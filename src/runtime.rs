//! Frame-to-frame safety state: sliding window, phase gate, sequence and
//! miss accounting. No I/O and no allocation happens per frame.

use crate::calibration_v3::Calibration;
use crate::format_v2::ModelV2;
use crate::frame::{
    ReplyFrame, RequestFrame, CORRECTION_SCALE, FLAG_MOTOR_ARMED, PHASE_ARMED, PHASE_FAULT,
    PHASE_IDLE,
};
use crate::model::{Model, Scratch};
use crate::model_v3::V3Model;
use crate::simd::dot;
use crate::student_runtime::LoadOptions;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub const MAX_CORRECTION: f32 = 0.1; // radians; |g| <= Gmax (concept §2)
pub const MAX_CONSECUTIVE_MISSES: u32 = 3;

/// Policy for when the assist output is allowed.
///
/// The concept requires silence while the motor works. `potok-uart` v1 has no
/// dedicated BURN phase, so the gate combines the phase byte with the
/// `MOTOR_ARMED` flag: `Flight` is allowed only when the motor is not armed
/// (coast / apogee / descent all report as `Flight` in this protocol).
#[derive(Debug, Clone)]
pub struct PhasePolicy {
    pub forbidden_phases: [bool; 8],
    pub forbid_motor_armed: bool,
}

impl Default for PhasePolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

impl PhasePolicy {
    pub fn conservative() -> Self {
        let mut forbidden_phases = [false; 8];
        forbidden_phases[PHASE_IDLE as usize] = true;
        forbidden_phases[PHASE_ARMED as usize] = true;
        forbidden_phases[PHASE_FAULT as usize] = true;
        Self {
            forbidden_phases,
            forbid_motor_armed: true,
        }
    }

    pub fn is_forbidden(&self, phase: u8, flags: u8) -> bool {
        let p = self
            .forbidden_phases
            .get(phase as usize)
            .copied()
            .unwrap_or(true);
        p || (self.forbid_motor_armed && flags & FLAG_MOTOR_ARMED != 0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    pub frames_ok: u64,
    pub frames_muted: u64,
    pub crc_errors: u64,
    pub seq_errors: u64,
    pub timeouts: u64,
    pub disables: u64,
}

/// The sensor part of a request frame, in floats. The remaining v3 physics
/// slots (8..14) are zero and the six conditioning slots (14..20) come from
/// `conditioning`; this is the smart-frame layout of `student_runtime.rs`.
const SENSOR_FEATURES: usize = 8;
const V3_CONDITIONING_START: usize = 14;
const V3_WINDOW: usize = 50;

/// Stateful v3 fast-model runner. `V3Model` is the SHA-verified reader; this
/// struct owns only the reusable hot-loop buffers and replays the Runtime's
/// 50-frame sensor window through the reader's dequantized weights. The tick
/// math is the canonical `StatefulTCN.step` mirrored in `student_runtime.rs`.
struct V3Engine {
    model: V3Model,
    calibration: Option<Calibration>,
    h: Vec<f32>,
    cand: Vec<f32>,
    next_source: Vec<f32>,
    source: Vec<f32>,
    total: Vec<f32>,
    z: Vec<f32>,
    frame: Vec<f32>,
    alloc: Vec<f32>,
    conditioning: [f32; 6],
}

impl V3Engine {
    fn new(model: V3Model, calibration: Option<Calibration>) -> Self {
        assert_eq!(model.in_features, crate::model_v3::IN_FEATURES);
        assert_eq!(model.out_features, crate::model_v3::OUT_FEATURES);
        let n = model.n_blocks * model.width;
        let mut alloc = vec![0.0f32; model.n_loops];
        let mut max = f32::NEG_INFINITY;
        for (l, dst) in alloc.iter_mut().enumerate().take(model.n_loops) {
            *dst = model.alloc[l];
            max = max.max(*dst);
        }
        let mut sum = 0.0f32;
        for a in alloc.iter_mut() {
            *a = (*a - max).exp();
            sum += *a;
        }
        for a in alloc.iter_mut() {
            *a /= sum;
        }
        Self {
            h: vec![0.0; n],
            cand: vec![0.0; model.width],
            next_source: vec![0.0; model.width],
            source: vec![0.0; model.width],
            total: vec![0.0; model.width],
            z: vec![0.0; model.width],
            frame: vec![0.0; model.in_features],
            alloc,
            conditioning: [0.0; 6],
            model,
            calibration,
        }
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
    }

    /// Replay the whole window from a clean state and return the last tick.
    /// The Runtime shifts its window one frame per request, so replay gives the
    /// same hidden state as an incremental step without carrying state across
    /// requests (and without a second inference path to keep in sync).
    fn forward(&mut self, window: &[f32]) -> [f32; 3] {
        assert_eq!(
            window.len() % SENSOR_FEATURES,
            0,
            "v3 window holds whole sensor frames"
        );
        self.reset();
        let mut out = [0.0f32; 3];
        for sensors in window.as_chunks::<SENSOR_FEATURES>().0 {
            out = self.step(sensors);
        }
        out
    }

    fn step(&mut self, sensors: &[f32]) -> [f32; 3] {
        self.frame.fill(0.0);
        let n = sensors.len().min(self.frame.len());
        self.frame[..n].copy_from_slice(&sensors[..n]);
        self.frame[V3_CONDITIONING_START..].copy_from_slice(&self.conditioning);

        let w = self.model.width;
        for o in 0..w {
            let row =
                &self.model.stem_w[o * self.model.in_features..(o + 1) * self.model.in_features];
            self.source[o] = dot(row, &self.frame) + self.model.stem_b[o];
        }
        self.total.fill(0.0);
        for iteration in 0..self.model.n_loops {
            for (b, block) in self.model.blocks.iter().enumerate() {
                let h_prev = &self.h[b * w..(b + 1) * w];
                for o in 0..w {
                    let rh = &block.rec_h_w[o * w..(o + 1) * w];
                    let rx = &block.rec_x_w[o * w..(o + 1) * w];
                    let acc = dot(rx, &self.source)
                        + dot(rh, h_prev)
                        + block.rec_x_b[o]
                        + block.rec_h_b[o];
                    self.cand[o] = 0.8 * h_prev[o] + 0.2 * acc.tanh();
                }
                for o in 0..w {
                    self.next_source[o] = dot(&block.proj_w[o * w..(o + 1) * w], &self.cand);
                }
                self.source.copy_from_slice(&self.next_source);
                self.h[b * w..(b + 1) * w].copy_from_slice(&self.cand);
            }
            let weight = self.alloc[iteration] * self.model.decay.powi(iteration as i32);
            for o in 0..w {
                self.total[o] += weight * self.source[o];
            }
        }
        for o in 0..w {
            self.next_source[o] = self.total[o] + self.source[o];
        }
        crate::model_v3::layer_norm(
            &self.next_source,
            &self.model.norm_w,
            &self.model.norm_b,
            &mut self.z,
        );
        let mut out = [0.0f32; 3];
        for (k, dst) in out.iter_mut().enumerate().take(self.model.out_features) {
            *dst = dot(&self.model.head_corr_w[k * w..(k + 1) * w], &self.z)
                + self.model.head_corr_b[k];
            *dst = dst.tanh() * self.model.correction_bound;
        }
        let logit = dot(&self.model.head_conf_w[..w], &self.z) + self.model.head_conf_b[0];
        out[2] = match &self.calibration {
            Some(c) if logit.is_finite() => c.apply_logit(logit as f64).unwrap_or(0.0) as f32,
            _ => 1.0 / (1.0 + (-logit).exp()),
        };
        // The calibrated threshold is the adaptive-compute cut: normally it
        // escalates to a windowed teacher. A bare Runtime has no teacher, so it
        // withholds the correction below the cut rather than guessing.
        if let Some(c) = &self.calibration {
            if (out[2] as f64) < c.threshold() {
                out[0] = 0.0;
                out[1] = 0.0;
            }
        }
        out
    }

    fn memory_bytes(&self) -> usize {
        self.model.memory_bytes()
            + (self.h.len()
                + self.cand.len()
                + self.next_source.len()
                + self.source.len()
                + self.total.len()
                + self.z.len()
                + self.frame.len()
                + self.alloc.len())
                * std::mem::size_of::<f32>()
    }
}

/// One of the three supported model formats, behind the same fixed window API.
enum Engine {
    PotM { model: Model, scratch: Scratch },
    V2(ModelV2),
    V3(Box<V3Engine>),
}

impl Engine {
    fn window_len(&self) -> usize {
        match self {
            Self::PotM { model, .. } => model.window,
            Self::V2(m) => m.window,
            Self::V3(_) => V3_WINDOW,
        }
    }

    fn max_correction(&self) -> f32 {
        match self {
            Self::PotM { .. } => MAX_CORRECTION,
            Self::V2(m) => m.g_max,
            Self::V3(e) => e.model.correction_bound,
        }
    }

    fn forward(&mut self, window: &[f32]) -> [f32; 3] {
        match self {
            Self::PotM { model, scratch } => {
                let (p, y, c) = model.forward(window, scratch);
                [p, y, c]
            }
            Self::V2(m) => m.forward(window),
            Self::V3(e) => e.forward(window),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::PotM { model, scratch } => model.memory_bytes() + scratch.memory_bytes(),
            Self::V2(m) => m.memory_bytes(),
            Self::V3(e) => e.memory_bytes(),
        }
    }

    fn potm(&self) -> Option<&Model> {
        match self {
            Self::PotM { model, .. } => Some(model),
            Self::V2(_) | Self::V3(_) => None,
        }
    }
}

/// Owns the model engine, the preallocated window and the safety state.
pub struct Runtime {
    engine: Engine,
    window: VecDeque<[f32; 8]>,
    flat: Vec<f32>,
    policy: PhasePolicy,
    last_seq: Option<u16>,
    misses: u32,
    disabled: bool,
    last_raw: [f32; 3],
    pub counters: Counters,
}

impl Runtime {
    pub fn new(model: Model) -> Self {
        Self::with_policy(model, PhasePolicy::default())
    }

    pub fn with_policy(model: Model, policy: PhasePolicy) -> Self {
        Self::from_engine(
            Engine::PotM {
                scratch: Scratch::new(&model),
                model,
            },
            policy,
        )
    }

    pub fn from_v2(model: ModelV2) -> Self {
        Self::with_policy_v2(model, PhasePolicy::default())
    }

    pub fn with_policy_v2(model: ModelV2, policy: PhasePolicy) -> Self {
        Self::from_engine(Engine::V2(model), policy)
    }

    pub fn from_v3(model: V3Model) -> Self {
        Self::with_policy_v3(model, PhasePolicy::default())
    }

    pub fn with_policy_v3(model: V3Model, policy: PhasePolicy) -> Self {
        Self::from_engine(Engine::V3(Box::new(V3Engine::new(model, None))), policy)
    }

    /// Load one of the three formats by its magic bytes: `POTOKM3\0` selects
    /// the v3 reader, `POTOKM1\0` the v2 reader, everything else the legacy
    /// POTM reader. Each reader verifies its own SHA-256 before parsing.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if bytes.starts_with(&crate::model_v3::MAGIC) {
            // POTOKM3 goes through the typed ±10 m/s² command envelope, the
            // same bound resolution as the v2 student reader (§2.3/§2.4).
            V3Model::from_bytes_with_options(&bytes, LoadOptions::default())
                .map(Self::from_v3)
                .map_err(|e| e.to_string())
        } else if bytes.starts_with(&crate::format_v2::MAGIC) {
            ModelV2::from_bytes(&bytes)
                .map(Self::from_v2)
                .map_err(|e| e.to_string())
        } else {
            Model::from_bytes(&bytes)
                .map(Self::new)
                .map_err(|e| e.to_string())
        }
    }

    /// Load a v3 model together with its `calibration_v3` export. The model
    /// still goes through [`Self::load`]'s SHA-verified reader; the calibration
    /// supplies the confidence transform and the adaptive-compute threshold.
    /// Other formats are refused: calibration is a v3 contract.
    pub fn load_with_calibration(
        model_path: impl AsRef<Path>,
        calibration_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let bytes = fs::read(model_path.as_ref())
            .map_err(|e| format!("cannot read {}: {e}", model_path.as_ref().display()))?;
        if !bytes.starts_with(&crate::model_v3::MAGIC) {
            return Err("calibration requires a POTOKM3 v3 model".to_owned());
        }
        let model = V3Model::from_bytes_with_options(&bytes, LoadOptions::default())
            .map_err(|e| e.to_string())?;
        let calibration = Calibration::from_file(calibration_path).map_err(|e| e.to_string())?;
        Ok(Self::from_engine(
            Engine::V3(Box::new(V3Engine::new(model, Some(calibration)))),
            PhasePolicy::default(),
        ))
    }

    fn from_engine(engine: Engine, policy: PhasePolicy) -> Self {
        let window_len = engine.window_len();
        let mut window = VecDeque::with_capacity(window_len);
        for _ in 0..window_len {
            window.push_back([0.0; 8]);
        }
        Self {
            engine,
            window,
            flat: vec![0.0; window_len * 8],
            policy,
            last_seq: None,
            misses: 0,
            disabled: false,
            last_raw: [0.0; 3],
            counters: Counters::default(),
        }
    }

    /// The legacy model, when this runtime holds one (used by the naive
    /// baseline benchmark). Returns `None` for v2 and v3 models.
    pub fn potm_model(&self) -> Option<&Model> {
        self.engine.potm()
    }

    pub fn is_v2(&self) -> bool {
        matches!(self.engine, Engine::V2(_))
    }

    pub fn is_v3(&self) -> bool {
        matches!(self.engine, Engine::V3(_))
    }

    /// The data-derived adaptive-compute cut, when this runtime was loaded
    /// with calibration. `None` for v2/legacy and for an uncalibrated v3 load.
    pub fn v3_threshold(&self) -> Option<f64> {
        match &self.engine {
            Engine::V3(e) => e.calibration.as_ref().map(|c| c.threshold()),
            _ => None,
        }
    }

    /// Six conditioning numbers appended to each v3 smart frame (physics slots
    /// 14..20). Ignored by the other formats.
    pub fn set_v3_conditioning(&mut self, conditioning: [f32; 6]) {
        if let Engine::V3(e) = &mut self.engine {
            e.conditioning = conditioning;
        }
    }

    /// The verified v3 reader, when this runtime holds one.
    pub fn v3_model(&self) -> Option<&V3Model> {
        match &self.engine {
            Engine::V3(e) => Some(&e.model),
            _ => None,
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    pub fn misses(&self) -> u32 {
        self.misses
    }

    /// Raw head output `[g_pitch, g_yaw, confidence]` of the last inference.
    pub fn last_raw(&self) -> [f32; 3] {
        self.last_raw
    }

    /// Current 50x8 sliding window, oldest first, flattened row-major.
    pub fn last_window(&self) -> &[f32] {
        &self.flat
    }

    /// Run the windowed model on an explicit flattened window without
    /// touching sequence, phase or miss state. Speculative dispatch owns its
    /// own window and uses this to make at most one teacher call per cycle.
    pub fn infer_window(&mut self, window: &[f32]) -> [f32; 3] {
        self.engine.forward(window)
    }

    /// Largest correction this engine's head can emit, in radians.
    pub fn max_correction(&self) -> f32 {
        self.engine.max_correction()
    }

    /// Total bytes owned by the dequantized weights and the reusable buffers.
    pub fn memory_bytes(&self) -> usize {
        self.engine.memory_bytes()
            + self.flat.len() * std::mem::size_of::<f32>()
            + self.window.capacity() * 8 * std::mem::size_of::<f32>()
    }

    /// Call when a frame arrived but failed CRC / magic / version / read timeout.
    /// Three consecutive failures latch the assistant off until restart.
    pub fn on_invalid_frame(&mut self) {
        if self.disabled {
            return;
        }
        self.misses += 1;
        self.disable_if_threshold();
    }

    pub fn on_timeout(&mut self) {
        self.counters.timeouts += 1;
        self.on_invalid_frame();
    }

    /// A protocol version mismatch is fatal immediately (no three-strike
    /// allowance): firmware and edge disagree on the wire format.
    pub fn disable_now(&mut self) {
        if !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }

    /// Produce the reply for one valid request. CRC was already checked by the
    /// codec; sequence and phase are checked here.
    pub fn on_request(&mut self, req: &RequestFrame) -> ReplyFrame {
        if self.disabled {
            return ReplyFrame::muted(req.seq, false);
        }

        if let Some(prev) = self.last_seq {
            let expected = prev.wrapping_add(1);
            if req.seq != expected {
                // Broken or duplicated packet: ignore it and resync on the new
                // sequence so one bad frame does not poison every later one.
                self.counters.seq_errors += 1;
                self.misses += 1;
                self.last_seq = Some(req.seq);
                self.disable_if_threshold();
                return ReplyFrame::muted(req.seq, !self.disabled);
            }
        }
        self.last_seq = Some(req.seq);

        // Any fully valid frame clears the consecutive-miss counter.
        self.misses = 0;

        if self.policy.is_forbidden(req.phase, req.flags) {
            self.counters.frames_muted += 1;
            // Muted by policy, not a fault: stay healthy so the controller keeps
            // listening, but send exactly zero correction.
            return ReplyFrame::muted(req.seq, true);
        }

        let features = req.features();
        self.window.pop_front();
        self.window.push_back(features);
        for (i, frame) in self.window.iter().enumerate() {
            self.flat[i * 8..i * 8 + 8].copy_from_slice(frame);
        }
        let [pitch, yaw, conf] = self.engine.forward(&self.flat);
        self.last_raw = [pitch, yaw, conf];

        self.counters.frames_ok += 1;
        let bound = self.engine.max_correction();
        let clamp = |v: f32| -> i16 {
            let v = v.clamp(-bound, bound);
            (v / CORRECTION_SCALE)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        };
        let confidence = conf.clamp(0.0, 1.0);
        ReplyFrame {
            seq: req.seq,
            g_pitch: clamp(pitch),
            g_yaw: clamp(yaw),
            confidence: (confidence * 255.0).round() as u8,
            ai_healthy: 1,
            version: 1,
        }
    }

    fn disable_if_threshold(&mut self) {
        if self.misses >= MAX_CONSECUTIVE_MISSES && !self.disabled {
            self.disabled = true;
            self.counters.disables += 1;
        }
    }
}
