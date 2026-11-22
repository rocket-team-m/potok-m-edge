// SPDX-License-Identifier: GPL-3.0-or-later
//! v3 speculative dispatch: a stateful student on every cycle, the windowed
//! teacher by exception, and the AI output watchguard on the chosen command.
//!
//! The v3 target is the authoritative fin command `[g_accel_y, -g_accel_x]`
//! in m/s^2 (`potok_ai.data_v3`); the angle-domain `g_theta` is diagnostic and
//! never reaches this dispatch. Guard bounds and rate limits are therefore in
//! the command's own m/s^2 units.
//!
//! Rules, in order:
//! 1. the student predicts on every valid cycle;
//! 2. low student confidence or invalid sensors, **or** the periodic forced
//!    check, call the teacher — all three share ONE call site, so a cycle can
//!    never make two teacher calls even when both conditions fire;
//! 3. low confidence / invalid sensors apply the teacher command; a forced
//!    check applies it too and additionally compares the two heads, logging a
//!    divergence and re-warming the student when they disagree;
//! 4. the chosen command passes through the pure [`guard`] before it is
//!    applied; any rejection emits zeros, and three consecutive rejections
//!    latch the assistant off until [`reset`];
//! 5. a malformed or non-finite frame bypasses both models and the guard.
//!
//! Prediction is behind [`V3Predictor`] so the dispatch is testable with
//! deterministic fixtures and does not depend on a particular v3 model file.

use std::fmt;

/// Frames kept for the windowed teacher. Same as the v2 dispatch.
pub const WINDOW: usize = 50;
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.7;
pub const DEFAULT_FORCED_CHECK_EVERY: u32 = 20;
pub const DEFAULT_DIVERGENCE_FACTOR: f32 = 3.0;
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;
pub const DEFAULT_MAX_MISSES: u32 = 3;

// ---------------------------------------------------------------------------
// Pure watchguard — Rust mirror of ai/src/potok_ai/watchguard.py (B02).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardReason {
    Ok,
    NonFinite,
    OutOfBounds,
    LowConfidence,
    RateLimit,
    Latched,
}

impl GuardReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NonFinite => "non_finite",
            Self::OutOfBounds => "out_of_bounds",
            Self::LowConfidence => "low_confidence",
            Self::RateLimit => "rate_limit",
            Self::Latched => "latched",
        }
    }
}

impl fmt::Display for GuardReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-channel command bounds and per-tick rate limits, in m/s^2.
/// Constructed through [`GuardConfig::new`], which validates loudly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardConfig {
    pub bounds: [(f32, f32); 2],
    pub max_rate: [f32; 2],
    pub min_confidence: f32,
    pub max_misses: u32,
}

impl GuardConfig {
    pub fn new(
        bounds: [(f32, f32); 2],
        max_rate: [f32; 2],
        min_confidence: f32,
        max_misses: u32,
    ) -> Result<Self, String> {
        for (low, high) in bounds {
            if !(low.is_finite() && high.is_finite() && low < high) {
                return Err(format!(
                    "bounds need finite low < high, got ({low}, {high})"
                ));
            }
        }
        for rate in max_rate {
            if !rate.is_finite() || rate < 0.0 {
                return Err(format!(
                    "max_rate must be finite and non-negative, got {rate}"
                ));
            }
        }
        if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
            return Err(format!(
                "min_confidence must be in [0, 1], got {min_confidence}"
            ));
        }
        if max_misses == 0 {
            return Err("max_misses must be >= 1".to_owned());
        }
        Ok(Self {
            bounds,
            max_rate,
            min_confidence,
            max_misses,
        })
    }
}

/// Conservative example bounds, not a flight-certified envelope: the caller
/// must supply the calibrated limits for its vehicle.
impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            bounds: [(-20.0, 20.0), (-20.0, 20.0)],
            max_rate: [20.0, 20.0],
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_misses: DEFAULT_MAX_MISSES,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GuardState {
    pub misses: u32,
    pub latched: bool,
    pub last_command: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardDecision {
    pub accepted: bool,
    pub reason: GuardReason,
    pub command: [f32; 2],
    pub state: GuardState,
}

/// The only way to clear a latch: fresh, immutable state.
pub fn reset() -> GuardState {
    GuardState::default()
}

/// Total, deterministic function: no clock, no globals, no mutation.
/// Check order is fixed so the rejection reason is reproducible:
/// `latched -> non-finite -> bounds -> confidence -> rate`.
pub fn guard(
    command: [f32; 2],
    confidence: f32,
    state: GuardState,
    config: &GuardConfig,
) -> GuardDecision {
    let zeros = [0.0f32; 2];
    if state.latched {
        return GuardDecision {
            accepted: false,
            reason: GuardReason::Latched,
            command: zeros,
            state,
        };
    }
    let reject = |reason| GuardDecision {
        accepted: false,
        reason,
        command: zeros,
        state: GuardState {
            misses: state.misses + 1,
            latched: state.misses + 1 >= config.max_misses,
            last_command: state.last_command,
        },
    };
    if !command[0].is_finite() || !command[1].is_finite() || !confidence.is_finite() {
        return reject(GuardReason::NonFinite);
    }
    if command
        .iter()
        .zip(config.bounds)
        .any(|(v, (low, high))| *v < low || *v > high)
    {
        return reject(GuardReason::OutOfBounds);
    }
    if confidence < config.min_confidence {
        return reject(GuardReason::LowConfidence);
    }
    if let Some(prev) = state.last_command {
        if command
            .iter()
            .zip(prev)
            .zip(config.max_rate)
            .any(|((v, p), rate)| (*v - p).abs() > rate)
        {
            return reject(GuardReason::RateLimit);
        }
    }
    GuardDecision {
        accepted: true,
        reason: GuardReason::Ok,
        command,
        state: GuardState {
            misses: 0,
            latched: false,
            last_command: Some(command),
        },
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// One prediction: authoritative 2-channel command plus a confidence in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V3Output {
    pub command: [f32; 2],
    pub confidence: f32,
}

/// Implemented by the student and teacher models. `reset` is called after a
/// divergence; stateful implementations must override it (the counter records
/// calls, not the effect). The closure impl has no resettable model state.
pub trait V3Predictor {
    fn predict(&mut self, input: &[f32]) -> V3Output;
    fn reset(&mut self) {}
}

impl<F: FnMut(&[f32]) -> V3Output> V3Predictor for F {
    fn predict(&mut self, input: &[f32]) -> V3Output {
        self(input)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3Source {
    Student,
    TeacherLowConfidence,
    TeacherForced,
    TeacherBoth,
    Zero,
}

impl V3Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Student => "student",
            Self::TeacherLowConfidence => "teacher/low-confidence",
            Self::TeacherForced => "teacher/forced-check",
            Self::TeacherBoth => "teacher/low-confidence+forced",
            Self::Zero => "zero",
        }
    }
}

impl fmt::Display for V3Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchConfig {
    /// Sensor frame width; the window holds `WINDOW` frames of this length.
    pub frame_len: usize,
    pub confidence_threshold: f32,
    pub forced_check_every: u32,
    pub divergence_factor: f32,
    pub guards: GuardConfig,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            frame_len: 8,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            forced_check_every: DEFAULT_FORCED_CHECK_EVERY,
            divergence_factor: DEFAULT_DIVERGENCE_FACTOR,
            guards: GuardConfig::default(),
        }
    }
}

impl DispatchConfig {
    fn validate(&self) -> Result<(), String> {
        if self.frame_len == 0 {
            return Err("frame_len must be >= 1".to_owned());
        }
        if !self.confidence_threshold.is_finite() {
            return Err("confidence_threshold must be finite".to_owned());
        }
        if self.forced_check_every == 0 {
            return Err("forced_check_every must be >= 1".to_owned());
        }
        if !self.divergence_factor.is_finite() {
            return Err("divergence_factor must be finite".to_owned());
        }
        if !(self.guards.bounds[0].0.is_finite()
            && self.guards.bounds[0].1.is_finite()
            && self.guards.bounds[1].0.is_finite()
            && self.guards.bounds[1].1.is_finite())
        {
            return Err("guard bounds must be finite".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct V3Counters {
    pub cycles: u64,
    pub invalid_cycles: u64,
    pub student_calls: u64,
    pub teacher_calls: u64,
    pub low_confidence: u64,
    pub forced_checks: u64,
    /// Cycles where both conditions fired; still exactly one teacher call.
    pub both_triggers: u64,
    pub divergences: u64,
    pub student_resets: u64,
    pub guard_accepted: u64,
    pub guard_rejected: u64,
    pub reject_non_finite: u64,
    pub reject_out_of_bounds: u64,
    pub reject_low_confidence: u64,
    pub reject_rate_limit: u64,
    pub reject_latched: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V3Reply {
    pub seq: u16,
    /// Guarded authoritative command; all zeros when `accepted` is false.
    pub command: [f32; 2],
    pub confidence: f32,
    /// Which model produced the pre-guard command (Zero for an invalid frame).
    pub source: V3Source,
    pub reason: GuardReason,
    pub accepted: bool,
}

/// Holds the student and windowed teacher side by side. Teacher calls happen
/// at exactly one site in [`V3Dispatch::on_cycle`].
pub struct V3Dispatch<S: V3Predictor, T: V3Predictor> {
    student: S,
    teacher: T,
    cfg: DispatchConfig,
    guard_state: GuardState,
    window: Vec<f32>,
    frames_since_check: u32,
    pub counters: V3Counters,
    pub last_source: V3Source,
    pub last_reason: GuardReason,
}

impl<S: V3Predictor, T: V3Predictor> V3Dispatch<S, T> {
    pub fn new(student: S, teacher: T, cfg: DispatchConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            student,
            teacher,
            window: vec![0.0; WINDOW * cfg.frame_len],
            cfg,
            guard_state: reset(),
            frames_since_check: 0,
            counters: V3Counters::default(),
            last_source: V3Source::Zero,
            last_reason: GuardReason::Ok,
        })
    }

    pub fn counters(&self) -> &V3Counters {
        &self.counters
    }

    pub fn guard_state(&self) -> GuardState {
        self.guard_state
    }

    /// Explicit latch clear, matching the guard's `reset()`.
    pub fn reset_guard(&mut self) {
        self.guard_state = reset();
    }

    pub fn student(&self) -> &S {
        &self.student
    }

    pub fn student_mut(&mut self) -> &mut S {
        &mut self.student
    }

    pub fn teacher(&self) -> &T {
        &self.teacher
    }

    pub fn teacher_mut(&mut self) -> &mut T {
        &mut self.teacher
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    fn push_frame(&mut self, frame: &[f32]) {
        let n = self.cfg.frame_len;
        self.window.copy_within(n..WINDOW * n, 0);
        self.window[(WINDOW - 1) * n..WINDOW * n].copy_from_slice(frame);
    }

    /// One cycle. `sensors_valid` false is the "invalid sensors" teacher
    /// trigger from the v2 dispatch, not an invalid frame: the frame itself
    /// still has to be the right width and finite.
    pub fn on_cycle(&mut self, seq: u16, frame: &[f32], sensors_valid: bool) -> V3Reply {
        if frame.len() != self.cfg.frame_len || !frame.iter().all(|v| v.is_finite()) {
            self.counters.invalid_cycles += 1;
            self.last_source = V3Source::Zero;
            self.last_reason = GuardReason::Ok;
            return V3Reply {
                seq,
                command: [0.0; 2],
                confidence: 0.0,
                source: V3Source::Zero,
                reason: GuardReason::Ok,
                accepted: false,
            };
        }

        self.push_frame(frame);
        self.frames_since_check += 1;
        self.counters.cycles += 1;

        // Rule 1: the student runs on every valid cycle.
        let student_out = self.student.predict(frame);
        self.counters.student_calls += 1;

        let low_conf = student_out.confidence < self.cfg.confidence_threshold || !sensors_valid;
        if low_conf {
            self.counters.low_confidence += 1;
        }
        let forced = self.frames_since_check >= self.cfg.forced_check_every;
        if forced {
            self.counters.forced_checks += 1;
        }
        if low_conf && forced {
            self.counters.both_triggers += 1;
        }

        let teacher_before = self.counters.teacher_calls;
        let (chosen, source) = if low_conf || forced {
            // THE single teacher call site. Both conditions share it.
            let teacher_out = self.teacher.predict(&self.window);
            self.counters.teacher_calls += 1;
            let g_max = self
                .cfg
                .guards
                .bounds
                .iter()
                .map(|(low, high)| low.abs().max(high.abs()))
                .fold(0.0f32, f32::max);
            let diff = (teacher_out.command[0] - student_out.command[0])
                .abs()
                .max((teacher_out.command[1] - student_out.command[1]).abs());
            if diff > self.cfg.divergence_factor * g_max {
                self.counters.divergences += 1;
                self.counters.student_resets += 1;
                self.student.reset();
            }
            self.frames_since_check = 0;
            let source = match (low_conf, forced) {
                (true, true) => V3Source::TeacherBoth,
                (true, false) => V3Source::TeacherLowConfidence,
                (false, true) => V3Source::TeacherForced,
                (false, false) => unreachable!(),
            };
            (teacher_out, source)
        } else {
            (student_out, V3Source::Student)
        };
        debug_assert!(
            self.counters.teacher_calls - teacher_before <= 1,
            "one cycle may call the teacher at most once"
        );

        // Rule 4: the guard is the last word, on both paths.
        let decision = guard(
            chosen.command,
            chosen.confidence,
            self.guard_state,
            &self.cfg.guards,
        );
        self.guard_state = decision.state;
        match decision.reason {
            GuardReason::Ok => self.counters.guard_accepted += 1,
            _ => self.counters.guard_rejected += 1,
        }
        match decision.reason {
            GuardReason::NonFinite => self.counters.reject_non_finite += 1,
            GuardReason::OutOfBounds => self.counters.reject_out_of_bounds += 1,
            GuardReason::LowConfidence => self.counters.reject_low_confidence += 1,
            GuardReason::RateLimit => self.counters.reject_rate_limit += 1,
            GuardReason::Latched => self.counters.reject_latched += 1,
            GuardReason::Ok => {}
        }

        self.last_source = source;
        self.last_reason = decision.reason;
        V3Reply {
            seq,
            command: decision.command,
            confidence: chosen.confidence,
            source,
            reason: decision.reason,
            accepted: decision.accepted,
        }
    }
}
