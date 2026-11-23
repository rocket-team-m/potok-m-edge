//! Gates for the v3 path through `Runtime::load`. The v3 bytes are built in
//! memory with the canonical layout from `model_v3_tests.rs`; no model binary
//! is added to the repository. What is proven here: the `POTOKM3\0` branch is
//! reachable, its SHA-256 is enforced, the v2/legacy branches are unchanged,
//! and a `calibration_v3` threshold is consumed by the dispatch.

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{RequestFrame, FLAG_IMU_OK, PHASE_LANDING, PROTOCOL_VERSION};
use potok_edge::model_v3::{
    COND_FEATURES, FP32_LAYER, IN_FEATURES, MAGIC, MODEL_KIND_FAST, OUT_FEATURES, PHYSICS_FEATURES,
    TARGET_VERSION, TSCT_LAYER, VERSION,
};
use potok_edge::runtime::Runtime;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct Spec {
    width: usize,
    rank: usize,
    n_blocks: usize,
    n_loops: usize,
    kernel: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            width: 8,
            rank: 2,
            n_blocks: 2,
            n_loops: 2,
            kernel: 3,
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn code(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }
    fn scale(&mut self) -> f32 {
        0.05 + (self.next() % 20) as f32 * 0.01
    }
}

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_name(b: &mut Vec<u8>, kind: u8, name: &str) {
    b.push(kind);
    push_u32(b, name.len() as u32);
    b.extend_from_slice(name.as_bytes());
}
fn f32_record(b: &mut Vec<u8>, name: &str, shape: &[usize], data: &[f32]) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    push_name(b, FP32_LAYER, name);
    push_u32(b, shape.len() as u32);
    for &d in shape {
        push_u32(b, d as u32);
    }
    for &v in data {
        push_f32(b, v);
    }
}
fn tsct_record(b: &mut Vec<u8>, rng: &mut Lcg, name: &str, rank: usize, w: usize) {
    push_name(b, TSCT_LAYER, name);
    push_u32(b, w as u32);
    push_u32(b, w as u32);
    push_u32(b, rank as u32);
    push_f32(b, 1.0);
    push_f32(b, 1.0);
    for _ in 0..w * rank {
        b.push(rng.code() as u8);
    }
    for _ in 0..rank {
        push_f32(b, rng.scale());
    }
    for _ in 0..w * rank {
        b.push(rng.code() as u8);
    }
}

fn build_body(spec: &Spec, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    for b in 0..spec.n_blocks {
        let prefix = format!("blocks.{b}");
        for suffix in ["rec_h.weight", "rec_x.weight"] {
            let data: Vec<f32> = (0..spec.width * spec.width)
                .map(|_| rng.f32() * 0.08)
                .collect();
            let mut rec = Vec::new();
            f32_record(
                &mut rec,
                &format!("{prefix}.{suffix}"),
                &[spec.width; 2],
                &data,
            );
            records.push((format!("{prefix}.{suffix}"), rec));
        }
        for suffix in ["rec_h.bias", "rec_x.bias"] {
            let data: Vec<f32> = (0..spec.width).map(|_| rng.f32() * 0.02).collect();
            let mut rec = Vec::new();
            f32_record(
                &mut rec,
                &format!("{prefix}.{suffix}"),
                &[spec.width],
                &data,
            );
            records.push((format!("{prefix}.{suffix}"), rec));
        }
        let mut rec = Vec::new();
        tsct_record(
            &mut rec,
            &mut rng,
            &format!("{prefix}.proj"),
            spec.rank,
            spec.width,
        );
        records.push((format!("{prefix}.proj"), rec));
    }

    let w = spec.width;
    let f = IN_FEATURES;
    let o = OUT_FEATURES;
    let globals: [(&str, Vec<usize>, Vec<f32>); 10] = [
        (
            "stem.weight",
            vec![w, f],
            (0..w * f).map(|_| rng.f32() * 0.3).collect(),
        ),
        (
            "stem.bias",
            vec![w],
            (0..w).map(|_| rng.f32() * 0.02).collect(),
        ),
        (
            "norm.weight",
            vec![w],
            (0..w)
                .map(|_| 0.9 + (rng.next() % 10) as f32 * 0.02)
                .collect(),
        ),
        (
            "norm.bias",
            vec![w],
            (0..w).map(|_| rng.f32() * 0.01).collect(),
        ),
        (
            "alloc_logits",
            vec![spec.n_loops],
            (0..spec.n_loops).map(|_| rng.f32() * 0.2).collect(),
        ),
        (
            "head_corr.weight",
            vec![o, w],
            (0..o * w).map(|_| rng.f32() * 0.3).collect(),
        ),
        (
            "head_corr.bias",
            vec![o],
            (0..o).map(|_| rng.f32() * 0.05).collect(),
        ),
        (
            "head_conf.weight",
            vec![1, w],
            (0..w).map(|_| rng.f32() * 0.3).collect(),
        ),
        ("head_conf.bias", vec![1], vec![rng.f32() * 0.05]),
        ("g_max", vec![1], vec![0.12]),
    ];
    for (name, shape, data) in globals {
        let mut rec = Vec::new();
        f32_record(&mut rec, name, &shape, &data);
        records.push((name.to_owned(), rec));
    }
    assert_eq!(records.len(), 5 * spec.n_blocks + 10);

    let header_len = 64 + 4 * spec.n_blocks;
    let mut body = vec![0u8; header_len];
    body[..8].copy_from_slice(&MAGIC);
    put_u32(&mut body, 8, VERSION);
    put_u32(&mut body, 12, MODEL_KIND_FAST);
    put_u32(&mut body, 16, IN_FEATURES as u32);
    put_u32(&mut body, 20, w as u32);
    put_u32(&mut body, 24, o as u32);
    put_u32(&mut body, 28, spec.rank as u32);
    put_u32(&mut body, 32, spec.n_blocks as u32);
    put_u32(&mut body, 36, spec.kernel as u32);
    put_u32(&mut body, 40, spec.n_loops as u32);
    body[44..48].copy_from_slice(&0.8f32.to_le_bytes());
    body[48..52].copy_from_slice(&0.12f32.to_le_bytes());
    put_u32(&mut body, 52, 1);
    put_u32(&mut body, 56, TARGET_VERSION);
    put_u32(&mut body, 60, records.len() as u32);
    for b in 0..spec.n_blocks {
        put_u32(&mut body, 64 + 4 * b, (b + 1) as u32);
    }
    for (_, rec) in records {
        body.extend_from_slice(&rec);
    }
    body
}

fn build(seed: u64) -> Vec<u8> {
    let body = build_body(&Spec::default(), seed);
    let mut bytes = body.clone();
    bytes.extend_from_slice(&Sha256::digest(&body));
    bytes
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("potok-v3-dispatch-{}-{name}", std::process::id()))
}

fn request(seq: u16) -> RequestFrame {
    RequestFrame {
        seq,
        time_us: seq as u32 * 20_000,
        phase: PHASE_LANDING,
        version: PROTOCOL_VERSION,
        flags: FLAG_IMU_OK,
        gyro: [0.02, -0.01, 0.005],
        accel: [0.05, 9.79, -0.03],
        mag: [1.0, 2.0, 3.0],
        height: 1500.0 + seq as f32,
        pitot_dp: (seq % 50) as i16,
    }
}

#[test]
fn runtime_load_reaches_the_v3_reader_and_dispatches() {
    let path = temp_path("model.bin");
    fs::write(&path, build(7)).unwrap();
    let mut rt = Runtime::load(&path).unwrap();
    assert!(rt.is_v3());
    assert!(!rt.is_v2());
    assert!(rt.potm_model().is_none());
    let model = rt.v3_model().expect("v3 reader must be reachable");
    assert_eq!(model.in_features, IN_FEATURES);
    assert_eq!(model.out_features, OUT_FEATURES);
    assert_eq!(PHYSICS_FEATURES + COND_FEATURES, IN_FEATURES);
    assert_eq!(model.target_version, TARGET_VERSION);
    let bound = 0.12f32.tanh();
    assert!((rt.max_correction() - bound).abs() < 1e-7);
    let limit = (bound / 0.001).round() as i16 + 1;
    for seq in 1..=6u16 {
        let reply = rt.on_request(&request(seq));
        assert_eq!(reply.seq, seq);
        assert_eq!(reply.ai_healthy, 1);
        assert!(reply.g_pitch.abs() <= limit && reply.g_yaw.abs() <= limit);
        assert_eq!(
            reply.confidence,
            (rt.last_raw()[2].clamp(0.0, 1.0) * 255.0).round() as u8
        );
    }
    assert!(rt.memory_bytes() > 0);
    assert!(rt.v3_threshold().is_none(), "uncalibrated load has no cut");
    fs::remove_file(&path).unwrap();
}

#[test]
fn runtime_load_refuses_a_corrupt_v3_checksum() {
    let path = temp_path("bad-sha.bin");
    let mut bytes = build(7);
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&path, &bytes).unwrap();
    let err = Runtime::load(&path).err().expect("load must fail");
    assert!(
        err.contains("SHA256") || err.contains("checksum"),
        "loud checksum refusal: {err}"
    );
    fs::remove_file(&path).unwrap();
}

#[test]
fn runtime_load_keeps_v2_and_legacy_paths() {
    let v2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_model.bin");
    let rt = Runtime::load(&v2).unwrap();
    assert!(rt.is_v2() && !rt.is_v3());
    assert!(rt.potm_model().is_none());
    assert!(rt.v3_model().is_none());
    assert!(rt.v3_threshold().is_none());

    let path = temp_path("legacy.bin");
    fs::write(&path, build_fixture(FixtureConfig::default(), 0x504F544B)).unwrap();
    let rt = Runtime::load(&path).unwrap();
    assert!(rt.potm_model().is_some() && !rt.is_v2() && !rt.is_v3());
    fs::remove_file(&path).unwrap();
}

#[test]
fn calibration_threshold_is_consumed_by_v3_dispatch() {
    let model_path = temp_path("cal-model.bin");
    fs::write(&model_path, build(7)).unwrap();
    let mut plain = Runtime::load(&model_path).unwrap();
    let window = vec![0.05f32; 50 * 8];
    let uncalibrated = plain.infer_window(&window);

    let cal_path = temp_path("cal.json");
    fs::write(
        &cal_path,
        r#"{"method":"temperature","temperature":2.0,"threshold":0.0}"#,
    )
    .unwrap();
    let mut open = Runtime::load_with_calibration(&model_path, &cal_path).unwrap();
    assert!(open.is_v3());
    assert_eq!(open.v3_threshold(), Some(0.0));
    let out = open.infer_window(&window);
    assert_eq!(
        (out[0], out[1]),
        (uncalibrated[0], uncalibrated[1]),
        "threshold 0 never withholds a correction"
    );

    fs::write(
        &cal_path,
        r#"{"method":"temperature","temperature":2.0,"threshold":1.0}"#,
    )
    .unwrap();
    let mut strict = Runtime::load_with_calibration(&model_path, &cal_path).unwrap();
    assert_eq!(strict.v3_threshold(), Some(1.0));
    let out = strict.infer_window(&window);
    assert!(out[2] < 1.0, "fixture confidence must be below the cut");
    assert_eq!(
        (out[0], out[1]),
        (0.0, 0.0),
        "a bare runtime withholds below the adaptive-compute cut"
    );

    let legacy_cal = Runtime::load_with_calibration(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_model.bin"),
        &cal_path,
    );
    assert!(legacy_cal.is_err(), "calibration is a v3-only contract");
    fs::remove_file(&model_path).unwrap();
    fs::remove_file(&cal_path).unwrap();
}

#[test]
fn v3_conditioning_reaches_the_smart_frame_tail() {
    let path = temp_path("cond.bin");
    fs::write(&path, build(9)).unwrap();
    let mut rt = Runtime::load(&path).unwrap();
    let window = vec![0.05f32; 50 * 8];
    let base = rt.infer_window(&window);
    rt.set_v3_conditioning([1.0; COND_FEATURES]);
    let changed = rt.infer_window(&window);
    assert_ne!(
        (base[0], base[1]),
        (changed[0], changed[1]),
        "conditioning slots 14..20 feed the stem"
    );
    fs::remove_file(&path).unwrap();
}
