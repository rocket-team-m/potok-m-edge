//! Typed command-envelope gate for the v3 fast-model reader (FIXRECIPE, 2026-11-23).
//!
//! `g_max` used to be compressed by `tanh`, capping every correction below 1
//! m/s² while the v3 target reaches ±10. This file pins the reader's mode
//! resolution, the ±10 default, the loud out-of-range refusal, unchanged
//! SHA/version compatibility, and the `Runtime::load` wiring. Every byte
//! sequence is built in memory; no model binary is added to the repository.

use potok_edge::model_v3::{
    FP32_LAYER, IN_FEATURES, MAGIC, MODEL_KIND_FAST, OUT_FEATURES, TARGET_VERSION, TSCT_LAYER,
    VERSION,
};
use potok_edge::student_runtime::{CommandBoundMode, CommandEnvelope, LoadOptions};
use potok_edge::{Runtime, V3Error, V3Model};
use sha2::{Digest, Sha256};

const WIDTH: usize = 8;
const RANK: usize = 2;
const BLOCKS: usize = 2;
const LOOPS: usize = 2;
const KERNEL: usize = 3;

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
}

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut [u8], at: usize, v: f32) {
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
fn tsct_record(b: &mut Vec<u8>, rng: &mut Lcg, name: &str) {
    push_name(b, TSCT_LAYER, name);
    push_u32(b, WIDTH as u32);
    push_u32(b, WIDTH as u32);
    push_u32(b, RANK as u32);
    push_f32(b, 1.0); // alpha_u
    push_f32(b, 1.0); // alpha_v
    for _ in 0..WIDTH * RANK {
        b.push(rng.code() as u8);
    }
    for _ in 0..RANK {
        push_f32(b, 0.1);
    }
    for _ in 0..WIDTH * RANK {
        b.push(rng.code() as u8);
    }
}

/// Body only (no digest), so corruption tests can reseal a modified body.
/// `corr_bias` drives the head to a known saturation point: with zero head
/// weights the emitted correction is `tanh(bias) * correction_bound`.
fn build_body(g_max: f32, corr_bias: [f32; 2]) -> Vec<u8> {
    let mut rng = Lcg(7);
    let mut body = vec![0u8; 64 + 4 * BLOCKS];
    body[..8].copy_from_slice(&MAGIC);
    put_u32(&mut body, 8, VERSION);
    put_u32(&mut body, 12, MODEL_KIND_FAST);
    put_u32(&mut body, 16, IN_FEATURES as u32);
    put_u32(&mut body, 20, WIDTH as u32);
    put_u32(&mut body, 24, OUT_FEATURES as u32);
    put_u32(&mut body, 28, RANK as u32);
    put_u32(&mut body, 32, BLOCKS as u32);
    put_u32(&mut body, 36, KERNEL as u32);
    put_u32(&mut body, 40, LOOPS as u32);
    put_f32(&mut body, 44, 0.8); // decay
    put_f32(&mut body, 48, g_max);
    put_u32(&mut body, 52, 1); // flags
    put_u32(&mut body, 56, TARGET_VERSION);
    put_u32(&mut body, 60, (5 * BLOCKS + 10) as u32);
    for b in 0..BLOCKS {
        put_u32(&mut body, 64 + 4 * b, (b + 1) as u32);
    }

    for b in 0..BLOCKS {
        let p = format!("blocks.{b}");
        for suffix in ["rec_h.weight", "rec_x.weight"] {
            let data: Vec<f32> = (0..WIDTH * WIDTH).map(|_| rng.f32() * 0.08).collect();
            f32_record(&mut body, &format!("{p}.{suffix}"), &[WIDTH, WIDTH], &data);
        }
        for suffix in ["rec_h.bias", "rec_x.bias"] {
            let data: Vec<f32> = (0..WIDTH).map(|_| rng.f32() * 0.02).collect();
            f32_record(&mut body, &format!("{p}.{suffix}"), &[WIDTH], &data);
        }
        tsct_record(&mut body, &mut rng, &format!("{p}.proj"));
    }

    let stem_w: Vec<f32> = (0..WIDTH * IN_FEATURES).map(|_| rng.f32() * 0.3).collect();
    let stem_b: Vec<f32> = (0..WIDTH).map(|_| rng.f32() * 0.02).collect();
    let norm_w = vec![1.0f32; WIDTH];
    let norm_b: Vec<f32> = (0..WIDTH).map(|_| rng.f32() * 0.01).collect();
    let alloc: Vec<f32> = (0..LOOPS).map(|_| rng.f32() * 0.2).collect();
    let corr_w = vec![0.0f32; OUT_FEATURES * WIDTH];
    let conf_w: Vec<f32> = (0..WIDTH).map(|_| rng.f32() * 0.3).collect();
    f32_record(&mut body, "stem.weight", &[WIDTH, IN_FEATURES], &stem_w);
    f32_record(&mut body, "stem.bias", &[WIDTH], &stem_b);
    f32_record(&mut body, "norm.weight", &[WIDTH], &norm_w);
    f32_record(&mut body, "norm.bias", &[WIDTH], &norm_b);
    f32_record(&mut body, "alloc_logits", &[LOOPS], &alloc);
    f32_record(
        &mut body,
        "head_corr.weight",
        &[OUT_FEATURES, WIDTH],
        &corr_w,
    );
    f32_record(&mut body, "head_corr.bias", &[OUT_FEATURES], &corr_bias);
    f32_record(&mut body, "head_conf.weight", &[1, WIDTH], &conf_w);
    f32_record(&mut body, "head_conf.bias", &[1], &[0.0]);
    f32_record(&mut body, "g_max", &[1], &[g_max]);
    body
}

fn seal(body: &[u8]) -> Vec<u8> {
    let mut bytes = body.to_vec();
    bytes.extend_from_slice(&Sha256::digest(body));
    bytes
}
fn fixture(g_max: f32) -> Vec<u8> {
    seal(&build_body(g_max, [0.0, 0.0]))
}
fn saturated(g_max: f32) -> Vec<u8> {
    seal(&build_body(g_max, [100.0, -100.0]))
}
fn rechecksum(bytes: &mut [u8]) {
    let split = bytes.len() - 32;
    let digest = Sha256::digest(&bytes[..split]);
    bytes[split..].copy_from_slice(&digest);
}

#[test]
fn v3_reader_shares_the_typed_ten_mps2_envelope() {
    assert_eq!(CommandEnvelope::DEFAULT.accel, 10.0);
    assert_eq!(LoadOptions::default().envelope, CommandEnvelope::DEFAULT);
    assert_eq!(LoadOptions::default().mode, CommandBoundMode::Auto);

    let m = V3Model::from_bytes(&fixture(10.0)).unwrap();
    assert_eq!(m.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(m.command_envelope, CommandEnvelope::DEFAULT);
    assert_eq!(m.g_max, 10.0);
    assert_eq!(m.correction_bound, 10.0);
}

#[test]
fn ten_mps2_v3_export_emits_linear_commands_up_to_the_envelope() {
    let m = V3Model::from_bytes(&saturated(10.0)).unwrap();
    assert_eq!(m.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(m.correction_bound, 10.0);

    let mut rt = Runtime::from_v3(m);
    let out = rt.infer_window(&[0.1; 8]);
    assert!(
        (out[0] - 10.0).abs() < 1e-3,
        "positive correction {0} is not +10 m/s²",
        out[0]
    );
    assert!(
        (out[1] + 10.0).abs() < 1e-3,
        "negative correction {0} is not -10 m/s²",
        out[1]
    );
    assert!((0.0..=1.0).contains(&out[2]));
    assert_eq!(rt.max_correction(), 10.0);
}

#[test]
fn legacy_v3_field_keeps_tanh_bound_and_can_be_forced_to_envelope() {
    let legacy = V3Model::from_bytes(&fixture(0.12)).unwrap();
    assert_eq!(legacy.bound_mode, CommandBoundMode::LegacyV2Tanh);
    assert!((legacy.correction_bound - 0.12f32.tanh()).abs() < 1e-7);

    let forced = V3Model::from_bytes_with_options(
        &fixture(0.12),
        LoadOptions {
            mode: CommandBoundMode::EnvelopeMps2,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(forced.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(forced.correction_bound, 0.12);

    // 1.0 is the legacy Python default: it must stay tanh-bounded.
    let one = V3Model::from_bytes(&fixture(1.0)).unwrap();
    assert_eq!(one.bound_mode, CommandBoundMode::LegacyV2Tanh);
    assert!((one.correction_bound - 1.0f32.tanh()).abs() < 1e-7);

    // A large legacy field stays legacy only when the caller opts in explicitly.
    let big = fixture(20.0);
    assert_eq!(
        V3Model::from_bytes(&big).unwrap_err(),
        V3Error::CommandOutOfEnvelope(20.0, 10.0)
    );
    let old = V3Model::from_bytes_with_options(
        &big,
        LoadOptions {
            mode: CommandBoundMode::LegacyV2Tanh,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert!((old.correction_bound - 20.0f32.tanh()).abs() < 1e-7);
}

#[test]
fn out_of_envelope_v3_export_is_refused_loudly() {
    let err = V3Model::from_bytes(&fixture(10.001)).unwrap_err();
    assert_eq!(err, V3Error::CommandOutOfEnvelope(10.001, 10.0));
    assert!(
        err.to_string()
            .contains("exceeds the command envelope ceiling 10 m/s^2"),
        "error must name the cause: {err}"
    );

    let tight = LoadOptions {
        mode: CommandBoundMode::EnvelopeMps2,
        envelope: CommandEnvelope::new(5.0).unwrap(),
    };
    assert_eq!(
        V3Model::from_bytes_with_options(&fixture(8.0), tight).unwrap_err(),
        V3Error::CommandOutOfEnvelope(8.0, 5.0)
    );

    let broken = LoadOptions {
        envelope: CommandEnvelope { accel: -1.0 },
        ..LoadOptions::default()
    };
    assert_eq!(
        V3Model::from_bytes_with_options(&fixture(10.0), broken).unwrap_err(),
        V3Error::InvalidCommandEnvelope(-1.0)
    );
}

#[test]
fn non_finite_or_nonpositive_bound_is_refused() {
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            V3Model::from_bytes(&fixture(bad)).unwrap_err(),
            V3Error::InvalidCommandBound(_)
        ));
    }
}

#[test]
fn header_and_record_bound_must_agree() {
    let mut bytes = fixture(10.0);
    put_f32(&mut bytes, 48, 20.0); // header only; record stays 10.0
    rechecksum(&mut bytes);
    assert_eq!(
        V3Model::from_bytes(&bytes).unwrap_err(),
        V3Error::CommandBoundMismatch(20.0, 10.0)
    );
}

#[test]
fn sha_and_export_version_compatibility_is_unchanged() {
    let bytes = fixture(10.0);
    let mut bad_sha = bytes.clone();
    *bad_sha.last_mut().unwrap() ^= 1;
    assert_eq!(
        V3Model::from_bytes(&bad_sha).unwrap_err(),
        V3Error::ChecksumMismatch
    );

    let mut v2 = bytes.clone();
    put_u32(&mut v2, 8, 2);
    rechecksum(&mut v2);
    assert_eq!(
        V3Model::from_bytes(&v2).unwrap_err(),
        V3Error::UnsupportedVersion(2)
    );

    let mut bad_target = bytes;
    put_u32(&mut bad_target, 56, 2);
    rechecksum(&mut bad_target);
    assert_eq!(
        V3Model::from_bytes(&bad_target).unwrap_err(),
        V3Error::BadTargetVersion(2)
    );
}

#[test]
fn runtime_load_wires_potokm3_through_the_command_envelope() {
    let dir = std::env::temp_dir().join(format!("potok-v3-envelope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fast.bin");
    std::fs::write(&path, saturated(10.0)).unwrap();

    let mut rt = Runtime::load(&path).unwrap();
    assert!(rt.is_v3());
    let m = rt.v3_model().unwrap();
    assert_eq!(m.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(m.command_envelope, CommandEnvelope::DEFAULT);
    assert_eq!(rt.max_correction(), 10.0);
    let out = rt.infer_window(&[0.1; 8]);
    assert!((out[0] - 10.0).abs() < 1e-3, "got {}", out[0]);
    assert!((out[1] + 10.0).abs() < 1e-3, "got {}", out[1]);

    // The live path keeps the loud refusal too: a v3 file above ±10 is not
    // silently clamped by Runtime::load.
    std::fs::write(&path, fixture(10.001)).unwrap();
    let err = Runtime::load(&path).err().unwrap();
    assert!(
        err.contains("exceeds the command envelope ceiling 10 m/s^2"),
        "{err}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn typed_flag_is_available_on_the_v3_file_loading_path() {
    let dir = std::env::temp_dir().join(format!("potok-v3-envelope-file-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fast.bin");
    std::fs::write(&path, fixture(10.0)).unwrap();

    let m = V3Model::from_file_with_options(
        &path,
        LoadOptions {
            mode: CommandBoundMode::EnvelopeMps2,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(m.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(m.correction_bound, 10.0);
    assert_eq!(
        V3Model::from_file_with_options(
            &path,
            LoadOptions {
                envelope: CommandEnvelope::new(5.0).unwrap(),
                ..LoadOptions::default()
            },
        )
        .unwrap_err(),
        V3Error::CommandOutOfEnvelope(10.0, 5.0)
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
