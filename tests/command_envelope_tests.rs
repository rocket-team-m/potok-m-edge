//! Typed command-envelope gate for the v2 student reader (FIXRECIPE, 2026-11-23).
//!
//! `g_max` used to be compressed by `tanh`, capping every correction below
//! 1 m/s² while the v3 target reaches ±10. New exports carry `g_max` as a
//! linear bound; this file pins the reader's mode resolution, the ±10 default,
//! the loud out-of-range refusal, and unchanged SHA/version compatibility.

use potok_edge::student_runtime::{
    build_student_fixture, model_kind, CommandBoundMode, CommandEnvelope, LoadOptions,
    StudentError, StudentFixtureConfig, StudentModel, StudentRuntime, HEADER_BYTES, VERSION,
};
use sha2::{Digest, Sha256};

fn fixture(g_max: f32) -> Vec<u8> {
    build_student_fixture(
        StudentFixtureConfig {
            g_max,
            ..StudentFixtureConfig::default()
        },
        7,
    )
}

fn rechecksum(bytes: &mut [u8]) {
    let split = bytes.len() - 32;
    let digest = Sha256::digest(&bytes[..split]);
    bytes[split..].copy_from_slice(&digest);
}

/// Drive the correction head to saturation so the emitted command equals the
/// active bound. Rewrites one named f32 record and refreshes the SHA trailer.
fn saturate_head(mut bytes: Vec<u8>, bias: [f32; 2]) -> Vec<u8> {
    let split = bytes.len() - 32;
    let mut off = HEADER_BYTES;
    while off < split {
        let kind = bytes[off];
        off += 1;
        let name_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let name = std::str::from_utf8(&bytes[off..off + name_len])
            .unwrap()
            .to_owned();
        off += name_len;
        let payload = match kind {
            2 => {
                let ndim = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                let mut n = 1usize;
                for d in 0..ndim {
                    n *= i32::from_le_bytes(
                        bytes[off + 4 + 4 * d..off + 8 + 4 * d].try_into().unwrap(),
                    ) as usize;
                }
                4 + 4 * ndim + 4 * n
            }
            1 => {
                let d_in = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                let d_out =
                    u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
                let rank =
                    u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()) as usize;
                20 + d_in * rank + 4 * rank + d_out * rank
            }
            other => panic!("unexpected layer kind {other}"),
        };
        if name == "head_corr.bias" {
            assert_eq!(kind, 2);
            let data = off + 8; // ndim + one dimension
            for (k, v) in bias.iter().enumerate() {
                bytes[data + 4 * k..data + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
            }
            rechecksum(&mut bytes);
            return bytes;
        }
        off += payload;
    }
    panic!("head_corr.bias record not found");
}

#[test]
fn command_envelope_is_typed_and_defaults_to_ten_mps2() {
    assert_eq!(CommandEnvelope::DEFAULT.accel, 10.0);
    assert_eq!(
        CommandEnvelope::new(10.0).unwrap(),
        CommandEnvelope::DEFAULT
    );
    for bad in [0.0, -10.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            CommandEnvelope::new(bad),
            Err(StudentError::InvalidCommandEnvelope(_))
        ));
    }
    assert_eq!(LoadOptions::default().envelope, CommandEnvelope::DEFAULT);
    assert_eq!(LoadOptions::default().mode, CommandBoundMode::Auto);
}

#[test]
fn ten_mps2_export_emits_linear_commands_up_to_the_envelope() {
    let bytes = saturate_head(fixture(10.0), [100.0, -100.0]);
    let model = StudentModel::from_bytes(&bytes).unwrap();
    assert_eq!(model.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(model.command_envelope, CommandEnvelope::DEFAULT);
    assert_eq!(model.g_max, 10.0);
    assert_eq!(model.correction_bound, 10.0);

    let mut rt = StudentRuntime::new(model);
    let out = rt.step(&[0.1; 10]);
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
}

#[test]
fn legacy_v2_field_keeps_tanh_bound_and_can_be_forced_to_envelope() {
    let bytes = fixture(0.1);
    let legacy = StudentModel::from_bytes(&bytes).unwrap();
    assert_eq!(legacy.bound_mode, CommandBoundMode::LegacyV2Tanh);
    assert!((legacy.correction_bound - 0.1f32.tanh()).abs() < 1e-7);

    let forced = StudentModel::from_bytes_with_options(
        &bytes,
        LoadOptions {
            mode: CommandBoundMode::EnvelopeMps2,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(forced.bound_mode, CommandBoundMode::EnvelopeMps2);
    assert_eq!(forced.correction_bound, 0.1);

    // 1.0 is the legacy Python default and the shipped smart fixture: it must
    // stay tanh-bounded even though tanh(1.0) < 1.
    let one = StudentModel::from_bytes(&fixture(1.0)).unwrap();
    assert_eq!(one.bound_mode, CommandBoundMode::LegacyV2Tanh);
    assert!((one.correction_bound - 1.0f32.tanh()).abs() < 1e-7);

    // A pre-FIXRECIPE export with a large shape field stays legacy only when
    // the caller opts in explicitly; the default Auto path refuses it.
    let big = fixture(20.0);
    assert_eq!(
        StudentModel::from_bytes(&big).unwrap_err(),
        StudentError::CommandOutOfEnvelope(20.0, 10.0)
    );
    let old = StudentModel::from_bytes_with_options(
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
fn out_of_envelope_export_is_refused_loudly() {
    let bytes = fixture(10.001);
    let err = StudentModel::from_bytes(&bytes).unwrap_err();
    assert_eq!(err, StudentError::CommandOutOfEnvelope(10.001, 10.0));
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
        StudentModel::from_bytes_with_options(&fixture(8.0), tight).unwrap_err(),
        StudentError::CommandOutOfEnvelope(8.0, 5.0)
    );

    // An out-of-range bound at the typed boundary is refused too.
    let broken = LoadOptions {
        envelope: CommandEnvelope { accel: -1.0 },
        ..LoadOptions::default()
    };
    assert!(matches!(
        StudentModel::from_bytes_with_options(&fixture(10.0), broken),
        Err(StudentError::InvalidCommandEnvelope(_))
    ));
}

#[test]
fn non_finite_or_nonpositive_bound_is_refused() {
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            StudentModel::from_bytes(&fixture(bad)),
            Err(StudentError::InvalidCommandBound(_))
        ));
    }
}

#[test]
fn header_and_record_bound_must_agree() {
    let mut bytes = fixture(10.0);
    bytes[44..48].copy_from_slice(&20.0f32.to_le_bytes());
    rechecksum(&mut bytes);
    assert_eq!(
        StudentModel::from_bytes(&bytes).unwrap_err(),
        StudentError::CommandBoundMismatch(20.0, 10.0)
    );
}

#[test]
fn sha_and_export_version_compatibility_is_unchanged() {
    let bytes = fixture(10.0);
    assert_eq!(model_kind(&bytes).unwrap(), 2);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        VERSION
    );

    let mut bad_sha = bytes.clone();
    *bad_sha.last_mut().unwrap() ^= 1;
    assert_eq!(
        StudentModel::from_bytes(&bad_sha).unwrap_err(),
        StudentError::ChecksumMismatch
    );

    let mut v3 = bytes.clone();
    v3[8..12].copy_from_slice(&3u32.to_le_bytes());
    rechecksum(&mut v3);
    assert_eq!(
        StudentModel::from_bytes(&v3).unwrap_err(),
        StudentError::UnsupportedVersion(3)
    );
    assert_eq!(
        model_kind(&v3).unwrap_err(),
        StudentError::UnsupportedVersion(3)
    );
}

#[test]
fn typed_flag_is_available_on_the_file_loading_path() {
    let dir = std::env::temp_dir().join(format!("potok-envelope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("student.bin");
    std::fs::write(&path, fixture(10.0)).unwrap();

    let rt = StudentRuntime::from_file_with_options(
        &path,
        LoadOptions {
            mode: CommandBoundMode::EnvelopeMps2,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(rt.model().correction_bound, 10.0);
    assert_eq!(
        StudentRuntime::from_file_with_options(
            &path,
            LoadOptions {
                envelope: CommandEnvelope::new(5.0).unwrap(),
                ..LoadOptions::default()
            },
        )
        .err()
        .unwrap(),
        StudentError::CommandOutOfEnvelope(10.0, 5.0)
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
