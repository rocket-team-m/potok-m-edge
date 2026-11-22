//! Gates for the canonical v3 export reader. Every byte sequence is built in
//! memory: no model binary is added to the repository.

use potok_edge::model_v3::{
    V3Error, V3Model, COND_FEATURES, FP32_LAYER, IN_FEATURES, MAGIC, MODEL_KIND_FAST, OUT_FEATURES,
    PHYSICS_FEATURES, TARGET_VERSION, TSCT_LAYER, VERSION,
};
use sha2::{Digest, Sha256};

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

#[derive(Default)]
struct Opts<'a> {
    omit: &'a [&'a str],
    duplicate_first: bool,
    corrupt_code: bool,
    bad_shape: bool,
    extra_trailing: bool,
    version: Option<u32>,
    kind: Option<u32>,
    target: Option<u32>,
    magic: Option<[u8; 8]>,
    n_records: Option<u32>,
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
fn f32_record(b: &mut Vec<u8>, name: &str, shape: &[usize], data: &[f32], bad_shape: bool) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    push_name(b, FP32_LAYER, name);
    let shape: Vec<usize> = if bad_shape {
        let mut s = shape.to_vec();
        s.push(1);
        s
    } else {
        shape.to_vec()
    };
    push_u32(b, shape.len() as u32);
    for &d in &shape {
        push_u32(b, d as u32);
    }
    // The reader decodes dimensions as i32; positive u32 values are identical.
    for &v in data {
        push_f32(b, v);
    }
}
fn tsct_record(b: &mut Vec<u8>, rng: &mut Lcg, name: &str, rank: usize, w: usize, corrupt: bool) {
    push_name(b, TSCT_LAYER, name);
    push_u32(b, w as u32);
    push_u32(b, w as u32);
    push_u32(b, rank as u32);
    push_f32(b, 1.0);
    push_f32(b, 1.0);
    for i in 0..w * rank {
        b.push(if corrupt && i == 0 {
            7
        } else {
            rng.code() as u8
        });
    }
    for _ in 0..rank {
        push_f32(b, rng.scale());
    }
    for _ in 0..w * rank {
        b.push(rng.code() as u8);
    }
}

/// Body only (no digest), so corruption tests can reseal a modified body.
fn build_body(spec: &Spec, seed: u64, opts: &Opts<'_>) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    for b in 0..spec.n_blocks {
        let prefix = format!("blocks.{b}");
        for (suffix, rows, cols) in [
            ("rec_h.weight", spec.width, spec.width),
            ("rec_x.weight", spec.width, spec.width),
        ] {
            let data: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 0.08).collect();
            let mut rec = Vec::new();
            f32_record(
                &mut rec,
                &format!("{prefix}.{suffix}"),
                &[rows, cols],
                &data,
                false,
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
                false,
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
            opts.corrupt_code && b == 0,
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
        f32_record(
            &mut rec,
            name,
            &shape,
            &data,
            opts.bad_shape && name == "norm.bias",
        );
        records.push((name.to_owned(), rec));
    }

    let canonical_records = 5 * spec.n_blocks + 10;
    records.retain(|(name, _)| !opts.omit.contains(&name.as_str()));
    // Keep the declared record count canonical: an omitted required layer is
    // replaced by a valid unused record, so the reader reaches the name lookup
    // instead of failing while counting.
    while records.len() < canonical_records {
        let name = format!("unused.{}", records.len());
        let mut rec = Vec::new();
        f32_record(&mut rec, &name, &[1], &[0.0], false);
        records.push((name, rec));
    }
    if opts.duplicate_first {
        // Collide a name without changing the declared record count.
        let name = records[0].0.clone();
        let mut rec = Vec::new();
        f32_record(&mut rec, &name, &[1], &[0.0], false);
        *records.last_mut().unwrap() = (name, rec);
    }
    if opts.extra_trailing {
        records.push(("extra".to_owned(), vec![0u8]));
    }

    let header_len = 64 + 4 * spec.n_blocks;
    let mut body = vec![0u8; header_len];
    body[..8].copy_from_slice(&opts.magic.unwrap_or(MAGIC));
    put_u32(&mut body, 8, opts.version.unwrap_or(VERSION));
    put_u32(&mut body, 12, opts.kind.unwrap_or(MODEL_KIND_FAST));
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
    put_u32(&mut body, 56, opts.target.unwrap_or(TARGET_VERSION));
    put_u32(
        &mut body,
        60,
        opts.n_records.unwrap_or(canonical_records as u32),
    );
    for b in 0..spec.n_blocks {
        put_u32(&mut body, 64 + 4 * b, (b + 1) as u32);
    }
    for (_, rec) in records {
        body.extend_from_slice(&rec);
    }
    body
}

fn seal(body: &[u8]) -> Vec<u8> {
    let mut bytes = body.to_vec();
    bytes.extend_from_slice(&Sha256::digest(body));
    bytes
}

fn build(spec: &Spec, seed: u64, opts: &Opts<'_>) -> Vec<u8> {
    seal(&build_body(spec, seed, opts))
}

#[test]
fn canonical_fixture_roundtrips_without_a_model_binary() {
    let spec = Spec::default();
    let bytes = build(&spec, 7, &Opts::default());
    assert_eq!(
        bytes,
        build(&spec, 7, &Opts::default()),
        "fixture is deterministic"
    );

    let m = V3Model::from_bytes(&bytes).unwrap();
    assert_eq!(m.in_features, IN_FEATURES);
    assert_eq!(PHYSICS_FEATURES + COND_FEATURES, IN_FEATURES);
    assert_eq!(m.width, spec.width);
    assert_eq!(m.out_features, OUT_FEATURES);
    assert_eq!(m.rank, spec.rank);
    assert_eq!(m.n_blocks, spec.n_blocks);
    assert_eq!(m.kernel, spec.kernel);
    assert_eq!(m.n_loops, spec.n_loops);
    assert_eq!(m.target_version, TARGET_VERSION);
    assert_eq!(m.dilations, vec![1, 2]);
    assert_eq!(m.correction_bound, 0.12f32.tanh());
    assert_eq!(m.stem_w.len(), spec.width * IN_FEATURES);
    assert_eq!(m.blocks.len(), spec.n_blocks);
    assert_eq!(m.blocks[0].proj_w.len(), spec.width * spec.width);
    assert!(m
        .blocks
        .iter()
        .flat_map(|b| b.proj_w.iter().chain(&b.rec_h_w).chain(&b.rec_x_w))
        .all(|v| v.is_finite()));
    assert!(m.memory_bytes() > 0);
}

#[test]
fn sha256_mismatch_is_rejected_before_parsing() {
    let mut bytes = build(&Spec::default(), 7, &Opts::default());
    *bytes.last_mut().unwrap() ^= 1;
    assert_eq!(
        V3Model::from_bytes(&bytes).unwrap_err(),
        V3Error::ChecksumMismatch
    );
}

#[test]
fn short_body_and_truncated_record_are_loud() {
    assert_eq!(
        V3Model::from_bytes(&[0u8; 16]).unwrap_err(),
        V3Error::Truncated
    );
    let body = build_body(&Spec::default(), 7, &Opts::default());
    let cut = seal(&body[..body.len() - 1]);
    assert_eq!(V3Model::from_bytes(&cut).unwrap_err(), V3Error::Truncated);
}

#[test]
fn magic_version_kind_target_and_counts_are_checked() {
    let spec = Spec::default();
    for (opts, want) in [
        (
            Opts {
                magic: Some(*b"BADMAGIC"),
                ..Opts::default()
            },
            V3Error::BadMagic,
        ),
        (
            Opts {
                version: Some(2),
                ..Opts::default()
            },
            V3Error::UnsupportedVersion(2),
        ),
        (
            Opts {
                kind: Some(1),
                ..Opts::default()
            },
            V3Error::BadModelKind(1),
        ),
        (
            Opts {
                target: Some(2),
                ..Opts::default()
            },
            V3Error::BadTargetVersion(2),
        ),
        (
            Opts {
                n_records: Some(99),
                ..Opts::default()
            },
            V3Error::InvalidField("header counts"),
        ),
    ] {
        assert_eq!(
            V3Model::from_bytes(&build(&spec, 7, &opts)).unwrap_err(),
            want
        );
    }
}

#[test]
fn corrupt_ternary_code_is_loud() {
    let opts = Opts {
        corrupt_code: true,
        ..Opts::default()
    };
    let err = V3Model::from_bytes(&build(&Spec::default(), 7, &opts)).unwrap_err();
    assert_eq!(err, V3Error::InvalidTernaryCode(7));
}

#[test]
fn missing_and_duplicate_layers_are_loud() {
    let opts = Opts {
        omit: &["norm.bias"],
        ..Opts::default()
    };
    assert_eq!(
        V3Model::from_bytes(&build(&Spec::default(), 7, &opts)).unwrap_err(),
        V3Error::MissingLayer("norm.bias".to_owned())
    );
    let opts = Opts {
        duplicate_first: true,
        ..Opts::default()
    };
    assert!(matches!(
        V3Model::from_bytes(&build(&Spec::default(), 7, &opts)).unwrap_err(),
        V3Error::DuplicateLayer(_)
    ));
}

#[test]
fn bad_shape_and_trailing_bytes_are_loud() {
    let opts = Opts {
        bad_shape: true,
        ..Opts::default()
    };
    assert_eq!(
        V3Model::from_bytes(&build(&Spec::default(), 7, &opts)).unwrap_err(),
        V3Error::ShapeMismatch("tensor shape")
    );
    let opts = Opts {
        extra_trailing: true,
        ..Opts::default()
    };
    assert_eq!(
        V3Model::from_bytes(&build(&Spec::default(), 7, &opts)).unwrap_err(),
        V3Error::InvalidField("trailing bytes")
    );
}

#[test]
fn from_file_reads_the_same_bytes() {
    let bytes = build(&Spec::default(), 7, &Opts::default());
    let path = std::env::temp_dir().join(format!("potok-v3-reader-{}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let from_file = V3Model::from_file(&path).unwrap();
    let from_bytes = V3Model::from_bytes(&bytes).unwrap();
    assert_eq!(from_file.memory_bytes(), from_bytes.memory_bytes());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn error_display_names_the_failure() {
    let msg = V3Error::ChecksumMismatch.to_string();
    assert!(msg.contains("SHA256"), "{msg}");
}
