// SPDX-License-Identifier: GPL-3.0-or-later
//! Hard, deterministic performance gates for the lowlat profile.
//!
//! This crate is pure Rust on the CPU (`Cargo.toml` has no GPU dependency and
//! nothing here calls a device API), so the whole benchmark is host-only.
//!
//! "Deterministic" refers to the *workload and the statistics*, not to the
//! wall clock. The frame bytes, model init and request sequence are fixed
//! functions of a literal seed; every reported figure is a pure function of
//! the collected durations, and the output of the timed loop has a pinned
//! checksum. Timing assertions are therefore structural (counts, finiteness,
//! quantile ordering, model/overhead accounting) with the tolerance constants
//! documented below — never a machine-specific microsecond threshold. Such a
//! threshold would be flaky on a shared dev box and meaningless on the Pi.

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{FLAG_IMU_OK, PHASE_LANDING};
use potok_edge::lowlat::LowlatLayout;
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use potok_edge::FrameLayout;
use potok_protocol::{
    crc16, encode_lowlat_request, LowlatRequestFrame, Phase, LOWLAT_REPLY_SIZE,
    LOWLAT_REQUEST_CRC_OFF, LOWLAT_REQUEST_SIZE, LOWLAT_REQUEST_VERSION,
};
use std::time::Instant;

const SEED: u64 = 0x504F544B;
const WARMUP_FRAMES: usize = 16;
const TIMED_FRAMES: usize = 128;
/// CRC calls per timed sample: amortises the timer read so one sample is a
/// measurement of CRC work, not of clock granularity.
const CRC_BATCH: usize = 32;

/// Documented tolerance for the model/overhead accounting check. Quantiles of
/// a sum are not the sum of quantiles, so `p50(total)` and `p50(model)+p50(overhead)`
/// need not be equal; a factor of 2 admits that gap and timer jitter while
/// still failing if a third, unaccounted stage appears in the cycle.
const SPLIT_TOLERANCE_FACTOR: f64 = 2.0;

/// Output fingerprint of the fixed `WARMUP_FRAMES + TIMED_FRAMES` request
/// sequence. Pinned in both debug and release: the workload is deterministic
/// and the figure is independent of how long each stage took.
const CYCLE_CHECKSUM: u64 = 0x9bfa;
/// Sum of the per-batch CRC accumulators, pinned across debug and release.
const CRC_CHECKSUM: u64 = 0x3ff180;

/// Linear-interpolation percentile (numpy `method="linear"` convention).
///
/// For `n` sorted samples the rank is `pos = q * (n - 1)` and the result
/// interpolates between the two neighbours. This differs from the nearest-rank
/// rule in `src/bin/bench.rs` by at most half a sample gap; the pinned literals
/// in `percentile_is_pinned_and_pure` make the choice unambiguous.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "empty sample set has no percentile");
    assert!((0.0..=1.0).contains(&q), "quantile {q} outside [0, 1]");
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    p50: f64,
    p99: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    Stats {
        p50: percentile(&sorted, 0.50),
        p99: percentile(&sorted, 0.99),
    }
}

fn to_us(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Fixed-seed request bytes. Feature values are a pure function of the frame
/// index, so two calls produce byte-identical wire data with no RNG involved.
fn lowlat_requests(count: usize) -> Vec<[u8; LOWLAT_REQUEST_SIZE]> {
    (0..count)
        .map(|i| {
            let mut features = [0i16; 14];
            for (k, f) in features.iter_mut().enumerate() {
                *f = ((i as i32 * 7 + k as i32 * 3) % 21 - 10) as i16;
            }
            let frame = LowlatRequestFrame {
                seq: (i + 1) as u16,
                time_us: i as u32 * 20_000,
                phase: Phase::try_from(PHASE_LANDING).unwrap(),
                version: LOWLAT_REQUEST_VERSION,
                flags: FLAG_IMU_OK,
                reserved: 0,
                features,
            };
            let mut bytes = [0u8; LOWLAT_REQUEST_SIZE];
            encode_lowlat_request(&frame, &mut bytes).unwrap();
            bytes
        })
        .collect()
}

struct Split {
    total: Vec<f64>,
    decode: Vec<f64>,
    model: Vec<f64>,
    encode: Vec<f64>,
    overhead: Vec<f64>,
    checksum: u64,
}

/// Full lowlat cycle: decode (magic, CRC, version, phase, dequantize) ->
/// model dispatch -> encode reply (quantize, CRC). Three timed stages; `total`
/// and `overhead` are derived per iteration from the same three timestamps.
fn bench_full_cycle(requests: &[[u8; LOWLAT_REQUEST_SIZE]]) -> Split {
    let layout = LowlatLayout::new();
    let mut rt =
        Runtime::new(Model::from_bytes(&build_fixture(FixtureConfig::default(), SEED)).unwrap());
    let mut reply_bytes = [0u8; LOWLAT_REPLY_SIZE];

    for bytes in requests.iter().take(WARMUP_FRAMES) {
        let req = layout.decode_request(bytes).unwrap();
        let reply = rt.on_request(&req);
        layout.encode_reply(&reply, &mut reply_bytes).unwrap();
    }

    let n = requests.len() - WARMUP_FRAMES;
    let mut total = Vec::with_capacity(n);
    let mut decode = Vec::with_capacity(n);
    let mut model = Vec::with_capacity(n);
    let mut encode = Vec::with_capacity(n);
    let mut overhead = Vec::with_capacity(n);
    let mut checksum = 0u64;
    for bytes in requests.iter().skip(WARMUP_FRAMES) {
        let t0 = Instant::now();
        let req = layout.decode_request(bytes).unwrap();
        let t1 = Instant::now();
        let reply = rt.on_request(&req);
        let t2 = Instant::now();
        layout.encode_reply(&reply, &mut reply_bytes).unwrap();
        let t3 = Instant::now();
        let d = to_us(t1.duration_since(t0));
        let m = to_us(t2.duration_since(t1));
        let e = to_us(t3.duration_since(t2));
        total.push(d + m + e);
        decode.push(d);
        model.push(m);
        encode.push(e);
        overhead.push(d + e);
        checksum = checksum
            .wrapping_add(u64::from(reply.seq))
            .wrapping_add(reply.g_pitch as u64)
            .wrapping_add(reply.g_yaw as u64)
            .wrapping_add(u64::from(reply.confidence));
    }
    Split {
        total,
        decode,
        model,
        encode,
        overhead,
        checksum,
    }
}

#[test]
fn percentile_is_pinned_and_pure() {
    let line: Vec<f64> = (1..=9).map(f64::from).collect();
    assert_eq!(percentile(&line, 0.0), 1.0);
    assert_eq!(percentile(&line, 0.5), 5.0);
    assert_eq!(percentile(&line, 0.99), 8.92);
    assert_eq!(percentile(&line, 1.0), 9.0);
    let ties = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_eq!(percentile(&ties, 0.5), 4.5);
    assert_eq!(percentile(&ties, 0.99), 8.86);

    let reordered = [9.0, 2.0, 7.0, 5.0, 4.0, 4.0, 5.0, 4.0];
    assert_eq!(
        percentile(&reordered, 0.5).to_bits(),
        percentile(&ties, 0.5).to_bits(),
        "percentile must not depend on input order"
    );
    let first = percentile(&line, 0.99);
    for _ in 0..8 {
        assert_eq!(percentile(&line, 0.99).to_bits(), first.to_bits());
    }
    assert!(stats(&line).p50 <= stats(&line).p99);
}

#[test]
#[should_panic(expected = "empty sample set")]
fn percentile_rejects_empty_samples() {
    percentile(&[], 0.5);
}

#[test]
#[should_panic(expected = "outside [0, 1]")]
fn percentile_rejects_quantile_above_one() {
    percentile(&[1.0, 2.0], 1.01);
}

#[test]
#[should_panic(expected = "outside [0, 1]")]
fn percentile_rejects_negative_quantile() {
    percentile(&[1.0, 2.0], -0.01);
}

#[test]
fn lowlat_full_cycle_split_is_deterministic_and_structurally_sound() {
    let requests = lowlat_requests(WARMUP_FRAMES + TIMED_FRAMES);
    let a = bench_full_cycle(&requests);
    let b = bench_full_cycle(&requests);

    // Same fixed workload -> same model output, regardless of timing noise.
    assert_eq!(
        a.checksum, b.checksum,
        "timed workload must be deterministic"
    );
    assert_eq!(
        a.checksum, CYCLE_CHECKSUM,
        "lowlat cycle output changed (debug and release must agree)"
    );

    for (name, samples) in [
        ("total", &a.total),
        ("decode", &a.decode),
        ("model", &a.model),
        ("encode", &a.encode),
        ("overhead", &a.overhead),
    ] {
        assert_eq!(samples.len(), TIMED_FRAMES, "{name} sample count");
        assert!(
            samples.iter().all(|s| s.is_finite() && *s >= 0.0),
            "{name}: non-finite or negative sample"
        );
        let st = stats(samples);
        assert!(
            st.p50 > 0.0 && st.p50 <= st.p99 && st.p99.is_finite(),
            "{name}: bad quantiles {st:?}"
        );
        assert_eq!(
            stats(samples).p50.to_bits(),
            st.p50.to_bits(),
            "{name}: statistic must recompute bit-identically"
        );
    }

    let t = stats(&a.total);
    let d = stats(&a.decode);
    let m = stats(&a.model);
    let e = stats(&a.encode);
    let o = stats(&a.overhead);
    eprintln!(
        "cycle p50/p99 {:.2}/{:.2} us | decode {:.3}/{:.3} | model {:.2}/{:.2} | encode {:.3}/{:.3} | overhead {:.3}/{:.3}",
        t.p50, t.p99, d.p50, d.p99, m.p50, m.p99, e.p50, e.p99, o.p50, o.p99
    );

    // Each timed iteration satisfies total >= model and total >= overhead, so
    // the same ordering must hold for every quantile (no timing threshold).
    assert!(
        m.p50 <= t.p50 && m.p99 <= t.p99,
        "model exceeds total: {m:?} {t:?}"
    );
    assert!(
        o.p50 <= t.p50 && o.p99 <= t.p99,
        "overhead exceeds total: {o:?} {t:?}"
    );
    // overhead = decode + encode elementwise, so each part is dominated by it.
    assert!(d.p50 <= o.p50 && d.p99 <= o.p99, "decode exceeds overhead");
    assert!(e.p50 <= o.p50 && e.p99 <= o.p99, "encode exceeds overhead");
    assert!(
        t.p50 <= (m.p50 + o.p50) * SPLIT_TOLERANCE_FACTOR
            && t.p99 <= (m.p99 + o.p99) * SPLIT_TOLERANCE_FACTOR,
        "model+overhead accounting outside the documented {SPLIT_TOLERANCE_FACTOR}x tolerance: {t:?} {m:?} {o:?}"
    );
}

#[test]
fn lowlat_crc_microbench_is_deterministic() {
    let requests = lowlat_requests(TIMED_FRAMES);
    let mut samples = Vec::with_capacity(TIMED_FRAMES);
    let mut checksum = 0u64;
    for _ in 0..WARMUP_FRAMES {
        let _ = crc16(std::hint::black_box(&requests[0][..LOWLAT_REQUEST_CRC_OFF]));
    }
    for bytes in &requests {
        let t0 = Instant::now();
        let mut acc = 0u16;
        for _ in 0..CRC_BATCH {
            // Input and accumulator are loop-invariant without these barriers:
            // release mode collapsed all 32 calls into one (3 ns per reported
            // call). black_box keeps the microbench measuring CRC work.
            let input = std::hint::black_box(&bytes[..LOWLAT_REQUEST_CRC_OFF]);
            acc = std::hint::black_box(acc.wrapping_add(crc16(input)));
        }
        samples.push(to_us(t0.elapsed()) / CRC_BATCH as f64);
        checksum = checksum.wrapping_add(u64::from(acc));
    }

    assert_eq!(samples.len(), TIMED_FRAMES);
    assert!(
        samples.iter().all(|s| s.is_finite() && *s >= 0.0),
        "non-finite CRC sample"
    );
    let st = stats(&samples);
    assert!(
        st.p50 > 0.0 && st.p50 <= st.p99 && st.p99.is_finite(),
        "{st:?}"
    );
    assert_eq!(
        checksum, CRC_CHECKSUM,
        "CRC of the fixed request bytes changed"
    );
    eprintln!("crc p50/p99 {:.4}/{:.4} us/call", st.p50, st.p99);
}

#[test]
fn crc_guard_detects_lowlat_single_bit_corruption() {
    let requests = lowlat_requests(1);
    let good = crc16(&requests[0][..LOWLAT_REQUEST_CRC_OFF]);
    let mut corrupt = requests[0];
    corrupt[5] ^= 0x01;
    assert_ne!(crc16(&corrupt[..LOWLAT_REQUEST_CRC_OFF]), good);
    assert!(LowlatLayout::new().decode_request(&corrupt).is_err());
}
