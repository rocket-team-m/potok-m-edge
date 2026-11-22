//! `bench_v3` — deterministic, CPU-only microbenchmark of the Potok-M v3
//! inference path.
//!
//! SPDX-License-Identifier: GPL-3.0-or-later
//!
//! It separates the time spent inside the model forward from the framing
//! overhead (request encode/decode, sliding-window maintenance and reply
//! encode), and prints a stable output checksum so two runs can be compared
//! for correctness. No model binary is read from disk: the canonical
//! checksummed model is built in memory from `build_fixture`, so the benchmark
//! is reproducible on any checkout, with no external data and no GPU.
//!
//! `--check` runs the deterministic pipeline twice and asserts the checksum is
//! identical — a runnable determinism gate for the benchmark itself.
//!
//! Usage:
//!   cargo run --release --bin bench_v3 -- [--frames 2000] [--warmup 100] [--seed 1] [--check]

use potok_edge::fixture::{build_fixture, FixtureConfig};
use potok_edge::frame::{
    build_request, FrameLayout, PotokUartV1, ReplyFrame, RequestFrame, CORRECTION_SCALE,
    FLAG_IMU_OK, PHASE_LANDING, REPLY_LEN, REQUEST_LEN,
};
use potok_edge::model::Model;
use potok_edge::runtime::Runtime;
use std::collections::VecDeque;
use std::env;
use std::time::{Duration, Instant};

const DEFAULT_FRAMES: usize = 2000;
const DEFAULT_WARMUP: usize = 100;
const DEFAULT_SEED: u64 = 0x504F_544B; // "POTK"

/// Deterministic 32-bit xorshift; independent of the platform RNG and of any
/// floating-point library, so the same seed yields the same frames everywhere.
struct Rng(u32);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = (seed as u32) ^ ((seed >> 32) as u32);
        if s == 0 {
            s = 0xA5A5_A5A5;
        }
        Self(s)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1), using a power-of-two scale for exact conversion.
    fn unit(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Replace the oldest row of the sliding window and rebuild the flat buffer.
/// Mirrors `Runtime::on_request` so the model/overhead split is honest.
fn push_window(window: &mut VecDeque<[f32; 8]>, flat: &mut [f32], features: [f32; 8]) {
    window.pop_front();
    window.push_back(features);
    for (i, row) in window.iter().enumerate() {
        flat[i * 8..i * 8 + 8].copy_from_slice(row);
    }
}

struct Config {
    frames: usize,
    warmup: usize,
    seed: u64,
    check: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut cfg = Config {
        frames: DEFAULT_FRAMES,
        warmup: DEFAULT_WARMUP,
        seed: DEFAULT_SEED,
        check: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" if i + 1 < args.len() => {
                cfg.frames = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid frame count");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--warmup" if i + 1 < args.len() => {
                cfg.warmup = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid warmup count");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                cfg.seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid seed");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--check" => {
                cfg.check = true;
                i += 1;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!("usage: bench_v3 [--frames N] [--warmup N] [--seed N] [--check]");
                std::process::exit(2);
            }
        }
    }
    if cfg.frames == 0 {
        eprintln!("frame count must be positive");
        std::process::exit(2);
    }
    cfg
}

fn make_requests(n: usize, seed: u64) -> Vec<RequestFrame> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| RequestFrame {
            seq: (i + 1) as u16,
            time_us: (i as u32) * 20_000,
            phase: PHASE_LANDING,
            version: 1,
            flags: FLAG_IMU_OK,
            gyro: [rng.unit() * 0.05, rng.unit() * 0.05, rng.unit() * 0.05],
            accel: [rng.unit() * 0.2, 9.8 + rng.unit() * 0.2, rng.unit() * 0.2],
            mag: [1.0, 2.0, 3.0],
            height: 1000.0 + i as f32 * 0.1,
            pitot_dp: (i % 100) as i16,
        })
        .collect()
}

struct Samples {
    checksum: u64,
    model: Vec<Duration>,
    overhead: Vec<Duration>,
    total: Vec<Duration>,
    memory: usize,
}

/// One deterministic run. Timings are collected only for the measured frames;
/// warm-up frames prime the paths without polluting the percentiles.
fn run(cfg: &Config, requests: &[RequestFrame]) -> Samples {
    let bytes = build_fixture(FixtureConfig::default(), cfg.seed);
    let model = Model::from_bytes(&bytes).expect("in-crate fixture must parse");
    let memory = model.memory_bytes();
    let mut rt = Runtime::new(model);

    let layout = PotokUartV1;
    let mut req_bytes = [0u8; REQUEST_LEN];
    let mut rep_bytes = [0u8; REPLY_LEN];

    let window_len = FixtureConfig::default().window;
    let mut window = VecDeque::with_capacity(window_len);
    for _ in 0..window_len {
        window.push_back([0.0; 8]);
    }
    let mut flat = vec![0.0f32; window_len * 8];

    let mut samples = Samples {
        checksum: 0,
        model: Vec::with_capacity(cfg.frames),
        overhead: Vec::with_capacity(cfg.frames),
        total: Vec::with_capacity(cfg.frames),
        memory,
    };

    let bound = rt.max_correction();
    let clamp = |v: f32| -> i16 {
        (v.clamp(-bound, bound) / CORRECTION_SCALE)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    };

    for n in 0..(cfg.warmup + cfg.frames) {
        let req = &requests[n % requests.len()];
        build_request(req, &mut req_bytes).expect("request encode");

        let t0 = Instant::now();
        let decoded = layout.decode_request(&req_bytes).expect("request decode");
        push_window(&mut window, &mut flat, decoded.features());
        let t1 = Instant::now();

        let out = rt.infer_window(&flat);
        let t2 = Instant::now();

        let reply = ReplyFrame {
            seq: decoded.seq,
            g_pitch: clamp(out[0]),
            g_yaw: clamp(out[1]),
            confidence: (out[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ai_healthy: 1,
            version: 1,
        };
        layout
            .encode_reply(&reply, &mut rep_bytes)
            .expect("reply encode");
        let t3 = Instant::now();

        samples.checksum = samples
            .checksum
            .wrapping_add(out[0].to_bits() as u64)
            .wrapping_add((out[1].to_bits() as u64).rotate_left(17))
            .wrapping_add((out[2].to_bits() as u64).rotate_left(31))
            .wrapping_add(u64::from(rep_bytes[0]));

        if n >= cfg.warmup {
            samples.model.push(t2 - t1);
            samples.overhead.push((t1 - t0) + (t3 - t2));
            samples.total.push(t3 - t0);
        }
    }
    samples
}

fn report(label: &str, sorted: &[Duration]) {
    println!(
        "  {label:<10} p50 {:>8.1} us / p99 {:>8.1} us",
        percentile(sorted, 0.50).as_secs_f64() * 1e6,
        percentile(sorted, 0.99).as_secs_f64() * 1e6
    );
}

fn main() {
    let cfg = parse_args();
    let requests = make_requests(cfg.frames.max(1), cfg.seed);

    let mut s = run(&cfg, &requests);

    if cfg.check {
        let again = run(&cfg, &requests);
        assert_eq!(
            s.checksum, again.checksum,
            "determinism check failed: same seed produced different output"
        );
        println!("determinism check: PASS (checksum {:#018x})", s.checksum);
    }

    s.model.sort_unstable();
    s.overhead.sort_unstable();
    s.total.sort_unstable();

    println!(
        "potok-edge bench_v3: {} measured frames, {} warmup, seed {:#x}",
        cfg.frames, cfg.warmup, cfg.seed
    );
    report("model", &s.model);
    report("overhead", &s.overhead);
    report("total", &s.total);
    println!(
        "  memory   : {} bytes ({} KiB) — dequantized weights + buffers",
        s.memory,
        s.memory / 1024
    );
    println!("  checksum : {:#018x}", s.checksum);
}
