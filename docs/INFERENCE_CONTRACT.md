<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Edge inference contract — potok-m-edge

This document states the on-device inference contract. The Rust sources named in
each section are the authority; where a document and the code disagree, the code
wins. Own code is `GPL-3.0-or-later` (`Cargo.toml:6`, `LICENSE`); a commercial
option exists (`LICENSE-COMMERCIAL.md`).

## 1. Boundary

Raspberry Pi 4 B is an advisor. STM32 owns flight phases, PID, interlocks and
servos. Two bounded corrections, a confidence and `ai_healthy` cross the wire;
the controller may reject them. On the active segment the assistant is silent.
Three consecutive misses or `ai_healthy = 0` disable the assistant for the rest
of the flight (`README.md:7-9`, `docs/src/content/docs/ai/concept.md` in
`potok-m`). The authoritative v3 command is `[g_accel_y, -g_accel_x]` in m/s²;
the angle target `g_theta` is diagnostic only (`src/speculative_v3.rs:5-8`).

## 2. Model formats

Two export families are read. Both are little-endian, f32 IEEE-754, signed
int8 ternary codes (`-1, 0, 1`), UTF-8 names without a NUL terminator, no
padding, and a 32-byte SHA-256 trailer.

### 2.1 Canonical layer records (shared by v2 and v3)

`kind: u8`, `name_len: u32`, `name: UTF-8`, then payload:

| kind | payload |
|---|---|
| `1` — TSCT | `d_in, d_out, rank: u32`; `alpha_u, alpha_v: f32`; `d_in*rank` codes; `rank` f32 scales; `d_out*rank` codes |
| `2` — f32 tensor | `ndim: u32`; `ndim` positive `i32` dimensions; row-major f32 data |

Factor order is `W[out, in] = U[out,k] · alpha_u · s[k] · V[in,k] · alpha_v`.
`U` indexes the output, `V` the input (`src/format_v2.rs:312-325`,
`src/model_v3.rs:380-389`). All factors are dequantized to dense f32 **once at
load**; the hot loop never reads a code byte and allocates no buffers.

### 2.2 v2 windowed teacher — `POTOKM1\0`, version 2

Header after the 8-byte magic (`src/format_v2.rs:14-15, 182-211`):

| field | value |
|---|---|
| `version: u32` | 2 |
| `in_features` | 8 |
| `width`, `rank`, `kernel`, `n_blocks` | positive |
| `out_features` | 3 (`g_pitch`, `g_yaw`, confidence) |
| `g_max: f32` | head bound |
| `dilations[n_blocks]: u32` | positive |
| `flags: u32` | stored verbatim |
| `n_records: u32` | `5 * n_blocks + 6` |

Required names: `stem.{weight,bias}`, `blocks.{i}.{dw.weight,dw.bias,norm.weight,norm.bias,pw}`,
`norm.{weight,bias}`, `head.{weight,bias}`. `pw` is a TSCT record; the rest are
f32 tensors with the shapes asserted in `src/format_v2.rs:286-337`. The window
is a fixed 50×8 frame. Forward: stem → per-block causal dilated depthwise conv
reading the frozen block input → LayerNorm after conv → TSCT pointwise with
residual → final LayerNorm → `tanh` head scaled by `g_max`, sigmoid confidence
(`src/format_v2.rs:395-457`).

### 2.3 v2 stateful student — `model_kind = 2`

The same magic and version, a 56-byte header with `model_kind = 2` at offset 12
(teacher files hold `in_features = 8` there), and `n_records` at offset 52
(`src/student_runtime.rs:1-30, 55-70, 122-138`). `in_features` is **10** for the
legacy frame (8 sensors + phase + flags) or **20** for the smart frame (14
physics features + 6 conditioning values: `mass_kg, area_m2, cd0,
static_margin_cal, thrust_peak_n, gyro_noise`). `out_features = 2`; loops
`1..=3`. Records: `blocks.{b}.proj` (TSCT), `blocks.{b}.rec_h/rec_x.{weight,bias}`,
`stem`, `norm`, `alloc_logits`, `head_corr`, `head_conf`, `g_max`. Forward
mirrors `potok_ai.student.py`: leaky recurrent state `h = 0.8·h + 0.2·tanh(...)`,
loop outputs accumulated by `softmax(alloc_logits)·decay^loop`,
`LayerNorm(total + source)`, corrections `tanh(|g_max|)·tanh(head_corr(z))`,
confidence `sigmoid(head_conf(z))` (`src/student_runtime.rs:32-46`).

### 2.4 v3 fast model — `POTOKM3\0`, version 3

64-byte fixed header plus `4 * n_blocks` dilation words
(`src/model_v3.rs:1-46, 54-68, 228-287`):

| offset | type | field | required value |
|---:|---|---|---|
| 0 | 8 bytes | magic | `POTOKM3\0` |
| 8 | u32 | version | 3 |
| 12 | u32 | model_kind | 2 (stateful fast model) |
| 16 | u32 | in_features | 20 (14 physics + 6 conditioning) |
| 20 | u32 | width | > 0 |
| 24 | u32 | out_features | 2 (roll, pitch corrections) |
| 28 | u32 | rank | > 0 |
| 32 | u32 | n_blocks | > 0 |
| 36 | u32 | kernel | > 0 |
| 40 | u32 | n_loops | 1..=3 |
| 44 | f32 | decay | finite, 0..=1 |
| 48 | f32 | g_max | finite, > 0 |
| 52 | u32 | flags | exactly 1 (bit 0 = ternary factors; others must be 0) |
| 56 | u32 | target_version | 3 |
| 60 | u32 | n_records | `5 * n_blocks + 10` |
| 64+4i | u32 | dilations[i] | positive |

Records: `blocks.{b}.{rec_h,rec_x}.{weight,bias}`, `blocks.{b}.proj` (TSCT),
`stem.{weight,bias}`, `norm.{weight,bias}`, `alloc_logits`,
`head_corr.{weight,bias}`, `head_conf.{weight,bias}`, `g_max` (shape `[1]`,
finite, > 0). The head bound exposed to consumers is `tanh(|g_max|)`
(`src/model_v3.rs:358-426`).

**Wiring state.** `V3Model` is a canonical reader exported from the library
(`src/lib.rs:15, 27`), and `src/speculative_v3.rs` is a predictor-agnostic
dispatch/guard with fixture-driven gates. `Runtime::load` currently dispatches
only on `POTOKM1\0` versus the legacy POTM reader (`src/runtime.rs:149-163`); a
`POTOKM3\0` file is not loaded by the production entry point, and calibration is
not applied in the dispatch path (§4). These are integration gaps, not format
differences.

## 3. SHA policy

- The trailer is **32 raw bytes** of SHA-256, appended after the last record.
  The digest covers every byte from offset 0 up to, but not including, those
  final 32 bytes (`src/format_v2.rs:168-177`, `src/model_v3.rs:226-236`,
  `src/student_runtime.rs:4-5, 69`).
- A mismatch is a hard error: no fallback, no partial load, no proceeding with
  an unverified file. All readers return before any layer is parsed.
- A file shorter than the header plus trailer is `Truncated`; a failed filesystem
  read is reported as a truncated file by the model readers, and as
  `cannot read <path>: <os error>` by `Runtime::load` (`src/runtime.rs:151-153`).
- The same raw-trailer rule applies to the v3 validation/target export from the
  simulator; its digest is a stable 32-byte trailer and changing a target
  column changes both digest and loaded command (`potok-m/ai/tests/test_v3_export_roundtrip.py`).
- SHA-256 is an integrity check, not authentication. CRC16 on the wire is a
  random-error check, not cryptographic protection (`potok-m/PROTOCOL_BUDGET.md`).

## 4. Calibration JSON

`src/calibration_v3.rs` reads two exports, with no dependency on `serde`:

| source | `method` | required fields |
|---|---|---|
| `potok_ai.calibration` → `calibration.json` | `temperature` | `temperature > 0` finite, `threshold` in [0,1]; **no** `bins`/`values` |
| `potok_ai.calibration` → `calibration.json` | `isotonic` | `bins`, `values`, `threshold` in [0,1] |
| `potok_ai.conformal` → `conformal.json` | `split-conformal` | `q_hat >= 0`, `tol > 0`, `alpha` in (0,1), positive-integer `n`, `coverage` in [0,1]; optional paired `bins`/`values` |

Transform of a pre-sigmoid logit `z` (`src/calibration_v3.rs:181-221`):

- temperature: `sigmoid(z / T)`.
- isotonic: `step(sigmoid(z))` — first upper edge `>= p`, holding the last value
  beyond the final edge.
- split-conformal: the same step when an exported map exists; otherwise the exact
  indicator `p >= 1 / (1 + q_hat)`. `p` is the export's confidence coordinate
  `1 / (1 + |error| / tol)`. At the boundary the two routes deliberately differ:
  the exported map uses `searchsorted(..., side="left")` and yields the lower
  value, the exact route yields 1 (`tests/calibration_v3_tests.rs:113-124`).
- Sigmoid clips `z` to `[-60, 60]`. Non-finite logits are refused.

Validation is total and loud: the root must be an object; duplicate JSON keys,
trailing characters, non-finite numbers, non-string `method`, missing keys,
wrong types, non-increasing or out-of-range bin edges, non-monotone or
out-of-range values, and `bins`/`values` supplied separately all fail. An `n+1`
edge array with a leading lower edge is accepted and normalised, matching the
Python reader. There is **no fallback to raw confidence**. The adaptive-compute
threshold is data-derived: `threshold` from the JSON for
temperature/isotonic, `1 / (1 + q_hat)` for conformal; never a constant.

`pub mod calibration_v3;` is present (`src/lib.rs:9`). No dispatch call site
consumes `Calibration` today: the v3 dispatch takes the confidence returned by
its predictor directly (`src/speculative_v3.rs:437-440, 489-494`). Until that
wiring exists, the contract for calibration is the reader and transform, not an
endpoint guarantee that every emitted confidence is calibrated.

## 5. Watchguard decisions

The guard is a pure, deterministic mirror of `potok_ai.watchguard.guard`
(`src/speculative_v3.rs:36-38`): no clock, no globals, no argument mutation.
Commands and limits are in m/s². Checks run in a fixed order so the reason is
reproducible (`src/speculative_v3.rs:150-211`):

| order | reason (`as_str`) | trigger | effect |
|---:|---|---|---|
| 1 | `latched` | state already latched | zeros, no state change |
| 2 | `non_finite` | command or confidence not finite | zeros, miss++ |
| 3 | `out_of_bounds` | any channel outside its inclusive bound | zeros, miss++ |
| 4 | `low_confidence` | `confidence < min_confidence` | zeros, miss++ |
| 5 | `rate_limit` | any `|v - last_accepted| > max_rate` | zeros, miss++ |
| — | `ok` | none of the above | command passed, misses cleared, `last_command` updated |

State semantics: a rejection emits `[0.0, 0.0]` and leaves `last_command` at the
**last accepted** value, so a spike cannot move the rate reference. Reaching
`max_misses` (default 3) latches the assistant off; a later valid sample does
not clear the latch. `reset()` returns fresh state and is the only latch clear
(`src/speculative_v3.rs:130-148`). Config validation is loud for non-finite or
inverted bounds, negative rates, `min_confidence` outside [0,1], and
`max_misses == 0` (`src/speculative_v3.rs:79-115`). The default bounds
`[(-20,20), (-20,20)]` are a conservative example, **not** a flight-certified
envelope; the caller must supply calibrated vehicle limits
(`src/speculative_v3.rs:117-128`).

Dispatch rules (`src/speculative_v3.rs:10-21, 417-519`):

1. the student predicts on every valid cycle;
2. low student confidence, invalid sensors, or the periodic forced check call
   the teacher — all three share **one** call site, capped at one teacher call
   per cycle by an assert and a benchmark invariant;
3. low confidence / invalid sensors apply the teacher command; a forced check
   also compares heads, logs a divergence and re-warms the student;
4. the chosen command passes through the guard before it is applied;
5. a malformed or non-finite frame bypasses both models and the guard, replies
   zero, and does not increment cycle counters.

Defaults: confidence threshold 0.7, forced check every 20 frames, divergence
factor 3.0, window 50 frames. `divergence_factor` at 3.0 is unreachable when
both heads are `tanh`-bounded by the same `g_max`; lowering it is the documented
test seam for the log+reset path (`BUILD_LOG.md:166-177`). Accepted vs rejected
guard calls and each reason are counted in `V3Counters`. The protocol's
`ai_healthy` is not a guard input; the integration must AND it with
`decision.accepted`.

## 6. Cycle budget

- Full cycle budget: **< 10 ms p99**, from request decode through dispatch to
  reply encode plus wire time (`potok-m/docs/src/content/docs/ai/concept.md`,
  `potok-m/PI4_BUDGET.md:31-45`). 50 Hz service loop: 20 ms period, 8 ms read
  timeout; three misses disable the assistant (`src/uart.rs:14-15, 98-99`).
- Wire: compat 70 B/cycle, ~608 µs at 921600 baud; lowlat 58 B/cycle, 232 µs at
  2 Mbit/s (`BUILD_LOG.md:342-348`, `potok-m/PROTOCOL_BUDGET.md`).
- Host measurements (x86_64 dev box, release, 2000 frames, real exports):
  normal 47.4/67.3 µs p50/p99, low-confidence 294.4/461.5 µs, forced-check
  285.4/372.5 µs; worst cycle 1.071 ms including 0.61 ms UART wire, under the
  10 ms budget (`BUILD_LOG.md:250-253`). The bench gate fails loudly above
  10 ms (`src/bin/bench_spec.rs:238-245`).
- Status: p99 on a real Pi 4 B is **not measured**; NEON is compile-checked for
  aarch64, not executed here (`BUILD_LOG.md:63-67`, `potok-m/PI4_BUDGET.md:41-45`).
  A debug-build integration test has previously reported p99 above 10 ms
  (11.5 ms), a pre-existing timing failure unrelated to correctness
  (`potok-m/.superpowers/sdd/rocketcraft-implementation/task-B16-report.md:56-59`).
  Do not promote host numbers to flight evidence.

## 7. p50/p99 measurement

- Percentile definition used by this repo:
  `sorted[round((len - 1) * q)]`, on durations sorted ascending — nearest-rank,
  with the sample count fixed by `--frames` (`src/bin/bench.rs:14-20`,
  `src/bin/bench_v3.rs:62-67`, `src/bin/bench_spec.rs:28-33`). It is not the
  interpolation convention used elsewhere in the project; quote the instrument
  with the number.
- `bench` times `decode_request → on_request → encode_reply` as the full cycle
  and reports model-only and overhead-only percentiles plus memory and an output
  checksum; `--compare` adds the naive control (`src/bin/bench.rs:86-95, 137-169`).
- `bench_v3` builds its fixture in memory (seed `0x504F544B`), discards
  `--warmup` frames (default 100) before collecting 2000 samples by default,
  and prints model / overhead / total p50 and p99 plus a checksum. `--check`
  runs the pipeline twice and asserts the checksum is identical
  (`src/bin/bench_v3.rs:162-235, 251-258`).
- `bench_spec` times three paths (normal, low-confidence, forced-check) and two
  profiles (compat, lowlat). Full-cycle p50/p99 include CRC and
  quantize/dequantize; wire time is computed as `bytes * 8 / baud` and printed
  separately. The budget gate uses `max(path p99) + 0.61 ms` wire
  (`src/bin/bench_spec.rs:26, 37-113, 204-245`).
- Interpretation: quote frames, warmup, seed, profile, baud, build mode and host.
  Host p99 spread on a shared box is noise, not a signal (`BUILD_LOG.md:80-86`).
  P50 deltas and the checksum are the trustworthy cross-run comparisons.

## 8. Not in this repository

- **No production model binary.** The binary export is produced by
  `potok-m-ai` and delivered at deployment, then loaded into RAM once at startup;
  there is no disk read in the loop (`README.md:19-21`, `potok-m/PI4_BUDGET.md:37`).
- Small deterministic fixture binaries live under `tests/fixtures/` and are
  generated by `tests/fixtures/generate.py`; they are test data, not flight
  models. Model and v3 export fixtures are otherwise built in memory
  (`tests/model_v3_tests.rs:1-2`, `src/bin/bench_v3.rs:9-11`).
- No training, no exporter, no calibration fit and no GPU code here — those
  belong to `potok-m-ai`. The canonical wire layout lives in `potok-m-protocol`
  (a read-only dependency); `src/frame.rs` is an isolated local copy behind
  `FrameLayout` (`BUILD_LOG.md:18-21`).

## 9. Failure messages

Exact `Display` strings, by source. `{...}` is a formatted value.

**v2 / student / v3 model readers** (`src/format_v2.rs:35-51`,
`src/student_runtime.rs:87-103`, `src/model_v3.rs:87-105`):

```
v2 model file truncated | not a POTOKM1 model file | unsupported export version {v}
SHA256 checksum mismatch | invalid v2 field: {s} | unknown layer kind {k}
unexpected layer {s} | missing layer {s} | shape mismatch: {s}
invalid ternary code {c} | layer name is not UTF-8

student model file truncated | not a POTOKM1 model file | unsupported export version {v}
model_kind {k} is not the stateful student (2) | student SHA256 checksum mismatch
invalid student field: {s} | unknown student layer kind {k} | missing student layer {s}
student shape mismatch: {s} | invalid ternary code {c} | student layer name is not UTF-8

v3 model file truncated | not a POTOKM3 model file | unsupported export version {v}
model_kind {k} is not the v3 fast model (2) | unsupported v3 target version {v}
v3 SHA256 checksum mismatch | invalid v3 field: {s} | unknown v3 layer kind {k}
missing v3 layer {s} | duplicate v3 layer {s} | v3 shape mismatch: {s}
invalid ternary code {c} | v3 layer name is not UTF-8
```

**Calibration** (`src/calibration_v3.rs:43-52`):

```
calibration file: {s} | calibration JSON: {s} | calibration JSON: missing {k}
calibration JSON: {s} | unsupported calibration method {s:?}
```

**Frame / transport** (`src/frame.rs:46-58`, `src/uart.rs:111, 135, 147`):

```
frame is {got} bytes, need {want} | bad frame magic 0x{m:04X}
protocol version {v} unsupported | CRC16 mismatch: got 0x{got:04X}, want 0x{want:04X}
unknown phase {p} | potok-edge: UART read error: {e}
potok-edge: inference loop started (50 Hz, 8 ms read timeout)
potok-edge: assistant disabled after 3 consecutive misses; AI stays off until restart
```

A protocol version mismatch is fatal on the **first** frame; the three-strike
rule applies to CRC, sequence and timeout failures only (`BUILD_LOG.md:24-26`,
`src/uart.rs:61-70`).

**Guard and dispatch configuration** (`src/speculative_v3.rs:86-107, 288-309`):

```
bounds need finite low < high, got ({low}, {high})
max_rate must be finite and non-negative, got {rate}
min_confidence must be in [0, 1], got {min_confidence}
max_misses must be >= 1
frame_len must be >= 1 | confidence_threshold must be finite
forced_check_every must be >= 1 | divergence_factor must be finite | guard bounds must be finite
```

Rejection reasons are `ok`, `non_finite`, `out_of_bounds`, `low_confidence`,
`rate_limit`, `latched` (`src/speculative_v3.rs:50-61`). Dispatch sources are
`student`, `teacher/low-confidence`, `teacher/forced-check`,
`teacher/low-confidence+forced`, `zero` (`src/speculative_v3.rs:247-256`).

**CLI and budget gates** (`src/bin/infer.rs:38-58`, `src/bin/bench.rs:44-59`,
`src/bin/bench_spec.rs:153-159, 238-245`, `src/speculative.rs:152-160`):

```
usage: infer --model MODEL.bin --device /dev/ttyS0 [--profile compat|lowlat] [--baud N]
unknown profile {other}; expected compat or lowlat
model load failed: {e} | cannot read {path}: {e}
cannot read teacher: {e} | teacher path holds a model_kind=2 student file
student path is not a model_kind=2 file
BUDGET EXCEEDED
```
