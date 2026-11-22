// SPDX-License-Identifier: GPL-3.0-or-later
//! Fixture-driven acceptance tests for v3 speculative dispatch and its guard.

use potok_edge::speculative_v3::{
    guard, reset, DispatchConfig, GuardConfig, GuardReason, V3Dispatch, V3Predictor, V3Source,
    WINDOW,
};

/// Deterministic command fixture: a fixed sequence, a confidence, and call/reset
/// probes. No model binary and no RNG in the hot path.
#[derive(Debug)]
struct FixturePredictor {
    commands: Vec<[f32; 2]>,
    pub confidence: f32,
    index: usize,
    calls: usize,
    resets: usize,
    last_input_len: usize,
}

impl FixturePredictor {
    fn new(command: [f32; 2], confidence: f32) -> Self {
        Self {
            commands: vec![command],
            confidence,
            index: 0,
            calls: 0,
            resets: 0,
            last_input_len: 0,
        }
    }

    fn sequence(commands: Vec<[f32; 2]>, confidence: f32) -> Self {
        Self {
            commands,
            confidence,
            index: 0,
            calls: 0,
            resets: 0,
            last_input_len: 0,
        }
    }
}

impl V3Predictor for FixturePredictor {
    fn predict(&mut self, input: &[f32]) -> potok_edge::speculative_v3::V3Output {
        self.last_input_len = input.len();
        let command = self.commands[self.index % self.commands.len()];
        self.index += 1;
        self.calls += 1;
        potok_edge::speculative_v3::V3Output {
            command,
            confidence: self.confidence,
        }
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

fn dispatch(
    student: FixturePredictor,
    teacher: FixturePredictor,
    cfg: DispatchConfig,
) -> V3Dispatch<FixturePredictor, FixturePredictor> {
    V3Dispatch::new(student, teacher, cfg).unwrap()
}

fn frame(n: usize) -> [f32; 2] {
    [n as f32 * 0.01, (n as f32 * 0.3).sin() * 0.5]
}

#[test]
fn guard_rejects_each_reason_in_a_stable_order() {
    let cfg = GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [0.4, 0.4], 0.5, 3).unwrap();
    let accepted = guard([0.2, 0.1], 0.9, reset(), &cfg);
    assert!(accepted.accepted);
    assert_eq!(accepted.reason, GuardReason::Ok);
    assert_eq!(accepted.command, [0.2, 0.1]);
    assert_eq!(accepted.state.last_command, Some([0.2, 0.1]));

    let cases = [
        ([f32::NAN, 0.1], 0.9, GuardReason::NonFinite),
        ([0.2, f32::INFINITY], 0.9, GuardReason::NonFinite),
        ([1.5, 0.1], 0.9, GuardReason::OutOfBounds),
        ([0.2, 0.1], 0.4, GuardReason::LowConfidence),
        ([0.9, 0.1], 0.9, GuardReason::RateLimit),
    ];
    for (command, confidence, want) in cases {
        let d = guard(command, confidence, accepted.state, &cfg);
        assert_eq!(
            d.reason, want,
            "command {command:?}, confidence {confidence}"
        );
        assert!(!d.accepted);
        assert_eq!(d.command, [0.0, 0.0]);
        assert_eq!(d.state.last_command, Some([0.2, 0.1]));
        assert_eq!(d.state.misses, 1);
    }
}

#[test]
fn guard_latch_is_sticky_after_three_misses_and_only_reset_clears_it() {
    let cfg = GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [1.0, 1.0], 0.5, 3).unwrap();
    let mut state = reset();
    for _ in 0..2 {
        let d = guard([0.1, 0.1], 0.1, state, &cfg);
        assert!(!d.state.latched);
        state = d.state;
    }
    let third = guard([0.1, 0.1], 0.1, state, &cfg);
    assert!(third.state.latched);
    assert_eq!(third.reason, GuardReason::LowConfidence);
    let after = guard([0.5, 0.5], 0.9, third.state, &cfg);
    assert_eq!(after.reason, GuardReason::Latched);
    assert_eq!(after.command, [0.0, 0.0]);
    assert!(after.state.latched);
    assert!(guard([0.5, 0.5], 0.9, reset(), &cfg).accepted);
}

#[test]
fn a_rejection_does_not_move_the_rate_reference() {
    let cfg = GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [0.3, 0.3], 0.5, 3).unwrap();
    let first = guard([0.9, 0.0], 0.9, reset(), &cfg);
    let rejected = guard([1.5, 0.0], 0.9, first.state, &cfg);
    assert_eq!(rejected.reason, GuardReason::OutOfBounds);
    // 0.7 is within 0.3 of the LAST ACCEPTED command (0.9), not of the zeros
    // emitted on rejection. If the reference had moved, this would be rejected.
    let next = guard([0.7, 0.0], 0.9, rejected.state, &cfg);
    assert!(next.accepted, "rejection moved the rate reference");
    assert_eq!(next.state.misses, 0);
}

#[test]
fn student_runs_every_cycle_and_teacher_is_capped_at_one_per_cycle() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 0.0,
        forced_check_every: 3,
        ..DispatchConfig::default()
    };
    let student = FixturePredictor::new([0.1, 0.1], 0.9);
    let teacher = FixturePredictor::new([0.2, -0.2], 0.9);
    let mut rt = dispatch(student, teacher, cfg);

    for n in 0..20 {
        let reply = rt.on_cycle(n as u16, &frame(n), true);
        assert!(reply.accepted);
        assert!(matches!(
            reply.source,
            V3Source::Student | V3Source::TeacherForced
        ));
    }
    assert_eq!(rt.counters.cycles, 20);
    assert_eq!(rt.counters.student_calls, 20);
    assert_eq!(rt.counters.teacher_calls, 6); // frames 3, 6, ..., 18
    assert!(rt.counters.teacher_calls <= rt.counters.cycles);
    assert_eq!(rt.counters.forced_checks, 6);
    assert_eq!(rt.counters.both_triggers, 0);
    assert_eq!(rt.teacher().calls, 6);
    assert_eq!(rt.teacher().last_input_len, WINDOW * 2);
}

#[test]
fn simultaneous_low_confidence_and_forced_check_still_call_teacher_once() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 1.1, // every student sample is "low"
        forced_check_every: 1,     // every cycle is forced
        ..DispatchConfig::default()
    };
    let mut rt = dispatch(
        FixturePredictor::new([0.1, 0.1], 0.9),
        FixturePredictor::new([0.3, -0.3], 0.9),
        cfg,
    );
    for n in 0..10 {
        let reply = rt.on_cycle(n as u16, &frame(n), true);
        assert_eq!(reply.source, V3Source::TeacherBoth);
        assert_eq!(reply.command, [0.3, -0.3]);
    }
    assert_eq!(rt.counters.teacher_calls, 10);
    assert_eq!(rt.counters.both_triggers, 10);
    assert_eq!(rt.counters.low_confidence, 10);
    assert_eq!(rt.counters.forced_checks, 10);
    assert_eq!(rt.counters.student_calls, 10);
}

#[test]
fn low_confidence_and_invalid_sensors_apply_the_teacher_command() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 0.7,
        forced_check_every: u32::MAX,
        ..DispatchConfig::default()
    };
    let mut low = dispatch(
        FixturePredictor::new([0.1, 0.1], 0.0),
        FixturePredictor::new([0.4, -0.5], 0.9),
        cfg,
    );
    assert_eq!(
        low.on_cycle(1, &frame(0), true).source,
        V3Source::TeacherLowConfidence
    );

    let mut sensors = dispatch(
        FixturePredictor::new([0.1, 0.1], 0.9),
        FixturePredictor::new([0.6, 0.7], 0.9),
        cfg,
    );
    let reply = sensors.on_cycle(2, &frame(1), false);
    assert_eq!(reply.source, V3Source::TeacherLowConfidence);
    assert_eq!(reply.command, [0.6, 0.7]);
    assert_eq!(sensors.counters.low_confidence, 1);
    assert_eq!(sensors.counters.teacher_calls, 1);
}

#[test]
fn the_guard_screens_the_teacher_path() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 1.1,
        forced_check_every: u32::MAX,
        ..DispatchConfig::default()
    };
    // Tighter-than-default bounds make the first teacher command illegal.
    let mut rt = dispatch(
        FixturePredictor::new([0.1, 0.1], 0.9),
        FixturePredictor::sequence(vec![[5.0, 0.0], [f32::NAN, 0.0], [0.2, 0.2]], 0.9),
        DispatchConfig {
            guards: GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [1.0, 1.0], 0.5, 3).unwrap(),
            ..cfg
        },
    );

    let out_of_bounds = rt.on_cycle(1, &frame(0), true);
    assert_eq!(out_of_bounds.reason, GuardReason::OutOfBounds);
    assert_eq!(out_of_bounds.command, [0.0, 0.0]);
    assert!(!out_of_bounds.accepted);
    assert_eq!(out_of_bounds.source, V3Source::TeacherLowConfidence);

    let non_finite = rt.on_cycle(2, &frame(1), true);
    assert_eq!(non_finite.reason, GuardReason::NonFinite);
    assert_eq!(non_finite.command, [0.0, 0.0]);

    rt.teacher_mut().confidence = 0.1;
    let low_conf = rt.on_cycle(3, &frame(2), true);
    assert_eq!(low_conf.reason, GuardReason::LowConfidence);
    assert_eq!(rt.counters.reject_out_of_bounds, 1);
    assert_eq!(rt.counters.reject_non_finite, 1);
    assert_eq!(rt.counters.reject_low_confidence, 1);
    assert_eq!(rt.counters.guard_rejected, 3);
    assert_eq!(rt.counters.guard_accepted, 0);
}

#[test]
fn divergence_resets_the_student_and_keeps_the_command_bounded() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 0.0,
        forced_check_every: 1,
        divergence_factor: -1.0, // any non-negative difference trips it
        ..DispatchConfig::default()
    };
    let student = FixturePredictor::new([0.1, 0.1], 0.9);
    let teacher = FixturePredictor::new([0.3, -0.3], 0.9);
    let mut rt = dispatch(student, teacher, cfg);
    let reply = rt.on_cycle(1, &frame(0), true);
    assert_eq!(rt.counters.divergences, 1);
    assert_eq!(rt.counters.student_resets, 1);
    assert_eq!(rt.student().resets, 1);
    assert_eq!(reply.command, [0.3, -0.3]); // forced check applies the teacher command
    assert_eq!(reply.source, V3Source::TeacherForced);
    assert!(reply.command.iter().all(|v| v.abs() <= 20.0));
}

#[test]
fn latch_blocks_output_until_an_explicit_reset() {
    let cfg = DispatchConfig {
        frame_len: 2,
        confidence_threshold: 0.0,
        forced_check_every: 1,
        guards: GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [1.0, 1.0], 0.5, 2).unwrap(),
        ..DispatchConfig::default()
    };
    let mut rt = dispatch(
        FixturePredictor::new([0.1, 0.1], 0.0),
        FixturePredictor::new([0.2, 0.2], 0.0),
        cfg,
    );
    assert_eq!(
        rt.on_cycle(1, &frame(0), true).reason,
        GuardReason::LowConfidence
    );
    let second = rt.on_cycle(2, &frame(1), true);
    assert!(!second.accepted);
    assert!(rt.guard_state().latched);
    let third = rt.on_cycle(3, &frame(2), true);
    assert_eq!(third.reason, GuardReason::Latched);
    assert_eq!(third.command, [0.0, 0.0]);
    assert_eq!(rt.counters.reject_latched, 1);

    rt.student_mut().confidence = 0.9;
    rt.teacher_mut().confidence = 0.9;
    rt.reset_guard();
    let fourth = rt.on_cycle(4, &frame(3), true);
    assert!(fourth.accepted, "reset must clear the latch");
    assert_eq!(fourth.command, [0.2, 0.2]);
    assert_eq!(rt.counters.guard_accepted, 1);
}

#[test]
fn malformed_or_non_finite_frames_bypass_both_models() {
    let cfg = DispatchConfig {
        frame_len: 2,
        ..DispatchConfig::default()
    };
    let student = FixturePredictor::new([0.1, 0.1], 0.9);
    let teacher = FixturePredictor::new([0.2, 0.2], 0.9);
    let mut rt = dispatch(student, teacher, cfg);
    let short = rt.on_cycle(1, &[1.0], true);
    let nan = rt.on_cycle(2, &[1.0, f32::NAN], true);
    for reply in [short, nan] {
        assert_eq!(reply.command, [0.0, 0.0]);
        assert_eq!(reply.source, V3Source::Zero);
        assert!(!reply.accepted);
    }
    assert_eq!(rt.counters.invalid_cycles, 2);
    assert_eq!(rt.counters.cycles, 0);
    assert_eq!(rt.student().calls, 0);
    assert_eq!(rt.teacher().calls, 0);
    assert_eq!(rt.counters.guard_accepted, 0);
}

#[test]
fn bad_configuration_fails_loudly() {
    assert!(GuardConfig::new([(1.0, -1.0), (-1.0, 1.0)], [1.0, 1.0], 0.5, 3).is_err());
    assert!(GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [-1.0, 1.0], 0.5, 3).is_err());
    assert!(GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [1.0, 1.0], 1.5, 3).is_err());
    assert!(GuardConfig::new([(-1.0, 1.0), (-1.0, 1.0)], [1.0, 1.0], 0.5, 0).is_err());
    let student = FixturePredictor::new([0.1, 0.1], 0.9);
    let teacher = FixturePredictor::new([0.2, 0.2], 0.9);
    assert!(V3Dispatch::new(
        student,
        teacher,
        DispatchConfig {
            frame_len: 0,
            ..DispatchConfig::default()
        }
    )
    .is_err());
    let student = FixturePredictor::new([0.1, 0.1], 0.9);
    let teacher = FixturePredictor::new([0.2, 0.2], 0.9);
    assert!(V3Dispatch::new(
        student,
        teacher,
        DispatchConfig {
            forced_check_every: 0,
            ..DispatchConfig::default()
        }
    )
    .is_err());
}

/// A deterministic two-model fixture replayed over a long run: two identical
/// dispatchers must return bit-identical commands under the one-call cap.
#[test]
fn a_long_fixture_replay_is_bitwise_reproducible() {
    fn commands(seed: u64, n: usize) -> Vec<[f32; 2]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut next = || {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                };
                [next() * 0.5, next() * 0.5]
            })
            .collect()
    }
    let cfg = DispatchConfig {
        frame_len: 2,
        forced_check_every: 7,
        ..DispatchConfig::default()
    };
    let mut a = dispatch(
        FixturePredictor::sequence(commands(11, 200), 0.8),
        FixturePredictor::sequence(commands(23, 200), 0.95),
        cfg,
    );
    let mut b = dispatch(
        FixturePredictor::sequence(commands(11, 200), 0.8),
        FixturePredictor::sequence(commands(23, 200), 0.95),
        cfg,
    );
    for n in 0..200 {
        let ra = a.on_cycle(n as u16, &frame(n), true);
        let rb = b.on_cycle(n as u16, &frame(n), true);
        assert_eq!(ra.command[0].to_bits(), rb.command[0].to_bits());
        assert_eq!(ra.command[1].to_bits(), rb.command[1].to_bits());
        assert_eq!(ra.source, rb.source);
        assert_eq!(ra.reason, rb.reason);
    }
    assert_eq!(a.counters, b.counters);
    assert_eq!(a.counters.cycles, 200);
    assert!(a.counters.teacher_calls <= 200);
}
