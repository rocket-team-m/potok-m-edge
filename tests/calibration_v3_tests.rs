//! v3 calibration reader gates. Reference outputs were produced by
//! `potok_ai.calibration.apply_calibration` and `potok_ai.conformal.apply_confidence`
//! in `/home/sehaxe/potok-m/ai` (`.venv`, 2026-11-23), not invented here.
//!
//! This repo's `src/lib.rs` is shared with other tasks, so the module is
//! included by path instead of being re-exported from the library crate.

#[path = "../src/calibration_v3.rs"]
mod calibration_v3;

use calibration_v3::{Calibration, CalibrationError, Method};

const TEMP_JSON: &str = r#"{
  "coverage": 0.6, "ece": 0.0, "loops": 3, "method": "temperature",
  "n": 4, "raw_ece": 0.29, "target_ece": 0.05, "temperature": 2.0,
  "temperature_ece": 0.1, "threshold": 0.3
}"#;

const ISO_JSON: &str = r#"{
  "bins": [2.0611536181902037e-09, 0.01798620996209156,
           0.9820137900379085, 0.9999999979388463],
  "coverage": 0.6, "ece": 0.0, "loops": 3, "method": "isotonic",
  "n": 160, "raw_ece": 0.2910068939883774, "target_ece": 0.05,
  "temperature": 10.0, "temperature_ece": 0.09999999999999998,
  "threshold": 0.3, "values": [0.3, 0.5, 0.7, 0.9]
}"#;

const CONF_JSON: &str = r#"{
  "alpha": 0.25, "bins": [0.2857142857142857, 1.0], "coverage": 0.875,
  "half_width": 0.5, "method": "split-conformal", "n": 8, "q_hat": 2.5,
  "tol": 0.2, "values": [0.0, 1.0]
}"#;

const CONF_NOBIN_JSON: &str = r#"{
  "alpha": 0.25, "coverage": 0.875, "half_width": 0.5,
  "method": "split-conformal", "n": 8, "q_hat": 2.5, "tol": 0.2
}"#;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "got {actual}, expected {expected}"
    );
}

#[test]
fn temperature_matches_python_logit_transform() {
    let c = Calibration::from_json(TEMP_JSON).expect("temperature export must load");
    assert_eq!(c.method(), Method::Temperature);
    assert_close(c.threshold(), 0.3);
    for (logit, expected) in [
        (-4.0, 0.11920292202211755),
        (0.0, 0.5),
        (3.0, 0.8175744761936437),
    ] {
        assert_close(c.apply_logit(logit).unwrap(), expected);
    }
}

#[test]
fn isotonic_matches_python_step_map_and_accepts_leading_edge() {
    let c = Calibration::from_json(ISO_JSON).expect("isotonic export must load");
    assert_eq!(c.method(), Method::Isotonic);
    let logits = [-30.0, -20.0, -4.0, 0.0, 4.0, 20.0, 30.0];
    let expected = [0.3, 0.3, 0.5, 0.7, 0.7, 0.9, 0.9];
    for (z, want) in logits.iter().zip(expected) {
        assert_close(c.apply_logit(*z).unwrap(), want);
    }

    // The Python reader accepts n+1 edges with a leading lower edge; the
    // extended map must give identical calibrated confidences.
    let extended = ISO_JSON.replace(
        "\"bins\": [2.0611536181902037e-09",
        "\"bins\": [0.0, 2.0611536181902037e-09",
    );
    let c_ext = Calibration::from_json(&extended).unwrap();
    for (z, want) in logits.iter().zip(expected) {
        assert_close(c_ext.apply_logit(*z).unwrap(), want);
    }
}

#[test]
fn conformal_bin_map_matches_python_decision() {
    let c = Calibration::from_json(CONF_JSON).expect("conformal export must load");
    assert_eq!(c.method(), Method::SplitConformal);
    // p_hat = 1/(1 + q_hat) is the data-derived threshold, not a constant.
    assert_close(c.threshold(), 0.2857142857142857);
    // z = -ln(s) means sigmoid(z) = 1/(1+s), the export's confidence coordinate.
    for (z, expected) in [
        (2.3025850929940455, 1.0),
        (std::f64::consts::LN_2, 1.0),
        (0.0, 1.0),
        (-std::f64::consts::LN_2, 1.0),
        (-1.6094379124341003, 0.0),
        (-2.995732273553991, 0.0),
    ] {
        assert_close(c.apply_logit(z).unwrap(), expected);
    }
}

#[test]
fn conformal_without_bin_map_uses_exact_quantile() {
    let c = Calibration::from_json(CONF_NOBIN_JSON).unwrap();
    assert_eq!(c.method(), Method::SplitConformal);
    for (z, expected) in [
        (2.3025850929940455, 1.0),
        (std::f64::consts::LN_2, 1.0),
        (-std::f64::consts::LN_2, 1.0),
        (-1.6094379124341003, 0.0),
    ] {
        assert_close(c.apply_logit(z).unwrap(), expected);
    }
    // The exported step map uses "first upper edge >= p", so at p == p_hat it
    // lands on the lower value; the exact route uses s <= q_hat and is 1.
    // Both are pinned to the Python reference, including their boundary.
    let boundary = -0.9162907318741551; // -ln(2.5), q_hat = 2.5
    assert_close(
        Calibration::from_json(CONF_JSON)
            .unwrap()
            .apply_logit(boundary)
            .unwrap(),
        0.0,
    );
    assert_close(c.apply_logit(boundary).unwrap(), 1.0);
}

#[test]
fn threshold_is_read_from_json_not_hardcoded() {
    let changed = TEMP_JSON.replace("\"threshold\": 0.3", "\"threshold\": 0.73");
    let c = Calibration::from_json(&changed).unwrap();
    assert_close(c.threshold(), 0.73);
    // The threshold is an adaptive-compute input, not part of the logit map.
    assert_close(c.apply_logit(0.0).unwrap(), 0.5);
}

#[test]
fn rejects_non_finite_logit_loudly() {
    let c = Calibration::from_json(TEMP_JSON).unwrap();
    assert_eq!(
        c.apply_logit(f64::NAN),
        Err(CalibrationError::Invalid("logit must be finite".into()))
    );
    assert!(c.apply_logit(f64::INFINITY).is_err());
}

#[test]
fn rejects_malformed_payloads_loudly() {
    assert!(matches!(
        Calibration::from_json(r#"{"temperature": 1.0, "threshold": 0.5}"#),
        Err(CalibrationError::Missing("method"))
    ));
    assert!(matches!(
        Calibration::from_json(r#"{"method": "bogus"}"#),
        Err(CalibrationError::UnknownMethod(_))
    ));
    assert!(Calibration::from_json(
        &TEMP_JSON.replace("\"temperature\": 2.0", "\"temperature\": 0.0")
    )
    .is_err());
    assert!(
        Calibration::from_json(&ISO_JSON.replace(", \"values\": [0.3, 0.5, 0.7, 0.9]", ""))
            .is_err()
    );
    assert!(Calibration::from_json(
        &ISO_JSON.replace("0.9820137900379085, 0.9999999979388463", "0.98, 0.98")
    )
    .is_err());
    assert!(Calibration::from_json(&ISO_JSON.replace(
        "\"values\": [0.3, 0.5, 0.7, 0.9]",
        "\"values\": [0.3, 1.5, 0.7, 0.9]"
    ))
    .is_err());
    assert!(
        Calibration::from_json(&CONF_JSON.replace("\"q_hat\": 2.5", "\"q_hat\": -1.0")).is_err()
    );
    assert!(Calibration::from_json(&CONF_JSON.replace(", \"values\": [0.0, 1.0]", "")).is_err());
    assert!(matches!(
        Calibration::from_json("{not json"),
        Err(CalibrationError::Json(_))
    ));
    assert!(matches!(
        Calibration::from_json(
            r#"{"method":"isotonic","threshold":0.5,"bins":[0.2],"values":[0.1],"bins":[0.3]}"#
        ),
        Err(CalibrationError::Json(_))
    ));
}

#[test]
fn reads_a_file_and_reports_io_errors() {
    let path = std::env::temp_dir().join(format!(
        "potok-calibration-v3-{}-{}.json",
        std::process::id(),
        "file"
    ));
    std::fs::write(&path, TEMP_JSON).unwrap();
    let c = Calibration::from_file(&path).unwrap();
    assert_close(c.threshold(), 0.3);
    assert_close(c.apply_logit(3.0).unwrap(), 0.8175744761936437);
    std::fs::remove_file(&path).unwrap();

    let missing = std::env::temp_dir().join(format!(
        "potok-calibration-v3-{}-absent.json",
        std::process::id()
    ));
    assert!(matches!(
        Calibration::from_file(&missing),
        Err(CalibrationError::Io(_))
    ));
}
