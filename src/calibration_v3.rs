//! v3 confidence calibration: read an exported JSON and transform a logit.
//!
//! Consumes either a `calibration.json` from `potok_ai.calibration` (methods
//! `temperature` and `isotonic`) or a `conformal.json` from `potok_ai.conformal`
//! (method `split-conformal`). The reader validates the whole payload and never
//! falls back to raw confidence: a malformed or unsupported export is an error.
//!
//! The transform mirrors the Python reference on the edge:
//!   * temperature: `sigmoid(z / T)`;
//!   * isotonic: `step(sigmoid(z))`, first upper edge >= p, last value held;
//!   * split-conformal: over `p = sigmoid(z)`, where the export's
//!     `p = 1 / (1 + |error| / tol)` coordinate is the confidence coordinate
//!     (`potok_ai.conformal.monotone_bin_map`). Canonical boundary is the
//!     INCLUSIVE LOWER EDGE: confident iff `p >= p_hat = 1/(1 + q_hat)`, so at
//!     `p == p_hat` the bin map and the exact `p >= 1/(1 + q_hat)` route agree.
//!     The conformal step reader uses "first upper edge > p" (last value held),
//!     matching `potok_ai.conformal.apply_confidence`.
//!
//! The adaptive-compute threshold is data-derived: `threshold` from the JSON
//! for temperature/isotonic, `1 / (1 + q_hat)` for conformal. It is never a
//! constant in this module.

use std::error::Error;
use std::fmt;
use std::path::Path;

const SIGMOID_CLIP: f64 = 60.0;
const EDGE_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Temperature,
    Isotonic,
    SplitConformal,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    Io(String),
    Json(String),
    Missing(&'static str),
    Invalid(String),
    UnknownMethod(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "calibration file: {s}"),
            Self::Json(s) => write!(f, "calibration JSON: {s}"),
            Self::Missing(k) => write!(f, "calibration JSON: missing {k}"),
            Self::Invalid(s) => write!(f, "calibration JSON: {s}"),
            Self::UnknownMethod(s) => write!(f, "unsupported calibration method {s:?}"),
        }
    }
}

impl Error for CalibrationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    method: Method,
    threshold: f64,
    temperature: Option<f64>,
    bins: Vec<f64>,
    values: Vec<f64>,
    q_hat: Option<f64>,
}

impl Calibration {
    pub fn from_json(text: &str) -> Result<Self, CalibrationError> {
        let root = Parser::new(text).parse()?;
        let Json::Object(fields) = root else {
            return Err(CalibrationError::Invalid("root must be an object".into()));
        };
        let method = json_string(&fields, "method")?;
        match method {
            "temperature" => {
                let temperature = finite_number(&fields, "temperature")?;
                if temperature <= 0.0 {
                    return Err(CalibrationError::Invalid(
                        "temperature must be finite and positive".into(),
                    ));
                }
                if fields.iter().any(|(k, _)| k == "bins" || k == "values") {
                    return Err(CalibrationError::Invalid(
                        "temperature payload must not carry bins/values".into(),
                    ));
                }
                let threshold = unit_number(&fields, "threshold")?;
                Ok(Self {
                    method: Method::Temperature,
                    threshold,
                    temperature: Some(temperature),
                    bins: Vec::new(),
                    values: Vec::new(),
                    q_hat: None,
                })
            }
            "isotonic" => {
                let threshold = unit_number(&fields, "threshold")?;
                let bins = json_numbers(&fields, "bins")?;
                let values = json_numbers(&fields, "values")?;
                let (bins, values) = validated_map(bins, values)?;
                Ok(Self {
                    method: Method::Isotonic,
                    threshold,
                    temperature: None,
                    bins,
                    values,
                    q_hat: None,
                })
            }
            "split-conformal" => {
                let q_hat = finite_number(&fields, "q_hat")?;
                if q_hat < 0.0 {
                    return Err(CalibrationError::Invalid(
                        "q_hat must be finite and non-negative".into(),
                    ));
                }
                let tol = finite_number(&fields, "tol")?;
                if tol <= 0.0 {
                    return Err(CalibrationError::Invalid(
                        "tol must be finite and positive".into(),
                    ));
                }
                let alpha = finite_number(&fields, "alpha")?;
                if !(0.0..1.0).contains(&alpha) {
                    return Err(CalibrationError::Invalid(
                        "alpha must lie strictly in (0, 1)".into(),
                    ));
                }
                let n = finite_number(&fields, "n")?;
                if n < 1.0 || n.fract() != 0.0 {
                    return Err(CalibrationError::Invalid(
                        "n must be a positive integer".into(),
                    ));
                }
                unit_number(&fields, "coverage")?;
                let has_bins = fields.iter().any(|(k, _)| k == "bins");
                let has_values = fields.iter().any(|(k, _)| k == "values");
                if has_bins != has_values {
                    return Err(CalibrationError::Invalid(
                        "bins and values must be supplied together".into(),
                    ));
                }
                let (bins, values) = if has_bins {
                    validated_map(
                        json_numbers(&fields, "bins")?,
                        json_numbers(&fields, "values")?,
                    )?
                } else {
                    (Vec::new(), Vec::new())
                };
                let p_hat = 1.0 / (1.0 + q_hat);
                let threshold = p_hat.clamp(EDGE_EPS, 1.0 - EDGE_EPS);
                Ok(Self {
                    method: Method::SplitConformal,
                    threshold,
                    temperature: None,
                    bins,
                    values,
                    q_hat: Some(q_hat),
                })
            }
            other => Err(CalibrationError::UnknownMethod(other.to_string())),
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CalibrationError> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| CalibrationError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_json(&text)
    }

    pub fn method(&self) -> Method {
        self.method
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Map a pre-sigmoid confidence logit to a calibrated confidence in [0, 1].
    pub fn apply_logit(&self, logit: f64) -> Result<f64, CalibrationError> {
        if !logit.is_finite() {
            return Err(CalibrationError::Invalid("logit must be finite".into()));
        }
        let p = match self.method {
            Method::Temperature => {
                let t = self
                    .temperature
                    .ok_or_else(|| CalibrationError::Invalid("temperature missing".into()))?;
                sigmoid(logit / t)
            }
            Method::Isotonic | Method::SplitConformal => sigmoid(logit),
        };
        match self.method {
            Method::Temperature => Ok(p),
            Method::Isotonic => Ok(self.step(p)),
            Method::SplitConformal => {
                if !self.bins.is_empty() {
                    Ok(self.conformal_step(p))
                } else {
                    let q = self
                        .q_hat
                        .ok_or_else(|| CalibrationError::Invalid("q_hat missing".into()))?;
                    Ok(if p >= 1.0 / (1.0 + q) { 1.0 } else { 0.0 })
                }
            }
        }
    }

    /// First upper edge >= p, holding the last value past the final edge.
    /// Used by isotonic, whose Python reference uses this convention.
    fn step(&self, p: f64) -> f64 {
        let index = self.bins.partition_point(|edge| *edge < p);
        self.values[index.min(self.values.len() - 1)]
    }

    /// Conformal step reader: first upper edge strictly greater than `p`,
    /// holding the last value past the final edge. This makes each bin's
    /// LOWER edge inclusive, so `p >= p_hat` is confident at equality.
    fn conformal_step(&self, p: f64) -> f64 {
        let index = self.bins.partition_point(|edge| *edge <= p);
        self.values[index.min(self.values.len() - 1)]
    }
}

fn sigmoid(z: f64) -> f64 {
    let z = z.clamp(-SIGMOID_CLIP, SIGMOID_CLIP);
    1.0 / (1.0 + (-z).exp())
}

fn validated_map(
    mut bins: Vec<f64>,
    values: Vec<f64>,
) -> Result<(Vec<f64>, Vec<f64>), CalibrationError> {
    // An n+1 edge array carries a leading lower edge; drop it as the Python
    // reader does. Anything else is a corrupt map, not a guess.
    if bins.len() == values.len() + 1 {
        bins.remove(0);
    }
    if values.is_empty() || bins.len() != values.len() {
        return Err(CalibrationError::Invalid(
            "bin map needs n upper edges and n values".into(),
        ));
    }
    if bins.iter().any(|b| !b.is_finite() || *b <= 0.0 || *b > 1.0) {
        return Err(CalibrationError::Invalid(
            "bin edges must be finite and in (0, 1]".into(),
        ));
    }
    if bins.windows(2).any(|w| w[1] <= w[0]) {
        return Err(CalibrationError::Invalid(
            "bin edges must be strictly increasing".into(),
        ));
    }
    if values
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(CalibrationError::Invalid(
            "bin values must be finite and in [0, 1]".into(),
        ));
    }
    if values.windows(2).any(|w| w[1] < w[0]) {
        return Err(CalibrationError::Invalid(
            "bin values must be monotone".into(),
        ));
    }
    Ok((bins, values))
}

fn json_string<'a>(
    fields: &'a [(String, Json)],
    key: &'static str,
) -> Result<&'a str, CalibrationError> {
    match fields.iter().find(|(k, _)| k == key) {
        None => Err(CalibrationError::Missing(key)),
        Some((_, Json::String(s))) => Ok(s),
        Some(_) => Err(CalibrationError::Invalid(format!("{key} must be a string"))),
    }
}

fn finite_number(fields: &[(String, Json)], key: &'static str) -> Result<f64, CalibrationError> {
    match fields.iter().find(|(k, _)| k == key) {
        None => Err(CalibrationError::Missing(key)),
        Some((_, Json::Number(v))) if v.is_finite() => Ok(*v),
        Some(_) => Err(CalibrationError::Invalid(format!(
            "{key} must be a finite number"
        ))),
    }
}

fn unit_number(fields: &[(String, Json)], key: &'static str) -> Result<f64, CalibrationError> {
    let value = finite_number(fields, key)?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(CalibrationError::Invalid(format!(
            "{key} must lie in [0, 1]"
        )))
    }
}

fn json_numbers(
    fields: &[(String, Json)],
    key: &'static str,
) -> Result<Vec<f64>, CalibrationError> {
    match fields.iter().find(|(k, _)| k == key) {
        None => Err(CalibrationError::Missing(key)),
        Some((_, Json::Array(items))) => items
            .iter()
            .map(|v| match v {
                Json::Number(n) if n.is_finite() => Ok(*n),
                _ => Err(CalibrationError::Invalid(format!(
                    "{key} must contain only finite numbers"
                ))),
            })
            .collect(),
        Some(_) => Err(CalibrationError::Invalid(format!("{key} must be an array"))),
    }
}

// Minimal JSON reader: only the shapes an export can contain. No dependency is
// added to the crate, and every malformed shape is a loud error.
#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

struct Parser<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn parse(mut self) -> Result<Json, CalibrationError> {
        let value = self.value()?;
        self.ws();
        if self.pos != self.text.len() {
            return Err(self.err("trailing characters"));
        }
        Ok(value)
    }

    fn err(&self, message: &str) -> CalibrationError {
        CalibrationError::Json(format!("{message} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.bump();
        }
    }

    fn value(&mut self) -> Result<Json, CalibrationError> {
        self.ws();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::String(self.string()?)),
            Some('t') => {
                self.literal("true")?;
                Ok(Json::Bool)
            }
            Some('f') => {
                self.literal("false")?;
                Ok(Json::Bool)
            }
            Some('n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some('-' | '0'..='9') => self.number(),
            _ => Err(self.err("unexpected token")),
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), CalibrationError> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn number(&mut self) -> Result<Json, CalibrationError> {
        let start = self.pos;
        while matches!(self.peek(), Some('-' | '+' | '.' | 'e' | 'E' | '0'..='9')) {
            self.bump();
        }
        self.text[start..self.pos]
            .parse::<f64>()
            .map(Json::Number)
            .map_err(|_| self.err("invalid number"))
    }

    fn string(&mut self) -> Result<String, CalibrationError> {
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string")),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let hi = self.hex4()?;
                        let codepoint = if (0xD800..=0xDBFF).contains(&hi) {
                            if !self.text[self.pos..].starts_with("\\u") {
                                return Err(self.err("bad surrogate pair"));
                            }
                            self.pos += 2;
                            let lo = self.hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&lo) {
                                return Err(self.err("bad surrogate pair"));
                            }
                            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                        } else {
                            hi
                        };
                        out.push(
                            char::from_u32(codepoint).ok_or_else(|| self.err("bad codepoint"))?,
                        );
                    }
                    _ => return Err(self.err("bad escape")),
                },
                Some(c) if c.is_control() => return Err(self.err("control character in string")),
                Some(c) => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, CalibrationError> {
        let start = self.pos;
        for _ in 0..4 {
            match self.bump() {
                Some(c) if c.is_ascii_hexdigit() => {}
                _ => return Err(self.err("bad unicode escape")),
            }
        }
        u32::from_str_radix(&self.text[start..self.pos], 16)
            .map_err(|_| self.err("bad unicode escape"))
    }

    fn object(&mut self) -> Result<Json, CalibrationError> {
        self.bump(); // '{'
        let mut fields = Vec::new();
        self.ws();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(Json::Object(fields));
        }
        loop {
            self.ws();
            if self.peek() != Some('"') {
                return Err(self.err("object key must be a string"));
            }
            let key = self.string()?;
            self.ws();
            if self.bump() != Some(':') {
                return Err(self.err("expected ':'"));
            }
            let value = self.value()?;
            if fields.iter().any(|(k, _)| k == &key) {
                return Err(self.err("duplicate object key"));
            }
            fields.push((key, value));
            self.ws();
            match self.bump() {
                Some(',') => {}
                Some('}') => return Ok(Json::Object(fields)),
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Json, CalibrationError> {
        self.bump(); // '['
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.bump() {
                Some(',') => {}
                Some(']') => return Ok(Json::Array(items)),
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }
}
