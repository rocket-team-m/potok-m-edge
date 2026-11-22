//! Canonical reader for the Potok-M v3 model export.
//!
//! The v3 export is the smart-frame fast model: 14 physics features followed
//! by 6 conditioning numbers, trained against the v3 command target
//! (`g_accel_y` / `g_accel_x`, see `ai/src/potok_ai/data_v3.py`). This module
//! only reads and dequantizes; it never writes a model and no model binary is
//! shipped in this repository. Test bytes are generated in memory.
//!
//! Layout (all integers little-endian, f32 IEEE-754, codes signed int8,
//! UTF-8 names without a NUL terminator, no padding):
//!
//! ```text
//! [header][layer record] * n_records[sha256: 32 raw bytes]
//! ```
//!
//! The SHA-256 covers every byte from offset 0 up to, but not including, the
//! final 32 bytes. A mismatch is a hard error: no fallback, no partial load.
//!
//! Header (64 bytes plus `4 * n_blocks` dilation words):
//!
//! | offset | type | field | value |
//! |---:|---|---|---|
//! | 0 | 8 bytes | magic | `POTOKM3\0` |
//! | 8 | u32 | version | 3 |
//! | 12 | u32 | model_kind | 2 (stateful fast model) |
//! | 16 | u32 | in_features | 20 (14 physics + 6 conditioning) |
//! | 20 | u32 | width | channel width |
//! | 24 | u32 | out_features | 2 (roll, pitch corrections) |
//! | 28 | u32 | rank | TSCT rank |
//! | 32 | u32 | n_blocks | block count |
//! | 36 | u32 | kernel | depthwise kernel |
//! | 40 | u32 | n_loops | loop iterations per tick, 1..=3 |
//! | 44 | f32 | decay | loop accumulation decay, 0..=1 |
//! | 48 | f32 | g_max | correction bound |
//! | 52 | u32 | flags | bit 0 = ternary factors (required); others must be 0 |
//! | 56 | u32 | target_version | 3, the acceleration-command contract |
//! | 60 | u32 | n_records | `5 * n_blocks + 10` |
//! | 64 + 4i | u32 | dilations[i] | one positive word per block |
//!
//! Layer records use the canonical v2 encoding: `kind = 1` TSCT (in/out/rank,
//! alpha_u/alpha_v, `in*rank` ternary codes, `rank` f32 scales, `out*rank`
//! ternary codes) and `kind = 2` f32 tensor (ndim, i32 dimensions, row-major
//! data). Required names are `blocks.{b}.{rec_h,rec_x}.{weight,bias}`,
//! `blocks.{b}.proj`, `stem.{weight,bias}`, `norm.{weight,bias}`,
//! `alloc_logits`, `head_corr.{weight,bias}`, `head_conf.{weight,bias}` and
//! `g_max`. Ternary factors are dequantized once; consumers never see codes.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"POTOKM3\0";
pub const VERSION: u32 = 3;
pub const TARGET_VERSION: u32 = 3;
pub const MODEL_KIND_FAST: u32 = 2;
pub const IN_FEATURES: usize = 20;
pub const PHYSICS_FEATURES: usize = 14;
pub const COND_FEATURES: usize = 6;
pub const OUT_FEATURES: usize = 2;
pub const MAX_LOOPS: usize = 3;
pub const TSCT_LAYER: u8 = 1;
pub const FP32_LAYER: u8 = 2;

const CHECKSUM_BYTES: usize = 32;
const FIXED_HEADER_BYTES: usize = 64;
const LN_EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3Error {
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    BadModelKind(u32),
    BadTargetVersion(u32),
    ChecksumMismatch,
    InvalidField(&'static str),
    UnknownKind(u8),
    MissingLayer(String),
    DuplicateLayer(String),
    ShapeMismatch(&'static str),
    InvalidTernaryCode(i8),
    Utf8,
}

impl fmt::Display for V3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "v3 model file truncated"),
            Self::BadMagic => write!(f, "not a POTOKM3 model file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported export version {v}"),
            Self::BadModelKind(k) => write!(f, "model_kind {k} is not the v3 fast model (2)"),
            Self::BadTargetVersion(v) => write!(f, "unsupported v3 target version {v}"),
            Self::ChecksumMismatch => write!(f, "v3 SHA256 checksum mismatch"),
            Self::InvalidField(s) => write!(f, "invalid v3 field: {s}"),
            Self::UnknownKind(k) => write!(f, "unknown v3 layer kind {k}"),
            Self::MissingLayer(s) => write!(f, "missing v3 layer {s}"),
            Self::DuplicateLayer(s) => write!(f, "duplicate v3 layer {s}"),
            Self::ShapeMismatch(s) => write!(f, "v3 shape mismatch: {s}"),
            Self::InvalidTernaryCode(c) => write!(f, "invalid ternary code {c}"),
            Self::Utf8 => write!(f, "v3 layer name is not UTF-8"),
        }
    }
}

impl std::error::Error for V3Error {}

struct Reader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], V3Error> {
        let end = self.off.checked_add(n).ok_or(V3Error::Truncated)?;
        let s = self.b.get(self.off..end).ok_or(V3Error::Truncated)?;
        self.off = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, V3Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, V3Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, V3Error> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, V3Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, V3Error> {
        let bytes = self.take(n.checked_mul(4).ok_or(V3Error::Truncated)?)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    fn codes(&mut self, n: usize) -> Result<Vec<i8>, V3Error> {
        let bytes = self.take(n)?;
        let v: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        if let Some(&bad) = v.iter().find(|c| !(-1..=1).contains(*c)) {
            return Err(V3Error::InvalidTernaryCode(bad));
        }
        Ok(v)
    }
}

enum LayerData {
    Tsct {
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha_u: f32,
        alpha_v: f32,
        u: Vec<i8>,
        s: Vec<f32>,
        v: Vec<i8>,
    },
    F32 {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
}

/// One dequantized recurrent block of the v3 fast model.
#[derive(Debug)]
pub struct V3Block {
    pub rec_h_w: Vec<f32>,
    pub rec_h_b: Vec<f32>,
    pub rec_x_w: Vec<f32>,
    pub rec_x_b: Vec<f32>,
    /// `[width, width]`, `W[out, in] = U[out,k]*alpha_u*s[k]*V[in,k]*alpha_v`.
    pub proj_w: Vec<f32>,
}

/// A fully dequantized v3 export. Weights are dense f32 so consumers do no
/// dequantization in the hot loop.
#[derive(Debug)]
pub struct V3Model {
    pub in_features: usize,
    pub width: usize,
    pub out_features: usize,
    pub rank: usize,
    pub n_blocks: usize,
    pub kernel: usize,
    pub n_loops: usize,
    pub decay: f32,
    pub g_max: f32,
    pub flags: u32,
    pub target_version: u32,
    pub dilations: Vec<usize>,
    /// `tanh(|g_max|)`, the head's correction bound.
    pub correction_bound: f32,
    pub stem_w: Vec<f32>,
    pub stem_b: Vec<f32>,
    pub blocks: Vec<V3Block>,
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub alloc: Vec<f32>,
    pub head_corr_w: Vec<f32>,
    pub head_corr_b: Vec<f32>,
    pub head_conf_w: Vec<f32>,
    pub head_conf_b: Vec<f32>,
}

impl V3Model {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, V3Error> {
        let bytes = fs::read(path).map_err(|_| V3Error::Truncated)?;
        Self::from_bytes(&bytes)
    }

    /// Verify SHA-256, parse the canonical header and records, dequantize every
    /// ternary factor once. Every refusal is loud; there is no partial result.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, V3Error> {
        if bytes.len() < FIXED_HEADER_BYTES + CHECKSUM_BYTES {
            return Err(V3Error::Truncated);
        }
        let split = bytes.len() - CHECKSUM_BYTES;
        let body = &bytes[..split];
        if Sha256::digest(body).as_slice() != &bytes[split..] {
            return Err(V3Error::ChecksumMismatch);
        }

        let mut r = Reader::new(body);
        if r.take(8)? != MAGIC {
            return Err(V3Error::BadMagic);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(V3Error::UnsupportedVersion(version));
        }
        let kind = r.u32()?;
        if kind != MODEL_KIND_FAST {
            return Err(V3Error::BadModelKind(kind));
        }
        let in_features = r.u32()? as usize;
        let width = r.u32()? as usize;
        let out_features = r.u32()? as usize;
        let rank = r.u32()? as usize;
        let n_blocks = r.u32()? as usize;
        let kernel = r.u32()? as usize;
        let n_loops = r.u32()? as usize;
        let decay = r.f32()?;
        let g_max = r.f32()?;
        let flags = r.u32()?;
        let target_version = r.u32()?;
        if target_version != TARGET_VERSION {
            return Err(V3Error::BadTargetVersion(target_version));
        }
        let n_records = r.u32()? as usize;

        if in_features != IN_FEATURES
            || width == 0
            || out_features != OUT_FEATURES
            || rank == 0
            || n_blocks == 0
            || kernel == 0
            || !(1..=MAX_LOOPS).contains(&n_loops)
            || !decay.is_finite()
            || !(0.0..=1.0).contains(&decay)
            || !g_max.is_finite()
            || g_max <= 0.0
            || flags != 1
            || n_records != 5 * n_blocks + 10
        {
            return Err(V3Error::InvalidField("header counts"));
        }
        let dilations: Vec<usize> = (0..n_blocks)
            .map(|_| r.u32().map(|d| d as usize))
            .collect::<Result<_, _>>()?;
        if dilations.contains(&0) {
            return Err(V3Error::InvalidField("dilation"));
        }

        let mut layers: HashMap<String, LayerData> = HashMap::with_capacity(n_records);
        for _ in 0..n_records {
            let kind = r.u8()?;
            let name_len = r.u32()? as usize;
            let name = std::str::from_utf8(r.take(name_len)?)
                .map_err(|_| V3Error::Utf8)?
                .to_owned();
            let layer = match kind {
                TSCT_LAYER => {
                    let d_in = r.u32()? as usize;
                    let d_out = r.u32()? as usize;
                    let d_rank = r.u32()? as usize;
                    let alpha_u = r.f32()?;
                    let alpha_v = r.f32()?;
                    let u = r.codes(d_in.checked_mul(d_rank).ok_or(V3Error::Truncated)?)?;
                    let s = r.f32s(d_rank)?;
                    let v = r.codes(d_out.checked_mul(d_rank).ok_or(V3Error::Truncated)?)?;
                    LayerData::Tsct {
                        in_dim: d_in,
                        out_dim: d_out,
                        rank: d_rank,
                        alpha_u,
                        alpha_v,
                        u,
                        s,
                        v,
                    }
                }
                FP32_LAYER => {
                    let ndim = r.u32()? as usize;
                    let mut shape = Vec::with_capacity(ndim);
                    let mut n = 1usize;
                    for _ in 0..ndim {
                        let d = r.i32()?;
                        if d <= 0 {
                            return Err(V3Error::ShapeMismatch("non-positive dimension"));
                        }
                        n = n.checked_mul(d as usize).ok_or(V3Error::Truncated)?;
                        shape.push(d as usize);
                    }
                    LayerData::F32 {
                        shape,
                        data: r.f32s(n)?,
                    }
                }
                other => return Err(V3Error::UnknownKind(other)),
            };
            if layers.insert(name.clone(), layer).is_some() {
                return Err(V3Error::DuplicateLayer(name));
            }
        }
        if r.off != body.len() {
            return Err(V3Error::InvalidField("trailing bytes"));
        }

        let tensor = |name: &str, want: &[usize]| -> Result<Vec<f32>, V3Error> {
            match layers.get(name) {
                Some(LayerData::F32 { shape, data }) if shape.as_slice() == want => {
                    Ok(data.clone())
                }
                Some(LayerData::F32 { .. }) => Err(V3Error::ShapeMismatch("tensor shape")),
                Some(_) => Err(V3Error::InvalidField("tensor layer kind")),
                None => Err(V3Error::MissingLayer(name.to_owned())),
            }
        };

        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let prefix = format!("blocks.{b}");
            let rec_h_w = tensor(&format!("{prefix}.rec_h.weight"), &[width, width])?;
            let rec_h_b = tensor(&format!("{prefix}.rec_h.bias"), &[width])?;
            let rec_x_w = tensor(&format!("{prefix}.rec_x.weight"), &[width, width])?;
            let rec_x_b = tensor(&format!("{prefix}.rec_x.bias"), &[width])?;
            let name = format!("{prefix}.proj");
            let (d_in, d_out, d_rank, alpha_u, alpha_v, uc, sc, vc) = match layers.get(&name) {
                Some(LayerData::Tsct {
                    in_dim,
                    out_dim,
                    rank: r,
                    alpha_u,
                    alpha_v,
                    u,
                    s,
                    v,
                }) => (*in_dim, *out_dim, *r, *alpha_u, *alpha_v, u, s, v),
                Some(_) => return Err(V3Error::InvalidField("proj kind")),
                None => return Err(V3Error::MissingLayer(name)),
            };
            if d_in != width || d_out != width || d_rank != rank || sc.len() != rank {
                return Err(V3Error::ShapeMismatch("TSCT factors"));
            }
            let mut proj_w = vec![0.0f32; width * width];
            for k in 0..rank {
                for out in 0..width {
                    let uu = uc[out * rank + k] as f32 * alpha_u;
                    for input in 0..width {
                        let vv = vc[input * rank + k] as f32 * alpha_v;
                        proj_w[out * width + input] += (uu * sc[k]) * vv;
                    }
                }
            }
            blocks.push(V3Block {
                rec_h_w,
                rec_h_b,
                rec_x_w,
                rec_x_b,
                proj_w,
            });
        }

        let stem_w = tensor("stem.weight", &[width, in_features])?;
        let stem_b = tensor("stem.bias", &[width])?;
        let norm_w = tensor("norm.weight", &[width])?;
        let norm_b = tensor("norm.bias", &[width])?;
        let alloc = tensor("alloc_logits", &[n_loops])?;
        let head_corr_w = tensor("head_corr.weight", &[out_features, width])?;
        let head_corr_b = tensor("head_corr.bias", &[out_features])?;
        let head_conf_w = tensor("head_conf.weight", &[1, width])?;
        let head_conf_b = tensor("head_conf.bias", &[1])?;
        let file_g_max = tensor("g_max", &[1])?[0];
        if !file_g_max.is_finite() || file_g_max <= 0.0 {
            return Err(V3Error::InvalidField("g_max tensor"));
        }

        Ok(Self {
            in_features,
            width,
            out_features,
            rank,
            n_blocks,
            kernel,
            n_loops,
            decay,
            g_max,
            flags,
            target_version,
            dilations,
            correction_bound: g_max.abs().tanh(),
            stem_w,
            stem_b,
            blocks,
            norm_w,
            norm_b,
            alloc,
            head_corr_w,
            head_corr_b,
            head_conf_w,
            head_conf_b,
        })
    }

    /// Bytes held by the dequantized weights.
    pub fn memory_bytes(&self) -> usize {
        let fixed = self.stem_w.len()
            + self.stem_b.len()
            + self.norm_w.len()
            + self.norm_b.len()
            + self.alloc.len()
            + self.head_corr_w.len()
            + self.head_corr_b.len()
            + self.head_conf_w.len()
            + self.head_conf_b.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.rec_h_w.len()
                    + b.rec_h_b.len()
                    + b.rec_x_w.len()
                    + b.rec_x_b.len()
                    + b.proj_w.len()
            })
            .sum();
        (fixed + blocks) * std::mem::size_of::<f32>()
    }
}

/// LayerNorm with the biased variance and epsilon used by the canonical
/// export, exposed for consumers that replay the v3 head.
pub fn layer_norm(src: &[f32], gain: &[f32], bias: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mean = src.iter().sum::<f32>() / n as f32;
    let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv * gain[i] + bias[i];
    }
}
