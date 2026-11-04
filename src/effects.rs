//! Local, offline video effects applied to packed RGB24 frames.
//!
//! * `blur` / `replace`: real person matting with Robust Video Matting (MobileNetV3, ONNX,
//!   GPL-3.0) run locally through ONNX Runtime. See `assets/models/README.md` for provenance.
//! * `chroma`: green-screen key (green dominance matte + spill suppression) over a background image.
//! * `soften`: edge-preserving image softening (box blur that keeps strong edges). Not a
//!   reimplementation of any vendor "makeup" feature.
//! * `mirror`, `overlay_image` (centred, never upscaled, alpha-aware), `board` (four-corner
//!   perspective correction to a full-frame rectangle).
//!
//! Frame order: board -> matte -> soften -> background -> mirror -> overlay.

use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
    time::SystemTime,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use image::{
    DynamicImage, ImageReader, Limits,
    imageops::{self, FilterType},
};
use ort::{
    environment::Environment,
    session::Session,
    value::{DynValue, Tensor, TensorRef},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_FILE: &str = "rvm_mobilenetv3_fp32.onnx";
pub const MODEL_SHA256: &str = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828";
const MODEL_URL: &str = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx";

const MIN_DIM: u32 = 16;
const MAX_DIM: u32 = 8192;
const MAX_IMAGE_BYTES: u64 = 64 << 20;
const MAX_IMAGE_DIM: u32 = 16384;
const MAX_IMAGE_ALLOC: u64 = 512 << 20;
/// Longest side fed to the matting network; the alpha is upsampled back to the frame size.
const SEG_LONG_SIDE: f64 = 512.0;
/// RVM `downsample_ratio`: its docs want the backbone input between 256 and 512 px.
const SEG_DOWNSAMPLE: f32 = 0.5;
/// Green dominance (G - max(R,B)) where the key starts / is complete.
const KEY_LO: i32 = 20;
const KEY_HI: i32 = 70;
/// Soften keeps detail whose local contrast is well above this (0..255 scale).
const SOFTEN_EDGE: i32 = 12;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Effects {
    /// `none`, `blur`, `replace` or `chroma`.
    pub background: String,
    /// Required for `replace` and `chroma`; cover-fitted to the frame.
    pub background_image: Option<PathBuf>,
    /// Background blur strength (Gaussian-like sigma in pixels at 720p), 0..=30; `blur` needs > 0.
    pub blur: f32,
    /// Edge-preserving image softening, 0..=1.
    pub soften: f32,
    pub mirror: bool,
    /// PNG/JPEG (alpha honoured), centred and shrunk to fit if larger than the frame.
    pub overlay_image: Option<PathBuf>,
    pub overlay_opacity: f32,
    /// Normalised corners TL, TR, BR, BL (clockwise on screen) of a region stretched to the frame.
    pub board: Option<[[f32; 2]; 4]>,
}

impl Default for Effects {
    fn default() -> Self {
        Self {
            background: "none".into(),
            background_image: None,
            blur: 12.0,
            soften: 0.0,
            mirror: false,
            overlay_image: None,
            overlay_opacity: 1.0,
            board: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    None,
    Blur,
    Replace,
    Chroma,
}

impl Effects {
    /// Pure range/consistency checks; no file access.
    pub fn validate(&self) -> Result<()> {
        self.mode().map(drop)
    }

    fn mode(&self) -> Result<Mode> {
        let mode = match self.background.as_str() {
            "" | "none" => Mode::None,
            "blur" => Mode::Blur,
            "replace" => Mode::Replace,
            "chroma" => Mode::Chroma,
            other => bail!("unknown background mode {other:?}; use none, blur, replace or chroma"),
        };
        ensure!(
            self.blur.is_finite() && (0.0..=30.0).contains(&self.blur),
            "blur must be within 0..30, got {}",
            self.blur
        );
        ensure!(
            self.soften.is_finite() && (0.0..=1.0).contains(&self.soften),
            "soften must be within 0..1, got {}",
            self.soften
        );
        ensure!(
            self.overlay_opacity.is_finite() && (0.0..=1.0).contains(&self.overlay_opacity),
            "overlay_opacity must be within 0..1, got {}",
            self.overlay_opacity
        );
        if mode == Mode::Blur {
            ensure!(
                self.blur > 0.0,
                "background \"blur\" needs blur > 0 (1..30)"
            );
        }
        if matches!(mode, Mode::Replace | Mode::Chroma) {
            ensure!(
                self.background_image.is_some(),
                "background {:?} needs background_image",
                self.background
            );
        }
        if let Some(b) = &self.board {
            check_board(b)?;
        }
        Ok(mode)
    }
}

// ---------------------------------------------------------------------------------------------
// Model discovery and ONNX Runtime
// ---------------------------------------------------------------------------------------------

fn oe(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{e}")
}

fn model_candidates() -> Vec<PathBuf> {
    if let Some(p) = std::env::var_os("LINK_CAMERA_RVM_MODEL").filter(|p| !p.is_empty()) {
        return vec![PathBuf::from(p)];
    }
    let mut v = Vec::new();
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/share")));
    if let Some(d) = data {
        v.push(d.join("link-camera/models").join(MODEL_FILE));
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        v.push(dir.join("assets/models").join(MODEL_FILE));
        v.push(dir.join("../share/link-camera/models").join(MODEL_FILE));
        v.push(dir.join("../../assets/models").join(MODEL_FILE)); // target/{debug,release}
    }
    v.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/models")
            .join(MODEL_FILE),
    );
    v
}

/// Finds the first candidate whose bytes match [`MODEL_SHA256`].
fn load_model() -> Result<(PathBuf, Vec<u8>)> {
    let mut notes = Vec::new();
    for p in model_candidates() {
        match p.metadata() {
            Ok(meta) if meta.is_file() && meta.len() == 14_975_696 => {}
            Ok(_) => {
                notes.push(format!(
                    "  {} (expected a regular 14,975,696-byte model file)",
                    p.display()
                ));
                continue;
            }
            Err(error) => {
                notes.push(format!("  {} ({error})", p.display()));
                continue;
            }
        }
        match std::fs::read(&p) {
            Ok(bytes) => {
                let sum = format!("{:x}", Sha256::digest(&bytes));
                if sum == MODEL_SHA256 {
                    return Ok((p, bytes));
                }
                notes.push(format!("  {} (sha256 {sum} does not match)", p.display()));
            }
            Err(e) => notes.push(format!("  {} ({e})", p.display())),
        }
    }
    bail!(
        "person segmentation model {MODEL_FILE} (sha256 {MODEL_SHA256}) not found; looked in:\n{}\n\
         Recover: mkdir -p ~/.local/share/link-camera/models && curl -L -o ~/.local/share/link-camera/models/{MODEL_FILE} {MODEL_URL}\n\
         (or point LINK_CAMERA_RVM_MODEL at a verified copy; see assets/models/README.md)",
        notes.join("\n")
    )
}

/// Path of the verified segmentation model, for diagnostics.
pub fn locate_model() -> Result<PathBuf> {
    load_model().map(|(p, _)| p)
}

fn ort_env() -> Result<&'static Environment> {
    static ENV: LazyLock<std::result::Result<Environment, String>> = LazyLock::new(|| {
        ort::init()
            .with_name("link-camera")
            .with_telemetry(false)
            .build()
            .map_err(|e| e.to_string())
    });
    ENV.as_ref()
        .map_err(|e| anyhow!("ONNX Runtime failed to initialise: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Resampling helpers
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Tap {
    i0: usize,
    i1: usize,
    /// Weight of `i1` in 1/256.
    w1: u32,
}

/// Bilinear taps mapping `dst` samples onto `src` samples (pixel-centre aligned).
fn taps(dst: usize, src: usize) -> Vec<Tap> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let s = ((d as f64 + 0.5) * scale - 0.5).clamp(0.0, (src - 1) as f64);
            let i0 = s.floor() as usize;
            Tap {
                i0,
                i1: (i0 + 1).min(src - 1),
                w1: ((s - i0 as f64) * 256.0).round() as u32,
            }
        })
        .collect()
}

fn lerp(a: u32, b: u32, w1: u32) -> u32 {
    (a * (256 - w1) + b * w1 + 128) >> 8
}

/// `fg * a/255 + bg * (1 - a/255)`.
fn mix(fg: u8, bg: u32, a: u32) -> u8 {
    ((fg as u32 * a + bg * (255 - a) + 127) / 255) as u8
}

fn blur_h(src: &[u8], dst: &mut [u8], w: usize, r: usize) {
    let div = (2 * r + 1) as u32;
    let (r, last) = (r as isize, w as isize - 1);
    for (srow, drow) in src.chunks_exact(w * 3).zip(dst.chunks_exact_mut(w * 3)) {
        for c in 0..3 {
            let at = |x: isize| srow[x.clamp(0, last) as usize * 3 + c] as u32;
            let mut sum: u32 = (-r..=r).map(at).sum();
            for x in 0..w as isize {
                drow[x as usize * 3 + c] = ((sum + div / 2) / div) as u8;
                sum = sum + at(x + r + 1) - at(x - r);
            }
        }
    }
}

fn blur_v(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize, sums: &mut Vec<u32>) {
    let rw = w * 3;
    let div = (2 * r + 1) as u32;
    let (ri, last) = (r as isize, h as isize - 1);
    sums.clear();
    sums.resize(rw, 0);
    for k in -ri..=ri {
        let row = &src[k.clamp(0, last) as usize * rw..][..rw];
        for (s, &v) in sums.iter_mut().zip(row) {
            *s += v as u32;
        }
    }
    for y in 0..h {
        for (d, s) in dst[y * rw..][..rw].iter_mut().zip(sums.iter()) {
            *d = ((s + div / 2) / div) as u8;
        }
        let add = &src[(y as isize + ri + 1).min(last) as usize * rw..][..rw];
        let sub = &src[(y as isize - ri).max(0) as usize * rw..][..rw];
        for ((s, &a), &b) in sums.iter_mut().zip(add).zip(sub) {
            *s = *s + a as u32 - b as u32;
        }
    }
}

/// In-place separable box blur (edge-replicating); `tmp` must be as large as `buf`.
fn box_blur(buf: &mut [u8], tmp: &mut [u8], sums: &mut Vec<u32>, w: usize, h: usize, r: usize) {
    blur_h(buf, tmp, w, r);
    blur_v(tmp, buf, w, h, r, sums);
}

// ---------------------------------------------------------------------------------------------
// Perspective
// ---------------------------------------------------------------------------------------------

fn check_board(b: &[[f32; 2]; 4]) -> Result<()> {
    for (i, p) in b.iter().enumerate() {
        ensure!(
            p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "board corner {} ({}, {}) must be finite and within 0..1",
            i + 1,
            p[0],
            p[1]
        );
    }
    let mut area2 = 0.0f64;
    for i in 0..4 {
        let (p, q, s) = (b[i], b[(i + 1) % 4], b[(i + 2) % 4]);
        let (e1, e2) = ([q[0] - p[0], q[1] - p[1]], [s[0] - q[0], s[1] - q[1]]);
        let cross = e1[0] as f64 * e2[1] as f64 - e1[1] as f64 * e2[0] as f64;
        ensure!(
            cross > 1e-6,
            "board corners must be a convex quadrilateral ordered top-left, top-right, bottom-right, bottom-left (corner {} is collinear, reversed or crossed)",
            (i + 1) % 4 + 1
        );
        area2 += p[0] as f64 * q[1] as f64 - q[0] as f64 * p[1] as f64;
    }
    ensure!(
        area2 / 2.0 >= 0.005,
        "board covers less than 0.5% of the frame"
    );
    Ok(())
}

/// Unit-square -> quad projective map (Heckbert), plus per-pixel evaluation in frame pixels.
struct Warp {
    /// a b c d e f g h of x=(a u+b v+c)/(g u+h v+1), y=(d u+e v+f)/(g u+h v+1).
    m: [f64; 8],
    w: usize,
    h: usize,
}

impl Warp {
    fn new(b: &[[f32; 2]; 4], w: usize, h: usize) -> Result<Self> {
        check_board(b)?;
        let p: Vec<(f64, f64)> = b.iter().map(|p| (p[0] as f64, p[1] as f64)).collect();
        let (sx, sy) = (
            p[0].0 - p[1].0 + p[2].0 - p[3].0,
            p[0].1 - p[1].1 + p[2].1 - p[3].1,
        );
        let m = if sx.abs() < 1e-12 && sy.abs() < 1e-12 {
            [
                p[1].0 - p[0].0,
                p[2].0 - p[1].0,
                p[0].0,
                p[1].1 - p[0].1,
                p[2].1 - p[1].1,
                p[0].1,
                0.0,
                0.0,
            ]
        } else {
            let (dx1, dx2, dy1, dy2) = (
                p[1].0 - p[2].0,
                p[3].0 - p[2].0,
                p[1].1 - p[2].1,
                p[3].1 - p[2].1,
            );
            let den = dx1 * dy2 - dx2 * dy1;
            ensure!(
                den.abs() > 1e-12,
                "board corners do not define a perspective transform"
            );
            let g = (sx * dy2 - dx2 * sy) / den;
            let hh = (dx1 * sy - sx * dy1) / den;
            [
                p[1].0 - p[0].0 + g * p[1].0,
                p[3].0 - p[0].0 + hh * p[3].0,
                p[0].0,
                p[1].1 - p[0].1 + g * p[1].1,
                p[3].1 - p[0].1 + hh * p[3].1,
                p[0].1,
                g,
                hh,
            ]
        };
        ensure!(
            m.iter().all(|v| v.is_finite()),
            "board corners produce a degenerate transform"
        );
        Ok(Self { m, w, h })
    }

    /// Normalised source position for a normalised destination position.
    fn map(&self, u: f64, v: f64) -> (f64, f64) {
        let m = &self.m;
        let d = m[6] * u + m[7] * v + 1.0;
        (
            (m[0] * u + m[1] * v + m[2]) / d,
            (m[3] * u + m[4] * v + m[5]) / d,
        )
    }

    /// Bilinear inverse warp of `src` into `dst` (both `w*h` RGB).
    fn apply(&self, src: &[u8], dst: &mut [u8]) {
        let (w, h) = (self.w, self.h);
        let (wf, hf) = (w as f32, h as f32);
        for y in 0..h {
            let v = (y as f64 + 0.5) / h as f64;
            for x in 0..w {
                let (sx, sy) = self.map((x as f64 + 0.5) / w as f64, v);
                let fx = (sx as f32 * wf - 0.5).clamp(0.0, wf - 1.0);
                let fy = (sy as f32 * hf - 0.5).clamp(0.0, hf - 1.0);
                let (x0, y0) = (fx as usize, fy as usize);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                let out = &mut dst[(y * w + x) * 3..][..3];
                for c in 0..3 {
                    let p = |xx: usize, yy: usize| src[(yy * w + xx) * 3 + c] as f32;
                    let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
                    let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
                    out[c] = (top + (bot - top) * ty + 0.5) as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Person matting (RVM)
// ---------------------------------------------------------------------------------------------

struct Segmenter {
    session: Session,
    iw: usize,
    ih: usize,
    input: Vec<f32>,
    rec: [DynValue; 4],
    ratio: Tensor<f32>,
    /// frame -> model plane taps
    down_x: Vec<Tap>,
    down_y: Vec<Tap>,
    /// model plane -> frame taps
    up_x: Vec<Tap>,
    up_y: Vec<Tap>,
}

fn zero_state() -> Result<DynValue> {
    Ok(Tensor::<f32>::from_array(([1i64, 1, 1, 1], vec![0f32]))
        .map_err(oe)?
        .into_dyn())
}

impl Segmenter {
    fn new(w: usize, h: usize) -> Result<Self> {
        let (path, bytes) = load_model()?;
        let threads = std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .min(4);
        let session = Session::builder(ort_env()?)
            .map_err(oe)?
            .with_intra_threads(threads)
            .map_err(oe)?
            .commit_from_memory(&bytes)
            .map_err(oe)
            .with_context(|| format!("loading segmentation model {}", path.display()))?;
        for n in ["src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"] {
            ensure!(
                session.inputs().iter().any(|i| i.name() == n),
                "model {} lacks input {n}",
                path.display()
            );
        }
        for n in ["pha", "r1o", "r2o", "r3o", "r4o"] {
            ensure!(
                session.outputs().iter().any(|o| o.name() == n),
                "model {} lacks output {n}",
                path.display()
            );
        }
        let s = (SEG_LONG_SIDE / w.max(h) as f64).min(1.0);
        let iw = ((w as f64 * s).round() as usize).max(MIN_DIM as usize);
        let ih = ((h as f64 * s).round() as usize).max(MIN_DIM as usize);
        Ok(Self {
            session,
            iw,
            ih,
            input: vec![0.0; 3 * iw * ih],
            rec: [zero_state()?, zero_state()?, zero_state()?, zero_state()?],
            ratio: Tensor::from_array(([1i64], vec![SEG_DOWNSAMPLE])).map_err(oe)?,
            down_x: taps(iw, w),
            down_y: taps(ih, h),
            up_x: taps(w, iw),
            up_y: taps(h, ih),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.rec = [zero_state()?, zero_state()?, zero_state()?, zero_state()?];
        Ok(())
    }

    /// Writes the person alpha (255 = person) for `rgb` into `alpha` (`w*h`).
    fn matte(&mut self, rgb: &[u8], w: usize, h: usize, alpha: &mut [u8]) -> Result<()> {
        let (iw, ih) = (self.iw, self.ih);
        let plane = iw * ih;
        for y in 0..ih {
            let ty = self.down_y[y];
            let (r0, r1) = (
                &rgb[ty.i0 * w * 3..][..w * 3],
                &rgb[ty.i1 * w * 3..][..w * 3],
            );
            let wy = ty.w1 as f32 / 256.0;
            for x in 0..iw {
                let tx = self.down_x[x];
                let wx = tx.w1 as f32 / 256.0;
                for c in 0..3 {
                    let p = |r: &[u8], i: usize| r[i * 3 + c] as f32;
                    let top = p(r0, tx.i0) + (p(r0, tx.i1) - p(r0, tx.i0)) * wx;
                    let bot = p(r1, tx.i0) + (p(r1, tx.i1) - p(r1, tx.i0)) * wx;
                    self.input[c * plane + y * iw + x] = (top + (bot - top) * wy) / 255.0;
                }
            }
        }

        let Self {
            session,
            input,
            rec,
            ratio,
            up_x,
            up_y,
            ..
        } = self;
        let src = TensorRef::from_array_view(([1i64, 3, ih as i64, iw as i64], &input[..]))
            .map_err(oe)?;
        let mut outputs = session
            .run(ort::inputs! {
                "src" => src,
                "r1i" => &rec[0],
                "r2i" => &rec[1],
                "r3i" => &rec[2],
                "r4i" => &rec[3],
                "downsample_ratio" => &*ratio,
            })
            .map_err(oe)
            .context("running person segmentation")?;
        {
            let (shape, pha) = outputs["pha"].try_extract_tensor::<f32>().map_err(oe)?;
            ensure!(
                shape.len() == 4
                    && shape[2] == ih as i64
                    && shape[3] == iw as i64
                    && pha.len() == plane,
                "segmentation model returned alpha of shape {shape}, expected [1,1,{ih},{iw}]"
            );
            for y in 0..h {
                let ty = up_y[y];
                let (r0, r1) = (&pha[ty.i0 * iw..][..iw], &pha[ty.i1 * iw..][..iw]);
                let wy = ty.w1 as f32 / 256.0;
                for (a, tx) in alpha[y * w..][..w].iter_mut().zip(up_x.iter()) {
                    let wx = tx.w1 as f32 / 256.0;
                    let top = r0[tx.i0] + (r0[tx.i1] - r0[tx.i0]) * wx;
                    let bot = r1[tx.i0] + (r1[tx.i1] - r1[tx.i0]) * wx;
                    *a = ((top + (bot - top) * wy) * 255.0 + 0.5) as u8; // saturating cast: NaN -> 0
                }
            }
        }
        for (slot, name) in rec.iter_mut().zip(["r1o", "r2o", "r3o", "r4o"]) {
            *slot = outputs
                .remove(name)
                .ok_or_else(|| anyhow!("segmentation model produced no {name}"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Pixel operations
// ---------------------------------------------------------------------------------------------

/// Green-screen matte: alpha 255 = keep (foreground), 0 = replace. Also suppresses green spill.
// shortcut: fixed green thresholds (KEY_LO/KEY_HI), no blue-screen or tunable key colour; add fields when needed.
fn chroma_key(rgb: &mut [u8], alpha: &mut [u8]) {
    for (px, a) in rgb.chunks_exact_mut(3).zip(alpha.iter_mut()) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let rb = r.max(b);
        let d = g - rb;
        *a = if d <= KEY_LO {
            255
        } else if d >= KEY_HI {
            0
        } else {
            (255 * (KEY_HI - d) / (KEY_HI - KEY_LO)) as u8
        };
        if d > 0 {
            px[1] = rb as u8;
        }
    }
}

fn composite_rgb(rgb: &mut [u8], bg: &[u8], alpha: &[u8]) {
    for ((px, b), &a) in rgb.chunks_exact_mut(3).zip(bg.chunks_exact(3)).zip(alpha) {
        match a {
            255 => {}
            0 => px.copy_from_slice(b),
            _ => {
                for c in 0..3 {
                    px[c] = mix(px[c], b[c] as u32, a as u32);
                }
            }
        }
    }
}

/// Edge-preserving softening: pull low-contrast detail towards a box-blurred copy.
fn soften(
    rgb: &mut [u8],
    blurred: &mut [u8],
    tmp: &mut [u8],
    sums: &mut Vec<u32>,
    w: usize,
    h: usize,
    strength: f32,
) {
    let r = ((w.min(h) as f32 / 240.0).round() as usize).max(1);
    blurred.copy_from_slice(rgb);
    box_blur(blurred, tmp, sums, w, h, r);
    box_blur(blurred, tmp, sums, w, h, r);
    let s = (strength * 256.0).round() as i32;
    let t2 = SOFTEN_EDGE * SOFTEN_EDGE;
    for (px, b) in rgb.chunks_exact_mut(3).zip(blurred.chunks_exact(3)) {
        let d = [
            px[0] as i32 - b[0] as i32,
            px[1] as i32 - b[1] as i32,
            px[2] as i32 - b[2] as i32,
        ];
        let e = d.iter().map(|v| v.abs()).max().unwrap_or(0);
        let keep = e * e * 256 / (e * e + t2).max(1);
        let m = 256 - s * (256 - keep) / 256;
        for c in 0..3 {
            px[c] = (b[c] as i32 + ((d[c] * m) >> 8)).clamp(0, 255) as u8;
        }
    }
}

fn mirror(rgb: &mut [u8], w: usize) {
    for row in rgb.chunks_exact_mut(w * 3) {
        for x in 0..w / 2 {
            for c in 0..3 {
                row.swap(x * 3 + c, (w - 1 - x) * 3 + c);
            }
        }
    }
}

/// Scale-to-cover and centre-crop to exactly `w*h` RGB.
fn cover(img: DynamicImage, w: usize, h: usize) -> Vec<u8> {
    let rgb = img.into_rgb8();
    let s = (w as f64 / rgb.width() as f64).max(h as f64 / rgb.height() as f64);
    // Crop before scaling so extreme aspect ratios cannot inflate the resized allocation.
    let cw = ((w as f64 / s).ceil() as u32).clamp(1, rgb.width());
    let ch = ((h as f64 / s).ceil() as u32).clamp(1, rgb.height());
    let cropped = imageops::crop_imm(
        &rgb,
        (rgb.width() - cw) / 2,
        (rgb.height() - ch) / 2,
        cw,
        ch,
    );
    imageops::resize(&*cropped, w as u32, h as u32, FilterType::Triangle).into_raw()
}

struct Overlay {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

fn fit_overlay(img: DynamicImage, w: usize, h: usize) -> Overlay {
    let mut rgba = img.into_rgba8();
    let (iw, ih) = (rgba.width() as usize, rgba.height() as usize);
    let s = (w as f64 / iw as f64).min(h as f64 / ih as f64);
    let (ow, oh) = if s < 1.0 {
        let (ow, oh) = (
            ((iw as f64 * s).round() as usize).clamp(1, w),
            ((ih as f64 * s).round() as usize).clamp(1, h),
        );
        rgba = imageops::resize(&rgba, ow as u32, oh as u32, FilterType::Triangle);
        (ow, oh)
    } else {
        (iw, ih)
    };
    Overlay {
        x: (w - ow) / 2,
        y: (h - oh) / 2,
        w: ow,
        h: oh,
        rgba: rgba.into_raw(),
    }
}

fn blend_overlay(rgb: &mut [u8], w: usize, ov: &Overlay, opacity256: u32) {
    for row in 0..ov.h {
        let dst = &mut rgb[((ov.y + row) * w + ov.x) * 3..][..ov.w * 3];
        let src = &ov.rgba[row * ov.w * 4..][..ov.w * 4];
        for (d, s) in dst.chunks_exact_mut(3).zip(src.chunks_exact(4)) {
            let a = (s[3] as u32 * opacity256 + 128) >> 8;
            if a == 0 {
                continue;
            }
            for c in 0..3 {
                d[c] = mix(s[c], d[c] as u32, a.min(255));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cached assets
// ---------------------------------------------------------------------------------------------

#[derive(PartialEq)]
struct Stamp {
    path: PathBuf,
    modified: Option<SystemTime>,
    len: u64,
}

struct Cached<T> {
    stamp: Stamp,
    val: T,
}

fn stamp(path: &Path) -> Result<Stamp> {
    let meta = std::fs::metadata(path).with_context(|| format!("image {}", path.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(
        meta.len() <= MAX_IMAGE_BYTES,
        "{} is larger than {} MiB",
        path.display(),
        MAX_IMAGE_BYTES >> 20
    );
    Ok(Stamp {
        path: path.to_path_buf(),
        modified: meta.modified().ok(),
        len: meta.len(),
    })
}

fn decode(path: &Path) -> Result<DynamicImage> {
    let mut reader = ImageReader::open(path)
        .with_context(|| format!("opening image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("reading image {}", path.display()))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC);
    reader.limits(limits);
    reader
        .decode()
        .with_context(|| format!("decoding image {} (PNG or JPEG expected)", path.display()))
}

/// `Some(new)` when the file changed or was never loaded; `None` when `slot` is still valid.
fn refresh<T>(
    slot: &Option<Cached<T>>,
    path: &Path,
    load: impl FnOnce(DynamicImage) -> T,
) -> Result<Option<Cached<T>>> {
    let stamp = stamp(path)?;
    if slot.as_ref().is_some_and(|c| c.stamp == stamp) {
        return Ok(None);
    }
    Ok(Some(Cached {
        val: load(decode(path)?),
        stamp,
    }))
}

/// Blurred, downscaled copy of the frame used as the background for `blur`.
struct BlurState {
    f: usize,
    sw: usize,
    sh: usize,
    radius: usize,
    small: Vec<u8>,
    tmp: Vec<u8>,
    x: Vec<Tap>,
    y: Vec<Tap>,
}

impl BlurState {
    fn new(w: usize, h: usize, blur: f32) -> Self {
        let r = blur * w.min(h) as f32 / 720.0;
        let f = if r >= 6.0 {
            4
        } else if r >= 3.0 {
            2
        } else {
            1
        };
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        Self {
            f,
            sw,
            sh,
            radius: ((r / f as f32).round() as usize).max(1),
            small: vec![0; sw * sh * 3],
            tmp: vec![0; sw * sh * 3],
            x: taps(w, sw),
            y: taps(h, sh),
        }
    }

    /// Box-average downscale then three box passes (~Gaussian).
    fn build(&mut self, rgb: &[u8], w: usize, h: usize, sums: &mut Vec<u32>) {
        let (f, sw, sh) = (self.f, self.sw, self.sh);
        for sy in 0..sh {
            let (y0, y1) = (sy * f, (sy * f + f).min(h));
            for sx in 0..sw {
                let (x0, x1) = (sx * f, (sx * f + f).min(w));
                let mut acc = [0u32; 3];
                for y in y0..y1 {
                    for px in rgb[(y * w + x0) * 3..(y * w + x1) * 3].chunks_exact(3) {
                        for c in 0..3 {
                            acc[c] += px[c] as u32;
                        }
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as u32;
                for c in 0..3 {
                    self.small[(sy * sw + sx) * 3 + c] = ((acc[c] + n / 2) / n) as u8;
                }
            }
        }
        for _ in 0..3 {
            box_blur(&mut self.small, &mut self.tmp, sums, sw, sh, self.radius);
        }
    }

    fn composite(&self, rgb: &mut [u8], alpha: &[u8], w: usize) {
        let rw = self.sw * 3;
        for (y, (row, arow)) in rgb
            .chunks_exact_mut(w * 3)
            .zip(alpha.chunks_exact(w))
            .enumerate()
        {
            let ty = self.y[y];
            let (r0, r1) = (
                &self.small[ty.i0 * rw..][..rw],
                &self.small[ty.i1 * rw..][..rw],
            );
            for (x, (px, &a)) in row.chunks_exact_mut(3).zip(arow).enumerate() {
                if a == 255 {
                    continue;
                }
                let tx = self.x[x];
                for c in 0..3 {
                    let top = lerp(r0[tx.i0 * 3 + c] as u32, r0[tx.i1 * 3 + c] as u32, tx.w1);
                    let bot = lerp(r1[tx.i0 * 3 + c] as u32, r1[tx.i1 * 3 + c] as u32, tx.w1);
                    px[c] = mix(px[c], lerp(top, bot, ty.w1), a as u32);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------------------------

/// Applies [`Effects`] to frames of one fixed size. Assets (model session, background, overlay)
/// are loaded in [`Processor::configure`] and cached across frames and reconfigurations.
pub struct Processor {
    w: usize,
    h: usize,
    fx: Effects,
    mode: Mode,
    opacity256: u32,
    warp: Option<Warp>,
    seg: Option<Segmenter>,
    bg: Option<Cached<Vec<u8>>>,
    overlay: Option<Cached<Overlay>>,
    blur: Option<BlurState>,
    alpha: Vec<u8>,
    scratch: Vec<u8>,
    tmp: Vec<u8>,
    sums: Vec<u32>,
}

impl Processor {
    /// Cheap: no model or image is loaded until [`Processor::configure`] needs it.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        ensure!(
            (MIN_DIM..=MAX_DIM).contains(&width) && (MIN_DIM..=MAX_DIM).contains(&height),
            "frame size {width}x{height} outside supported {MIN_DIM}..{MAX_DIM}"
        );
        Ok(Self {
            w: width as usize,
            h: height as usize,
            fx: Effects::default(),
            mode: Mode::None,
            opacity256: 256,
            warp: None,
            seg: None,
            bg: None,
            overlay: None,
            blur: None,
            alpha: Vec::new(),
            scratch: Vec::new(),
            tmp: Vec::new(),
            sums: Vec::new(),
        })
    }

    /// Validates and loads everything first; on error the previous configuration stays active.
    /// May take about a second the first time `blur`/`replace` loads the model.
    pub fn configure(&mut self, fx: Effects) -> Result<()> {
        let mode = fx.mode()?;
        let (w, h) = (self.w, self.h);
        let warp = fx.board.as_ref().map(|b| Warp::new(b, w, h)).transpose()?;
        let wants_seg = matches!(mode, Mode::Blur | Mode::Replace);
        let new_bg = match (&fx.background_image, mode) {
            (Some(p), Mode::Replace | Mode::Chroma) => refresh(&self.bg, p, |i| cover(i, w, h))?,
            _ => None,
        };
        let new_ov = match &fx.overlay_image {
            Some(p) => refresh(&self.overlay, p, |i| fit_overlay(i, w, h))?,
            None => None,
        };
        if wants_seg {
            crate::preferences::require_ai_enabled("Person blur and background replacement")?;
        }
        let new_seg = if wants_seg && self.seg.is_none() {
            Some(Segmenter::new(w, h)?)
        } else {
            None
        };

        let was_seg = matches!(self.mode, Mode::Blur | Mode::Replace);
        if let Some(s) = new_seg {
            self.seg = Some(s);
        } else if wants_seg && !was_seg {
            if let Some(s) = &mut self.seg {
                s.reset()?; // stale recurrent state from an earlier session
            }
        }
        if matches!(mode, Mode::Replace | Mode::Chroma) {
            self.bg = new_bg.or(self.bg.take());
        } else {
            self.bg = None;
        }
        self.overlay = if fx.overlay_image.is_some() {
            new_ov.or(self.overlay.take())
        } else {
            None
        };
        self.blur = (mode == Mode::Blur).then(|| BlurState::new(w, h, fx.blur));
        if mode != Mode::None && self.alpha.len() != w * h {
            self.alpha = vec![0; w * h];
        }
        if (warp.is_some() || fx.soften > 0.0) && self.scratch.len() != w * h * 3 {
            self.scratch = vec![0; w * h * 3];
            self.tmp = vec![0; w * h * 3];
        }
        self.opacity256 = (fx.overlay_opacity * 256.0).round() as u32;
        self.warp = warp;
        self.mode = mode;
        self.fx = fx;
        Ok(())
    }

    /// Applies the configured effects in place to a `width*height` RGB24 frame.
    pub fn process(&mut self, rgb: &mut [u8]) -> Result<()> {
        let (w, h) = (self.w, self.h);
        ensure!(
            rgb.len() == w * h * 3,
            "frame has {} bytes, expected {} for {w}x{h} RGB24",
            rgb.len(),
            w * h * 3
        );
        if let Some(warp) = &self.warp {
            self.scratch.copy_from_slice(rgb);
            warp.apply(&self.scratch, rgb);
        }
        match self.mode {
            Mode::Blur | Mode::Replace => {
                let seg = self.seg.as_mut().ok_or_else(|| {
                    anyhow!("segmentation model not loaded; call configure first")
                })?;
                seg.matte(rgb, w, h, &mut self.alpha)?;
            }
            Mode::Chroma => chroma_key(rgb, &mut self.alpha),
            Mode::None => {}
        }
        if self.fx.soften > 0.0 {
            soften(
                rgb,
                &mut self.scratch,
                &mut self.tmp,
                &mut self.sums,
                w,
                h,
                self.fx.soften,
            );
        }
        match self.mode {
            Mode::Blur => {
                if let Some(b) = &mut self.blur {
                    b.build(rgb, w, h, &mut self.sums);
                    b.composite(rgb, &self.alpha, w);
                }
            }
            Mode::Replace | Mode::Chroma => {
                let bg = self
                    .bg
                    .as_ref()
                    .ok_or_else(|| anyhow!("background image not loaded; call configure first"))?;
                composite_rgb(rgb, &bg.val, &self.alpha);
            }
            Mode::None => {}
        }
        if self.fx.mirror {
            mirror(rgb, w);
        }
        if let (Some(ov), true) = (&self.overlay, self.opacity256 > 0) {
            blend_overlay(rgb, w, &ov.val, self.opacity256);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("link-camera-effects-{}-{name}", std::process::id()))
    }

    fn solid_png(name: &str, w: u32, h: u32, rgba: [u8; 4]) -> PathBuf {
        let path = temp(name);
        image::RgbaImage::from_pixel(w, h, image::Rgba(rgba))
            .save(&path)
            .unwrap();
        path
    }

    fn px(buf: &[u8], w: usize, x: usize, y: usize) -> [u8; 3] {
        let i = (y * w + x) * 3;
        [buf[i], buf[i + 1], buf[i + 2]]
    }

    #[test]
    fn rejects_invalid_effects() {
        let base = Effects::default;
        let bad = [
            Effects {
                background: "glow".into(),
                ..base()
            },
            Effects {
                blur: 31.0,
                ..base()
            },
            Effects {
                blur: f32::NAN,
                ..base()
            },
            Effects {
                background: "blur".into(),
                blur: 0.0,
                ..base()
            },
            Effects {
                soften: -0.1,
                ..base()
            },
            Effects {
                overlay_opacity: 1.5,
                ..base()
            },
            Effects {
                background: "replace".into(),
                ..base()
            },
            Effects {
                background: "chroma".into(),
                ..base()
            },
        ];
        for fx in bad {
            assert!(fx.validate().is_err(), "{fx:?}");
        }
        assert!(serde_json::from_str::<Effects>(r#"{"mirorr":true}"#).is_err());
        assert!(Effects::default().validate().is_ok());
        assert!(Processor::new(8, 8).is_err());
        assert!(Processor::new(640, 100_000).is_err());
    }

    #[test]
    fn board_rejects_degenerate_polygons() {
        let ok = [[0.1, 0.1], [0.9, 0.15], [0.85, 0.9], [0.05, 0.8]];
        assert!(check_board(&ok).is_ok());
        let mut cases = vec![];
        cases.push([ok[0], ok[1], ok[3], ok[2]]); // bow-tie
        cases.push([ok[3], ok[2], ok[1], ok[0]]); // counter-clockwise
        cases.push([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]); // collinear corner
        cases.push([[0.2, 0.2], [0.21, 0.2], [0.21, 0.21], [0.2, 0.21]]); // tiny
        cases.push([[0.0, 0.0], [1.2, 0.0], [1.0, 1.0], [0.0, 1.0]]); // out of range
        cases.push([[0.0, 0.0], [f32::NAN, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        cases.push([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.5, 0.0]]); // concave-ish/zero area
        for c in cases {
            assert!(check_board(&c).is_err(), "{c:?}");
        }
    }

    #[test]
    fn warp_maps_corners_and_unit_square_is_identity() {
        let quad = [[0.2, 0.1], [0.8, 0.1], [0.95, 0.9], [0.05, 0.9]];
        let wp = Warp::new(&quad, 64, 32).unwrap();
        for ((u, v), q) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .into_iter()
            .zip(quad)
        {
            let (x, y) = wp.map(u, v);
            assert!((x - q[0] as f64).abs() < 1e-9 && (y - q[1] as f64).abs() < 1e-9);
        }
        let (cx, cy) = wp.map(0.5, 0.5);
        assert!((0.05..0.95).contains(&cx) && (0.1..0.9).contains(&cy));

        let (w, h) = (64usize, 32usize);
        let src: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % w * 4) as u8, (i / w * 8) as u8, 7])
            .collect();
        let mut out = vec![0; src.len()];
        Warp::new(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], w, h)
            .unwrap()
            .apply(&src, &mut out);
        assert!(src.iter().zip(&out).all(|(a, b)| a.abs_diff(*b) <= 1));

        // left half stretched to full width: last column samples near the old middle
        Warp::new(&[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], w, h)
            .unwrap()
            .apply(&src, &mut out);
        assert!(
            px(&out, w, w - 1, 0)[0].abs_diff(125) <= 3,
            "{:?}",
            px(&out, w, w - 1, 0)
        );
    }

    #[test]
    fn blur_keeps_flat_regions_and_mirror_is_involution() {
        let (w, h) = (20usize, 10usize);
        let mut flat = vec![77u8; w * h * 3];
        let (mut tmp, mut sums) = (vec![0; w * h * 3], Vec::new());
        box_blur(&mut flat, &mut tmp, &mut sums, w, h, 4);
        assert!(flat.iter().all(|&v| v == 77));

        let mut img: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let orig = img.clone();
        mirror(&mut img, w);
        assert_eq!(px(&img, w, 0, 3), px(&orig, w, w - 1, 3));
        mirror(&mut img, w);
        assert_eq!(img, orig);
    }

    #[test]
    fn soften_is_noop_at_zero_and_smooths_noise_but_keeps_edges() {
        let (w, h) = (48usize, 48usize);
        // left half dark with +-3 noise, right half bright: strong edge in the middle
        let img: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let base = if i % w < w / 2 { 40 } else { 200 };
                let v = base + if i % 2 == 0 { 3 } else { 0 };
                [v, v, v]
            })
            .collect();
        let (mut s, mut t, mut sums) = (vec![0; img.len()], vec![0; img.len()], Vec::new());
        let mut zero = img.clone();
        soften(&mut zero, &mut s, &mut t, &mut sums, w, h, 0.0);
        assert_eq!(zero, img);
        let mut out = img.clone();
        soften(&mut out, &mut s, &mut t, &mut sums, w, h, 1.0);
        let noise = |b: &[u8]| (b[(24 * w + 5) * 3] as i32 - b[(24 * w + 6) * 3] as i32).abs();
        assert!(noise(&out) < noise(&img));
        assert!(px(&out, w, 2, 24)[0] < 60 && px(&out, w, w - 3, 24)[0] > 180);
        assert!(
            (px(&out, w, w / 2 - 1, 24)[0] as i32 - 40).abs() < 20,
            "edge preserved"
        );
    }

    #[test]
    fn cover_centres_extreme_aspect_ratios_without_an_oversized_intermediate() {
        for (width, height) in [(MAX_IMAGE_DIM, 1), (1, MAX_IMAGE_DIM)] {
            let middle = width.max(height) / 2;
            let image = image::RgbImage::from_fn(width, height, |x, y| {
                if x.max(y).abs_diff(middle) <= 4 {
                    image::Rgb([10, 20, 200])
                } else {
                    image::Rgb([200, 0, 0])
                }
            });
            let output = cover(DynamicImage::ImageRgb8(image), 1920, 1080);
            assert_eq!(output.len(), 1920 * 1080 * 3);
            assert!(
                output
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .all(|p| *p == [10, 20, 200])
            );
        }
    }

    #[test]
    fn chroma_replaces_green_keeps_skin() {
        let bg = solid_png("chroma-bg.png", 32, 16, [10, 20, 200, 255]);
        let mut p = Processor::new(16, 16).unwrap();
        p.configure(Effects {
            background: "chroma".into(),
            background_image: Some(bg.clone()),
            ..Effects::default()
        })
        .unwrap();
        let mut frame: Vec<u8> = (0..256)
            .flat_map(|i| {
                if i % 16 < 8 {
                    [0, 200, 0]
                } else {
                    [200, 150, 120]
                }
            })
            .collect();
        p.process(&mut frame).unwrap();
        assert!(
            px(&frame, 16, 2, 5)
                .iter()
                .zip([10, 20, 200])
                .all(|(a, b)| a.abs_diff(b) <= 2),
            "{:?}",
            px(&frame, 16, 2, 5)
        );
        assert_eq!(px(&frame, 16, 12, 5), [200, 150, 120]);
        assert!(p.process(&mut frame[..30]).is_err());
        std::fs::remove_file(bg).unwrap();
    }

    #[test]
    fn overlay_is_centred_and_opacity_applies() {
        let red = solid_png("overlay.png", 8, 8, [255, 0, 0, 255]);
        let mut p = Processor::new(32, 32).unwrap();
        let fx = |o| Effects {
            overlay_image: Some(red.clone()),
            overlay_opacity: o,
            ..Effects::default()
        };
        let run = |p: &mut Processor| {
            let mut f = vec![0u8; 32 * 32 * 3];
            p.process(&mut f).unwrap();
            f
        };
        p.configure(fx(1.0)).unwrap();
        let f = run(&mut p);
        assert_eq!(px(&f, 32, 16, 16), [255, 0, 0]);
        assert_eq!(px(&f, 32, 11, 16), [0, 0, 0]);
        assert_eq!(px(&f, 32, 12, 16), [255, 0, 0]);
        p.configure(fx(0.5)).unwrap();
        assert!(run(&mut p)[(16 * 32 + 16) * 3].abs_diff(128) <= 1);
        p.configure(fx(0.0)).unwrap();
        assert!(run(&mut p).iter().all(|&v| v == 0));
        std::fs::remove_file(red).unwrap();
    }

    #[test]
    fn asset_paths_are_validated() {
        let mut p = Processor::new(32, 32).unwrap();
        let missing = Effects {
            overlay_image: Some(temp("missing.png")),
            ..Effects::default()
        };
        assert!(p.configure(missing).is_err());
        let dir = Effects {
            background: "replace".into(),
            background_image: Some(std::env::temp_dir()),
            ..Effects::default()
        };
        assert!(
            p.configure(dir)
                .unwrap_err()
                .to_string()
                .contains("not a regular file")
        );
        let junk = temp("junk.png");
        std::fs::write(&junk, b"not an image").unwrap();
        let fx = Effects {
            overlay_image: Some(junk.clone()),
            ..Effects::default()
        };
        assert!(p.configure(fx).is_err());
        std::fs::remove_file(junk).unwrap();
        // failed configures must not have changed the active (empty) configuration
        let mut f = vec![9u8; 32 * 32 * 3];
        p.process(&mut f).unwrap();
        assert!(f.iter().all(|&v| v == 9));
    }
}
