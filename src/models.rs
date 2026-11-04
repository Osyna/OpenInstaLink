//! On-machine model contracts. Results stay paired with the exact submitted frame.

pub mod history;
pub mod local;
mod worker;
pub use local::{Cancelled, Client};
pub use worker::{Engine, EngineStatus};

use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use image::{ImageEncoder, ImageFormat, ImageReader, Limits};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::media::Frame;

pub const MODEL_IDS: [&str; 8] = [
    "mediapipe-hand",
    "mediapipe-pose",
    "mediapipe-face",
    "depth-anything-v2-small",
    "marigold-depth-v1-1",
    "sam2.1-tiny",
    "florence-2-base",
    "yolo11n",
];
pub const MAX_PIXELS: u64 = 40_000_000;
pub const MAX_OUTPUT_BYTES: usize = 32 * 1024 * 1024;

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub output: &'static str,
    pub interval_ms: u64,
}

pub const MODELS: [ModelSpec; 8] = [
    ModelSpec {
        id: MODEL_IDS[0],
        label: "MediaPipe Hand",
        output: "json",
        interval_ms: 33,
    },
    ModelSpec {
        id: MODEL_IDS[1],
        label: "MediaPipe Pose",
        output: "json",
        interval_ms: 33,
    },
    ModelSpec {
        id: MODEL_IDS[2],
        label: "MediaPipe Face",
        output: "json",
        interval_ms: 33,
    },
    ModelSpec {
        id: MODEL_IDS[3],
        label: "Depth Anything V2 Small",
        output: "image",
        interval_ms: 100,
    },
    ModelSpec {
        id: MODEL_IDS[4],
        label: "Marigold Depth v1.1",
        output: "image",
        interval_ms: 1000,
    },
    ModelSpec {
        id: MODEL_IDS[5],
        label: "SAM 2.1 Tiny",
        output: "json",
        interval_ms: 100,
    },
    ModelSpec {
        id: MODEL_IDS[6],
        label: "Florence-2 Base",
        output: "json",
        interval_ms: 500,
    },
    ModelSpec {
        id: MODEL_IDS[7],
        label: "YOLO11n",
        output: "json",
        interval_ms: 33,
    },
];

pub fn model_spec(id: &str) -> Result<&'static ModelSpec> {
    MODELS
        .iter()
        .find(|s| s.id == id)
        .with_context(|| format!("unsupported model {id}; use models list"))
}

pub fn defaults(model: &str) -> Result<Value> {
    Ok(match model_spec(model)?.id {
        "mediapipe-hand" => json!({"max_results":2,"min_confidence":0.5}),
        "mediapipe-pose" => json!({"max_results":1,"min_confidence":0.5}),
        "mediapipe-face" => json!({"max_results":1,"min_confidence":0.5,"blendshapes":true}),
        "depth-anything-v2-small" => json!({"colormap":"gray"}),
        "marigold-depth-v1-1" => json!({"steps":4,"ensemble":1,"colormap":"gray"}),
        "sam2.1-tiny" => json!({"multimask":false}),
        "florence-2-base" => json!({"task":"caption"}),
        "yolo11n" => json!({"threshold":0.25,"top_k":100}),
        _ => unreachable!(),
    })
}

fn numeric(options: &Value, key: &str, lo: f64, hi: f64, integer: bool) -> Result<()> {
    if let Some(value) = options.get(key) {
        let number = value
            .as_f64()
            .with_context(|| format!("{key} must be a number"))?;
        ensure!(
            number.is_finite() && number >= lo && number <= hi,
            "{key} must be between {lo} and {hi}"
        );
        ensure!(
            !integer || value.as_u64().is_some(),
            "{key} must be an integer"
        );
    }
    Ok(())
}

fn boolean(options: &Value, key: &str) -> Result<()> {
    ensure!(
        options.get(key).is_none_or(Value::is_boolean),
        "{key} must be true or false"
    );
    Ok(())
}

fn choice(options: &Value, key: &str, choices: &[&str]) -> Result<()> {
    if let Some(value) = options.get(key) {
        ensure!(
            value.as_str().is_some_and(|s| choices.contains(&s)),
            "{key} must be one of {}",
            choices.join(", ")
        );
    }
    Ok(())
}

fn coordinates(value: &Value, count: usize, name: &str) -> Result<Vec<f64>> {
    let string = value
        .as_str()
        .with_context(|| format!("{name} must be comma-separated pixel coordinates"))?;
    ensure!(string.len() <= 160, "{name} is too long");
    let values = string
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("{name} contains an invalid coordinate"))?;
    ensure!(
        values.len() == count && values.iter().all(|v| v.is_finite()),
        "{name} needs {count} finite coordinates"
    );
    Ok(values)
}

pub fn validate_options(model: &str, options: &Value, width: u32, height: u32) -> Result<()> {
    model_spec(model)?;
    ensure!(
        width > 0 && height > 0 && u64::from(width) * u64::from(height) <= MAX_PIXELS,
        "image must contain 1..={MAX_PIXELS} pixels"
    );
    let object = options
        .as_object()
        .context("model options must be a JSON object")?;
    let allowed: &[&str] = match model {
        "mediapipe-hand" | "mediapipe-pose" => &["max_results", "min_confidence"],
        "mediapipe-face" => &["max_results", "min_confidence", "blendshapes"],
        "depth-anything-v2-small" => &["colormap"],
        "marigold-depth-v1-1" => &["steps", "ensemble", "colormap"],
        "sam2.1-tiny" => &["points", "box", "multimask"],
        "florence-2-base" => &["task"],
        "yolo11n" => &["threshold", "top_k"],
        _ => unreachable!(),
    };
    for key in object.keys() {
        ensure!(
            allowed.contains(&key.as_str()),
            "{model} does not accept option {key}"
        );
    }
    match model {
        "mediapipe-hand" | "mediapipe-pose" | "mediapipe-face" => {
            numeric(options, "max_results", 1.0, 10.0, true)?;
            numeric(options, "min_confidence", 0.1, 0.9, false)?;
            boolean(options, "blendshapes")?;
        }
        "depth-anything-v2-small" | "marigold-depth-v1-1" => {
            choice(options, "colormap", &["gray", "inferno"])?;
            numeric(options, "steps", 1.0, 10.0, true)?;
            numeric(options, "ensemble", 1.0, 5.0, true)?;
        }
        "florence-2-base" => choice(options, "task", &["caption", "detect", "ocr", "regions"])?,
        "yolo11n" => {
            numeric(options, "threshold", 0.0, 1.0, false)?;
            numeric(options, "top_k", 1.0, 300.0, true)?;
        }
        "sam2.1-tiny" => {
            boolean(options, "multimask")?;
            if let Some(points) = options.get("points") {
                let points = points
                    .as_array()
                    .context("points must be an array of 'x,y' strings")?;
                ensure!(points.len() <= 16, "SAM accepts at most 16 positive points");
                for point in points {
                    let p = coordinates(point, 2, "point")?;
                    ensure!(
                        p[0] >= 0.0
                            && p[0] <= f64::from(width)
                            && p[1] >= 0.0
                            && p[1] <= f64::from(height),
                        "point is outside the {width}x{height} image"
                    );
                }
            }
            if let Some(value) = options.get("box") {
                let b = coordinates(value, 4, "box")?;
                ensure!(
                    b[0] >= 0.0
                        && b[1] >= 0.0
                        && b[0] < b[2]
                        && b[1] < b[3]
                        && b[2] <= f64::from(width)
                        && b[3] <= f64::from(height),
                    "box must be x1,y1,x2,y2, ordered and inside the {width}x{height} image"
                );
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[derive(Clone)]
pub struct RuntimeConfig {
    pub timeout: Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub disk_bytes: u64,
    #[serde(default)]
    pub device: String,
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub loaded: bool,
    #[serde(default)]
    pub loading: bool,
    #[serde(default)]
    pub available: Option<bool>,
    #[serde(default)]
    pub unavailable_reason: Option<String>,
    #[serde(default)]
    pub presentation: Value,
    #[serde(default)]
    pub option_defaults: Value,
}

#[derive(Clone)]
pub struct RawOutput {
    pub job_id: String,
    pub content_type: String,
    pub bytes: Arc<Vec<u8>>,
    pub elapsed: Duration,
    pub warning: Option<String>,
}

impl std::fmt::Debug for RawOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawOutput")
            .field("job_id", &self.job_id)
            .field("content_type", &self.content_type)
            .field("bytes_len", &self.bytes.len())
            .field("elapsed", &self.elapsed)
            .field("warning", &self.warning)
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub visibility: Option<f64>,
    #[serde(default)]
    pub presence: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hand {
    pub handedness: String,
    pub score: f64,
    pub landmarks: Vec<Point>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pose {
    pub landmarks: Vec<Point>,
    pub world_landmarks: Vec<Point>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blendshape {
    pub name: String,
    pub score: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Face {
    pub landmarks: Vec<Point>,
    #[serde(default)]
    pub blendshapes: Vec<Blendshape>,
    #[serde(default)]
    pub transformation_matrix: Option<[[f64; 4]; 4]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Box2d {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Detection {
    pub label: String,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub class_id: Option<u32>,
    #[serde(rename = "box")]
    pub bounds: Box2d,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Region {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(rename = "box")]
    pub bounds: Box2d,
    #[serde(default)]
    pub quad: Option<[[f64; 2]; 4]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskOutline {
    pub score: f64,
    pub area_fraction: f64,
    pub bbox: Option<Box2d>,
    pub polygon: Vec<[f64; 2]>,
}
#[derive(Clone, Debug)]
pub enum AnalysisData {
    Hands(Vec<Hand>),
    Poses(Vec<Pose>),
    Faces(Vec<Face>),
    Detections(Vec<Detection>),
    Regions(Vec<Region>),
    Caption,
    Segmentation(Vec<MaskOutline>),
    RelativeDepth,
}

#[derive(Clone)]
pub struct Analysis {
    pub model: String,
    pub source: Arc<Frame>,
    pub output: RawOutput,
    pub text: String,
    pub annotated: Arc<Frame>,
    pub depth: Option<Arc<Frame>>,
    pub data: AnalysisData,
}

fn points_valid(points: &[Point], count: usize, image: &Dimensions, world: bool) -> Result<()> {
    ensure!(
        points.len() == count,
        "expected {count} landmarks, received {}",
        points.len()
    );
    for p in points {
        ensure!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "non-finite landmark"
        );
        // Off-image joints are valid, but unbounded coordinates must not reach rasterization.
        ensure!(
            world
                || (p.x.abs() <= f64::from(image.width) * 8.0
                    && p.y.abs() <= f64::from(image.height) * 8.0),
            "landmark coordinates are not image pixels"
        );
        for score in [p.visibility, p.presence].into_iter().flatten() {
            probability(score)?;
        }
        ensure!(
            p.name.as_ref().is_none_or(|s| s.len() <= 128),
            "landmark name is too long"
        );
    }
    Ok(())
}
fn probability(n: f64) -> Result<()> {
    ensure!(
        n.is_finite() && (0.0..=1.0).contains(&n),
        "invalid probability {n}"
    );
    Ok(())
}
fn box_valid(b: &Box2d, image: &Dimensions) -> Result<()> {
    ensure!(
        [b.x1, b.y1, b.x2, b.y2].iter().all(|n| n.is_finite()),
        "non-finite box"
    );
    ensure!(
        b.x1 <= b.x2
            && b.y1 <= b.y2
            && b.x1 >= -1.0
            && b.y1 >= -1.0
            && b.x2 <= f64::from(image.width) + 1.0
            && b.y2 <= f64::from(image.height) + 1.0,
        "box is not inside the source image"
    );
    Ok(())
}
fn xy_valid(p: [f64; 2], image: &Dimensions) -> Result<()> {
    ensure!(
        p[0].is_finite()
            && p[1].is_finite()
            && p[0] >= -1.0
            && p[1] >= -1.0
            && p[0] <= f64::from(image.width) + 1.0
            && p[1] <= f64::from(image.height) + 1.0,
        "outline point is outside the source image"
    );
    Ok(())
}

#[derive(Deserialize)]
struct JsonHeader {
    image: Dimensions,
    text: String,
    #[serde(default)]
    model: Option<String>,
}
#[derive(Deserialize)]
struct HandsPayload {
    hands: Vec<Hand>,
}
#[derive(Deserialize)]
struct PosesPayload {
    poses: Vec<Pose>,
}
#[derive(Deserialize)]
struct FacesPayload {
    faces: Vec<Face>,
}
#[derive(Deserialize)]
struct DetectionPayload {
    detections: Vec<Detection>,
}
#[derive(Deserialize)]
struct RegionsPayload {
    regions: Vec<Region>,
}
#[derive(Deserialize)]
struct SegmentationPayload {
    masks: Vec<MaskOutline>,
}
#[derive(Deserialize)]
struct FlorenceTask {
    task: String,
}

pub fn interpret(model: &str, source: Arc<Frame>, output: RawOutput) -> Result<Analysis> {
    let spec = model_spec(model)?;
    ensure!(
        source.width > 0
            && source.height > 0
            && u64::from(source.width) * u64::from(source.height) <= MAX_PIXELS,
        "invalid source dimensions"
    );
    ensure!(
        source.rgb.len() as u64 == u64::from(source.width) * u64::from(source.height) * 3,
        "source RGB size does not match dimensions"
    );
    ensure!(
        !output.bytes.is_empty() && output.bytes.len() <= MAX_OUTPUT_BYTES,
        "invalid model result size"
    );
    let mime = output.content_type.split(';').next().unwrap_or("").trim();
    if spec.output == "image" {
        ensure!(
            mime == "image/png",
            "{model} returned {mime}, expected image/png"
        );
        let (w, h) =
            ImageReader::with_format(Cursor::new(output.bytes.as_slice()), ImageFormat::Png)
                .into_dimensions()
                .context("invalid depth PNG")?;
        ensure!(
            (w, h) == (source.width, source.height),
            "depth result {w}x{h} does not match source {}x{}",
            source.width,
            source.height
        );
        let mut reader =
            ImageReader::with_format(Cursor::new(output.bytes.as_slice()), ImageFormat::Png);
        let mut limits = Limits::default();
        limits.max_image_width = Some(w);
        limits.max_image_height = Some(h);
        limits.max_alloc = Some(u64::from(w) * u64::from(h) * 8 + 1024 * 1024);
        reader.limits(limits);
        let rgb = reader
            .decode()
            .context("cannot decode depth PNG")?
            .into_rgb8()
            .into_raw();
        let depth = Arc::new(Frame {
            width: w,
            height: h,
            rgb,
            sequence: source.sequence,
        });
        return Ok(Analysis {
            model: model.into(),
            source,
            output,
            text: "Relative depth · near = bright · not metres; brightness is normalized per image"
                .into(),
            annotated: depth.clone(),
            depth: Some(depth),
            data: AnalysisData::RelativeDepth,
        });
    }
    ensure!(
        mime == "application/json",
        "{model} returned {mime}, expected application/json"
    );
    let header: JsonHeader =
        serde_json::from_slice(&output.bytes).context("invalid model JSON envelope")?;
    ensure!(
        (header.image.width, header.image.height) == (source.width, source.height),
        "model result dimensions do not match submitted frame"
    );
    ensure!(
        header.model.as_deref().is_none_or(|id| id == model),
        "model result belongs to a different model"
    );
    ensure!(
        header.text.len() <= 64 * 1024,
        "model result text exceeds 64 KiB"
    );
    let mut text = header.text;
    let dims = header.image;
    let data = match model {
        "mediapipe-hand" => {
            let p: HandsPayload =
                serde_json::from_slice(&output.bytes).context("invalid hand result")?;
            ensure!(p.hands.len() <= 10, "too many hands");
            for h in &p.hands {
                ensure!(
                    matches!(h.handedness.as_str(), "Left" | "Right"),
                    "invalid hand side"
                );
                probability(h.score)?;
                points_valid(&h.landmarks, 21, &dims, false)?;
            }
            AnalysisData::Hands(p.hands)
        }
        "mediapipe-pose" => {
            let p: PosesPayload =
                serde_json::from_slice(&output.bytes).context("invalid pose result")?;
            ensure!(p.poses.len() <= 10, "too many poses");
            for p in &p.poses {
                points_valid(&p.landmarks, 33, &dims, false)?;
                points_valid(&p.world_landmarks, 33, &dims, true)?;
            }
            AnalysisData::Poses(p.poses)
        }
        "mediapipe-face" => {
            let p: FacesPayload =
                serde_json::from_slice(&output.bytes).context("invalid face result")?;
            ensure!(p.faces.len() <= 10, "too many faces");
            for face in &p.faces {
                points_valid(&face.landmarks, 478, &dims, false)?;
                ensure!(face.blendshapes.len() <= 52, "too many blendshapes");
                for b in &face.blendshapes {
                    ensure!(b.name.len() <= 128, "blendshape name too long");
                    probability(b.score)?;
                }
                if let Some(matrix) = face.transformation_matrix {
                    ensure!(
                        matrix.iter().flatten().all(|v| v.is_finite()),
                        "non-finite face transformation"
                    );
                }
            }
            AnalysisData::Faces(p.faces)
        }
        "yolo11n" => AnalysisData::Detections(parse_detections(&output.bytes, &dims, 300)?),
        "florence-2-base" => {
            let task: FlorenceTask =
                serde_json::from_slice(&output.bytes).context("missing Florence task")?;
            match task.task.as_str() {
                "caption" => AnalysisData::Caption,
                "detect" => AnalysisData::Detections(parse_detections(&output.bytes, &dims, 1024)?),
                "ocr" | "regions" => {
                    let p: RegionsPayload = serde_json::from_slice(&output.bytes)
                        .context("invalid Florence regions")?;
                    ensure!(p.regions.len() <= 1024, "too many regions");
                    for r in &p.regions {
                        box_valid(&r.bounds, &dims)?;
                        ensure!(
                            r.text.as_ref().is_none_or(|s| s.len() <= 64 * 1024),
                            "region text too long"
                        );
                        if let Some(q) = r.quad {
                            for xy in q {
                                xy_valid(xy, &dims)?;
                            }
                        }
                    }
                    AnalysisData::Regions(p.regions)
                }
                other => bail!("unsupported Florence result task {other}"),
            }
        }
        "sam2.1-tiny" => {
            let p: SegmentationPayload =
                serde_json::from_slice(&output.bytes).context("invalid SAM result")?;
            ensure!(p.masks.len() <= 3, "too many SAM candidates");
            for m in &p.masks {
                probability(m.score)?;
                probability(m.area_fraction)?;
                if let Some(b) = &m.bbox {
                    box_valid(b, &dims)?;
                }
                ensure!(m.polygon.len() <= 200, "SAM outline exceeds 200 vertices");
                for &xy in &m.polygon {
                    xy_valid(xy, &dims)?;
                }
            }
            text.push_str("\nOutline only: the local SAM backend returns the largest component, not a full-resolution alpha mask.");
            AnalysisData::Segmentation(p.masks)
        }
        _ => unreachable!(),
    };
    let annotated = match &data {
        AnalysisData::Caption => source.clone(),
        _ => {
            let mut rendered = Frame {
                width: source.width,
                height: source.height,
                rgb: source.rgb.clone(),
                sequence: source.sequence,
            };
            annotate(&mut rendered, &data, &mut text);
            Arc::new(rendered)
        }
    };
    Ok(Analysis {
        model: model.into(),
        source,
        output,
        text,
        annotated,
        depth: None,
        data,
    })
}

fn parse_detections(bytes: &[u8], dims: &Dimensions, limit: usize) -> Result<Vec<Detection>> {
    let p: DetectionPayload = serde_json::from_slice(bytes).context("invalid detection result")?;
    ensure!(p.detections.len() <= limit, "too many detections");
    for d in &p.detections {
        box_valid(&d.bounds, dims)?;
        ensure!(d.label.len() <= 4096, "detection label too long");
        if let Some(s) = d.score {
            probability(s)?;
        }
    }
    Ok(p.detections)
}

const COLORS: [[u8; 3]; 6] = [
    [255, 210, 70],
    [70, 220, 255],
    [255, 130, 160],
    [120, 245, 150],
    [210, 160, 255],
    [255, 170, 80],
];
const HAND_EDGES: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];
const POSE_EDGES: [(usize, usize); 22] = [
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
];

fn visible(p: &Point) -> bool {
    p.visibility.is_none_or(|v| v >= 0.5) && p.presence.is_none_or(|v| v >= 0.5)
}
fn annotate(frame: &mut Frame, data: &AnalysisData, text: &mut String) {
    match data {
        AnalysisData::Hands(hands) => {
            for (i, h) in hands.iter().enumerate() {
                joints(frame, &h.landmarks, &HAND_EDGES, COLORS[i % 6]);
                text.push_str(&format!(
                    "\n{}: {} hand · {:.0}% (selfie handedness convention)",
                    i + 1,
                    h.handedness,
                    h.score * 100.0
                ));
            }
        }
        AnalysisData::Poses(poses) => {
            for (i, p) in poses.iter().enumerate() {
                joints(frame, &p.landmarks, &POSE_EDGES, COLORS[i % 6]);
                for (a, b) in [(28, 30), (30, 32), (27, 31), (28, 32)] {
                    if visible(&p.landmarks[a]) && visible(&p.landmarks[b]) {
                        line(
                            frame,
                            [p.landmarks[a].x, p.landmarks[a].y],
                            [p.landmarks[b].x, p.landmarks[b].y],
                            COLORS[i % 6],
                        );
                    }
                }
            }
        }
        AnalysisData::Faces(faces) => {
            for (i, f) in faces.iter().enumerate() {
                for p in &f.landmarks {
                    dot(
                        frame,
                        p.x.round() as i32,
                        p.y.round() as i32,
                        1,
                        COLORS[i % 6],
                    );
                }
                if !f.blendshapes.is_empty() {
                    text.push_str(&format!(
                        "\nFace {}: {}",
                        i + 1,
                        f.blendshapes
                            .iter()
                            .take(5)
                            .map(|b| format!("{} {:.0}%", b.name, b.score * 100.0))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
        AnalysisData::Detections(detections) => {
            for (i, d) in detections.iter().enumerate() {
                rectangle(frame, &d.bounds, COLORS[i % 6]);
                number(
                    frame,
                    d.bounds.x1 as i32 + 3,
                    d.bounds.y1 as i32 + 3,
                    i + 1,
                    COLORS[i % 6],
                );
                text.push_str(&format!(
                    "\n{}: {}{}",
                    i + 1,
                    d.label,
                    d.score
                        .map(|s| format!(" · {:.0}%", s * 100.0))
                        .unwrap_or_default()
                ));
            }
        }
        AnalysisData::Regions(regions) => {
            for (i, r) in regions.iter().enumerate() {
                if let Some(q) = &r.quad {
                    polygon(frame, q, COLORS[i % 6]);
                } else {
                    rectangle(frame, &r.bounds, COLORS[i % 6]);
                }
                number(
                    frame,
                    r.bounds.x1 as i32 + 3,
                    r.bounds.y1 as i32 + 3,
                    i + 1,
                    COLORS[i % 6],
                );
                if let Some(s) = &r.text {
                    text.push_str(&format!("\n{}: {}", i + 1, s));
                }
            }
        }
        AnalysisData::Segmentation(masks) => {
            for (i, m) in masks.iter().enumerate() {
                polygon(frame, &m.polygon, COLORS[i % 6]);
            }
        }
        AnalysisData::Caption | AnalysisData::RelativeDepth => {}
    }
}
fn joints(frame: &mut Frame, points: &[Point], edges: &[(usize, usize)], color: [u8; 3]) {
    for &(a, b) in edges {
        if visible(&points[a]) && visible(&points[b]) {
            line(
                frame,
                [points[a].x, points[a].y],
                [points[b].x, points[b].y],
                color,
            );
        }
    }
    for p in points {
        if visible(p) {
            dot(frame, p.x.round() as i32, p.y.round() as i32, 2, color);
        }
    }
}
fn rectangle(frame: &mut Frame, b: &Box2d, c: [u8; 3]) {
    polygon(
        frame,
        &[[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]],
        c,
    );
}
fn polygon(frame: &mut Frame, points: &[[f64; 2]], c: [u8; 3]) {
    if points.len() < 2 {
        return;
    }
    for pair in points.windows(2) {
        line(frame, pair[0], pair[1], c);
    }
    line(frame, *points.last().unwrap(), points[0], c);
}
fn dot(frame: &mut Frame, x: i32, y: i32, r: i32, c: [u8; 3]) {
    for yy in y - r..=y + r {
        for xx in x - r..=x + r {
            if xx >= 0 && yy >= 0 && xx < frame.width as i32 && yy < frame.height as i32 {
                let n = (yy as usize * frame.width as usize + xx as usize) * 3;
                frame.rgb[n..n + 3].copy_from_slice(&c);
            }
        }
    }
}
fn line(frame: &mut Frame, mut a: [f64; 2], mut b: [f64; 2], c: [u8; 3]) {
    // Liang–Barsky clips valid off-image joints before the integer rasterizer.
    let delta = [b[0] - a[0], b[1] - a[1]];
    let mut lo: f64 = 0.0;
    let mut hi: f64 = 1.0;
    for (p, q) in [
        (-delta[0], a[0]),
        (delta[0], f64::from(frame.width - 1) - a[0]),
        (-delta[1], a[1]),
        (delta[1], f64::from(frame.height - 1) - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                lo = lo.max(t);
            } else {
                hi = hi.min(t);
            }
            if lo > hi {
                return;
            }
        }
    }
    b = [a[0] + hi * delta[0], a[1] + hi * delta[1]];
    a = [a[0] + lo * delta[0], a[1] + lo * delta[1]];
    let (mut x, mut y, x2, y2) = (
        a[0].round() as i32,
        a[1].round() as i32,
        b[0].round() as i32,
        b[1].round() as i32,
    );
    let dx = (x2 - x).abs();
    let sx = if x < x2 { 1 } else { -1 };
    let dy = -(y2 - y).abs();
    let sy = if y < y2 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        dot(frame, x, y, 1, c);
        if x == x2 && y == y2 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x += sx;
        }
        if twice <= dx {
            error += dx;
            y += sy;
        }
    }
}
fn number(frame: &mut Frame, x: i32, y: i32, n: usize, c: [u8; 3]) {
    const DIGITS: [[u8; 5]; 10] = [
        [7, 5, 5, 5, 7],
        [2, 6, 2, 2, 7],
        [7, 1, 7, 4, 7],
        [7, 1, 7, 1, 7],
        [5, 5, 7, 1, 1],
        [7, 4, 7, 1, 7],
        [7, 4, 7, 5, 7],
        [7, 1, 1, 1, 1],
        [7, 5, 7, 5, 7],
        [7, 5, 7, 1, 7],
    ];
    for (i, d) in n.to_string().bytes().enumerate() {
        for (row, bits) in DIGITS[(d - b'0') as usize].iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    dot(frame, x + i as i32 * 8 + col * 2, y + row as i32 * 2, 1, c);
                }
            }
        }
    }
}

fn new_output(path: &Path, extension: &str) -> Result<std::fs::File> {
    ensure!(
        path.extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(extension)),
        "output must end in .{extension}"
    );
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "cannot create {} (existing files are never overwritten)",
                path.display()
            )
        })
}
pub fn save_result(analysis: &Analysis, path: &Path) -> Result<()> {
    let extension = if analysis.depth.is_some() {
        "png"
    } else {
        "json"
    };
    let mut file = new_output(path, extension)?;
    let result = file
        .write_all(analysis.output.bytes.as_slice())
        .and_then(|_| file.sync_all());
    if let Err(error) = result {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error).context("cannot save model result");
    }
    Ok(())
}
pub fn save_visualization(analysis: &Analysis, path: &Path) -> Result<()> {
    let file = new_output(path, "png")?;
    let frame = &analysis.annotated;
    if let Err(error) = image::codecs::png::PngEncoder::new(file).write_image(
        &frame.rgb,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    ) {
        let _ = std::fs::remove_file(path);
        return Err(error).context("cannot save model visualization");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(sequence: u64) -> Arc<Frame> {
        Arc::new(Frame {
            width: 32,
            height: 24,
            rgb: vec![0; 32 * 24 * 3],
            sequence,
        })
    }
    fn output(value: Value) -> RawOutput {
        RawOutput {
            job_id: "a".repeat(32),
            content_type: "application/json".into(),
            bytes: Arc::new(serde_json::to_vec(&value).unwrap()),
            elapsed: Duration::from_millis(12),
            warning: None,
        }
    }
    #[test]
    fn sam_coordinates_validate_exact_input_and_point_limit() {
        assert!(
            validate_options(
                "sam2.1-tiny",
                &json!({"points":["31,23"],"box":"0,0,32,24"}),
                32,
                24
            )
            .is_ok()
        );
        for v in [
            json!({"points":["33,1"]}),
            json!({"points":["NaN,2"]}),
            json!({"points":vec!["1,1";17]}),
            json!({"box":"20,0,10,24"}),
            json!({"box":[0,0,1,1]}),
        ] {
            assert!(validate_options("sam2.1-tiny", &v, 32, 24).is_err(), "{v}");
        }
    }
    #[test]
    fn options_reject_cross_model_fields_and_type_confusion() {
        for (m, v) in [
            ("yolo11n", json!({"threshold":true})),
            ("mediapipe-face", json!({"blendshapes":"true"})),
            ("mediapipe-pose", json!({"max_results":1.5})),
            ("marigold-depth-v1-1", json!({"ensemble":6})),
            ("florence-2-base", json!({"task":"question"})),
            ("depth-anything-v2-small", json!({"steps":4})),
        ] {
            assert!(validate_options(m, &v, 32, 24).is_err(), "{m}: {v}");
        }
    }
    #[test]
    fn boxes_are_drawn_on_the_submitted_frame_not_mutated_source() {
        let source = frame(71);
        let result=interpret("yolo11n",source.clone(),output(json!({"model":"yolo11n","text":"person","image":{"width":32,"height":24},"detections":[{"label":"person","score":0.9,"box":{"x1":4,"y1":4,"x2":20,"y2":20}}]}))).unwrap();
        assert_eq!(result.source.sequence, 71);
        assert_eq!(result.annotated.sequence, 71);
        assert_eq!(
            &source.rgb[(4 * 32 + 4) * 3..(4 * 32 + 4) * 3 + 3],
            &[0, 0, 0]
        );
        assert_eq!(
            &result.annotated.rgb[(4 * 32 + 4) * 3..(4 * 32 + 4) * 3 + 3],
            &COLORS[0]
        );
        assert_eq!(
            &result.annotated.rgb[(23 * 32 + 31) * 3..(23 * 32 + 31) * 3 + 3],
            &[0, 0, 0]
        );
    }
    #[test]
    fn foreign_model_or_dimensions_are_not_rendered() {
        for v in [
            json!({"model":"mediapipe-hand","text":"","image":{"width":32,"height":24},"detections":[]}),
            json!({"model":"yolo11n","text":"","image":{"width":64,"height":24},"detections":[]}),
        ] {
            assert!(interpret("yolo11n", frame(1), output(v)).is_err());
        }
    }
    #[test]
    fn no_detection_is_valid_but_missing_detection_schema_is_not() {
        let value = json!({"model":"yolo11n","text":"no objects","image":{"width":32,"height":24},"detections":[]});
        let result = interpret("yolo11n", frame(2), output(value.clone())).unwrap();
        assert_eq!(result.annotated.rgb, vec![0; 32 * 24 * 3]);
        let mut missing = value;
        missing.as_object_mut().unwrap().remove("detections");
        assert!(interpret("yolo11n", frame(2), output(missing)).is_err());
    }
    #[test]
    fn polygon_is_an_outline_not_a_fabricated_alpha_mask() {
        let result=interpret("sam2.1-tiny",frame(2),output(json!({"text":"1 mask","image":{"width":32,"height":24},"masks":[{"score":0.8,"area_fraction":0.3,"bbox":{"x1":4,"y1":4,"x2":20,"y2":20},"polygon":[[4,4],[20,4],[20,20],[4,20]]}]}))).unwrap();
        assert_eq!(
            &result.annotated.rgb[(4 * 32 + 10) * 3..(4 * 32 + 10) * 3 + 3],
            &COLORS[0]
        );
        assert_eq!(
            &result.annotated.rgb[(12 * 32 + 12) * 3..(12 * 32 + 12) * 3 + 3],
            &[0, 0, 0]
        );
    }
    #[test]
    fn depth_matches_source_and_rejects_wrong_size_before_decode() {
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                &vec![127; 32 * 24 * 3],
                32,
                24,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let raw = RawOutput {
            job_id: "b".repeat(32),
            content_type: "image/png".into(),
            bytes: Arc::new(bytes),
            elapsed: Duration::ZERO,
            warning: None,
        };
        let result = interpret("depth-anything-v2-small", frame(3), raw.clone()).unwrap();
        assert_eq!(result.depth.unwrap().rgb, vec![127; 32 * 24 * 3]);
        let other = Arc::new(Frame {
            width: 16,
            height: 24,
            rgb: vec![0; 16 * 24 * 3],
            sequence: 3,
        });
        assert!(interpret("marigold-depth-v1-1", other, raw).is_err());
    }
    #[test]
    fn off_image_skeleton_lines_are_clipped_not_projected_to_border() {
        let mut f = Frame {
            width: 32,
            height: 24,
            rgb: vec![0; 32 * 24 * 3],
            sequence: 1,
        };
        line(&mut f, [-100.0, 12.0], [100.0, 12.0], COLORS[0]);
        assert_eq!(&f.rgb[(12 * 32) * 3..(12 * 32) * 3 + 3], &COLORS[0]);
        let before = f.rgb.clone();
        line(&mut f, [-100.0, -10.0], [100.0, -10.0], COLORS[1]);
        assert_eq!(f.rgb, before);
    }
    #[test]
    fn landmark_families_preserve_world_pose_expressions_and_source_geometry() {
        let point = json!({"x":8.0,"y":8.0,"z":0.1,"visibility":0.9,"presence":0.9});
        let mut hand = json!({"model":"mediapipe-hand","image":{"width":32,"height":24},"text":"1 hand","hands":[{"handedness":"Left","score":0.9,"landmarks":vec![point.clone();21]}]});
        let result = interpret("mediapipe-hand", frame(4), output(hand.clone())).unwrap();
        assert_eq!(
            &result.annotated.rgb[(8 * 32 + 8) * 3..(8 * 32 + 8) * 3 + 3],
            &COLORS[0]
        );
        hand["hands"][0]["landmarks"].as_array_mut().unwrap().pop();
        assert!(interpret("mediapipe-hand", frame(4), output(hand)).is_err());
        let world = json!({"x":0.25,"y":-0.5,"z":0.75});
        let pose = json!({"model":"mediapipe-pose","image":{"width":32,"height":24},"text":"1 pose","poses":[{"landmarks":vec![point.clone();33],"world_landmarks":vec![world;33]}]});
        let result = interpret("mediapipe-pose", frame(5), output(pose)).unwrap();
        let AnalysisData::Poses(poses) = result.data else {
            panic!("pose data lost")
        };
        assert_eq!(poses[0].world_landmarks[0].y, -0.5);
        let matrix = [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let face = json!({"model":"mediapipe-face","image":{"width":32,"height":24},"text":"1 face","faces":[{"landmarks":vec![point;478],"blendshapes":[{"name":"jawOpen","score":0.8}],"transformation_matrix":matrix}]});
        let result = interpret("mediapipe-face", frame(6), output(face)).unwrap();
        let AnalysisData::Faces(faces) = result.data else {
            panic!("face data lost")
        };
        assert_eq!(faces[0].transformation_matrix.unwrap()[0][3], 0.3);
        assert_eq!(faces[0].blendshapes[0].score, 0.8);
    }
    #[test]
    fn result_exports_never_truncate_an_existing_file() {
        let result = interpret(
            "florence-2-base",
            frame(1),
            output(json!({"image":{"width":32,"height":24},"text":"a room","task":"caption"})),
        )
        .unwrap();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "link-camera-model-export-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        let raw = dir.join("result.json");
        let visual = dir.join("result.png");
        std::fs::write(&raw, b"keep raw").unwrap();
        std::fs::write(&visual, b"keep image").unwrap();
        assert!(save_result(&result, &raw).is_err());
        assert!(save_visualization(&result, &visual).is_err());
        assert_eq!(std::fs::read(&raw).unwrap(), b"keep raw");
        assert_eq!(std::fs::read(&visual).unwrap(), b"keep image");
        std::fs::remove_file(raw).unwrap();
        std::fs::remove_file(visual).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
