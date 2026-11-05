//! Insta360 Link control through the Linux `uvcvideo` driver (upstream `insta360-link`).
//!
//! Unsupported or unreadable selectors leave other controls available. A failed USB transport
//! aborts the scan instead of multiplying kernel timeouts. `Control.value == ""` is unknown,
//! never a default to write. No raw selectors or firmware operations are exposed.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use insta360_link as link;
use link::{
    AudioMode, CameraMode, Exposure, Features, Focus, Framing, Gestures, ImageSettings, Model,
    PanTiltUpdate, PowerLine, Preset, State, TrackingArea, TrackingSpeed, WhiteBalance,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub model: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlKind {
    Number,
    Toggle,
    Choice,
}

/// `min`/`max`/`step` are 0 for toggles and choices. Toggle values are `true`/`false`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Control {
    pub key: String,
    pub label: String,
    pub group: String,
    pub kind: ControlKind,
    pub value: String,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub options: Vec<String>,
    /// Hardware default from V4L2 metadata, in the same unit as `value`; `None` when the camera
    /// does not report one (never a guess).
    #[serde(default)]
    pub default: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub device: Device,
    pub firmware: String,
    pub serial: String,
    pub streaming: bool,
    pub controls: Vec<Control>,
    pub warnings: Vec<String>,
    /// Real tracking capability and state; see [`TrackingStatus`].
    pub tracking: TrackingStatus,
}

/// A camera-stored view. Slots are 0-9.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PresetInfo {
    pub slot: u8,
    pub name: String,
    pub mode: String,
    pub pan: f64,
    pub tilt: f64,
    pub zoom: f64,
}

/// A pause-track zone: rectangle as fractions of the frame from the top-left, plus the gimbal
/// angle it was defined at.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PauseAreaInfo {
    pub active: bool,
    pub pan: f64,
    pub tilt: f64,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackingAreaInfo {
    pub enabled: bool,
    pub pan: [f64; 2],
    pub tilt: [f64; 2],
}

/// What "follow the subject" means on this camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrackingCapability {
    /// Model not recognised, so nothing is claimed.
    None,
    /// Fixed camera (Link 2C family): the picture is reframed, no motor moves. `mode` = `auto-framing`.
    AutoFraming,
    /// Motorized gimbal follows a person. `mode` = `track`.
    SubjectTracking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrackingState {
    /// The camera does not report enough to say.
    Unavailable,
    /// No program is streaming: modes cannot be read or changed.
    NotStreaming,
    /// Privacy mode is on.
    Privacy,
    Off,
    /// Between modes.
    Switching,
    /// Tracking was requested and the camera reports nobody to follow.
    WaitingForSubject,
    Active,
    /// Another special mode (whiteboard, DeskView, overhead) is running.
    OtherMode,
}

/// Read from the camera on every snapshot. Pause-track zones cannot be read as "currently
/// holding", so a pause by a zone is never claimed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackingStatus {
    pub capability: TrackingCapability,
    /// `Some(false)` on Link 2C / 2C Pro; `None` for an unrecognised model.
    pub physical_gimbal: Option<bool>,
    /// The `mode` control value that starts following on this camera.
    pub mode_key: Option<String>,
    pub state: TrackingState,
    /// Short text for a status line.
    pub label: String,
    /// Why it is unavailable, paused, waiting or off; empty when there is nothing to explain.
    pub reason: String,
}

pub fn devices() -> Result<Vec<Device>> {
    let list = link::Camera::list().context("listing Insta360 Link cameras")?;
    Ok(list.iter().map(device_of).collect())
}

fn device_of(info: &link::DeviceInfo) -> Device {
    Device {
        id: info.id.to_string(),
        name: info
            .product
            .clone()
            .unwrap_or_else(|| info.model.name().to_owned()),
        model: info.model.to_string(),
    }
}

/// Wraps the upstream camera. Not `Send`: keep it on the thread that opened it.
pub struct CameraBackend {
    camera: link::Camera,
    ranges: OnceCell<std::result::Result<HashMap<String, V4l>, String>>,
    /// Readback taken by the last autosave, reused by the next `snapshot` within `READBACK_TTL`.
    readback: RefCell<Option<(Instant, Reading)>>,
}

const MODE_KEYS: [&str; 6] = [
    "normal",
    "track",
    "whiteboard",
    "overhead",
    "deskview",
    "auto-framing",
];
const FRAMING_KEYS: [&str; 4] = ["head", "half-body", "whole-body", "manual"];
const SPEED_KEYS: [&str; 3] = ["slow", "normal", "fast"];
const AUDIO_KEYS: [&str; 3] = ["voice-suppression", "voice-focus", "music-balance"];
const POWER_KEYS: [&str; 4] = ["off", "50", "60", "auto"];
const AUTO_MANUAL: [&str; 2] = ["auto", "manual"];

/// `(key, label, group)` of every switch held in `Features`.
const FEATURE_KEYS: [(&str, &str, &str); 9] = [
    ("hdr", "HDR", "Picture"),
    ("horizontal-flip", "Horizontal flip", "View"),
    ("vertical-flip", "Vertical flip", "View"),
    ("horizontal-correction", "Horizontal correction", "View"),
    ("ai-zoom", "AI Zoom", "Framing"),
    ("smart-adjustment", "Smart adjustment", "Framing"),
    ("noise-canceling", "Noise canceling", "Audio"),
    ("mute-in-privacy", "Mute in privacy", "Audio"),
    ("gestures", "Hand gestures", "Gestures"),
];
const GESTURE_KEYS: [(&str, &str); 3] = [
    ("gesture-palm", "Palm toggles tracking"),
    ("gesture-zoom", "L gesture zooms"),
    ("gesture-whiteboard", "V toggles whiteboard"),
];

/// Link 2C and 2C Pro have no motor: `Some(false)`. `None`: unknown model, nothing claimed.
fn physical_gimbal(model: Model) -> Option<bool> {
    match model {
        Model::Link2C | Model::Link2CPro => Some(false),
        Model::Link | Model::Link2 | Model::Link2Pro => Some(true),
        _ => None,
    }
}

/// The reference app offers Auto Framing (never single tracking or overhead) without a gimbal.
const FIXED_MODE_KEYS: [&str; 4] = ["normal", "auto-framing", "whiteboard", "deskview"];

fn mode_keys(model: Model) -> &'static [&'static str] {
    if physical_gimbal(model) == Some(false) {
        &FIXED_MODE_KEYS
    } else {
        &MODE_KEYS
    }
}

/// These modes steer pan, tilt and zoom themselves: a saved view must not be replayed over them
/// (a manual zoom would switch the camera's framing to manual).
fn camera_driven(mode: Option<CameraMode>) -> bool {
    matches!(mode, Some(CameraMode::Tracking | CameraMode::AutoFraming))
}

/// Tracking with nobody in view leaves the camera busy and never reporting a mode: for a
/// tracking request that busy readback is the expected outcome.
fn mode_path(mode: Option<CameraMode>) -> &'static str {
    if mode == Some(CameraMode::Tracking) {
        "!mode_tracking"
    } else {
        "mode"
    }
}

/// `saved` is the mode the user last asked for, used only to tell "waiting for a subject" from a
/// plain mode change while the camera reports busy.
fn tracking_status_of(model: Model, r: &Reading, saved: Option<CameraMode>) -> TrackingStatus {
    let gimbal = physical_gimbal(model);
    let (capability, key, target, name) = match gimbal {
        Some(false) => (
            TrackingCapability::AutoFraming,
            Some("auto-framing"),
            Some(CameraMode::AutoFraming),
            "Auto framing",
        ),
        Some(true) => (
            TrackingCapability::SubjectTracking,
            Some("track"),
            Some(CameraMode::Tracking),
            "Tracking",
        ),
        None => (TrackingCapability::None, None, None, "Tracking"),
    };
    let mut reason = String::new();
    let state = if capability == TrackingCapability::None {
        reason = format!("{model}: tracking support is not known for this model");
        TrackingState::Unavailable
    } else if r.streaming == Some(false) {
        reason =
            "No program is streaming video; the camera applies modes only while streaming".into();
        TrackingState::NotStreaming
    } else if r.privacy == Some(true) {
        reason = "Privacy mode is on; the camera follows nobody until it is switched off".into();
        TrackingState::Privacy
    } else if r.mode_busy {
        if capability == TrackingCapability::SubjectTracking && saved == Some(CameraMode::Tracking)
        {
            reason = "Tracking is on but the camera reports nobody to follow".into();
            TrackingState::WaitingForSubject
        } else {
            reason = "The camera is between modes".into();
            TrackingState::Switching
        }
    } else {
        match r.state.mode {
            None => {
                let why = r
                    .failures
                    .iter()
                    .find(|f| f.starts_with("mode:"))
                    .map_or("the camera does not report a mode", String::as_str);
                reason = format!("Cannot read the mode: {why}");
                TrackingState::Unavailable
            }
            Some(m) if Some(m) == target => TrackingState::Active,
            Some(CameraMode::Normal) => TrackingState::Off,
            Some(m) => {
                reason = if m == CameraMode::Tracking {
                    "The camera reports `track` mode, which needs a gimbal this camera does not have".into()
                } else {
                    format!(
                        "`{m}` mode is running; {name} is off until the camera returns to normal mode"
                    )
                };
                TrackingState::OtherMode
            }
        }
    };
    let mut notes = Vec::new();
    if gimbal == Some(false)
        && matches!(
            state,
            TrackingState::Active | TrackingState::Off | TrackingState::OtherMode
        )
    {
        notes.push("This camera has no motorized gimbal; Auto Framing reframes the picture only");
    }
    if state == TrackingState::Active {
        let reframing = matches!(
            r.state.tracking.framing,
            Some(Framing::Head | Framing::HalfBody | Framing::WholeBody)
        );
        if reframing && r.state.features.ai_zoom == Some(false) {
            notes.push("AI Zoom is off, so the framing style is stored but does not reframe");
        }
        if capability == TrackingCapability::SubjectTracking && r.pause_areas == Some(true) {
            notes.push("Pause-track zones are on: tracking holds while the subject is inside one");
        }
    }
    if !notes.is_empty() {
        if !reason.is_empty() {
            notes.insert(0, reason.as_str());
        }
        reason = notes.join(". ");
    }
    let label = match state {
        TrackingState::Unavailable => format!("{name} unavailable"),
        TrackingState::NotStreaming => format!("{name} needs a video stream"),
        TrackingState::Privacy => format!("{name} paused by privacy mode"),
        TrackingState::Off => format!("{name} off"),
        TrackingState::Switching => "Camera is changing mode".to_owned(),
        TrackingState::WaitingForSubject => format!("{name} waiting for a subject"),
        TrackingState::Active => format!("{name} on"),
        TrackingState::OtherMode => format!(
            "{name} off ({} mode)",
            r.state.mode.map_or_else(String::new, |m| m.to_string())
        ),
    };
    TrackingStatus {
        capability,
        physical_gimbal: gimbal,
        mode_key: key.map(str::to_owned),
        state,
        label,
        reason,
    }
}

fn feature_slot<'a>(f: &'a mut Features, key: &str) -> Option<&'a mut Option<bool>> {
    Some(match key {
        "hdr" => &mut f.hdr,
        "horizontal-flip" => &mut f.horizontal_flip,
        "vertical-flip" => &mut f.vertical_flip,
        "horizontal-correction" => &mut f.horizontal_correction,
        "ai-zoom" => &mut f.ai_zoom,
        "smart-adjustment" => &mut f.smart_adjustment,
        "noise-canceling" => &mut f.noise_canceling,
        "mute-in-privacy" => &mut f.mute_in_privacy,
        "gestures" => &mut f.gestures,
        _ => return None,
    })
}

fn gesture_slot<'a>(g: &'a mut Gestures, key: &str) -> Option<&'a mut Option<bool>> {
    Some(match key {
        "gesture-palm" => &mut g.tracking,
        "gesture-zoom" => &mut g.zoom,
        "gesture-whiteboard" => &mut g.whiteboard,
        _ => return None,
    })
}

/// Everything read from the camera in one pass.
#[derive(Default)]
struct Reading {
    state: State,
    streaming: Option<bool>,
    mode_busy: bool,
    position_unknown: bool,
    privacy: Option<bool>,
    pause_areas: Option<bool>,
    tracking_area: Option<TrackingArea>,
    /// Reads that failed for a reason other than "not supported".
    failures: Vec<String>,
}

impl Reading {
    fn take<T>(&mut self, what: &str, result: link::Result<T>) -> Result<Option<T>> {
        match result {
            Ok(value) => Ok(Some(value)),
            Err(link::Error::Unsupported) => Ok(None),
            Err(error) if transport_lost(&error) => Err(error)
                .with_context(|| format!("camera USB communication failed while reading {what}; unplug/reconnect the camera")),
            Err(error) => {
                self.failures.push(format!("{what}: {error}"));
                Ok(None)
            }
        }
    }
}

fn transport_lost(error: &link::Error) -> bool {
    match error {
        link::Error::Io { source, .. } => {
            matches!(source.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::NotConnected)
                // Linux EIO, ENXIO, ENODEV, ESHUTDOWN, ETIMEDOUT and EREMOTEIO.
                // EPIPE/EINVAL can belong to one unsupported selector, not a lost device.
                || matches!(source.raw_os_error(), Some(5 | 6 | 19 | 108 | 110 | 121))
        }
        _ => false,
    }
}

#[cfg(feature = "gui")]
pub(crate) fn is_transport_failure(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<link::Error>())
        .any(transport_lost)
}

impl CameraBackend {
    pub fn open(device: Option<&str>) -> Result<Self> {
        let camera = match device {
            Some(id) => link::Camera::open(id).with_context(|| format!("opening {id}"))?,
            None => link::Camera::open_first().context("opening the first Insta360 Link camera")?,
        };
        Ok(Self::new(camera))
    }

    fn new(camera: link::Camera) -> Self {
        Self {
            camera,
            ranges: OnceCell::new(),
            readback: RefCell::default(),
        }
    }

    pub fn device(&self) -> Device {
        device_of(self.camera.info())
    }

    fn id(&self) -> &str {
        self.camera.info().id.as_str()
    }

    /// Ranges of the standard controls from the kernel, read once. The upstream crate validates
    /// every write against the same ranges but does not expose them.
    fn ranges(&self) -> &std::result::Result<HashMap<String, V4l>, String> {
        self.ranges
            .get_or_init(|| v4l_list(self.id()).map_err(|e| format!("{e:#}")))
    }

    fn range(&self, name: &str, nominal: (f64, f64, f64)) -> (f64, f64, f64) {
        match self.ranges() {
            Ok(map) => match map.get(name) {
                Some(V4l {
                    min: Some(min),
                    max: Some(max),
                    step,
                    ..
                }) => (
                    f64::from(*min),
                    f64::from(*max),
                    f64::from(step.unwrap_or(1).max(1)),
                ),
                _ => nominal,
            },
            Err(_) => nominal,
        }
    }

    fn read(&self, extras: bool) -> Result<Reading> {
        let cam = &self.camera;
        let mut r = Reading::default();
        r.streaming = r.take("streaming state", cam.is_streaming())?;
        match cam.image() {
            Ok(image) => r.state.image = image,
            Err(error) => {
                // image() fails as a whole if any one control fails (often exposure). Fall back to
                // the individual V4L2 values so the other picture controls stay available.
                r.take::<()>("image settings", Err(error))?;
                match v4l_list(self.id()).map(|map| v4l_image(&map)) {
                    Ok(image) => r.state.image = image,
                    Err(e) => r.failures.push(format!("image settings via V4L2: {e:#}")),
                }
                r.state.image.exposure = r.take("exposure", cam.exposure())?;
            }
        }
        // Not streaming, the camera reports Normal whatever it will do once video starts.
        if r.streaming != Some(false) {
            match cam.mode() {
                Ok(mode) => r.state.mode = Some(mode),
                Err(link::Error::Busy) => r.mode_busy = true,
                Err(error) => {
                    r.take::<()>("mode", Err(error))?;
                }
            }
        }
        r.state.audio_mode = r.take("audio mode", cam.audio_mode())?;
        r.state.tracking.framing = r.take("framing", cam.framing())?;
        r.state.tracking.speed = r.take("tracking speed", cam.tracking_speed())?;
        r.state.features = r
            .take("feature switches", cam.features())?
            .unwrap_or_default();
        r.state.gestures = r
            .take("gesture switches", cam.gestures())?
            .unwrap_or_default();
        r.state.gimbal.zoom = r.take("zoom", cam.zoom())?;
        match cam.position() {
            Ok(p) => {
                r.state.gimbal.pan = Some(p.pan);
                r.state.gimbal.tilt = Some(p.tilt);
            }
            Err(link::Error::PositionUnknown) => r.position_unknown = true,
            Err(error) => {
                r.take::<()>("pan/tilt", Err(error))?;
            }
        }
        if extras {
            r.privacy = r.take("privacy", cam.privacy())?;
            r.pause_areas = r.take("pause-track areas", cam.pause_areas_enabled())?;
            if cam.info().model == Model::Link2Pro {
                r.tracking_area = r.take("tracking area", cam.tracking_area())?;
            }
        }
        Ok(r)
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let cached = self
            .readback
            .take()
            .filter(|(at, _)| at.elapsed() < READBACK_TTL);
        let mut r = match cached {
            Some((_, r)) => r,
            None => self.read(true)?,
        };
        let firmware = r
            .take("firmware version", self.camera.firmware_version())?
            .unwrap_or_default();
        let serial = r
            .take("serial number", self.camera.serial_number())?
            .unwrap_or_default();
        let tracking = self.tracking_status(&r);
        let mut warnings = std::mem::take(&mut r.failures);
        if let Err(reason) = self.ranges() {
            warnings.push(format!(
                "control ranges are nominal, not read from the camera: {reason}"
            ));
        }
        if r.streaming == Some(false) {
            warnings.push(
                "camera is not streaming: mode is hidden and vendor settings (mode, framing, \
                 exposure, switches, presets, areas, privacy) cannot be written"
                    .into(),
            );
        }
        if r.mode_busy {
            warnings.push("mode: camera is between modes or tracking has nobody to follow".into());
        }
        if r.position_unknown {
            warnings.push("pan/tilt: camera reports an invalid position; set both axes".into());
        }
        Ok(Snapshot {
            device: self.device(),
            firmware,
            serial,
            streaming: r.streaming.unwrap_or(false),
            controls: self.controls(&r),
            warnings,
            tracking,
        })
    }

    /// One control, freshly read.
    pub fn control(&self, key: &str) -> Result<Control> {
        let snapshot = self.snapshot()?;
        snapshot
            .controls
            .into_iter()
            .find(|c| c.key == key)
            .ok_or_else(|| {
                anyhow!(
                    "control `{key}` is not available on this camera ({})",
                    snapshot.warnings.join("; ")
                )
            })
    }

    fn controls(&self, r: &Reading) -> Vec<Control> {
        let mut out = Vec::new();
        let state = &r.state;
        let image = &state.image;
        let model = self.camera.info().model;
        let num = |key: &str,
                   label: &str,
                   group: &str,
                   value: Option<String>,
                   (min, max, step): (f64, f64, f64)| Control {
            key: key.into(),
            label: label.into(),
            group: group.into(),
            kind: ControlKind::Number,
            value: value.unwrap_or_default(),
            min,
            max,
            step,
            options: Vec::new(),
            default: None,
        };
        let choice =
            |key: &str, label: &str, group: &str, value: Option<String>, options: &[&str]| {
                Control {
                    key: key.into(),
                    label: label.into(),
                    group: group.into(),
                    kind: ControlKind::Choice,
                    value: value.unwrap_or_default(),
                    min: 0.0,
                    max: 0.0,
                    step: 0.0,
                    options: options.iter().map(|s| (*s).to_owned()).collect(),
                    default: None,
                }
            };
        let toggle = |key: &str, label: &str, group: &str, on: bool| Control {
            key: key.into(),
            label: label.into(),
            group: group.into(),
            kind: ControlKind::Toggle,
            value: on.to_string(),
            min: 0.0,
            max: 0.0,
            step: 0.0,
            options: Vec::new(),
            default: None,
        };

        for (key, label, value, name, nominal) in [
            (
                "brightness",
                "Brightness",
                image.brightness,
                "brightness",
                (0.0, 100.0, 1.0),
            ),
            (
                "contrast",
                "Contrast",
                image.contrast,
                "contrast",
                (0.0, 100.0, 1.0),
            ),
            (
                "saturation",
                "Saturation",
                image.saturation,
                "saturation",
                (0.0, 100.0, 1.0),
            ),
            (
                "sharpness",
                "Sharpness",
                image.sharpness,
                "sharpness",
                (0.0, 100.0, 1.0),
            ),
            ("hue", "Hue", image.hue, "hue", (-15.0, 15.0, 1.0)),
        ] {
            if let Some(v) = value {
                out.push(num(
                    key,
                    label,
                    "Picture",
                    Some(v.to_string()),
                    self.range(name, nominal),
                ));
            }
        }

        // Manual numbers stay readable while the matching automatic mode is on, from V4L2.
        let live = if matches!(image.white_balance, Some(WhiteBalance::Auto))
            || matches!(image.focus, Some(Focus::Auto))
        {
            v4l_list(self.id()).ok()
        } else {
            None
        };
        let live = |name: &str| {
            live.as_ref()
                .and_then(|m| m.get(name))
                .and_then(|c| c.value)
                .map(|v| v.to_string())
        };
        if let Some(wb) = image.white_balance {
            let (mode, temperature) = match wb {
                WhiteBalance::Auto => ("auto", live("white_balance_temperature")),
                WhiteBalance::Manual { temperature } => ("manual", Some(temperature.to_string())),
            };
            out.push(choice(
                "white-balance",
                "White balance",
                "Picture",
                Some(mode.into()),
                &AUTO_MANUAL,
            ));
            out.push(num(
                "white-balance-temperature",
                "White balance (K)",
                "Picture",
                temperature,
                self.range("white_balance_temperature", (2000.0, 10000.0, 1.0)),
            ));
        }
        if let Some(focus) = image.focus {
            let (mode, position) = match focus {
                Focus::Auto => ("auto", live("focus_absolute")),
                Focus::Manual { position } => ("manual", Some(position.to_string())),
            };
            out.push(choice(
                "focus",
                "Focus",
                "Picture",
                Some(mode.into()),
                &AUTO_MANUAL,
            ));
            out.push(num(
                "focus-position",
                "Focus position",
                "Picture",
                position,
                self.range("focus_absolute", (0.0, 100.0, 1.0)),
            ));
        }
        if let Some(p) = image.power_line {
            out.push(choice(
                "power-line",
                "Anti-flicker",
                "Picture",
                Some(p.to_string()),
                &POWER_KEYS,
            ));
        }
        if let Some(e) = image.exposure {
            let mode = if matches!(e, Exposure::Auto { .. }) {
                "auto"
            } else {
                "manual"
            };
            out.push(choice(
                "exposure",
                "Exposure",
                "Picture",
                Some(mode.into()),
                &AUTO_MANUAL,
            ));
            match e {
                Exposure::Auto { compensation } => {
                    out.push(num(
                        "exposure-compensation",
                        "Exposure compensation (EV)",
                        "Picture",
                        Some(compensation.to_string()),
                        (-5.0, 5.0, 0.01),
                    ));
                }
                Exposure::Manual { iso, shutter } => {
                    out.push(num(
                        "exposure-iso",
                        "ISO",
                        "Picture",
                        Some(iso.to_string()),
                        (1.0, 65535.0, 1.0),
                    ));
                    out.push(num(
                        "exposure-shutter",
                        "Shutter (1/n s)",
                        "Picture",
                        Some(shutter.to_string()),
                        (1.0, 65535.0, 1.0),
                    ));
                }
            }
        }

        if let Some(zoom) = state.gimbal.zoom {
            let range = self
                .camera
                .zoom_limits()
                .map_or((1.0, 4.0, 0.01), |r| (*r.start(), *r.end(), 0.01));
            out.push(num("zoom", "Zoom", "View", Some(zoom.to_string()), range));
        }
        if let Ok(limits) = self.camera.position_limits() {
            let known = state.gimbal.pan.is_some() && state.gimbal.tilt.is_some();
            if known || r.position_unknown {
                let range = |l: &std::ops::RangeInclusive<f64>| (*l.start(), *l.end(), 1.0);
                out.push(num(
                    "pan",
                    "Pan (°)",
                    "View",
                    state.gimbal.pan.map(|v| v.to_string()),
                    range(&limits.pan),
                ));
                out.push(num(
                    "tilt",
                    "Tilt (°)",
                    "View",
                    state.gimbal.tilt.map(|v| v.to_string()),
                    range(&limits.tilt),
                ));
            }
        }
        if state.mode.is_some() || r.mode_busy {
            let mut keys: Vec<&str> = mode_keys(model).to_vec();
            let current = state.mode.map(|m| m.to_string());
            // Never hide what the camera reports, even a mode this model should not offer.
            if let Some(c) = current.as_deref().filter(|c| !keys.contains(c)) {
                keys.push(c);
            }
            out.push(choice("mode", "Mode", "View", current.clone(), &keys));
        }
        if let Some(on) = r.privacy {
            out.push(toggle("privacy", "Privacy mode", "View", on));
        }

        let mut features = state.features;
        for (key, label, group) in FEATURE_KEYS {
            if let Some(on) = feature_slot(&mut features, key).and_then(|s| *s) {
                out.push(toggle(key, label, group, on));
            }
        }
        if let Some(f) = state.tracking.framing {
            out.push(choice(
                "framing",
                "Framing",
                "Framing",
                Some(f.to_string()),
                &FRAMING_KEYS,
            ));
        }
        if let Some(s) = state.tracking.speed {
            out.push(choice(
                "tracking-speed",
                "Tracking speed",
                "Framing",
                Some(s.to_string()),
                &SPEED_KEYS,
            ));
        }
        if let Some(on) = r.pause_areas {
            out.push(toggle("pause-areas", "Pause-track areas", "Framing", on));
        }
        if let Some(on) = r.tracking_area.as_ref().map(|a| a.enabled) {
            out.push(toggle(
                "tracking-area",
                "Tracking area limits",
                "Framing",
                on,
            ));
        }
        if let Some(m) = state.audio_mode {
            out.push(choice(
                "audio-mode",
                "Microphone mode",
                "Audio",
                Some(m.to_string()),
                &AUDIO_KEYS,
            ));
        }
        let mut gestures = state.gestures;
        for (key, label) in GESTURE_KEYS {
            let label = if key == "gesture-palm" && physical_gimbal(model) == Some(false) {
                "Palm toggles auto framing"
            } else {
                label
            };
            if let Some(on) = gesture_slot(&mut gestures, key).and_then(|s| *s) {
                out.push(toggle(key, label, "Gestures", on));
            }
        }
        for c in out.iter_mut().filter(|c| c.kind == ControlKind::Number) {
            c.default = self.number_default(&c.key, c.min, c.max);
        }
        // Stable grouping for UIs that render in order.
        let rank = |g: &str| {
            ["Picture", "View", "Framing", "Audio", "Gestures"]
                .iter()
                .position(|x| *x == g)
                .unwrap_or(9)
        };
        out.sort_by_key(|c| rank(&c.group));
        out
    }

    /// Validated write through the upstream typed API. Vendor settings need an active stream.
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        let expect = if key == "mode" {
            vec![mode_path(CameraMode::from_str(value.trim()).ok()).to_owned()]
        } else {
            expected_after_set(key)
        };
        self.persisted(&expect, || {
            self.write(key, value.trim())
                .with_context(|| format!("cannot set {key} to `{}`", value.trim()))
        })
    }

    fn write(&self, key: &str, value: &str) -> Result<()> {
        let cam = &self.camera;
        let image = |f: &dyn Fn(&mut ImageSettings)| -> Result<()> {
            let mut s = ImageSettings::default();
            f(&mut s);
            Ok(cam.set_image(&s)?)
        };
        match key {
            "brightness" => {
                let v = int(value)?;
                image(&|s| s.brightness = Some(v))
            }
            "contrast" => {
                let v = int(value)?;
                image(&|s| s.contrast = Some(v))
            }
            "saturation" => {
                let v = int(value)?;
                image(&|s| s.saturation = Some(v))
            }
            "sharpness" => {
                let v = int(value)?;
                image(&|s| s.sharpness = Some(v))
            }
            "hue" => {
                let v = int(value)?;
                image(&|s| s.hue = Some(v))
            }
            "white-balance" => match value {
                "auto" => image(&|s| s.white_balance = Some(WhiteBalance::Auto)),
                "manual" => {
                    if matches!(
                        cam.image()?.white_balance,
                        Some(WhiteBalance::Manual { .. })
                    ) {
                        return Ok(());
                    }
                    let t = v4l_value(self.id(), "white_balance_temperature")
                        .context("cannot read the current temperature; set white-balance-temperature instead")?;
                    image(&|s| {
                        s.white_balance = Some(WhiteBalance::Manual {
                            temperature: t.max(0) as u32,
                        })
                    })
                }
                _ => bail!("expected `auto` or `manual`"),
            },
            "white-balance-temperature" => {
                let temperature = value
                    .parse::<u32>()
                    .map_err(|_| anyhow!("expected a temperature in kelvin"))?;
                image(&|s| s.white_balance = Some(WhiteBalance::Manual { temperature }))
            }
            "focus" => match value {
                "auto" => image(&|s| s.focus = Some(Focus::Auto)),
                "manual" => {
                    if matches!(cam.image()?.focus, Some(Focus::Manual { .. })) {
                        return Ok(());
                    }
                    let p = v4l_value(self.id(), "focus_absolute").context(
                        "cannot read the current focus position; set focus-position instead",
                    )?;
                    image(&|s| {
                        s.focus = Some(Focus::Manual {
                            position: p.max(0) as u32,
                        })
                    })
                }
                _ => bail!("expected `auto` or `manual`"),
            },
            "focus-position" => {
                let position = value
                    .parse::<u32>()
                    .map_err(|_| anyhow!("expected a focus position"))?;
                image(&|s| s.focus = Some(Focus::Manual { position }))
            }
            "power-line" => {
                let p = PowerLine::from_str(value)?;
                image(&|s| s.power_line = Some(p))
            }
            "exposure" => {
                let current = cam.exposure()?;
                match (value, current) {
                    ("auto", Exposure::Auto { .. }) | ("manual", Exposure::Manual { .. }) => Ok(()),
                    // The camera forgets the compensation while manual, so neutral is the honest start.
                    ("auto", _) => Ok(cam.set_exposure(&Exposure::Auto { compensation: 0.0 })?),
                    // Start from what the camera is using now so the picture does not jump.
                    ("manual", _) => {
                        let (iso, shutter) = cam.exposure_in_effect()?;
                        Ok(cam.set_exposure(&Exposure::Manual { iso, shutter })?)
                    }
                    _ => bail!("expected `auto` or `manual`"),
                }
            }
            "exposure-compensation" => Ok(cam.set_exposure(&Exposure::Auto {
                compensation: float(value)?,
            })?),
            "exposure-iso" | "exposure-shutter" => {
                let v = value
                    .parse::<u16>()
                    .map_err(|_| anyhow!("expected an integer from 1 to 65535"))?;
                let (mut iso, mut shutter) = match cam.exposure()? {
                    Exposure::Manual { iso, shutter } => (iso, shutter),
                    Exposure::Auto { .. } => cam.exposure_in_effect()?,
                };
                if key == "exposure-iso" {
                    iso = v
                } else {
                    shutter = v
                }
                Ok(cam.set_exposure(&Exposure::Manual { iso, shutter })?)
            }
            "zoom" => Ok(cam.set_zoom(float(value)?)?),
            "pan" => self.move_to(Some(float(value)?), None),
            "tilt" => self.move_to(None, Some(float(value)?)),
            "mode" => {
                let mode = CameraMode::from_str(value)?;
                self.check_mode(mode)?;
                Ok(cam.set_mode(mode)?)
            }
            "framing" => Ok(cam.set_framing(Framing::from_str(value)?)?),
            "tracking-speed" => Ok(cam.set_tracking_speed(TrackingSpeed::from_str(value)?)?),
            "audio-mode" => Ok(cam.set_audio_mode(AudioMode::from_str(value)?)?),
            "privacy" => Ok(cam.set_privacy(boolean(value)?)?),
            "pause-areas" => Ok(cam.set_pause_areas_enabled(boolean(value)?)?),
            "tracking-area" => {
                let mut area = cam.tracking_area()?;
                area.enabled = boolean(value)?;
                Ok(cam.set_tracking_area(&area)?)
            }
            _ => {
                let on = boolean(value);
                let mut features = Features::default();
                if let Some(slot) = feature_slot(&mut features, key) {
                    *slot = Some(on?);
                    return Ok(cam.set_features(&features)?);
                }
                let mut gestures = Gestures::default();
                if let Some(slot) = gesture_slot(&mut gestures, key) {
                    *slot = Some(on?);
                    return Ok(cam.set_gestures(&gestures)?);
                }
                bail!(
                    "unknown control `{key}`; known keys: {}",
                    KNOWN_KEYS.join(", ")
                )
            }
        }
    }

    pub fn position(&self) -> Result<(f64, f64)> {
        let p = self.camera.position()?;
        Ok((p.pan, p.tilt))
    }

    /// Moves pan/tilt through the standard V4L2 controls; an axis left `None` keeps its angle.
    /// The camera decides whether that is a motor or a digital view; Unsupported is reported as is.
    pub fn set_position(&self, pan: Option<f64>, tilt: Option<f64>) -> Result<()> {
        let mut expect = Vec::new();
        if pan.is_some() {
            expect.push("gimbal.pan".to_owned());
        }
        if tilt.is_some() {
            expect.push("gimbal.tilt".to_owned());
        }
        self.persisted(&expect, || self.move_to(pan, tilt))
    }

    fn move_to(&self, pan: Option<f64>, tilt: Option<f64>) -> Result<()> {
        for v in [pan, tilt].into_iter().flatten() {
            if !v.is_finite() {
                bail!("angle must be a finite number");
            }
        }
        Ok(self.camera.set_position(PanTiltUpdate { pan, tilt })?)
    }

    pub fn recenter(&self) -> Result<()> {
        self.set_position(Some(0.0), Some(0.0))
    }

    pub fn mode_status(&self) -> Result<Option<String>> {
        Ok(self.camera.mode_status()?.map(|s| format!("{s:?}")))
    }

    /// Used preset slots (0-9). Needs an active stream.
    pub fn presets(&self) -> Result<Vec<PresetInfo>> {
        let mut out = Vec::new();
        for slot in 0..link::PRESET_SLOTS {
            if let Some(p) = self
                .camera
                .preset(slot)
                .with_context(|| format!("reading preset slot {slot}"))?
            {
                out.push(preset_info(slot, p));
            }
        }
        Ok(out)
    }

    pub fn save_preset(&self, slot: u8, name: &str) -> Result<PresetInfo> {
        Ok(preset_info(slot, self.camera.save_preset(slot, name)?))
    }

    pub fn recall_preset(&self, slot: u8) -> Result<PresetInfo> {
        self.persisted(&["gimbal.zoom".to_owned()], || {
            (self.camera.recall_preset(slot))
                .map(|p| preset_info(slot, p))
                .map_err(anyhow::Error::from)
        })
    }

    pub fn delete_preset(&self, slot: u8) -> Result<()> {
        Ok(self.camera.delete_preset(slot)?)
    }

    pub fn pause_areas(&self) -> Result<Vec<PauseAreaInfo>> {
        Ok(self.pause_areas_raw()?)
    }

    fn pause_areas_raw(&self) -> link::Result<Vec<PauseAreaInfo>> {
        Ok(self
            .camera
            .pause_areas()?
            .into_iter()
            .map(|a| PauseAreaInfo {
                active: a.active,
                pan: a.pan,
                tilt: a.tilt,
                x: a.x,
                y: a.y,
                width: a.width,
                height: a.height,
            })
            .collect())
    }

    /// Replaces all zones (at most 6); an empty slice clears them. Ranges are validated upstream.
    pub fn set_pause_areas(&self, areas: &[PauseAreaInfo]) -> Result<()> {
        self.persisted(&["!pause_zones".to_owned()], || {
            (self.camera.set_pause_areas(&pause_areas_link(areas))).map_err(anyhow::Error::from)
        })
    }

    pub fn tracking_area(&self) -> Result<TrackingAreaInfo> {
        Ok(tracking_info(&self.camera.tracking_area()?))
    }

    pub fn set_tracking_area(&self, area: &TrackingAreaInfo) -> Result<()> {
        self.persisted(&["!tracking_area".to_owned()], || {
            (self.camera.set_tracking_area(&tracking_link(area))).map_err(anyhow::Error::from)
        })
    }

    /// Every setting that could be read. Unsupported and transiently busy settings are `None`;
    /// any other read failure is an error so a profile is never saved silently incomplete.
    pub fn export_state(&self) -> Result<State> {
        let r = self.read(false)?;
        if !r.failures.is_empty() {
            bail!(
                "cannot read the complete camera state: {}",
                r.failures.join("; ")
            );
        }
        Ok(r.state)
    }

    /// Applies the `Some` settings in order and stops at the first the camera rejects, so a
    /// failure can leave earlier settings applied.
    pub fn apply_state(&self, state: &State) -> Result<()> {
        if let Some(mode) = state.mode {
            self.check_mode(mode)?;
        }
        let mut state = *state;
        if camera_driven(state.mode) {
            state.gimbal = Default::default();
        }
        let mut expect = applied_paths(&state)?;
        for p in expect.iter_mut().filter(|p| p.as_str() == "mode") {
            *p = mode_path(state.mode).to_owned();
        }
        self.persisted(&expect, || {
            self.camera.apply(&state).context("applying camera state")
        })
    }

    pub fn save_profile(&self, name: &str) -> Result<()> {
        let path = profile_path(name)?;
        let mut json = serde_json::to_vec_pretty(&self.export_state()?)?;
        json.push(b'\n');
        let dir = path.parent().expect("profile path has a parent");
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        // Dot-prefixed temp file: never listed as a profile, replaced atomically by rename.
        let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
        let _ = fs::remove_file(&tmp);
        let written = (|| -> std::io::Result<()> {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            file.write_all(&json)?;
            file.sync_all()?;
            fs::rename(&tmp, &path)
        })();
        if let Err(error) = written {
            let _ = fs::remove_file(&tmp);
            return Err(error).with_context(|| format!("writing profile {}", path.display()));
        }
        Ok(())
    }

    pub fn load_profile(&self, name: &str) -> Result<()> {
        let path = profile_path(name)?;
        self.apply_state(&read_state_file(&path)?)
    }
}

impl CameraBackend {
    fn tracking_status(&self, r: &Reading) -> TrackingStatus {
        let saved = if r.mode_busy { self.saved_mode() } else { None };
        tracking_status_of(self.camera.info().model, r, saved)
    }

    /// The mode last saved for this camera (what the user asked for); lock-free read of an
    /// atomically replaced file. Any problem means "unknown".
    fn saved_mode(&self) -> Option<CameraMode> {
        let identity = identity_of(self.id()).ok()?;
        let path = saved_path(&cameras_dir().ok()?, &identity);
        load_saved(&path, &identity).ok()??.state.mode
    }

    fn mode_allowed(&self, mode: CameraMode) -> bool {
        mode_keys(self.camera.info().model).contains(&mode.to_string().as_str())
    }

    fn check_mode(&self, mode: CameraMode) -> Result<()> {
        if self.mode_allowed(mode) {
            return Ok(());
        }
        bail!(
            "{} has no motorized gimbal, so `{mode}` mode is unavailable; {}",
            self.camera.info().model,
            if mode == CameraMode::Tracking {
                "use `auto-framing` to reframe automatically"
            } else {
                "choose normal, auto-framing, whiteboard or deskview"
            }
        )
    }

    /// The camera's own V4L2 default for a numeric control, converted to the control's unit.
    fn number_default(&self, key: &str, min: f64, max: f64) -> Option<f64> {
        // (V4L2 name, V4L2 units per control unit): the upstream crate shows zoom in x and
        // pan/tilt in degrees, V4L2 holds 1/100 x and 1/3600 degree.
        let (name, per_unit) = match key {
            "brightness" | "contrast" | "saturation" | "sharpness" | "hue" => (key, 1.0),
            "white-balance-temperature" => ("white_balance_temperature", 1.0),
            "focus-position" => ("focus_absolute", 1.0),
            "zoom" => ("zoom_absolute", 100.0),
            "pan" => ("pan_absolute", 3600.0),
            "tilt" => ("tilt_absolute", 3600.0),
            _ => return None,
        };
        scaled_default(self.ranges().as_ref().ok()?.get(name)?, min, max, per_unit)
    }
}

/// `default / per_unit`, only when the V4L2 range maps onto the control's own range with that
/// scale (a mismatch means the unit assumption is wrong, so no default is better than a wrong one).
fn scaled_default(v: &V4l, min: f64, max: f64, per_unit: f64) -> Option<f64> {
    let (vmin, vmax, vdef) = (f64::from(v.min?), f64::from(v.max?), f64::from(v.default?));
    let tol = 1e-6 * per_unit.max(1.0);
    let fits = ((vmax - vmin) - (max - min) * per_unit).abs() <= tol * (max - min).abs().max(1.0)
        && (vmin - min * per_unit).abs() <= tol * min.abs().max(1.0);
    (fits && (vmin..=vmax).contains(&vdef)).then(|| vdef / per_unit)
}

/// Reads a `State` JSON file (profile or exported), rejecting unknown fields and huge files.
pub fn read_state_file(path: &std::path::Path) -> Result<State> {
    const MAX: u64 = 1 << 20;
    let mut bytes = Vec::new();
    fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(MAX + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() as u64 > MAX {
        bail!("{} is larger than 1 MiB", path.display());
    }
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as camera state", path.display()))
}

fn preset_info(slot: u8, p: Preset) -> PresetInfo {
    PresetInfo {
        slot,
        name: p.name,
        mode: p.mode.to_string(),
        pan: p.pan,
        tilt: p.tilt,
        zoom: p.zoom,
    }
}

const KNOWN_KEYS: &[&str] = &[
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "hue",
    "white-balance",
    "white-balance-temperature",
    "focus",
    "focus-position",
    "power-line",
    "exposure",
    "exposure-compensation",
    "exposure-iso",
    "exposure-shutter",
    "hdr",
    "zoom",
    "pan",
    "tilt",
    "mode",
    "privacy",
    "horizontal-flip",
    "vertical-flip",
    "horizontal-correction",
    "framing",
    "tracking-speed",
    "ai-zoom",
    "smart-adjustment",
    "pause-areas",
    "tracking-area",
    "audio-mode",
    "noise-canceling",
    "mute-in-privacy",
    "gestures",
    "gesture-palm",
    "gesture-zoom",
    "gesture-whiteboard",
];

fn int(value: &str) -> Result<i32> {
    value.parse().map_err(|_| anyhow!("expected an integer"))
}

fn float(value: &str) -> Result<f64> {
    match value.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => bail!("expected a finite number"),
    }
}

fn boolean(value: &str) -> Result<bool> {
    match value {
        "true" | "on" | "1" => Ok(true),
        "false" | "off" | "0" => Ok(false),
        _ => bail!("expected `true` or `false`"),
    }
}

// ---- profiles -------------------------------------------------------------------------------

/// 1-64 characters of `[A-Za-z0-9._-]`, not starting with `.`: no separators, no traversal, and
/// never a temp file name.
pub fn validate_profile_name(name: &str) -> Result<()> {
    let ok = (1..=64).contains(&name.len())
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        bail!(
            "invalid profile name `{name}`: use 1-64 of A-Z a-z 0-9 . _ - and do not start with `.`"
        )
    }
}

fn state_base() -> Result<PathBuf> {
    Ok(
        match std::env::var_os("XDG_STATE_HOME").map(PathBuf::from) {
            Some(p) if p.is_absolute() => p,
            _ => PathBuf::from(
                std::env::var_os("HOME")
                    .filter(|h| !h.is_empty())
                    .context("HOME is not set")?,
            )
            .join(".local/state"),
        },
    )
}

fn profile_dir() -> Result<PathBuf> {
    Ok(state_base()?.join("link-camera/profiles"))
}

fn profile_path(name: &str) -> Result<PathBuf> {
    validate_profile_name(name)?;
    Ok(profile_dir()?.join(format!("{name}.json")))
}

pub fn profiles() -> Result<Vec<String>> {
    let dir = profile_dir()?;
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("listing {}", dir.display())),
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|f| f.strip_suffix(".json").map(str::to_owned))
        .filter(|n| validate_profile_name(n).is_ok())
        .collect();
    names.sort();
    Ok(names)
}

pub fn delete_profile(name: &str) -> Result<()> {
    let path = profile_path(name)?;
    fs::remove_file(&path)
        .with_context(|| format!("deleting profile `{name}` ({})", path.display()))
}

// ---- automatic per-camera persistence ------------------------------------------------------
//
// Every successful user mutation saves what the camera then reports, under a key made from the
// cached sysfs USB identity (never `/dev/videoN`). Values a read could not return keep their
// previously saved value. Restoring bypasses all of this.

const SAVED_VERSION: u32 = 1;
const SAVED_MAX_BYTES: u64 = 64 << 10;
const READBACK_TTL: Duration = Duration::from_secs(2);
#[cfg(not(test))]
const LOCK_WAIT: Duration = Duration::from_secs(10);
#[cfg(test)]
const LOCK_WAIT: Duration = Duration::from_millis(100);
const SYS_VIDEO: &str = "/sys/class/video4linux";
/// Groups of `State` merged leaf by leaf; every other key (and every leaf) is replaced whole.
const STATE_GROUPS: [&str; 5] = ["gimbal", "tracking", "image", "features", "gestures"];
/// Readback tolerance when deciding that a saved number is already in effect.
const SAME_EPSILON: f64 = 0.005;

/// What [`CameraBackend::restore_saved_settings`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RestoreStatus {
    /// Nothing is saved for this camera.
    Missing,
    /// The whole saved state is in effect (applied now or already matching), streaming.
    Applied,
    /// Not streaming: only standard picture controls were applied; the rest waits for a stream.
    WaitingForStream,
}

/// A change reached the camera but could not be saved. Never retry the hardware write.
#[derive(Debug)]
pub struct AppliedNotSaved(anyhow::Error);

impl std::fmt::Display for AppliedNotSaved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("applied to the camera but not saved")
    }
}

/// The cause stays in the chain (use `{:#}`), so `is_transport_failure` still sees a lost USB link.
impl std::error::Error for AppliedNotSaved {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// True when `error` means the camera accepted the change but saving failed.
pub fn applied_but_not_saved(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<AppliedNotSaved>())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    version: u32,
    identity: String,
    #[serde(default)]
    state: State,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    privacy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pause_areas_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pause_areas: Option<Vec<PauseAreaInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tracking_area: Option<TrackingAreaInfo>,
}

impl Saved {
    fn new(identity: &str) -> Self {
        Self {
            version: SAVED_VERSION,
            identity: identity.to_owned(),
            state: State::default(),
            privacy: None,
            pause_areas_enabled: None,
            pause_areas: None,
            tracking_area: None,
        }
    }
}

fn pause_areas_link(areas: &[PauseAreaInfo]) -> Vec<link::PauseArea> {
    areas
        .iter()
        .map(|a| link::PauseArea {
            active: a.active,
            pan: a.pan,
            tilt: a.tilt,
            x: a.x,
            y: a.y,
            width: a.width,
            height: a.height,
        })
        .collect()
}

fn tracking_info(a: &TrackingArea) -> TrackingAreaInfo {
    TrackingAreaInfo {
        enabled: a.enabled,
        pan: [*a.pan.start(), *a.pan.end()],
        tilt: [*a.tilt.start(), *a.tilt.end()],
    }
}

fn tracking_link(a: &TrackingAreaInfo) -> TrackingArea {
    TrackingArea {
        enabled: a.enabled,
        pan: a.pan[0]..=a.pan[1],
        tilt: a.tilt[0]..=a.tilt[1],
    }
}

/// The USB device directory behind `/dev/videoN`, from the kernel's cached sysfs tree.
fn usb_device_dir(class: &Path, node: &str) -> Result<PathBuf> {
    let name = Path::new(node)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| {
            n.strip_prefix("video")
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        })
        .ok_or_else(|| anyhow!("{node} is not a /dev/videoN node"))?;
    let mut dir = fs::canonicalize(class.join(name).join("device"))
        .with_context(|| format!("resolving the USB device of {node}"))?;
    for _ in 0..6 {
        if dir.join("idVendor").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    bail!("{node} is not behind a USB device")
}

fn sysfs_attr(dir: &Path, name: &str) -> Option<String> {
    let mut text = String::new();
    fs::File::open(dir.join(name))
        .ok()?
        .take(256)
        .read_to_string(&mut text)
        .ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Stable key of a physical camera: vendor/product plus USB serial, else the USB port path.
/// A camera without a serial string is only recognised while it stays in the same port.
fn identity_in(class: &Path, node: &str) -> Result<String> {
    let dir = usb_device_dir(class, node)?;
    let attr = |name: &str| {
        sysfs_attr(&dir, name).with_context(|| format!("{} has no {name}", dir.display()))
    };
    let (vid, pid) = (attr("idVendor")?, attr("idProduct")?);
    Ok(match sysfs_attr(&dir, "serial") {
        Some(serial) => format!("usb:{vid}:{pid}:serial:{serial}"),
        None => {
            let port = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            format!("usb:{vid}:{pid}:port:{port}")
        }
    })
}

fn instance_in(class: &Path, node: &str) -> Result<String> {
    let dir = usb_device_dir(class, node)?;
    let name = Path::new(node).file_name().unwrap_or_default();
    let attr = |dir: &Path, name: &str| {
        sysfs_attr(dir, name).with_context(|| format!("{} has no {name}", dir.display()))
    };
    Ok(format!(
        "{}:bus{}:dev{}:node{}",
        identity_in(class, node)?,
        attr(&dir, "busnum")?,
        attr(&dir, "devnum")?,
        attr(&class.join(name), "dev")?
    ))
}

fn identity_of(node: &str) -> Result<String> {
    identity_in(Path::new(SYS_VIDEO), node)
}

/// Identity of one USB enumeration of the camera behind `device`: it changes when the camera is
/// unplugged and plugged back in (the kernel assigns a new device number; it can repeat only after
/// the 127-number bus space wraps). Read from cached sysfs, no camera I/O.
pub fn physical_instance(device: &str) -> Result<String> {
    instance_in(Path::new(SYS_VIDEO), device)
}

/// The physical USB device behind a camera node, as the kernel reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsbDevice {
    /// Kernel device path without the `/sys` prefix (`/devices/pci…/usb5/5-2/5-2.4/5-2.4.1`);
    /// every interface of the camera, including its microphone, lives below it.
    pub sysfs: String,
    pub vendor: u16,
    pub product: u16,
}

/// Identifies the USB device of `/dev/videoN` from cached sysfs; no camera I/O.
pub fn usb_device(node: &str) -> Result<UsbDevice> {
    let class = Path::new(SYS_VIDEO);
    let dir = usb_device_dir(class, node)?;
    let id = |name: &str| {
        sysfs_attr(&dir, name)
            .and_then(|v| u16::from_str_radix(&v, 16).ok())
            .with_context(|| format!("{} has no valid {name}", dir.display()))
    };
    let sysfs = dir
        .to_str()
        .and_then(|p| p.strip_prefix("/sys"))
        .ok_or_else(|| anyhow!("{} is not under /sys", dir.display()))?;
    Ok(UsbDevice {
        sysfs: sysfs.to_owned(),
        vendor: id("idVendor")?,
        product: id("idProduct")?,
    })
}

fn cameras_dir() -> Result<PathBuf> {
    Ok(state_base()?.join("link-camera/cameras"))
}

fn saved_path(dir: &Path, identity: &str) -> PathBuf {
    let hash = format!("{:x}", Sha256::digest(identity.as_bytes()));
    dir.join(format!("{}.json", &hash[..32]))
}

/// Whether automatic settings are saved for the camera behind `device` (a `/dev/videoN` node).
/// Reads no camera state; a corrupt saved file is an error, not "absent".
pub fn has_saved_settings(device: &str) -> Result<bool> {
    let identity = identity_of(device)?;
    Ok(load_saved(&saved_path(&cameras_dir()?, &identity), &identity)?.is_some())
}

fn load_saved(path: &Path, identity: &str) -> Result<Option<Saved>> {
    let mut bytes = Vec::new();
    match fs::File::open(path) {
        Ok(file) => {
            file.take(SAVED_MAX_BYTES + 1)
                .read_to_end(&mut bytes)
                .with_context(|| format!("reading {}", path.display()))?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
    }
    if bytes.len() as u64 > SAVED_MAX_BYTES {
        bail!(
            "{} is larger than 64 KiB; repair or delete it",
            path.display()
        );
    }
    let saved: Saved = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is not valid saved camera settings; repair or delete it",
            path.display()
        )
    })?;
    if saved.version != SAVED_VERSION {
        bail!(
            "{} has unsupported version {}",
            path.display(),
            saved.version
        );
    }
    if saved.identity != identity {
        bail!(
            "{} belongs to another camera ({})",
            path.display(),
            saved.identity
        );
    }
    if saved
        .pause_areas
        .as_ref()
        .is_some_and(|z| z.len() > link::PAUSE_AREA_SLOTS as usize)
    {
        bail!("{} lists too many pause-track zones", path.display());
    }
    Ok(Some(saved))
}

/// Exclusive per-camera lock. It is held across the hardware write, the readback and the save
/// of one user change, and across a whole automatic restore, so concurrent GUI/CLI/daemon
/// operations neither corrupt nor overwrite each other's settings. Dropping it releases it.
struct SavedLock {
    _file: fs::File,
    path: PathBuf,
    identity: String,
}

fn lock_saved(dir: &Path, identity: &str) -> Result<SavedLock> {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    let path = saved_path(dir, identity);
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(path.with_extension("lock"))
        .with_context(|| format!("opening the lock for {}", path.display()))?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(fs::TryLockError::WouldBlock) if start.elapsed() < LOCK_WAIT => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(fs::TryLockError::WouldBlock) => {
                bail!(
                    "another link-camera process is changing this camera's settings; nothing was written"
                )
            }
            Err(fs::TryLockError::Error(e)) => return Err(e).context("locking saved settings"),
        }
    }
    Ok(SavedLock {
        _file: file,
        path,
        identity: identity.to_owned(),
    })
}

impl SavedLock {
    /// A corrupt file is an error, never "absent".
    fn load(&self) -> Result<Option<Saved>> {
        load_saved(&self.path, &self.identity)
    }

    /// Private atomic replace (temp file, fsync, rename).
    fn store(&self, saved: &Saved) -> Result<()> {
        let mut json = serde_json::to_vec_pretty(saved)?;
        json.push(b'\n');
        let tmp = self
            .path
            .with_extension(format!("{}.tmp", std::process::id()));
        let _ = fs::remove_file(&tmp);
        let written = (|| -> std::io::Result<()> {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            file.write_all(&json)?;
            file.sync_all()?;
            fs::rename(&tmp, &self.path)
        })();
        if let Err(error) = written {
            let _ = fs::remove_file(&tmp);
            return Err(error).with_context(|| format!("writing {}", self.path.display()));
        }
        Ok(())
    }
}

/// Overlays `new` on `old`: leaves the camera did not report keep their old value.
fn merge_state(old: &State, new: &State) -> Result<State> {
    let mut merged = serde_json::to_value(old)?;
    let (Some(base), Value::Object(new)) = (merged.as_object_mut(), serde_json::to_value(new)?)
    else {
        bail!("camera state is not an object");
    };
    for (key, value) in new {
        match (STATE_GROUPS.contains(&key.as_str()), value) {
            (true, Value::Object(group)) => {
                match base.entry(key).or_insert_with(|| Value::Object(Map::new())) {
                    Value::Object(old) => old.extend(group),
                    other => *other = Value::Object(group),
                }
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
    Ok(serde_json::from_value(merged)?)
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() < SAME_EPSILON,
            _ => x == y,
        },
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w))
        }
        _ => a == b,
    }
}

fn put(out: &mut Map<String, Value>, group: &str, leaf: &str, value: &Value) {
    if let Value::Object(g) = out
        .entry(group.to_owned())
        .or_insert_with(|| Value::Object(Map::new()))
    {
        g.insert(leaf.to_owned(), value.clone());
    }
}

/// The part of `saved` not already in effect in `current`. Settings the SDK couples are kept
/// together: a mode change re-applies pan/tilt/zoom/framing, a zoom change re-applies framing
/// (manual zoom switches framing to manual), a pan or tilt change re-applies both axes. A mode
/// the camera reports as busy is treated as in effect only for a saved Tracking (waiting for a
/// subject); any other saved mode must be written, which also leaves a stuck tracking request.
/// Tracking and Auto Framing steer the view, so no saved pan/tilt/zoom is replayed over them.
fn differs(saved: &State, current: &State, mode_busy: bool) -> Result<State> {
    let Value::Object(want) = serde_json::to_value(saved)? else {
        bail!("camera state is not an object");
    };
    let have = serde_json::to_value(current)?;
    let mut out = Map::new();
    let driven = camera_driven(saved.mode);
    for (key, value) in &want {
        if driven && key == "gimbal" {
            continue;
        }
        match value {
            Value::Object(leaves) if STATE_GROUPS.contains(&key.as_str()) => {
                for (leaf, v) in leaves {
                    if !have
                        .get(key)
                        .and_then(|g| g.get(leaf))
                        .is_some_and(|c| same(c, v))
                    {
                        put(&mut out, key, leaf, v);
                    }
                }
            }
            _ => {
                let in_effect =
                    (key == "mode" && mode_busy && saved.mode == Some(CameraMode::Tracking))
                        || have.get(key).is_some_and(|c| same(c, value));
                if !in_effect {
                    out.insert(key.clone(), value.clone());
                }
            }
        }
    }
    let changed = |g: &str, l: &str| out.get(g).and_then(|x| x.get(l)).is_some();
    let (mode, zoom) = (out.contains_key("mode"), changed("gimbal", "zoom"));
    let moved = changed("gimbal", "pan") || changed("gimbal", "tilt");
    let mut coupled: Vec<(&str, &str)> = Vec::new();
    if mode && !driven {
        coupled.extend([("gimbal", "pan"), ("gimbal", "tilt"), ("gimbal", "zoom")]);
    }
    if mode || zoom {
        coupled.push(("tracking", "framing"));
    }
    if moved {
        coupled.extend([("gimbal", "pan"), ("gimbal", "tilt")]);
    }
    for (g, l) in coupled {
        if let Some(v) = want.get(g).and_then(|x| x.get(l)) {
            put(&mut out, g, l, v);
        }
    }
    Ok(serde_json::from_value(Value::Object(out))?)
}

/// Readback entries that must be present before a change is saved, so a changed setting the
/// camera cannot read back is never reported saved with its stale old value. Entries are
/// `mode`/`audio_mode`/`group.leaf` paths of `State`, or `!privacy`, `!pause_areas_enabled`,
/// `!tracking_area` and `!pause_zones` (the zone layout, read separately).
fn expected_after_set(key: &str) -> Vec<String> {
    let paths: &[&str] = match key {
        "brightness" | "contrast" | "saturation" | "sharpness" | "hue" => {
            return vec![format!("image.{key}")];
        }
        "white-balance" | "white-balance-temperature" => &["image.white_balance"],
        "focus" | "focus-position" => &["image.focus"],
        "power-line" => &["image.power_line"],
        "exposure" | "exposure-compensation" | "exposure-iso" | "exposure-shutter" => {
            &["image.exposure"]
        }
        "zoom" => &["gimbal.zoom"],
        "pan" | "tilt" => &["gimbal.pan", "gimbal.tilt"],
        "mode" => &["mode"],
        "framing" => &["tracking.framing"],
        "tracking-speed" => &["tracking.speed"],
        "audio-mode" => &["audio_mode"],
        "privacy" => &["!privacy"],
        "pause-areas" => &["!pause_areas_enabled"],
        "tracking-area" => &["!tracking_area"],
        "gestures" => &["features.gestures"],
        "gesture-palm" => &["gestures.tracking"],
        "gesture-zoom" => &["gestures.zoom"],
        "gesture-whiteboard" => &["gestures.whiteboard"],
        _ => return vec![format!("features.{}", key.replace('-', "_"))],
    };
    paths.iter().map(|p| (*p).to_owned()).collect()
}

/// Every setting an applied `State` sets, as readback paths.
fn applied_paths(state: &State) -> Result<Vec<String>> {
    let Value::Object(map) = serde_json::to_value(state)? else {
        bail!("camera state is not an object");
    };
    let mut out = Vec::new();
    for (key, value) in map {
        match value {
            Value::Object(group) if STATE_GROUPS.contains(&key.as_str()) => {
                out.extend(group.keys().map(|leaf| format!("{key}.{leaf}")));
            }
            _ => out.push(key),
        }
    }
    Ok(out)
}

fn readable(r: &Reading, state: &Value, path: &str) -> bool {
    match path.strip_prefix('!') {
        Some("privacy") => r.privacy.is_some(),
        Some("pause_areas_enabled") => r.pause_areas.is_some(),
        Some("tracking_area") => r.tracking_area.is_some(),
        Some("mode_tracking") => r.mode_busy || r.state.mode == Some(CameraMode::Tracking),
        Some("pause_zones") => true,
        Some(_) => false,
        None => path.split('.').try_fold(state, |v, k| v.get(k)).is_some(),
    }
}

impl CameraBackend {
    /// Whether some program is streaming video from the camera. Errors, including an
    /// unsupported query, are reported as they are.
    pub fn is_streaming(&self) -> Result<bool> {
        self.camera
            .is_streaming()
            .context("reading the streaming state")
    }

    /// One user change as one operation under the per-camera lock: lock, load the saved file
    /// (a corrupt one refuses the change), run the hardware write `op`, read back, verify and
    /// save. Errors before `op` (identity, lock, corrupt file) are ordinary and nothing was
    /// written, so retrying is safe. An `op` error is ordinary too. After `op` succeeded any
    /// readback or save failure is [`AppliedNotSaved`] with its cause kept: never replay `op`.
    /// Only the public mutators call this, never each other, so one operation is one commit.
    fn persisted<T>(&self, expect: &[String], op: impl FnOnce() -> Result<T>) -> Result<T> {
        let identity = identity_of(self.id())?;
        self.persisted_in(&cameras_dir()?, &identity, expect, op)
    }

    fn persisted_in<T>(
        &self,
        dir: &Path,
        identity: &str,
        expect: &[String],
        op: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let lock = lock_saved(dir, identity)?;
        let old = lock.load()?;
        let value = op()?;
        self.save_readback(&lock, old, expect)
            .map_err(|e| anyhow::Error::new(AppliedNotSaved(e)))?;
        Ok(value)
    }

    /// Saves a fresh readback merged over `old`, under `lock`. Every `expect` entry must be
    /// present in the readback; otherwise nothing is written (the old file stays) and the error
    /// says which setting could not be verified. `!pause_zones` also refreshes the zone layout.
    fn save_readback(&self, lock: &SavedLock, old: Option<Saved>, expect: &[String]) -> Result<()> {
        let mut saved = old.unwrap_or_else(|| Saved::new(&lock.identity));
        let r = self.read(true)?;
        let have = serde_json::to_value(r.state)?;
        let missing: Vec<&str> = expect
            .iter()
            .map(String::as_str)
            .filter(|p| !readable(&r, &have, p))
            .map(|p| p.trim_start_matches('!'))
            .collect();
        if !missing.is_empty() {
            let why = if r.failures.is_empty() {
                String::new()
            } else {
                format!(" ({})", r.failures.join("; "))
            };
            bail!(
                "the camera did not report {} after the change{why}; the previously saved value was kept",
                missing.join(", ")
            );
        }
        saved.state = merge_state(&saved.state, &r.state)?;
        if r.mode_busy && expect.iter().any(|p| p == "!mode_tracking") {
            saved.state.mode = Some(CameraMode::Tracking);
        }
        saved.privacy = r.privacy.or(saved.privacy);
        saved.pause_areas_enabled = r.pause_areas.or(saved.pause_areas_enabled);
        if let Some(area) = &r.tracking_area {
            saved.tracking_area = Some(tracking_info(area));
        }
        if expect.iter().any(|p| p == "!pause_zones") {
            saved.pause_areas = Some(
                self.pause_areas_raw()
                    .map_err(|e| anyhow::Error::from(e).context("reading pause-track zones"))?,
            );
        }
        lock.store(&saved)?;
        *self.readback.borrow_mut() = Some((Instant::now(), r));
        Ok(())
    }

    /// Re-applies the settings saved for this physical camera, holding the same per-camera lock
    /// as user changes so it can never overwrite a value changed meanwhile. Never saves, never
    /// starts or owns capture, never writes privacy (reopening privacy needs an explicit user
    /// action). Writes only what differs from what the camera reports. Not streaming, or
    /// streaming state unknown or unreadable: standard picture controls only, `WaitingForStream`.
    /// Streaming: the whole saved state through the typed SDK (mode, then zoom, then framing,
    /// position last), then zones and tracking area. A failure to get the lock is an error with
    /// nothing written; a hardware error is returned as is and must not be retried.
    pub fn restore_saved_settings(&self) -> Result<RestoreStatus> {
        let identity = identity_of(self.id())?;
        self.restore_in(&cameras_dir()?, &identity)
    }

    fn restore_in(&self, dir: &Path, identity: &str) -> Result<RestoreStatus> {
        let exists = saved_path(dir, identity)
            .try_exists()
            .with_context(|| format!("checking {}", dir.display()))?;
        if !exists {
            return Ok(RestoreStatus::Missing);
        }
        let lock = lock_saved(dir, identity)?;
        match lock.load()? {
            Some(saved) => self.restore(&saved),
            None => Ok(RestoreStatus::Missing),
        }
    }

    fn restore(&self, saved: &Saved) -> Result<RestoreStatus> {
        let r = self.read(true)?;
        if r.streaming != Some(true) {
            let want = State {
                image: ImageSettings {
                    exposure: None,
                    ..saved.state.image
                },
                ..State::default()
            };
            let todo = differs(&want, &r.state, false)?.image;
            if todo != ImageSettings::default() {
                self.camera
                    .set_image(&todo)
                    .context("restoring picture controls")?;
            }
            return Ok(RestoreStatus::WaitingForStream);
        }
        let mut want = saved.state;
        if want.mode.is_some_and(|m| !self.mode_allowed(m)) {
            // e.g. a `track` saved for a camera without a gimbal: it cannot be applied here.
            want.mode = None;
        }
        let todo = differs(&want, &r.state, r.mode_busy)?;
        if todo != State::default() {
            self.camera
                .apply(&todo)
                .context("restoring saved camera state")?;
        }
        let mut zones_written = false;
        if let Some(want) = &saved.pause_areas {
            match self.pause_areas_raw() {
                Ok(have) => {
                    if !same(&serde_json::to_value(&have)?, &serde_json::to_value(want)?) {
                        self.camera
                            .set_pause_areas(&pause_areas_link(want))
                            .context("restoring pause-track zones")?;
                        zones_written = true;
                    }
                }
                Err(link::Error::Unsupported) => {}
                Err(e) => return Err(anyhow::Error::from(e).context("reading pause-track zones")),
            }
        }
        if let (Some(on), Some(have)) = (saved.pause_areas_enabled, r.pause_areas) {
            if on != have || zones_written {
                self.camera
                    .set_pause_areas_enabled(on)
                    .context("restoring pause-track areas")?;
            }
        }
        if let (Some(want), Some(have)) = (&saved.tracking_area, &r.tracking_area) {
            if !same(
                &serde_json::to_value(want)?,
                &serde_json::to_value(tracking_info(have))?,
            ) {
                self.camera
                    .set_tracking_area(&tracking_link(want))
                    .context("restoring the tracking area")?;
            }
        }
        Ok(RestoreStatus::Applied)
    }
}

// ---- V4L2 control ranges (read-only `v4l2-ctl --list-ctrls`) --------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
struct V4l {
    min: Option<i32>,
    max: Option<i32>,
    step: Option<i32>,
    value: Option<i32>,
    default: Option<i32>,
}

fn v4l_list(device: &str) -> Result<HashMap<String, V4l>> {
    let out = Command::new("v4l2-ctl")
        .args(["-d", device, "--list-ctrls"])
        .stdin(Stdio::null())
        .output()
        .context("running v4l2-ctl")?;
    if !out.status.success() {
        bail!(
            "v4l2-ctl failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(parse_v4l(&String::from_utf8_lossy(&out.stdout)))
}

fn v4l_value(device: &str, name: &str) -> Result<i32> {
    v4l_list(device)?
        .get(name)
        .and_then(|c| c.value)
        .ok_or_else(|| anyhow!("v4l2-ctl does not report {name}"))
}

/// Lines look like `name 0x00980900 (int) : min=0 max=100 step=1 default=50 value=50 flags=...`.
fn parse_v4l(text: &str) -> HashMap<String, V4l> {
    text.lines()
        .filter_map(|line| {
            let (head, tail) = line.split_once(" : ")?;
            let name = head.split_whitespace().next()?;
            let field = |key: &str| {
                tail.split_whitespace()
                    .find_map(|t| t.strip_prefix(key)?.strip_prefix('=')?.parse().ok())
            };
            Some((
                name.to_owned(),
                V4l {
                    min: field("min"),
                    max: field("max"),
                    step: field("step"),
                    value: field("value"),
                    default: field("default"),
                },
            ))
        })
        .collect()
}

fn v4l_image(map: &HashMap<String, V4l>) -> ImageSettings {
    let get = |name: &str| map.get(name).and_then(|c| c.value);
    ImageSettings {
        brightness: get("brightness"),
        contrast: get("contrast"),
        saturation: get("saturation"),
        sharpness: get("sharpness"),
        hue: get("hue"),
        white_balance: match get("white_balance_automatic") {
            Some(0) => get("white_balance_temperature").map(|t| WhiteBalance::Manual {
                temperature: t.max(0) as u32,
            }),
            Some(_) => Some(WhiteBalance::Auto),
            None => None,
        },
        focus: match get("focus_automatic_continuous") {
            Some(0) => get("focus_absolute").map(|p| Focus::Manual {
                position: p.max(0) as u32,
            }),
            Some(_) => Some(Focus::Auto),
            None => None,
        },
        power_line: get("power_line_frequency").and_then(|v| match v {
            0 => Some(PowerLine::Off),
            1 => Some(PowerLine::Hz50),
            2 => Some(PowerLine::Hz60),
            3 => Some(PowerLine::Auto),
            _ => None,
        }),
        exposure: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailedTransport(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl link::transport::Transport for FailedTransport {
        fn xu_query(
            &self,
            _: u8,
            _: u8,
            _: link::transport::XuQuery,
            _: &mut [u8],
        ) -> std::io::Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(std::io::Error::from_raw_os_error(110))
        }
        fn query_control(
            &self,
            _: link::transport::Control,
        ) -> std::io::Result<Option<link::transport::ControlInfo>> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(std::io::Error::from_raw_os_error(110))
        }
        fn get_controls(&self, _: &mut [(link::transport::Control, i32)]) -> std::io::Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(std::io::Error::from_raw_os_error(110))
        }
        fn set_controls(&self, _: &[(link::transport::Control, i32)]) -> std::io::Result<()> {
            panic!("reading a snapshot must never write camera settings")
        }
    }

    #[test]
    fn snapshot_stops_after_first_usb_transport_timeout() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let kind = link::xu::UnitKind::Control;
        let info = link::DeviceInfo::new(
            "/camera-timeout-regression",
            0x4c03,
            vec![link::xu::ExtensionUnit {
                id: 9,
                guid: kind.guid(),
                kind: Some(kind),
                controls: vec![0xff; 8],
            }],
        );
        let camera = CameraBackend::new(link::Camera::with_transport(
            info,
            FailedTransport(calls.clone()),
        ));
        assert!(
            camera.snapshot().is_err(),
            "a failed device transport must not produce a successful snapshot"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "later getters must not multiply the USB timeout"
        );
    }

    #[test]
    fn optional_selector_stalls_are_not_device_transport_loss() {
        assert!(!transport_lost(&link::Error::Unsupported));
        let selector = link::Error::Io {
            context: "optional selector".into(),
            source: std::io::Error::from_raw_os_error(32),
        };
        assert!(
            !transport_lost(&selector),
            "EPIPE can reject one selector on a healthy device"
        );
        let device = link::Error::Io {
            context: "device access".into(),
            source: std::io::Error::from_raw_os_error(5),
        };
        assert!(
            transport_lost(&device),
            "EIO must stop the remaining control scan"
        );
    }

    #[test]
    fn profile_names_reject_traversal_and_temp_files() {
        for bad in [
            "",
            ".",
            "..",
            "../x",
            "a/b",
            ".hidden",
            "a b",
            "é",
            &"x".repeat(65),
            "a\0b",
        ] {
            assert!(validate_profile_name(bad).is_err(), "{bad:?}");
        }
        for good in ["desk", "Meeting-2.json", "a_b.c", &"x".repeat(64)] {
            assert!(validate_profile_name(good).is_ok(), "{good:?}");
        }
    }

    #[test]
    fn v4l_listing_keeps_ranges_and_out_of_menu_values() {
        let map = parse_v4l(
            "User Controls\n\n  hue 0x00980903 (int)    : min=-15 max=15 step=1 default=0 value=-3 flags=has-min-max\n\
             power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=3 value=3 (Auto)\n\
             white_balance_automatic 0x0098090c (bool)   : default=1 value=1\n",
        );
        assert_eq!(
            map["hue"],
            V4l {
                min: Some(-15),
                max: Some(15),
                step: Some(1),
                value: Some(-3),
                default: Some(0)
            }
        );
        assert_eq!(v4l_image(&map).power_line, Some(PowerLine::Auto));
        assert_eq!(map["white_balance_automatic"].min, None);
        assert_eq!(v4l_image(&map).white_balance, Some(WhiteBalance::Auto));
    }

    #[test]
    fn toggles_and_numbers_are_strict() {
        assert!(boolean("yes").is_err());
        assert!(boolean("ON").is_err());
        assert!(float("NaN").is_err() && float("inf").is_err());
        assert_eq!(int("-5").unwrap(), -5);
        assert!(int("5.5").is_err());
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("link-camera-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A sysfs tree where class/video0 and class/video5 sit behind USB device `port`.
    fn fake_sysfs(root: &Path, serial: Option<&str>, port: &str, devnum: &str) -> PathBuf {
        let usb = root.join("devices/usb1").join(port);
        let iface = usb.join(format!("{port}:1.0"));
        fs::create_dir_all(&iface).unwrap();
        for (name, value) in [
            ("idVendor", "2e1a"),
            ("idProduct", "4c03"),
            ("busnum", "1"),
            ("devnum", devnum),
        ] {
            fs::write(usb.join(name), format!("{value}\n")).unwrap();
        }
        if let Some(serial) = serial {
            fs::write(usb.join("serial"), format!("{serial}\n")).unwrap();
        }
        for (node, dev) in [("video0", "81:0"), ("video5", "81:5")] {
            let dir = root.join("class").join(node);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("dev"), format!("{dev}\n")).unwrap();
            std::os::unix::fs::symlink(&iface, dir.join("device")).unwrap();
        }
        root.join("class")
    }

    #[test]
    fn identity_ignores_node_numbers_and_prefers_the_serial() {
        let root = scratch("sysfs");
        let class = fake_sysfs(&root, Some("SN1"), "1-2.3", "7");
        let id = identity_in(&class, "/dev/video0").unwrap();
        assert_eq!(id, "usb:2e1a:4c03:serial:SN1");
        assert_eq!(identity_in(&class, "/dev/video5").unwrap(), id);
        let before = instance_in(&class, "/dev/video0").unwrap();
        fs::write(root.join("devices/usb1/1-2.3/devnum"), "8\n").unwrap();
        assert_ne!(
            instance_in(&class, "/dev/video0").unwrap(),
            before,
            "a reconnect gets a new instance"
        );
        assert_eq!(identity_in(&class, "/dev/video0").unwrap(), id);
        fs::remove_file(root.join("devices/usb1/1-2.3/serial")).unwrap();
        assert_eq!(
            identity_in(&class, "/dev/video0").unwrap(),
            "usb:2e1a:4c03:port:1-2.3"
        );
        for bad in ["/dev/video", "/dev/videoX", "/dev/null", "/dev/video9"] {
            assert!(identity_in(&class, bad).is_err(), "{bad}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merging_keeps_values_the_camera_did_not_report() {
        let old = State {
            mode: Some(CameraMode::Tracking),
            gimbal: link::GimbalState {
                zoom: Some(2.0),
                ..Default::default()
            },
            image: ImageSettings {
                brightness: Some(10),
                white_balance: Some(WhiteBalance::Manual { temperature: 5000 }),
                ..Default::default()
            },
            ..Default::default()
        };
        let new = State {
            image: ImageSettings {
                brightness: Some(20),
                white_balance: Some(WhiteBalance::Auto),
                ..Default::default()
            },
            ..Default::default()
        };
        let merged = merge_state(&old, &new).unwrap();
        assert_eq!(merged.mode, Some(CameraMode::Tracking));
        assert_eq!(merged.gimbal.zoom, Some(2.0));
        assert_eq!(merged.image.brightness, Some(20));
        assert_eq!(
            merged.image.white_balance,
            Some(WhiteBalance::Auto),
            "enum values are replaced whole"
        );
        assert_eq!(merge_state(&old, &State::default()).unwrap(), old);
    }

    #[test]
    fn restore_diff_keeps_coupled_settings_together() {
        let saved = State {
            mode: Some(CameraMode::Normal),
            gimbal: link::GimbalState {
                pan: Some(5.0),
                tilt: Some(0.0),
                zoom: Some(2.0),
            },
            tracking: link::TrackingState {
                framing: Some(Framing::Head),
                ..Default::default()
            },
            image: ImageSettings {
                brightness: Some(70),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(differs(&saved, &saved, false).unwrap(), State::default());
        let mut current = saved;
        current.gimbal.zoom = Some(1.0);
        let todo = differs(&saved, &current, false).unwrap();
        assert_eq!(todo.gimbal.zoom, Some(2.0));
        assert_eq!(
            todo.tracking.framing,
            Some(Framing::Head),
            "manual zoom flips framing to manual"
        );
        assert_eq!(
            (todo.mode, todo.gimbal.pan, todo.image),
            (None, None, ImageSettings::default())
        );
        let mut current = saved;
        current.mode = Some(CameraMode::Tracking);
        let todo = differs(&saved, &current, false).unwrap();
        assert_eq!(todo.mode, Some(CameraMode::Normal));
        assert_eq!((todo.gimbal.pan, todo.gimbal.zoom), (Some(5.0), Some(2.0)));
        current.mode = None;
        assert_eq!(
            differs(&saved, &current, true).unwrap().mode,
            Some(CameraMode::Normal),
            "a busy camera is not in normal mode: leave the tracking request"
        );
        let mut tracking = saved;
        tracking.mode = Some(CameraMode::Tracking);
        assert_eq!(
            differs(&tracking, &current, true).unwrap().mode,
            None,
            "tracking waiting for a subject is in effect"
        );
    }

    #[test]
    fn restore_never_replays_a_view_over_camera_driven_modes() {
        let mut saved = State {
            mode: Some(CameraMode::AutoFraming),
            gimbal: link::GimbalState {
                pan: Some(5.0),
                tilt: Some(0.0),
                zoom: Some(2.0),
            },
            tracking: link::TrackingState {
                framing: Some(Framing::Head),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut current = saved;
        current.mode = Some(CameraMode::Normal);
        current.gimbal.zoom = Some(1.0);
        let todo = differs(&saved, &current, false).unwrap();
        assert_eq!(todo.mode, Some(CameraMode::AutoFraming));
        assert_eq!(todo.gimbal, link::GimbalState::default());
        assert_eq!(todo.tracking.framing, Some(Framing::Head));
        current.mode = saved.mode;
        assert_eq!(
            differs(&saved, &current, false).unwrap(),
            State::default(),
            "auto framing in effect: its live zoom is not rewritten"
        );
        saved.mode = Some(CameraMode::Normal);
        assert_eq!(
            differs(&saved, &current, false).unwrap().gimbal.zoom,
            Some(2.0)
        );
    }

    #[test]
    fn fixed_cameras_offer_auto_framing_not_gimbal_modes() {
        assert_eq!(physical_gimbal(Model::Link2C), Some(false));
        assert_eq!(physical_gimbal(Model::Unknown(1)), None);
        for key in ["track", "overhead"] {
            assert!(!mode_keys(Model::Link2C).contains(&key));
            assert!(mode_keys(Model::Link2).contains(&key));
        }
        assert!(mode_keys(Model::Link2CPro).contains(&"auto-framing"));
    }

    fn read_with(mode: Option<CameraMode>, busy: bool) -> Reading {
        let mut r = Reading::default();
        r.streaming = Some(true);
        r.state.mode = mode;
        r.mode_busy = busy;
        r
    }

    #[test]
    fn tracking_status_is_honest_per_model() {
        let status = |m, r: &Reading, saved| tracking_status_of(m, r, saved);
        let on = status(
            Model::Link2C,
            &read_with(Some(CameraMode::AutoFraming), false),
            None,
        );
        assert_eq!(
            (on.state, on.capability, on.physical_gimbal),
            (
                TrackingState::Active,
                TrackingCapability::AutoFraming,
                Some(false)
            )
        );
        assert_eq!(on.mode_key.as_deref(), Some("auto-framing"));
        assert!(on.reason.contains("no motorized gimbal"));
        let waiting = status(
            Model::Link2,
            &read_with(None, true),
            Some(CameraMode::Tracking),
        );
        assert_eq!(waiting.state, TrackingState::WaitingForSubject);
        assert_eq!(
            status(Model::Link2, &read_with(None, true), None).state,
            TrackingState::Switching
        );
        assert_eq!(
            status(
                Model::Link2C,
                &read_with(Some(CameraMode::Tracking), false),
                None
            )
            .state,
            TrackingState::OtherMode,
            "a fixed camera never claims subject tracking"
        );
        let mut idle = read_with(None, false);
        idle.streaming = Some(false);
        assert_eq!(
            status(Model::Link2C, &idle, None).state,
            TrackingState::NotStreaming
        );
        let mut private = read_with(Some(CameraMode::Normal), false);
        private.privacy = Some(true);
        assert_eq!(
            status(Model::Link2, &private, None).state,
            TrackingState::Privacy
        );
        assert_eq!(
            status(
                Model::Unknown(9),
                &read_with(Some(CameraMode::Normal), false),
                None
            )
            .capability,
            TrackingCapability::None
        );
        assert_eq!(
            status(Model::Link2C, &read_with(None, false), None).state,
            TrackingState::Unavailable
        );
    }

    #[test]
    fn numeric_defaults_need_a_matching_v4l_range() {
        let v = |min, max, default| V4l {
            min: Some(min),
            max: Some(max),
            default: Some(default),
            ..V4l::default()
        };
        assert_eq!(scaled_default(&v(0, 100, 50), 0.0, 100.0, 1.0), Some(50.0));
        assert_eq!(
            scaled_default(&v(100, 400, 100), 1.0, 4.0, 100.0),
            Some(1.0)
        );
        assert_eq!(
            scaled_default(&v(-522_000, 522_000, 0), -145.0, 145.0, 3600.0),
            Some(0.0)
        );
        assert_eq!(
            scaled_default(&v(0, 100, 50), 0.0, 255.0, 1.0),
            None,
            "range mismatch"
        );
        assert_eq!(
            scaled_default(&v(0, 100, 500), 0.0, 100.0, 1.0),
            None,
            "default outside range"
        );
        assert_eq!(scaled_default(&V4l::default(), 0.0, 100.0, 1.0), None);
    }

    #[test]
    fn saved_files_are_private_locked_strict_and_never_overwritten_when_corrupt() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("store");
        let id = "usb:1:2:serial:X";
        let path = saved_path(&dir, id);
        assert!(load_saved(&path, id).unwrap().is_none());
        update_saved(&dir, id, |old| {
            assert!(old.is_none());
            let mut saved = Saved::new(id);
            saved.privacy = Some(true);
            Ok(saved)
        })
        .unwrap();
        update_saved(&dir, id, |old| {
            let mut saved = old.unwrap();
            assert_eq!(saved.privacy, Some(true));
            saved.state.image.brightness = Some(30);
            Ok(saved)
        })
        .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            load_saved(&path, id)
                .unwrap()
                .unwrap()
                .state
                .image
                .brightness,
            Some(30)
        );
        assert!(
            load_saved(&path, "usb:other").is_err(),
            "another camera's file is rejected"
        );

        let good = fs::read(&path).unwrap();
        assert!(update_saved(&dir, id, |_| bail!("readback failed")).is_err());
        assert_eq!(
            fs::read(&path).unwrap(),
            good,
            "a failed update keeps the old file"
        );

        for bad in [
            "{not json".to_owned(),
            format!(r#"{{"version":1,"identity":"{id}","bogus":1}}"#),
            format!(r#"{{"version":9,"identity":"{id}"}}"#),
            " ".repeat(SAVED_MAX_BYTES as usize + 1),
        ] {
            fs::write(&path, &bad).unwrap();
            assert!(load_saved(&path, id).is_err());
            assert!(update_saved(&dir, id, |_| Ok(Saved::new(id))).is_err());
            assert_eq!(
                fs::read(&path).unwrap(),
                bad.as_bytes(),
                "corrupt data is never silently replaced"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[derive(Clone, Default)]
    struct Fake(std::sync::Arc<std::sync::Mutex<FakeState>>);

    #[derive(Default)]
    struct FakeState {
        brightness: i32,
        streaming: bool,
        writes: Vec<(link::transport::Control, i32)>,
        xu_writes: usize,
    }

    /// Only brightness and the device-status selector exist.
    impl link::transport::Transport for Fake {
        fn xu_query(
            &self,
            _: u8,
            _: u8,
            query: link::transport::XuQuery,
            data: &mut [u8],
        ) -> std::io::Result<()> {
            use link::transport::XuQuery;
            let mut state = self.0.lock().unwrap();
            match query {
                XuQuery::GetInfo => data[0] = 3,
                XuQuery::GetLen => data.copy_from_slice(&4u16.to_le_bytes()),
                XuQuery::GetCur => data.copy_from_slice(&[0, 0, 0, u8::from(state.streaming)]),
                XuQuery::SetCur => state.xu_writes += 1,
                _ => return Err(std::io::ErrorKind::Unsupported.into()),
            }
            Ok(())
        }
        fn query_control(
            &self,
            control: link::transport::Control,
        ) -> std::io::Result<Option<link::transport::ControlInfo>> {
            Ok((control == link::transport::Control::Brightness).then_some(
                link::transport::ControlInfo {
                    minimum: 0,
                    maximum: 100,
                    step: 1,
                    default: 50,
                },
            ))
        }
        fn get_controls(
            &self,
            controls: &mut [(link::transport::Control, i32)],
        ) -> std::io::Result<()> {
            let state = self.0.lock().unwrap();
            for (control, value) in controls {
                if *control == link::transport::Control::Brightness {
                    *value = state.brightness;
                }
            }
            Ok(())
        }
        fn set_controls(
            &self,
            controls: &[(link::transport::Control, i32)],
        ) -> std::io::Result<()> {
            let mut state = self.0.lock().unwrap();
            for &(control, value) in controls {
                if control == link::transport::Control::Brightness {
                    state.brightness = value;
                }
                state.writes.push((control, value));
            }
            Ok(())
        }
    }

    fn fake_backend(brightness: i32, streaming: bool) -> (CameraBackend, Fake) {
        let fake = Fake::default();
        {
            let mut state = fake.0.lock().unwrap();
            state.brightness = brightness;
            state.streaming = streaming;
        }
        let kind = link::xu::UnitKind::Control;
        let info = link::DeviceInfo::new(
            "/dev/video99999",
            0x4c04,
            vec![link::xu::ExtensionUnit {
                id: 9,
                guid: kind.guid(),
                kind: Some(kind),
                controls: vec![0, 0x04],
            }],
        );
        (
            CameraBackend::new(link::Camera::with_transport(info, fake.clone())),
            fake,
        )
    }

    fn saved_with(state: State) -> Saved {
        let mut saved = Saved::new("usb:test");
        saved.state = state;
        saved
    }

    #[test]
    fn idle_restore_applies_only_standard_controls() {
        let (backend, fake) = fake_backend(50, false);
        let saved = saved_with(State {
            mode: Some(CameraMode::Tracking),
            gimbal: link::GimbalState {
                zoom: Some(2.0),
                ..Default::default()
            },
            image: ImageSettings {
                brightness: Some(70),
                exposure: Some(Exposure::Auto { compensation: 1.0 }),
                ..Default::default()
            },
            ..Default::default()
        });
        assert_eq!(
            backend.restore(&saved).unwrap(),
            RestoreStatus::WaitingForStream
        );
        let state = fake.0.lock().unwrap();
        assert_eq!(state.writes, [(link::transport::Control::Brightness, 70)]);
        assert_eq!(state.xu_writes, 0, "vendor controls wait for a stream");
    }

    #[test]
    fn streaming_restore_writes_only_the_difference() {
        let saved = saved_with(State {
            image: ImageSettings {
                brightness: Some(70),
                ..Default::default()
            },
            ..Default::default()
        });
        let (backend, fake) = fake_backend(70, true);
        assert_eq!(backend.restore(&saved).unwrap(), RestoreStatus::Applied);
        assert!(
            fake.0.lock().unwrap().writes.is_empty(),
            "already in effect: no write"
        );
        let (backend, fake) = fake_backend(50, true);
        assert_eq!(backend.restore(&saved).unwrap(), RestoreStatus::Applied);
        assert_eq!(fake.0.lock().unwrap().brightness, 70);
    }

    #[test]
    fn autosave_saves_the_readback_and_keeps_what_it_could_not_read() {
        let dir = scratch("autosave");
        let id = "usb:test";
        let (backend, _fake) = fake_backend(70, false);
        update_saved(&dir, id, |_| {
            let mut saved = Saved::new(id);
            saved.state.mode = Some(CameraMode::Tracking);
            saved.state.image.brightness = Some(10);
            saved.pause_areas = Some(Vec::new());
            Ok(saved)
        })
        .unwrap();
        let lock = lock_saved(&dir, id).unwrap();
        backend
            .save_readback(
                &lock,
                lock.load().unwrap(),
                &["image.brightness".to_owned()],
            )
            .unwrap();
        drop(lock);
        let saved = load_saved(&saved_path(&dir, id), id).unwrap().unwrap();
        assert_eq!(
            saved.state.image.brightness,
            Some(70),
            "the readback replaces the old value"
        );
        assert_eq!(
            saved.state.mode,
            Some(CameraMode::Tracking),
            "mode is not read while idle"
        );
        assert!(saved.pause_areas.is_some());
        assert!(
            backend.readback.borrow().is_some(),
            "the next snapshot reuses the readback"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn update_saved(
        dir: &Path,
        id: &str,
        update: impl FnOnce(Option<Saved>) -> Result<Saved>,
    ) -> Result<()> {
        let lock = lock_saved(dir, id)?;
        let saved = update(lock.load()?)?;
        lock.store(&saved)
    }

    #[test]
    fn a_user_change_is_one_locked_write_readback_and_save() {
        let dir = scratch("persisted");
        let id = "usb:test";
        let (backend, fake) = fake_backend(50, false);
        update_saved(&dir, id, |_| {
            let mut saved = Saved::new(id);
            saved.state.image.contrast = Some(10);
            Ok(saved)
        })
        .unwrap();
        backend
            .persisted_in(&dir, id, &["image.brightness".to_owned()], || {
                assert!(
                    lock_saved(&dir, id).is_err(),
                    "the lock is held during the hardware write"
                );
                Ok(backend.camera.set_image(&ImageSettings {
                    brightness: Some(70),
                    ..Default::default()
                })?)
            })
            .unwrap();
        let saved = load_saved(&saved_path(&dir, id), id).unwrap().unwrap();
        assert_eq!(
            saved.state.image.brightness,
            Some(70),
            "the readback taken after the write is saved"
        );
        assert_eq!(
            saved.state.image.contrast,
            Some(10),
            "unrelated values it could not read are kept"
        );
        assert_eq!(
            fake.0.lock().unwrap().writes,
            [(link::transport::Control::Brightness, 70)]
        );
        assert!(
            backend.readback.borrow().is_some(),
            "the next snapshot reuses the readback"
        );
        assert!(
            lock_saved(&dir, id).is_ok(),
            "the lock is released afterwards"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failures_before_the_write_are_ordinary_and_write_nothing() {
        let dir = scratch("preflight");
        let id = "usb:test";
        let (backend, _fake) = fake_backend(50, false);
        let mut ran = false;
        // No sysfs identity for the fake node.
        let error = backend
            .persisted(&[], || {
                ran = true;
                Ok(())
            })
            .unwrap_err();
        assert!(!ran && !applied_but_not_saved(&error));

        let held = lock_saved(&dir, id).unwrap();
        let error = backend
            .persisted_in(&dir, id, &[], || {
                ran = true;
                Ok(())
            })
            .unwrap_err();
        assert!(!ran && !applied_but_not_saved(&error), "{error:#}");
        drop(held);

        fs::write(saved_path(&dir, id), "{not json").unwrap();
        let error = backend
            .persisted_in(&dir, id, &[], || {
                ran = true;
                Ok(())
            })
            .unwrap_err();
        assert!(
            !ran && !applied_but_not_saved(&error),
            "a corrupt file refuses the change"
        );
        assert_eq!(fs::read(saved_path(&dir, id)).unwrap(), b"{not json");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_rejected_hardware_write_saves_nothing_and_is_not_applied() {
        let dir = scratch("rejected");
        let id = "usb:test";
        let (backend, _fake) = fake_backend(50, false);
        let error = backend
            .persisted_in(&dir, id, &["image.brightness".to_owned()], || {
                Err::<(), _>(anyhow!("write rejected"))
            })
            .unwrap_err();
        assert!(!applied_but_not_saved(&error));
        assert!(!saved_path(&dir, id).exists());
        assert!(backend.readback.borrow().is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_change_that_cannot_be_read_back_keeps_the_old_file_and_is_applied_but_not_saved() {
        let dir = scratch("unverified");
        let id = "usb:test";
        let (backend, _fake) = fake_backend(70, false);
        update_saved(&dir, id, |_| {
            let mut saved = Saved::new(id);
            saved.state.image.contrast = Some(10);
            Ok(saved)
        })
        .unwrap();
        let before = fs::read(saved_path(&dir, id)).unwrap();
        let mut ran = false;
        // The fake has no contrast control: the readback lacks the setting just changed.
        let error = backend
            .persisted_in(&dir, id, &["image.contrast".to_owned()], || {
                ran = true;
                Ok(())
            })
            .unwrap_err();
        assert!(ran && applied_but_not_saved(&error));
        assert!(format!("{error:#}").contains("image.contrast"), "{error:#}");
        assert_eq!(fs::read(saved_path(&dir, id)).unwrap(), before);
        assert!(backend.readback.borrow().is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(feature = "gui")]
    #[test]
    fn a_lost_usb_link_during_the_readback_stays_visible_as_a_transport_failure() {
        let dir = scratch("lost");
        let kind = link::xu::UnitKind::Control;
        let info = link::DeviceInfo::new(
            "/camera-lost-after-write",
            0x4c03,
            vec![link::xu::ExtensionUnit {
                id: 9,
                guid: kind.guid(),
                kind: Some(kind),
                controls: vec![0xff; 8],
            }],
        );
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend =
            CameraBackend::new(link::Camera::with_transport(info, FailedTransport(calls)));
        let error = backend
            .persisted_in(&dir, "usb:test", &[], || Ok(()))
            .unwrap_err();
        assert!(applied_but_not_saved(&error));
        assert!(is_transport_failure(&error), "{error:#}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn restore_waits_for_the_same_lock_as_user_changes() {
        let dir = scratch("restore-lock");
        let id = "usb:test";
        let (backend, fake) = fake_backend(50, false);
        update_saved(&dir, id, |_| {
            let mut saved = Saved::new(id);
            saved.state.image.brightness = Some(70);
            Ok(saved)
        })
        .unwrap();
        let held = lock_saved(&dir, id).unwrap();
        assert!(backend.restore_in(&dir, id).is_err());
        assert!(
            fake.0.lock().unwrap().writes.is_empty(),
            "no hardware write without the lock"
        );
        drop(held);
        assert_eq!(
            backend.restore_in(&dir, id).unwrap(),
            RestoreStatus::WaitingForStream
        );
        assert_eq!(
            fake.0.lock().unwrap().writes,
            [(link::transport::Control::Brightness, 70)]
        );
        assert_eq!(
            backend.restore_in(&dir, "usb:none").unwrap(),
            RestoreStatus::Missing
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_streaming_state_restores_only_standard_controls() {
        let (_, fake) = fake_backend(50, true);
        // Drop the device-status unit: is_streaming() is then Unsupported (unknown).
        let info = link::DeviceInfo::new("/dev/video99999", 0x4c04, Vec::new());
        let backend = CameraBackend::new(link::Camera::with_transport(info, fake.clone()));
        let saved = saved_with(State {
            mode: Some(CameraMode::Tracking),
            image: ImageSettings {
                brightness: Some(70),
                ..Default::default()
            },
            ..Default::default()
        });
        assert_eq!(
            backend.restore(&saved).unwrap(),
            RestoreStatus::WaitingForStream
        );
        let state = fake.0.lock().unwrap();
        assert_eq!(state.writes, [(link::transport::Control::Brightness, 70)]);
        assert_eq!(state.xu_writes, 0);
    }
}
