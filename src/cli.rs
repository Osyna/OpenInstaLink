//! Command line: `link-camera` with no subcommand opens the GUI; everything else runs headless
//! on the same backend and media layers.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, CommandFactory, Parser, Subcommand};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::backend::{self, CameraBackend, Control, ControlKind, PauseAreaInfo, TrackingAreaInfo};
use crate::effects::Effects;
use crate::media::{Frame, Preview, VideoConfig};
use crate::models::{
    Cancelled, Client, MODEL_IDS, ModelInfo, RuntimeConfig, defaults, interpret, model_spec,
    save_result, save_visualization,
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(
    name = "link-camera",
    version,
    about = "Local Insta360 Link controller (GUI and CLI)"
)]
struct Cli {
    /// Camera device node (default: first Insta360 Link found), e.g. /dev/video0
    #[arg(long, global = true)]
    device: Option<String>,
    /// Machine-readable JSON output where a command prints data
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open the graphical interface (default)
    Gui,
    #[cfg(feature = "gui")]
    #[command(name = "__camera-worker", hide = true)]
    CameraWorker,
    /// List connected cameras
    List,
    /// Device identity, streaming state, every control and warnings
    Status,
    /// List the MJPEG capture size/frame-rate combinations the camera advertises
    Formats,
    /// List the controls this camera supports with current values
    Controls {
        /// Only this group (Picture, View, Framing, Audio, Gestures)
        #[arg(long)]
        group: Option<String>,
    },
    /// Set one control (see `controls` for keys, ranges and options)
    Set {
        key: String,
        #[arg(allow_hyphen_values = true)]
        value: String,
    },
    /// Set pan/tilt degrees (standard V4L2 controls); an omitted axis keeps its angle
    Position {
        #[arg(long, allow_hyphen_values = true)]
        pan: Option<f64>,
        #[arg(long, allow_hyphen_values = true)]
        tilt: Option<f64>,
    },
    /// Reset pan/tilt to 0,0
    Center,
    /// Camera-stored views (slots 0-9)
    #[command(subcommand)]
    Preset(PresetCmd),
    /// Pause-track zones and tracking-area limits
    #[command(subcommand)]
    Area(AreaCmd),
    /// Saved settings profiles (local JSON files)
    #[command(subcommand)]
    Profile(ProfileCmd),
    /// Save one still frame (.png/.jpg) from the processed stream
    Snapshot {
        path: PathBuf,
        #[command(flatten)]
        media: MediaArgs,
    },
    /// Record the processed stream until Ctrl-C or --duration
    Record {
        path: PathBuf,
        /// Stop after this many seconds
        #[arg(long, value_parser = seconds_arg)]
        duration: Option<f64>,
        /// Audio input name for the recording (omit for video only)
        #[arg(long)]
        audio: Option<String>,
        /// Also feed an existing V4L2 output node (v4l2loopback)
        #[arg(long)]
        virtual_output: Option<PathBuf>,
        #[command(flatten)]
        media: MediaArgs,
    },
    /// Feed the processed stream to an existing V4L2 output node until Ctrl-C or --duration
    Stream {
        /// Existing V4L2 output node, e.g. /dev/video10 (the module is never loaded for you)
        #[arg(long)]
        virtual_output: PathBuf,
        #[arg(long, value_parser = seconds_arg)]
        duration: Option<f64>,
        #[command(flatten)]
        media: MediaArgs,
    },
    /// Serve the local web remote
    Serve {
        #[arg(long, default_value = "127.0.0.1:9360")]
        listen: SocketAddr,
        /// Access token (required off loopback; generated if omitted on loopback)
        #[arg(long)]
        token: Option<String>,
    },
    /// Saved settings (optional AI is off until you turn it on)
    #[command(subcommand)]
    Settings(SettingsCmd),
    /// Restore saved camera settings without owning capture; manage the login helper
    #[command(subcommand)]
    Restore(RestoreCmd),
    /// On-machine vision models: library, downloads and offline inference (no server)
    #[command(subcommand)]
    Models(ModelsCmd),
    /// Create, inspect, stop and remove app-owned virtual cameras
    #[command(subcommand)]
    VirtualCamera(VirtualCmd),
    /// Inspect microphone activity, mute and volume without opening capture
    #[command(subcommand)]
    Mic(MicCmd),
}

#[derive(Subcommand)]
enum VirtualCmd {
    List,
    /// Show driver, tool and privilege prerequisites
    Status,
    /// Create a node; --authorize permits a polkit prompt and loading the installed module
    Create {
        #[arg(long, default_value = "Link Camera")]
        label: String,
        #[arg(long)]
        authorize: bool,
    },
    /// Stop this app's producer on one node, without opening the physical camera
    Stop {
        path: PathBuf,
    },
    /// Stop every virtual output owned by this app, across its GUI/CLI instances
    StopAll,
    /// Stop and remove one node created by this app
    Remove {
        path: PathBuf,
        #[arg(long)]
        authorize: bool,
    },
    /// Stop and remove all nodes created by this app; other virtual cameras are untouched
    RemoveAll {
        #[arg(long)]
        authorize: bool,
    },
}

#[derive(Subcommand)]
enum MicCmd {
    Sources,
    /// Match the physical camera's microphone; report ambiguity instead of choosing another mic
    Status,
    Mute {
        source: String,
        #[arg(action = clap::ArgAction::Set)]
        muted: bool,
    },
    Volume {
        source: String,
        #[arg(value_parser = clap::value_parser!(u32).range(0..=100))]
        percent: u32,
    },
}

#[derive(Subcommand)]
enum RestoreCmd {
    /// Apply saved hardware settings once (vendor settings wait for an active stream)
    Apply,
    /// Enable the no-capture helper for this user's login session and future logins
    Enable,
    /// Disable automatic restoration for other applications
    Disable,
    /// Watch connected cameras without opening a video stream
    Watch,
    #[command(name = "__monitor", hide = true)]
    Monitor,
}

#[derive(Subcommand)]
enum SettingsCmd {
    /// Show the saved settings and where they live
    Show,
    /// Turn optional AI (model API, neural person blur/replacement) on or off; nothing connects by itself
    Ai {
        #[arg(value_parser = clap::value_parser!(bool), action = clap::ArgAction::Set)]
        enabled: bool,
    },
}

#[derive(Subcommand)]
enum PresetCmd {
    List,
    /// Store the current mode, position and zoom
    Save {
        #[arg(value_parser = clap::value_parser!(u8).range(0..=9))]
        slot: u8,
        name: String,
    },
    Recall {
        #[arg(value_parser = clap::value_parser!(u8).range(0..=9))]
        slot: u8,
    },
    Delete {
        #[arg(value_parser = clap::value_parser!(u8).range(0..=9))]
        slot: u8,
    },
}

#[derive(Subcommand)]
enum AreaCmd {
    /// Pause-track zones (up to 6); switch them on with `set pause-areas true`
    #[command(subcommand)]
    Pause(PauseCmd),
    /// Tracking-area limits (Link 2 Pro only); switch with `set tracking-area true`
    #[command(subcommand)]
    Tracking(TrackingCmd),
}

#[derive(Subcommand)]
enum PauseCmd {
    List,
    /// Replace all zones. Each --rect is X,Y,W,H as fractions of the frame from the top-left
    Set {
        #[arg(long = "rect", required = true, num_args = 1, value_parser = rect_arg)]
        rects: Vec<[f64; 4]>,
        /// Gimbal angle the zones are defined at (default: current position)
        #[arg(long, allow_hyphen_values = true)]
        pan: Option<f64>,
        #[arg(long, allow_hyphen_values = true)]
        tilt: Option<f64>,
    },
    Clear,
}

#[derive(Subcommand)]
enum TrackingCmd {
    Show,
    /// Set limits in degrees; `--disabled` stores them without enforcing
    Set {
        #[arg(long, allow_hyphen_values = true, value_parser = pair_arg, value_name = "MIN,MAX")]
        pan: [f64; 2],
        #[arg(long, allow_hyphen_values = true, value_parser = pair_arg, value_name = "MIN,MAX")]
        tilt: [f64; 2],
        #[arg(long)]
        disabled: bool,
    },
}

#[derive(Subcommand)]
enum ProfileCmd {
    List,
    /// Save the camera's current settings under NAME
    Save {
        name: String,
    },
    /// Apply a saved profile (stops at the first setting the camera rejects)
    Load {
        name: String,
    },
    Delete {
        name: String,
    },
    /// Write the current state JSON to PATH (stdout if omitted; never overwrites)
    Export {
        path: Option<PathBuf>,
    },
    /// Apply a state JSON file
    Apply {
        path: PathBuf,
    },
}

/// Inference deadline for the owned local process.
#[derive(Args)]
struct RuntimeArgs {
    /// Stop the local inference process after this many seconds, 1-3600 (default 300)
    #[arg(long, value_parser = timeout_arg)]
    timeout: Option<f64>,
}

impl RuntimeArgs {
    fn client(&self) -> Result<Client> {
        let mut config = RuntimeConfig::default();
        if let Some(seconds) = self.timeout {
            config.timeout = Duration::from_secs_f64(seconds);
        }
        Client::new(config)
    }
}

#[derive(Args)]
struct ModelRunArgs {
    #[arg(value_parser = clap::builder::PossibleValuesParser::new(MODEL_IDS))]
    model: String,
    /// PNG/JPEG file to analyse instead of a camera frame (the camera is not opened)
    #[arg(long)]
    input: Option<PathBuf>,
    /// Options as a JSON object merged over the model defaults, e.g. '{"threshold":0.4}'
    #[arg(long, value_parser = options_arg)]
    options: Option<Map<String, Value>>,
    /// Save the raw result (.json, or .png for depth models); JSON models print to stdout without it
    #[arg(long)]
    output: Option<PathBuf>,
    /// Save the annotated result image (.png)
    #[arg(long)]
    visualization: Option<PathBuf>,
    #[command(flatten)]
    runtime: RuntimeArgs,
    /// Camera capture settings; ignored with --input
    #[command(flatten)]
    media: MediaArgs,
}

#[derive(Args)]
struct ModelWatchArgs {
    #[arg(value_parser = clap::builder::PossibleValuesParser::new(MODEL_IDS))]
    model: String,
    #[arg(long, value_parser = options_arg)]
    options: Option<Map<String, Value>>,
    /// Minimum seconds between submissions, 0-3600 (0 runs as fast as new frames allow)
    #[arg(long, value_parser = interval_arg)]
    interval: Option<f64>,
    /// Stop after this many seconds (checked between inferences)
    #[arg(long, value_parser = seconds_arg)]
    duration: Option<f64>,
    /// Write each raw result here as MODEL-STAMP-NNNN.json/.png (required for depth models)
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Also write the annotated image next to each raw result
    #[arg(long, requires = "output_dir")]
    visualization: bool,
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[command(flatten)]
    media: MediaArgs,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List local installed/downloadable models, options and licenses (no camera needed)
    List,
    /// Download this model and provision its local runtime; Ctrl-C cancels owned work
    Download {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(MODEL_IDS))]
        model: String,
    },
    /// Delete this app's downloaded model files (also available with AI disabled)
    Delete {
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(MODEL_IDS))]
        model: String,
    },
    /// Run one model once on an image file or one camera frame
    Run(ModelRunArgs),
    /// Analyse the newest camera frame repeatedly and print one JSON line per result until Ctrl-C or --duration
    Watch(ModelWatchArgs),
}

/// Capture format and local effects shared by snapshot, record and stream.
#[derive(Args)]
struct MediaArgs {
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// Effects as a JSON file (the Effects structure); flags below override it
    #[arg(long)]
    effects: Option<PathBuf>,
    #[arg(long, value_parser = ["none", "blur", "replace", "chroma"])]
    background: Option<String>,
    #[arg(long)]
    background_image: Option<PathBuf>,
    /// Background blur strength, 0-30
    #[arg(long, value_parser = blur_arg)]
    blur: Option<f32>,
    /// Softening, 0-1
    #[arg(long, value_parser = unit_arg)]
    soften: Option<f32>,
    #[arg(long)]
    mirror: bool,
    #[arg(long)]
    overlay: Option<PathBuf>,
    /// Overlay opacity, 0-1
    #[arg(long, value_parser = unit_arg)]
    overlay_opacity: Option<f32>,
    /// Perspective rectangle as 8 normalized numbers TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy
    #[arg(long, value_parser = board_arg, allow_hyphen_values = true)]
    board: Option<[[f32; 2]; 4]>,
}

impl MediaArgs {
    fn video(&self) -> VideoConfig {
        VideoConfig {
            width: self.width,
            height: self.height,
            fps: self.fps,
        }
    }

    /// `None` when no effect was requested, so the processing path stays untouched.
    fn effects(&self) -> Result<Option<Effects>> {
        let mut effects = match &self.effects {
            Some(path) => Some(read_json::<Effects>(path)?),
            None => None,
        };
        let flags = self.background.is_some()
            || self.background_image.is_some()
            || self.blur.is_some()
            || self.soften.is_some()
            || self.mirror
            || self.overlay.is_some()
            || self.overlay_opacity.is_some()
            || self.board.is_some();
        if !flags {
            return Ok(effects);
        }
        let e = effects.get_or_insert_with(Effects::default);
        if let Some(v) = &self.background {
            e.background = v.clone()
        }
        if let Some(v) = &self.background_image {
            e.background_image = Some(v.clone())
        }
        if let Some(v) = self.blur {
            e.blur = v
        }
        if let Some(v) = self.soften {
            e.soften = v
        }
        if self.mirror {
            e.mirror = true
        }
        if let Some(v) = &self.overlay {
            e.overlay_image = Some(v.clone())
        }
        if let Some(v) = self.overlay_opacity {
            e.overlay_opacity = v
        }
        if let Some(v) = self.board {
            e.board = Some(v)
        }
        Ok(effects)
    }
}

// ---- argument parsers -----------------------------------------------------------------------

fn numbers(text: &str, count: usize) -> std::result::Result<Vec<f64>, String> {
    let values: Vec<f64> = text
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .map_err(|_| format!("`{t}` is not a number"))
        })
        .collect::<std::result::Result<_, _>>()?;
    if values.len() != count || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("expected {count} finite comma-separated numbers"));
    }
    Ok(values)
}

fn ranged(text: &str, min: f32, max: f32) -> std::result::Result<f32, String> {
    match text.parse::<f32>() {
        Ok(v) if (min..=max).contains(&v) => Ok(v),
        _ => Err(format!("expected a number from {min} to {max}")),
    }
}

fn blur_arg(text: &str) -> std::result::Result<f32, String> {
    ranged(text, 0.0, 30.0)
}

fn unit_arg(text: &str) -> std::result::Result<f32, String> {
    ranged(text, 0.0, 1.0)
}

fn seconds_arg(text: &str) -> std::result::Result<f64, String> {
    match text.parse::<f64>() {
        Ok(v) if v > 0.0 && v <= 31_536_000.0 => Ok(v),
        _ => Err("expected a positive number of seconds (at most one year)".into()),
    }
}

fn timeout_arg(text: &str) -> std::result::Result<f64, String> {
    match text.parse::<f64>() {
        Ok(v) if (1.0..=3600.0).contains(&v) => Ok(v),
        _ => Err("expected a number of seconds from 1 to 3600".into()),
    }
}

fn interval_arg(text: &str) -> std::result::Result<f64, String> {
    match text.parse::<f64>() {
        Ok(v) if (0.0..=3600.0).contains(&v) => Ok(v),
        _ => Err("expected a number of seconds from 0 to 3600".into()),
    }
}

fn options_arg(text: &str) -> std::result::Result<Map<String, Value>, String> {
    if text.len() > 64 << 10 {
        return Err("options are larger than 64 KiB".into());
    }
    match serde_json::from_str(text) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err("options must be a JSON object".into()),
        Err(error) => Err(format!("options are not valid JSON: {error}")),
    }
}

fn pair_arg(text: &str) -> std::result::Result<[f64; 2], String> {
    let v = numbers(text, 2)?;
    Ok([v[0], v[1]])
}

fn rect_arg(text: &str) -> std::result::Result<[f64; 4], String> {
    let v = numbers(text, 4)?;
    let [x, y, w, h] = [v[0], v[1], v[2], v[3]];
    if x < 0.0 || y < 0.0 || w <= 0.0 || h <= 0.0 || x + w > 1.0 || y + h > 1.0 {
        return Err("rectangle must lie inside the frame (0..1) with positive size".into());
    }
    Ok([x, y, w, h])
}

fn board_arg(text: &str) -> std::result::Result<[[f32; 2]; 4], String> {
    let v = numbers(text, 8)?;
    if v.iter().any(|n| !(0.0..=1.0).contains(n)) {
        return Err("board coordinates must be normalized (0..1)".into());
    }
    let p = |i: usize| [v[i] as f32, v[i + 1] as f32];
    Ok([p(0), p(2), p(4), p(6)])
}

/// Off loopback the remote can be reached by others, so it needs an explicit token.
fn check_listen(listen: SocketAddr, token: Option<&str>) -> Result<()> {
    let loopback = listen.ip().is_loopback();
    if !loopback && token.is_none_or(|t| t.trim().is_empty()) {
        bail!("listening on {listen} (not loopback) requires an explicit non-empty --token");
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    const MAX: u64 = 1 << 20;
    let len = std::fs::metadata(path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    if len > MAX {
        bail!("{} is larger than 1 MiB", path.display());
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

// ---- entry point ----------------------------------------------------------------------------

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        None if cfg!(feature = "gui") => gui(cli.device.clone()),
        None => {
            Cli::command().print_help()?;
            bail!("this build has no GUI (built without the `gui` feature); choose a subcommand")
        }
        Some(Cmd::Gui) => gui(cli.device.clone()),
        Some(command) => {
            let changed = matches!(
                command,
                Cmd::Set { .. }
                    | Cmd::Position { .. }
                    | Cmd::Center
                    | Cmd::Preset(PresetCmd::Recall { .. })
                    | Cmd::Profile(ProfileCmd::Load { .. } | ProfileCmd::Apply { .. })
                    | Cmd::Area(
                        AreaCmd::Pause(PauseCmd::Set { .. } | PauseCmd::Clear)
                            | AreaCmd::Tracking(TrackingCmd::Set { .. })
                    )
            );
            if matches!(command, Cmd::Serve { .. }) {
                if let Err(error) = crate::restore::ensure_user_service() {
                    eprintln!("link-camera: automatic restore unavailable: {error:#}");
                }
            }
            let result = cli.dispatch(command);
            if changed && result.is_ok() {
                if let Err(error) = crate::restore::ensure_user_service() {
                    eprintln!(
                        "link-camera: settings applied and saved, but automatic restore is unavailable: {error:#}"
                    );
                }
            }
            result
        }
    }
}

#[cfg(feature = "gui")]
fn gui(device: Option<String>) -> Result<()> {
    crate::gui::run(device)
}

#[cfg(not(feature = "gui"))]
fn gui(_: Option<String>) -> Result<()> {
    bail!("this build has no GUI (built without the `gui` feature)")
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn settings(cmd: &SettingsCmd, json: bool) -> Result<()> {
    use crate::preferences;
    if let SettingsCmd::Ai { enabled } = cmd {
        preferences::set_ai_enabled(*enabled)?;
    }
    let prefs = preferences::load()?;
    let ai_enabled = prefs.ai_enabled;
    let path = preferences::path()?;
    if json {
        return print_json(
            &json!({ "ai_enabled": ai_enabled, "capture": prefs.capture, "effects": prefs.effects, "path": path }),
        );
    }
    println!(
        "optional AI: {} ({})",
        if ai_enabled { "enabled" } else { "disabled" },
        path.display()
    );
    Ok(())
}

fn print_controls(controls: &[Control]) {
    let mut group = "";
    for c in controls {
        if c.group != group {
            println!("{}", c.group);
            group = &c.group;
        }
        let value = if c.value.is_empty() { "?" } else { &c.value };
        let spec = match c.kind {
            ControlKind::Number => format!("{}..{} step {}", c.min, c.max, c.step),
            ControlKind::Choice => c.options.join("|"),
            ControlKind::Toggle => "true|false".into(),
        };
        println!("  {:<26} {:<12} {}", c.key, value, spec);
    }
}

impl Cli {
    fn open(&self) -> Result<CameraBackend> {
        CameraBackend::open(self.device.as_deref())
    }

    /// Runs `action`; if the camera needs a stream, holds a temporary capture and retries once.
    /// Other errors, and the retry's own error, are returned unchanged.
    fn streaming<T>(
        &self,
        backend: &CameraBackend,
        mut action: impl FnMut() -> Result<T>,
    ) -> Result<T> {
        match action() {
            Err(error)
                if !backend::applied_but_not_saved(&error)
                    && matches!(
                        error.downcast_ref::<insta360_link::Error>(),
                        Some(insta360_link::Error::NotStreaming)
                    ) =>
            {
                eprintln!(
                    "link-camera: camera is not streaming; holding a temporary capture to apply this"
                );
                let preview = Preview::start(&backend.device().id, &VideoConfig::default())
                    .context("cannot start a temporary stream (is another program using the camera exclusively?)")?;
                preview
                    .wait_first_frame(Duration::from_secs(15))
                    .context("temporary stream produced no frame")?;
                let result = action();
                drop(preview);
                result
            }
            other => other,
        }
    }

    fn start_preview(&self, backend: &CameraBackend, media: &MediaArgs) -> Result<Preview> {
        let preview =
            Preview::start(&backend.device().id, &media.video()).context("starting capture")?;
        if let Some(effects) = media.effects()? {
            preview
                .set_effects(effects)
                .context("configuring effects")?;
        }
        preview
            .wait_first_frame(Duration::from_secs(15))
            .context("waiting for the first frame")?;
        Ok(preview)
    }

    fn dispatch(&self, command: &Cmd) -> Result<()> {
        match command {
            Cmd::Gui => unreachable!("handled in run"),
            #[cfg(feature = "gui")]
            Cmd::CameraWorker => crate::gui::camera_worker_process(),
            Cmd::Restore(action) => match action {
                RestoreCmd::Apply => {
                    let status = self.open()?.restore_saved_settings()?;
                    if self.json {
                        print_json(&status)
                    } else {
                        println!("{status:?}");
                        Ok(())
                    }
                }
                RestoreCmd::Enable => crate::restore::enable_user_service(),
                RestoreCmd::Disable => crate::restore::disable_user_service(),
                RestoreCmd::Watch => crate::restore::watch(),
                RestoreCmd::Monitor => crate::restore::monitor(
                    self.device
                        .as_deref()
                        .context("restore worker requires --device")?,
                ),
            },
            Cmd::List => {
                let devices = backend::devices()?;
                if self.json {
                    return print_json(&devices);
                }
                if devices.is_empty() {
                    eprintln!("no Insta360 Link series camera found");
                }
                for d in devices {
                    println!("{}\t{}\t{}", d.id, d.name, d.model);
                }
                Ok(())
            }
            Cmd::Formats => {
                let camera = self.open()?;
                let formats = crate::media::formats(&camera.device().id)?;
                if self.json {
                    return print_json(&formats);
                }
                for format in formats {
                    println!("{}x{} @ {} fps", format.width, format.height, format.fps);
                }
                Ok(())
            }
            Cmd::Status => {
                let s = self.open()?.snapshot()?;
                if self.json {
                    return print_json(&s);
                }
                println!("{} ({}) {}", s.device.name, s.device.model, s.device.id);
                println!(
                    "firmware {}  serial {}  streaming {}",
                    or_unknown(&s.firmware),
                    or_unknown(&s.serial),
                    s.streaming
                );
                print_controls(&s.controls);
                s.warnings.iter().for_each(|w| eprintln!("warning: {w}"));
                Ok(())
            }
            Cmd::Controls { group } => {
                let s = self.open()?.snapshot()?;
                let controls: Vec<Control> = s
                    .controls
                    .into_iter()
                    .filter(|c| {
                        group
                            .as_ref()
                            .is_none_or(|g| c.group.eq_ignore_ascii_case(g))
                    })
                    .collect();
                if self.json {
                    print_json(&controls)?;
                } else {
                    print_controls(&controls);
                }
                s.warnings.iter().for_each(|w| eprintln!("warning: {w}"));
                Ok(())
            }
            Cmd::Set { key, value } => {
                let b = self.open()?;
                self.streaming(&b, || b.set(key, value))?;
                // The write succeeded; a failed readback must not turn it into a failure.
                match b.control(key) {
                    Ok(c) if self.json => print_json(&c)?,
                    Ok(c) => println!(
                        "{} = {}",
                        c.key,
                        if c.value.is_empty() { "?" } else { &c.value }
                    ),
                    Err(e) => eprintln!("set applied; readback unavailable: {e:#}"),
                }
                Ok(())
            }
            Cmd::Position { pan, tilt } => {
                if pan.is_none() && tilt.is_none() {
                    bail!("give --pan and/or --tilt (degrees)");
                }
                let b = self.open()?;
                b.set_position(*pan, *tilt)?;
                print_position(&b, self.json)
            }
            Cmd::Center => {
                let b = self.open()?;
                b.recenter()?;
                print_position(&b, self.json)
            }
            Cmd::Preset(cmd) => self.preset(cmd),
            Cmd::Area(cmd) => self.area(cmd),
            Cmd::Profile(cmd) => self.profile(cmd),
            Cmd::Snapshot { path, media } => {
                refuse_existing(path)?;
                let b = self.open()?;
                let preview = self.start_preview(&b, media)?;
                preview
                    .snapshot(path)
                    .with_context(|| format!("saving {}", path.display()))?;
                println!("{}", path.display());
                Ok(())
            }
            Cmd::Record {
                path,
                duration,
                audio,
                virtual_output,
                media,
            } => {
                refuse_existing(path)?;
                let b = self.open()?;
                let preview = self.start_preview(&b, media)?;
                if let Some(node) = virtual_output {
                    preview
                        .set_virtual_output(Some(node))
                        .with_context(|| format!("virtual output {}", node.display()))?;
                }
                install_ctrlc()?;
                preview
                    .start_recording(path, audio.as_deref())
                    .context("starting recording")?;
                eprintln!("recording to {} (Ctrl-C to stop)", path.display());
                let waited = wait(&preview, *duration);
                let stopped = preview.stop_recording();
                let frames = preview.status().frames;
                let mut errors = Vec::new();
                if let Err(e) = waited {
                    errors.push(format!("{e:#}"));
                }
                match stopped {
                    Ok(()) => println!("recorded {} ({frames} frames captured)", path.display()),
                    Err(e) => errors.push(format!("finalizing {}: {e:#}", path.display())),
                }
                if errors.is_empty() {
                    Ok(())
                } else {
                    bail!("{}", errors.join("; "))
                }
            }
            Cmd::Stream {
                virtual_output,
                duration,
                media,
            } => {
                let b = self.open()?;
                let preview = self.start_preview(&b, media)?;
                preview
                    .set_virtual_output(Some(virtual_output))
                    .with_context(|| format!("virtual output {}", virtual_output.display()))?;
                install_ctrlc()?;
                eprintln!("streaming to {} (Ctrl-C to stop)", virtual_output.display());
                let waited = wait(&preview, *duration);
                let cleared = preview.set_virtual_output(None);
                waited?;
                cleared.context("closing the virtual output")
            }
            Cmd::Serve { listen, token } => {
                check_listen(*listen, token.as_deref())?;
                crate::remote::serve(self.device.clone(), *listen, token.clone())
            }
            Cmd::Settings(cmd) => settings(cmd, self.json),
            Cmd::Models(cmd) => self.models(cmd),
            Cmd::VirtualCamera(cmd) => self.virtual_camera(cmd),
            Cmd::Mic(cmd) => self.microphone(cmd),
        }
    }

    fn virtual_camera(&self, cmd: &VirtualCmd) -> Result<()> {
        use crate::virtual_camera::{self as virtuals, CreateOptions, Elevation};
        let elevation = |authorize: bool| {
            if authorize {
                Elevation::Pkexec
            } else {
                Elevation::Refuse
            }
        };
        match cmd {
            VirtualCmd::List => {
                let devices = virtuals::list()?;
                if self.json {
                    return print_json(&devices);
                }
                for node in devices {
                    print!(
                        "{}\t{}\t{}",
                        node.path.display(),
                        node.label,
                        if node.owned { "owned" } else { "unmanaged" }
                    );
                    match node.producer_pid {
                        Some(pid) => println!("\tproducer {pid}"),
                        None => println!("\tstopped"),
                    }
                }
                Ok(())
            }
            VirtualCmd::Status => {
                let status = virtuals::prerequisites();
                if self.json {
                    return print_json(&status);
                }
                let missing = status.missing();
                if missing.is_empty() {
                    println!("Virtual camera prerequisites are available.");
                } else {
                    for action in missing {
                        println!("{action}");
                    }
                }
                Ok(())
            }
            VirtualCmd::Create { label, authorize } => {
                let node = virtuals::create(&CreateOptions {
                    label: label.clone(),
                    elevation: elevation(*authorize),
                    load_module: *authorize,
                })?;
                print_one(self.json, &node, || {
                    format!("{}\t{}", node.path.display(), node.label)
                })
            }
            VirtualCmd::Stop { path } => {
                let stopped = virtuals::stop(path)?;
                print_one(self.json, &json!({"path":path,"stopped":stopped}), || {
                    format!(
                        "{}: {}",
                        path.display(),
                        if stopped {
                            "producer stopped"
                        } else {
                            "no app-owned producer running"
                        }
                    )
                })
            }
            VirtualCmd::StopAll => virtual_report(virtuals::stop_all(), self.json),
            VirtualCmd::Remove { path, authorize } => {
                virtuals::remove(path, elevation(*authorize))?;
                print_one(self.json, &json!({"path":path,"removed":true}), || {
                    format!("{}: removed", path.display())
                })
            }
            VirtualCmd::RemoveAll { authorize } => {
                virtual_report(virtuals::remove_all(elevation(*authorize)), self.json)
            }
        }
    }

    fn microphone(&self, cmd: &MicCmd) -> Result<()> {
        use crate::audio;
        match cmd {
            MicCmd::Sources => {
                let sources = audio::sources()?;
                if self.json {
                    return print_json(&sources);
                }
                for source in sources {
                    println!(
                        "{}\t{}\tstate {:?}\tmuted {:?}\tvolume {:?}%",
                        source.name,
                        plain(&source.description),
                        source.state,
                        source.muted,
                        source.volume_percent
                    );
                }
                Ok(())
            }
            MicCmd::Status => {
                let devices;
                let node = if let Some(node) = self.device.as_deref() {
                    node
                } else {
                    devices = backend::devices()?;
                    &devices
                        .first()
                        .context("no camera found; specify --device or use mic sources")?
                        .id
                };
                let status = audio::camera_mic(node)?;
                print_json(&status)
            }
            MicCmd::Mute { source, muted } => {
                let status = audio::set_muted(source, *muted)?;
                print_json(&status)
            }
            MicCmd::Volume { source, percent } => {
                let status = audio::set_volume(source, *percent)?;
                print_json(&status)
            }
        }
    }

    fn preset(&self, cmd: &PresetCmd) -> Result<()> {
        let b = self.open()?;
        match cmd {
            PresetCmd::List => {
                let presets = self.streaming(&b, || b.presets())?;
                if self.json {
                    return print_json(&presets);
                }
                for p in presets {
                    println!(
                        "{}\t{}\t{}\tpan {} tilt {} zoom {}",
                        p.slot, p.name, p.mode, p.pan, p.tilt, p.zoom
                    );
                }
                Ok(())
            }
            PresetCmd::Save { slot, name } => {
                let p = self.streaming(&b, || b.save_preset(*slot, name))?;
                print_one(self.json, &p, || {
                    format!(
                        "saved slot {} `{}` ({}, pan {} tilt {} zoom {})",
                        p.slot, p.name, p.mode, p.pan, p.tilt, p.zoom
                    )
                })
            }
            PresetCmd::Recall { slot } => {
                let p = self.streaming(&b, || b.recall_preset(*slot))?;
                print_one(self.json, &p, || {
                    format!("recalled slot {} `{}` ({})", p.slot, p.name, p.mode)
                })
            }
            PresetCmd::Delete { slot } => self.streaming(&b, || b.delete_preset(*slot)),
        }
    }

    fn area(&self, cmd: &AreaCmd) -> Result<()> {
        let b = self.open()?;
        match cmd {
            AreaCmd::Pause(PauseCmd::List) => {
                let areas = b.pause_areas()?;
                if self.json {
                    return print_json(&areas);
                }
                for (i, a) in areas.iter().enumerate() {
                    println!(
                        "{}\t{}\tx {} y {} w {} h {}\tat pan {} tilt {}",
                        i + 1,
                        if a.active { "active" } else { "inactive" },
                        a.x,
                        a.y,
                        a.width,
                        a.height,
                        a.pan,
                        a.tilt
                    );
                }
                Ok(())
            }
            AreaCmd::Pause(PauseCmd::Set { rects, pan, tilt }) => {
                let (cur_pan, cur_tilt) = match (pan, tilt) {
                    (Some(p), Some(t)) => (*p, *t),
                    _ => {
                        let (p, t) = b.position().context(
                            "cannot read the current gimbal angle; pass both --pan and --tilt",
                        )?;
                        (pan.unwrap_or(p), tilt.unwrap_or(t))
                    }
                };
                let areas: Vec<PauseAreaInfo> = rects
                    .iter()
                    .map(|&[x, y, width, height]| PauseAreaInfo {
                        active: true,
                        pan: cur_pan,
                        tilt: cur_tilt,
                        x,
                        y,
                        width,
                        height,
                    })
                    .collect();
                self.streaming(&b, || b.set_pause_areas(&areas))
            }
            AreaCmd::Pause(PauseCmd::Clear) => self.streaming(&b, || b.set_pause_areas(&[])),
            AreaCmd::Tracking(TrackingCmd::Show) => {
                let a = b.tracking_area()?;
                print_one(self.json, &a, || {
                    format!(
                        "{} pan {}..{} tilt {}..{}",
                        if a.enabled { "enabled" } else { "disabled" },
                        a.pan[0],
                        a.pan[1],
                        a.tilt[0],
                        a.tilt[1]
                    )
                })
            }
            AreaCmd::Tracking(TrackingCmd::Set {
                pan,
                tilt,
                disabled,
            }) => {
                let area = TrackingAreaInfo {
                    enabled: !disabled,
                    pan: *pan,
                    tilt: *tilt,
                };
                self.streaming(&b, || b.set_tracking_area(&area))
            }
        }
    }

    fn profile(&self, cmd: &ProfileCmd) -> Result<()> {
        match cmd {
            ProfileCmd::List => {
                let names = backend::profiles()?;
                if self.json {
                    return print_json(&names);
                }
                names.iter().for_each(|n| println!("{n}"));
                Ok(())
            }
            ProfileCmd::Delete { name } => backend::delete_profile(name),
            ProfileCmd::Save { name } => self.open()?.save_profile(name),
            ProfileCmd::Load { name } => {
                let b = self.open()?;
                self.streaming(&b, || b.load_profile(name))
            }
            ProfileCmd::Export { path } => {
                let mut json = serde_json::to_string_pretty(&self.open()?.export_state()?)?;
                json.push('\n');
                match path {
                    None => print!("{json}"),
                    Some(path) => {
                        use std::io::Write;
                        std::fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(path)
                            .and_then(|mut f| f.write_all(json.as_bytes()))
                            .with_context(|| format!("writing {}", path.display()))?;
                    }
                }
                Ok(())
            }
            ProfileCmd::Apply { path } => {
                let state = backend::read_state_file(path)?;
                let b = self.open()?;
                self.streaming(&b, || b.apply_state(&state))
            }
        }
    }

    fn models(&self, cmd: &ModelsCmd) -> Result<()> {
        match cmd {
            ModelsCmd::List => {
                crate::preferences::require_ai_enabled("`models list`")?;
                let catalog = Client::new(RuntimeConfig::default())?.catalog()?;
                if self.json {
                    return print_json(&catalog);
                }
                catalog.iter().for_each(print_model);
                Ok(())
            }
            ModelsCmd::Download { model } => {
                crate::preferences::require_ai_enabled("`models download`")?;
                install_ctrlc()?;
                eprintln!("Downloading {model} and its local runtime; Ctrl-C cancels.");
                let client = Client::new(RuntimeConfig::default())?;
                match client.download(model, &INTERRUPTED) {
                    Err(error) if error.is::<Cancelled>() => return Ok(()),
                    result => result?,
                }
                let catalog = client.catalog()?;
                let installed = catalog
                    .iter()
                    .find(|entry| entry.id == *model)
                    .context("download completed but model status is unavailable")?;
                print_one(self.json, installed, || {
                    format!("{model}: installed ({} bytes)", installed.disk_bytes)
                })
            }
            ModelsCmd::Delete { model } => {
                Client::new(RuntimeConfig::default())?.delete(model)?;
                print_one(self.json, &json!({"model":model,"deleted":true}), || {
                    format!("{model}: deleted")
                })
            }
            ModelsCmd::Run(args) => self.model_run(args),
            ModelsCmd::Watch(args) => self.model_watch(args),
        }
    }

    fn model_run(&self, args: &ModelRunArgs) -> Result<()> {
        let ModelRunArgs {
            model,
            input,
            options,
            output,
            visualization,
            runtime,
            media,
        } = args;
        let input = input.as_deref();
        let options = options.as_ref();
        let output = output.as_deref();
        let visualization = visualization.as_deref();
        let png = is_png_model(model);
        match output {
            Some(path) => check_new_file(path, if png { "png" } else { "json" })?,
            None if png => bail!(
                "{model} returns a PNG depth map; give --output FILE.png (binary is never written to the terminal)"
            ),
            None => {}
        }
        if let Some(path) = visualization {
            check_new_file(path, "png")?;
        }
        if output.is_some() && output == visualization {
            bail!("--output and --visualization must be different files");
        }
        crate::preferences::require_ai_enabled("`models run`")?;
        let client = runtime.client()?;
        let options = merge_options(model, options)?;
        let frame = match input {
            Some(path) => Arc::new(load_image(path)?),
            None => {
                let b = self.open()?;
                let preview = self.start_preview(&b, media)?;
                preview
                    .latest_frame()
                    .context("no camera frame available")?
            }
        };
        install_ctrlc()?;
        let raw = client.run(model, &options, &frame, &INTERRUPTED)?;
        let analysis = interpret(model, frame, raw)?;
        if let Some(path) = output {
            save_result(&analysis, path).with_context(|| format!("saving {}", path.display()))?;
        }
        if let Some(path) = visualization {
            save_visualization(&analysis, path)
                .with_context(|| format!("saving {}", path.display()))?;
        }
        let raw = &analysis.output;
        match output {
            None => {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                stdout.write_all(&raw.bytes)?;
                if raw.bytes.last() != Some(&b'\n') {
                    stdout.write_all(b"\n")?;
                }
                if let Some(path) = visualization {
                    eprintln!("{}", path.display());
                }
            }
            Some(path) if self.json => print_json(&json!({
                "model": model,
                "job_id": raw.job_id,
                "content_type": raw.content_type,
                "bytes": raw.bytes.len(),
                "elapsed_ms": raw.elapsed.as_millis() as u64,
                "source": {"width": analysis.source.width, "height": analysis.source.height},
                "text": analysis.text,
                "output": path,
                "visualization": visualization,
                "warning": raw.warning,
            }))?,
            Some(path) => {
                if !analysis.text.is_empty() {
                    println!("{}", analysis.text);
                }
                println!("{}", path.display());
                if let Some(path) = visualization {
                    println!("{}", path.display());
                }
            }
        }
        Ok(())
    }

    fn model_watch(&self, args: &ModelWatchArgs) -> Result<()> {
        let ModelWatchArgs {
            model,
            options,
            interval,
            duration,
            output_dir,
            visualization,
            runtime,
            media,
        } = args;
        let options = options.as_ref();
        let output_dir = output_dir.as_deref();
        let visualization = *visualization;
        use std::io::Write;
        let png = is_png_model(model);
        if png && output_dir.is_none() {
            bail!(
                "{model} returns PNG depth maps; give --output-dir DIR (binary is never written to the terminal)"
            );
        }
        crate::preferences::require_ai_enabled("`models watch`")?;
        let client = runtime.client()?;
        let options = merge_options(model, options)?;
        let interval = interval.map_or_else(
            || Duration::from_millis(model_spec(model).map_or(1000, |s| s.interval_ms)),
            Duration::from_secs_f64,
        );
        if let Some(dir) = output_dir {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let b = self.open()?;
        let preview = self.start_preview(&b, media)?;
        install_ctrlc()?;
        let deadline = duration.map(|s| Instant::now() + Duration::from_secs_f64(s));
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let (mut last, mut count, mut next) = (None, 0u64, Instant::now());
        let mut stdout = std::io::stdout().lock();
        loop {
            let frame = loop {
                if INTERRUPTED.load(Ordering::SeqCst)
                    || deadline.is_some_and(|d| Instant::now() >= d)
                {
                    return Ok(());
                }
                if let Some(error) = preview.status().error {
                    bail!("media error: {error}");
                }
                if Instant::now() >= next {
                    if let Some(frame) = preview.latest_frame().filter(|f| Some(f.sequence) != last)
                    {
                        break frame;
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            };
            last = Some(frame.sequence);
            next = Instant::now() + interval;
            let raw = match client.run(model, &options, &frame, &INTERRUPTED) {
                Ok(raw) => raw,
                Err(error) if error.is::<Cancelled>() => return Ok(()),
                Err(error) => return Err(error),
            };
            let analysis = interpret(model, frame, raw)?;
            count += 1;
            let raw = &analysis.output;
            let mut line = json!({
                "sequence": count,
                "model": model,
                "source_sequence": analysis.source.sequence,
                "job_id": raw.job_id,
                "elapsed_ms": raw.elapsed.as_millis() as u64,
                "text": analysis.text,
            });
            match output_dir {
                Some(dir) => {
                    let base = format!("{model}-{stamp}-{count:04}");
                    let path = dir.join(format!("{base}.{}", if png { "png" } else { "json" }));
                    save_result(&analysis, &path)
                        .with_context(|| format!("saving {}", path.display()))?;
                    line["output"] = json!(path);
                    if visualization {
                        let path = dir.join(format!("{base}-viz.png"));
                        save_visualization(&analysis, &path)
                            .with_context(|| format!("saving {}", path.display()))?;
                        line["visualization"] = json!(path);
                    }
                }
                None => {
                    line["result"] = serde_json::from_slice(&raw.bytes)
                        .context("the model result is not valid JSON")?;
                }
            }
            if let Some(warning) = &raw.warning {
                line["warning"] = json!(warning);
            }
            writeln!(stdout, "{line}")?;
            stdout.flush()?;
            if let Some(warning) = &raw.warning {
                bail!("stopping so private jobs do not pile up on the server: {warning}");
            }
        }
    }
}

fn or_unknown(text: &str) -> &str {
    if text.is_empty() { "unknown" } else { text }
}

fn print_one<T: Serialize>(json: bool, value: &T, human: impl FnOnce() -> String) -> Result<()> {
    if json {
        print_json(value)
    } else {
        Ok(println!("{}", human()))
    }
}

fn print_position(b: &CameraBackend, json: bool) -> Result<()> {
    let (pan, tilt) = b
        .position()
        .context("position set, but it cannot be read back")?;
    print_one(json, &[pan, tilt], || format!("pan {pan} tilt {tilt}"))
}

fn refuse_existing(path: &Path) -> Result<()> {
    if path.exists() {
        bail!("{} already exists; refusing to overwrite", path.display());
    }
    Ok(())
}

fn install_ctrlc() -> Result<()> {
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .context("installing the Ctrl-C handler")
}

/// Blocks until Ctrl-C, the duration elapses, or the media layer reports an error.
fn wait(preview: &Preview, duration: Option<f64>) -> Result<()> {
    let deadline = duration.map(|s| Instant::now() + Duration::from_secs_f64(s));
    loop {
        if INTERRUPTED.load(Ordering::SeqCst) || deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(());
        }
        if let Some(error) = preview.status().error {
            bail!("media error: {error}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Both depth models return a PNG; every other model returns JSON.
fn is_png_model(model: &str) -> bool {
    model_spec(model).is_ok_and(|spec| spec.output == "image")
}

/// Fails early before any model loading or camera work when an output cannot be created.
fn check_new_file(path: &Path, extension: &str) -> Result<()> {
    let matches = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(extension));
    if !matches {
        bail!("{} must end in .{extension}", path.display());
    }
    refuse_existing(path)
}

/// Model defaults with the user's top-level keys on top; the client validates the result.
fn merge_options(model: &str, given: Option<&Map<String, Value>>) -> Result<Value> {
    let mut merged = defaults(model)?;
    let base = merged
        .as_object_mut()
        .context("the model defaults are not a JSON object")?;
    if let Some(given) = given {
        base.extend(given.clone());
    }
    Ok(merged)
}

/// Decodes bounded PNG/JPEG input upright, compositing alpha onto a visible white background.
fn load_image(path: &Path) -> Result<Frame> {
    use image::ImageDecoder;
    const MAX_FILE: u64 = 64 << 20;
    const MAX_SIDE: u32 = 8192;
    let meta = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if !meta.is_file() || meta.len() > MAX_FILE {
        bail!("{} is not a regular file of at most 64 MiB", path.display());
    }
    let mut reader = image::ImageReader::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("reading {}", path.display()))?;
    if !matches!(
        reader.format(),
        Some(image::ImageFormat::Png | image::ImageFormat::Jpeg)
    ) {
        bail!("{} is not a PNG or JPEG image", path.display());
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decoding {}", path.display()))?;
    let orientation = decoder
        .orientation()
        .with_context(|| format!("decoding {}", path.display()))?;
    let mut image = image::DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decoding {}", path.display()))?;
    image.apply_orientation(orientation);
    let rgb = if image.color().has_alpha() {
        let mut rgba = image.into_rgba8();
        for pixel in rgba.pixels_mut() {
            let alpha = u16::from(pixel[3]);
            for channel in &mut pixel.0[..3] {
                *channel = ((u16::from(*channel) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
            }
        }
        image::DynamicImage::ImageRgba8(rgba).into_rgb8()
    } else {
        image.into_rgb8()
    };
    Ok(Frame {
        width: rgb.width(),
        height: rgb.height(),
        rgb: rgb.into_raw(),
        sequence: 0,
    })
}

/// Server-supplied text for a terminal: no control characters.
fn plain(text: &str) -> String {
    text.chars().filter(|c| !c.is_control()).collect()
}

fn listed(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(text) => plain(text),
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .map(plain)
            .collect::<Vec<_>>()
            .join("; "),
        _ => String::new(),
    };
    (!text.is_empty()).then_some(text)
}

fn print_model(model: &ModelInfo) {
    let spec = model_spec(&model.id).ok();
    let state = if !model.installed {
        "not downloaded".into()
    } else if model.available == Some(false) {
        format!(
            "unavailable: {}",
            plain(
                model
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("no reason given")
            )
        )
    } else if model.loaded {
        "loaded".into()
    } else if model.loading {
        "loading".into()
    } else {
        "not loaded (loads on first use)".into()
    };
    println!("{}\t{}\t{state}", model.id, spec.map_or("", |s| s.label));
    println!(
        "  device: {} · disk: {} bytes",
        plain(&model.device),
        model.disk_bytes
    );
    let interval = spec.map_or(String::new(), |s| {
        format!(", recommended interval {} ms", s.interval_ms)
    });
    println!(
        "  {} -> {}{interval}",
        plain(&model.input),
        plain(&model.output)
    );
    for (label, key) in [
        ("summary", "summary"),
        ("limitations", "limitations"),
        ("license", "license"),
    ] {
        if let Some(text) = listed(&model.presentation[key]) {
            println!("  {label}: {text}");
        }
    }
    if model
        .option_defaults
        .as_object()
        .is_some_and(|o| !o.is_empty())
    {
        println!("  defaults: {}", model.option_defaults);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> std::result::Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("link-camera").chain(args.iter().copied()))
    }

    #[test]
    fn negative_values_and_trailing_global_flags_parse() {
        let cli = parse(&["set", "hue", "-5", "--device", "/dev/video2"]).unwrap();
        assert_eq!(cli.device.as_deref(), Some("/dev/video2"));
        assert!(matches!(cli.command, Some(Cmd::Set { ref value, .. }) if value == "-5"));
        assert!(parse(&["position", "--pan", "-12.5"]).is_ok());
        assert!(
            parse(&[
                "area", "tracking", "set", "--pan", "-100,100", "--tilt", "-30,30"
            ])
            .is_ok()
        );
    }

    #[test]
    fn invalid_ranges_are_rejected_at_parse_time() {
        assert!(parse(&["preset", "recall", "10"]).is_err());
        assert!(parse(&["snapshot", "a.png", "--blur", "31"]).is_err());
        assert!(parse(&["snapshot", "a.png", "--overlay-opacity", "nan"]).is_err());
        assert!(parse(&["record", "a.mkv", "--duration", "0"]).is_err());
        assert!(
            parse(&["stream"]).is_err(),
            "stream without an output node does nothing"
        );
    }

    #[test]
    fn geometry_arguments_stay_inside_the_frame() {
        assert_eq!(rect_arg("0.1,0.2,0.5,0.5").unwrap(), [0.1, 0.2, 0.5, 0.5]);
        assert!(rect_arg("0.6,0,0.5,0.5").is_err());
        assert!(rect_arg("0,0,0,0.5").is_err());
        assert!(rect_arg("0,0,nan,0.5").is_err());
        assert!(board_arg("0,0,1,0,1,1,0,1").is_ok());
        assert!(board_arg("0,0,1,0,1,1,0,1.5").is_err());
        assert!(board_arg("0,0,1,0").is_err());
    }

    #[test]
    fn remote_off_loopback_needs_a_token() {
        let lan: SocketAddr = "0.0.0.0:9360".parse().unwrap();
        let local: SocketAddr = "127.0.0.1:9360".parse().unwrap();
        assert!(check_listen(lan, None).is_err());
        assert!(check_listen(lan, Some("  ")).is_err());
        assert!(check_listen(lan, Some("secret")).is_ok());
        assert!(check_listen(local, None).is_ok());
    }

    #[test]
    fn models_commands_parse_and_reject_bad_arguments() {
        assert!(
            parse(&["models", "run", "gpt-4"]).is_err(),
            "only the eight models"
        );
        assert!(parse(&["models", "run", "yolo11n", "--options", "[1]"]).is_err());
        assert!(parse(&["models", "run", "yolo11n", "--options", "{"]).is_err());
        assert!(parse(&["models", "run", "yolo11n", "--timeout", "0"]).is_err());
        assert!(parse(&["models", "watch", "yolo11n", "--interval", "-1"]).is_err());
        assert!(parse(&["models", "watch", "yolo11n", "--interval", "nan"]).is_err());
        assert!(
            parse(&["models", "watch", "yolo11n", "--visualization"]).is_err(),
            "visualization files need --output-dir"
        );
    }

    #[test]
    fn depth_models_refuse_terminal_output_before_any_network_or_camera_work() {
        let cli = parse(&[
            "models",
            "run",
            "marigold-depth-v1-1",
            "--input",
            "none.png",
        ])
        .unwrap();
        let error = cli.dispatch(cli.command.as_ref().unwrap()).unwrap_err();
        assert!(error.to_string().contains("--output"), "{error:#}");
        let cli = parse(&["models", "watch", "depth-anything-v2-small"]).unwrap();
        let error = cli.dispatch(cli.command.as_ref().unwrap()).unwrap_err();
        assert!(error.to_string().contains("--output-dir"), "{error:#}");
    }

    #[test]
    fn output_files_need_the_right_extension_and_never_overwrite() {
        let dir = std::env::temp_dir().join(format!("link-camera-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let existing = dir.join("taken.json");
        std::fs::write(&existing, b"{}").unwrap();
        assert!(check_new_file(&existing, "json").is_err());
        assert!(check_new_file(&dir.join("fresh.JSON"), "json").is_ok());
        assert!(check_new_file(&dir.join("fresh.png"), "json").is_err());
        assert!(check_new_file(&dir.join("fresh"), "png").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn transparent_image_input_uses_visible_white_not_hidden_rgb() {
        let dir = std::env::temp_dir().join(format!("link-camera-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("a.png");
        image::RgbaImage::from_pixel(5, 3, image::Rgba([10, 20, 30, 0]))
            .save(&png)
            .unwrap();
        let frame = load_image(&png).unwrap();
        assert!(
            frame
                .rgb
                .chunks_exact(3)
                .all(|pixel| pixel == [255, 255, 255])
        );
        let text = dir.join("a.txt");
        std::fs::write(&text, b"not an image").unwrap();
        assert!(load_image(&text).is_err());
        assert!(load_image(&dir).is_err(), "directories are not images");
        assert!(load_image(&dir.join("missing.png")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

fn virtual_report(report: crate::virtual_camera::Report, json: bool) -> Result<()> {
    if json {
        print_json(&report)?;
    } else {
        for path in &report.stopped {
            println!("stopped {}", path.display());
        }
        for path in &report.removed {
            println!("removed {}", path.display());
        }
        for error in &report.failures {
            eprintln!("{error}");
        }
        println!(
            "{} producers stopped, {} nodes removed",
            report.stopped.len(),
            report.removed.len()
        );
    }
    if !report.failures.is_empty() {
        bail!(
            "virtual camera operation had {} failure(s)",
            report.failures.len()
        );
    }
    Ok(())
}
