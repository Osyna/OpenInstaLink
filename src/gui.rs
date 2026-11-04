//! Slint desktop window. Camera, media, inference and exports run off the UI thread.
//! A deadline-bound subprocess owns the camera; one UI timer polls replies and frames.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use slint::{
    CloseRequestResponse, ComponentHandle, Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer,
    SharedString, VecModel,
};

use crate::audio::{self, CameraMic, Source};
use crate::backend::{
    self, CameraBackend, Control, ControlKind, Device, PauseAreaInfo, PresetInfo, Snapshot,
    TrackingAreaInfo, TrackingCapability, TrackingState,
};
use crate::effects::Effects;
use crate::media::{Frame, MediaStatus, Preview, VideoConfig};
use crate::models::history::{History as ResultHistory, Record as HistoryRecord};
use crate::models::{
    self, Analysis, Client, Engine, EngineStatus, MODEL_IDS, ModelInfo, RuntimeConfig,
};
use crate::preferences;
use crate::virtual_camera::{self, CreateOptions, Elevation, VirtualCamera};

slint::include_modules!();

const BACKGROUNDS: [&str; 4] = ["none", "blur", "replace", "chroma"];
const CONTAINERS: [&str; 2] = ["mkv", "mp4"];
const FRAME_TIMEOUT: Duration = Duration::from_secs(10);
const STALL: Duration = Duration::from_secs(4);
const REFRESH_EVERY: Duration = Duration::from_secs(5);
const SCRIPT_SAVE_AFTER: Duration = Duration::from_secs(1);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_CONTROL_MESSAGE: u64 = 2 << 20;
const AUDIO_REFRESH: Duration = Duration::from_secs(5);

// ---- worker protocol -------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
enum BackendCmd {
    Devices,
    Open { epoch: u64, device: String },
    Close,
    Refresh,
    Set { key: String, value: String },
    ProfileSave(String),
    ProfileLoad(String),
    ProfileDelete(String),
    PresetSave { slot: u8, name: String },
    PresetRecall(u8),
    PresetDelete(u8),
    Recenter,
    AreasLoad,
    AreasSet(Vec<PauseAreaInfo>),
    TrackingLoad,
    TrackingSet(TrackingAreaInfo),
}

#[derive(Default, Serialize, Deserialize)]
enum Commit {
    #[default]
    None,
    Key(String),
    All,
}

/// Exactly one `Reply` per command except `Close`.
#[derive(Default, Serialize, Deserialize)]
struct Reply {
    epoch: u64,
    global: bool,
    opened: bool,
    periodic: bool,
    transport_lost: bool,
    commit: Commit,
    ok: Option<String>,
    err: Option<String>,
    snapshot: Option<Snapshot>,
    devices: Option<Vec<Device>>,
    profiles: Option<Vec<String>>,
    select_profile: Option<String>,
    presets: Option<std::result::Result<Vec<PresetInfo>, String>>,
    areas: Option<std::result::Result<Vec<PauseAreaInfo>, String>>,
    tracking: Option<std::result::Result<TrackingAreaInfo, String>>,
}

impl Reply {
    fn hardware_error(&mut self, error: anyhow::Error) -> String {
        self.transport_lost |= backend::is_transport_failure(&error);
        format!("{error:#}")
    }
}

enum MediaCmd {
    Start {
        epoch: u64,
        device: String,
        config: VideoConfig,
        effects: Option<Effects>,
    },
    Formats {
        epoch: u64,
        device: String,
    },
    Stop(Arc<Preview>),
    Snapshot(Arc<Preview>, PathBuf),
    Record(Arc<Preview>, PathBuf, Option<String>),
    Pause(Arc<Preview>, bool),
    StopRecord(Arc<Preview>),
    Effects {
        preview: Arc<Preview>,
        effects: Effects,
        fallback: Effects,
    },
    Virtual(Arc<Preview>, Option<PathBuf>),
}

#[derive(Default)]
struct MediaReply {
    epoch: u64,
    started: Option<std::result::Result<Arc<Preview>, String>>,
    effects_err: Option<String>,
    effects: Option<Effects>,
    ok: Option<String>,
    err: Option<String>,
    virtual_err: bool,
    formats: Option<std::result::Result<Vec<VideoConfig>, String>>,
}

enum Event {
    Backend(Reply),
    Media(MediaReply),
    Audio(AudioReply),
    Vcam(VcamReply),
    Browse(i32, std::result::Result<Option<String>, String>),
    AiSaved(std::result::Result<String, String>),
    AiHistoryRestored {
        token: u64,
        record: HistoryRecord,
        result: std::result::Result<Arc<Analysis>, String>,
    },
    Library(LibraryDone),
}

// ---- microphone and virtual camera workers ---------------------------------------------------

enum AudioCmd {
    Scan { video: Option<String> },
    Mute { name: String, muted: bool },
    Volume { name: String, percent: u32 },
}

/// Exactly one per command. `set` carries the source name the setter targeted, with its verified
/// readback or the reason it failed.
#[derive(Default)]
struct AudioReply {
    sources: Option<std::result::Result<Vec<Source>, String>>,
    camera: Option<std::result::Result<CameraMic, String>>,
    set: Option<(String, std::result::Result<Source, String>)>,
    /// A setter command was answered (also true for a skipped, superseded volume write).
    setter: bool,
}

/// The audio service is queried on its own thread: pactl calls are bounded but not instant, and
/// the UI must never wait on them. A volume drag keeps only its newest write.
fn audio_worker(commands: Receiver<AudioCmd>, events: Sender<Event>) {
    let mut backlog = VecDeque::new();
    while let Some((command, skipped)) = next_coalesced(
        &commands,
        &mut backlog,
        |now, next| matches!((now, next), (AudioCmd::Volume { name: a, .. }, AudioCmd::Volume { name: b, .. }) if a == b),
    ) {
        for _ in 0..skipped {
            let _ = events.send(Event::Audio(AudioReply {
                setter: true,
                ..AudioReply::default()
            }));
        }
        let text = |e: anyhow::Error| format!("{e:#}");
        let mut r = AudioReply::default();
        match command {
            AudioCmd::Scan { video } => {
                r.sources = Some(audio::sources().map_err(text));
                r.camera = video.map(|node| audio::camera_mic(&node).map_err(text));
            }
            AudioCmd::Mute { name, muted } => {
                let result = audio::set_muted(&name, muted).map_err(text);
                r.set = Some((name, result));
            }
            AudioCmd::Volume { name, percent } => {
                let result = audio::set_volume(&name, percent).map_err(text);
                r.set = Some((name, result));
            }
        }
        r.setter = r.set.is_some();
        if events.send(Event::Audio(r)).is_err() {
            break;
        }
    }
}

enum VcamCmd {
    Refresh,
    Create(String),
    Remove(PathBuf),
    Stop(PathBuf),
}

/// Every command ends with a fresh listing, so the picker always shows what is on the system.
#[derive(Default)]
struct VcamReply {
    list: Option<std::result::Result<Vec<VirtualCamera>, String>>,
    missing: Vec<String>,
    ok: Option<String>,
    err: Option<String>,
    /// Answers the independent stop-all thread, not a queued command.
    stop_all: bool,
}

/// Stop-all runs on its own thread, never behind a queued administrator prompt: it needs no
/// elevation and must end existing outputs at once.
fn stop_all_reply() -> VcamReply {
    let report = virtual_camera::stop_all();
    VcamReply {
        stop_all: true,
        ok: Some(match report.stopped.len() {
            0 => "No Link Camera output was running.".to_owned(),
            1 => "Stopped 1 Link Camera output.".to_owned(),
            n => format!("Stopped {n} Link Camera outputs."),
        }),
        err: (!report.failures.is_empty()).then(|| report.failures.join("\n")),
        list: Some(virtual_camera::list().map_err(|e| format!("{e:#}"))),
        missing: virtual_camera::prerequisites().missing(),
    }
}

/// Creating or removing a camera can wait on a polkit prompt (the helpers time out). The window
/// stays open until every queued command has answered, and this thread is joined at shutdown, so
/// a root helper can never finish after the app is gone and leave an unowned node.
fn vcam_worker(commands: Receiver<VcamCmd>, events: Sender<Event>) {
    while let Ok(command) = commands.recv() {
        let mut r = VcamReply::default();
        let shown = |p: &Path| p.display().to_string();
        match command {
            VcamCmd::Refresh => {}
            VcamCmd::Create(label) => {
                let options = CreateOptions {
                    label,
                    elevation: Elevation::Pkexec,
                    load_module: true,
                };
                match virtual_camera::create(&options) {
                    Ok(cam) => {
                        r.ok = Some(format!("Created {} at {}.", cam.label, shown(&cam.path)))
                    }
                    Err(e) => r.err = Some(format!("{e:#}")),
                }
            }
            VcamCmd::Remove(path) => match virtual_camera::remove(&path, Elevation::Pkexec) {
                Ok(()) => r.ok = Some(format!("Removed {}.", shown(&path))),
                Err(e) => r.err = Some(format!("{e:#}")),
            },
            VcamCmd::Stop(path) => match virtual_camera::stop(&path) {
                Ok(true) => r.ok = Some(format!("Stopped the feed to {}.", shown(&path))),
                Ok(false) => {
                    r.ok = Some(format!(
                        "Nothing from Link Camera was sending to {}.",
                        shown(&path)
                    ))
                }
                Err(e) => r.err = Some(format!("{e:#}")),
            },
        }
        r.list = Some(virtual_camera::list().map_err(|e| format!("{e:#}")));
        r.missing = virtual_camera::prerequisites().missing();
        if events.send(Event::Vcam(r)).is_err() {
            break;
        }
    }
}

/// Result of the model library job (download or delete) that ran on its own thread.
struct LibraryDone {
    model: &'static str,
    download: bool,
    cancelled: bool,
    result: std::result::Result<(), String>,
}

fn explain(error: String) -> String {
    if error.to_lowercase().contains("streaming") {
        format!(
            "{error}. This setting only works while the camera streams: start the preview, then try again."
        )
    } else {
        error
    }
}

// ---- backend worker --------------------------------------------------------------------------

fn finish(camera: &CameraBackend, reply: &mut Reply, result: Result<Option<String>>) {
    if result
        .as_ref()
        .err()
        .is_some_and(backend::is_transport_failure)
    {
        reply.err = result.err().map(|e| reply.hardware_error(e));
        return;
    }
    match camera.snapshot() {
        Ok(snapshot) => reply.snapshot = Some(snapshot),
        Err(e) => reply.err = Some(reply.hardware_error(e.context("cannot re-read the camera"))),
    }
    match result {
        Ok(message) => reply.ok = message,
        Err(e) => {
            let message = reply.hardware_error(e);
            reply.err = Some(match reply.err.take() {
                Some(readback) => format!("{message}; readback failed: {readback}"),
                None => message,
            });
        }
    }
}

fn load_views(camera: &CameraBackend, reply: &mut Reply, snapshot: &Snapshot) {
    reply.presets = Some(camera.presets().map_err(|e| reply.hardware_error(e)));
    if reply.transport_lost {
        return;
    }
    let has = |key: &str| snapshot.controls.iter().any(|c| c.key == key);
    if has("pause-areas") {
        reply.areas = Some(camera.pause_areas().map_err(|e| reply.hardware_error(e)));
        if reply.transport_lost {
            return;
        }
    }
    if has("tracking-area") {
        reply.tracking = Some(camera.tracking_area().map_err(|e| reply.hardware_error(e)));
    }
}

/// Next queued command. A write that a later queued write to the same target replaces is
/// answered with an empty reply and skipped, so dragging a control never leaves a long
/// backlog of hardware writes behind.
fn next_coalesced<C>(
    commands: &Receiver<C>,
    backlog: &mut VecDeque<C>,
    superseded: impl Fn(&C, &C) -> bool,
) -> Option<(C, usize)> {
    let mut command = match backlog.pop_front() {
        Some(c) => c,
        None => commands.recv().ok()?,
    };
    backlog.extend(commands.try_iter());
    let mut skipped = 0;
    while backlog
        .front()
        .is_some_and(|next| superseded(&command, next))
    {
        skipped += 1;
        command = backlog.pop_front()?;
    }
    Some((command, skipped))
}

fn read_control_message<T: DeserializeOwned>(reader: &mut impl BufRead) -> Result<Option<T>> {
    crate::process::read_message(reader, MAX_CONTROL_MESSAGE)
}

struct CameraProcess {
    child: Option<Child>,
    input: Option<BufWriter<ChildStdin>>,
    replies: Receiver<std::result::Result<Reply, String>>,
    quarantined: bool,
}

impl CameraProcess {
    fn start() -> Result<Self> {
        // /proc/self/exe keeps the worker on this binary's version across atomic app updates.
        let mut child = Command::new("/proc/self/exe")
            .arg("__camera-worker")
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting the camera control worker")?;
        let stdout = child.stdout.take().expect("piped worker stdout");
        let input = BufWriter::new(child.stdin.take().expect("piped worker stdin"));
        let (tx, replies) = mpsc::channel();
        let process = Self {
            child: Some(child),
            input: Some(input),
            replies,
            quarantined: false,
        };
        std::thread::Builder::new()
            .name("camera-replies".into())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                loop {
                    let reply = match read_control_message(&mut reader) {
                        Ok(Some(reply)) => Ok(reply),
                        Ok(None) => Err("camera control worker exited before replying".into()),
                        Err(error) => Err(format!("invalid camera control reply: {error:#}")),
                    };
                    let failed = reply.is_err();
                    if tx.send(reply).is_err() || failed {
                        break;
                    }
                }
            })
            .context("starting the camera reply reader")?;
        Ok(process)
    }

    fn request(&mut self, command: &BackendCmd, timeout: Duration) -> Result<Option<Reply>> {
        ensure!(
            !self.quarantined,
            "the previous camera request is still blocked in the kernel; unplug the camera before reconnecting"
        );
        let input = self
            .input
            .as_mut()
            .context("camera worker input is closed")?;
        serde_json::to_writer(&mut *input, command)?;
        input.write_all(b"\n")?;
        input.flush()?;
        if matches!(command, BackendCmd::Close) {
            return Ok(None);
        }
        match self.replies.recv_timeout(timeout) {
            Ok(Ok(reply)) => Ok(Some(reply)),
            Ok(Err(error)) => bail!("{error}"),
            Err(mpsc::RecvTimeoutError::Disconnected) => bail!("camera reply reader stopped"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.stop();
                bail!(
                    "camera controls did not answer within {} seconds; worker cancellation was requested, but kernel USB I/O may still be pending. Any setting being applied has an unknown outcome and was not retried",
                    timeout.as_secs()
                );
            }
        }
    }

    fn stop(&mut self) {
        if self.quarantined {
            return;
        }
        self.quarantined = true;
        if let Some(child) = self.child.as_mut() {
            if let Err(error) = crate::process::stop_group(child) {
                eprintln!("link-camera: cannot stop the camera worker group: {error}");
            }
        }
        self.input = None;
    }

    fn reaped(&mut self) -> Result<bool> {
        if self
            .child
            .as_mut()
            .context("camera worker is missing")?
            .try_wait()?
            .is_some()
        {
            self.child = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Drop for CameraProcess {
    fn drop(&mut self) {
        self.stop();
        if let Some(mut child) = self.child.take() {
            // A driver in uninterruptible sleep must not hold window shutdown hostage.
            let _ = std::thread::Builder::new()
                .name("camera-reap".into())
                .spawn(move || {
                    let _ = child.wait();
                });
        }
    }
}

/// Private stdio entry point; it never initializes Slint or starts a video stream.
pub(crate) fn camera_worker_process() -> Result<()> {
    let (tx, commands) = mpsc::channel();
    let (events, replies) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("camera-controls".into())
        .spawn(move || backend_worker_in_process(commands, events))?;
    let mut input = BufReader::new(std::io::stdin().lock());
    let mut output = BufWriter::new(std::io::stdout().lock());
    while let Some(command) = read_control_message::<BackendCmd>(&mut input)? {
        let close = matches!(command, BackendCmd::Close);
        tx.send(command).context("camera worker stopped")?;
        if !close {
            let Event::Backend(reply) = replies
                .recv()
                .context("camera worker stopped before replying")?
            else {
                bail!("unexpected camera worker event");
            };
            serde_json::to_writer(&mut output, &reply)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    drop(tx);
    worker
        .join()
        .map_err(|_| anyhow!("camera control worker panicked"))?;
    Ok(())
}

fn backend_worker(commands: Receiver<BackendCmd>, events: Sender<Event>) {
    let mut worker: Option<CameraProcess> = None;
    let mut epoch = 0;
    let mut backlog = VecDeque::new();
    while let Some((command, skipped)) = next_coalesced(
        &commands,
        &mut backlog,
        |now, next| matches!((now, next), (BackendCmd::Set { key: a, .. }, BackendCmd::Set { key: b, .. }) if a == b),
    ) {
        for _ in 0..skipped {
            let _ = events.send(Event::Backend(Reply {
                epoch,
                ..Reply::default()
            }));
        }
        if let BackendCmd::Open { epoch: next, .. } = &command {
            epoch = *next;
        }
        let result = (|| -> Result<Option<Reply>> {
            if let Some(process) = worker.as_mut().filter(|p| p.quarantined) {
                if process.reaped()? {
                    worker = None;
                } else {
                    bail!(
                        "the previous camera request is still blocked in the kernel; unplug the camera before reconnecting"
                    );
                }
            }
            if worker.is_none() {
                if matches!(command, BackendCmd::Close) {
                    return Ok(None);
                }
                worker = Some(CameraProcess::start()?);
            }
            worker
                .as_mut()
                .expect("started camera worker")
                .request(&command, CONTROL_TIMEOUT)
        })();
        let reply = match result {
            Ok(Some(reply)) => reply,
            Ok(None) => continue,
            Err(error) => {
                if let Some(process) = worker.as_mut() {
                    process.stop();
                }
                if matches!(command, BackendCmd::Close) {
                    continue;
                }
                Reply {
                    epoch,
                    global: matches!(command, BackendCmd::Devices | BackendCmd::ProfileDelete(_)),
                    opened: matches!(command, BackendCmd::Open { .. }),
                    periodic: matches!(command, BackendCmd::Refresh),
                    transport_lost: true,
                    err: Some(format!("{error:#}")),
                    ..Reply::default()
                }
            }
        };
        if events.send(Event::Backend(reply)).is_err() {
            break;
        }
    }
}

fn backend_worker_in_process(commands: Receiver<BackendCmd>, events: Sender<Event>) {
    let mut camera: Option<CameraBackend> = None;
    let mut epoch = 0u64;
    let mut backlog = VecDeque::new();
    while let Some((command, skipped)) = next_coalesced(
        &commands,
        &mut backlog,
        |now, next| matches!((now, next), (BackendCmd::Set { key: a, .. }, BackendCmd::Set { key: b, .. }) if a == b),
    ) {
        for _ in 0..skipped {
            let _ = events.send(Event::Backend(Reply {
                epoch,
                ..Reply::default()
            }));
        }
        let mut r = Reply {
            epoch,
            periodic: matches!(&command, BackendCmd::Refresh),
            ..Reply::default()
        };
        let changed = matches!(
            &command,
            BackendCmd::Set { .. }
                | BackendCmd::Recenter
                | BackendCmd::ProfileLoad(_)
                | BackendCmd::PresetRecall(_)
                | BackendCmd::AreasSet(_)
                | BackendCmd::TrackingSet(_)
        );
        match command {
            BackendCmd::Close => {
                camera = None;
                continue;
            }
            BackendCmd::Devices => {
                r.global = true;
                match backend::devices() {
                    Ok(list) => r.devices = Some(list),
                    Err(e) => r.err = Some(format!("{e:#}")),
                }
                r.profiles = backend::profiles().ok();
            }
            BackendCmd::Open {
                epoch: new_epoch,
                device,
            } => {
                camera = None;
                epoch = new_epoch;
                r.epoch = epoch;
                r.opened = true;
                r.profiles = backend::profiles().ok();
                match CameraBackend::open(Some(&device)).and_then(|c| {
                    c.restore_saved_settings()
                        .context("restoring saved camera settings")?;
                    c.snapshot().map(|s| (c, s))
                }) {
                    Ok((opened, snapshot)) => {
                        load_views(&opened, &mut r, &snapshot);
                        r.snapshot = Some(snapshot);
                        camera = Some(opened);
                    }
                    Err(e) => r.err = Some(r.hardware_error(e)),
                }
            }
            BackendCmd::ProfileDelete(name) => {
                r.global = true;
                match backend::delete_profile(&name) {
                    Ok(()) => r.ok = Some(format!("Deleted profile {name}")),
                    Err(e) => r.err = Some(format!("{e:#}")),
                }
                r.profiles = backend::profiles().ok();
            }
            other => match camera.as_ref() {
                None => {
                    r.err =
                        Some("No camera is connected. Start the preview or press Reconnect.".into())
                }
                Some(c) => run_camera_command(c, other, &mut r),
            },
        }
        if changed && r.err.is_none() {
            if let Err(error) = crate::restore::ensure_user_service() {
                let warning = format!(
                    "Settings saved, but automatic restoration for other apps is unavailable: {error:#}"
                );
                if let Some(snapshot) = r.snapshot.as_mut() {
                    snapshot.warnings.push(warning);
                } else {
                    r.ok = Some(warning);
                }
            }
        }
        if r.transport_lost {
            camera = None;
            r.snapshot = None;
            if r.err.is_none() {
                r.err = Some("the camera stopped answering USB control requests; see the control warning for details".into());
            }
        }
        if events.send(Event::Backend(r)).is_err() {
            break;
        }
    }
}

fn run_camera_command(c: &CameraBackend, command: BackendCmd, r: &mut Reply) {
    match command {
        BackendCmd::Refresh => match c.snapshot() {
            Ok(s) => r.snapshot = Some(s),
            Err(e) => r.err = Some(r.hardware_error(e)),
        },
        BackendCmd::Set { key, value } => {
            let result = c.set(&key, &value).map(|()| None);
            r.commit = Commit::Key(key);
            finish(c, r, result);
        }
        BackendCmd::ProfileSave(name) => {
            let result = c
                .save_profile(&name)
                .map(|()| Some(format!("Saved profile {name}")));
            r.select_profile = Some(name);
            r.profiles = backend::profiles().ok();
            match result {
                Ok(m) => r.ok = m,
                Err(e) => r.err = Some(r.hardware_error(e)),
            }
        }
        BackendCmd::ProfileLoad(name) => {
            let result = c
                .load_profile(&name)
                .map(|()| Some(format!("Applied profile {name}")));
            r.commit = Commit::All;
            finish(c, r, result);
        }
        BackendCmd::PresetSave { slot, name } => {
            let result = c
                .save_preset(slot, &name)
                .map(|_| Some(format!("Saved view {name} in slot {slot}")));
            r.commit = Commit::All;
            finish(c, r, result);
            if !r.transport_lost {
                r.presets = Some(c.presets().map_err(|e| r.hardware_error(e)));
            }
        }
        BackendCmd::PresetRecall(slot) => {
            let result = c
                .recall_preset(slot)
                .map(|p| Some(format!("Recalled view {} (slot {slot})", p.name)));
            r.commit = Commit::All;
            finish(c, r, result);
        }
        BackendCmd::PresetDelete(slot) => {
            let result = c
                .delete_preset(slot)
                .map(|()| Some(format!("Deleted view in slot {slot}")));
            finish(c, r, result);
            if !r.transport_lost {
                r.presets = Some(c.presets().map_err(|e| r.hardware_error(e)));
            }
        }
        BackendCmd::Recenter => {
            let result = c.recenter().map(|()| Some("View centered".to_owned()));
            r.commit = Commit::All;
            finish(c, r, result);
        }
        BackendCmd::AreasLoad => r.areas = Some(c.pause_areas().map_err(|e| r.hardware_error(e))),
        BackendCmd::AreasSet(areas) => {
            match c.set_pause_areas(&areas) {
                Ok(()) => r.ok = Some("Pause-tracking zones applied".into()),
                Err(e) => r.err = Some(r.hardware_error(e)),
            }
            if !r.transport_lost {
                r.areas = Some(c.pause_areas().map_err(|e| r.hardware_error(e)));
            }
        }
        BackendCmd::TrackingLoad => {
            r.tracking = Some(c.tracking_area().map_err(|e| r.hardware_error(e)))
        }
        BackendCmd::TrackingSet(area) => {
            match c.set_tracking_area(&area) {
                Ok(()) => r.ok = Some("Tracking area applied".into()),
                Err(e) => r.err = Some(r.hardware_error(e)),
            }
            if !r.transport_lost {
                r.tracking = Some(c.tracking_area().map_err(|e| r.hardware_error(e)));
            }
        }
        BackendCmd::Devices
        | BackendCmd::Open { .. }
        | BackendCmd::Close
        | BackendCmd::ProfileDelete(_) => {}
    }
}

// ---- media worker ----------------------------------------------------------------------------

fn media_worker(commands: Receiver<MediaCmd>, events: Sender<Event>) {
    let mut backlog = VecDeque::new();
    while let Some((command, skipped)) = next_coalesced(
        &commands,
        &mut backlog,
        |now, next| matches!((now, next), (MediaCmd::Effects { preview: a, .. }, MediaCmd::Effects { preview: b, .. }) if Arc::ptr_eq(a, b)),
    ) {
        for _ in 0..skipped {
            let _ = events.send(Event::Media(MediaReply::default()));
        }
        let mut r = MediaReply::default();
        let fail = |e: anyhow::Error| Some(format!("{e:#}"));
        match command {
            MediaCmd::Formats { epoch, device } => {
                r.epoch = epoch;
                r.formats = Some(crate::media::formats(&device).map_err(|e| format!("{e:#}")));
            }
            MediaCmd::Start {
                epoch,
                device,
                config,
                effects,
            } => {
                r.epoch = epoch;
                let started = Preview::start(&device, &config)
                    .and_then(|p| p.wait_first_frame(FRAME_TIMEOUT).map(|_| p));
                r.started = Some(match started {
                    Ok(preview) => {
                        if let Some(effects) = effects {
                            match preview.set_effects(effects.clone()) {
                                Ok(()) => r.effects = Some(effects),
                                Err(error) => r.effects_err = fail(error),
                            }
                        }
                        Ok(Arc::new(preview))
                    }
                    Err(e) => Err(format!("{e:#}")),
                });
            }
            MediaCmd::Stop(preview) => drop(preview),
            MediaCmd::Snapshot(preview, path) => match preview.snapshot(&path) {
                Ok(()) => r.ok = Some(format!("Saved snapshot {}", path.display())),
                Err(e) => r.err = fail(e),
            },
            MediaCmd::Record(preview, path, audio) => {
                match preview.start_recording(&path, audio.as_deref()) {
                    Ok(()) => r.ok = Some(format!("Recording to {}", path.display())),
                    Err(e) => r.err = fail(e),
                }
            }
            MediaCmd::Pause(preview, paused) => match preview.pause_recording(paused) {
                Ok(()) => {
                    r.ok = Some(
                        if paused {
                            "Recording paused"
                        } else {
                            "Recording resumed"
                        }
                        .into(),
                    )
                }
                Err(e) => r.err = fail(e),
            },
            MediaCmd::StopRecord(preview) => match preview.stop_recording() {
                Ok(()) => r.ok = Some("Recording saved".into()),
                Err(e) => r.err = fail(e),
            },
            MediaCmd::Effects {
                preview,
                effects,
                fallback,
            } => match preview.set_effects(effects.clone()) {
                Ok(()) => r.effects = Some(effects),
                Err(e) => {
                    r.effects = Some(fallback);
                    r.err = fail(e);
                }
            },
            MediaCmd::Virtual(preview, path) => {
                let started = path.is_some();
                match preview.set_virtual_output(path.as_deref()) {
                    Ok(()) => {
                        r.ok = Some(
                            if started {
                                "Virtual camera started"
                            } else {
                                "Virtual camera stopped"
                            }
                            .into(),
                        )
                    }
                    Err(e) => {
                        r.virtual_err = true;
                        r.err = fail(e);
                    }
                }
            }
        }
        if events.send(Event::Media(r)).is_err() {
            break;
        }
    }
}

// ---- pure helpers ----------------------------------------------------------------------------

fn humanize(text: &str) -> String {
    let spaced = text.replace(['-', '_'], " ");
    let mut chars = spaced.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

fn kind_id(control: &Control) -> i32 {
    match control.kind {
        ControlKind::Number => 0,
        ControlKind::Toggle => 1,
        ControlKind::Choice => 2,
    }
}

fn same_mode(a: &VideoConfig, b: &VideoConfig) -> bool {
    a.width == b.width && a.height == b.height && a.fps == b.fps
}

/// The current mode if the camera reports it, otherwise the reported mode nearest 1280x720 at 30 fps.
fn choose_mode(modes: &[VideoConfig], current: &VideoConfig) -> Option<VideoConfig> {
    modes
        .iter()
        .find(|m| same_mode(m, current))
        .cloned()
        .or_else(|| {
            modes
                .iter()
                .min_by_key(|m| {
                    (
                        (i64::from(m.width) * i64::from(m.height) - 921_600).abs(),
                        (i64::from(m.fps) - 30).abs(),
                    )
                })
                .cloned()
        })
}

fn resolutions_of(modes: &[VideoConfig]) -> Vec<(u32, u32)> {
    let mut sizes: Vec<(u32, u32)> = modes.iter().map(|m| (m.width, m.height)).collect();
    sizes.sort_by_key(|&(w, h)| (u64::from(w) * u64::from(h), w));
    sizes.dedup();
    sizes
}

fn fps_of(modes: &[VideoConfig], width: u32, height: u32) -> Vec<u32> {
    let mut rates: Vec<u32> = modes
        .iter()
        .filter(|m| m.width == width && m.height == height)
        .map(|m| m.fps)
        .collect();
    rates.sort_unstable();
    rates.dedup();
    rates
}

fn same_control(a: &Control, b: &Control) -> bool {
    a.key == b.key
        && a.label == b.label
        && a.group == b.group
        && kind_id(a) == kind_id(b)
        && a.value == b.value
        && a.min == b.min
        && a.max == b.max
        && a.step == b.step
        && a.options == b.options
        && a.default == b.default
}

/// The hardware default of a ranged number control, only when the camera reported one that lies
/// inside the range. Never invented: no default means no reset.
fn default_of(c: &Control) -> Option<f64> {
    c.default.filter(|d| {
        matches!(c.kind, ControlKind::Number) && c.max > c.min && (c.min..=c.max).contains(d)
    })
}

fn decimals(step: f64) -> usize {
    if step <= 0.0 {
        2
    } else {
        format!("{step}")
            .split_once('.')
            .map_or(0, |(_, f)| f.len())
            .min(4)
    }
}

/// Slider value snapped to the control's step grid, as the text the backend accepts.
fn slider_text(value: f64, c: &Control) -> String {
    let mut v = value;
    if c.step > 0.0 {
        v = c.min + ((v - c.min) / c.step).round() * c.step;
    }
    v = v.clamp(c.min, c.max);
    let text = format!("{:.*}", decimals(c.step), v);
    if text
        .trim_start_matches('-')
        .chars()
        .all(|ch| ch == '0' || ch == '.')
    {
        text.trim_start_matches('-').to_owned()
    } else {
        text
    }
}

fn build_rows(controls: &[Control], revs: &[i32]) -> Vec<ControlRow> {
    controls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let heading = if i == 0 || controls[i - 1].group != c.group {
                c.group.clone()
            } else {
                String::new()
            };
            let options: Vec<SharedString> = c.options.iter().map(|o| humanize(o).into()).collect();
            ControlRow {
                key: c.key.clone().into(),
                label: c.label.clone().into(),
                heading: heading.into(),
                kind: kind_id(c),
                text: if matches!(c.kind, ControlKind::Number) {
                    c.value.clone().into()
                } else {
                    SharedString::new()
                },
                num: c.value.trim().parse::<f32>().unwrap_or(c.min as f32),
                lo: c.min as f32,
                hi: c.max as f32,
                inc: c.step as f32,
                has_def: default_of(c).is_some(),
                def: default_of(c).unwrap_or(0.0) as f32,
                checked: c.value == "true",
                options: ModelRc::new(VecModel::from(options)),
                sel: c
                    .options
                    .iter()
                    .position(|o| *o == c.value)
                    .map_or(-1, |p| p as i32),
                rev: revs[i],
            }
        })
        .collect()
}

fn trim_num(v: f32) -> String {
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

fn parse_unit(text: &str, what: &str) -> std::result::Result<f32, String> {
    let v: f32 = text
        .trim()
        .parse()
        .map_err(|_| format!("{what}: `{}` is not a number", text.trim()))?;
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{what} must be between 0 and 1"))
    }
}

fn parse_f64(text: &str, what: &str) -> std::result::Result<f64, String> {
    let v: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("{what}: `{}` is not a number", text.trim()))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("{what} must be a finite number"))
    }
}

fn effects_active(e: &Effects) -> bool {
    (!e.background.is_empty() && e.background != "none")
        || e.soften > 0.0
        || e.mirror
        || e.overlay_image.is_some()
        || e.board.is_some()
}

fn initial_effects() -> Effects {
    Effects {
        background: "none".into(),
        blur: 10.0,
        overlay_opacity: 1.0,
        ..Effects::default()
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
}

fn user_dir(name: &str) -> PathBuf {
    let dir = home().join(name);
    if dir.is_dir() { dir } else { home() }
}

fn script_store() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home().join(".config"));
    base.join("link-camera/teleprompter.txt")
}

/// First `link-camera-NNNN.ext` that does not exist yet, so nothing is ever overwritten.
fn next_path(dir: &str, ext: &str) -> std::result::Result<PathBuf, String> {
    let dir = dir.trim();
    if dir.is_empty() {
        return Err("Choose an output folder on the Output tab first.".into());
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Cannot use folder {}: {e}", dir.display()))?;
    (1..10_000)
        .map(|n| dir.join(format!("link-camera-{n:04}.{ext}")))
        .find(|p| !p.exists())
        .ok_or_else(|| {
            format!(
                "Folder {} already holds 9999 captures; choose another folder.",
                dir.display()
            )
        })
}

fn write_script(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)
}

fn read_script(path: &Path) -> std::result::Result<String, String> {
    const MAX: u64 = 1 << 20;
    let meta =
        std::fs::metadata(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() > MAX {
        return Err(format!(
            "{} is not a regular text file under 1 MiB.",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8 text.", path.display()))
}

fn choose_path(target: i32) -> std::result::Result<Option<String>, String> {
    let directory = matches!(target, 2 | 3 | 5);
    let attempts: [(&str, Vec<&str>); 2] = [
        (
            "zenity",
            if directory {
                vec!["--file-selection", "--directory"]
            } else {
                vec!["--file-selection"]
            },
        ),
        (
            "kdialog",
            vec![if directory {
                "--getexistingdirectory"
            } else {
                "--getopenfilename"
            }],
        ),
    ];
    for (program, args) in attempts {
        match std::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
        {
            Ok(out) if out.status.success() => {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_owned();
                return Ok((!path.is_empty()).then_some(path));
            }
            Ok(_) => return Ok(None), // cancelled
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot run {program}: {e}")),
        }
    }
    Err("No file chooser found (install zenity or kdialog); type the path instead.".into())
}

#[derive(Clone, Default)]
struct Zone {
    active: bool,
    x: String,
    y: String,
    w: String,
    h: String,
    pan: String,
    tilt: String,
}

impl Zone {
    fn row(&self) -> AreaRow {
        AreaRow {
            active: self.active,
            x: self.x.clone().into(),
            y: self.y.clone().into(),
            w: self.w.clone().into(),
            h: self.h.clone().into(),
            pan: self.pan.clone().into(),
            tilt: self.tilt.clone().into(),
        }
    }
}

// ---- AI models: pure helpers -----------------------------------------------------------------

/// Local inference has no network round trip: look at the engine at about frame cadence.
const AI_POLL: Duration = Duration::from_millis(16);
/// The SAM model accepts at most 16 positive points.
const SAM_MAX_POINTS: usize = 16;
const FLORENCE_TASKS: [&str; 4] = ["caption", "detect", "ocr", "regions"];
const COLORMAPS: [&str; 2] = ["gray", "inferno"];
/// Smallest drag, in display pixels, that counts as drawing a box rather than a click.
const MIN_BOX_DRAG: f32 = 6.0;

/// Which option group the UI shows: 0 hands/pose, 1 face, 2 YOLO, 3 Florence, 4 SAM,
/// 5 Depth Anything, 6 Marigold.
fn model_kind(id: &str) -> i32 {
    match id {
        "mediapipe-face" => 1,
        "yolo11n" => 2,
        "florence-2-base" => 3,
        "sam2.1-tiny" => 4,
        "depth-anything-v2-small" => 5,
        "marigold-depth-v1-1" => 6,
        _ => 0,
    }
}

fn model_label(id: &str) -> String {
    models::model_spec(id).map_or_else(|_| id.to_owned(), |s| s.label.to_owned())
}

/// Whether two catalog reads show the same library state. The engine re-reads the catalog after
/// every run, and a live session must not rebuild the model picker 30 times a second.
fn same_library(a: &[ModelInfo], b: &[ModelInfo]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.id == y.id
                && x.installed == y.installed
                && x.loaded == y.loaded
                && x.loading == y.loading
                && x.available == y.available
                && x.unavailable_reason == y.unavailable_reason
                && x.disk_bytes == y.disk_bytes
                && x.device == y.device
        })
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn text_of(value: &serde_json::Value) -> Option<String> {
    let text = match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join("; "),
        _ => return None,
    };
    let text = text.trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Maps a pointer position over an `Image` drawn with `image-fit: contain` in a `view` of
/// that exact size to image pixels. A position in the letterbox margin is `None`, or with
/// `clamp` pinned to the nearest image edge.
fn contain_to_pixel(
    view: (f32, f32),
    image: (u32, u32),
    at: (f32, f32),
    clamp: bool,
) -> Option<(f64, f64)> {
    let (vw, vh) = (f64::from(view.0), f64::from(view.1));
    let (iw, ih) = (f64::from(image.0), f64::from(image.1));
    let (ax, ay) = (f64::from(at.0), f64::from(at.1));
    if !(vw > 0.0 && vh > 0.0 && iw > 0.0 && ih > 0.0 && ax.is_finite() && ay.is_finite()) {
        return None;
    }
    let scale = (vw / iw).min(vh / ih);
    let x = (ax - (vw - iw * scale) / 2.0) / scale;
    let y = (ay - (vh - ih * scale) / 2.0) / scale;
    let inside = (0.0..=iw).contains(&x) && (0.0..=ih).contains(&y);
    (inside || clamp).then(|| (x.clamp(0.0, iw), y.clamp(0.0, ih)))
}

fn parse_coord(text: &str, what: &str) -> std::result::Result<f64, String> {
    match text.trim().parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
        _ => Err(format!(
            "{what}: `{}` is not a non-negative number.",
            text.trim()
        )),
    }
}

fn parse_point(token: &str) -> std::result::Result<(f64, f64), String> {
    match token.split(',').collect::<Vec<_>>().as_slice() {
        [x, y] => Ok((parse_coord(x, "Point")?, parse_coord(y, "Point")?)),
        _ => Err(format!("Point `{}` must look like x,y.", token.trim())),
    }
}

fn point_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split([';', '\n'])
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

fn parse_points(text: &str) -> std::result::Result<Vec<(f64, f64)>, String> {
    let points = point_tokens(text)
        .map(parse_point)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if points.len() > SAM_MAX_POINTS {
        return Err(format!("At most {SAM_MAX_POINTS} points are allowed."));
    }
    Ok(points)
}

/// Points that parse, for drawing markers while the user is still typing.
fn lenient_points(text: &str) -> Vec<(f64, f64)> {
    point_tokens(text)
        .filter_map(|t| parse_point(t).ok())
        .collect()
}

/// `x1,y1,x2,y2` normalised to top-left then bottom-right; empty text means no box.
fn parse_box(text: &str) -> std::result::Result<Option<[f64; 4]>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let numbers = text
        .split(',')
        .map(|p| parse_coord(p, "Box"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let [x1, y1, x2, y2] = match numbers.as_slice() {
        [a, b, c, d] => [*a, *b, *c, *d],
        _ => return Err("Box must look like x1,y1,x2,y2.".into()),
    };
    let (left, right, top, bottom) = (x1.min(x2), x1.max(x2), y1.min(y2), y1.max(y2));
    if right <= left || bottom <= top {
        return Err("Box must have a positive width and height.".into());
    }
    Ok(Some([left, top, right, bottom]))
}

fn fmt_coord(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        trim_num(v as f32)
    }
}

fn plot(rgb: &mut [u8], size: (i64, i64), x: i64, y: i64, color: [u8; 3]) {
    if (0..size.0).contains(&x)
        && (0..size.1).contains(&y)
        && let Some(px) = rgb.get_mut(((y * size.0 + x) * 3) as usize..)
        && px.len() >= 3
    {
        px[..3].copy_from_slice(&color);
    }
}

/// Draws SAM prompts (point crosses, box outline) into packed RGB at source resolution.
fn draw_prompts(
    rgb: &mut [u8],
    width: u32,
    height: u32,
    points: &[(f64, f64)],
    rect: Option<[f64; 4]>,
) {
    let size = (i64::from(width), i64::from(height));
    let arm = (size.0.max(size.1) / 80).max(6);
    let inner = (arm / 6).max(1);
    for (half, color) in [(inner + 1, [0, 0, 0]), (inner, [0, 255, 64])] {
        let reach = arm + half - inner;
        for &(px, py) in points {
            let (cx, cy) = (px as i64, py as i64);
            for d in -reach..=reach {
                for t in -half..=half {
                    plot(rgb, size, cx + d, cy + t, color);
                    plot(rgb, size, cx + t, cy + d, color);
                }
            }
        }
        if let Some([x1, y1, x2, y2]) = rect {
            let (x1, y1, x2, y2) = (x1 as i64, y1 as i64, x2 as i64, y2 as i64);
            for t in -half..=half {
                for x in (x1 - half)..=(x2 + half) {
                    plot(rgb, size, x, y1 + t, color);
                    plot(rgb, size, x, y2 + t, color);
                }
                for y in (y1 - half)..=(y2 + half) {
                    plot(rgb, size, x1 + t, y, color);
                    plot(rgb, size, x2 + t, y, color);
                }
            }
        }
    }
}

fn frame_ok(frame: &Frame) -> bool {
    frame.width > 0
        && frame.height > 0
        && frame.rgb.len() == frame.width as usize * frame.height as usize * 3
}

fn rgb_image(width: u32, height: u32, rgb: &[u8]) -> Option<Image> {
    (width > 0 && height > 0 && rgb.len() == width as usize * height as usize * 3).then(|| {
        let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
        buffer.make_mut_bytes().copy_from_slice(rgb);
        Image::from_rgb8(buffer)
    })
}

/// The API result as shown in the read-only raw view.
fn raw_view(content_type: &str, bytes: &[u8]) -> String {
    if content_type.contains("json") {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .and_then(|v| serde_json::to_string_pretty(&v))
            .unwrap_or_else(|_| String::from_utf8_lossy(bytes).into_owned())
    } else if content_type.contains("png") {
        format!(
            "PNG image, {} bytes. Relative depth: near is bright; not a metric distance. Use Save raw result to keep the file.",
            bytes.len()
        )
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// First `link-camera-<model>-NNNN<suffix>.<ext>` that does not exist yet.
fn next_model_path(
    dir: &str,
    model: &str,
    suffix: &str,
    ext: &str,
) -> std::result::Result<PathBuf, String> {
    let dir = dir.trim();
    if dir.is_empty() {
        return Err("Choose a folder for AI results first.".into());
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Cannot use folder {}: {e}", dir.display()))?;
    (1..10_000)
        .map(|n| dir.join(format!("link-camera-{model}-{n:04}{suffix}.{ext}")))
        .find(|p| !p.exists())
        .ok_or_else(|| {
            format!(
                "Folder {} already holds 9999 results for {model}; choose another folder.",
                dir.display()
            )
        })
}

enum ModelState {
    /// Optional AI is off, or the local library has not been read.
    Off,
    Checking,
    Missing,
    NotInstalled,
    /// Installed, but the local runtime cannot run it.
    Unavailable(String),
    Ready {
        loaded: bool,
        loading: bool,
    },
}

fn state_tag(state: &ModelState) -> &'static str {
    match state {
        ModelState::Off => "",
        ModelState::Checking => "checking",
        ModelState::Missing => "not listed",
        ModelState::NotInstalled => "not installed",
        ModelState::Unavailable(_) => "cannot run",
        ModelState::Ready { loading: true, .. } => "loading",
        ModelState::Ready { loaded: true, .. } => "loaded",
        ModelState::Ready { .. } => "installed",
    }
}

/// A download or delete running on its own thread. The engine is dropped while it runs, so no
/// model process holds the files and nothing can run half-installed.
struct LibraryJob {
    model: &'static str,
    download: bool,
    cancel: Arc<AtomicBool>,
    started: Instant,
    handle: JoinHandle<()>,
}

/// The request the engine is working on. A reply is only shown while its token is current.
struct Inflight {
    token: u64,
    model: &'static str,
    seq: u64,
}

#[derive(Default)]
struct Ai {
    engine: Option<Engine>,
    status: EngineStatus,
    /// Bumped by every invalidation; nothing from an older token is ever displayed.
    token: u64,
    inflight: Option<Inflight>,
    result: Option<Arc<Analysis>>,
    result_info: String,
    latest: Option<Arc<Analysis>>,
    latest_record: Option<HistoryRecord>,
    history: ResultHistory,
    history_rows: Rc<VecModel<HistoryRow>>,
    history_selected: Option<HistoryRecord>,
    history_paused: bool,
    history_view_token: u64,
    history_loading: Option<JoinHandle<()>>,
    history_pending: Option<(u64, HistoryRecord)>,
    completed: u64,
    last_error: Option<String>,
    live: bool,
    library: Option<LibraryJob>,
    lib_note: (String, bool),
    closing: bool,
    /// Show the next published result on the stage (the user just asked for one).
    follow: bool,
    /// An effects change that moves pixels was requested and has not reached the preview yet.
    geometry_pending: bool,
    model: usize,
    stage: i32,
    /// Stage to return to when this analysis session ends.
    prior_stage: Option<i32>,
    frozen: Option<Arc<Frame>>,
    /// (token, frame sequence, options) of the last submission; live only submits something new.
    last_key: Option<(u64, u64, String)>,
    next_at: Option<Instant>,
    last_poll: Option<Instant>,
    note: (String, bool),
    saving: Vec<JoinHandle<()>>,
    enabled: bool,
    settings_note: (String, bool),
    prefs_checked: Option<Instant>,
    local_pref_override: bool,
}

impl Ai {
    fn working(&self) -> bool {
        self.status.busy || self.status.stopping || self.inflight.is_some()
    }

    fn begin_session(&mut self) {
        self.prior_stage.get_or_insert(self.stage);
    }

    fn invalidate_history_view(&mut self) {
        self.history_view_token = self.history_view_token.wrapping_add(1);
        self.history_pending = None;
    }

    fn select_stage(&mut self, stage: i32) {
        self.stage = stage;
        if self.prior_stage.is_some() {
            self.prior_stage = Some(stage);
        }
        self.follow = false;
        self.invalidate_history_view();
    }

    fn invalidate_request(&mut self) {
        self.token = self.token.wrapping_add(1);
        self.inflight = None;
        self.follow = false;
        self.last_key = None;
    }

    fn stop_session(&mut self) {
        self.live = false;
        self.invalidate_request();
        self.invalidate_history_view();
        self.next_at = None;
        if let Some(stage) = self.prior_stage.take() {
            self.stage = match stage {
                2 if self.frozen.is_none() => 0,
                1 if self.result.is_none() => 0,
                _ => stage,
            };
        }
    }
}

fn initial_ai(prefs: &Result<preferences::Preferences>) -> Ai {
    let mut ai = Ai::default();
    match prefs {
        Ok(prefs) => ai.enabled = prefs.ai_enabled,
        Err(e) => {
            ai.settings_note = (
                format!("Optional AI stays off because the settings cannot be read: {e:#}"),
                true,
            );
        }
    }
    ai
}

// ---- application state (UI thread only) ------------------------------------------------------

struct App {
    win: slint::Weak<MainWindow>,
    events: Receiver<Event>,
    event_tx: Sender<Event>,
    backend_tx: Option<Sender<BackendCmd>>,
    media_tx: Option<Sender<MediaCmd>>,
    audio_tx: Option<Sender<AudioCmd>>,
    vcam_tx: Option<Sender<VcamCmd>>,
    workers: Vec<JoinHandle<()>>,

    devices: Vec<Device>,
    device: Option<usize>,
    wanted: Option<String>,
    config: VideoConfig,
    fallback: Option<VideoConfig>,
    formats: Vec<VideoConfig>,
    formats_device: Option<String>,
    res_list: Vec<(u32, u32)>,
    fps_list: Vec<u32>,
    mode_warning: Option<String>,
    epoch: u64,
    auto: bool,
    connecting: bool,
    preview: Option<Arc<Preview>>,
    stage_error: Option<String>,
    media: MediaStatus,
    progress: (u64, Instant),
    media_error_shown: Option<String>,
    last_seq: u64,
    has_frame: bool,
    recording_path: Option<PathBuf>,

    snapshot: Option<Snapshot>,
    controls: Vec<Control>,
    row_revs: Vec<i32>,
    revs: HashMap<String, i32>,
    rows: Rc<VecModel<ControlRow>>,
    profiles: Vec<String>,
    presets: Vec<PresetInfo>,
    zones: Vec<Zone>,
    tracking: Option<TrackingAreaInfo>,
    effects: Effects,
    sources: Vec<Source>,
    camera_mic: Option<std::result::Result<CameraMic, String>>,
    audio_error: Option<String>,
    audio_pending: usize,
    mic_pending: usize,
    audio_checked: Instant,
    vcams: Vec<VirtualCamera>,
    vcam_missing: Vec<String>,
    vcam_pending: usize,
    vcam_stopping: bool,
    vcam_stop: Option<JoinHandle<()>>,
    vcam_note: (String, bool),
    last_vout: Option<String>,
    last_page: i32,

    bpending: usize,
    mpending: usize,
    status: (String, bool),
    last_status_poll: Instant,
    last_refresh: Instant,
    script_dirty: Option<Instant>,
    ai: Ai,
}

impl App {
    fn ui(&self) -> Option<MainWindow> {
        self.win.upgrade()
    }

    fn say(&mut self, message: impl Into<String>) {
        self.status = (message.into(), false);
    }

    fn fail(&mut self, message: impl Into<String>) {
        self.status = (message.into(), true);
    }

    fn send_backend(&mut self, command: BackendCmd) {
        let close = matches!(command, BackendCmd::Close | BackendCmd::Refresh);
        if self
            .backend_tx
            .as_ref()
            .is_some_and(|tx| tx.send(command).is_ok())
        {
            if !close {
                self.bpending += 1;
            }
        } else {
            self.fail("The camera worker stopped; restart the application.");
        }
    }

    fn send_media(&mut self, command: MediaCmd) {
        if self
            .media_tx
            .as_ref()
            .is_some_and(|tx| tx.send(command).is_ok())
        {
            self.mpending += 1;
        } else {
            self.fail("The media worker stopped; restart the application.");
        }
    }

    fn send_audio(&mut self, command: AudioCmd) -> bool {
        let sent = self
            .audio_tx
            .as_ref()
            .is_some_and(|tx| tx.send(command).is_ok());
        if sent {
            self.audio_pending += 1;
        } else {
            self.fail("The audio worker stopped; restart the application.");
        }
        sent
    }

    fn send_vcam(&mut self, command: VcamCmd) {
        if self.ai.closing {
            return;
        }
        if self
            .vcam_tx
            .as_ref()
            .is_some_and(|tx| tx.send(command).is_ok())
        {
            self.vcam_pending += 1;
        } else {
            self.fail("The virtual camera worker stopped; restart the application.");
        }
    }

    // -- connection lifecycle --

    /// Drops the preview and camera session. Late replies are ignored through the epoch.
    fn disconnect(&mut self) {
        self.epoch += 1;
        self.ai_reset(true);
        self.connecting = false;
        if let Some(preview) = self.preview.take() {
            self.send_media(MediaCmd::Stop(preview));
        }
        self.send_backend(BackendCmd::Close);
        self.media = MediaStatus::default();
        self.media_error_shown = None;
        self.recording_path = None;
        self.snapshot = None;
        self.presets.clear();
        self.zones.clear();
        self.tracking = None;
        self.clear_has_frame();
        self.set_controls_empty("Start the preview to read and change camera controls.");
        self.push_views();
        self.set_note(
            |ui, n| ui.set_presets_note(n),
            "Start the preview to read the camera's saved views.".into(),
        );
    }

    fn connect(&mut self) {
        let Some(device) = self.device.and_then(|i| self.devices.get(i)).cloned() else {
            return;
        };
        self.disconnect();
        self.connecting = true;
        self.stage_error = None;
        if self.formats_device.as_deref() == Some(device.id.as_str()) {
            self.start_preview();
        } else {
            self.send_media(MediaCmd::Formats {
                epoch: self.epoch,
                device: device.id,
            });
            self.say("Reading the camera's video modes…");
        }
    }

    fn start_preview(&mut self) {
        let Some(device) = self
            .device
            .and_then(|i| self.devices.get(i))
            .map(|d| d.id.clone())
        else {
            self.connecting = false;
            return;
        };
        let effects = effects_active(&self.effects).then(|| self.effects.clone());
        self.send_media(MediaCmd::Start {
            epoch: self.epoch,
            device,
            config: self.config.clone(),
            effects,
        });
        self.say("Starting preview…");
    }

    fn push_formats(&mut self) {
        self.res_list = resolutions_of(&self.formats);
        self.fps_list = fps_of(&self.formats, self.config.width, self.config.height);
        let Some(ui) = self.ui() else { return };
        let sizes: Vec<SharedString> = self
            .res_list
            .iter()
            .map(|(w, h)| SharedString::from(format!("{w}x{h}")))
            .collect();
        let rates: Vec<SharedString> = self
            .fps_list
            .iter()
            .map(|f| SharedString::from(f.to_string()))
            .collect();
        ui.set_resolutions(ModelRc::new(VecModel::from(sizes)));
        ui.set_fps_list(ModelRc::new(VecModel::from(rates)));
    }

    fn open_backend(&mut self) {
        let Some(device) = self
            .device
            .and_then(|i| self.devices.get(i))
            .map(|d| d.id.clone())
        else {
            self.connecting = false;
            return;
        };
        self.send_backend(BackendCmd::Open {
            epoch: self.epoch,
            device,
        });
    }

    fn preview_failed(&mut self, reason: String) {
        self.auto = false;
        self.disconnect();
        self.stage_error = Some(format!(
            "{reason}. Check the cable and that no other program uses the camera, then press Reconnect."
        ));
        self.fail(format!("Preview stopped: {reason}"));
    }

    fn refresh_devices(&mut self) {
        self.send_backend(BackendCmd::Devices);
    }

    fn reconnect(&mut self) {
        if self.bpending > 0 {
            self.fail("Wait for the current camera control request to finish before reconnecting.");
            self.sync();
            return;
        }
        self.auto = true;
        self.fallback = None;
        self.formats_device = None; // re-read the modes: the camera may have been replugged
        if self.devices.is_empty() {
            self.refresh_devices();
        } else {
            self.connect();
        }
        self.sync();
    }

    fn toggle_preview(&mut self) {
        if self.preview.is_some() {
            self.auto = false;
            self.disconnect();
            self.stage_error = None;
            self.say("Preview stopped");
        } else {
            self.reconnect();
            return;
        }
        self.sync();
    }

    fn select_device(&mut self, index: i32) {
        let index = usize::try_from(index)
            .ok()
            .filter(|i| *i < self.devices.len());
        if index.is_none() || index == self.device {
            self.sync();
            return;
        }
        self.device = index;
        self.refresh_audio();
        self.reconnect();
    }

    fn change_mode(&mut self, config: VideoConfig) {
        if same_mode(&config, &self.config) {
            return;
        }
        let previous = std::mem::replace(&mut self.config, config);
        self.push_formats();
        if self.preview.is_some() || self.connecting {
            self.fallback = Some(previous);
            self.auto = true;
            self.connect();
        }
        self.sync();
    }

    fn select_resolution(&mut self, index: i32) {
        let Some(&(width, height)) = usize::try_from(index)
            .ok()
            .and_then(|i| self.res_list.get(i))
        else {
            return;
        };
        // Keep the frame rate if this size offers it, otherwise take the nearest offered one.
        let fps = self
            .formats
            .iter()
            .filter(|m| m.width == width && m.height == height)
            .map(|m| m.fps)
            .min_by_key(|f| f.abs_diff(self.config.fps));
        if let Some(fps) = fps {
            self.change_mode(VideoConfig { width, height, fps });
        }
    }

    fn select_fps(&mut self, index: i32) {
        if let Some(&fps) = usize::try_from(index)
            .ok()
            .and_then(|i| self.fps_list.get(i))
        {
            self.change_mode(VideoConfig {
                fps,
                ..self.config.clone()
            });
        }
    }

    fn dismiss_status(&mut self) {
        self.status = (String::new(), false);
        self.sync();
    }

    // -- event handling --

    fn on_devices(&mut self, list: Vec<Device>) {
        let previous = self
            .device
            .and_then(|i| self.devices.get(i))
            .map(|d| d.id.clone());
        self.devices = list;
        let mut selected = None;
        let mut skip_default = false;
        if let Some(wanted) = self.wanted.take() {
            selected = self
                .devices
                .iter()
                .position(|d| d.id == wanted || d.name == wanted);
            if selected.is_none() {
                skip_default = true;
                self.fail(format!("Camera `{wanted}` was not found. Pick another camera from the list or press Refresh devices."));
            }
        }
        if selected.is_none() {
            selected = previous
                .as_ref()
                .and_then(|id| self.devices.iter().position(|d| d.id == *id));
        }
        if selected.is_none() && previous.is_none() && !self.devices.is_empty() && !skip_default {
            selected = Some(0);
        }
        let vanished = previous.is_some() && selected.is_none();
        self.device = selected;
        if vanished {
            self.auto = false;
            self.disconnect();
            self.stage_error = Some("The camera was unplugged or is no longer listed. Plug it in and press Refresh devices.".into());
            self.fail("The selected camera is no longer connected.");
        }
        if self.devices.is_empty() && self.preview.is_some() {
            self.disconnect();
        }
        let names: Vec<SharedString> = self
            .devices
            .iter()
            .map(|d| format!("{} ({})", d.name, d.id).into())
            .collect();
        if let Some(ui) = self.ui() {
            ui.set_devices(ModelRc::new(VecModel::from(names)));
        }
        self.refresh_audio();
        if self.auto && self.preview.is_none() && !self.connecting && self.device.is_some() {
            self.connect();
        }
    }

    fn on_event(&mut self, event: Event) {
        match event {
            Event::Backend(r) => self.on_backend(r),
            Event::Media(r) => self.on_media(r),
            Event::Audio(r) => self.on_audio(r),
            Event::Vcam(r) => self.on_vcam(r),
            Event::Browse(target, result) => self.on_browse(target, result),
            Event::AiSaved(result) => self.on_ai_saved(result),
            Event::AiHistoryRestored {
                token,
                record,
                result,
            } => self.on_ai_history_restored(token, record, result),
            Event::Library(done) => self.on_library_done(done),
        }
        self.sync();
    }

    fn on_backend(&mut self, r: Reply) {
        if !r.periodic {
            self.bpending = self.bpending.saturating_sub(1);
        }
        if let Some(profiles) = r.profiles {
            self.set_profiles(profiles, r.select_profile.as_deref());
        }
        if r.transport_lost && r.epoch == self.epoch {
            self.connecting = false;
            self.auto = false;
            self.snapshot = None;
            self.presets.clear();
            self.zones.clear();
            self.tracking = None;
            self.set_controls_empty("Camera controls stopped responding. Unplug/reconnect the camera, then press Reconnect.");
            self.push_views();
            self.push_zones();
        }
        if r.global {
            if let Some(list) = r.devices {
                self.on_devices(list);
            }
            match (r.err, r.ok) {
                (Some(e), _) => self.fail(e),
                (None, Some(m)) => self.say(m),
                _ => {}
            }
            return;
        }
        if r.epoch != self.epoch {
            return;
        }
        if r.opened {
            self.connecting = false;
        }
        let editing = self.ui().is_some_and(|ui| ui.get_controls_editing());
        if let Some(snapshot) = r.snapshot.filter(|_| !(r.periodic && editing)) {
            self.apply_snapshot(snapshot, r.commit);
        } else if matches!(r.commit, Commit::Key(_) | Commit::All) {
            self.bump(&r.commit);
            self.set_rows(self.controls.clone());
        }
        if let Some(result) = r.presets {
            match result {
                Ok(list) => {
                    self.presets = list;
                    self.set_note(|ui, n| ui.set_presets_note(n), String::new());
                }
                Err(e) => {
                    self.presets.clear();
                    self.set_note(
                        |ui, n| ui.set_presets_note(n),
                        format!("Saved views unavailable: {}", explain(e)),
                    );
                }
            }
            self.push_views();
        }
        if let Some(result) = r.areas {
            match result {
                Ok(areas) => {
                    self.zones = areas.iter().map(zone_from).collect();
                    self.set_note(|ui, n| ui.set_areas_note(n), String::new());
                }
                Err(e) => self.set_note(
                    |ui, n| ui.set_areas_note(n),
                    format!("Zones unavailable: {}", explain(e)),
                ),
            }
            self.push_zones();
        }
        if let Some(result) = r.tracking {
            match result {
                Ok(area) => {
                    if let Some(ui) = self.ui() {
                        ui.set_ta_pan_min(trim_num(area.pan[0] as f32).into());
                        ui.set_ta_pan_max(trim_num(area.pan[1] as f32).into());
                        ui.set_ta_tilt_min(trim_num(area.tilt[0] as f32).into());
                        ui.set_ta_tilt_max(trim_num(area.tilt[1] as f32).into());
                    }
                    self.tracking = Some(area);
                    self.set_note(|ui, n| ui.set_tracking_note(n), String::new());
                }
                Err(e) => self.set_note(
                    |ui, n| ui.set_tracking_note(n),
                    format!("Tracking area unavailable: {}", explain(e)),
                ),
            }
        }
        if let Some(e) = r.err {
            if r.opened {
                self.set_controls_empty(&format!("Camera controls are unavailable: {e}"));
            }
            if r.transport_lost {
                self.fail(format!("Camera controls unavailable: {e}. Unplug/reconnect the camera, then press Reconnect."));
            } else if !(r.periodic && self.status.1) {
                self.fail(explain(e));
            }
        } else if let Some(m) = r.ok {
            self.say(m);
        }
    }

    fn on_media(&mut self, r: MediaReply) {
        self.mpending = self.mpending.saturating_sub(1);
        if self.mpending == 0 {
            self.ai.geometry_pending = false;
        }
        if let Some(result) = r.formats {
            if r.epoch != self.epoch {
                return;
            }
            match result {
                Ok(modes) if !modes.is_empty() => {
                    if let Some(chosen) = choose_mode(&modes, &self.config) {
                        self.config = chosen;
                    }
                    self.formats_device = self
                        .device
                        .and_then(|i| self.devices.get(i))
                        .map(|d| d.id.clone());
                    self.formats = modes;
                    self.mode_warning = None;
                }
                other => {
                    let why = match other {
                        Ok(_) => "the camera reported no MJPEG modes".to_owned(),
                        Err(e) => e,
                    };
                    self.formats.clear();
                    self.mode_warning = Some(format!(
                        "Cannot list the camera's video modes ({why}). Trying {}x{} at {} fps; mode selection is unavailable.",
                        self.config.width, self.config.height, self.config.fps
                    ));
                }
            }
            self.push_formats();
            self.start_preview();
            return;
        }
        if let Some(started) = r.started {
            match started {
                Ok(preview) if r.epoch != self.epoch => self.send_media(MediaCmd::Stop(preview)),
                Ok(preview) => {
                    self.preview = Some(preview);
                    self.fallback = None;
                    self.media = MediaStatus::default();
                    self.progress = (0, Instant::now());
                    self.last_seq = 0;
                    match self.mode_warning.take() {
                        Some(warning) => self.fail(warning),
                        None => self.say("Preview running"),
                    }
                    if let Err(error) = preferences::set_capture(&self.config) {
                        self.fail(format!(
                            "Preview started, but its format was not saved: {error:#}"
                        ));
                    }
                    if let Some(e) = r.effects_err {
                        self.effects = initial_effects();
                        self.push_effects();
                        self.fail(format!(
                            "Effects were not applied, so they were turned off: {e}"
                        ));
                    }
                    self.open_backend();
                }
                Err(e) if r.epoch == self.epoch => {
                    self.connecting = false;
                    if let Some(previous) = self.fallback.take() {
                        self.config = previous;
                        self.fail(format!("The camera did not accept that video mode ({e}). Restoring the previous mode."));
                        self.auto = true;
                        self.connect();
                        return;
                    }
                    self.auto = false;
                    self.stage_error = Some(e.clone());
                    self.fail(format!("Preview could not start: {e}"));
                    self.open_backend();
                }
                Err(_) => {}
            }
            if r.epoch == self.epoch {
                if let Some(effects) = r.effects {
                    self.accept_effects(effects);
                }
            }
            return;
        }
        if let Some(effects) = r.effects {
            self.accept_effects(effects);
            if r.err.is_some() {
                self.push_effects();
            }
        }
        if r.virtual_err {
            self.vcam_note = (r.err.clone().unwrap_or_default(), true);
        }
        match (r.err, r.ok) {
            (Some(e), _) => self.fail(e),
            (None, Some(m)) => self.say(m),
            _ => {}
        }
        self.poll_status();
    }

    /// The microphone currently chosen in the picker (index 0 is "No audio").
    fn selected_source(&self, ui: &MainWindow) -> Option<&Source> {
        usize::try_from(ui.get_audio_index())
            .ok()
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| self.sources.get(i))
    }

    fn on_audio(&mut self, r: AudioReply) {
        self.audio_pending = self.audio_pending.saturating_sub(1);
        if r.setter {
            self.mic_pending = self.mic_pending.saturating_sub(1);
        }
        self.audio_checked = Instant::now();
        let Some(ui) = self.ui() else { return };
        let chosen = self.selected_source(&ui).map(|s| s.name.clone());
        if let Some(result) = r.sources {
            match result {
                Ok(list) => {
                    self.sources = list;
                    let lost = chosen
                        .as_ref()
                        .is_some_and(|n| !self.sources.iter().any(|s| &s.name == n));
                    self.audio_error = lost.then(|| {
                        "The microphone you chose is no longer listed, so recordings will have no sound until you pick another.".to_owned()
                    });
                }
                Err(e) => {
                    self.sources.clear();
                    self.audio_error = Some(format!(
                        "The audio service is unavailable: {e}. Recordings will have no sound."
                    ));
                }
            }
        }
        if let Some(camera) = r.camera {
            self.camera_mic = Some(camera);
        }
        if let Some((name, result)) = r.set {
            match result {
                Ok(updated) => {
                    if let Some(slot) = self.sources.iter_mut().find(|s| s.name == name) {
                        *slot = updated;
                    }
                    self.audio_error = None;
                }
                Err(e) => {
                    self.audio_error = Some(format!("Could not change the microphone: {e}"));
                }
            }
        }
        // The choice stays on the same named source; it never falls back to another one.
        let index = chosen
            .as_ref()
            .and_then(|n| self.sources.iter().position(|s| &s.name == n))
            .map_or(0, |i| i + 1);
        ui.set_audio_index(index as i32);
        self.push_audio(&ui);
    }

    fn plain_label(source: &Source) -> String {
        if source.description.is_empty() {
            source.name.clone()
        } else {
            source.description.clone()
        }
    }

    /// Picker entries, the notes and the microphone panel, all from what the audio service
    /// reported. Unreported mute or level stays "unknown" instead of being guessed.
    fn push_audio(&self, ui: &MainWindow) {
        let camera_source = match &self.camera_mic {
            Some(Ok(CameraMic::Found { source, .. })) => Some(source.name.as_str()),
            _ => None,
        };
        let mut labels: Vec<SharedString> = vec!["No audio".into()];
        labels.extend(self.sources.iter().map(|s| {
            let base = Self::plain_label(s);
            if camera_source == Some(s.name.as_str()) {
                SharedString::from(format!("{base} (this camera)"))
            } else {
                SharedString::from(base)
            }
        }));
        ui.set_audio_sources(ModelRc::new(VecModel::from(labels)));
        let selected = self.selected_source(ui);
        let note = match (&self.audio_error, self.sources.is_empty()) {
            (Some(e), _) => e.clone(),
            (None, true) => {
                "The audio service lists no microphones; recordings will have no sound.".to_owned()
            }
            (None, false) => String::new(),
        };
        ui.set_audio_note(note.into());
        let (camera_note, camera_available) = match (&self.camera_mic, self.device.is_some()) {
            (_, false) | (None, _) => (String::new(), false),
            (Some(Ok(CameraMic::Found { source, matched_by })), _) => (
                format!(
                    "This camera's microphone: {} (matched by {matched_by}).",
                    Self::plain_label(source)
                ),
                self.sources.iter().any(|s| s.name == source.name)
                    && selected.is_none_or(|s| s.name != source.name),
            ),
            (Some(Ok(CameraMic::NotFound { reason })), _) => (
                format!("No microphone could be tied to this camera: {reason}"),
                false,
            ),
            (Some(Ok(CameraMic::Ambiguous { sources })), _) => (
                format!(
                    "Several microphones could belong to this camera ({}). Pick the right one yourself; nothing was chosen for you.",
                    sources
                        .iter()
                        .map(Self::plain_label)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                false,
            ),
            (Some(Err(e)), _) => (
                format!("Cannot look up this camera's microphone: {e}"),
                false,
            ),
        };
        ui.set_camera_mic_note(camera_note.into());
        ui.set_camera_mic_available(camera_available);
        ui.set_mic_listed(selected.is_some());
        let Some(s) = selected else {
            ui.set_mic_title(SharedString::new());
            ui.set_mic_detail(SharedString::new());
            ui.set_mic_state(SharedString::new());
            ui.set_mic_muted_known(false);
            ui.set_mic_volume_known(false);
            return;
        };
        let (state, tone) = match s.state.as_deref() {
            Some("RUNNING") => ("In use by an app".to_owned(), 1),
            Some("IDLE") => ("Idle, not capturing".to_owned(), 0),
            Some("SUSPENDED") => ("Suspended, not capturing".to_owned(), 0),
            Some(other) => (format!("Reported as {other}"), 0),
            None => ("Activity not reported".to_owned(), 0),
        };
        ui.set_mic_title(Self::plain_label(s).into());
        ui.set_mic_detail(s.name.clone().into());
        ui.set_mic_state(state.into());
        ui.set_mic_state_tone(tone);
        ui.set_mic_muted_known(s.muted.is_some());
        ui.set_mic_muted(s.muted.unwrap_or(false));
        ui.set_mic_volume_known(s.volume_percent.is_some());
        ui.set_mic_volume(s.volume_percent.unwrap_or(0) as f32);
    }

    fn on_browse(&mut self, target: i32, result: std::result::Result<Option<String>, String>) {
        let path = match result {
            Ok(Some(path)) => path,
            Ok(None) => return,
            Err(e) => return self.fail(e),
        };
        let Some(ui) = self.ui() else { return };
        match target {
            0 => ui.set_fx_bg_image(path.into()),
            1 => ui.set_fx_overlay(path.into()),
            2 => ui.set_snapshot_dir(path.into()),
            3 => ui.set_record_dir(path.into()),
            5 => ui.set_ai_out_dir(path.into()),
            _ => ui.set_script_path(path.into()),
        }
        if target < 2 {
            self.apply_effects();
        }
    }

    fn set_note(&self, set: impl Fn(&MainWindow, SharedString), text: String) {
        if let Some(ui) = self.ui() {
            set(&ui, text.into());
        }
    }

    // -- camera controls --

    fn bump(&mut self, commit: &Commit) {
        match commit {
            Commit::None => {}
            Commit::Key(key) => *self.revs.entry(key.clone()).or_default() += 1,
            Commit::All => {
                for c in &self.controls {
                    *self.revs.entry(c.key.clone()).or_default() += 1;
                }
            }
        }
    }

    fn apply_snapshot(&mut self, snapshot: Snapshot, commit: Commit) {
        self.bump(&commit);
        let mut order: Vec<&str> = Vec::new();
        for c in &snapshot.controls {
            if !order.contains(&c.group.as_str()) {
                order.push(&c.group);
            }
        }
        let mut controls = snapshot.controls.clone();
        controls.sort_by_key(|c| order.iter().position(|g| *g == c.group));
        self.set_rows(controls);

        let mut info = format!("{} · {}", snapshot.device.name, snapshot.device.model);
        let detail: Vec<String> = [
            ("Firmware", &snapshot.firmware),
            ("Serial", &snapshot.serial),
        ]
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, v)| format!("{k} {v}"))
        .collect();
        if !detail.is_empty() {
            info = format!("{info}\n{}", detail.join(" · "));
        }
        if !snapshot.streaming {
            info.push_str("\nThe camera is not streaming; some settings are hidden or read-only.");
        }
        let note = if self.controls.is_empty() {
            "The camera reported no adjustable controls.".to_owned()
        } else {
            String::new()
        };
        let has = |key: &str| self.controls.iter().any(|c| c.key == key);
        if let Some(ui) = self.ui() {
            ui.set_device_info(info.into());
            ui.set_controls_note(note.into());
            let warnings = if snapshot.warnings.is_empty() {
                String::new()
            } else {
                format!(
                    "Unavailable on this camera: {}",
                    snapshot.warnings.join("; ")
                )
            };
            ui.set_warnings(warnings.into());
            ui.set_has_gimbal(has("pan"));
            ui.set_has_pause_areas(has("pause-areas"));
            ui.set_has_tracking_area(has("tracking-area"));
            let t = &snapshot.tracking;
            let capability = match t.capability {
                TrackingCapability::AutoFraming => {
                    "Fixed camera: Auto Framing reframes the picture; no motor moves."
                }
                TrackingCapability::SubjectTracking => {
                    "Gimbal camera: tracking follows a subject with the motor."
                }
                TrackingCapability::None => "Tracking support is not known for this model.",
            };
            ui.set_tracking_label(t.label.clone().into());
            let detail = if t.reason.is_empty() {
                capability
            } else {
                t.reason.as_str()
            };
            ui.set_tracking_detail(detail.into());
            ui.set_tracking_tone(match t.state {
                TrackingState::Active => 1,
                TrackingState::Unavailable => 3,
                _ => 0,
            });
        }
        self.snapshot = Some(snapshot);
        self.last_refresh = Instant::now();
    }

    fn set_controls_empty(&mut self, note: &str) {
        self.controls.clear();
        self.row_revs.clear();
        self.rows.set_vec(Vec::new());
        if let Some(ui) = self.ui() {
            ui.set_controls_note(note.into());
            ui.set_warnings(SharedString::new());
            ui.set_device_info(SharedString::new());
            ui.set_has_gimbal(false);
            ui.set_has_pause_areas(false);
            ui.set_has_tracking_area(false);
            ui.set_tracking_label(SharedString::new());
            ui.set_tracking_detail(SharedString::new());
            ui.set_tracking_tone(0);
        }
    }

    /// Updates the model in place when the control list keeps its shape, so focus and scroll
    /// position survive; replaces it otherwise.
    fn set_rows(&mut self, controls: Vec<Control>) {
        let revs: Vec<i32> = controls
            .iter()
            .map(|c| self.revs.get(&c.key).copied().unwrap_or(0))
            .collect();
        let rows = build_rows(&controls, &revs);
        let same_shape = self.controls.len() == controls.len()
            && self
                .controls
                .iter()
                .zip(&controls)
                .all(|(a, b)| a.key == b.key && a.group == b.group && kind_id(a) == kind_id(b));
        if same_shape {
            for (i, row) in rows.into_iter().enumerate() {
                if !same_control(&self.controls[i], &controls[i]) || self.row_revs[i] != revs[i] {
                    self.rows.set_row_data(i, row);
                }
            }
        } else {
            self.rows.set_vec(rows);
        }
        self.controls = controls;
        self.row_revs = revs;
    }

    fn write_control(&mut self, key: String, value: String) {
        self.send_backend(BackendCmd::Set { key, value });
    }

    fn reject(&mut self, index: usize, message: String) {
        if let Some(key) = self.controls.get(index).map(|c| c.key.clone()) {
            *self.revs.entry(key).or_default() += 1;
            self.set_rows(self.controls.clone());
        }
        self.fail(message);
        self.sync();
    }

    fn set_number(&mut self, index: i32, text: SharedString) {
        let Some(control) = usize::try_from(index)
            .ok()
            .and_then(|i| self.controls.get(i))
            .cloned()
        else {
            return;
        };
        let text = text.trim().to_owned();
        let label = control.label.clone();
        let value = match text.parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => {
                return self.reject(
                    index as usize,
                    format!("{label}: `{text}` is not a number."),
                );
            }
        };
        if control.max > control.min && !(control.min..=control.max).contains(&value) {
            return self.reject(
                index as usize,
                format!(
                    "{label} must be between {} and {}.",
                    control.min, control.max
                ),
            );
        }
        self.write_control(control.key, text);
        self.sync();
    }

    fn set_float(&mut self, index: i32, value: f32) {
        let Some(control) = usize::try_from(index)
            .ok()
            .and_then(|i| self.controls.get(i))
            .cloned()
        else {
            return;
        };
        let text = slider_text(f64::from(value), &control);
        self.write_control(control.key, text);
        self.sync();
    }

    fn set_toggle(&mut self, index: i32, on: bool) {
        if let Some(key) = usize::try_from(index)
            .ok()
            .and_then(|i| self.controls.get(i))
            .map(|c| c.key.clone())
        {
            self.write_control(key, on.to_string());
            self.sync();
        }
    }

    fn set_choice(&mut self, index: i32, option: i32) {
        let value = usize::try_from(index)
            .ok()
            .and_then(|i| self.controls.get(i))
            .and_then(|c| {
                usize::try_from(option)
                    .ok()
                    .and_then(|o| c.options.get(o))
                    .map(|v| (c.key.clone(), v.clone()))
            });
        if let Some((key, value)) = value {
            self.write_control(key, value);
            self.sync();
        }
    }

    fn refresh_controls(&mut self) {
        self.send_backend(BackendCmd::Refresh);
        self.sync();
    }

    // -- profiles / views --

    fn set_profiles(&mut self, profiles: Vec<String>, select: Option<&str>) {
        let Some(ui) = self.ui() else { return };
        let current = usize::try_from(ui.get_profile_index())
            .ok()
            .and_then(|i| self.profiles.get(i))
            .cloned();
        let want = select.map(str::to_owned).or(current);
        let index = want
            .and_then(|w| profiles.iter().position(|p| *p == w))
            .unwrap_or(0);
        self.profiles = profiles;
        let labels: Vec<SharedString> = self
            .profiles
            .iter()
            .map(|p| SharedString::from(p.as_str()))
            .collect();
        ui.set_profiles(ModelRc::new(VecModel::from(labels)));
        ui.set_profile_index(index as i32);
    }

    fn selected_profile(&self) -> Option<String> {
        let ui = self.ui()?;
        usize::try_from(ui.get_profile_index())
            .ok()
            .and_then(|i| self.profiles.get(i))
            .cloned()
    }

    fn profile_save(&mut self) {
        let Some(ui) = self.ui() else { return };
        let name = ui.get_profile_name().trim().to_owned();
        if let Err(e) = backend::validate_profile_name(&name) {
            return self.fail(format!("{e:#}"));
        }
        self.send_backend(BackendCmd::ProfileSave(name));
        self.sync();
    }

    fn profile_load(&mut self) {
        if let Some(name) = self.selected_profile() {
            self.send_backend(BackendCmd::ProfileLoad(name));
            self.sync();
        }
    }

    fn profile_delete(&mut self) {
        if let Some(name) = self.selected_profile() {
            self.send_backend(BackendCmd::ProfileDelete(name));
            self.sync();
        }
    }

    fn push_views(&self) {
        let Some(ui) = self.ui() else { return };
        let labels: Vec<SharedString> = self
            .presets
            .iter()
            .map(|p| {
                format!(
                    "Slot {}: {} ({}, zoom {:.2})",
                    p.slot, p.name, p.mode, p.zoom
                )
                .into()
            })
            .collect();
        ui.set_preset_list(ModelRc::new(VecModel::from(labels)));
        if usize::try_from(ui.get_preset_index()).is_ok_and(|i| i >= self.presets.len()) {
            ui.set_preset_index(0);
        }
    }

    fn selected_slot(&self) -> Option<u8> {
        let ui = self.ui()?;
        usize::try_from(ui.get_preset_index())
            .ok()
            .and_then(|i| self.presets.get(i))
            .map(|p| p.slot)
    }

    fn preset_recall(&mut self) {
        if let Some(slot) = self.selected_slot() {
            self.send_backend(BackendCmd::PresetRecall(slot));
            self.sync();
        }
    }

    fn preset_delete(&mut self) {
        if let Some(slot) = self.selected_slot() {
            self.send_backend(BackendCmd::PresetDelete(slot));
            self.sync();
        }
    }

    fn preset_save(&mut self) {
        let Some(ui) = self.ui() else { return };
        let name = ui.get_preset_name().trim().to_owned();
        if name.is_empty() {
            return self.fail("Enter a name for the view first.");
        }
        let slot = ui.get_preset_slot().clamp(0, 9) as u8;
        self.send_backend(BackendCmd::PresetSave { slot, name });
        self.sync();
    }

    fn recenter(&mut self) {
        self.send_backend(BackendCmd::Recenter);
        self.sync();
    }

    // -- pause zones / tracking area --

    fn push_zones(&self) {
        if let Some(ui) = self.ui() {
            let rows: Vec<AreaRow> = self.zones.iter().map(Zone::row).collect();
            ui.set_pause_zones(ModelRc::new(VecModel::from(rows)));
        }
    }

    fn area_add(&mut self) {
        if self.zones.len() < 6 {
            self.zones.push(Zone {
                active: true,
                pan: "0".into(),
                tilt: "0".into(),
                ..Zone::default()
            });
            self.push_zones();
        }
    }

    fn area_remove(&mut self, index: i32) {
        if let Ok(i) = usize::try_from(index)
            && i < self.zones.len()
        {
            self.zones.remove(i);
            self.push_zones();
        }
    }

    fn area_edit(&mut self, index: i32, field: i32, text: SharedString) {
        let Some(zone) = usize::try_from(index)
            .ok()
            .and_then(|i| self.zones.get_mut(i))
        else {
            return;
        };
        let text = text.to_string();
        match field {
            0 => zone.x = text,
            1 => zone.y = text,
            2 => zone.w = text,
            3 => zone.h = text,
            4 => zone.pan = text,
            _ => zone.tilt = text,
        }
    }

    fn area_active(&mut self, index: i32, on: bool) {
        if let Some(zone) = usize::try_from(index)
            .ok()
            .and_then(|i| self.zones.get_mut(i))
        {
            zone.active = on;
        }
    }

    fn areas_apply(&mut self) {
        let mut areas = Vec::new();
        for (n, z) in self.zones.iter().enumerate() {
            let name = |f: &str| format!("Zone {} {f}", n + 1);
            let parsed = (|| -> std::result::Result<PauseAreaInfo, String> {
                let (x, y) = (parse_unit(&z.x, &name("x"))?, parse_unit(&z.y, &name("y"))?);
                let (w, h) = (
                    parse_unit(&z.w, &name("width"))?,
                    parse_unit(&z.h, &name("height"))?,
                );
                if w <= 0.0 || h <= 0.0 || x + w > 1.0001 || y + h > 1.0001 {
                    return Err(format!(
                        "Zone {} must have a positive size and stay inside the frame.",
                        n + 1
                    ));
                }
                Ok(PauseAreaInfo {
                    active: z.active,
                    pan: parse_f64(&z.pan, &name("pan"))?,
                    tilt: parse_f64(&z.tilt, &name("tilt"))?,
                    x: f64::from(x),
                    y: f64::from(y),
                    width: f64::from(w),
                    height: f64::from(h),
                })
            })();
            match parsed {
                Ok(area) => areas.push(area),
                Err(e) => return self.fail(e),
            }
        }
        self.send_backend(BackendCmd::AreasSet(areas));
        self.sync();
    }

    fn areas_reload(&mut self) {
        self.send_backend(BackendCmd::AreasLoad);
        self.sync();
    }

    fn tracking_apply(&mut self) {
        let Some(ui) = self.ui() else { return };
        let read = || -> std::result::Result<[f64; 4], String> {
            Ok([
                parse_f64(&ui.get_ta_pan_min(), "Pan minimum")?,
                parse_f64(&ui.get_ta_pan_max(), "Pan maximum")?,
                parse_f64(&ui.get_ta_tilt_min(), "Tilt minimum")?,
                parse_f64(&ui.get_ta_tilt_max(), "Tilt maximum")?,
            ])
        };
        let [pan_lo, pan_hi, tilt_lo, tilt_hi] = match read() {
            Ok(v) => v,
            Err(e) => return self.fail(e),
        };
        if pan_lo > pan_hi || tilt_lo > tilt_hi {
            return self.fail("Each minimum must not exceed its maximum.");
        }
        let enabled = self
            .controls
            .iter()
            .find(|c| c.key == "tracking-area")
            .map(|c| c.value == "true")
            .or_else(|| self.tracking.as_ref().map(|t| t.enabled))
            .unwrap_or(false);
        self.send_backend(BackendCmd::TrackingSet(TrackingAreaInfo {
            enabled,
            pan: [pan_lo, pan_hi],
            tilt: [tilt_lo, tilt_hi],
        }));
        self.sync();
    }

    fn tracking_reload(&mut self) {
        self.send_backend(BackendCmd::TrackingLoad);
        self.sync();
    }

    // -- effects --

    fn push_effects(&self) {
        let Some(ui) = self.ui() else { return };
        let e = &self.effects;
        ui.set_fx_bg_index(
            BACKGROUNDS
                .iter()
                .position(|b| *b == e.background)
                .unwrap_or(0) as i32,
        );
        ui.set_fx_blur(e.blur);
        ui.set_fx_soften(e.soften);
        ui.set_fx_mirror(e.mirror);
        ui.set_fx_bg_image(
            e.background_image
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
                .into(),
        );
        ui.set_fx_overlay(
            e.overlay_image
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
                .into(),
        );
        ui.set_fx_opacity(e.overlay_opacity);
        ui.set_fx_board_on(e.board.is_some());
        if let Some(b) = e.board {
            self.push_corners(&b);
        }
    }

    fn push_corners(&self, b: &[[f32; 2]; 4]) {
        let Some(ui) = self.ui() else { return };
        let t = |v: f32| SharedString::from(trim_num(v));
        ui.set_fx_tlx(t(b[0][0]));
        ui.set_fx_tly(t(b[0][1]));
        ui.set_fx_trx(t(b[1][0]));
        ui.set_fx_try(t(b[1][1]));
        ui.set_fx_brx(t(b[2][0]));
        ui.set_fx_bry(t(b[2][1]));
        ui.set_fx_blx(t(b[3][0]));
        ui.set_fx_bly(t(b[3][1]));
    }

    fn accept_effects(&mut self, effects: Effects) {
        self.effects = effects;
        if !self.ai.enabled && matches!(self.effects.background.as_str(), "blur" | "replace") {
            let mut disabled = self.effects.clone();
            disabled.background = "none".into();
            disabled.background_image = None;
            self.request_effects(disabled);
            if let Some(ui) = self.ui() {
                ui.set_fx_bg_index(0);
            }
        } else if let Err(error) = preferences::set_effects(&self.effects) {
            self.fail(format!(
                "Effects applied, but the setup was not saved: {error:#}"
            ));
        }
    }

    fn request_effects(&mut self, new: Effects) {
        if new.mirror != self.effects.mirror || new.board != self.effects.board {
            self.ai_geometry_changed();
        }
        match self.preview.clone() {
            Some(preview) => {
                let fallback = self.effects.clone();
                self.send_media(MediaCmd::Effects {
                    preview,
                    effects: new,
                    fallback,
                });
            }
            None => {
                self.effects = new;
                match preferences::set_effects(&self.effects) {
                    Ok(()) => {
                        self.say("Effects are saved and will be applied when the preview starts.")
                    }
                    Err(error) => self.fail(format!(
                        "Effects selected, but the setup was not saved: {error:#}"
                    )),
                }
            }
        }
        self.sync();
    }

    fn apply_effects(&mut self) {
        let Some(ui) = self.ui() else { return };
        let mut e = self.effects.clone();
        let index = usize::try_from(ui.get_fx_bg_index())
            .unwrap_or(0)
            .min(BACKGROUNDS.len() - 1);
        e.background = BACKGROUNDS[index].into();
        if !self.ai.enabled && matches!(e.background.as_str(), "blur" | "replace") {
            self.push_effects();
            self.say("Blur and background replacement need optional AI. Turn it on in Settings.");
            return;
        }
        e.blur = ui.get_fx_blur();
        e.soften = ui.get_fx_soften();
        e.mirror = ui.get_fx_mirror();
        e.overlay_opacity = ui.get_fx_opacity();
        let path = |s: SharedString| (!s.trim().is_empty()).then(|| PathBuf::from(s.trim()));
        e.background_image = path(ui.get_fx_bg_image());
        e.overlay_image = path(ui.get_fx_overlay());
        if matches!(e.background.as_str(), "replace" | "chroma") && e.background_image.is_none() {
            // Keep the background choice staged without blocking independent image controls.
            e.background = self.effects.background.clone();
            e.background_image = self.effects.background_image.clone();
            self.request_effects(e);
            self.say("Choose a background image to activate it. Other effects remain active.");
            self.sync();
            return;
        }
        if !matches!(e.background.as_str(), "replace" | "chroma") {
            e.background_image = None;
        }
        self.request_effects(e);
    }

    fn apply_board(&mut self) {
        let Some(ui) = self.ui() else { return };
        let mut e = self.effects.clone();
        if ui.get_fx_board_on() {
            let fields = [
                ("Top-left x", ui.get_fx_tlx()),
                ("Top-left y", ui.get_fx_tly()),
                ("Top-right x", ui.get_fx_trx()),
                ("Top-right y", ui.get_fx_try()),
                ("Bottom-right x", ui.get_fx_brx()),
                ("Bottom-right y", ui.get_fx_bry()),
                ("Bottom-left x", ui.get_fx_blx()),
                ("Bottom-left y", ui.get_fx_bly()),
            ];
            let mut v = [0f32; 8];
            for (slot, (what, text)) in v.iter_mut().zip(&fields) {
                match parse_unit(text, what) {
                    Ok(n) => *slot = n,
                    Err(msg) => {
                        self.fail(format!(
                            "{msg}. Enter all eight corner values, or press Reset to full frame."
                        ));
                        ui.set_fx_board_on(e.board.is_some());
                        return;
                    }
                }
            }
            e.board = Some([[v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], v[7]]]);
        } else {
            e.board = None;
        }
        self.request_effects(e);
    }

    fn reset_board(&mut self) {
        self.push_corners(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        self.apply_board();
    }

    fn browse(&mut self, target: i32) {
        // A modal chooser would block this thread; run it elsewhere and deliver the answer as an event.
        let events = self.events_sender();
        std::thread::spawn(move || {
            let _ = events.send(Event::Browse(target, choose_path(target)));
        });
    }

    // -- capture --

    fn require_preview(&mut self) -> Option<Arc<Preview>> {
        let preview = self.preview.clone();
        if preview.is_none() {
            self.fail("Start the preview first.");
        }
        preview
    }

    fn snapshot(&mut self) {
        let (Some(preview), Some(ui)) = (self.require_preview(), self.ui()) else {
            return;
        };
        match next_path(&ui.get_snapshot_dir(), "png") {
            Ok(path) => self.send_media(MediaCmd::Snapshot(preview, path)),
            Err(e) => self.fail(e),
        }
        self.sync();
    }

    fn record(&mut self) {
        let (Some(preview), Some(ui)) = (self.require_preview(), self.ui()) else {
            return;
        };
        let ext = CONTAINERS[usize::try_from(ui.get_container_index())
            .unwrap_or(0)
            .min(CONTAINERS.len() - 1)];
        let audio = self
            .selected_source(&ui)
            .map(|s| format!("pulse:{}", s.name));
        match next_path(&ui.get_record_dir(), ext) {
            Ok(path) => {
                self.recording_path = Some(path.clone());
                self.send_media(MediaCmd::Record(preview, path, audio));
            }
            Err(e) => self.fail(e),
        }
        self.sync();
    }

    fn pause_record(&mut self) {
        if let Some(preview) = self.require_preview() {
            let paused = !self.media.paused;
            self.send_media(MediaCmd::Pause(preview, paused));
        }
        self.sync();
    }

    fn stop_record(&mut self) {
        if let Some(preview) = self.require_preview() {
            self.send_media(MediaCmd::StopRecord(preview));
            self.say("Finishing the recording…");
        }
        self.sync();
    }

    /// Starts or stops the selected virtual camera. Stopping never needs a preview.
    fn vcam_toggle(&mut self) {
        let Some(ui) = self.ui() else { return };
        let Some(cam) = self.selected_vcam(&ui).cloned() else {
            return self.fail("Choose a virtual camera first.");
        };
        if self.vcam_sending_here(&cam) {
            match self.preview.clone() {
                Some(preview) => self.send_media(MediaCmd::Virtual(preview, None)),
                None => self.send_vcam(VcamCmd::Stop(cam.path)),
            }
        } else if cam.producer_pid.is_some() {
            self.send_vcam(VcamCmd::Stop(cam.path));
        } else {
            let Some(preview) = self.require_preview() else {
                return;
            };
            self.vcam_note = (String::new(), false);
            self.send_media(MediaCmd::Virtual(preview, Some(cam.path)));
        }
        self.sync();
    }

    fn vcam_refresh(&mut self) {
        self.vcam_note = (String::new(), false);
        self.send_vcam(VcamCmd::Refresh);
        self.sync();
    }

    fn vcam_create(&mut self, name: SharedString) {
        let name = name.trim().to_owned();
        if name.is_empty() {
            return self.fail("Name the virtual camera first.");
        }
        self.vcam_note = (
            "Creating the virtual camera… approve the administrator prompt if one appears.".into(),
            false,
        );
        self.send_vcam(VcamCmd::Create(name));
        self.sync();
    }

    fn vcam_remove(&mut self) {
        let Some(ui) = self.ui() else { return };
        let Some(cam) = self.selected_vcam(&ui).filter(|c| c.owned).cloned() else {
            return self.fail("Only a virtual camera created by Link Camera can be removed here.");
        };
        self.vcam_note = (
            "Removing the virtual camera… approve the administrator prompt if one appears.".into(),
            false,
        );
        self.send_vcam(VcamCmd::Remove(cam.path));
        self.sync();
    }

    /// Ends every feed Link Camera sends, from this window, other windows or the command line.
    fn vcam_stop_all(&mut self) {
        if self.vcam_stopping || self.ai.closing {
            return;
        }
        let tx = self.events_sender();
        let spawned = std::thread::Builder::new()
            .name("camera-virtual-stop".into())
            .spawn(move || {
                let _ = tx.send(Event::Vcam(stop_all_reply()));
            });
        match spawned {
            Ok(handle) => {
                self.vcam_stop = Some(handle);
                self.vcam_stopping = true;
                self.vcam_note = ("Stopping every Link Camera output…".into(), false);
            }
            Err(e) => self.fail(format!("Cannot start the stop request: {e}")),
        }
        self.sync();
    }

    fn selected_vcam(&self, ui: &MainWindow) -> Option<&VirtualCamera> {
        usize::try_from(ui.get_vcam_index())
            .ok()
            .and_then(|i| self.vcams.get(i))
    }

    /// This window's preview is the one feeding the node.
    fn vcam_sending_here(&self, cam: &VirtualCamera) -> bool {
        self.media
            .virtual_device
            .as_deref()
            .is_some_and(|d| Path::new(d) == cam.path)
    }

    fn on_vcam(&mut self, r: VcamReply) {
        if r.stop_all {
            self.vcam_stopping = false;
            if let Some(handle) = self.vcam_stop.take() {
                let _ = handle.join();
            }
        } else {
            self.vcam_pending = self.vcam_pending.saturating_sub(1);
        }
        let Some(ui) = self.ui() else { return };
        let chosen = self.selected_vcam(&ui).map(|c| c.path.clone());
        let list_error = match r.list {
            Some(Ok(list)) => {
                self.vcams = list;
                None
            }
            Some(Err(e)) => {
                self.vcams.clear();
                Some(e)
            }
            None => None,
        };
        self.vcam_missing = r.missing;
        if r.ok.is_some() || r.err.is_some() {
            let text = [r.ok.as_deref(), r.err.as_deref()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("\n");
            self.vcam_note = (text, r.err.is_some());
        } else if let Some(e) = list_error {
            self.vcam_note = (format!("Cannot list virtual cameras: {e}"), true);
        }
        if let Some(e) = r.err {
            self.fail(e);
        } else if let Some(m) = r.ok {
            self.say(m);
        }
        let labels: Vec<SharedString> = self
            .vcams
            .iter()
            .map(|c| {
                let owner = if c.owned {
                    "made by Link Camera"
                } else {
                    "existing device"
                };
                let sending = if self.vcam_sending_here(c) || c.producer_pid.is_some() {
                    " · sending"
                } else {
                    ""
                };
                SharedString::from(format!(
                    "{} · {} · {owner}{sending}",
                    c.path.display(),
                    c.label
                ))
            })
            .collect();
        // Keep the same device selected; with none chosen yet, offer the first one Link Camera made.
        let index = match chosen {
            Some(path) => self.vcams.iter().position(|c| c.path == path),
            None => self
                .vcams
                .iter()
                .position(|c| c.owned)
                .or(if self.vcams.is_empty() { None } else { Some(0) }),
        };
        ui.set_vcam_list(ModelRc::new(VecModel::from(labels)));
        ui.set_vcam_index(index.map_or(-1, |i| i as i32));
    }

    fn sync_vcam(&self, ui: &MainWindow) {
        let cam = self.selected_vcam(ui);
        let status = match cam {
            Some(c) => {
                let state = if self.vcam_sending_here(c) {
                    "This window is sending the processed picture to it.".to_owned()
                } else if let Some(pid) = c.producer_pid {
                    format!("Another Link Camera process (pid {pid}) is sending to it.")
                } else {
                    "Idle: nothing from Link Camera is sending to it.".to_owned()
                };
                let who = if c.owned {
                    "Created by Link Camera."
                } else {
                    "An existing v4l2loopback device; Link Camera never removes it."
                };
                format!("{state} {who}")
            }
            None if self.vcams.is_empty() => {
                let mut text = "No virtual camera was found.".to_owned();
                for line in &self.vcam_missing {
                    text.push('\n');
                    text.push_str(line);
                }
                text
            }
            None => "Choose a virtual camera above.".to_owned(),
        };
        ui.set_vcam_busy(self.vcam_pending > 0);
        ui.set_vcam_stopping(self.vcam_stopping);
        ui.set_vcam_selected_owned(cam.is_some_and(|c| c.owned));
        ui.set_vcam_selected_active(
            cam.is_some_and(|c| self.vcam_sending_here(c) || c.producer_pid.is_some()),
        );
        ui.set_vcam_status(status.into());
        ui.set_vcam_note(self.vcam_note.0.clone().into());
        ui.set_vcam_note_error(self.vcam_note.1);
    }

    /// Reads the microphones and this camera's microphone from the audio service, off the UI thread.
    fn refresh_audio(&mut self) {
        let video = self
            .device
            .and_then(|i| self.devices.get(i))
            .map(|d| d.id.clone());
        self.audio_checked = Instant::now();
        self.send_audio(AudioCmd::Scan { video });
    }

    fn audio_selected(&mut self, _index: i32) {
        if let Some(ui) = self.ui() {
            self.push_audio(&ui);
        }
    }

    /// Selects the microphone the system tied to this camera by its USB path, never a guess.
    fn use_camera_mic(&mut self) {
        let Some(ui) = self.ui() else { return };
        let Some(Ok(CameraMic::Found { source, .. })) = &self.camera_mic else {
            return;
        };
        if let Some(i) = self.sources.iter().position(|s| s.name == source.name) {
            ui.set_audio_index(i as i32 + 1);
            self.push_audio(&ui);
        }
    }

    fn set_mic_muted(&mut self, on: bool) {
        let Some(ui) = self.ui() else { return };
        let Some(name) = self.selected_source(&ui).map(|s| s.name.clone()) else {
            return;
        };
        if self.send_audio(AudioCmd::Mute { name, muted: on }) {
            self.mic_pending += 1;
        }
        self.sync();
    }

    fn set_mic_volume(&mut self, value: f32) {
        let Some(ui) = self.ui() else { return };
        let Some(name) = self.selected_source(&ui).map(|s| s.name.clone()) else {
            return;
        };
        let percent = value.round().clamp(0.0, 100.0) as u32;
        if self.send_audio(AudioCmd::Volume { name, percent }) {
            self.mic_pending += 1;
        }
        self.sync();
    }

    // -- script --

    fn script_edited(&mut self) {
        self.script_dirty = Some(Instant::now());
    }

    fn save_script_store(&mut self) {
        if self.script_dirty.take().is_none() {
            return;
        }
        let Some(ui) = self.ui() else { return };
        let path = script_store();
        if let Err(e) = write_script(&path, &ui.get_script_text()) {
            self.script_dirty = Some(Instant::now());
            self.fail(format!("Cannot save the script to {}: {e}", path.display()));
        }
    }

    fn script_load(&mut self) {
        let Some(ui) = self.ui() else { return };
        match read_script(Path::new(ui.get_script_path().trim())) {
            Ok(text) => {
                ui.set_script_text(text.into());
                ui.set_prompter_offset(0.0);
                ui.set_prompter_playing(false);
                self.script_dirty = Some(Instant::now());
                self.say("Script loaded");
            }
            Err(e) => self.fail(e),
        }
    }

    fn script_save(&mut self) {
        let Some(ui) = self.ui() else { return };
        let path = PathBuf::from(ui.get_script_path().trim());
        match write_script(&path, &ui.get_script_text()) {
            Ok(()) => self.say(format!("Script saved to {}", path.display())),
            Err(e) => self.fail(format!("Cannot save {}: {e}", path.display())),
        }
    }

    // -- timer --

    fn events_sender(&self) -> Sender<Event> {
        self.event_tx.clone()
    }

    fn clear_has_frame(&mut self) {
        self.has_frame = false;
        self.last_seq = 0;
        if let Some(ui) = self.ui() {
            ui.set_has_frame(false);
            ui.set_frame(Image::default());
            ui.set_frame_info(SharedString::new());
        }
    }

    fn show_frame(&mut self) {
        let Some(frame) = self.preview.as_ref().and_then(|p| p.latest_frame()) else {
            return;
        };
        if frame.sequence == self.last_seq && self.has_frame {
            return;
        }
        let (w, h) = (frame.width, frame.height);
        if w == 0 || h == 0 || frame.rgb.len() != w as usize * h as usize * 3 {
            return;
        }
        let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(w, h);
        buffer.make_mut_bytes().copy_from_slice(&frame.rgb);
        self.last_seq = frame.sequence;
        self.has_frame = true;
        if let Some(ui) = self.ui() {
            ui.set_frame(Image::from_rgb8(buffer));
            ui.set_has_frame(true);
            ui.set_frame_info(
                format!(
                    "{w}x{h} · {} fps requested · frame {}",
                    self.config.fps, frame.sequence
                )
                .into(),
            );
        }
    }

    fn poll_status(&mut self) {
        let Some(preview) = self.preview.clone() else {
            return;
        };
        let status = preview.status();
        if status.frames != self.progress.0 {
            self.progress = (status.frames, Instant::now());
        }
        let stalled = !self.connecting && self.progress.1.elapsed() > STALL;
        if stalled {
            let reason = status
                .error
                .clone()
                .unwrap_or_else(|| "the camera stopped delivering frames".to_owned());
            return self.preview_failed(reason);
        }
        // The error can be a recoverable sink failure (e.g. virtual output) while frames keep flowing.
        if status.error != self.media_error_shown {
            if let Some(e) = &status.error {
                self.fail(format!("Media warning: {e}"));
            }
            self.media_error_shown = status.error.clone();
        }
        if !status.recording {
            self.recording_path = None;
        }
        self.media = status;
        // A feed that started or ended (here, by stop-all or after a failure) changes the picker.
        if self.media.virtual_device != self.last_vout {
            self.last_vout = self.media.virtual_device.clone();
            if self.vcam_pending == 0 {
                self.send_vcam(VcamCmd::Refresh);
            }
        }
    }

    fn tick(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            self.on_event(event);
        }
        self.show_frame();
        self.ai_tick();
        if self.last_status_poll.elapsed() >= Duration::from_millis(250) {
            self.last_status_poll = Instant::now();
            self.poll_status();
            self.sync();
        }
        let editing = self.ui().is_some_and(|ui| ui.get_controls_editing());
        if self.preview.is_some()
            && self.snapshot.is_some()
            && !self.connecting
            && self.bpending == 0
            && !editing
            && self.last_refresh.elapsed() >= REFRESH_EVERY
        {
            self.last_refresh = Instant::now();
            self.send_backend(BackendCmd::Refresh);
        }
        if self
            .script_dirty
            .is_some_and(|t| t.elapsed() >= SCRIPT_SAVE_AFTER)
        {
            self.save_script_store();
        }
        // The Output page shows live microphone and virtual camera state, so keep it fresh there.
        let page = self.ui().map_or(0, |ui| ui.get_active_page());
        if page == 3
            && (self.last_page != 3 || self.audio_checked.elapsed() >= AUDIO_REFRESH)
            && self.audio_pending == 0
        {
            self.refresh_audio();
            if self.vcam_pending == 0 {
                self.send_vcam(VcamCmd::Refresh);
            }
        }
        self.last_page = page;
    }

    // -- derived UI state --

    fn sync(&self) {
        let Some(ui) = self.ui() else { return };
        let selected = self.device.and_then(|i| self.devices.get(i));
        ui.set_device_index(self.device.map_or(-1, |i| i as i32));
        ui.set_previewing(self.preview.is_some());
        ui.set_recording(self.media.recording);
        ui.set_paused(self.media.paused);
        ui.set_media_busy(self.mpending > 0 || self.connecting);
        ui.set_backend_busy(self.bpending > 0);
        ui.set_can_stop_preview(
            self.preview.is_some() && !self.media.recording && self.mpending == 0,
        );
        ui.set_preview_starting(self.connecting && self.preview.is_none());
        ui.set_resolution_index(
            self.res_list
                .iter()
                .position(|&(w, h)| w == self.config.width && h == self.config.height)
                .map_or(-1, |i| i as i32),
        );
        ui.set_fps_index(
            self.fps_list
                .iter()
                .position(|&f| f == self.config.fps)
                .map_or(-1, |i| i as i32),
        );
        ui.set_vout_active(self.media.virtual_device.clone().unwrap_or_default().into());
        self.sync_vcam(&ui);
        ui.set_mic_busy(self.mic_pending > 0);
        let remote = match selected {
            Some(d) => format!("link-camera --device {} serve", d.id),
            None => "link-camera serve".to_owned(),
        };
        ui.set_remote_command(remote.into());
        let recording_note = match (&self.recording_path, self.media.recording) {
            (Some(path), true) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                format!(
                    "{} {name}",
                    if self.media.paused {
                        "Paused:"
                    } else {
                        "Recording:"
                    }
                )
            }
            _ => String::new(),
        };
        ui.set_recording_note(recording_note.into());
        let (title, hint) = if self.devices.is_empty() && self.preview.is_none() {
            ("No camera found".to_owned(), "Connect an Insta360 Link camera and press Refresh devices. If it is plugged in, check that your user may open /dev/video* (video group).".to_owned())
        } else if let Some(error) = &self.stage_error {
            ("Preview unavailable".to_owned(), error.clone())
        } else if self.connecting {
            (
                "Starting preview…".to_owned(),
                "Opening the camera and waiting for the first frame.".to_owned(),
            )
        } else if self.preview.is_some() {
            ("Waiting for video…".to_owned(), String::new())
        } else {
            (
                "Preview stopped".to_owned(),
                "Press Start preview to see the camera and change its settings.".to_owned(),
            )
        };
        ui.set_stage_title(title.into());
        ui.set_stage_hint(hint.into());
        ui.set_status_text(self.status.0.clone().into());
        ui.set_status_error(self.status.1);
        self.ai_sync(&ui);
    }

    // -- AI models --

    fn ai_id(&self) -> &'static str {
        MODEL_IDS[self.ai.model.min(MODEL_IDS.len() - 1)]
    }

    fn ai_note(&mut self, text: impl Into<String>, error: bool) {
        self.ai.note = (text.into(), error);
    }

    /// The catalog entry of `id` as last read from the local library.
    fn ai_info(&self, id: &str) -> Option<&ModelInfo> {
        self.ai
            .status
            .catalog
            .as_ref()
            .and_then(|c| c.iter().find(|m| m.id == id))
    }

    fn ai_state(&self, id: &str) -> ModelState {
        let Some(catalog) = &self.ai.status.catalog else {
            return if self.ai.engine.is_some() && self.ai.status.busy {
                ModelState::Checking
            } else {
                ModelState::Off
            };
        };
        match catalog.iter().find(|m| m.id == id) {
            None => ModelState::Missing,
            Some(m) if !m.installed => ModelState::NotInstalled,
            Some(m) if m.available == Some(false) => ModelState::Unavailable(
                m.unavailable_reason
                    .clone()
                    .unwrap_or_else(|| "no reason given".into()),
            ),
            Some(m) => ModelState::Ready {
                loaded: m.loaded,
                loading: m.loading,
            },
        }
    }

    /// Whether a new live session can start with the installed local model.
    fn ai_can_live(&self) -> bool {
        self.ai.enabled
            && !self.ai.closing
            && !self.ai.working()
            && self.ai.library.is_none()
            && self.ai.engine.is_some()
            && matches!(self.ai_state(self.ai_id()), ModelState::Ready { .. })
    }

    /// SAM analyzes its frozen frame when there is one, every other model the newest frame.
    fn ai_source_frame(&self) -> Option<Arc<Frame>> {
        if model_kind(self.ai_id()) == 4
            && let Some(frozen) = &self.ai.frozen
        {
            return Some(frozen.clone());
        }
        self.preview
            .as_ref()
            .and_then(|p| p.latest_frame())
            .filter(|f| frame_ok(f))
    }

    /// Why Run is unavailable, if it is.
    fn ai_run_block(&self, manual: bool) -> Option<String> {
        let ai = &self.ai;
        if !ai.enabled {
            return Some("Optional AI is off. Turn it on in Settings.".into());
        }
        if ai.closing {
            return Some("The window is closing.".into());
        }
        if let Some(job) = &ai.library {
            return Some(format!(
                "{} {} is in progress; analysis resumes when it finishes.",
                if job.download {
                    "Downloading"
                } else {
                    "Deleting"
                },
                model_label(job.model)
            ));
        }
        if ai.engine.is_none() {
            return Some(
                "The local model library is not ready. Turn optional AI off and on in Settings to retry."
                    .into(),
            );
        }
        if ai.status.stopping {
            return Some("Stopping and unloading the model…".into());
        }
        if manual && ai.live {
            return Some("Live analysis is on. Stop it before running a single analysis.".into());
        }
        if ai.working() {
            let text = if ai.status.catalog.is_none() {
                "Reading the local model library…"
            } else if ai.inflight.is_none() {
                "Stopping the previous analysis…"
            } else {
                "An analysis is running."
            };
            return Some(text.into());
        }
        match self.ai_state(self.ai_id()) {
            ModelState::Missing => {
                return Some("This build does not list the selected model.".into());
            }
            ModelState::NotInstalled => {
                return Some(
                    "The selected model is not installed. Press Download in the Model section."
                        .into(),
                );
            }
            ModelState::Unavailable(why) => {
                return Some(format!("The selected model cannot run: {why}."));
            }
            ModelState::Off | ModelState::Checking => {
                return Some("The model list is not ready yet.".into());
            }
            ModelState::Ready { .. } => {}
        }
        if ai.geometry_pending {
            return Some("Waiting for the mirror or board change to reach the preview…".into());
        }
        if self.ai_source_frame().is_none() {
            return Some("Start the camera preview first.".into());
        }
        None
    }

    fn ai_model_info(&self) -> String {
        let id = self.ai_id();
        let mut lines: Vec<String> = vec![match self.ai_state(id) {
            ModelState::Off => "Turn on optional AI in Settings to see this model's state.".into(),
            ModelState::Checking => "Reading the local model library.".into(),
            ModelState::Missing => "This build does not list this model.".into(),
            ModelState::NotInstalled => {
                "Not installed. Nothing runs and nothing is contacted until you download it.".into()
            }
            ModelState::Unavailable(why) => format!("Installed, but it cannot run: {why}."),
            ModelState::Ready { loading: true, .. } => {
                "Loading into memory now; the first run waits for it.".into()
            }
            ModelState::Ready { loaded: true, .. } => "Loaded and ready.".into(),
            ModelState::Ready { .. } => {
                "Installed. The first run loads it into memory and takes longer.".into()
            }
        }];
        if let Some(m) = self.ai_info(id) {
            let field = |key: &str| text_of(&m.presentation[key]);
            if let Some(s) = field("summary") {
                lines.push(s);
            }
            if let Some(s) = field("limitations") {
                lines.push(format!("Limits: {s}"));
            }
            if let Some(s) = field("license") {
                lines.push(format!("License: {s}"));
            }
            if let Some(s) = field("source") {
                lines.push(format!("Source: {s}"));
            }
            if !m.installed
                && let Some(s) = field("download_size")
            {
                lines.push(format!("Download size: {s}."));
            } else if m.installed
                && let Some(s) = field("revision")
            {
                lines.push(format!("Revision: {s}"));
            }
        }
        match model_kind(id) {
            4 => lines.push("The result is the outline of the largest region (up to 200 points), not a full mask.".into()),
            5 | 6 => lines.push("Relative depth only: near is bright, values are not metres.".into()),
            _ => {}
        }
        lines.join("\n")
    }

    fn ai_push_models(&self) {
        let Some(ui) = self.ui() else { return };
        let labels: Vec<SharedString> = MODEL_IDS
            .iter()
            .map(|id| {
                let label = model_label(id);
                match state_tag(&self.ai_state(id)) {
                    "" => label.into(),
                    tag => format!("{label} · {tag}").into(),
                }
            })
            .collect();
        let id = self.ai_id();
        let ms = models::model_spec(id).map_or(0, |s| s.interval_ms);
        ui.set_ai_models(ModelRc::new(VecModel::from(labels)));
        ui.set_ai_model_index(self.ai.model as i32);
        ui.set_ai_kind(model_kind(id));
        ui.set_ai_model_info(self.ai_model_info().into());
        let presentation = self.ai_info(id).map(|info| &info.presentation);
        let field = |key: &str| presentation.and_then(|p| p[key].as_str()).unwrap_or("");
        ui.set_ai_purpose(field("summary").into());
        ui.set_ai_best_for(field("best_for").into());
        ui.set_ai_outputs(field("outputs").into());
        ui.set_ai_performance(field("performance").into());
        ui.set_ai_limitations(field("limitations").into());
        ui.set_ai_model_license(field("license").into());
        ui.set_ai_interval_note(
            format!(
                "{} starts at a {ms} ms gap between live runs. Lower it for faster updates, raise it to spare the GPU. Live analysis runs one frame at a time on the newest camera frame and stops on any error.",
                model_label(id),
            )
            .into(),
        );
    }

    fn ai_load_defaults(&mut self) {
        let id = self.ai_id();
        let (Some(ui), Ok(d)) = (self.ui(), models::defaults(id)) else {
            return;
        };
        let num = |key: &str| d.get(key).and_then(serde_json::Value::as_f64);
        let flag = |key: &str| d.get(key).and_then(serde_json::Value::as_bool);
        let pick = |key: &str, names: &[&str]| {
            d.get(key)
                .and_then(serde_json::Value::as_str)
                .and_then(|v| names.iter().position(|n| *n == v))
                .map(|i| i as i32)
        };
        if let Some(v) = num("max_results") {
            ui.set_ai_max_results(v as f32);
        }
        if let Some(v) = num("min_confidence") {
            ui.set_ai_min_conf(v as f32);
        }
        if let Some(v) = num("threshold") {
            ui.set_ai_threshold(v as f32);
        }
        if let Some(v) = num("top_k") {
            ui.set_ai_top_k(v as f32);
        }
        if let Some(v) = num("steps") {
            ui.set_ai_steps(v as f32);
        }
        if let Some(v) = num("ensemble") {
            ui.set_ai_ensemble(v as f32);
        }
        if let Some(v) = flag("blendshapes") {
            ui.set_ai_blend(v);
        }
        if let Some(v) = flag("multimask") {
            ui.set_ai_multimask(v);
        }
        if let Some(v) = pick("task", &FLORENCE_TASKS) {
            ui.set_ai_task_index(v);
        }
        if let Some(v) = pick("colormap", &COLORMAPS) {
            ui.set_ai_colormap_index(v);
        }
        let interval = models::model_spec(id).map_or(1.0, |s| s.interval_ms as f32 / 1000.0);
        ui.set_ai_interval(interval);
        let f = |key: &str| num(key).unwrap_or(0.0) as f32;
        ui.set_ai_def(AiDefaults {
            max_results: f("max_results"),
            min_conf: f("min_confidence"),
            threshold: f("threshold"),
            top_k: f("top_k"),
            steps: f("steps"),
            ensemble: f("ensemble"),
            interval,
        });
    }

    /// The options object for the selected model, exactly as the UI shows it, validated against
    /// the size of the frame that will be analyzed.
    fn ai_options(
        &self,
        ui: &MainWindow,
        id: &str,
        size: (u32, u32),
    ) -> std::result::Result<serde_json::Value, String> {
        use serde_json::json;
        let mut options = models::defaults(id).map_err(|e| format!("{e:#}"))?;
        let round2 = |v: f32| (f64::from(v) * 100.0).round() / 100.0;
        let object = options
            .as_object_mut()
            .ok_or_else(|| "The model's default options are not an object.".to_owned())?;
        match model_kind(id) {
            0 | 1 => {
                object.insert(
                    "max_results".into(),
                    json!(ui.get_ai_max_results().round() as i64),
                );
                object.insert("min_confidence".into(), json!(round2(ui.get_ai_min_conf())));
                if model_kind(id) == 1 {
                    object.insert("blendshapes".into(), json!(ui.get_ai_blend()));
                }
            }
            2 => {
                object.insert("threshold".into(), json!(round2(ui.get_ai_threshold())));
                object.insert("top_k".into(), json!(ui.get_ai_top_k().round() as i64));
            }
            3 => {
                let task = FLORENCE_TASKS[usize::try_from(ui.get_ai_task_index())
                    .unwrap_or(0)
                    .min(FLORENCE_TASKS.len() - 1)];
                object.insert("task".into(), json!(task));
            }
            4 => {
                let points = parse_points(&ui.get_ai_points())?;
                let rect = parse_box(&ui.get_ai_box())?;
                if points.is_empty() {
                    object.remove("points");
                } else {
                    let list: Vec<String> = points
                        .iter()
                        .map(|(x, y)| format!("{},{}", fmt_coord(*x), fmt_coord(*y)))
                        .collect();
                    object.insert("points".into(), json!(list));
                }
                match rect {
                    Some([x1, y1, x2, y2]) => {
                        let text = format!(
                            "{},{},{},{}",
                            fmt_coord(x1),
                            fmt_coord(y1),
                            fmt_coord(x2),
                            fmt_coord(y2)
                        );
                        object.insert("box".into(), json!(text));
                    }
                    None => {
                        object.remove("box");
                    }
                }
                object.insert("multimask".into(), json!(ui.get_ai_multimask()));
            }
            kind => {
                let colormap = COLORMAPS[usize::try_from(ui.get_ai_colormap_index())
                    .unwrap_or(0)
                    .min(COLORMAPS.len() - 1)];
                object.insert("colormap".into(), json!(colormap));
                if kind == 6 {
                    object.insert("steps".into(), json!(ui.get_ai_steps().round() as i64));
                    object.insert(
                        "ensemble".into(),
                        json!(ui.get_ai_ensemble().round() as i64),
                    );
                }
            }
        }
        models::validate_options(id, &options, size.0, size.1).map_err(|e| format!("{e:#}"))?;
        Ok(options)
    }

    fn ai_interval(&self, ui: &MainWindow) -> Duration {
        Duration::from_secs_f32(ui.get_ai_interval().max(0.0).min(10.0))
    }

    /// Cancels obsolete requests; completed history remains tied to its original source frame.
    fn ai_reset(&mut self, camera: bool) {
        self.ai.invalidate_request();
        if let Some(engine) = &self.ai.engine {
            engine.cancel();
        }
        if camera {
            self.ai.live = false;
            self.ai.frozen = None;
            self.ai.geometry_pending = false;
        }
        if camera && self.ai.stage == 2 {
            self.ai.stage = 0;
        }
        if let Some(ui) = self.ui() {
            if camera {
                ui.set_ai_frozen_image(Image::default());
            }
        }
    }

    fn ai_geometry_changed(&mut self) {
        if self.preview.is_some() {
            self.ai_reset(true);
            self.ai.geometry_pending = true;
        }
    }

    /// Persists the switch. Turning AI off takes effect first: live stops, results are
    /// invalidated, a running download is cancelled and the engine is dropped.
    fn ai_enabled_toggled(&mut self, on: bool) {
        self.ai_enabled_changed(on, true);
    }

    fn ai_enabled_changed(&mut self, on: bool, persist: bool) {
        if self.ai.closing || on == self.ai.enabled {
            return self.sync();
        }
        if on {
            let saved = if persist {
                preferences::set_ai_enabled(true)
            } else {
                Ok(())
            };
            self.ai.local_pref_override = persist && saved.is_err();
            self.ai.settings_note = match saved {
                Ok(()) => {
                    self.ai.enabled = true;
                    self.ai_start_engine();
                    (
                        "Optional AI is on. Models run on this computer; nothing is downloaded until you confirm it in AI tools."
                            .into(),
                        false,
                    )
                }
                Err(e) => (
                    format!("Cannot save the setting, so optional AI stays off: {e:#}"),
                    true,
                ),
            };
            return self.sync();
        }
        // Disable first, so a failed save can never leave AI running.
        self.ai.enabled = false;
        self.ai.live = false;
        self.ai_reset(true);
        self.ai_clear_history_state();
        if matches!(self.effects.background.as_str(), "blur" | "replace") {
            let mut e = self.effects.clone();
            e.background = "none".into();
            e.background_image = None;
            self.request_effects(e);
            if let Some(ui) = self.ui() {
                ui.set_fx_bg_index(0);
            }
        }
        let saved = if persist {
            preferences::set_ai_enabled(false)
        } else {
            Ok(())
        };
        self.ai.local_pref_override = persist && saved.is_err();
        if let Some(job) = &self.ai.library {
            job.cancel.store(true, Ordering::Release);
        }
        self.ai_drop_engine();
        self.ai.settings_note = match saved {
            Err(e) => (
                format!(
                    "Optional AI is off for this session, but the setting could not be saved: {e:#}"
                ),
                true,
            ),
            Ok(()) => ("Optional AI is off.".into(), false),
        };
        self.sync();
    }

    /// Starts the engine, which only reads the local model files: no process, no network.
    fn ai_start_engine(&mut self) {
        let ai = &self.ai;
        if ai.closing || !ai.enabled || ai.engine.is_some() || ai.library.is_some() {
            return;
        }
        match Engine::new(RuntimeConfig::default()) {
            Ok(engine) => {
                self.ai.status = engine.status();
                self.ai.engine = Some(engine);
                self.ai.completed = 0;
                self.ai.last_error = None;
            }
            Err(e) => self.ai_note(format!("Cannot start the local model library: {e:#}"), true),
        }
        self.ai_push_models();
    }

    fn ai_refresh_library(&mut self) {
        if !self.ai.enabled || self.ai.closing || self.ai.library.is_some() {
            return;
        }
        if let Some(engine) = &self.ai.engine {
            let result = engine.refresh_catalog();
            self.ai.status = engine.status();
            if let Err(error) = result {
                self.ai_note(format!("Cannot refresh local models: {error:#}"), true);
            }
        } else {
            self.ai_start_engine();
        }
        self.sync();
    }

    /// Drops the engine, which stops its model process and joins its thread. The catalog is kept
    /// only while a library job runs, so the library panel does not blank out meanwhile.
    fn ai_drop_engine(&mut self) {
        self.ai.engine = None;
        if self.ai.library.is_none() {
            self.ai.status = EngineStatus::default();
        }
        self.ai.completed = 0;
        self.ai.last_error = None;
        self.ai_push_models();
    }

    fn ai_download(&mut self) {
        self.ai_library_start(true);
    }

    fn ai_delete(&mut self) {
        self.ai_library_start(false);
    }

    /// Downloads or deletes the selected model on its own thread. Only an explicit, confirmed
    /// press gets here; analysis never downloads anything.
    fn ai_library_start(&mut self, download: bool) {
        let id = self.ai_id();
        if self.ai.closing || !self.ai.enabled || self.ai.library.is_some() {
            return;
        }
        let what = if download {
            "Model download"
        } else {
            "Model delete"
        };
        if let Err(error) = preferences::require_ai_enabled(what) {
            self.ai_enabled_changed(false, false);
            self.ai.lib_note = (format!("{error:#}"), true);
            return self.sync();
        }
        match (self.ai_state(id), download) {
            (ModelState::NotInstalled | ModelState::Unavailable(_), true) => {}
            (ModelState::Ready { .. } | ModelState::Unavailable(_), false) => {}
            _ => {
                self.ai.lib_note = (
                    if download {
                        "That model is already installed, or the library is not ready."
                    } else {
                        "That model is not installed."
                    }
                    .into(),
                    true,
                );
                return self.sync();
            }
        }
        self.ai.live = false;
        self.ai_reset(false);
        // Dropping the engine stops its resident model process, so nothing holds the files.
        self.ai.engine = None;
        self.ai.completed = 0;
        self.ai.last_error = None;
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let tx = self.events_sender();
        let spawned = std::thread::Builder::new()
            .name("model-library".into())
            .spawn(move || {
                let result = Client::new(RuntimeConfig::default()).and_then(|client| {
                    if download {
                        client.download(id, &flag)
                    } else {
                        client.delete(id)
                    }
                });
                let cancelled = result
                    .as_ref()
                    .err()
                    .is_some_and(|e| e.downcast_ref::<models::Cancelled>().is_some());
                let _ = tx.send(Event::Library(LibraryDone {
                    model: id,
                    download,
                    cancelled,
                    result: result.map_err(|e| format!("{e:#}")),
                }));
            });
        match spawned {
            Ok(handle) => {
                self.ai.library = Some(LibraryJob {
                    model: id,
                    download,
                    cancel,
                    started: Instant::now(),
                    handle,
                });
                self.ai.lib_note = (String::new(), false);
            }
            Err(e) => {
                self.ai.lib_note = (format!("Cannot start the model job: {e}"), true);
                self.ai_start_engine();
            }
        }
        self.ai_push_models();
        self.sync();
    }

    fn ai_cancel_download(&mut self) {
        if let Some(job) = self.ai.library.as_ref().filter(|j| j.download) {
            job.cancel.store(true, Ordering::Release);
            self.ai.lib_note = ("Stopping the download…".into(), false);
        }
        self.sync();
    }

    /// The library thread finished: report it and read the library again with a fresh engine.
    fn on_library_done(&mut self, done: LibraryDone) {
        if let Some(job) = self.ai.library.take() {
            let _ = job.handle.join();
        }
        let label = model_label(done.model);
        self.ai.lib_note = match (&done.result, done.cancelled) {
            (Ok(()), _) if done.download => {
                (format!("{label} is downloaded and ready to run."), false)
            }
            (Ok(()), _) => (format!("{label} was deleted from this computer."), false),
            (Err(_), true) => (
                format!("The {label} download was stopped. Nothing partial was kept."),
                false,
            ),
            (Err(e), false) => (
                format!(
                    "{} of {label} failed: {e}",
                    if done.download { "Download" } else { "Delete" }
                ),
                true,
            ),
        };
        let note = self.ai.lib_note.clone();
        if note.1 {
            self.fail(note.0);
        } else {
            self.say(note.0);
        }
        if !self.ai.enabled {
            self.ai.status = EngineStatus::default();
        } else {
            self.ai_start_engine();
        }
        self.ai_push_models();
    }

    fn ai_select_model(&mut self, index: i32) {
        let Some(i) = usize::try_from(index).ok().filter(|i| *i < MODEL_IDS.len()) else {
            return;
        };
        if i != self.ai.model {
            self.ai.model = i;
            self.ai_reset(false);
            self.ai_load_defaults();
            self.ai_note("", false);
        }
        self.ai_push_models();
        self.sync();
    }

    fn ai_defaults(&mut self) {
        self.ai_load_defaults();
        self.sync();
    }

    fn ai_run(&mut self) {
        self.ai_submit(true);
        self.sync();
    }

    fn ai_stop(&mut self) {
        self.ai.stop_session();
        if let Some(engine) = &self.ai.engine {
            engine.stop();
            self.ai.status = engine.status();
        }
        self.ai_note(
            if self.ai.status.stopping {
                "Stopping and unloading the model…"
            } else {
                "Stopped."
            },
            false,
        );
        self.sync();
    }

    fn ai_live_toggled(&mut self, on: bool) {
        if on == self.ai.live {
            return self.sync();
        }
        if on {
            if let Some(st) = self.ai.engine.as_ref().map(Engine::status) {
                self.ai_apply(st);
            }
            if !self.ai_can_live() {
                self.ai_note(
                    self.ai_run_block(false)
                        .unwrap_or_else(|| "Live analysis needs an installed model.".into()),
                    !self.ai.working() && !self.ai.live,
                );
                return self.sync();
            }
            self.ai.begin_session();
            self.ai.last_key = None;
            self.ai.next_at = None;
            self.ai.follow = true;
        }
        self.ai.live = on;
        self.sync();
    }

    fn ai_set_stage(&mut self, mode: i32) {
        match mode {
            1 if self.ai.result.is_none() => {
                self.ai_note("There is no analysis result to show yet.", true)
            }
            2 if self.ai.frozen.is_none() => {
                self.ai_note("No frame is frozen. Freeze one in the SAM options.", true)
            }
            0..=2 => self.ai.select_stage(mode),
            _ => {}
        }
        self.sync();
    }

    fn ai_source_toggled(&mut self, on: bool) {
        if let (Some(ui), Some(a)) = (self.ui(), self.ai.result.clone()) {
            let frame = if on { &a.source } else { &a.annotated };
            if let Some(image) = rgb_image(frame.width, frame.height, &frame.rgb) {
                ui.set_ai_image(image);
            }
        }
    }

    fn ai_freeze(&mut self) {
        let frame = self
            .preview
            .as_ref()
            .and_then(|p| p.latest_frame())
            .filter(|f| frame_ok(f));
        let Some(frame) = frame else {
            self.ai_note("Start the camera preview first.", true);
            return self.sync();
        };
        if let Some(ui) = self.ui() {
            ui.set_ai_points(SharedString::new());
            ui.set_ai_box(SharedString::new());
        }
        self.ai_note(
            format!(
                "Frozen camera frame {}. Pick points or draw a box on the stage.",
                frame.sequence
            ),
            false,
        );
        self.ai.frozen = Some(frame);
        self.ai.select_stage(2);
        self.ai_draw_frozen();
        self.sync();
    }

    fn ai_unfreeze(&mut self) {
        self.ai.frozen = None;
        if self.ai.stage == 2 {
            self.ai.select_stage(0);
        }
        if let Some(ui) = self.ui() {
            ui.set_ai_points(SharedString::new());
            ui.set_ai_box(SharedString::new());
            ui.set_ai_frozen_image(Image::default());
        }
        self.sync();
    }

    fn ai_clear_prompts(&mut self) {
        if let Some(ui) = self.ui() {
            ui.set_ai_points(SharedString::new());
            ui.set_ai_box(SharedString::new());
        }
        self.ai_draw_frozen();
    }

    fn ai_prompts_edited(&mut self) {
        self.ai_draw_frozen();
    }

    /// Repaints the frozen frame with the current points and box.
    fn ai_draw_frozen(&self) {
        let (Some(ui), Some(frame)) = (self.ui(), self.ai.frozen.as_ref()) else {
            return;
        };
        let points = lenient_points(&ui.get_ai_points());
        let rect = parse_box(&ui.get_ai_box()).ok().flatten();
        let mut buffer = SharedPixelBuffer::<Rgb8Pixel>::new(frame.width, frame.height);
        let bytes = buffer.make_mut_bytes();
        bytes.copy_from_slice(&frame.rgb);
        draw_prompts(bytes, frame.width, frame.height, &points, rect);
        ui.set_ai_frozen_image(Image::from_rgb8(buffer));
    }

    fn ai_pick(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, w: f32, h: f32) {
        match self.ai_pick_inner((x0, y0), (x1, y1), (w, h)) {
            Ok(message) => self.ai_note(message, false),
            Err(e) => self.ai_note(e, true),
        }
        self.ai_draw_frozen();
        self.sync();
    }

    /// `press`/`release` are pointer positions over the stage image, `view` its size.
    fn ai_pick_inner(
        &mut self,
        press: (f32, f32),
        release: (f32, f32),
        view: (f32, f32),
    ) -> std::result::Result<String, String> {
        let (Some(ui), Some(frame)) = (self.ui(), self.ai.frozen.clone()) else {
            return Err("Freeze a camera frame first.".into());
        };
        let size = (frame.width, frame.height);
        let (iw, ih) = (f64::from(size.0), f64::from(size.1));
        if ui.get_ai_pick_mode() == 0 {
            let (x, y) = contain_to_pixel(view, size, release, false)
                .ok_or("That click is in the empty margin around the frame; click on the image.")?;
            let existing = ui.get_ai_points().to_string();
            if parse_points(&existing)?.len() >= SAM_MAX_POINTS {
                return Err(format!(
                    "At most {SAM_MAX_POINTS} points; clear some first."
                ));
            }
            let (px, py) = (x.floor().min(iw - 1.0), y.floor().min(ih - 1.0));
            let mut text = existing.trim().trim_end_matches(';').trim_end().to_owned();
            if !text.is_empty() {
                text.push_str("; ");
            }
            text.push_str(&format!("{},{}", fmt_coord(px), fmt_coord(py)));
            ui.set_ai_points(text.into());
            Ok(format!(
                "Added point {},{} on frozen frame {}.",
                fmt_coord(px),
                fmt_coord(py),
                frame.sequence
            ))
        } else {
            if (release.0 - press.0).hypot(release.1 - press.1) < MIN_BOX_DRAG {
                return Err(
                    "Drag across the object to draw a box, or switch the pointer action to points."
                        .into(),
                );
            }
            let (a, b) = (
                contain_to_pixel(view, size, press, true),
                contain_to_pixel(view, size, release, true),
            );
            let (Some(a), Some(b)) = (a, b) else {
                return Err("The frozen frame is not drawn yet.".into());
            };
            let (x1, y1) = (a.0.min(b.0).floor(), a.1.min(b.1).floor());
            let (x2, y2) = (a.0.max(b.0).ceil().min(iw), a.1.max(b.1).ceil().min(ih));
            if x2 <= x1 || y2 <= y1 {
                return Err("That box is empty; drag across part of the image.".into());
            }
            let text = format!(
                "{},{},{},{}",
                fmt_coord(x1),
                fmt_coord(y1),
                fmt_coord(x2),
                fmt_coord(y2)
            );
            ui.set_ai_box(text.clone().into());
            Ok(format!("Box {text} on frozen frame {}.", frame.sequence))
        }
    }

    fn ai_submit(&mut self, manual: bool) {
        if !self.ai.enabled {
            self.ai.live = false;
            return;
        }
        // The timer re-reads the setting every second, so live runs do not touch the file.
        if manual && let Err(error) = preferences::require_ai_enabled("Model analysis") {
            self.ai_enabled_changed(false, false);
            self.ai_note(format!("{error:#}"), true);
            return self.sync();
        }
        let Some(st) = self.ai.engine.as_ref().map(Engine::status) else {
            if manual {
                self.ai_note("The local model library is not ready.", true);
            }
            return;
        };
        // Take in any completion first so the fresh baseline can only count the new request.
        if self.ai_apply(st) {
            self.sync();
        }
        if let Some(reason) = self.ai_run_block(manual) {
            if manual {
                self.ai_note(reason, !self.ai.working() && !self.ai.live);
            }
            return;
        }
        let id = self.ai_id();
        let (Some(frame), Some(ui)) = (self.ai_source_frame(), self.ui()) else {
            return;
        };
        let options = match self.ai_options(&ui, id, (frame.width, frame.height)) {
            Ok(options) => options,
            Err(e) => {
                self.ai.live = false;
                self.ai_note(e, true);
                return self.sync();
            }
        };
        let key = (self.ai.token, frame.sequence, options.to_string());
        if !manual && self.ai.last_key.as_ref() == Some(&key) {
            return;
        }
        let seq = frame.sequence;
        let interval = self.ai_interval(&ui);
        let sent = self
            .ai
            .engine
            .as_ref()
            .map(|e| e.submit(id, options, frame));
        match sent {
            Some(Ok(())) => {
                self.ai.begin_session();
                self.ai.last_key = Some(key);
                self.ai.inflight = Some(Inflight {
                    token: self.ai.token,
                    model: id,
                    seq,
                });
                self.ai.status.busy = true;
                self.ai.next_at = Some(Instant::now() + interval);
                if manual {
                    self.ai.follow = true;
                }
                self.ai_note(
                    format!("Analyzing frame {seq} with {}…", model_label(id)),
                    false,
                );
            }
            Some(Err(e)) => {
                self.ai.live = false;
                self.ai_note(format!("{e:#}"), true);
                self.fail(format!("Model analysis: {e:#}"));
            }
            None => {}
        }
        self.sync();
    }

    fn ai_poll(&mut self) {
        let Some(st) = self.ai.engine.as_ref().map(Engine::status) else {
            return;
        };
        if self.ai_apply(st) {
            self.sync();
        }
    }

    fn ai_apply(&mut self, st: EngineStatus) -> bool {
        let models_changed = match (&self.ai.status.catalog, &st.catalog) {
            (None, None) => self.ai.status.busy != st.busy || self.ai.status.error != st.error,
            (Some(a), Some(b)) => !Arc::ptr_eq(a, b) && !same_library(a, b),
            _ => true,
        };
        let changed = models_changed
            || self.ai.status.busy != st.busy
            || self.ai.status.stopping != st.stopping
            || self.ai.status.error != st.error
            || self.ai.completed != st.completed;
        if self.ai.status.stopping && !st.stopping && !st.busy && st.error.is_none() {
            self.ai_note("Stopped.", false);
        }
        if st.completed != self.ai.completed {
            self.ai.completed = st.completed;
            self.ai_completed(&st);
        }
        if st.error != self.ai.last_error {
            self.ai.last_error = st.error.clone();
            if let Some(e) = &st.error {
                // Any error, including a cleanup warning next to a good result, ends live mode.
                self.ai.live = false;
                if st.catalog.is_some() {
                    self.ai_note(e.clone(), true);
                    self.fail(format!("Model analysis: {e}"));
                } else {
                    self.fail(format!("Local model library unavailable: {e}"));
                }
            }
        }
        self.ai.status = st;
        if models_changed {
            self.ai_push_models();
        }
        changed
    }

    /// A job finished. It is shown only if it is the request still current for this token, the
    /// same model and the same source frame; anything else is a late result and is dropped.
    fn ai_completed(&mut self, st: &EngineStatus) {
        let Some(sent) = self.ai.inflight.take() else {
            return;
        };
        if sent.token != self.ai.token {
            return;
        }
        match &st.result {
            Some(a) if a.model == sent.model && a.source.sequence == sent.seq => {
                let Some(snapshot) = st.snapshot.clone() else {
                    self.ai.live = false;
                    return self.ai_note(
                        "The result has no history snapshot and was discarded.",
                        true,
                    );
                };
                let record = self.ai.history.push(snapshot);
                if self.ai.history_rows.row_count() == self.ai.history.len() {
                    self.ai.history_rows.remove(self.ai.history.len() - 1);
                }
                self.ai.history_rows.insert(
                    0,
                    HistoryRow {
                        token: record.id.to_string().into(),
                        title: format!("#{} · {}", record.id, model_label(&record.snapshot.model))
                            .into(),
                        detail: format!(
                            "Frame {} · {:.0} ms · {}×{}",
                            record.snapshot.sequence,
                            record.snapshot.elapsed.as_secs_f64() * 1000.0,
                            record.snapshot.width,
                            record.snapshot.height
                        )
                        .into(),
                    },
                );
                self.ai.latest = Some(a.clone());
                self.ai.latest_record = Some(record.clone());
                if let Some(ui) = self.ui() {
                    ui.invoke_ai_history_inserted();
                }
                self.ai_note(format!("Analyzed frame {}.", a.source.sequence), false);
                if !self.ai.history_paused || self.ai.history_selected.is_none() {
                    self.ai.history_selected = Some(record);
                    self.ai.history_view_token = self.ai.history_view_token.wrapping_add(1);
                    self.ai.history_pending = None;
                    self.ai_publish(a.clone());
                }
            }
            Some(_) => {
                self.ai.live = false;
                self.ai_note(
                    "A result that did not match the request was discarded.",
                    true,
                );
            }
            None => {
                self.ai.live = false;
                let e = st
                    .error
                    .clone()
                    .unwrap_or_else(|| "The analysis finished without a result.".into());
                self.ai_note(e.clone(), true);
                self.fail(format!("Model analysis: {e}"));
            }
        }
    }

    fn ai_select_history(&mut self, token: SharedString) {
        if !self.ai.enabled || self.ai.closing {
            return;
        }
        let Some(record) = token
            .parse::<u64>()
            .ok()
            .and_then(|id| self.ai.history.get(id).cloned())
        else {
            return;
        };
        self.ai.history_paused = true;
        self.ai.follow = false;
        self.ai.history_view_token = self.ai.history_view_token.wrapping_add(1);
        if self
            .ai
            .latest_record
            .as_ref()
            .is_some_and(|latest| latest.id == record.id)
            && let Some(analysis) = self.ai.latest.clone()
        {
            self.ai.history_pending = None;
            self.ai.history_selected = Some(record);
            self.ai.select_stage(1);
            self.ai_publish(analysis);
        } else {
            self.ai.history_pending = Some((self.ai.history_view_token, record));
            self.ai_start_history_restore();
        }
        self.sync();
    }

    fn ai_start_history_restore(&mut self) {
        if self.ai.history_loading.is_some() {
            return;
        }
        let Some((token, record)) = self.ai.history_pending.take() else {
            return;
        };
        let tx = self.events_sender();
        match std::thread::Builder::new()
            .name("history-restore".into())
            .spawn(move || {
                let result = record
                    .snapshot
                    .restore()
                    .map(Arc::new)
                    .map_err(|error| format!("{error:#}"));
                let _ = tx.send(Event::AiHistoryRestored {
                    token,
                    record,
                    result,
                });
            }) {
            Ok(handle) => self.ai.history_loading = Some(handle),
            Err(error) => self.ai_note(format!("Cannot open the history result: {error}"), true),
        }
    }

    fn on_ai_history_restored(
        &mut self,
        token: u64,
        record: HistoryRecord,
        result: std::result::Result<Arc<Analysis>, String>,
    ) {
        if let Some(handle) = self.ai.history_loading.take() {
            let _ = handle.join();
        }
        if self.ai.enabled
            && !self.ai.closing
            && self.ai.history_paused
            && token == self.ai.history_view_token
        {
            match result {
                Ok(analysis) => {
                    self.ai.history_selected = Some(record);
                    self.ai.select_stage(1);
                    self.ai_publish(analysis);
                }
                Err(error) => {
                    self.ai_note(format!("Cannot open the history result: {error}"), true)
                }
            }
        }
        self.ai_start_history_restore();
    }

    fn ai_history_follow_toggled(&mut self, on: bool) {
        if !self.ai.enabled || self.ai.closing || self.ai.history_paused == !on {
            return;
        }
        self.ai.history_paused = !on;
        if on {
            self.ai.history_view_token = self.ai.history_view_token.wrapping_add(1);
            self.ai.history_pending = None;
            if let (Some(record), Some(analysis)) =
                (self.ai.latest_record.clone(), self.ai.latest.clone())
            {
                self.ai.history_selected = Some(record);
                self.ai.select_stage(1);
                self.ai_publish(analysis);
            }
        }
        self.sync();
    }

    fn ai_history_raw_toggled(&mut self, on: bool) {
        let Some(ui) = self.ui() else { return };
        ui.set_ai_history_raw(on);
        if on && self.ai.enabled {
            self.ai.history_paused = true;
            if let Some(analysis) = &self.ai.result {
                ui.set_ai_raw(
                    raw_view(&analysis.output.content_type, &analysis.output.bytes).into(),
                );
            }
        }
        self.sync();
    }

    fn ai_clear_history_state(&mut self) {
        self.ai.history.clear();
        self.ai.history_rows.set_vec(Vec::new());
        self.ai.history_view_token = self.ai.history_view_token.wrapping_add(1);
        self.ai.history_pending = None;
        self.ai.history_selected = None;
        self.ai.latest_record = None;
        self.ai.latest = None;
        self.ai.result = None;
        self.ai.result_info.clear();
        if self.ai.stage == 1 {
            self.ai.stage = 0;
        }
        if let Some(ui) = self.ui() {
            ui.set_ai_image(Image::default());
            ui.set_ai_text(SharedString::new());
            ui.set_ai_raw(SharedString::new());
        }
    }

    fn ai_history_clear(&mut self) {
        self.ai_clear_history_state();
        self.ai_note(
            "History cleared. New results will start a fresh list.",
            false,
        );
        self.sync();
    }

    fn ai_publish(&mut self, a: Arc<Analysis>) {
        let Some(ui) = self.ui() else { return };
        let shown = if ui.get_ai_show_source() {
            &a.source
        } else {
            &a.annotated
        };
        let Some(image) = rgb_image(shown.width, shown.height, &shown.rgb) else {
            self.ai.live = false;
            return self.ai_note(
                "The result image has an invalid size and was discarded.",
                true,
            );
        };
        ui.set_ai_image(image);
        ui.set_ai_text(a.text.clone().into());
        if ui.get_ai_history_raw() {
            ui.set_ai_raw(raw_view(&a.output.content_type, &a.output.bytes).into());
        }
        self.ai.result_info = format!(
            "#{} · {}\nFrame {} · {}×{} · {:.0} ms",
            self.ai
                .history_selected
                .as_ref()
                .map_or(0, |record| record.id),
            model_label(&a.model),
            a.source.sequence,
            a.source.width,
            a.source.height,
            a.output.elapsed.as_secs_f64() * 1000.0
        );
        if self.ai.follow {
            self.ai.stage = 1;
            self.ai.follow = false;
        }
        self.ai.result = Some(a);
    }

    fn ai_tick(&mut self) {
        if self.ai.closing {
            self.ai_poll();
            if !self.ai_working() {
                // Finished exports have enqueued their replies; do not hide a failed save.
                while let Ok(event) = self.events.try_recv() {
                    self.on_event(event);
                }
                self.ai_poll();
                self.ai.closing = false;
                if self.ai.status.error.is_some() || self.ai.note.1 {
                    self.say(
                        "Analysis or export stopped with an error. Review it before closing again.",
                    );
                    self.sync();
                } else if let Some(ui) = self.ui() {
                    let _ = ui.hide();
                }
            }
            return;
        }
        if !self.ai.local_pref_override
            && self
                .ai
                .prefs_checked
                .is_none_or(|at| at.elapsed() >= Duration::from_secs(1))
        {
            self.ai.prefs_checked = Some(Instant::now());
            match preferences::load() {
                Ok(prefs) if prefs.ai_enabled != self.ai.enabled => {
                    self.ai_enabled_changed(prefs.ai_enabled, false)
                }
                Ok(_) => {}
                Err(error) => {
                    self.ai_enabled_changed(false, false);
                    self.ai.settings_note = (format!("Optional AI stays off: {error:#}"), true);
                    self.sync();
                }
            }
        }
        if self.ai.engine.is_none() {
            return;
        }
        if self.ai.last_poll.is_none_or(|t| t.elapsed() >= AI_POLL) {
            self.ai.last_poll = Some(Instant::now());
            self.ai_poll();
        }
        if !self.ai.enabled {
            return;
        }
        if self.ai.live && !self.ai.working() && self.ai.next_at.is_none_or(|t| Instant::now() >= t)
        {
            self.ai_submit(false);
        }
    }

    /// True while a model download/delete, a result export or a virtual camera change (which may
    /// wait on an administrator prompt) is still running.
    fn ai_working(&mut self) -> bool {
        self.ai.saving.retain(|h| !h.is_finished());
        !self.ai.saving.is_empty()
            || self.ai.history_loading.is_some()
            || self.vcam_pending > 0
            || self.vcam_stopping
            || self
                .ai
                .library
                .as_ref()
                .is_some_and(|job| !job.handle.is_finished())
    }

    fn close_requested(&mut self) -> CloseRequestResponse {
        self.save_script_store();
        self.ai.live = false;
        // Cancelling kills the owned local processes at once, so there is nothing to drain.
        self.ai_reset(false);
        self.ai.history_view_token = self.ai.history_view_token.wrapping_add(1);
        self.ai.history_pending = None;
        if let Some(job) = &self.ai.library {
            job.cancel.store(true, Ordering::Release);
        }
        if !self.ai_working() {
            return CloseRequestResponse::HideWindow;
        }
        self.ai.closing = true;
        self.say(if self.vcam_pending > 0 {
            "Closing: waiting for the virtual camera change to finish. If an administrator prompt is open, answer or dismiss it. This window closes by itself."
        } else {
            "Closing: stopping the model download or finishing background result work. This window closes by itself."
        });
        self.sync();
        CloseRequestResponse::KeepWindowShown
    }

    fn ai_save_raw(&mut self) {
        self.ai_save(false);
    }

    fn ai_save_visual(&mut self) {
        self.ai_save(true);
    }

    fn ai_save(&mut self, visual: bool) {
        if self.ai.history_loading.is_some() || self.ai.history_pending.is_some() {
            self.ai_note(
                "Wait for the selected history result to finish opening.",
                false,
            );
            return self.sync();
        }
        if self.ai.saving.iter().any(|job| !job.is_finished()) {
            return;
        }
        let Some(analysis) = self.ai.result.clone() else {
            self.ai_note("There is no analysis result to save yet.", true);
            return self.sync();
        };
        let Some(ui) = self.ui() else { return };
        let json = models::model_spec(&analysis.model).is_ok_and(|s| s.output == "json");
        let (suffix, ext) = match (visual, json) {
            (true, _) => ("-annotated", "png"),
            (false, true) => ("", "json"),
            (false, false) => ("", "png"),
        };
        let path = match next_model_path(&ui.get_ai_out_dir(), &analysis.model, suffix, ext) {
            Ok(path) => path,
            Err(e) => {
                self.ai_note(e, true);
                return self.sync();
            }
        };
        let tx = self.events_sender();
        let spawned = std::thread::Builder::new()
            .name("model-export".into())
            .spawn(move || {
                let saved = if visual {
                    models::save_visualization(&analysis, &path)
                } else {
                    models::save_result(&analysis, &path)
                };
                let _ = tx.send(Event::AiSaved(
                    saved
                        .map(|()| path.display().to_string())
                        .map_err(|e| format!("{e:#}")),
                ));
            });
        match spawned {
            Ok(handle) => {
                self.ai.saving.push(handle);
                self.ai_note("Saving…", false);
            }
            Err(e) => self.ai_note(format!("Cannot start the export: {e}"), true),
        }
        self.sync();
    }

    fn on_ai_saved(&mut self, result: std::result::Result<String, String>) {
        match result {
            Ok(path) => {
                self.ai_note(format!("Saved {path}"), false);
                self.say(format!("Saved {path}"));
            }
            Err(e) => {
                self.ai_note(format!("Cannot save: {e}"), true);
                self.fail(format!("Cannot save the AI result: {e}"));
            }
        }
    }

    fn ai_sync(&self, ui: &MainWindow) {
        let ai = &self.ai;
        ui.set_ai_enabled(ai.enabled);
        ui.set_settings_note(ai.settings_note.0.clone().into());
        ui.set_settings_error(ai.settings_note.1);
        ui.set_ai_ready(ai.live || self.ai_can_live());
        ui.set_ai_busy(ai.working());
        ui.set_ai_stopping(ai.status.stopping);
        ui.set_ai_can_stop(
            ai.engine.is_some()
                && (ai.working()
                    || ai.live
                    || ai.prior_stage.is_some()
                    || ai.status.catalog.as_ref().is_some_and(|models| {
                        models.iter().any(|model| model.loaded || model.loading)
                    })),
        );
        ui.set_ai_saving(ai.saving.iter().any(|job| !job.is_finished()));
        ui.set_ai_live(ai.live);
        ui.set_ai_has_result(ai.result.is_some());
        ui.set_ai_frozen(ai.frozen.is_some());
        ui.set_stage_mode(ai.stage);
        ui.set_ai_model_index(ai.model as i32);
        self.ai_sync_library(ui);
        let block = self.ai_run_block(true);
        ui.set_ai_can_run(block.is_none());
        ui.set_ai_run_hint(block.unwrap_or_default().into());
        ui.set_ai_note(ai.note.0.clone().into());
        ui.set_ai_note_error(ai.note.1);
        ui.set_ai_history_count(self.ai.history.len() as i32);
        ui.set_ai_history_follow(!self.ai.history_paused);
        ui.set_ai_history_loading(self.ai.history_loading.is_some());
        ui.set_ai_history_selected(
            self.ai
                .history_selected
                .as_ref()
                .map(|record| record.id.to_string())
                .unwrap_or_default()
                .into(),
        );
        ui.set_ai_history_memory(human_bytes(self.ai.history.stored_bytes() as u64).into());
        let frozen_note = match &ai.frozen {
            Some(f) => format!(
                "Frozen camera frame {} ({}×{}). SAM analyzes exactly this frame, not the live camera.",
                f.sequence, f.width, f.height
            ),
            None => "No frozen frame: SAM analyzes the newest camera frame when you run it. Freeze a frame to pick points on exactly the image that is analyzed.".to_owned(),
        };
        let result_info = if ai.result.is_some() && self.has_frame && !ai.history_paused {
            format!(
                "{}. The camera is now at frame {}.",
                ai.result_info, self.last_seq
            )
        } else {
            ai.result_info.clone()
        };
        ui.set_stage_caption(
            match ai.stage {
                1 => result_info.as_str(),
                2 => frozen_note.as_str(),
                _ => "",
            }
            .into(),
        );
        ui.set_ai_frozen_note(frozen_note.into());
        ui.set_ai_result_info(ai.result_info.clone().into());
    }

    /// Local library panel for the selected model: state, device, disk use and which actions apply.
    fn ai_sync_library(&self, ui: &MainWindow) {
        let ai = &self.ai;
        let id = self.ai_id();
        let state = self.ai_state(id);
        let info = self.ai_info(id);
        let job = ai.library.as_ref();
        let (text, tone) = match (job, &state) {
            (Some(j), _) if j.model == id => (
                if j.download {
                    "Downloading…"
                } else {
                    "Deleting…"
                },
                2,
            ),
            (Some(_), _) => ("Library busy", 2),
            (None, ModelState::Off) => ("AI is off", 0),
            (None, ModelState::Checking) => ("Reading library…", 0),
            (None, ModelState::Missing) => ("Not listed", 3),
            (None, ModelState::NotInstalled) => ("Not installed", 2),
            (None, ModelState::Unavailable(_)) => ("Installed · cannot run", 3),
            (None, ModelState::Ready { loading: true, .. }) => ("Loading…", 2),
            (None, ModelState::Ready { loaded: true, .. }) => ("Loaded", 1),
            (None, ModelState::Ready { .. }) => ("Installed", 1),
        };
        let device = info
            .map(|m| m.device.as_str())
            .filter(|d| !d.is_empty())
            .map(|d| format!("Running on {d}"))
            .unwrap_or_default();
        let disk = match info {
            Some(m) if m.installed => format!("{} on disk.", human_bytes(m.disk_bytes)),
            Some(m) => match text_of(&m.presentation["download_size"]) {
                Some(size) => format!("Not on this computer. Download size: {size}."),
                None => "Not on this computer.".to_owned(),
            },
            None => String::new(),
        };
        let idle = ai.enabled && !ai.closing && job.is_none();
        let job_note = job.map_or_else(String::new, |j| {
            let label = model_label(j.model);
            let secs = j.started.elapsed().as_secs();
            if !j.download {
                format!("Deleting {label}…")
            } else if j.cancel.load(Ordering::Acquire) {
                format!("Stopping the {label} download…")
            } else {
                format!(
                    "Downloading {label}… {}:{:02} elapsed. Large models take several minutes and the first download also sets up the runtime. No progress figure is available.",
                    secs / 60,
                    secs % 60
                )
            }
        });
        ui.set_ai_lib_state(text.into());
        ui.set_ai_lib_tone(tone);
        ui.set_ai_lib_device(device.into());
        ui.set_ai_lib_disk(disk.into());
        ui.set_ai_lib_job(job.is_some());
        ui.set_ai_lib_can_cancel(
            job.is_some_and(|j| j.download && !j.cancel.load(Ordering::Acquire)),
        );
        ui.set_ai_lib_job_note(job_note.into());
        ui.set_ai_lib_can_download(
            idle && matches!(state, ModelState::NotInstalled | ModelState::Unavailable(_)),
        );
        ui.set_ai_lib_can_delete(
            idle && matches!(state, ModelState::Ready { .. } | ModelState::Unavailable(_)),
        );
        ui.set_ai_lib_note(ai.lib_note.0.clone().into());
        ui.set_ai_lib_note_error(ai.lib_note.1);
    }

    fn shutdown(&mut self) {
        self.save_script_store();
        // Stop the recording and the stream on the worker (these calls block), then wait for it.
        if let Some(preview) = self.preview.take() {
            if self.media.recording {
                let _ = self
                    .media_tx
                    .as_ref()
                    .map(|tx| tx.send(MediaCmd::StopRecord(preview.clone())));
            }
            let _ = self
                .media_tx
                .as_ref()
                .map(|tx| tx.send(MediaCmd::Stop(preview)));
        }
        let _ = self
            .backend_tx
            .as_ref()
            .map(|tx| tx.send(BackendCmd::Close));
        self.media_tx = None;
        self.backend_tx = None;
        self.audio_tx = None;
        self.vcam_tx = None;
        if let Some(handle) = self.vcam_stop.take() {
            let _ = handle.join();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        // Cancelling kills the download's process group at once; delete is short.
        if let Some(job) = self.ai.library.take() {
            job.cancel.store(true, Ordering::Release);
            let _ = job.handle.join();
        }
        self.ai.engine = None;
        self.ai.history_pending = None;
        if let Some(handle) = self.ai.history_loading.take() {
            let _ = handle.join();
        }
        // Anything still queued in the channel (e.g. a preview that finished starting) drops here.
        while self.events.try_recv().is_ok() {}
    }
}

fn zone_from(a: &PauseAreaInfo) -> Zone {
    let t = |v: f64| trim_num(v as f32);
    Zone {
        active: a.active,
        x: t(a.x),
        y: t(a.y),
        w: t(a.width),
        h: t(a.height),
        pan: t(a.pan),
        tilt: t(a.tilt),
    }
}

/// Opens the camera window. `device` is a `/dev/videoN` id or camera name to select at start.
pub fn run(device: Option<String>) -> Result<()> {
    let win = MainWindow::new().map_err(|e| anyhow!("cannot create the window: {e}"))?;

    let (event_tx, events) = mpsc::channel();
    let (backend_tx, backend_rx) = mpsc::channel();
    let (media_tx, media_rx) = mpsc::channel();
    let (audio_tx, audio_rx) = mpsc::channel();
    let (vcam_tx, vcam_rx) = mpsc::channel();
    let mut workers = vec![
        std::thread::Builder::new()
            .name("camera-backend".into())
            .spawn({
                let tx = event_tx.clone();
                move || backend_worker(backend_rx, tx)
            })?,
        std::thread::Builder::new()
            .name("camera-media".into())
            .spawn({
                let tx = event_tx.clone();
                move || media_worker(media_rx, tx)
            })?,
        std::thread::Builder::new()
            .name("camera-audio".into())
            .spawn({
                let tx = event_tx.clone();
                move || audio_worker(audio_rx, tx)
            })?,
    ];
    // Joined at shutdown once its channel is closed; the window stays open until a pending
    // create or remove (which may wait on a polkit prompt) has finished.
    workers.push(
        std::thread::Builder::new()
            .name("camera-virtual".into())
            .spawn({
                let tx = event_tx.clone();
                move || vcam_worker(vcam_rx, tx)
            })?,
    );

    let rows = Rc::new(VecModel::<ControlRow>::default());
    win.set_controls(ModelRc::from(rows.clone()));
    win.set_snapshot_dir(user_dir("Pictures").display().to_string().into());
    win.set_record_dir(user_dir("Videos").display().to_string().into());
    win.set_audio_sources(ModelRc::new(VecModel::from(vec![SharedString::from(
        "No audio",
    )])));
    win.set_fx_tlx("0".into());
    win.set_fx_tly("0".into());
    win.set_fx_trx("1".into());
    win.set_fx_try("0".into());
    win.set_fx_brx("1".into());
    win.set_fx_bry("1".into());
    win.set_fx_blx("0".into());
    win.set_fx_bly("1".into());
    win.set_ai_out_dir(user_dir("Pictures").display().to_string().into());
    if let Ok(text) = std::fs::read_to_string(script_store()) {
        win.set_script_text(text.into());
    }

    let prefs = preferences::load();
    let ai = initial_ai(&prefs);
    let config = prefs
        .as_ref()
        .ok()
        .and_then(|p| p.capture.clone())
        .unwrap_or_default();
    let mut effects = prefs
        .as_ref()
        .ok()
        .and_then(|p| p.effects.clone())
        .unwrap_or_else(initial_effects);
    if !ai.enabled && matches!(effects.background.as_str(), "blur" | "replace") {
        effects.background = "none".into();
        effects.background_image = None;
    }
    let app = Rc::new(RefCell::new(App {
        win: win.as_weak(),
        events,
        event_tx,
        backend_tx: Some(backend_tx),
        media_tx: Some(media_tx),
        audio_tx: Some(audio_tx),
        vcam_tx: Some(vcam_tx),
        workers,
        devices: Vec::new(),
        device: None,
        wanted: device,
        config,
        fallback: None,
        formats: Vec::new(),
        formats_device: None,
        res_list: Vec::new(),
        fps_list: Vec::new(),
        mode_warning: None,
        epoch: 0,
        auto: true,
        connecting: false,
        preview: None,
        stage_error: None,
        media: MediaStatus::default(),
        progress: (0, Instant::now()),
        media_error_shown: None,
        last_seq: 0,
        has_frame: false,
        recording_path: None,
        snapshot: None,
        controls: Vec::new(),
        row_revs: Vec::new(),
        revs: HashMap::new(),
        rows,
        profiles: Vec::new(),
        presets: Vec::new(),
        zones: Vec::new(),
        tracking: None,
        effects,
        sources: Vec::new(),
        camera_mic: None,
        audio_error: None,
        audio_pending: 0,
        mic_pending: 0,
        audio_checked: Instant::now(),
        vcams: Vec::new(),
        vcam_missing: Vec::new(),
        vcam_pending: 0,
        vcam_stopping: false,
        vcam_stop: None,
        vcam_note: (String::new(), false),
        last_vout: None,
        last_page: 0,
        bpending: 0,
        mpending: 0,
        status: (String::new(), false),
        last_status_poll: Instant::now(),
        last_refresh: Instant::now(),
        script_dirty: None,
        ai,
    }));

    macro_rules! wire {
        ($on:ident => $method:ident($($arg:ident),*)) => {{
            let app = app.clone();
            win.$on(move |$($arg),*| app.borrow_mut().$method($($arg),*));
        }};
    }
    wire!(on_refresh_devices => refresh_devices());
    wire!(on_select_device => select_device(i));
    wire!(on_reconnect => reconnect());
    wire!(on_toggle_preview => toggle_preview());
    wire!(on_select_resolution => select_resolution(i));
    wire!(on_select_fps => select_fps(i));
    wire!(on_dismiss_status => dismiss_status());
    wire!(on_refresh_controls => refresh_controls());
    wire!(on_set_number => set_number(i, t));
    wire!(on_set_float => set_float(i, v));
    wire!(on_set_toggle => set_toggle(i, b));
    wire!(on_set_choice => set_choice(i, o));
    wire!(on_profile_save => profile_save());
    wire!(on_profile_load => profile_load());
    wire!(on_profile_delete => profile_delete());
    wire!(on_preset_recall => preset_recall());
    wire!(on_preset_delete => preset_delete());
    wire!(on_preset_save => preset_save());
    wire!(on_recenter => recenter());
    wire!(on_area_add => area_add());
    wire!(on_area_remove => area_remove(i));
    wire!(on_area_edit => area_edit(i, f, t));
    wire!(on_area_active => area_active(i, b));
    wire!(on_areas_apply => areas_apply());
    wire!(on_areas_reload => areas_reload());
    wire!(on_tracking_apply => tracking_apply());
    wire!(on_tracking_reload => tracking_reload());
    wire!(on_apply_effects => apply_effects());
    wire!(on_apply_board => apply_board());
    wire!(on_reset_board => reset_board());
    wire!(on_browse => browse(i));
    wire!(on_snapshot => snapshot());
    wire!(on_record => record());
    wire!(on_pause_record => pause_record());
    wire!(on_stop_record => stop_record());
    wire!(on_vcam_refresh => vcam_refresh());
    wire!(on_vcam_create => vcam_create(n));
    wire!(on_vcam_remove => vcam_remove());
    wire!(on_vcam_toggle => vcam_toggle());
    wire!(on_vcam_stop_all => vcam_stop_all());
    wire!(on_refresh_audio => refresh_audio());
    wire!(on_audio_selected => audio_selected(i));
    wire!(on_use_camera_mic => use_camera_mic());
    wire!(on_set_mic_muted => set_mic_muted(b));
    wire!(on_set_mic_volume => set_mic_volume(v));
    wire!(on_script_edited => script_edited());
    wire!(on_script_load => script_load());
    wire!(on_script_save => script_save());
    wire!(on_ai_refresh_library => ai_refresh_library());
    wire!(on_ai_download => ai_download());
    wire!(on_ai_cancel_download => ai_cancel_download());
    wire!(on_ai_delete => ai_delete());
    wire!(on_ai_select_model => ai_select_model(i));
    wire!(on_ai_defaults => ai_defaults());
    wire!(on_ai_run => ai_run());
    wire!(on_ai_stop => ai_stop());
    wire!(on_ai_live_toggled => ai_live_toggled(b));
    wire!(on_ai_freeze => ai_freeze());
    wire!(on_ai_unfreeze => ai_unfreeze());
    wire!(on_ai_clear_prompts => ai_clear_prompts());
    wire!(on_ai_prompts_edited => ai_prompts_edited());
    wire!(on_ai_pick => ai_pick(x0, y0, x1, y1, w, h));
    wire!(on_ai_set_stage => ai_set_stage(i));
    wire!(on_ai_source_toggled => ai_source_toggled(b));
    wire!(on_ai_save_raw => ai_save_raw());
    wire!(on_ai_save_visual => ai_save_visual());
    wire!(on_ai_select_history => ai_select_history(token));
    wire!(on_ai_history_follow_toggled => ai_history_follow_toggled(on));
    wire!(on_ai_history_clear => ai_history_clear());
    wire!(on_ai_history_raw_toggled => ai_history_raw_toggled(on));
    wire!(on_ai_enabled_toggled => ai_enabled_toggled(b));

    {
        let app = app.clone();
        win.window()
            .on_close_requested(move || app.borrow_mut().close_requested());
    }

    let timer = slint::Timer::default();
    {
        let app = app.clone();
        timer.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(16),
            move || app.borrow_mut().tick(),
        );
    }

    {
        let mut a = app.borrow_mut();
        win.set_ai_history(ModelRc::from(a.ai.history_rows.clone()));
        a.push_effects();
        a.push_views();
        a.set_controls_empty("Start the preview to read and change camera controls.");
        a.refresh_devices();
        a.refresh_audio();
        a.ai_load_defaults();
        a.ai_start_engine();
        a.send_vcam(VcamCmd::Refresh);
        a.ai_push_models();
        if a.ai.settings_note.1 {
            let message = a.ai.settings_note.0.clone();
            a.fail(message);
        }
        a.sync();
    }

    let result = win.run();
    app.borrow_mut().shutdown();
    drop(timer);
    result.map_err(|e| anyhow!("window event loop failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ai_test_frame(sequence: u64) -> Arc<Frame> {
        Arc::new(Frame {
            width: 1,
            height: 1,
            rgb: vec![20, 30, 40],
            sequence,
        })
    }

    #[test]
    fn analysis_session_restores_original_stage_across_repeated_runs() {
        let mut ai = Ai::default();
        ai.begin_session();
        ai.stage = 1;
        ai.begin_session();
        ai.live = true;
        ai.follow = true;
        ai.inflight = Some(Inflight {
            token: ai.token,
            model: MODEL_IDS[0],
            seq: 12,
        });
        let old_token = ai.token;
        let old_view = ai.history_view_token;
        ai.stop_session();
        assert_eq!(ai.stage, 0);
        assert_eq!(ai.prior_stage, None);
        assert!(!ai.live && !ai.follow && ai.inflight.is_none());
        assert_ne!(ai.token, old_token);
        assert_ne!(ai.history_view_token, old_view);
        ai.stop_session();
        assert_eq!(ai.stage, 0);
    }

    #[test]
    fn stop_restores_valid_frozen_stage_but_not_an_invalidated_source() {
        let mut ai = Ai::default();
        ai.frozen = Some(ai_test_frame(9));
        ai.stage = 2;
        ai.begin_session();
        ai.stage = 1;
        ai.stop_session();
        assert_eq!(ai.stage, 2);
        ai.begin_session();
        ai.stage = 1;
        ai.frozen = None;
        ai.stop_session();
        assert_eq!(ai.stage, 0);
    }

    #[test]
    fn explicit_stage_selection_supersedes_automatic_session_restore() {
        let mut ai = Ai::default();
        ai.frozen = Some(ai_test_frame(4));
        ai.begin_session();
        ai.stage = 1;
        ai.follow = true;
        let old_view = ai.history_view_token;
        ai.select_stage(2);
        assert!(!ai.follow);
        assert_ne!(ai.history_view_token, old_view);
        ai.stop_session();
        assert_eq!(ai.stage, 2);
        ai.begin_session();
        ai.select_stage(0);
        ai.stop_session();
        assert_eq!(ai.stage, 0);
    }

    #[test]
    fn own_inflight_and_cleanup_each_block_a_new_analysis() {
        let mut ai = Ai::default();
        assert!(!ai.working());
        ai.inflight = Some(Inflight {
            token: ai.token,
            model: MODEL_IDS[0],
            seq: 1,
        });
        assert!(ai.working());
        ai.inflight = None;
        ai.status.stopping = true;
        assert!(ai.working());
        ai.status.stopping = false;
        ai.status.busy = true;
        assert!(ai.working());
        ai.status.busy = false;
        assert!(!ai.working());
    }

    #[test]
    fn stop_invalidates_pending_restore_without_discarding_frame_matched_history() {
        let frame = ai_test_frame(7);
        let analysis = Arc::new(models::interpret(
            "florence-2-base",
            frame,
            models::RawOutput {
                job_id: "test-7".into(),
                content_type: "application/json".into(),
                bytes: Arc::new(br#"{"model":"florence-2-base","image":{"width":1,"height":1},"text":"a room","task":"caption"}"#.to_vec()),
                elapsed: Duration::from_millis(1),
                warning: None,
            },
        ).unwrap());
        let mut ai = Ai::default();
        let record = ai.history.push(Arc::new(
            models::history::Snapshot::capture(&analysis).unwrap(),
        ));
        ai.result = Some(analysis.clone());
        ai.latest = Some(analysis);
        ai.history_selected = Some(record.clone());
        ai.history_pending = Some((ai.history_view_token, record.clone()));
        ai.begin_session();
        ai.stage = 1;
        let old_view = ai.history_view_token;
        ai.stop_session();
        assert!(ai.history_pending.is_none());
        assert_ne!(ai.history_view_token, old_view);
        assert_eq!(ai.history.len(), 1);
        assert_eq!(ai.history_selected.as_ref().unwrap().id, record.id);
        assert_eq!(ai.result.as_ref().unwrap().source.sequence, 7);
        assert_eq!(
            ai.history
                .get(record.id)
                .unwrap()
                .snapshot
                .restore()
                .unwrap()
                .source
                .sequence,
            7
        );
    }

    #[test]
    fn blocked_control_worker_is_stopped_and_cannot_receive_more_writes() {
        use std::os::unix::process::ExitStatusExt;
        let mut child = Command::new("sleep")
            .arg("30")
            .process_group(0)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let input = BufWriter::new(child.stdin.take().unwrap());
        let (_keep_reader_alive, replies) = mpsc::channel();
        let mut process = CameraProcess {
            child: Some(child),
            input: Some(input),
            replies,
            quarantined: false,
        };
        let started = Instant::now();
        let error = process
            .request(&BackendCmd::Refresh, Duration::from_millis(20))
            .err()
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(error.to_string().contains("unknown outcome"));
        assert!(
            process
                .request(
                    &BackendCmd::Set {
                        key: "privacy".into(),
                        value: "true".into()
                    },
                    Duration::from_secs(1)
                )
                .is_err()
        );
        let status = process.child.as_mut().unwrap().wait().unwrap();
        process.child = None;
        assert_eq!(status.signal(), Some(9));
    }

    #[test]
    fn control_protocol_rejects_partial_and_oversized_messages() {
        assert!(read_control_message::<BackendCmd>(&mut &b"\"Refresh\""[..]).is_err());
        let oversized = vec![b' '; MAX_CONTROL_MESSAGE as usize + 1];
        assert!(read_control_message::<BackendCmd>(&mut oversized.as_slice()).is_err());
    }

    fn number(min: f64, max: f64, step: f64) -> Control {
        Control {
            key: "k".into(),
            label: "K".into(),
            group: "G".into(),
            kind: ControlKind::Number,
            value: String::new(),
            min,
            max,
            step,
            options: vec![],
            default: None,
        }
    }

    #[test]
    fn slider_values_snap_to_the_step_grid_and_range() {
        assert_eq!(slider_text(4.6, &number(0.0, 10.0, 1.0)), "5");
        assert_eq!(slider_text(0.34, &number(0.0, 1.0, 0.1)), "0.3");
        assert_eq!(slider_text(99.0, &number(2000.0, 6500.0, 100.0)), "2000");
        assert_eq!(slider_text(-0.04, &number(-1.0, 1.0, 0.1)), "0.0");
    }

    #[test]
    fn only_a_reported_in_range_default_can_reset_a_slider() {
        let mut c = number(0.0, 255.0, 1.0);
        assert_eq!(
            default_of(&c),
            None,
            "no reported default means no reset target"
        );
        c.default = Some(128.0);
        assert_eq!(default_of(&c), Some(128.0));
        c.default = Some(300.0);
        assert_eq!(
            default_of(&c),
            None,
            "a default outside the range is not trusted"
        );
        c.kind = ControlKind::Toggle;
        c.default = Some(1.0);
        assert_eq!(default_of(&c), None);
    }

    #[test]
    fn model_sizes_read_in_binary_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GB");
    }

    #[test]
    fn mode_choice_keeps_a_reported_mode_else_the_nearest_reported_one() {
        let m = |width, height, fps| VideoConfig { width, height, fps };
        let modes = [m(1920, 1080, 30), m(3840, 2160, 24), m(1280, 720, 24)];
        let kept = choose_mode(&modes, &m(1920, 1080, 30)).unwrap();
        assert_eq!((kept.width, kept.fps), (1920, 30));
        let nearest = choose_mode(&modes, &m(1280, 720, 30)).unwrap();
        assert_eq!((nearest.width, nearest.fps), (1280, 24));
        assert!(choose_mode(&[], &m(1280, 720, 30)).is_none());
    }

    #[test]
    fn normalized_inputs_reject_out_of_range_and_text() {
        assert!(parse_unit("0.5", "x").is_ok());
        assert!(parse_unit("1.5", "x").is_err());
        assert!(parse_unit("abc", "x").is_err());
    }

    #[test]
    fn pointer_positions_map_through_the_contain_letterbox() {
        // 1000x500 view, 400x400 image: drawn 500x500 at x = 250.
        let wide = |x, y, clamp| contain_to_pixel((1000.0, 500.0), (400, 400), (x, y), clamp);
        assert_eq!(wide(250.0, 0.0, false), Some((0.0, 0.0)));
        assert_eq!(wide(500.0, 250.0, false), Some((200.0, 200.0)));
        assert_eq!(wide(750.0, 500.0, false), Some((400.0, 400.0)));
        assert_eq!(wide(100.0, 250.0, false), None);
        assert_eq!(wide(100.0, 250.0, true), Some((0.0, 200.0)));
        assert_eq!(wide(900.0, 600.0, true), Some((400.0, 400.0)));
        // 500x1000 view, 800x400 image: drawn 500x250 at y = 375.
        let tall = |x, y| contain_to_pixel((500.0, 1000.0), (800, 400), (x, y), false);
        assert_eq!(tall(250.0, 500.0), Some((400.0, 200.0)));
        assert_eq!(tall(250.0, 300.0), None);
        assert_eq!(
            contain_to_pixel((0.0, 500.0), (4, 4), (1.0, 1.0), true),
            None
        );
        assert_eq!(
            contain_to_pixel((500.0, 500.0), (4, 4), (f32::NAN, 1.0), true),
            None
        );
    }

    #[test]
    fn sam_prompts_parse_and_normalise() {
        assert_eq!(
            parse_points("10,20; 30.5 , 40\n7,8;").unwrap(),
            vec![(10.0, 20.0), (30.5, 40.0), (7.0, 8.0)]
        );
        assert!(parse_points("10").is_err());
        assert!(parse_points("a,b").is_err());
        assert!(parse_points("-1,2").is_err());
        assert!(parse_points(&"1,1;".repeat(17)).is_err());
        assert_eq!(
            lenient_points("1,2; bad; 3,4"),
            vec![(1.0, 2.0), (3.0, 4.0)]
        );
        assert_eq!(parse_box("  ").unwrap(), None);
        assert_eq!(
            parse_box("50,60,10,20").unwrap(),
            Some([10.0, 20.0, 50.0, 60.0])
        );
        assert!(parse_box("1,2,3").is_err());
        assert!(parse_box("1,1,1,9").is_err());
        assert_eq!(fmt_coord(12.0), "12");
        assert_eq!(fmt_coord(12.5), "12.5");
    }

    #[test]
    fn prompt_markers_land_on_the_chosen_pixels() {
        let (w, h) = (200u32, 100u32);
        let mut rgb = vec![10u8; w as usize * h as usize * 3];
        draw_prompts(
            &mut rgb,
            w,
            h,
            &[(100.0, 50.0)],
            Some([20.0, 10.0, 60.0, 40.0]),
        );
        let at = |x: usize, y: usize| &rgb[(y * w as usize + x) * 3..][..3];
        assert_eq!(at(100, 50), [0, 255, 64]);
        assert_eq!(at(20, 25), [0, 255, 64]);
        assert_eq!(at(150, 90), [10, 10, 10]);
        // Marks near and past the edges stay inside the buffer.
        draw_prompts(
            &mut rgb,
            w,
            h,
            &[(0.0, 0.0), (199.0, 99.0)],
            Some([0.0, 0.0, 200.0, 100.0]),
        );
    }
}
