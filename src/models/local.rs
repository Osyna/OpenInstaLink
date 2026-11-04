//! On-machine model runtime. Camera frames never leave the process tree: one persistent Python worker
//! (assets/local_ai, embedded in this binary) keeps the current model resident and talks over framed
//! stdin/stdout. Everything lives under `$XDG_DATA_HOME/link-camera/local-ai`:
//!
//! ```text
//! locks/<name>.lock   flock files (cross-process; model locks: shared = running, exclusive = download/delete)
//! models/<id>/        published only after a complete download; `.installed.json` is written last
//! runtime/<fp>/       managed Python 3.11 venv, built in place (venvs are not relocatable); usable only
//! once `.ready.json` exists, which is written after its import probe passed (only `bin/python` is used)
//! code/<hash>/        the embedded worker + plugins, materialised once per build
//! tmp/ cache/ home/ python/   staging, HF/matplotlib caches, the worker's private HOME, uv's Python
//! ```
//!
//! Nothing is downloaded by `catalog` or `run`; only an explicit `download` touches the network (runtime
//! packages through uv/pip, weights through the plugin). Inference runs with offline environment flags
//! and Python socket connects blocked. Every child is its own process group, is killed as a group on
//! cancel/timeout/drop and is always reaped.

use std::fmt;
use std::fs::{self, DirBuilder, File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    MAX_OUTPUT_BYTES, MAX_PIXELS, MODELS, ModelInfo, RawOutput, RuntimeConfig, defaults,
    model_spec, validate_options,
};
use crate::media::Frame;
use crate::process::stop_group;

const CODE: [(&str, &str); 4] = [
    ("worker.py", include_str!("../../assets/local_ai/worker.py")),
    (
        "landmarks.py",
        include_str!("../../assets/local_ai/landmarks.py"),
    ),
    (
        "detection.py",
        include_str!("../../assets/local_ai/detection.py"),
    ),
    ("scene.py", include_str!("../../assets/local_ai/scene.py")),
];
const REQUIREMENTS: &str = include_str!("../../assets/local_ai/requirements.txt");

const MAX_HEADER: usize = 1 << 20;
const MAX_REQUEST_BODY: usize = MAX_PIXELS as usize * 3;
const TAIL: usize = 16 << 10;
const POLL: Duration = Duration::from_millis(25);
/// How long a killed process group may take to exit before its leader goes to a reaper thread.
const STOP_GRACE: Duration = Duration::from_secs(2);
const SETUP_TIMEOUT: Duration = Duration::from_secs(2 * 3600);
const INSTALL_TIMEOUT: Duration = Duration::from_secs(6 * 3600);
const PROBE_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_TREE_ENTRIES: usize = 200_000;
const MARKER: &str = ".installed.json";
const READY: &str = ".ready.json";
const OVERRIDE_ENV: &str = "LINK_CAMERA_AI_PYTHON";

/// The run (or download) was cancelled by the caller; the owned child was already killed and reaped.
#[derive(Debug)]
pub struct Cancelled {
    pub job_id: Option<String>,
}

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// What the user sees before downloading: summary, limits, licence, upstream source, download size.
/// After a download the plugin's own `metadata()` (recorded in the marker) replaces source/licence.
struct Facts {
    summary: &'static str,
    best_for: &'static str,
    outputs: &'static str,
    performance: &'static str,
    limitations: &'static str,
    license: &'static str,
    source: &'static str,
    size: &'static str,
}

const MEDIAPIPE_LICENSE: &str =
    "Apache-2.0 (Google MediaPipe; model bundle per its MediaPipe model card)";
const MEDIAPIPE_SOURCE: &str = "https://storage.googleapis.com/mediapipe-models";

fn facts(model: &str) -> Facts {
    match model {
        "mediapipe-hand" => Facts {
            summary: "Locates finger joints and estimates whether each hand is left or right.",
            best_for: "Inspecting hand placement and finger positions in a camera frame.",
            outputs: "21 landmarks per hand, left/right handedness and its confidence score.",
            performance: "Runs on the CPU; no NVIDIA GPU required.",
            limitations: "Small, occluded or motion-blurred hands can be missed. Landmarks do not classify gestures.",
            license: MEDIAPIPE_LICENSE,
            source: MEDIAPIPE_SOURCE,
            size: "about 8 MB",
        },
        "mediapipe-pose" => Facts {
            summary: "Estimates body posture from 33 landmarks in each image.",
            best_for: "Inspecting body alignment when the person is clearly visible in the camera.",
            outputs: "33 image landmarks per pose and estimated 3D world landmarks.",
            performance: "Uses the lite pose model on the CPU; no NVIDIA GPU required.",
            limitations: "Partial bodies and occlusion reduce reliability. World coordinates are estimates, not calibrated measurements. No persistent video tracking.",
            license: MEDIAPIPE_LICENSE,
            source: MEDIAPIPE_SOURCE,
            size: "about 6 MB",
        },
        "mediapipe-face" => Facts {
            summary: "Maps facial geometry, with optional blendshape scores and a head transform.",
            best_for: "Inspecting facial landmark placement, facial movement and head orientation.",
            outputs: "478 landmarks per face; optional top blendshape scores and head transformation matrix.",
            performance: "Runs on the CPU; no NVIDIA GPU required.",
            limitations: "Frontal, unobstructed faces work best. Does not identify people; blendshape scores do not establish emotions.",
            license: MEDIAPIPE_LICENSE,
            source: MEDIAPIPE_SOURCE,
            size: "about 4 MB",
        },
        "depth-anything-v2-small" => Facts {
            summary: "Estimates which parts of a frame are nearer or farther away.",
            best_for: "Frequent relative-depth snapshots of a scene without diffusion inference.",
            outputs: "A relative-depth PNG in grayscale or inferno colors, with nearer areas brighter.",
            performance: "The lighter, single-pass depth option; uses NVIDIA CUDA when available, otherwise the CPU.",
            limitations: "Depth is relative within each image, not distance in metres. Brightness is normalized per frame, so it can change between frames.",
            license: "Apache-2.0",
            source: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
            size: "about 100 MB",
        },
        "marigold-depth-v1-1" => Facts {
            summary: "Uses diffusion to estimate a scene's relative depth from one image.",
            best_for: "Comparing depth structure in individual frames when slower inference is acceptable.",
            outputs: "A relative-depth PNG in grayscale or inferno colors, with nearer areas brighter.",
            performance: "Requires an NVIDIA GPU with CUDA. Slower than Depth Anything; more steps or ensemble samples add work.",
            limitations: "No CPU fallback. Depth is relative, not metres, and estimates can vary between runs. Per-frame normalization prevents direct brightness comparisons.",
            license: "CreativeML OpenRAIL++-M; use restrictions apply",
            source: "https://huggingface.co/prs-eth/marigold-depth-v1-1",
            size: "about 3 GB",
        },
        "sam2.1-tiny" => Facts {
            summary: "Segments a selected object from point or box prompts.",
            best_for: "Outlining a particular object using point coordinates or a bounding box.",
            outputs: "Largest-component outlines, bounding boxes, mask scores and area fractions.",
            performance: "Uses NVIDIA CUDA when available, otherwise the CPU.",
            limitations: "Defaults to the image centre without a prompt. Outlines omit holes and disconnected regions; no full alpha mask or video tracking.",
            license: "Apache-2.0",
            source: "https://huggingface.co/facebook/sam2.1-hiera-tiny",
            size: "about 160 MB",
        },
        "florence-2-base" => Facts {
            summary: "Generates a caption, object detections, text readings or region descriptions.",
            best_for: "Describing a scene or reading visible text; choose the task to match the question.",
            outputs: "Caption text, labeled boxes, OCR text with regions, or region descriptions, depending on the task.",
            performance: "Uses NVIDIA CUDA when available; generation can be slow on the CPU.",
            limitations: "Generated descriptions, labels and text can be wrong or invented. Small or blurred text is harder to read; verify important results.",
            license: "MIT",
            source: "https://huggingface.co/florence-community/Florence-2-base",
            size: "about 1 GB",
        },
        _ => Facts {
            summary: "Finds objects from the 80 COCO classes and marks their locations.",
            best_for: "Locating familiar objects such as people, vehicles, pets and everyday items.",
            outputs: "Object class labels, confidence scores and bounding boxes.",
            performance: "Uses the small YOLO11n model with CUDA when supported by the runtime, otherwise the CPU.",
            limitations: "Limited to the COCO classes; cannot recognize arbitrary objects. Small, obscured or unusual views can be missed or mislabeled.",
            license: "AGPL-3.0 (Ultralytics; commercial terms apply for proprietary use)",
            source: "https://github.com/ultralytics/assets/releases/tag/v8.4.0",
            size: "about 11 MB",
        },
    }
}

#[derive(Default)]
struct Status {
    /// Resident model and the device the worker reported for it.
    loaded: Option<(String, String)>,
    loading: Option<String>,
}

/// Reusable local runtime handle. Cheap to keep; owns at most one worker process.
pub struct Client {
    timeout: Duration,
    root: PathBuf,
    python_override: Option<PathBuf>,
    worker: Mutex<Option<Worker>>,
    cleanup: Mutex<Option<Receiver<()>>>,
    status: Mutex<Status>,
    runs: AtomicU64,
}

impl Client {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| Path::new(&h).join(".local/share"))
                    .filter(|p| p.is_absolute())
            })
            .context("neither XDG_DATA_HOME nor HOME is an absolute path")?;
        let python = match std::env::var_os(OVERRIDE_ENV).filter(|v| !v.is_empty()) {
            Some(path) => {
                let path = PathBuf::from(path);
                ensure!(
                    path.is_absolute(),
                    "{OVERRIDE_ENV} must be an absolute path to a prepared Python 3.11 interpreter"
                );
                Some(path)
            }
            None => None,
        };
        Self::at(data.join("link-camera/local-ai"), config, python)
    }

    fn at(root: PathBuf, config: RuntimeConfig, python_override: Option<PathBuf>) -> Result<Self> {
        ensure!(
            (Duration::from_secs(1)..=Duration::from_secs(3600)).contains(&config.timeout),
            "model timeout must be between 1 and 3600 seconds"
        );
        Ok(Self {
            timeout: config.timeout,
            root,
            python_override,
            worker: Mutex::new(None),
            cleanup: Mutex::new(None),
            status: Mutex::new(Status::default()),
            runs: AtomicU64::new(0),
        })
    }

    /// All eight models with install/runtime/device state. Starts no process and uses no network.
    pub fn catalog(&self) -> Result<Vec<ModelInfo>> {
        let owns_worker = match self.worker.try_lock() {
            Ok(mut worker) => {
                if worker.as_mut().is_some_and(Worker::exited) {
                    *worker = None;
                    self.status.lock().unwrap().loaded = None;
                }
                worker.is_some()
            }
            Err(std::sync::TryLockError::WouldBlock) => true,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner().is_some(),
        };
        let externally_owned = !owns_worker && self.resident_busy()?;
        let runtime = self.python().is_some();
        let status = self.status.lock().unwrap();
        let mut out = Vec::with_capacity(MODELS.len());
        for spec in &MODELS {
            let facts = facts(spec.id);
            let marker = self.installed(spec.id);
            let installed = marker.is_some();
            let (available, reason) = if !installed {
                (
                    false,
                    Some(format!(
                        "not downloaded ({}); downloading it also authorises the {} download",
                        facts.size,
                        if runtime {
                            "model weights"
                        } else {
                            "Python runtime (several GB) and model weights"
                        }
                    )),
                )
            } else if !runtime {
                (
                    false,
                    Some(
                        "the local AI runtime is missing; download a model again to set it up"
                            .into(),
                    ),
                )
            } else if externally_owned {
                (
                    false,
                    Some("another local AI worker owns this model library; Stop or close its window (or stop its CLI command), then Refresh here".into()),
                )
            } else {
                (true, None)
            };
            let meta = marker.as_ref().map(|m| &m["metadata"]);
            let pick = |key: &str, fallback: &str| {
                meta.and_then(|m| m[key].as_str())
                    .unwrap_or(fallback)
                    .to_string()
            };
            let resident = status.loaded.as_ref().filter(|(m, _)| m == spec.id);
            out.push(ModelInfo {
                id: spec.id.into(),
                input: "image".into(),
                output: spec.output.into(),
                loaded: resident.is_some(),
                loading: status.loading.as_deref() == Some(spec.id),
                available: Some(available),
                unavailable_reason: reason,
                presentation: json!({
                    "summary": facts.summary,
                    "best_for": facts.best_for,
                    "outputs": facts.outputs,
                    "performance": facts.performance,
                    "limitations": facts.limitations,
                    "license": pick("license", facts.license),
                    "source": pick("source", facts.source),
                    "revision": meta.and_then(|m| m["revision"].as_str()),
                    "download_size": facts.size,
                }),
                option_defaults: defaults(spec.id)?,
                installed,
                disk_bytes: if installed {
                    tree_bytes(&self.model_dir(spec.id), false).unwrap_or(0)
                } else {
                    0
                },
                device: resident.map(|(_, d)| d.clone()).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// Runs one frame through the resident worker, loading `model` first if it is not the loaded one.
    /// Cancellation and the total deadline kill the worker's process group immediately.
    pub fn run(
        &self,
        model: &str,
        options: &Value,
        frame: &Frame,
        cancel: &AtomicBool,
    ) -> Result<RawOutput> {
        let spec = model_spec(model)?;
        validate_options(model, options, frame.width, frame.height)?;
        ensure!(
            frame.rgb.len() as u64 == u64::from(frame.width) * u64::from(frame.height) * 3,
            "the frame buffer does not match its dimensions"
        );
        let started = Instant::now();
        let deadline = started + self.timeout;
        let cancelled = || Cancelled { job_id: None };
        if cancel.load(Ordering::SeqCst) {
            return Err(cancelled().into());
        }
        ensure!(
            self.installed(model).is_some(),
            "{} is not downloaded; download it first (this is the only step that uses the network)",
            spec.label
        );
        let python = self
            .python()
            .context("the local AI runtime is not set up; downloading a model sets it up")?;
        self.prepare()?;
        let mut worker = loop {
            match self.worker.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(p)) => break p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    if cancel.load(Ordering::SeqCst) {
                        return Err(cancelled().into());
                    }
                    ensure!(
                        Instant::now() < deadline,
                        "timed out waiting for the previous model run"
                    );
                    thread::sleep(POLL);
                }
            }
        };
        // Shared model lock: a download or delete (here or in another process) cannot replace the
        // files while this run holds it.
        let lock = self.open_lock(model)?;
        ensure!(
            try_lock(&lock, false)?,
            "{} is being downloaded or deleted; try again when that finishes",
            spec.label
        );
        ensure!(
            self.installed(model).is_some(),
            "{} is no longer installed",
            spec.label
        );
        let code = self.code_dir()?;
        if cancel.load(Ordering::SeqCst) {
            return Err(cancelled().into());
        }
        if worker.as_mut().is_some_and(Worker::exited) {
            *worker = None;
            self.status.lock().unwrap().loaded = None;
        }
        if worker.is_none() {
            *worker = Some(self.spawn_worker(&python, &code)?);
        }
        let w = worker.as_mut().expect("worker was just created");
        let loading = w.model.as_deref() != Some(model);
        if loading {
            self.status.lock().unwrap().loading = Some(model.into());
        }
        let header = json!({
            "op": "infer",
            "model": model,
            "dir": self.model_dir(model),
            "options": options,
            "width": frame.width,
            "height": frame.height,
        });
        let result = w.infer(header, &frame.rgb, cancel, deadline);
        let mut status = self.status.lock().unwrap();
        status.loading = None;
        let (reply, bytes) = match result {
            Ok(done) => done,
            Err(error) => {
                *worker = None; // drop reaps the killed group; the resident model is gone with it
                status.loaded = None;
                return Err(error);
            }
        };
        w.model = reply["loaded"].as_str().map(str::to_string);
        status.loaded = w
            .model
            .clone()
            .map(|m| (m, reply["device"].as_str().unwrap_or("unknown").to_string()));
        drop(status);
        if reply["ok"].as_bool() != Some(true) {
            bail!(
                "{} failed: {}",
                spec.label,
                clean(reply["error"].as_str().unwrap_or("no reason given"))
            );
        }
        let content_type = reply["content_type"].as_str().unwrap_or("");
        ensure!(
            matches!(content_type, "application/json" | "image/png"),
            "unexpected result type `{}`",
            clean(content_type)
        );
        if cancel.load(Ordering::SeqCst) {
            *worker = None;
            self.status.lock().unwrap().loaded = None;
            return Err(cancelled().into());
        }
        Ok(RawOutput {
            job_id: format!("local-{}", self.runs.fetch_add(1, Ordering::Relaxed) + 1),
            content_type: content_type.into(),
            bytes: Arc::new(bytes),
            elapsed: started.elapsed(),
            warning: None,
        })
    }

    /// Called off the UI thread; the lease remains held until even a delayed kernel exit completes.
    pub(super) fn stop(&self) {
        *self.worker.lock().unwrap() = None;
        if let Some(done) = self.cleanup.lock().unwrap().take() {
            let _ = done.recv();
        }
        let mut status = self.status.lock().unwrap();
        status.loaded = None;
        status.loading = None;
    }

    /// Explicit download: sets up the managed runtime if missing, then fetches the model's weights into
    /// a staging directory and publishes it atomically. Blocking (call it off the GUI thread); it polls
    /// `cancel`, which kills the owned child group at once and leaves nothing published.
    pub fn download(&self, model: &str, cancel: &AtomicBool) -> Result<()> {
        model_spec(model)?;
        self.prepare()?;
        let lock = self.open_lock(model)?;
        ensure!(
            try_lock(&lock, true)?,
            "{model} is in use, being downloaded or being deleted elsewhere"
        );
        if let Ok(mut worker) = self.worker.try_lock() {
            if worker
                .as_ref()
                .is_some_and(|w| w.model.as_deref() == Some(model))
            {
                *worker = None;
                self.status.lock().unwrap().loaded = None;
            }
        }
        let code = self.code_dir()?;
        let python = self.ensure_runtime(&code, cancel)?;
        if self.installed(model).is_some() {
            return Ok(());
        }
        let stage = self.root.join("tmp").join(format!("model-{model}"));
        remove_owned(&stage)?;
        make_dir(&stage)?;
        let result = (|| {
            let mut cmd = self.command(&python, &code, true);
            cmd.args(["install", model]).arg(&stage);
            let done = run_child(cmd, cancel, INSTALL_TIMEOUT)?;
            ensure!(
                done.status.success(),
                "downloading {model} failed ({}): {}. Nothing was installed.",
                done.status,
                tail_text(&done.stderr)
            );
            let marker =
                marker(&stage).context("the download finished without its completion marker")?;
            ensure!(
                marker["model"] == model,
                "the download marker names another model"
            );
            tree_bytes(&stage, true)?;
            let target = self.model_dir(model);
            remove_owned(&target)?;
            fs::rename(&stage, &target).with_context(|| format!("publishing {}", target.display()))
        })();
        let _ = remove_owned(&stage);
        result
    }

    /// Removes exactly `models/<id>` (links inside are unlinked, never followed). External model caches
    /// are never touched. Refuses while a run is active here or in another process.
    pub fn delete(&self, model: &str) -> Result<()> {
        let spec = model_spec(model)?;
        self.prepare()?;
        let mut worker = match self.worker.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                bail!(
                    "a model run is active; cancel it before deleting {}",
                    spec.label
                )
            }
        };
        let lock = self.open_lock(model)?;
        ensure!(
            try_lock(&lock, true)?,
            "{} is in use or being downloaded; it was not deleted",
            spec.label
        );
        let dir = self.model_dir(model);
        match fs::symlink_metadata(&dir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => bail!(
                "refusing to delete {}: it is not a directory owned by link-camera",
                dir.display()
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("{} is not installed", spec.label)
            }
            Err(e) => return Err(e).with_context(|| format!("inspecting {}", dir.display())),
        }
        if worker
            .as_ref()
            .is_some_and(|w| w.model.as_deref() == Some(model))
        {
            *worker = None;
            self.status.lock().unwrap().loaded = None;
        }
        // The worker's private copies must go too: remove the directory first, then staging leftovers.
        fs::remove_dir_all(&dir).with_context(|| format!("deleting {}", dir.display()))?;
        let _ = remove_owned(&self.root.join("tmp").join(format!("model-{model}")));
        Ok(())
    }

    // ---- layout -----------------------------------------------------------------------------

    fn model_dir(&self, model: &str) -> PathBuf {
        self.root.join("models").join(model)
    }

    fn runtime_dir(&self) -> PathBuf {
        self.root.join("runtime").join(runtime_id())
    }

    /// Creates the app-owned directories (0700). Every component must be a real directory.
    fn prepare(&self) -> Result<()> {
        DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        owned_dir(&self.root)?;
        for dir in [
            "models", "locks", "tmp", "code", "cache", "home", "python", "runtime",
        ] {
            owned_dir(&self.root.join(dir))?;
        }
        Ok(())
    }

    fn open_lock(&self, name: &str) -> Result<LockFile> {
        let path = self.root.join("locks").join(format!("{name}.lock"));
        if let Ok(meta) = fs::symlink_metadata(&path) {
            ensure!(
                meta.is_file(),
                "{} is not a regular lock file",
                path.display()
            );
        }
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)
            .map(LockFile)
            .with_context(|| format!("opening {}", path.display()))
    }

    fn resident_busy(&self) -> Result<bool> {
        let path = self.root.join("locks/resident-worker.lock");
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
        {
            Ok(file) => LockFile(file),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error).with_context(|| format!("opening {}", path.display())),
        };
        ensure!(file.metadata()?.is_file(), "invalid resident worker lock");
        Ok(!try_lock(&file, true)?)
    }

    /// The marker of a completely published model (real directory, regular marker file).
    fn installed(&self, model: &str) -> Option<Value> {
        let dir = self.model_dir(model);
        fs::symlink_metadata(&dir).ok()?.is_dir().then_some(())?;
        marker(&dir)
    }

    /// The interpreter to use, if the runtime is ready. A managed runtime counts only once its
    /// import probe passed (its marker is written last and published by rename).
    fn python(&self) -> Option<PathBuf> {
        if let Some(path) = &self.python_override {
            return path.is_file().then(|| path.clone());
        }
        let dir = self.runtime_dir();
        let python = dir.join("venv/bin/python");
        let ready = fs::symlink_metadata(dir.join(READY)).ok()?.is_file();
        (ready && python.exists()).then_some(python)
    }

    /// Writes the embedded worker and plugins once per content hash, atomically.
    fn code_dir(&self) -> Result<PathBuf> {
        self.prepare()?;
        let mut hash = Sha256::new();
        for (name, text) in CODE {
            hash.update(name);
            hash.update(text);
        }
        let target = self
            .root
            .join("code")
            .join(hex(&hash.finalize())[..16].to_string());
        if fs::symlink_metadata(&target).is_ok_and(|m| m.is_dir()) {
            return Ok(target);
        }
        let stage =
            self.root
                .join("code")
                .join(format!(".stage-{}-{}", std::process::id(), nanos()));
        make_dir(&stage)?;
        let written = (|| {
            for (name, text) in CODE {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(stage.join(name))?;
                file.write_all(text.as_bytes())?;
            }
            fs::rename(&stage, &target)
        })();
        let _ = remove_owned(&stage);
        match written {
            Ok(()) => Ok(target),
            // A concurrent process published the same content first.
            Err(_) if fs::symlink_metadata(&target).is_ok_and(|m| m.is_dir()) => Ok(target),
            Err(e) => {
                Err(e).with_context(|| format!("writing worker code to {}", target.display()))
            }
        }
    }

    /// The only constructor of Python commands: fixed argv, cleared environment, private HOME/caches.
    fn command(&self, python: &Path, code: &Path, online: bool) -> Command {
        let cache = self.root.join("cache");
        let mut cmd = Command::new(python);
        cmd.args(["-I", "-B"])
            .arg(code.join("worker.py"))
            .env_clear()
            .current_dir(self.root.join("home"))
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .env("HOME", self.root.join("home"))
            .env("TMPDIR", self.root.join("tmp"))
            .env("XDG_CACHE_HOME", &cache)
            .env("HF_HOME", cache.join("huggingface"))
            .env("MPLCONFIGDIR", cache.join("matplotlib"))
            .env("LANG", "C.UTF-8")
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .env("DO_NOT_TRACK", "1")
            .env("YOLO_AUTOINSTALL", "False");
        let mut pass = vec![
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
        ];
        if online {
            pass.extend([
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "https_proxy",
                "http_proxy",
                "all_proxy",
                "no_proxy",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
            ]);
        } else {
            for key in [
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "YOLO_OFFLINE",
            ] {
                cmd.env(key, "1");
            }
        }
        for key in pass {
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }
        cmd
    }

    fn spawn_worker(&self, python: &Path, code: &Path) -> Result<Worker> {
        let lock = self.open_lock("resident-worker")?;
        ensure!(
            try_lock(&lock, true)?,
            "another local AI worker owns this model library; Stop or close its window (or stop its CLI command), then Refresh here"
        );
        let mut cmd = self.command(python, code, false);
        cmd.arg("serve");
        let fd = lock.as_raw_fd();
        // Keep the lease in Python too, so an abruptly exited parent cannot orphan an unowned worker.
        // SAFETY: the lock keeps fd valid through spawn; fcntl is async-signal-safe.
        unsafe {
            cmd.pre_exec(move || {
                if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = Owned::spawn(cmd, Stdio::piped())
            .with_context(|| format!("starting the AI worker with {}", python.display()))?;
        let (released, cleanup) = mpsc::channel();
        child.lease = Some(ResidentLease {
            lock: Some(lock),
            released,
        });
        let (stdin, stdout, stderr) = child.pipes()?;
        let stdin = stdin.context("worker has no stdin")?;
        set_nonblocking(&stdin)?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || read_replies(stdout, tx));
        let (stderr, _) = collect(stderr, TAIL);
        *self.cleanup.lock().unwrap() = Some(cleanup);
        Ok(Worker {
            child,
            stdin,
            rx,
            stderr,
            model: None,
            next_id: 0,
        })
    }

    // ---- runtime setup (download only) ----------------------------------------------------------

    fn ensure_runtime(&self, code: &Path, cancel: &AtomicBool) -> Result<PathBuf> {
        if let Some(python) = &self.python_override {
            ensure!(
                python.is_file(),
                "{OVERRIDE_ENV} points at {}, which is not a file",
                python.display()
            );
            self.probe(python, code, cancel)?;
            return Ok(python.clone());
        }
        if let Some(python) = self.python() {
            return Ok(python);
        }
        let lock = self.open_lock("runtime")?;
        lock_wait(&lock, cancel)?;
        if let Some(python) = self.python() {
            return Ok(python);
        }
        // Built where it will live: venv scripts and pyvenv.cfg record absolute paths, so a staged venv
        // must not be renamed. Atomicity comes from the marker, which is written last and is the only
        // thing `python()` accepts; a half-built directory is invisible and replaced here.
        let target = self.runtime_dir();
        remove_owned(&target)?;
        make_dir(&target)?;
        let built = self.build_runtime(&target, code, cancel);
        if built.is_err() {
            let _ = remove_owned(&target);
        }
        built?;
        self.python()
            .context("the runtime was built but is not usable")
    }

    fn build_runtime(&self, stage: &Path, code: &Path, cancel: &AtomicBool) -> Result<()> {
        let requirements = stage.join("requirements.txt");
        fs::write(&requirements, REQUIREMENTS)?;
        let venv = stage.join("venv");
        let python = venv.join("bin/python");
        let step = |mut cmd: Command, what: &str| -> Result<()> {
            cmd.current_dir(stage)
                .env("PIP_DISABLE_PIP_VERSION_CHECK", "1");
            for key in ["VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"] {
                cmd.env_remove(key);
            }
            let done = run_child(cmd, cancel, SETUP_TIMEOUT)?;
            ensure!(
                done.status.success(),
                "{what} failed ({}): {}. Check the internet connection and that the disk has several free GB; \
                 nothing was installed.",
                done.status,
                tail_text(&done.stderr)
            );
            Ok(())
        };
        if let Some(uv) = find_in_path("uv") {
            let uv_cmd = |args: &[&str]| {
                let mut cmd = Command::new(&uv);
                cmd.args(args)
                    // The app chooses its own Python; inherited uv selectors can conflict.
                    .env_remove("UV_PYTHON_PREFERENCE")
                    .env_remove("UV_MANAGED_PYTHON")
                    .env_remove("UV_NO_MANAGED_PYTHON")
                    .env("UV_PYTHON_INSTALL_DIR", self.root.join("python"))
                    .env("UV_NO_PROGRESS", "1");
                cmd
            };
            let mut venv_cmd = uv_cmd(&[
                "venv",
                "--no-config",
                "--managed-python",
                "--python",
                "3.11",
            ]);
            venv_cmd.arg(&venv);
            step(venv_cmd, "creating the Python 3.11 environment with uv")?;
            let mut install = uv_cmd(&["pip", "install", "--no-config", "--python"]);
            install.arg(&python).arg("-r").arg(&requirements);
            step(install, "installing the AI runtime packages with uv")?;
        } else if let Some(py311) = find_in_path("python3.11") {
            let mut make = Command::new(&py311);
            make.args(["-m", "venv"]).arg(&venv);
            step(make, "creating the Python 3.11 environment")?;
            let mut install = Command::new(&python);
            install
                .args(["-m", "pip", "install", "--no-input", "-r"])
                .arg(&requirements);
            step(install, "installing the AI runtime packages with pip")?;
        } else {
            bail!(
                "no Python 3.11 provider found: install uv (https://docs.astral.sh/uv/) so link-camera can \
                 set up its own Python 3.11, install python3.11, or set {OVERRIDE_ENV} to a prepared interpreter"
            );
        }
        let report = self.probe(&python, code, cancel)?;
        fs::write(
            stage.join(READY),
            serde_json::to_vec(&json!({ "runtime": runtime_id(), "probe": report }))?,
        )?;
        Ok(())
    }

    /// Imports the runtime's packages offline; an incomplete runtime is reported, never published.
    fn probe(&self, python: &Path, code: &Path, cancel: &AtomicBool) -> Result<Value> {
        let mut cmd = self.command(python, code, false);
        cmd.arg("probe");
        let done = run_child(cmd, cancel, PROBE_TIMEOUT)?;
        ensure!(
            done.status.success(),
            "the AI runtime check failed ({}): {}",
            done.status,
            tail_text(&done.stderr)
        );
        let report: Value = serde_json::from_slice(&done.stdout).with_context(|| {
            format!(
                "the AI runtime check printed no report: {}",
                tail_text(&done.stderr)
            )
        })?;
        ensure!(
            report["ok"] == true,
            "the AI runtime is incomplete: {}. System libraries needed: libseccomp2 (network sandbox) and, for MediaPipe, libegl1 and libgles2.",
            clean(&report["missing"].to_string())
        );
        Ok(report)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // Dropping the worker kills its group and reaps it.
        if let Ok(mut worker) = self.worker.lock() {
            *worker = None;
        }
    }
}

// ---- owned child processes ---------------------------------------------------------------------

/// A child that leads its own process group. Dropping it kills the group and reaps the leader.
/// `child` is `None` once stopped. A leader that does not exit within `STOP_GRACE` after SIGKILL
/// (e.g. stuck in a GPU driver call) is handed, unreaped, to a detached reaper thread: it stays a
/// zombie-to-be until the kernel lets it go, so its PID cannot be reused and signalled by mistake.
struct Owned {
    child: Option<Child>,
    reaped: bool,
    lease: Option<ResidentLease>,
}

impl Owned {
    fn spawn(mut cmd: Command, stdin: Stdio) -> Result<Self> {
        cmd.process_group(0)
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let program = cmd.get_program().to_owned();
        let child = cmd
            .spawn()
            .with_context(|| format!("starting {}", Path::new(&program).display()))?;
        Ok(Self {
            child: Some(child),
            reaped: false,
            lease: None,
        })
    }

    /// stdin, stdout and stderr of the running child.
    fn pipes(&mut self) -> Result<(Option<ChildStdin>, ChildStdout, ChildStderr)> {
        let child = self
            .child
            .as_mut()
            .context("the child was already stopped")?;
        Ok((
            child.stdin.take(),
            child.stdout.take().context("child has no stdout")?,
            child.stderr.take().context("child has no stderr")?,
        ))
    }

    fn is_stopped(&self) -> bool {
        self.child.is_none() || self.reaped
    }

    /// Observes the leader's exit without reaping it (`waitid` + `WNOWAIT`), then kills its private
    /// process group while the zombie still pins the PID, and only then reaps. Orphaned descendants
    /// (a launcher that exits while its workers run) therefore never survive, and the group is never
    /// signalled by a number that could have been recycled.
    fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        let child = self
            .child
            .as_mut()
            .context("the child was already stopped")?;
        // SAFETY: an all-zero siginfo_t is a valid out-parameter for waitid.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        loop {
            // SAFETY: `info` is valid for writes; the pid is our unreaped child.
            let rc = unsafe {
                libc::waitid(
                    libc::P_PID,
                    child.id() as libc::id_t,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if rc == 0 {
                break;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error).context("checking the child process");
            }
        }
        // SAFETY: waitid succeeded and filled the SIGCHLD fields.
        if unsafe { info.si_pid() } == 0 {
            return Ok(None);
        }
        let _ = stop_group(child);
        let status = child.wait()?;
        self.reaped = true;
        Ok(Some(status))
    }

    fn stop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        if self.reaped {
            return;
        }
        // Group kill first, while the leader is still unreaped (its PID cannot have been recycled).
        let _ = stop_group(&mut child);
        let deadline = Instant::now() + STOP_GRACE;
        loop {
            match child.try_wait() {
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                Ok(None) => {
                    let lease = self.lease.take();
                    thread::spawn(move || {
                        let _ = child.wait();
                        drop(lease);
                    });
                    return;
                }
                Ok(Some(_)) | Err(_) => {
                    self.reaped = true;
                    return;
                }
            }
        }
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        self.stop();
    }
}

type Tail = Arc<Mutex<Vec<u8>>>;

/// Keeps the last `cap` bytes of a pipe in the returned buffer, always draining it so the child never
/// blocks on output. The channel yields once the pipe is closed.
fn collect(mut pipe: impl Read + Send + 'static, cap: usize) -> (Tail, Receiver<()>) {
    let tail = Tail::default();
    let (done, closed) = mpsc::channel();
    let sink = Arc::clone(&tail);
    thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n @ 1..) = pipe.read(&mut buf) {
            let mut tail = sink.lock().unwrap();
            tail.extend_from_slice(&buf[..n]);
            let excess = tail.len().saturating_sub(cap);
            tail.drain(..excess);
        }
        let _ = done.send(());
    });
    (tail, closed)
}

struct Finished {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Runs a short-lived owned child to completion with bounded output. Cancel or timeout kills the whole
/// group before returning, so no descendant survives.
fn run_child(cmd: Command, cancel: &AtomicBool, limit: Duration) -> Result<Finished> {
    let mut owned = Owned::spawn(cmd, Stdio::null())?;
    let (_, stdout, stderr) = owned.pipes()?;
    let (stdout, stdout_closed) = collect(stdout, TAIL * 4);
    let (stderr, stderr_closed) = collect(stderr, TAIL);
    let deadline = Instant::now() + limit;
    let status = loop {
        if let Some(status) = owned.try_wait()? {
            break status;
        }
        if cancel.load(Ordering::SeqCst) {
            owned.stop();
            return Err(Cancelled { job_id: None }.into());
        }
        if Instant::now() >= deadline {
            owned.stop();
            bail!("timed out after {} s and was stopped", limit.as_secs());
        }
        thread::sleep(POLL);
    };
    // A straggler holding a pipe open must not hang us: the readers are detached, output is best effort.
    for closed in [stdout_closed, stderr_closed] {
        let _ = closed.recv_timeout(Duration::from_secs(2));
    }
    let (stdout, stderr) = (
        stdout.lock().unwrap().clone(),
        stderr.lock().unwrap().clone(),
    );
    Ok(Finished {
        status,
        stdout,
        stderr,
    })
}

// ---- resident worker ---------------------------------------------------------------------------

enum Msg {
    Reply(Value, Vec<u8>),
    Closed(String),
}

struct Worker {
    child: Owned,
    stdin: ChildStdin,
    rx: Receiver<Msg>,
    stderr: Tail,
    /// The model the worker reported as loaded after its last reply.
    model: Option<String>,
    next_id: u64,
}

fn worker_stopped(stderr: &Tail, why: &str) -> anyhow::Error {
    thread::sleep(Duration::from_millis(100)); // let the stderr reader catch the last lines
    anyhow!(
        "the AI worker stopped ({why}): {}",
        tail_text(&stderr.lock().unwrap())
    )
}

impl Worker {
    fn exited(&mut self) -> bool {
        self.child.is_stopped() || !matches!(self.child.try_wait(), Ok(None))
    }

    /// One request/reply on the calling thread: the pipe is non-blocking and every wait polls
    /// `cancel` and `deadline`. On cancel, deadline or a broken stream the group is killed (and the
    /// leader reaped, boundedly) before this returns.
    fn infer(
        &mut self,
        mut header: Value,
        rgb: &[u8],
        cancel: &AtomicBool,
        deadline: Instant,
    ) -> Result<(Value, Vec<u8>)> {
        self.next_id += 1;
        let id = self.next_id;
        header["id"] = id.into();
        let head = frame_head(&header, rgb.len())?;
        while self.rx.try_recv().is_ok() {}
        let outcome = self.exchange(id, &head, rgb, cancel, deadline);
        if outcome.is_err() {
            self.child.stop();
        }
        outcome
    }

    fn exchange(
        &mut self,
        id: u64,
        head: &[u8],
        rgb: &[u8],
        cancel: &AtomicBool,
        deadline: Instant,
    ) -> Result<(Value, Vec<u8>)> {
        let aborted = || -> Result<()> {
            if cancel.load(Ordering::SeqCst) {
                return Err(Cancelled { job_id: None }.into());
            }
            ensure!(
                Instant::now() < deadline,
                "the model run timed out and the worker was stopped"
            );
            Ok(())
        };
        let total = head.len() + rgb.len();
        let mut sent = 0;
        while sent < total {
            aborted()?;
            let chunk = if sent < head.len() {
                &head[sent..]
            } else {
                &rgb[sent - head.len()..]
            };
            match self.stdin.write(chunk) {
                Ok(0) => return Err(worker_stopped(&self.stderr, "its input closed")),
                Ok(n) => sent += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Wait for pipe space, not a reply that requires the complete frame.
                    match self.rx.try_recv() {
                        Ok(Msg::Closed(why)) => return Err(worker_stopped(&self.stderr, &why)),
                        Ok(Msg::Reply(..)) => {
                            bail!("the AI worker answered before the frame was sent")
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                        Err(mpsc::TryRecvError::Disconnected) => {
                            return Err(worker_stopped(&self.stderr, "its output closed"));
                        }
                    }
                    let mut descriptor = libc::pollfd {
                        fd: self.stdin.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    // SAFETY: the borrowed stdin descriptor stays open for this call.
                    if unsafe { libc::poll(&mut descriptor, 1, POLL.as_millis() as i32) } < 0 {
                        let error = std::io::Error::last_os_error();
                        if error.kind() != std::io::ErrorKind::Interrupted {
                            return Err(error).context("waiting to send the camera frame");
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                    return Err(worker_stopped(&self.stderr, "its input closed"));
                }
                Err(e) => return Err(e).context("sending the frame to the AI worker"),
            }
        }
        loop {
            match self.rx.recv_timeout(POLL) {
                Ok(Msg::Reply(reply, body)) if reply["id"] == id => {
                    aborted()?;
                    return Ok((reply, body));
                }
                Ok(Msg::Reply(..)) => bail!("the AI worker answered out of order"),
                Ok(Msg::Closed(why)) => return Err(worker_stopped(&self.stderr, &why)),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(worker_stopped(&self.stderr, "its output closed"));
                }
                Err(RecvTimeoutError::Timeout) => aborted()?,
            }
        }
    }
}

fn read_replies(stdout: impl Read, tx: mpsc::Sender<Msg>) {
    let mut stdout = std::io::BufReader::new(stdout);
    loop {
        match read_frame(&mut stdout, MAX_OUTPUT_BYTES) {
            Ok(Some((header, body))) => {
                if tx.send(Msg::Reply(header, body)).is_err() {
                    return;
                }
            }
            Ok(None) => {
                let _ = tx.send(Msg::Closed("exited".into()));
                return;
            }
            Err(error) => {
                let _ = tx.send(Msg::Closed(format!("{error:#}")));
                return;
            }
        }
    }
}

// ---- framing -----------------------------------------------------------------------------------

/// `>II` header length and body length, JSON header, body. Limits are checked before allocating.
fn read_frame(reader: &mut impl Read, max_body: usize) -> Result<Option<(Value, Vec<u8>)>> {
    let mut prefix = [0u8; 8];
    let mut got = 0;
    while got < prefix.len() {
        match reader.read(&mut prefix[got..])? {
            0 if got == 0 => return Ok(None),
            0 => bail!("the stream ended inside a frame"),
            n => got += n,
        }
    }
    let header_len = u32::from_be_bytes(prefix[..4].try_into().unwrap()) as usize;
    let body_len = u32::from_be_bytes(prefix[4..].try_into().unwrap()) as usize;
    ensure!(
        header_len <= MAX_HEADER,
        "reply header exceeds {MAX_HEADER} bytes"
    );
    ensure!(body_len <= max_body, "reply body exceeds {max_body} bytes");
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let mut body = vec![0u8; body_len];
    reader.read_exact(&mut body)?;
    Ok(Some((
        serde_json::from_slice(&header).context("reply header is not JSON")?,
        body,
    )))
}

/// The `>II` prefix and JSON header of a request; the body follows separately.
fn frame_head(header: &Value, body_len: usize) -> Result<Vec<u8>> {
    let header = serde_json::to_vec(header)?;
    ensure!(
        header.len() <= MAX_HEADER && body_len <= MAX_REQUEST_BODY,
        "request is too large"
    );
    let mut out = Vec::with_capacity(8 + header.len());
    out.extend_from_slice(&(header.len() as u32).to_be_bytes());
    out.extend_from_slice(&(body_len as u32).to_be_bytes());
    out.extend_from_slice(&header);
    Ok(out)
}

/// Makes the parent's end of a pipe non-blocking, so a stuck child can never block a write and
/// cancellation stays responsive. The child's end is a separate file description and is unaffected.
fn set_nonblocking(fd: &impl AsRawFd) -> Result<()> {
    let fd = fd.as_raw_fd();
    // SAFETY: `fd` is a live pipe descriptor owned by the caller's ChildStdin.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error()).context("configuring the worker pipe");
    }
    Ok(())
}

// ---- filesystem and locks ----------------------------------------------------------------------

fn make_dir(path: &Path) -> Result<()> {
    DirBuilder::new()
        .mode(0o700)
        .create(path)
        .with_context(|| format!("creating {}", path.display()))
}

/// A directory this app may use: created 0700 if absent, never a symlink or file.
fn owned_dir(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => bail!(
            "{} must be a real directory (not a symlink or file); move it away so link-camera can create it",
            path.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => match make_dir(path) {
            Err(_) if fs::symlink_metadata(path).is_ok_and(|m| m.is_dir()) => Ok(()),
            other => other,
        },
        Err(e) => Err(e).with_context(|| format!("inspecting {}", path.display())),
    }
}

/// Deletes a directory tree, or unlinks a file/symlink, without ever following a link.
fn remove_owned(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
    .with_context(|| format!("removing {}", path.display()))
}

/// Sum of regular file sizes without following links. `strict` rejects links and special files.
fn tree_bytes(root: &Path, strict: bool) -> Result<u64> {
    let (mut total, mut seen, mut stack) = (0u64, 0usize, vec![root.to_path_buf()]);
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            seen += 1;
            ensure!(
                seen <= MAX_TREE_ENTRIES,
                "{} holds too many files",
                root.display()
            );
            let meta = entry.metadata()?; // does not follow symlinks
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total += meta.len();
            } else {
                ensure!(
                    !strict,
                    "{} is a link or special file; refusing to publish it",
                    entry.path().display()
                );
            }
        }
    }
    Ok(total)
}

fn marker(dir: &Path) -> Option<Value> {
    let path = dir.join(MARKER);
    let meta = fs::symlink_metadata(&path).ok()?;
    if !meta.is_file() || meta.len() > 64 << 10 {
        return None;
    }
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

struct ResidentLease {
    lock: Option<LockFile>,
    released: mpsc::Sender<()>,
}

impl Drop for ResidentLease {
    fn drop(&mut self) {
        drop(self.lock.take());
        let _ = self.released.send(());
    }
}

struct LockFile(File);

impl std::ops::Deref for LockFile {
    type Target = File;

    fn deref(&self) -> &File {
        &self.0
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        // A concurrent fork may retain this open file description after our descriptor closes.
        let _ = self.0.unlock();
    }
}

fn try_lock(file: &File, exclusive: bool) -> Result<bool> {
    match if exclusive {
        file.try_lock()
    } else {
        file.try_lock_shared()
    } {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

/// Waits for an exclusive lock held by another process (e.g. a runtime setup), honouring `cancel`.
fn lock_wait(file: &File, cancel: &AtomicBool) -> Result<()> {
    while !try_lock(file, true)? {
        if cancel.load(Ordering::SeqCst) {
            return Err(Cancelled { job_id: None }.into());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let home = std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/bin"));
    std::env::split_paths(&path)
        .chain(home)
        .map(|dir| dir.join(name))
        .find(|p| fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

/// Runtime identity: a new requirements file builds a new runtime next to the old one.
fn runtime_id() -> String {
    let mut hash = Sha256::new();
    hash.update(b"python3.11\0");
    hash.update(REQUIREMENTS);
    hex(&hash.finalize())[..16].to_string()
}

/// Worker text shown in a UI: no control characters, bounded.
fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(600)
        .collect()
}

/// The end of a captured stream (where the real error is), single line, bounded.
fn tail_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let flat: Vec<char> = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    flat[flat.len().saturating_sub(600)..]
        .iter()
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "link-camera-{name}-{}-{}",
            std::process::id(),
            nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn client(root: &Path, python: Option<PathBuf>) -> Client {
        Client::at(root.join("ai"), RuntimeConfig::default(), python).unwrap()
    }

    fn frame() -> Frame {
        Frame {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
            sequence: 0,
        }
    }

    fn alive(pid: u32) -> bool {
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        stat.rsplit(") ")
            .next()
            .is_some_and(|rest| !rest.starts_with('Z'))
    }

    /// A reply frame as the Python worker writes it (`>II`, JSON, body), built by hand.
    fn reply(header: &[u8], body: &[u8]) -> Vec<u8> {
        let mut wire = (header.len() as u32).to_be_bytes().to_vec();
        wire.extend_from_slice(&(body.len() as u32).to_be_bytes());
        wire.extend_from_slice(header);
        wire.extend_from_slice(body);
        wire
    }

    #[test]
    fn frame_parser_enforces_boundaries() {
        let wire = reply(br#"{"id":7}"#, b"abc");
        assert!(
            read_frame(&mut Cursor::new(&wire), 2).is_err(),
            "body over the cap"
        );
        assert!(
            read_frame(&mut Cursor::new(Vec::<u8>::new()), 16)
                .unwrap()
                .is_none()
        );
        assert!(
            read_frame(&mut Cursor::new(&wire[..5]), 16).is_err(),
            "truncated prefix"
        );
        assert!(
            read_frame(&mut Cursor::new(&wire[..10]), 16).is_err(),
            "truncated header"
        );
        assert!(
            read_frame(&mut Cursor::new(&wire[..wire.len() - 1]), 16).is_err(),
            "truncated body"
        );
        assert!(read_frame(&mut Cursor::new(reply(b"not json", b"")), 16).is_err());
        let mut huge = vec![0xff; 8];
        huge.extend_from_slice(b"{}");
        assert!(
            read_frame(&mut Cursor::new(huge), usize::MAX).is_err(),
            "header over the cap"
        );
        let mut wire = (MAX_HEADER as u32 + 1).to_be_bytes().to_vec();
        wire.extend_from_slice(&0u32.to_be_bytes());
        assert!(
            read_frame(&mut Cursor::new(wire), usize::MAX).is_err(),
            "header just over the cap"
        );
    }

    /// A launcher that exits at once (like a package-manager wrapper) must not leave its background
    /// descendants behind: the group is killed as soon as the leader's exit is observed.
    #[test]
    fn leader_exit_kills_orphaned_descendants() {
        let dir = scratch("orphan");
        let pidfile = dir.join("pid");
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("sleep 300 & echo $! > \"$1\"")
            .arg("sh")
            .arg(&pidfile);
        let done = run_child(cmd, &AtomicBool::new(false), Duration::from_secs(60)).unwrap();
        assert!(done.status.success());
        let pid: u32 = fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        for _ in 0..40 {
            if !alive(pid) {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(!alive(pid), "a descendant outlived its exited leader");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancel_kills_the_child_and_its_descendants() {
        let dir = scratch("cancel");
        let pidfile = dir.join("pid");
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("sleep 300 & echo $! > \"$1\"; wait")
            .arg("sh")
            .arg(&pidfile);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let trigger = thread::spawn(move || {
            thread::sleep(Duration::from_millis(400));
            flag.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let error = run_child(cmd, &cancel, Duration::from_secs(60))
            .err()
            .expect("cancelled");
        trigger.join().unwrap();
        assert!(error.is::<Cancelled>());
        assert!(started.elapsed() < Duration::from_secs(10));
        let pid: u32 = fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        for _ in 0..40 {
            if !alive(pid) {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(
            !alive(pid),
            "the descendant of the cancelled child is still running"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// A worker that never reads its input fills the pipe: the send must not block past the deadline
    /// or cancellation, and the group is killed and reaped.
    #[test]
    fn unresponsive_worker_cannot_block_send_and_is_killed() {
        for cancelled in [false, true] {
            let mut cmd = Command::new("sleep");
            cmd.arg("300");
            let mut child = Owned::spawn(cmd, Stdio::piped()).unwrap();
            let pid = child.child.as_ref().unwrap().id();
            let (stdin, stdout, stderr) = child.pipes().unwrap();
            let stdin = stdin.unwrap();
            set_nonblocking(&stdin).unwrap();
            drop((stdout, stderr));
            let (_tx, rx) = mpsc::channel();
            let mut worker = Worker {
                child,
                stdin,
                rx,
                stderr: Tail::default(),
                model: None,
                next_id: 0,
            };
            let cancel = AtomicBool::new(cancelled);
            let started = Instant::now();
            let rgb = vec![0u8; 4 << 20]; // far beyond the 64 KiB pipe buffer
            let error = worker
                .infer(
                    json!({}),
                    &rgb,
                    &cancel,
                    started + Duration::from_millis(400),
                )
                .unwrap_err();
            assert_eq!(error.is::<Cancelled>(), cancelled, "{error:#}");
            assert!(started.elapsed() < Duration::from_secs(10));
            assert!(worker.exited(), "the worker must be stopped");
            assert!(
                !alive(pid),
                "the worker process is still running or unreaped"
            );
        }
    }

    #[test]
    fn timeout_kills_the_child() {
        let mut cmd = Command::new("sleep");
        cmd.arg("300");
        let started = Instant::now();
        let error = run_child(cmd, &AtomicBool::new(false), Duration::from_millis(300))
            .err()
            .unwrap();
        assert!(format!("{error}").contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn delete_only_removes_the_managed_model_and_respects_locks() {
        let dir = scratch("delete");
        let outside = dir.join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("weights.bin"), b"external cache").unwrap();
        let ai = client(&dir, None);
        ai.prepare().unwrap();
        let model = ai.model_dir("yolo11n");

        // A model directory that is really a link elsewhere is never followed or removed through.
        std::os::unix::fs::symlink(&outside, &model).unwrap();
        assert!(ai.delete("yolo11n").is_err());
        assert!(outside.join("weights.bin").exists());
        fs::remove_file(&model).unwrap();

        // A managed model containing a link to external data: the link is dropped, the target survives.
        fs::create_dir(&model).unwrap();
        fs::write(
            model.join(MARKER),
            br#"{"format":1,"model":"yolo11n","metadata":{}}"#,
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.join("weights.bin"), model.join("link")).unwrap();
        assert!(
            ai.catalog()
                .unwrap()
                .iter()
                .any(|m| m.id == "yolo11n" && m.installed)
        );

        // Another holder of the model lock (a run, here or elsewhere) blocks deletion.
        let busy = ai.open_lock("yolo11n").unwrap();
        assert!(try_lock(&busy, false).unwrap());
        assert!(ai.delete("yolo11n").is_err());
        assert!(model.exists());
        // A concurrent fork can retain the same open file description until its exec.
        let inherited = busy.try_clone().unwrap();
        drop(busy);

        ai.delete("yolo11n").unwrap();
        drop(inherited);
        assert!(!model.exists());
        assert_eq!(
            fs::read(outside.join("weights.bin")).unwrap(),
            b"external cache"
        );
        assert!(
            ai.delete("yolo11n").is_err(),
            "deleting what is not installed is reported"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resident_lease_blocks_duplicates_and_stop_releases_only_its_worker() {
        let dir = scratch("resident-lease");
        let python = dir.join("python");
        fs::write(&python, "#!/bin/sh\nexec sleep 300\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
        let first = client(&dir, Some(python.clone()));
        let second = client(&dir, Some(python.clone()));
        let other_dir = scratch("other-resident");
        let other = client(&other_dir, Some(python.clone()));
        first.prepare().unwrap();
        other.prepare().unwrap();
        let worker = first.spawn_worker(&python, &dir).unwrap();
        let pid = worker.child.child.as_ref().unwrap().id();
        *first.worker.lock().unwrap() = Some(worker);
        let unrelated = other.spawn_worker(&python, &dir).unwrap();
        let unrelated_pid = unrelated.child.child.as_ref().unwrap().id();
        *other.worker.lock().unwrap() = Some(unrelated);
        let error = second
            .spawn_worker(&python, &dir)
            .err()
            .expect("duplicate lease");
        assert!(format!("{error:#}").contains("another local AI worker"));
        assert!(second.resident_busy().unwrap());
        let model = first.model_dir("yolo11n");
        fs::create_dir(&model).unwrap();
        fs::write(
            model.join(MARKER),
            br#"{"format":1,"model":"yolo11n","metadata":{}}"#,
        )
        .unwrap();
        first.status.lock().unwrap().loaded = Some(("yolo11n".into(), "cpu".into()));
        let catalog = second.catalog().unwrap();
        let info = catalog.iter().find(|m| m.id == "yolo11n").unwrap();
        assert_eq!(info.available, Some(false));
        assert!(
            info.unavailable_reason
                .as_deref()
                .unwrap()
                .contains("another local AI worker")
        );
        first.stop();
        first.stop();
        assert!(!alive(pid));
        assert!(
            alive(unrelated_pid),
            "Stop must not signal another store's worker"
        );
        assert!(!second.resident_busy().unwrap());
        assert!(
            first
                .catalog()
                .unwrap()
                .iter()
                .all(|m| !m.loaded && !m.loading)
        );
        let mut replacement = second.spawn_worker(&python, &dir).unwrap();
        let pending = replacement.child.lease.take().unwrap();
        drop(replacement);
        assert!(
            second.resident_busy().unwrap(),
            "a delayed reaper must retain ownership"
        );
        assert!(matches!(
            second.cleanup.lock().unwrap().as_ref().unwrap().try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        drop(pending);
        second.stop();
        other.stop();
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(other_dir).unwrap();
    }

    #[test]
    fn resident_lease_is_released_after_start_failure_and_active_cancellation() {
        let dir = scratch("resident-active");
        let ai = Arc::new(client(&dir, None));
        ai.prepare().unwrap();
        assert!(ai.spawn_worker(&dir.join("missing-python"), &dir).is_err());
        assert!(!ai.resident_busy().unwrap());
        let python = dir.join("python");
        fs::write(&python, "#!/bin/sh\nexec sleep 300\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
        let worker = ai.spawn_worker(&python, &dir).unwrap();
        let pid = worker.child.child.as_ref().unwrap().id();
        *ai.worker.lock().unwrap() = Some(worker);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let running = ai.clone();
        let (started, ready) = mpsc::channel();
        let thread = thread::spawn(move || {
            let mut worker = running.worker.lock().unwrap();
            started.send(()).unwrap();
            worker.as_mut().unwrap().infer(
                json!({}),
                &[],
                &flag,
                Instant::now() + Duration::from_secs(10),
            )
        });
        ready.recv().unwrap();
        cancel.store(true, Ordering::SeqCst);
        ai.stop();
        assert!(thread.join().unwrap().unwrap_err().is::<Cancelled>());
        assert!(!alive(pid));
        assert!(!ai.resident_busy().unwrap());
        assert!(ai.worker.lock().unwrap().is_none());
        drop(ai);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_releases_a_dead_workers_lease() {
        let dir = scratch("resident-dead");
        let python = dir.join("python");
        fs::write(&python, "#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
        let ai = client(&dir, Some(python.clone()));
        let peer = client(&dir, Some(python.clone()));
        ai.prepare().unwrap();
        let mut worker = ai.spawn_worker(&python, &dir).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while worker.child.try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline, "test worker did not exit");
            thread::sleep(POLL);
        }
        *ai.worker.lock().unwrap() = Some(worker);
        ai.status.lock().unwrap().loaded = Some(("yolo11n".into(), "cpu".into()));
        assert!(peer.resident_busy().unwrap());
        assert!(
            ai.catalog()
                .unwrap()
                .iter()
                .all(|m| !m.loaded && !m.loading)
        );
        assert!(!peer.resident_busy().unwrap());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_lists_all_models_without_a_runtime() {
        let dir = scratch("catalog");
        let ai = client(&dir, None);
        let list = ai.catalog().unwrap();
        assert_eq!(list.len(), super::super::MODEL_IDS.len());
        assert!(
            list.iter()
                .all(|m| !m.installed && m.available == Some(false) && m.device.is_empty())
        );
        assert!(!dir.join("ai").exists(), "catalog must not create anything");
        let error = ai
            .run("yolo11n", &json!({}), &frame(), &AtomicBool::new(false))
            .unwrap_err();
        assert!(format!("{error}").contains("not downloaded"), "{error}");
        fs::remove_dir_all(dir).unwrap();
    }

    /// The worker's OS-level network ban, exercised through libc socket() in a real worker process:
    /// IPv4/IPv6 are refused with EPERM while AF_UNIX (local CUDA) still works.
    #[test]
    fn worker_confinement_blocks_inet_sockets_but_not_unix() {
        let python = PathBuf::from("/usr/bin/python3");
        if !python.is_file() {
            return;
        }
        let dir = scratch("netcheck");
        let ai = client(&dir, Some(python.clone()));
        ai.prepare().unwrap();
        let code = ai.code_dir().unwrap();
        let mut cmd = ai.command(&python, &code, false);
        cmd.arg("netcheck");
        let done = run_child(cmd, &AtomicBool::new(false), Duration::from_secs(60)).unwrap();
        assert!(
            done.status.success(),
            "netcheck failed (is libseccomp2 installed?): {}",
            tail_text(&done.stderr)
        );
        let report: Value = serde_json::from_slice(&done.stdout).unwrap();
        assert_eq!(
            report["inet"], 1,
            "AF_INET socket() must fail with EPERM: {report}"
        );
        assert_eq!(
            report["inet6"], 1,
            "AF_INET6 socket() must fail with EPERM: {report}"
        );
        assert_eq!(
            report["unix"], true,
            "AF_UNIX must stay available: {report}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// Drives the real embedded worker.py over real pipes with the host's python3 (no packages needed:
    /// the model load fails honestly, the worker survives and answers again).
    #[test]
    fn real_worker_reports_load_failures_and_survives() {
        // A pyenv shim would not survive the cleared environment; use the real interpreter or skip.
        let python = PathBuf::from("/usr/bin/python3");
        if !python.is_file() {
            return;
        }
        let dir = scratch("worker");
        let ai = client(&dir, Some(python));
        ai.prepare().unwrap();
        let model = ai.model_dir("florence-2-base");
        fs::create_dir(&model).unwrap();
        fs::write(
            model.join(MARKER),
            br#"{"format":1,"model":"florence-2-base","metadata":{}}"#,
        )
        .unwrap();
        let options = json!({"task": "caption"});
        let cancel = AtomicBool::new(false);
        for _ in 0..2 {
            let error = ai
                .run("florence-2-base", &options, &frame(), &cancel)
                .unwrap_err();
            let text = format!("{error:#}");
            assert!(
                text.contains("failed:") && !text.contains("worker stopped"),
                "{text}"
            );
        }
        assert!(
            ai.worker.lock().unwrap().is_some(),
            "an in-worker error must not kill the worker"
        );
        cancel.store(true, Ordering::SeqCst);
        let cancelled = ai
            .run("florence-2-base", &options, &frame(), &cancel)
            .unwrap_err();
        assert!(cancelled.is::<Cancelled>());
        drop(ai);
        fs::remove_dir_all(dir).unwrap();
    }
}
