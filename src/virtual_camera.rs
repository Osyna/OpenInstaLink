//! Local virtual camera lifecycle: create/list/remove v4l2loopback nodes owned by Link Camera and
//! stop the FFmpeg producers that feed them, from any Link Camera process (GUI, CLI, other instance).
//!
//! Ownership lives in one lock-protected JSON file (`$XDG_STATE_HOME/link-camera/virtual-camera/`):
//! - A node is recorded as ours when its number + card label + boot id matches sysfs. Deletion
//!   rechecks the label after authorization, but the driver's number-only delete cannot atomically
//!   preserve identity against concurrent device removal/recreation by other administrative tools.
//! - A producer is ours only when its `/proc` start time and the private `LINK_CAMERA_VCAM` token in
//!   its environment both match its record, and it is signalled through a pidfd opened *before* that
//!   check, so a reused PID is never killed. Nothing here ever matches processes by name.
//!
//! Kernel modules are never unloaded; nodes are added and deleted one by one with
//! `v4l2loopback-ctl`. Elevation (`pkexec`) happens only when the caller allows it.

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use std::ffi::{c_int, c_uint, c_void};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Environment variable carrying a producer's private token (read back from `/proc/<pid>/environ`).
pub(crate) const PRODUCER_ENV: &str = "LINK_CAMERA_VCAM";
const SYSFS_CLASS: &str = "/sys/class/video4linux";
const SYSFS_VIRTUAL: &str = "/sys/devices/virtual/video4linux";
const MODULE_DIR: &str = "/sys/module/v4l2loopback";
const CONTROL_NODE: &str = "/dev/v4l2loopback";
const DEFAULT_LABEL: &str = "Link Camera";
/// 31-byte V4L2 card limit minus the ` #xxxxxx` suffix `create` appends.
const LABEL_MAX: usize = 23;

/// Whether an administrator prompt (`pkexec`) may be shown when a step needs root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elevation {
    /// Fail with the exact command to run yourself.
    Refuse,
    /// Run just the privileged helper through `pkexec` (polkit asks the user).
    Pkexec,
}

#[derive(Clone, Debug)]
pub struct CreateOptions {
    /// Base name shown to other applications: 1-23 printable ASCII characters (a ` #xxxxxx` suffix is added).
    pub label: String,
    pub elevation: Elevation,
    /// Also load the installed driver with zero implicit nodes before creating an owned one.
    pub load_module: bool,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            label: DEFAULT_LABEL.into(),
            elevation: Elevation::Refuse,
            load_module: false,
        }
    }
}

/// One v4l2loopback node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VirtualCamera {
    /// `/dev/videoN`.
    pub path: PathBuf,
    pub number: u32,
    pub label: String,
    /// Created by Link Camera and still the same device (only these can be removed here).
    pub owned: bool,
    /// PID of the live Link Camera FFmpeg feeding the node, from any instance.
    pub producer_pid: Option<u32>,
}

/// What a bulk operation did; `failures` are per-item, human-readable and never empty on partial failure.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Report {
    pub stopped: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub failures: Vec<String>,
}

/// What the host has for the virtual camera; `missing()` gives exact actions.
#[derive(Clone, Debug, Serialize)]
pub struct Prerequisites {
    pub ffmpeg: bool,
    pub ctl: Option<PathBuf>,
    pub pkexec: bool,
    pub module_installed: bool,
    pub module_loaded: bool,
    /// The current user can open `/dev/v4l2loopback` (no elevation needed to add/delete nodes).
    pub control_writable: bool,
    pub root: bool,
}

impl Prerequisites {
    pub fn missing(&self) -> Vec<String> {
        let mut todo = Vec::new();
        if !self.ffmpeg {
            todo.push("Install ffmpeg; it feeds the processed video to the virtual camera.".into());
        }
        if self.ctl.is_none() {
            todo.push("Install v4l2loopback-ctl (package v4l2loopback-utils; Arch: `sudo pacman -S v4l2loopback-utils`, Debian/Ubuntu: `sudo apt install v4l2loopback-utils`). Version 0.13 or newer is needed for per-device add/delete.".into());
        }
        if !self.module_loaded {
            todo.push(if self.module_installed {
                "Load the installed kernel module without implicit devices: `sudo modprobe v4l2loopback devices=0` (or authorize Link Camera through pkexec). Existing loaded modules and other cameras are left alone.".into()
            } else {
                "Install the v4l2loopback kernel module for the running kernel (v4l2loopback-dkms plus the kernel headers, or your distribution's prebuilt package); `modinfo v4l2loopback` must succeed.".into()
            });
        } else if !self.control_writable && !self.root && !self.pkexec {
            todo.push(format!(
                "Adding and removing nodes needs write access to {CONTROL_NODE}: install polkit (pkexec) so Link Camera can ask for administrator rights, or run v4l2loopback-ctl as root."
            ));
        }
        todo
    }
}

pub fn prerequisites() -> Prerequisites {
    let module_loaded = Path::new(MODULE_DIR).exists();
    Prerequisites {
        ffmpeg: find_program("ffmpeg").is_some(),
        ctl: trusted_program("v4l2loopback-ctl"),
        pkexec: trusted_program("pkexec").is_some(),
        module_loaded,
        module_installed: module_loaded
            || trusted_program("modinfo").is_some_and(|modinfo| {
                Command::new(modinfo)
                    .args(["-n", "v4l2loopback"])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            }),
        control_writable: control_writable(),
        root: is_root(),
    }
}

/// Every v4l2loopback node, flagged with ownership and the live producer, sorted by number.
pub fn list() -> Result<Vec<VirtualCamera>> {
    list_in(&Store::default_store()?)
}

/// Add a node for this app (`v4l2loopback-ctl add`; other devices and the module stay untouched).
pub fn create(options: &CreateOptions) -> Result<VirtualCamera> {
    // A random suffix makes the card label (checked exactly against sysfs) unique to this node.
    let label = format!(
        "{} #{}",
        valid_label(&options.label)?,
        &random_token()?[..6]
    );
    let store = Store::default_store()?;
    // Fail before any privileged change if ownership cannot be recorded afterwards.
    store
        .update(|_| ())
        .context("the virtual camera state file is not usable; fix it before creating a camera")?;
    let mut pre = prerequisites();
    if options.load_module && pre.ctl.is_some() && pre.module_installed && !pre.module_loaded {
        let modprobe = trusted_program("modprobe")
            .ok_or_else(|| anyhow!("modprobe not found in a system directory; run `sudo modprobe v4l2loopback devices=0`"))?;
        privileged(
            &modprobe,
            &["v4l2loopback", "devices=0"],
            options.elevation,
            false,
            "modprobe v4l2loopback devices=0",
        )?;
        wait_for(|| Path::new(CONTROL_NODE).exists(), Duration::from_secs(3));
        pre = prerequisites();
    }
    let missing = pre.missing();
    ensure!(
        missing.is_empty(),
        "the virtual camera cannot be set up yet:\n- {}",
        missing.join("\n- ")
    );
    let ctl = pre.ctl.expect("checked by missing()");
    let stdout = privileged(
        &ctl,
        &["add", "-n", &label, "-x", "1"],
        options.elevation,
        pre.control_writable,
        "v4l2loopback-ctl add",
    )?;
    // The tool prints `/dev/videoN` (or `N`); fall back to the one new node carrying our label.
    let printed = stdout.lines().find_map(|line| {
        let line = line.trim();
        line.strip_prefix("/dev/video")
            .unwrap_or(line)
            .parse::<u32>()
            .ok()
    });
    let mut found = None;
    wait_for(
        || {
            let candidates: Vec<u32> = nodes_now()
                .into_iter()
                .filter(|(number, name)| *name == label && printed.is_none_or(|p| p == *number))
                .map(|(number, _)| number)
                .collect();
            found =
                (candidates.len() == 1 && node_path(candidates[0]).exists()).then(|| candidates[0]);
            found.is_some()
        },
        Duration::from_secs(3),
    );
    let number = found.ok_or_else(|| {
        anyhow!("v4l2loopback-ctl reported success but no new node labelled {label:?} appeared; run `v4l2loopback-ctl list`")
    })?;
    if let Err(error) = store.update(|state| {
        state.nodes.push(Node {
            number,
            label: label.clone(),
            boot: boot_id(),
        })
    }) {
        let node = node_path(number).display().to_string();
        let rollback = delete_owned(
            &[(number, label.as_str())],
            options.elevation,
            pre.control_writable,
        );
        bail!(
            "created {node} but could not record ownership ({error:#}); {}",
            match rollback {
                Ok(_) => "the new node was deleted again".to_owned(),
                Err(e) => format!(
                    "deleting it failed too ({e:#}); remove it yourself: sudo v4l2loopback-ctl delete {number}"
                ),
            }
        );
    }
    Ok(VirtualCamera {
        path: node_path(number),
        number,
        label,
        owned: true,
        producer_pid: None,
    })
}

/// Delete a recorded node after stopping its producer; see the module's number-only delete limit.
pub fn remove(path: &Path, elevation: Elevation) -> Result<()> {
    let number = node_number(path)?;
    let mut report = Report::default();
    remove_in(
        &Store::default_store()?,
        Some(&[number]),
        elevation,
        &mut report,
    );
    finish(report)
}

/// Stop the Link Camera producer feeding `path` (any instance); `Ok(false)` when none is running.
pub fn stop(path: &Path) -> Result<bool> {
    let number = node_number(path)?;
    let mut report = Report::default();
    stop_in(&Store::default_store()?, Some(&[number]), &mut report);
    let stopped = !report.stopped.is_empty();
    finish(report)?;
    Ok(stopped)
}

/// Stop every producer recorded by any Link Camera process. Never touches other FFmpeg processes
/// and never needs elevation or an active preview.
pub fn stop_all() -> Report {
    let mut report = Report::default();
    match Store::default_store() {
        Ok(store) => stop_in(&store, None, &mut report),
        Err(error) => report.failures.push(format!("{error:#}")),
    }
    report
}

/// `stop_all` plus deleting every node Link Camera created.
pub fn remove_all(elevation: Elevation) -> Report {
    let mut report = Report::default();
    match Store::default_store() {
        Ok(store) => remove_in(&store, None, elevation, &mut report),
        Err(error) => report.failures.push(format!("{error:#}")),
    }
    report
}

fn finish(report: Report) -> Result<()> {
    ensure!(report.failures.is_empty(), "{}", report.failures.join("; "));
    Ok(())
}

/// Refuse real cameras; unknown (no sysfs entry) is allowed so existing loopback setups keep working.
pub(crate) fn ensure_loopback(path: &Path) -> Result<()> {
    let number = node_number(path)?;
    let entry = Path::new(SYSFS_CLASS).join(format!("video{number}"));
    if let Ok(real) = entry.canonicalize() {
        ensure!(
            real.starts_with(SYSFS_VIRTUAL),
            "{} is a real camera, not a v4l2loopback virtual camera; create one first",
            path.display()
        );
    }
    Ok(())
}

fn stop_in(store: &Store, only: Option<&[u32]>, report: &mut Report) {
    let producers = match store.update(|state| state.producers.clone()) {
        Ok(producers) => producers,
        Err(error) => return report.failures.push(format!("{error:#}")),
    };
    for producer in producers
        .into_iter()
        .filter(|p| only.is_none_or(|nodes| nodes.contains(&p.node)))
    {
        let path = node_path(producer.node);
        let token = producer.token.clone();
        // Mark and read under one lock so an `attach` racing with us either sees `stopping`
        // (and fails) or has already recorded the FFmpeg we are about to stop.
        let marked = store.update(|state| {
            state
                .producers
                .iter_mut()
                .find(|p| p.token == token)
                .map(|p| {
                    p.stopping = true;
                    p.ffmpeg.clone()
                })
        });
        let ffmpeg = match marked {
            Ok(Some(Some(ffmpeg))) => ffmpeg,
            // Still starting: the cancelled reservation stays (the node remains claimed) until its
            // owner sees the request in `attach`, kills its child and releases it.
            Ok(Some(None)) => {
                report.stopped.push(path);
                continue;
            }
            Ok(None) => continue,
            Err(error) => {
                report
                    .failures
                    .push(format!("{}: {error:#}", path.display()));
                continue;
            }
        };
        match terminate(&ffmpeg, &token) {
            Ok(killed) => {
                let _ = store.update(|state| state.producers.retain(|p| p.token != token));
                if killed {
                    report.stopped.push(path);
                }
            }
            Err(error) => report
                .failures
                .push(format!("{}: {error:#}", path.display())),
        }
    }
}

fn remove_in(store: &Store, wanted: Option<&[u32]>, elevation: Elevation, report: &mut Report) {
    let _removal = match RemovalGuard::acquire(store, wanted) {
        Ok(reservation) => reservation,
        Err(error) => return report.failures.push(format!("{error:#}")),
    };
    stop_in(store, wanted, report);
    let listed = match list_in(store) {
        Ok(listed) => listed,
        Err(error) => return report.failures.push(format!("{error:#}")),
    };
    let mut targets: Vec<&VirtualCamera> = listed
        .iter()
        .filter(|c| c.owned && wanted.is_none_or(|w| w.contains(&c.number)))
        .collect();
    for number in wanted.unwrap_or_default() {
        if !targets.iter().any(|c| c.number == *number) {
            report.failures.push(format!(
                "/dev/video{number} was not created by Link Camera (or is already gone); other devices are left alone. To remove it yourself: sudo v4l2loopback-ctl delete {number}"
            ));
        }
    }
    let claimed = match store.update(|state| {
        state
            .producers
            .iter()
            .map(|producer| producer.node)
            .collect::<Vec<_>>()
    }) {
        Ok(claimed) => claimed,
        Err(error) => return report.failures.push(format!("{error:#}")),
    };
    targets.retain(|target| {
        if claimed.contains(&target.number) {
            report.failures.push(format!(
                "{} still has a stopping output; wait for its cleanup before removing it again",
                target.path.display()
            ));
            false
        } else {
            true
        }
    });
    if targets.is_empty() {
        return;
    }
    let nodes: Vec<(u32, &str)> = targets
        .iter()
        .map(|c| (c.number, c.label.as_str()))
        .collect();
    let outcome = delete_owned(&nodes, elevation, control_writable());
    let now = match list_in(store) {
        Ok(now) => now,
        Err(error) => return report.failures.push(format!("{error:#}")),
    };
    let failed = outcome.as_ref().err().map(|error| format!("{error:#}"));
    for target in targets {
        if now.iter().any(|node| node.number == target.number) {
            if failed.is_none() {
                report.failures.push(format!(
                    "{} is still present after delete (in use by an application?)",
                    target.path.display()
                ));
            }
        } else {
            report.removed.push(target.path.clone());
        }
    }
    report.failures.extend(failed);
}

fn list_in(store: &Store) -> Result<Vec<VirtualCamera>> {
    let (nodes, owned, producers) = store.update(|state| {
        // Snapshot under the state lock so a concurrent create cannot lose its new record.
        let nodes = nodes_now();
        state.nodes.retain(|n| {
            nodes
                .iter()
                .any(|(number, label)| *number == n.number && *label == n.label)
        });
        (nodes, state.nodes.clone(), state.producers.clone())
    })?;
    Ok(nodes
        .into_iter()
        .map(|(number, label)| VirtualCamera {
            path: node_path(number),
            number,
            owned: owned.iter().any(|n| n.number == number && n.label == label),
            producer_pid: producers
                .iter()
                .find(|p| p.node == number)
                .and_then(|p| p.ffmpeg.as_ref().map(|f| f.pid)),
            label,
        })
        .collect())
}

/// shortcut: any virtual V4L2 node counts as loopback; upgrade if other virtual drivers matter.
fn nodes_now() -> Vec<(u32, String)> {
    if !Path::new(MODULE_DIR).exists() {
        return Vec::new();
    }
    let mut nodes: Vec<(u32, String)> = fs::read_dir(SYSFS_CLASS)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let number = entry
                .file_name()
                .to_str()?
                .strip_prefix("video")?
                .parse()
                .ok()?;
            entry
                .path()
                .canonicalize()
                .ok()?
                .starts_with(SYSFS_VIRTUAL)
                .then_some(())?;
            let label = fs::read_to_string(entry.path().join("name")).ok()?;
            Some((number, label.trim().to_owned()))
        })
        .collect();
    nodes.sort();
    nodes
}

fn node_path(number: u32) -> PathBuf {
    PathBuf::from(format!("/dev/video{number}"))
}

fn node_number(path: &Path) -> Result<u32> {
    path.parent()
        .is_some_and(|p| p == Path::new("/dev"))
        .then(|| {
            path.file_name()?
                .to_str()?
                .strip_prefix("video")?
                .parse()
                .ok()
        })
        .flatten()
        .ok_or_else(|| anyhow!("{} is not a /dev/videoN device", path.display()))
}

/// The user's part of the card label; `create` appends a random per-node suffix (31-byte V4L2 limit).
fn valid_label(label: &str) -> Result<String> {
    let label = label.trim();
    ensure!(
        (1..=LABEL_MAX).contains(&label.len())
            && label.bytes().all(|b| b == b' ' || b.is_ascii_graphic()),
        "the camera name must be 1-{LABEL_MAX} printable ASCII characters"
    );
    Ok(label.to_owned())
}

// ---- ownership store -----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Node {
    number: u32,
    label: String,
    boot: String,
}

/// A process identified by PID *and* kernel start time, so a reused PID never matches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Proc {
    pid: u32,
    start: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Producer {
    node: u32,
    token: String,
    owner: Proc,
    ffmpeg: Option<Proc>,
    /// A stop was requested; the owner must not report the exit as a failure.
    stopping: bool,
}

#[derive(Serialize, Deserialize)]
struct Removal {
    nodes: Vec<u32>,
    token: String,
    owner: Proc,
    boot: String,
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    nodes: Vec<Node>,
    producers: Vec<Producer>,
    #[serde(default)]
    removals: Vec<Removal>,
}

#[derive(Clone, Debug)]
struct Store(PathBuf);

impl Store {
    fn default_store() -> Result<Self> {
        let base = match std::env::var_os("XDG_STATE_HOME").map(PathBuf::from) {
            Some(p) if p.is_absolute() => p,
            _ => PathBuf::from(
                std::env::var_os("HOME")
                    .filter(|h| !h.is_empty())
                    .context("HOME is not set")?,
            )
            .join(".local/state"),
        };
        Ok(Self(base.join("link-camera/virtual-camera/state.json")))
    }

    /// Run `change` on the pruned state under an exclusive lock; write back only if it changed.
    fn update<T>(&self, change: impl FnOnce(&mut State) -> T) -> Result<T> {
        let dir = self.0.parent().context("state path has no directory")?;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .with_context(|| format!("cannot create {}", dir.display()))?;
        let lock = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(self.0.with_extension("lock"))
            .context("cannot open the virtual camera lock")?;
        lock.lock()
            .context("cannot lock the virtual camera state")?;
        let old = match fs::read(&self.0) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error).context("cannot read the virtual camera state"),
        };
        let mut state: State = if old.is_empty() {
            State::default()
        } else {
            serde_json::from_slice(&old)
                .with_context(|| format!("{} is corrupt; fix or delete it", self.0.display()))?
        };
        let boot = boot_id();
        state.nodes.retain(|n| n.boot == boot);
        state.producers.retain(|p| {
            alive(&p.owner, None) || p.ffmpeg.as_ref().is_some_and(|f| alive(f, Some(&p.token)))
        });
        state
            .removals
            .retain(|r| r.boot == boot && alive(&r.owner, None));
        let result = change(&mut state);
        let new = serde_json::to_vec_pretty(&state)?;
        if new != old {
            let temp = self.0.with_extension("tmp");
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .mode(0o600)
                .open(&temp)?;
            std::io::Write::write_all(&mut file, &new)?;
            fs::rename(&temp, &self.0)?;
        }
        Ok(result)
    }
}

/// Held across producer shutdown, authorization and deletion; no new producer may claim its nodes.
struct RemovalGuard {
    store: Store,
    token: String,
}

impl RemovalGuard {
    fn acquire(store: &Store, wanted: Option<&[u32]>) -> Result<Self> {
        let token = random_token()?;
        let pid = std::process::id();
        let owner = Proc {
            pid,
            start: proc_stat(pid)
                .context("cannot read own process start time")?
                .0,
        };
        store.update(|state| {
            let nodes = wanted.map_or_else(
                || {
                    state
                        .nodes
                        .iter()
                        .map(|node| node.number)
                        .collect::<Vec<_>>()
                },
                <[u32]>::to_vec,
            );
            ensure!(
                !state
                    .removals
                    .iter()
                    .any(|other| other.nodes.iter().any(|node| nodes.contains(node))),
                "another Link Camera removal is already in progress for these nodes"
            );
            state.removals.push(Removal {
                nodes,
                token: token.clone(),
                owner,
                boot: boot_id(),
            });
            Ok(())
        })??;
        Ok(Self {
            store: store.clone(),
            token,
        })
    }
}

impl Drop for RemovalGuard {
    fn drop(&mut self) {
        let _ = self
            .store
            .update(|state| state.removals.retain(|r| r.token != self.token));
    }
}

/// One producer slot per node, held for the lifetime of the FFmpeg writer and released on drop.
pub(crate) struct Claim {
    store: Store,
    token: String,
}

impl Claim {
    /// Reserve `node` before FFmpeg is spawned; fails if another live producer already feeds it.
    pub(crate) fn acquire(node: &Path) -> Result<Self> {
        Self::acquire_in(Store::default_store()?, node)
    }

    fn acquire_in(store: Store, node: &Path) -> Result<Self> {
        let number = node_number(node)?;
        let token = random_token()?;
        let pid = std::process::id();
        let owner = Proc {
            pid,
            start: proc_stat(pid)
                .context("cannot read own process start time")?
                .0,
        };
        let producer = Producer {
            node: number,
            token: token.clone(),
            owner,
            ffmpeg: None,
            stopping: false,
        };
        store.update(|state| {
            ensure!(
                !state.removals.iter().any(|r| r.nodes.contains(&number)),
                "{} is being removed by Link Camera; wait for removal to finish",
                node.display()
            );
            match state.producers.iter().find(|p| p.node == number) {
                Some(other) => Err(anyhow!(
                    "{} is already fed by another Link Camera output ({}); stop it first",
                    node.display(),
                    other
                        .ffmpeg
                        .as_ref()
                        .map_or("starting".to_owned(), |f| format!("pid {}", f.pid))
                )),
                None => {
                    state.producers.push(producer);
                    Ok(())
                }
            }
        })??;
        Ok(Self { store, token })
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Record the spawned FFmpeg (spawned with `PRODUCER_ENV=token`) so any instance can stop it.
    pub(crate) fn attach(&self, child: &Child) -> Result<()> {
        let start = proc_stat(child.id())
            .context("cannot read the FFmpeg start time")?
            .0;
        let proc = Proc {
            pid: child.id(),
            start,
        };
        self.store.update(|state| {
            let producer = state
                .producers
                .iter_mut()
                .find(|p| p.token == self.token)
                .ok_or_else(|| anyhow!("the output was stopped before it started"))?;
            // A stop requested while FFmpeg was being spawned wins; the owner then kills its child
            // and only afterwards releases this reservation.
            ensure!(
                !producer.stopping,
                "the output was stopped before it started"
            );
            producer.ffmpeg = Some(proc);
            Ok(())
        })?
    }

    /// True when a stop was requested through `stop`/`stop_all` (so an FFmpeg exit is intentional).
    pub(crate) fn stop_requested(&self) -> bool {
        self.store
            .update(|state| {
                state
                    .producers
                    .iter()
                    .find(|p| p.token == self.token)
                    .is_none_or(|p| p.stopping)
            })
            .unwrap_or(false)
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        let _ = self
            .store
            .update(|state| state.producers.retain(|p| p.token != self.token));
    }
}

// ---- processes -----------------------------------------------------------------------------

fn boot_id() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

fn random_token() -> Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn is_root() -> bool {
    fs::metadata("/proc/self").is_ok_and(|m| m.uid() == 0)
}

/// `(start time in clock ticks since boot, state letter)` from `/proc/<pid>/stat`.
fn proc_stat(pid: u32) -> Option<(u64, char)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name may contain spaces and parentheses; fields start after the last ')'.
    let mut fields = stat[stat.rfind(')')? + 2..].split(' ');
    let state = fields.next()?.chars().next()?;
    Some((fields.nth(18)?.parse().ok()?, state))
}

/// Same process (PID + start time, plus the private token when given) and not a zombie.
fn alive(proc: &Proc, token: Option<&str>) -> bool {
    let Some((start, state)) = proc_stat(proc.pid) else {
        return false;
    };
    if start != proc.start || matches!(state, 'Z' | 'X') {
        return false;
    }
    token.is_none_or(|token| {
        let wanted = format!("{PRODUCER_ENV}={token}");
        fs::read(format!("/proc/{}/environ", proc.pid))
            .is_ok_and(|env| env.split(|b| *b == 0).any(|kv| kv == wanted.as_bytes()))
    })
}

const ESRCH: i32 = 3;

fn pidfd_open(pid: u32) -> std::io::Result<OwnedFd> {
    // SAFETY: plain syscall with integer arguments; a non-negative result is a new fd we own.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as c_int, 0 as c_uint) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd` was just returned by the kernel and nothing else owns it.
    Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

fn pidfd_signal(fd: &OwnedFd, signal: c_int) -> std::io::Result<()> {
    // SAFETY: the pidfd is valid; a null siginfo and zero flags are the documented defaults.
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd.as_raw_fd(),
            signal,
            std::ptr::null::<c_void>(),
            0 as c_uint,
        )
    };
    match std::io::Error::last_os_error() {
        _ if result == 0 => Ok(()),
        error if error.raw_os_error() == Some(ESRCH) => Ok(()),
        error => Err(error),
    }
}

/// Stop the recorded FFmpeg: SIGTERM, then SIGKILL after 3 s. `Ok(false)` when it is already gone or
/// the PID now belongs to something else (never signalled). The pidfd is opened before the identity
/// check, so the PID cannot be recycled between check and signal.
fn terminate(proc: &Proc, token: &str) -> Result<bool> {
    let fd = match pidfd_open(proc.pid) {
        Ok(fd) => fd,
        Err(error) if error.raw_os_error() == Some(ESRCH) => return Ok(false),
        Err(error) => return Err(error).context("cannot pin the FFmpeg process"),
    };
    if !alive(proc, Some(token)) {
        return Ok(false);
    }
    for (signal, grace) in [(15, Duration::from_secs(3)), (9, Duration::from_secs(2))] {
        pidfd_signal(&fd, signal).context("cannot signal FFmpeg")?;
        if wait_for(|| !alive(proc, Some(token)), grace) {
            return Ok(true);
        }
    }
    bail!("FFmpeg (pid {}) did not exit after SIGKILL", proc.pid)
}

fn wait_for(mut done: impl FnMut() -> bool, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if done() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

// ---- helper programs -----------------------------------------------------------------------

/// A program found on the user's `PATH` (only for FFmpeg, which is never elevated).
fn find_program(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| {
            p.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        })
}

/// A helper that may run as root: looked up only in system directories, never `PATH`, and accepted
/// only if the resolved file and its directory are root-owned and not group/world-writable.
fn trusted_program(name: &str) -> Option<PathBuf> {
    let root_only = |path: &Path| {
        path.metadata()
            .is_ok_and(|m| m.uid() == 0 && m.permissions().mode() & 0o022 == 0)
    };
    [
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/usr/local/sbin",
        "/usr/local/bin",
    ]
    .iter()
    .filter_map(|dir| Path::new(dir).join(name).canonicalize().ok())
    .find(|file| {
        file.is_file()
                && file.metadata().is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
                // The canonical path has no symlinks, so every ancestor must be root-only too:
                // a user-writable directory anywhere above would let the file be swapped.
                && file.ancestors().all(root_only)
    })
}

fn control_writable() -> bool {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(CONTROL_NODE)
        .is_ok()
}

struct Failure {
    code: Option<i32>,
    text: String,
}

fn run(mut command: Command, timeout: Duration) -> std::result::Result<String, Failure> {
    let fail = |text: String| Failure { code: None, text };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|e| fail(format!("cannot start: {e}")))?;
    let deadline = Instant::now() + timeout;
    let stdout =
        crate::process::drain_output(child.stdout.take().expect("stdout was piped"), 64 * 1024);
    let stderr =
        crate::process::drain_output(child.stderr.take().expect("stderr was piped"), 64 * 1024);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(25)),
            Ok(None) => {
                let _ = child.kill();
                // Never block on a helper stuck in the kernel; a reaper thread collects it later.
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Err(fail(format!("timed out after {} s", timeout.as_secs())));
            }
            Err(error) => return Err(fail(error.to_string())),
        }
    };
    let collect = |receiver: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>| {
        // Descendants can inherit the pipes; EOF must obey the same deadline as the helper.
        let bytes = receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|error| match error {
                std::sync::mpsc::RecvTimeoutError::Timeout => {
                    fail(format!("timed out after {} s", timeout.as_secs()))
                }
                std::sync::mpsc::RecvTimeoutError::Disconnected => {
                    fail("helper output reader stopped".into())
                }
            })?
            .map_err(|error| fail(format!("cannot read helper output: {error}")))?;
        Ok::<_, Failure>(String::from_utf8_lossy(&bytes).trim().to_owned())
    };
    let stdout = collect(stdout)?;
    let stderr = collect(stderr)?;
    if status.success() {
        Ok(stdout)
    } else {
        Err(Failure {
            code: status.code(),
            text: if stderr.is_empty() { stdout } else { stderr },
        })
    }
}

/// Run a helper that needs root. Direct when already allowed (`accessible`/root), else `pkexec` if
/// the caller opted in, else an error carrying the exact command to run by hand.
fn privileged(
    program: &Path,
    args: &[&str],
    elevation: Elevation,
    accessible: bool,
    what: &str,
) -> Result<String> {
    let mut direct = Command::new(program);
    direct.args(args);
    if accessible || is_root() {
        return run(direct, Duration::from_secs(20))
            .map_err(|f| anyhow!("{what} failed: {}", f.text));
    }
    ensure!(
        elevation == Elevation::Pkexec,
        "{what} needs administrator rights. Allow elevation (pkexec) or run: sudo {}",
        shell_words(program, args)
    );
    let pkexec = trusted_program("pkexec").ok_or_else(|| {
        anyhow!(
            "{what} needs administrator rights but pkexec (polkit) is missing; run: sudo {}",
            shell_words(program, args)
        )
    })?;
    let mut elevated = Command::new(pkexec);
    elevated.arg(program).args(args);
    run(elevated, Duration::from_secs(120)).map_err(|f| match f.code {
        Some(126) => anyhow!("{what}: administrator authorization was dismissed or denied"),
        Some(127) => anyhow!(
            "{what}: pkexec could not authenticate (needs a polkit agent in this session; otherwise run: sudo {}): {}",
            shell_words(program, args),
            f.text
        ),
        _ => anyhow!("{what} failed: {}", f.text),
    })
}

/// Recheck each label inside the privileged step, after authorization. The driver deletes by number,
/// so concurrent administrative changes can still race between this check and the ioctl.
/// Arguments: `v4l2loopback-ctl`, sysfs class directory, then number/label pairs.
const DELETE_OWNED: &str = concat!(
    r#"ctl=$1 sys=$2; shift 2; rc=0; while [ $# -gt 1 ]; do "#,
    r#"if [ "$(cat "$sys/video$1/name")" = "$2" ]; then "$ctl" delete "/dev/video$1" || rc=1; "#,
    r#"else echo "/dev/video$1 is no longer the camera Link Camera created; left alone" >&2; rc=1; fi; "#,
    r#"shift 2; done; exit $rc"#,
);

/// Delete owned `(number, label)` nodes through [`DELETE_OWNED`], elevated like other root steps.
fn delete_owned(nodes: &[(u32, &str)], elevation: Elevation, accessible: bool) -> Result<String> {
    let ctl = trusted_program("v4l2loopback-ctl").ok_or_else(|| {
        anyhow!("v4l2loopback-ctl is not installed in a root-owned system directory (package v4l2loopback-utils)")
    })?;
    let sh = trusted_program("sh")
        .ok_or_else(|| anyhow!("sh is not installed in a root-owned system directory"))?;
    let args = delete_owned_args(&ctl, Path::new(SYSFS_VIRTUAL), nodes);
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    privileged(&sh, &args, elevation, accessible, "v4l2loopback-ctl delete")
}

fn delete_owned_args(ctl: &Path, sys: &Path, nodes: &[(u32, &str)]) -> Vec<String> {
    let mut args: Vec<String> = ["-c", DELETE_OWNED, "link-camera"].map(String::from).into();
    args.extend([ctl.display().to_string(), sys.display().to_string()]);
    for (number, label) in nodes {
        args.extend([number.to_string(), (*label).to_owned()]);
    }
    args
}

/// For messages only: a command line the user can paste; never executed through a shell.
fn shell_words(program: &Path, args: &[&str]) -> String {
    std::iter::once(program.to_string_lossy().into_owned())
        .chain(args.iter().map(|a| (*a).to_owned()))
        .map(|word| {
            if !word.is_empty()
                && word
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_./=:@".contains(&b))
            {
                word
            } else {
                format!("'{}'", word.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_drains_both_pipes_past_the_retained_limit() {
        for code in [0, 37] {
            let mut command = Command::new("sh");
            command.args([
                "-c",
                r#"out=0123456789abcdef; err=FEDCBA9876543210; i=0
while [ "$i" -lt 8 ]; do out=$out$out; err=$err$err; i=$((i + 1)); done
i=0
while [ "$i" -lt 64 ]; do
    printf '%s' "$out"
    printf '%s' "$err" >&2
    i=$((i + 1))
done
exit "$1""#,
                "link-camera-helper-test",
                &code.to_string(),
            ]);
            let timeout = Duration::from_secs(10);
            let started = Instant::now();
            let result = run(command, timeout);
            assert!(started.elapsed() < timeout, "helper exceeded its deadline");
            match (code, result) {
                (0, Ok(stdout)) => assert_eq!(stdout, "0123456789abcdef".repeat(4096)),
                (37, Err(failure)) => {
                    assert_eq!(failure.code, Some(37));
                    assert_eq!(failure.text, "FEDCBA9876543210".repeat(4096));
                }
                (_, Ok(_)) => panic!("failing helper reported success"),
                (_, Err(failure)) => panic!("successful helper failed: {}", failure.text),
            }
        }
    }

    fn sleeper(token: Option<&str>) -> Child {
        let mut command = Command::new("sleep");
        command.arg("60");
        if let Some(token) = token {
            command.env(PRODUCER_ENV, token);
        }
        command.spawn().unwrap()
    }

    fn temp_store() -> (Store, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "link-camera-vcam-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        (Store(dir.join("state.json")), dir)
    }

    #[test]
    fn stop_all_ends_only_recorded_producers_and_frees_the_node() {
        let (store, dir) = temp_store();
        let node = Path::new("/dev/video97");

        let claim = Claim::acquire_in(store.clone(), node).unwrap();
        assert!(
            Claim::acquire_in(store.clone(), node).is_err(),
            "one producer per node"
        );
        let mut producer = sleeper(Some(claim.token()));
        claim.attach(&producer).unwrap();

        // A live process whose PID is recorded but whose token differs stands for a reused PID.
        let mut stranger = sleeper(None);
        let stranger_start = proc_stat(stranger.id()).unwrap().0;
        store
            .update(|state| {
                state.producers.push(Producer {
                    node: 98,
                    token: "not-its-token".into(),
                    owner: Proc {
                        pid: std::process::id(),
                        start: proc_stat(std::process::id()).unwrap().0,
                    },
                    ffmpeg: Some(Proc {
                        pid: stranger.id(),
                        start: stranger_start,
                    }),
                    stopping: false,
                });
            })
            .unwrap();

        let mut report = Report::default();
        stop_in(&store, None, &mut report);

        assert_eq!(report.stopped, vec![node.to_path_buf()]);
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert!(
            !producer.wait().unwrap().success(),
            "producer was signalled"
        );
        assert!(
            stranger.try_wait().unwrap().is_none(),
            "unrelated process survived"
        );
        assert!(
            claim.stop_requested(),
            "owner can tell the exit was requested"
        );
        assert!(
            Claim::acquire_in(store.clone(), node).is_ok(),
            "node is free again"
        );

        let _ = stranger.kill();
        let _ = stranger.wait();
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn stop_all_cancels_a_start_in_progress_and_keeps_the_node_reserved() {
        let (store, dir) = temp_store();
        let node = Path::new("/dev/video96");
        let claim = Claim::acquire_in(store.clone(), node).unwrap();

        // FFmpeg is not recorded yet when stop-all runs.
        let mut report = Report::default();
        stop_in(&store, None, &mut report);
        assert_eq!(report.stopped, vec![node.to_path_buf()]);
        assert!(claim.stop_requested());

        // The late FFmpeg must not become a writer, and the node stays claimed until released.
        let mut late = sleeper(Some(claim.token()));
        assert!(
            claim.attach(&late).is_err(),
            "cancelled start must not attach"
        );
        assert!(
            Claim::acquire_in(store.clone(), node).is_err(),
            "still reserved"
        );
        let _ = late.kill();
        let _ = late.wait();
        drop(claim);
        assert!(
            Claim::acquire_in(store.clone(), node).is_ok(),
            "released after cleanup"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn removal_keeps_nodes_reserved_after_the_old_producer_releases_them() {
        let (store, dir) = temp_store();
        let node = Path::new("/dev/video96");
        let producer = Claim::acquire_in(store.clone(), node).unwrap();
        let removal = RemovalGuard::acquire(&store, Some(&[96])).unwrap();
        let mut report = Report::default();
        stop_in(&store, Some(&[96]), &mut report);
        assert!(producer.stop_requested());
        drop(producer);

        assert!(Claim::acquire_in(store.clone(), node).is_err());
        assert!(RemovalGuard::acquire(&store, Some(&[96])).is_err());
        let unrelated = Claim::acquire_in(store.clone(), Path::new("/dev/video95")).unwrap();
        stop_in(&store, None, &mut report);
        assert!(Claim::acquire_in(store.clone(), node).is_err());
        drop(removal);
        let replacement = Claim::acquire_in(store.clone(), node).unwrap();
        drop(replacement);
        drop(unrelated);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_dead_removal_owner_cannot_leave_a_permanent_reservation() {
        let (store, dir) = temp_store();
        let removal = RemovalGuard::acquire(&store, Some(&[96])).unwrap();
        store
            .update(|state| state.removals[0].owner.start += 1)
            .unwrap();
        let replacement = Claim::acquire_in(store.clone(), Path::new("/dev/video96")).unwrap();
        drop(replacement);
        drop(removal);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn privileged_delete_rechecks_each_label_itself() {
        let (_, dir) = temp_store();
        let sys = dir.join("sys");
        for (number, label) in [(5, "Link Camera #aaaaaa"), (6, "OBS Virtual Camera")] {
            fs::create_dir_all(sys.join(format!("video{number}"))).unwrap();
            fs::write(
                sys.join(format!("video{number}/name")),
                format!("{label}\n"),
            )
            .unwrap();
        }
        // Since ownership was checked, 6 went to another application's device and 7 disappeared.
        let nodes = [
            (5, "Link Camera #aaaaaa"),
            (6, "Link Camera #bbbbbb"),
            (7, "Link Camera #cccccc"),
        ];
        // `echo` stands in for v4l2loopback-ctl and prints each delete it is asked for.
        let args = delete_owned_args(Path::new("/bin/echo"), &sys, &nodes);
        let out = Command::new("sh").args(&args).output().unwrap();

        assert_eq!(String::from_utf8_lossy(&out.stdout), "delete /dev/video5\n");
        assert!(!out.status.success(), "skipped nodes are reported");
        let errors = String::from_utf8_lossy(&out.stderr);
        assert!(
            errors.contains("/dev/video6 is no longer")
                && errors.contains("/dev/video7 is no longer"),
            "{errors}"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
