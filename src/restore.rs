//! Restores device settings without opening a video stream or running inference.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};

use crate::backend::{self, CameraBackend};

const UNIT: &str = "link-camera-restore.service";
const MARKER: &str = "# Managed by Link Camera.\n";
const INTERVAL: Duration = Duration::from_secs(2);
const DEADLINE: Duration = Duration::from_secs(15);
static SERVICE_READY: AtomicBool = AtomicBool::new(false);
static STOP: AtomicBool = AtomicBool::new(false);

fn config_home() -> Result<PathBuf> {
    match std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        Some(path) if path.is_absolute() => Ok(path),
        _ => {
            Ok(PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?).join(".config"))
        }
    }
}

fn unit_arg(value: &str) -> Result<String> {
    ensure!(
        !value.chars().any(char::is_control),
        "service paths cannot contain control characters"
    );
    Ok(format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
    ))
}

fn unit_contents(executable: &Path) -> Result<String> {
    let executable = unit_arg(
        executable
            .to_str()
            .context("application path is not UTF-8")?,
    )?;
    let home = PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?);
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home.join(".local/state"));
    let mut environment = String::new();
    for (key, value) in [
        ("XDG_STATE_HOME", state),
        ("XDG_CONFIG_HOME", config_home()?),
    ] {
        let value = value.to_str().context("XDG directory is not UTF-8")?;
        environment.push_str(&format!(
            "Environment={}\n",
            unit_arg(&format!("{key}={value}"))?
        ));
    }
    Ok(format!(
        "{MARKER}[Unit]\nDescription=Restore saved Link Camera hardware settings\n\n[Service]\nType=simple\n{environment}ExecStart=:{executable} restore watch\nKillMode=control-group\nTimeoutStopSec=5\nNoNewPrivileges=true\n\n[Install]\nWantedBy=default.target\n"
    ))
}

fn systemctl(args: &[&str]) -> Result<()> {
    let output = Command::new("timeout")
        .args([
            "--signal=KILL",
            "5s",
            "systemctl",
            "--user",
            "--no-ask-password",
            "--no-pager",
            "--no-block",
        ])
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("running the user service manager (systemd and coreutils timeout are required)")?;
    ensure!(
        output.status.success(),
        "user service command failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

/// Called after a saved user change, not for discovery, reads or AI preference changes.
pub fn ensure_user_service() -> Result<()> {
    install_user_service(false)
}

pub fn enable_user_service() -> Result<()> {
    install_user_service(true)
}

fn install_user_service(force: bool) -> Result<()> {
    if !force && SERVICE_READY.load(Ordering::Acquire) {
        return Ok(());
    }
    let dir = config_home()?.join("systemd/user");
    let path = dir.join(UNIT);
    let executable = std::env::current_exe()?;
    ensure!(
        executable.is_file(),
        "the running application was replaced; restart it before configuring automatic restore"
    );
    let contents = unit_contents(&executable)?;
    let previous = match fs::read_to_string(&path) {
        Ok(previous) => {
            ensure!(
                previous.starts_with(MARKER),
                "refusing to replace the existing unmanaged service {}",
                path.display()
            );
            Some(previous)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    if !force && previous.is_some() {
        let status = Command::new("timeout")
            .args([
                "--signal=KILL",
                "5s",
                "systemctl",
                "--user",
                "is-enabled",
                UNIT,
            ])
            .stdin(Stdio::null())
            .output()?;
        if !status.status.success() && String::from_utf8_lossy(&status.stdout).trim() == "disabled"
        {
            SERVICE_READY.store(true, Ordering::Release);
            return Ok(());
        }
        ensure!(
            status.status.success(),
            "cannot determine restore-service state: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        );
    }
    if previous.as_deref() != Some(&contents) {
        fs::create_dir_all(&dir)?;
        let temp = dir.join(format!(".{UNIT}.{}.tmp", std::process::id()));
        let result = (|| -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&temp)?;
            file.write_all(contents.as_bytes())?;
            file.sync_all()?;
            fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result.with_context(|| format!("writing {}", path.display()))?;
        systemctl(&["daemon-reload"])?;
        systemctl(&["restart", UNIT])?;
    }
    systemctl(&["enable", "--now", UNIT])?;
    SERVICE_READY.store(true, Ordering::Release);
    Ok(())
}

pub fn disable_user_service() -> Result<()> {
    systemctl(&["disable", "--now", UNIT])?;
    SERVICE_READY.store(false, Ordering::Release);
    Ok(())
}

/// One process per camera. The parent owns its deadline, including a blocked first ioctl.
pub fn monitor(device: &str) -> Result<()> {
    let camera = CameraBackend::open(Some(device))?;
    let mut was_streaming = false;
    let mut first = true;
    loop {
        let streaming = camera.is_streaming()?;
        if first || (streaming && !was_streaming) {
            camera.restore_saved_settings()?;
            first = false;
        }
        serde_json::to_writer(std::io::stdout().lock(), &streaming)?;
        println!();
        std::io::stdout().flush()?;
        was_streaming = streaming;
        thread::sleep(INTERVAL);
    }
}

struct Monitor {
    child: Option<Child>,
    replies: Receiver<std::result::Result<bool, String>>,
    generation: String,
    last_reply: Instant,
    blocked: bool,
}

impl Monitor {
    fn start(device: &str, generation: String) -> Result<Self> {
        let mut child = Command::new("/proc/self/exe")
            .args(["--device", device, "restore", "__monitor"])
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting the settings restore worker")?;
        let stdout = child.stdout.take().expect("piped restore worker stdout");
        let (tx, replies) = mpsc::channel();
        let monitor = Self {
            child: Some(child),
            replies,
            generation,
            last_reply: Instant::now(),
            blocked: false,
        };
        thread::Builder::new()
            .name("restore-replies".into())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                loop {
                    let reply = match crate::process::read_message(&mut reader, 256) {
                        Ok(Some(value)) => Ok(value),
                        Ok(None) => Err(
                            "restore worker exited; reconnect the camera before retrying".into(),
                        ),
                        Err(error) => Err(format!("invalid restore worker response: {error:#}")),
                    };
                    let failed = reply.is_err();
                    if tx.send(reply).is_err() || failed {
                        break;
                    }
                }
            })?;
        Ok(monitor)
    }

    fn stop(&mut self) {
        if self.blocked {
            return;
        }
        self.blocked = true;
        if let Some(child) = self.child.as_mut() {
            if let Err(error) = crate::process::stop_group(child) {
                eprintln!("link-camera restore: cannot stop owned worker: {error}");
            }
        }
    }

    fn check(&mut self, device: &str) {
        if !self.blocked {
            while let Ok(reply) = self.replies.try_recv() {
                match reply {
                    Ok(_) => self.last_reply = Instant::now(),
                    Err(error) => {
                        eprintln!(
                            "link-camera restore: {device}: {error}; no automatic retry for this connection"
                        );
                        self.stop();
                        break;
                    }
                }
            }
            if !self.blocked && self.last_reply.elapsed() > DEADLINE {
                eprintln!(
                    "link-camera restore: {device}: 15-second USB deadline exceeded; pending writes have unknown outcomes. Reconnect the camera before retrying."
                );
                self.stop();
            }
        }
    }

    fn reaped(&mut self) -> Result<bool> {
        if let Some(child) = self.child.as_mut() {
            if child.try_wait()?.is_none() {
                return Ok(false);
            }
            self.child = None;
        }
        Ok(true)
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop();
        if let Some(mut child) = self.child.take() {
            let _ = thread::Builder::new()
                .name("restore-reap".into())
                .spawn(move || {
                    let _ = child.wait();
                });
        }
    }
}

pub fn watch() -> Result<()> {
    STOP.store(false, Ordering::Release);
    ctrlc::set_handler(|| STOP.store(true, Ordering::Release))?;
    let mut monitors: HashMap<String, Monitor> = HashMap::new();
    while !STOP.load(Ordering::Acquire) {
        let mut present = HashMap::new();
        for device in backend::devices()? {
            let result = (|| -> Result<()> {
                let generation = backend::physical_instance(&device.id)?;
                present.insert(device.id.clone(), generation.clone());
                if let Some(monitor) = monitors.get_mut(&device.id) {
                    monitor.check(&device.id);
                    if monitor.generation != generation {
                        monitor.stop();
                    }
                    if monitor.blocked && monitor.reaped()? && monitor.generation != generation {
                        monitors.remove(&device.id);
                    }
                }
                if !monitors.contains_key(&device.id) && backend::has_saved_settings(&device.id)? {
                    monitors.insert(device.id.clone(), Monitor::start(&device.id, generation)?);
                }
                Ok(())
            })();
            if let Err(error) = result {
                eprintln!(
                    "link-camera restore: cannot monitor {}: {error:#}",
                    device.id
                );
            }
        }
        let removed: Vec<_> = monitors
            .keys()
            .filter(|id| !present.contains_key(*id))
            .cloned()
            .collect();
        for id in removed {
            let monitor = monitors.get_mut(&id).expect("known restore worker");
            monitor.stop();
            if monitor.reaped()? {
                monitors.remove(&id);
            }
        }
        thread::sleep(INTERVAL);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_arguments_preserve_literal_paths_without_unit_injection() {
        assert_eq!(
            unit_arg("/a b/100%/$x/\"q\"\\app").unwrap(),
            "\"/a b/100%%/$x/\\\"q\\\"\\\\app\""
        );
        assert!(unit_arg("/app\nExecStart=/bad").is_err());
        assert!(unit_arg("/app\0").is_err());
    }
}
