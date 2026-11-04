//! Camera microphone status and volume through the PulseAudio/PipeWire `pactl` tool.
//!
//! Everything comes from the audio server's own source list (`pactl --format=json list
//! sources`): no capture stream is opened, the system default source is never changed and a
//! monitor source is never offered as a microphone. The camera's source is found by the USB
//! device path the kernel reports for it; when that is impossible the result says so instead of
//! picking a "close" source. Waiting for `pactl` and its output is bounded by [`TIMEOUT`].
//! Timed-out processes are asked to stop; writes with unknown outcomes are never retried.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::{self, UsbDevice};

const TIMEOUT: Duration = Duration::from_secs(5);
const MAX_OUTPUT: usize = 4 << 20;
/// Readback tolerance for volume: hardware gain steps are coarser than one percent.
const VOLUME_SLACK: u32 = 5;

/// One capture source of the audio server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// Server index. Changes when the device is re-plugged; do not store it.
    pub id: u32,
    /// Stable name: pass it to the setters, and to recording as `pulse:<name>`.
    pub name: String,
    pub description: String,
    /// `None` when the server does not report it.
    pub muted: Option<bool>,
    /// Loudest channel, in percent of 100 % (the server can report more).
    pub volume_percent: Option<u32>,
    /// Server-reported activity: `RUNNING` (something is capturing), `IDLE` or `SUSPENDED`.
    /// `None` when the server reports none. Not a level meter.
    #[serde(default)]
    pub state: Option<String>,
}

/// How the camera's microphone was looked up.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum CameraMic {
    Found {
        source: Source,
        matched_by: String,
    },
    /// The server lists no capture source for this camera; `reason` says what was checked.
    NotFound {
        reason: String,
    },
    /// Several sources could be the camera's; none is chosen.
    Ambiguous {
        sources: Vec<Source>,
    },
}

/// A listed source plus the identity properties used for matching.
struct Listed {
    source: Source,
    sysfs: Option<String>,
    vendor: Option<u16>,
    product: Option<u16>,
}

/// All capture sources (monitors excluded). An error means the audio service is unavailable.
pub fn sources() -> Result<Vec<Source>> {
    Ok(list()?.into_iter().map(|l| l.source).collect())
}

/// One capture source by its stable name, freshly read.
pub fn source(name: &str) -> Result<Source> {
    list()?
        .into_iter()
        .map(|l| l.source)
        .find(|s| s.name == name)
        .ok_or_else(|| {
            anyhow!("`{name}` is not a capture source of the audio server (unplugged or renamed?)")
        })
}

/// The microphone of the camera behind `video_node` (`/dev/videoN`).
pub fn camera_mic(video_node: &str) -> Result<CameraMic> {
    let device = backend::usb_device(video_node)?;
    Ok(match_camera(&device, list()?))
}

pub fn set_muted(name: &str, muted: bool) -> Result<Source> {
    check_name(name)?;
    source(name)?;
    write(&["set-source-mute", name, if muted { "1" } else { "0" }])?;
    let after = source(name)?;
    if after.muted != Some(muted) {
        bail!(
            "`{}` reports muted={:?} after asking for muted={muted}; the change did not take effect",
            after.name,
            after.muted
        );
    }
    Ok(after)
}

/// `percent` is 0..=100; amplifying above 100 % only adds noise.
pub fn set_volume(name: &str, percent: u32) -> Result<Source> {
    check_name(name)?;
    if percent > 100 {
        bail!("volume must be 0-100 %, got {percent}");
    }
    source(name)?;
    write(&["set-source-volume", name, &format!("{percent}%")])?;
    let after = source(name)?;
    match after.volume_percent {
        Some(v) if v.abs_diff(percent) <= VOLUME_SLACK => Ok(after),
        other => bail!(
            "`{}` reports volume {other:?} % after asking for {percent} %; the change did not take effect",
            after.name
        ),
    }
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 256
        || name.starts_with('-')
        || name.chars().any(char::is_control)
    {
        bail!("invalid audio source name `{}`", name.escape_debug());
    }
    Ok(())
}

fn list() -> Result<Vec<Listed>> {
    let text = run("pactl", &["--format=json", "list", "sources"], TIMEOUT)
        .context("reading microphones (needs pactl 16+ with PulseAudio or PipeWire)")?;
    parse_sources(&text)
}

/// A setter's failure to answer in time is an unknown outcome, not a failure to apply.
fn write(args: &[&str]) -> Result<()> {
    run("pactl", args, TIMEOUT).map(drop).with_context(|| {
        format!(
            "pactl {} (outcome unknown if it timed out; not retried)",
            args[0]
        )
    })
}

/// Returns successful stdout, bounding process exit and output collection by one deadline.
fn run(program: &str, args: &[&str], timeout: Duration) -> Result<String> {
    let mut child = Command::new(program)
        .args(args)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => anyhow!("`{program}` is not installed"),
            _ => anyhow!("cannot run `{program}`: {e}"),
        })?;
    let deadline = Instant::now() + timeout;
    let out = crate::process::drain_output(
        child.stdout.take().expect("stdout was piped"),
        MAX_OUTPUT + 1,
    );
    let err = crate::process::drain_output(
        child.stderr.take().expect("stderr was piped"),
        (64 << 10) + 1,
    );
    let status = loop {
        match child.try_wait().context("waiting for the process")? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                bail!(
                    "`{program}` did not answer within {} s; cancellation requested",
                    timeout.as_secs_f32()
                );
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let collect = |receiver: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>| {
        receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .with_context(|| {
                format!(
                    "waiting for `{program}` output ({} s deadline)",
                    timeout.as_secs_f32()
                )
            })?
            .with_context(|| format!("reading `{program}` output"))
    };
    let (out, err) = (collect(out)?, collect(err)?);
    if !status.success() {
        bail!(
            "`{program}` failed ({status}): {}",
            String::from_utf8_lossy(&err).trim()
        );
    }
    if out.len() > MAX_OUTPUT {
        bail!("`{program}` printed more than {MAX_OUTPUT} bytes");
    }
    String::from_utf8(out).map_err(|_| anyhow!("`{program}` printed invalid UTF-8"))
}

fn percent(text: &str) -> Option<u32> {
    text.trim().strip_suffix('%')?.trim().parse().ok()
}

fn hex_id(text: Option<&str>) -> Option<u16> {
    u16::from_str_radix(text?.trim().trim_start_matches("0x"), 16).ok()
}

fn parse_sources(text: &str) -> Result<Vec<Listed>> {
    let Value::Array(items) = serde_json::from_str(text).context("parsing the source list")? else {
        bail!("the source list is not an array");
    };
    let mut out = Vec::new();
    for item in items {
        let text = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_owned);
        let (Some(name), Some(id)) = (
            text(&item, "name"),
            item.get("index")
                .and_then(Value::as_u64)
                .and_then(|i| u32::try_from(i).ok()),
        ) else {
            bail!("a listed source has no name or index");
        };
        let prop = |key: &str| {
            item.get("properties")
                .and_then(|p| p.get(key))
                .and_then(Value::as_str)
        };
        let monitor = name.ends_with(".monitor")
            || prop("device.class") == Some("monitor")
            || item
                .get("monitor_source")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty());
        if monitor {
            continue;
        }
        let volume = item
            .get("volume")
            .and_then(Value::as_object)
            .and_then(|channels| {
                channels
                    .values()
                    .filter_map(|c| percent(c.get("value_percent")?.as_str()?))
                    .max()
            });
        out.push(Listed {
            sysfs: prop("sysfs.path").map(str::to_owned),
            vendor: hex_id(prop("device.vendor.id")),
            product: hex_id(prop("device.product.id")),
            source: Source {
                id,
                description: text(&item, "description").unwrap_or_else(|| name.clone()),
                name,
                muted: item.get("mute").and_then(Value::as_bool),
                state: text(&item, "state").filter(|s| !s.is_empty()),
                volume_percent: volume,
            },
        });
    }
    Ok(out)
}

fn match_camera(device: &UsbDevice, listed: Vec<Listed>) -> CameraMic {
    // Every interface of the camera sits below its USB device in sysfs, so the microphone's
    // card path starts with the camera's device path. Exact, and tells identical cameras apart.
    let below = format!("{}/", device.sysfs);
    let by_path: Vec<&Listed> = listed
        .iter()
        .filter(|l| l.sysfs.as_deref().is_some_and(|p| p.starts_with(&below)))
        .collect();
    match by_path.as_slice() {
        [one] => {
            return CameraMic::Found {
                source: one.source.clone(),
                matched_by: "USB device path".to_owned(),
            };
        }
        [] => {}
        many => {
            return CameraMic::Ambiguous {
                sources: many.iter().map(|l| l.source.clone()).collect(),
            };
        }
    }
    // No source is provably this camera's. Same-model sources the server reports no path for
    // might be (vendor/product ids are model identity, not device identity): offer them for an
    // explicit choice. A source with a path elsewhere is another device and is not offered.
    let unverifiable: Vec<Source> = listed
        .iter()
        .filter(|l| {
            l.sysfs.is_none()
                && l.vendor == Some(device.vendor)
                && l.product == Some(device.product)
        })
        .map(|l| l.source.clone())
        .collect();
    if unverifiable.is_empty() {
        CameraMic::NotFound {
            reason: format!(
                "the audio server lists no capture source on USB device {:04x}:{:04x} ({}); the camera microphone may be unavailable to it",
                device.vendor, device.product, device.sysfs
            ),
        }
    } else {
        CameraMic::Ambiguous {
            sources: unverifiable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMERA_PATH: &str = "/devices/pci0000:00/0000:11:00.4/usb5/5-2/5-2.4/5-2.4.1";

    fn entry(index: u32, name: &str, extra: &str) -> String {
        format!(
            r#"{{"index":{index},"name":"{name}","description":"d {index}","state":"IDLE","mute":false,
            "volume":{{"front-left":{{"value_percent":"30%"}},"front-right":{{"value_percent":"45%"}}}},
            "monitor_source":"","properties":{{{extra}}}}}"#
        )
    }

    fn device() -> UsbDevice {
        UsbDevice {
            sysfs: CAMERA_PATH.into(),
            vendor: 0x2e1a,
            product: 0x4c03,
        }
    }

    #[test]
    fn parses_state_and_never_lists_monitors() {
        let text = format!(
            "[{},{}]",
            entry(1, "alsa_output.x.monitor", ""),
            entry(2, "alsa_input.mic", "")
        );
        let listed = parse_sources(&text).unwrap();
        assert_eq!(listed.len(), 1);
        let s = &listed[0].source;
        assert_eq!(
            (s.id, s.muted, s.volume_percent),
            (2, Some(false), Some(45))
        );
        assert_eq!(s.state.as_deref(), Some("IDLE"));
        let by_flag =
            r#"[{"index":3,"name":"weird","monitor_source":"alsa_output.x","properties":{}}]"#;
        assert!(parse_sources(by_flag).unwrap().is_empty());
    }

    #[test]
    fn matches_only_the_cameras_own_usb_path() {
        let camera = entry(
            7,
            "alsa_input.usb-Insta360",
            &format!(r#""sysfs.path":"{CAMERA_PATH}/5-2.4.1:1.2/sound/card5""#),
        );
        // Same prefix text, different device: 5-2.4.10 is not below 5-2.4.1.
        let sibling = entry(
            8,
            "alsa_input.sibling",
            r#""sysfs.path":"/devices/pci0000:00/0000:11:00.4/usb5/5-2/5-2.4/5-2.4.10/5-2.4.10:1.2/sound/card6""#,
        );
        let listed = parse_sources(&format!("[{sibling},{camera}]")).unwrap();
        match match_camera(&device(), listed) {
            CameraMic::Found { source, matched_by } => {
                assert_eq!(source.name, "alsa_input.usb-Insta360");
                assert!(matched_by.contains("path"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn never_substitutes_another_microphone() {
        let other = entry(
            9,
            "alsa_input.other",
            r#""sysfs.path":"/devices/elsewhere/usb1/1-5/1-5:1.0/sound/card0","device.vendor.id":"0x2e1a","device.product.id":"0x4c03""#,
        );
        let listed = parse_sources(&format!("[{other}]")).unwrap();
        assert!(matches!(
            match_camera(&device(), listed),
            CameraMic::NotFound { .. }
        ));
        // Without a device path a same-model source cannot be proven to be this camera, even
        // when it is the only one: it is offered for an explicit choice, never as Found.
        let ids = r#""device.vendor.id":"0x2e1a","device.product.id":"0x4c03""#;
        let one = parse_sources(&format!("[{}]", entry(1, "a", ids))).unwrap();
        assert!(
            matches!(match_camera(&device(), one), CameraMic::Ambiguous { sources } if sources.len() == 1)
        );
        let two =
            parse_sources(&format!("[{},{}]", entry(1, "a", ids), entry(2, "b", ids))).unwrap();
        assert!(matches!(
            match_camera(&device(), two),
            CameraMic::Ambiguous { .. }
        ));
    }

    #[test]
    fn a_same_model_camera_elsewhere_is_not_this_camera() {
        // Two identical cameras; only the other one exposes a microphone (with a device path).
        let ids = r#""device.vendor.id":"0x2e1a","device.product.id":"0x4c03""#;
        let other = entry(
            5,
            "alsa_input.usb-Insta360_other",
            &format!(
                r#""sysfs.path":"/devices/pci0000:00/0000:11:00.4/usb5/5-2/5-2.3/5-2.3:1.2/sound/card4",{ids}"#
            ),
        );
        let listed = parse_sources(&format!("[{other}]")).unwrap();
        assert!(matches!(
            match_camera(&device(), listed),
            CameraMic::NotFound { .. }
        ));
    }

    #[test]
    fn validates_names_and_volumes_before_any_process_runs() {
        for bad in ["", "-x", "a\nb"] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
        assert!(check_name("alsa_input.usb-Insta360_Link_2C-02.mono-fallback").is_ok());
        assert!(set_volume("alsa_input.x", 101).is_err());
    }

    #[test]
    fn process_exit_and_inherited_output_share_a_deadline() {
        let directory = std::env::temp_dir().join(format!(
            "link-camera-audio-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let pid_file = directory.join("pid");
        let release_file = directory.join("release");
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker_pid_file = pid_file.clone();
        let worker_release_file = release_file.clone();
        let worker = std::thread::spawn(move || {
            let started = Instant::now();
            let result = run(
                "sh",
                &[
                    "-c",
                    r#"printf '%s' "$$" > "$1"; while [ ! -e "$2" ]; do sleep 0.01; done"#,
                    "audio-test",
                    worker_pid_file.to_str().unwrap(),
                    worker_release_file.to_str().unwrap(),
                ],
                Duration::from_secs(2),
            );
            let _ = sender.send((result, started.elapsed()));
        });
        let ready_deadline = Instant::now() + Duration::from_secs(1);
        let pid = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file)
                && let Ok(pid) = text.parse::<u32>()
            {
                break pid;
            }
            assert!(Instant::now() < ready_deadline, "fixture did not start");
            std::thread::sleep(Duration::from_millis(10));
        };
        // Keep the pipe open after the helper exits, as an inherited descendant pipe can be.
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(format!("/proc/{pid}/fd/1"))
            .unwrap();
        std::fs::write(&release_file, b"").unwrap();
        let result = receiver.recv_timeout(Duration::from_secs(3));
        drop(writer);
        worker.join().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        let (result, elapsed) = result.expect("inherited output escaped the deadline");
        assert!(result.is_err());
        assert!(elapsed < Duration::from_secs(3));
    }

    #[test]
    fn stalled_and_failed_processes_return_errors() {
        let start = Instant::now();
        assert!(run("sleep", &["30"], Duration::from_millis(150)).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(run("false", &[], TIMEOUT).is_err());
        assert!(run("link-camera-no-such-tool", &[], TIMEOUT).is_err());
    }
}
