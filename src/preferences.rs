//! Saved application preferences: `$XDG_CONFIG_HOME/link-camera/settings.json`.
//! Optional AI is off until the user turns it on. A missing file means defaults; a file that is
//! unreadable or malformed is an error, and callers must treat AI as disabled while it is.

use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{effects::Effects, media::VideoConfig};

const MAX_BYTES: u64 = 64 << 10;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Preferences {
    /// Optional local models and neural person blur/replacement. Default off.
    pub ai_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<VideoConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effects: Option<Effects>,
}

pub fn path() -> Result<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        Some(p) if p.is_absolute() => p,
        _ => PathBuf::from(
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .context("HOME is not set")?,
        )
        .join(".config"),
    };
    Ok(base.join("link-camera/settings.json"))
}

pub fn load() -> Result<Preferences> {
    load_from(&path()?)
}

pub fn set_ai_enabled(enabled: bool) -> Result<()> {
    update_at(&path()?, |prefs| {
        let mut changed = prefs.ai_enabled != enabled;
        prefs.ai_enabled = enabled;
        if !enabled {
            if let Some(fx) = prefs
                .effects
                .as_mut()
                .filter(|fx| matches!(fx.background.as_str(), "blur" | "replace"))
            {
                fx.background = "none".into();
                fx.background_image = None;
                changed = true;
            }
        }
        changed
    })
}

pub fn set_capture(config: &VideoConfig) -> Result<()> {
    config.validate()?;
    update_at(&path()?, |prefs| {
        if prefs.capture.as_ref() == Some(config) {
            return false;
        }
        prefs.capture = Some(config.clone());
        true
    })
}

pub fn set_effects(effects: &Effects) -> Result<()> {
    effects.validate()?;
    update_at(&path()?, |prefs| {
        if prefs.effects.as_ref() == Some(effects) {
            return false;
        }
        prefs.effects = Some(effects.clone());
        true
    })
}

fn update_at(path: &Path, change: impl FnOnce(&mut Preferences) -> bool) -> Result<()> {
    let dir = path.parent().context("settings path has no parent")?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    let lock = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(dir.join(".settings.lock"))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(fs::TryLockError::WouldBlock) => {
            bail!("another process is updating the application settings")
        }
        Err(fs::TryLockError::Error(error)) => {
            return Err(error).context("locking application settings");
        }
    }
    let mut prefs = load_from(path)?;
    if change(&mut prefs) {
        save_to(path, &prefs)?;
    }
    Ok(())
}

/// Denies optional-AI work unless the saved setting is on. `what` names the denied action.
pub fn require_ai_enabled(what: &str) -> Result<()> {
    gate(load(), what)
}

fn gate(loaded: Result<Preferences>, what: &str) -> Result<()> {
    if loaded
        .context("cannot read settings, so optional AI stays disabled")?
        .ai_enabled
    {
        return Ok(());
    }
    bail!(
        "{what} needs optional AI, which is disabled. Turn it on in Settings or run `link-camera settings ai true`"
    )
}

pub fn load_from(path: &Path) -> Result<Preferences> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Preferences::default()),
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("cannot read {}", path.display()))?;
    if bytes.len() as u64 > MAX_BYTES {
        bail!("{} is too large to be a settings file", path.display());
    }
    let prefs: Preferences = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is not a valid settings file; fix or delete it",
            path.display()
        )
    })?;
    if let Some(capture) = &prefs.capture {
        capture.validate().context("invalid saved capture format")?;
    }
    if let Some(effects) = &prefs.effects {
        effects.validate().context("invalid saved effects")?;
    }
    Ok(prefs)
}

pub fn save_to(path: &Path, prefs: &Preferences) -> Result<()> {
    let dir = path.parent().context("settings path has no parent")?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .with_context(|| format!("cannot create {}", dir.display()))?;
    let mut json = serde_json::to_vec_pretty(prefs)?;
    json.push(b'\n');
    if json.len() as u64 > MAX_BYTES {
        bail!("application settings exceed 64 KiB; the previous settings were kept");
    }
    // Dot-prefixed temp file replaced atomically by rename; a crash never leaves a torn file.
    let tmp = dir.join(format!(".settings.{}.tmp", std::process::id()));
    let _ = fs::remove_file(&tmp);
    let written = (|| -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&json)?;
        file.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("cannot write {}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("link-camera-prefs-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn missing_file_means_ai_off_and_enabling_persists_privately() {
        let d = dir("persist");
        let file = d.join("nested/settings.json");
        assert!(!load_from(&file).unwrap().ai_enabled);
        save_to(
            &file,
            &Preferences {
                ai_enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(load_from(&file).unwrap().ai_enabled);
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        save_to(
            &file,
            &Preferences {
                ai_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!load_from(&file).unwrap().ai_enabled);
        assert_eq!(fs::read_dir(file.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn corrupt_or_unreadable_settings_are_errors_not_defaults() {
        let d = dir("corrupt");
        fs::create_dir_all(&d).unwrap();
        for (name, body) in [
            ("bad.json", &b"{ai_enabled"[..]),
            ("type.json", &b"{\"ai_enabled\":\"yes\"}"[..]),
            ("unknown.json", &b"{\"ai_enabled\":true,\"x\":1}"[..]),
            ("big.json", &[b' '; (MAX_BYTES + 1) as usize][..]),
        ] {
            let file = d.join(name);
            fs::write(&file, body).unwrap();
            assert!(load_from(&file).is_err(), "{name}");
        }
        // A directory where the file should be is unreadable, not "missing".
        assert!(load_from(&d).is_err());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn gate_allows_only_an_explicit_on_and_fails_closed_on_errors() {
        let d = dir("gate");
        let file = d.join("settings.json");
        let denied = gate(load_from(&file), "x").unwrap_err().to_string();
        assert!(denied.contains("disabled"), "{denied}");
        save_to(
            &file,
            &Preferences {
                ai_enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(gate(load_from(&file), "x").is_ok());
        fs::write(&file, b"nope").unwrap();
        assert!(gate(load_from(&file), "x").is_err());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn updates_preserve_other_setup_and_reject_corrupt_or_locked_files() {
        let d = dir("updates");
        let file = d.join("settings.json");
        let prefs = Preferences {
            capture: Some(VideoConfig {
                width: 1920,
                height: 1080,
                fps: 30,
            }),
            effects: Some(Effects {
                mirror: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        save_to(&file, &prefs).unwrap();
        update_at(&file, |p| {
            p.ai_enabled = true;
            true
        })
        .unwrap();
        let saved = load_from(&file).unwrap();
        assert!(saved.ai_enabled);
        assert_eq!(saved.capture, prefs.capture);
        assert_eq!(saved.effects, prefs.effects);
        let lock = fs::OpenOptions::new()
            .write(true)
            .open(d.join(".settings.lock"))
            .unwrap();
        lock.lock().unwrap();
        assert!(
            update_at(&file, |p| {
                p.ai_enabled = false;
                true
            })
            .is_err()
        );
        drop(lock);
        let previous = fs::read(&file).unwrap();
        let oversized = Preferences {
            effects: Some(Effects {
                overlay_image: Some(PathBuf::from("x".repeat(MAX_BYTES as usize + 1))),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(save_to(&file, &oversized).is_err());
        assert_eq!(fs::read(&file).unwrap(), previous);
        fs::write(&file, b"bad").unwrap();
        assert!(update_at(&file, |_| true).is_err());
        assert_eq!(fs::read(&file).unwrap(), b"bad");
        fs::remove_dir_all(d).unwrap();
    }
}
