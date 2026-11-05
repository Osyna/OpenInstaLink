use std::io::{BufRead, Read};
use std::process::Child;

use anyhow::{Result, ensure};
use serde::de::DeserializeOwned;

pub(crate) fn read_message<T: DeserializeOwned>(
    reader: &mut impl BufRead,
    max: u64,
) -> Result<Option<T>> {
    let mut bytes = Vec::new();
    let n = reader.take(max + 1).read_until(b'\n', &mut bytes)?;
    if n == 0 {
        return Ok(None);
    }
    ensure!(
        n as u64 <= max && bytes.last() == Some(&b'\n'),
        "worker message is incomplete or exceeds {max} bytes"
    );
    Ok(Some(serde_json::from_slice(&bytes)?))
}

/// The caller must spawn a private process group and keep its leader unreaped until this returns.
pub(crate) fn stop_group(child: &mut Child) -> std::io::Result<()> {
    unsafe extern "C" {
        fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
    }
    // SAFETY: callers spawn a private group and retain its unreaped leader; Linux pid_t is c_int.
    if unsafe { kill(-(child.id() as i32), 9) } == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(3) {
        Ok(())
    } else {
        Err(error)
    }
}

pub(crate) fn drain_output(
    mut pipe: impl Read + Send + 'static,
    limit: usize,
) -> std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 8192];
        let result = loop {
            match pipe.read(&mut buffer) {
                Ok(0) => break Ok(bytes),
                Ok(count) => {
                    let retained = count.min(limit.saturating_sub(bytes.len()));
                    bytes.extend_from_slice(&buffer[..retained]);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => break Err(error),
            }
        };
        let _ = sender.send(result);
    });
    receiver
}
