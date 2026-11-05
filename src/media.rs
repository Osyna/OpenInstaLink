use crate::effects::{Effects, Processor};
use anyhow::{Context, Result, anyhow, bail, ensure};
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
        }
    }
}

impl VideoConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (64..=3840).contains(&self.width)
                && (64..=3840).contains(&self.height)
                && u64::from(self.width) * u64::from(self.height) <= 3840 * 2160,
            "video dimensions must be even, at least 64x64, and no larger than 4K (landscape or portrait)"
        );
        ensure!(
            self.width % 2 == 0 && self.height % 2 == 0,
            "video dimensions must be even"
        );
        ensure!(
            (1..=60).contains(&self.fps),
            "frame rate must be between 1 and 60"
        );
        Ok(())
    }

    fn size(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

/// MJPEG capture modes reported by the selected camera, not guessed resolution presets.
pub fn formats(device: &str) -> Result<Vec<VideoConfig>> {
    let device = video_node(Path::new(device))?;
    let mut command = Command::new("v4l2-ctl");
    command.arg("-d").arg(device).arg("--list-formats-ext");
    let (status, stdout, stderr) = run_bounded(command, Duration::from_secs(10)).context(
        "camera format query (v4l2-ctl; is v4l-utils installed and the camera responsive?)",
    )?;
    ensure!(
        status.success(),
        "camera format query failed: {}",
        String::from_utf8_lossy(&stderr).trim()
    );
    let formats = parse_formats(&String::from_utf8_lossy(&stdout));
    ensure!(
        !formats.is_empty(),
        "camera did not advertise supported discrete MJPEG modes with integer frame rates"
    );
    Ok(formats)
}

fn parse_formats(text: &str) -> Vec<VideoConfig> {
    let mut formats = Vec::new();
    let mut mjpeg = false;
    let mut size = None;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') && line.contains("]:") {
            mjpeg = line.contains("'MJPG'");
            size = None;
        } else if mjpeg {
            if let Some(dimensions) = line.strip_prefix("Size: Discrete ") {
                size = dimensions
                    .split_once('x')
                    .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
            } else if line.starts_with("Interval: Discrete ") {
                let fps = line
                    .rsplit_once('(')
                    .and_then(|(_, value)| value.strip_suffix(" fps)"))
                    .and_then(|value| value.parse::<f64>().ok());
                if let (Some((width, height)), Some(fps)) = (size, fps) {
                    if fps.is_finite() && fps.fract() == 0.0 && (1.0..=60.0).contains(&fps) {
                        let config = VideoConfig {
                            width,
                            height,
                            fps: fps as u32,
                        };
                        if config.validate().is_ok() && !formats.contains(&config) {
                            formats.push(config);
                        }
                    }
                }
            }
        }
    }
    formats
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub sequence: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MediaStatus {
    pub frames: u64,
    pub error: Option<String>,
    pub recording: bool,
    pub paused: bool,
    pub virtual_device: Option<String>,
}

#[derive(Default)]
struct Shared {
    frame: Option<Arc<Frame>>,
    status: MediaStatus,
}

type Reply = mpsc::Sender<std::result::Result<(), String>>;
enum Request {
    Effects(Effects, Reply),
    Record(PathBuf, Option<String>, Reply),
    Pause(bool, Reply),
    StopRecord(Reply),
    Virtual(Option<PathBuf>, Reply),
    Close,
}

pub struct Preview {
    shared: Arc<Mutex<Shared>>,
    requests: mpsc::Sender<Request>,
    capture: Arc<Mutex<Child>>,
    worker: Option<JoinHandle<()>>,
}

impl Preview {
    pub fn start(device: &str, config: &VideoConfig) -> Result<Self> {
        config.validate()?;
        let source = video_node(Path::new(device))?;
        let size = config.size();
        let (stdout, writer) =
            UnixStream::pair().context("cannot create bounded camera capture socket")?;
        let mut command = ffmpeg();
        command
            .args([
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-framerate",
                &config.fps.to_string(),
                "-video_size",
                &size,
                "-i",
            ])
            .arg(&source)
            .args([
                "-an", "-threads", "2", "-s", &size, "-pix_fmt", "rgb24", "-f", "rawvideo",
                "pipe:1",
            ])
            .stdout(Stdio::from(OwnedFd::from(writer)));
        let mut capture = command
            .spawn()
            .context("cannot start FFmpeg; install ffmpeg and check camera permissions")?;
        let capture_errors = ErrorLog::drain(capture.stderr.take().expect("piped FFmpeg stderr"));
        let capture = Arc::new(Mutex::new(capture));
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (requests, receiver) = mpsc::channel();
        let worker_shared = shared.clone();
        let worker_capture = capture.clone();
        let config = config.clone();
        let worker = thread::Builder::new()
            .name("camera-capture".into())
            .spawn(move || {
                let result = capture_loop(stdout, receiver, &worker_shared, &source, &config);
                kill_bounded(&mut worker_capture.lock().unwrap());
                if let Err(error) = result {
                    let detail = capture_errors.text();
                    worker_shared.lock().unwrap().status.error = Some(if detail.is_empty() {
                        format!("{error:#}")
                    } else {
                        format!("{error:#}: {detail}")
                    });
                }
            });
        match worker {
            Ok(worker) => Ok(Self {
                shared,
                requests,
                capture,
                worker: Some(worker),
            }),
            Err(error) => {
                kill_bounded(&mut capture.lock().unwrap());
                Err(error).context("cannot start camera capture thread")
            }
        }
    }

    pub fn latest_frame(&self) -> Option<Arc<Frame>> {
        self.shared.lock().unwrap().frame.clone()
    }
    pub fn status(&self) -> MediaStatus {
        self.shared.lock().unwrap().status.clone()
    }

    pub fn wait_first_frame(&self, timeout: Duration) -> Result<Arc<Frame>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(error) = self.status().error {
                bail!("{error}");
            }
            if let Some(frame) = self.latest_frame() {
                return Ok(frame);
            }
            ensure!(
                Instant::now() < deadline,
                "camera did not deliver a frame within {} seconds; check the lens cover, selected format and other camera users",
                timeout.as_secs()
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn request(&self, make: impl FnOnce(Reply) -> Request) -> Result<()> {
        let (reply, result) = mpsc::channel();
        self.requests
            .send(make(reply))
            .map_err(|_| anyhow!("preview is stopped; start it again"))?;
        match result.recv_timeout(Duration::from_secs(45)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(_) => bail!("camera processing did not respond; restart preview"),
        }
    }

    pub fn set_effects(&self, config: Effects) -> Result<()> {
        self.request(|reply| Request::Effects(config, reply))
    }
    pub fn start_recording(&self, path: &Path, audio: Option<&str>) -> Result<()> {
        self.request(|reply| Request::Record(path.to_owned(), audio.map(str::to_owned), reply))
    }
    pub fn pause_recording(&self, paused: bool) -> Result<()> {
        self.request(|reply| Request::Pause(paused, reply))
    }
    pub fn stop_recording(&self) -> Result<()> {
        self.request(Request::StopRecord)
    }
    pub fn set_virtual_output(&self, path: Option<&Path>) -> Result<()> {
        self.request(|reply| Request::Virtual(path.map(Path::to_owned), reply))
    }

    pub fn snapshot(&self, path: &Path) -> Result<()> {
        let frame = self
            .latest_frame()
            .ok_or_else(|| anyhow!("no preview frame available"))?;
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        ensure!(
            matches!(extension.as_str(), "png" | "jpg" | "jpeg"),
            "snapshot filename must end in .png, .jpg or .jpeg"
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| {
                format!(
                    "cannot create {}; existing files are never overwritten",
                    path.display()
                )
            })?;
        if extension == "png" {
            image::codecs::png::PngEncoder::new(file).write_image(
                &frame.rgb,
                frame.width,
                frame.height,
                image::ExtendedColorType::Rgb8,
            )?;
        } else {
            image::codecs::jpeg::JpegEncoder::new_with_quality(file, 95).encode(
                &frame.rgb,
                frame.width,
                frame.height,
                image::ExtendedColorType::Rgb8,
            )?;
        }
        Ok(())
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::Close);
        kill_bounded(&mut self.capture.lock().unwrap());
        // Output finalization may take time; bound the wait even if an output worker stalls.
        if let Some(worker) = self.worker.take() {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !worker.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
    }
}

fn capture_loop(
    mut stdout: UnixStream,
    receiver: mpsc::Receiver<Request>,
    shared: &Arc<Mutex<Shared>>,
    source: &Path,
    config: &VideoConfig,
) -> Result<()> {
    // Deadlines keep controls responsive even when capture stalls partway through a frame.
    stdout.set_read_timeout(Some(Duration::from_millis(50)))?;
    let mut processor = Processor::new(config.width, config.height)?;
    let frame_size = config.width as usize * config.height as usize * 3;
    let mut rgb = vec![0; frame_size];
    let mut offset = 0;
    let mut sequence = 0;
    let mut recording: Option<Recording> = None;
    let mut virtual_output: Option<(PathBuf, PipeOutput)> = None;
    let mut effect_replies = Vec::new();
    let result = (|| -> Result<()> {
        loop {
            loop {
                let request = match receiver.try_recv() {
                    Ok(request) => request,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
                };
                let (reply, result) = match request {
                    Request::Close => return Ok(()),
                    Request::Effects(effects, reply) => match processor.configure(effects) {
                        Ok(()) => {
                            shared.lock().unwrap().status.error = None;
                            effect_replies.push(reply);
                            continue;
                        }
                        Err(error) => (reply, Err(error)),
                    },
                    Request::Record(path, audio, reply) => {
                        let result = if recording.is_some() {
                            Err(anyhow!("already recording"))
                        } else {
                            Recording::start(&path, config, audio).map(|new| {
                                recording = Some(new);
                            })
                        };
                        (reply, result)
                    }
                    Request::Pause(paused, reply) => {
                        let result = recording
                            .as_mut()
                            .ok_or_else(|| anyhow!("not recording"))
                            .and_then(|recording| recording.set_paused(paused));
                        (reply, result)
                    }
                    Request::StopRecord(reply) => {
                        let result = recording
                            .take()
                            .ok_or_else(|| anyhow!("not recording"))
                            .and_then(|recording| recording.finish());
                        (reply, result)
                    }
                    Request::Virtual(path, reply) => {
                        let result = match path {
                            None => virtual_output
                                .take()
                                .map_or(Ok(()), |(_, output)| output.finish()),
                            Some(path) => (|| {
                                let path = video_node(&path)?;
                                ensure!(
                                    path != source,
                                    "virtual output cannot be the camera's capture device"
                                );
                                if virtual_output
                                    .as_ref()
                                    .is_some_and(|(open, _)| *open == path)
                                {
                                    return Ok(());
                                }
                                crate::virtual_camera::ensure_loopback(&path)?;
                                let output = PipeOutput::virtual_camera(&path, config)?;
                                if let Some((_, previous)) = virtual_output.replace((path, output))
                                {
                                    let _ = previous.finish();
                                }
                                Ok(())
                            })(),
                        };
                        (reply, result)
                    }
                };
                {
                    let mut state = shared.lock().unwrap();
                    if result.is_ok() {
                        state.status.error = None;
                    }
                    state.status.recording = recording.is_some();
                    state.status.paused = recording.as_ref().is_some_and(|r| r.paused);
                    state.status.virtual_device = virtual_output
                        .as_ref()
                        .map(|(path, _)| path.display().to_string());
                }
                let _ = reply.send(result.map_err(|error| format!("{error:#}")));
            }
            match stdout.read(&mut rgb[offset..]) {
                Ok(0) => bail!("camera video stream ended"),
                Ok(count) => offset += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                Err(error) => return Err(error).context("camera video stream ended"),
            }
            if offset < frame_size {
                continue;
            }
            offset = 0;
            // Do not publish unprocessed frames if a privacy effect fails.
            if let Err(error) = processor.process(&mut rgb) {
                for reply in effect_replies.drain(..) {
                    let _ = reply.send(Err(format!("{error:#}")));
                }
                return Err(error);
            }
            if let Some(Err(error)) = recording.as_mut().map(|output| output.write(&rgb)) {
                let mut message = format!("recording stopped: {error:#}");
                if let Some(output) = recording.take() {
                    if let Err(finalize) = output.finish() {
                        message.push_str(&format!("; {finalize:#}"));
                    }
                }
                let mut state = shared.lock().unwrap();
                state.status.recording = false;
                state.status.paused = false;
                state.status.error = Some(message);
            }
            if let Some(Err(error)) = virtual_output
                .as_mut()
                .map(|(_, output)| output.write(&rgb))
            {
                // A stop/stop-all from this or another process is intentional, not a failure.
                let requested = virtual_output
                    .take()
                    .is_some_and(|(_, output)| output.stop_requested());
                let mut state = shared.lock().unwrap();
                state.status.virtual_device = None;
                if !requested {
                    state.status.error = Some(format!("virtual output stopped: {error:#}"));
                }
            }
            sequence += 1;
            let frame = Arc::new(Frame {
                width: config.width,
                height: config.height,
                rgb,
                sequence,
            });
            let previous = {
                let mut state = shared.lock().unwrap();
                state.status.frames = sequence;
                state.frame.replace(frame)
            };
            // Acknowledgement is a snapshot barrier: callers must never save the preceding raw frame.
            for reply in effect_replies.drain(..) {
                let _ = reply.send(Ok(()));
            }
            rgb = previous
                .and_then(Arc::into_inner)
                .map(|frame| frame.rgb)
                .unwrap_or_else(|| vec![0; frame_size]);
        }
    })();
    if let Some(recording) = recording {
        if let Err(error) = recording.finish() {
            shared.lock().unwrap().status.error =
                Some(format!("recording finalization failed: {error:#}"));
        }
    }
    if let Some((_, output)) = virtual_output {
        let _ = output.finish();
    }
    let mut state = shared.lock().unwrap();
    state.status.recording = false;
    state.status.paused = false;
    state.status.virtual_device = None;
    result
}

fn ffmpeg() -> Command {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .stdin(Stdio::null())
        .stderr(Stdio::piped());
    command
}

fn video_node(path: &Path) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("cannot access video device {}", path.display()))?;
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    ensure!(
        path.parent() == Some(Path::new("/dev"))
            && name.strip_prefix("video").is_some_and(
                |suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
            )
            && path.metadata()?.file_type().is_char_device(),
        "{} is not a /dev/videoN character device",
        path.display()
    );
    Ok(path)
}

#[derive(Clone)]
struct ErrorLog(Arc<Mutex<VecDeque<u8>>>);
impl ErrorLog {
    fn drain(mut stderr: impl Read + Send + 'static) -> Self {
        let log = Self(Arc::new(Mutex::new(VecDeque::with_capacity(8192))));
        let output = log.clone();
        thread::spawn(move || {
            let mut bytes = [0; 1024];
            loop {
                let count = match stderr.read(&mut bytes) {
                    Ok(0) => break,
                    Ok(count) => count,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                };
                let mut log = output.0.lock().unwrap();
                for &byte in &bytes[..count] {
                    if log.len() == 8192 {
                        log.pop_front();
                    }
                    log.push_back(byte);
                }
            }
        });
        log
    }
    fn text(&self) -> String {
        let log = self.0.lock().unwrap();
        String::from_utf8_lossy(&log.iter().copied().collect::<Vec<_>>())
            .trim()
            .to_owned()
    }
}

struct PipeOutput {
    child: Option<Child>,
    input: Option<UnixStream>,
    errors: ErrorLog,
    /// Virtual-camera slot, released after the child is gone (field drops after `Drop::drop`).
    claim: Option<crate::virtual_camera::Claim>,
}
impl PipeOutput {
    fn spawn(mut command: Command) -> Result<Self> {
        let (input, reader) =
            UnixStream::pair().context("cannot create bounded video output socket")?;
        input.set_write_timeout(Some(Duration::from_secs(5)))?;
        command.stdin(Stdio::from(OwnedFd::from(reader)));
        let mut child = command.spawn().context("cannot start FFmpeg output")?;
        let input = Some(input);
        let errors = ErrorLog::drain(child.stderr.take().expect("piped output stderr"));
        Ok(Self {
            child: Some(child),
            input,
            errors,
            claim: None,
        })
    }
    fn raw_input(command: &mut Command, config: &VideoConfig) {
        command.args([
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &config.size(),
            "-framerate",
            &config.fps.to_string(),
            "-i",
            "pipe:0",
        ]);
    }
    fn segment(path: &Path, config: &VideoConfig, audio: Option<&str>) -> Result<Self> {
        let mut command = ffmpeg();
        // Preserve real elapsed time when local processing cannot keep up with the camera.
        command.args(["-use_wallclock_as_timestamps", "1"]);
        Self::raw_input(&mut command, config);
        if let Some(audio) = audio.filter(|s| !s.is_empty()) {
            let (format, source) = if let Some(source) = audio.strip_prefix("pulse:") {
                ("pulse", source)
            } else {
                ("alsa", audio.strip_prefix("alsa:").unwrap_or(audio))
            };
            command.args([
                "-thread_queue_size",
                "64",
                "-f",
                format,
                "-i",
                source,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-shortest",
            ]);
        }
        command
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "20",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "vfr",
                "-f",
                "matroska",
                "-n",
            ])
            .arg(path)
            .stdout(Stdio::null());
        Self::spawn(command)
    }
    fn virtual_camera(path: &Path, config: &VideoConfig) -> Result<Self> {
        let claim = crate::virtual_camera::Claim::acquire(path)?;
        let mut command = ffmpeg();
        Self::raw_input(&mut command, config);
        command
            .args(["-threads", "2", "-pix_fmt", "yuyv422", "-f", "v4l2"])
            .arg(path)
            .env(crate::virtual_camera::PRODUCER_ENV, claim.token())
            .stdout(Stdio::null());
        let mut output = Self::spawn(command)?;
        // The claim is owned by the output before `attach`, so a failed or cancelled start kills
        // the child first and only then releases the reservation.
        let claim = output.claim.insert(claim);
        // A wedged sink must not stall the preview for long; the output is dropped on timeout.
        if let Some(input) = &output.input {
            input.set_write_timeout(Some(Duration::from_secs(1)))?;
        }
        claim.attach(output.child.as_ref().expect("output child exists"))?;
        Ok(output)
    }
    fn stop_requested(&self) -> bool {
        self.claim
            .as_ref()
            .is_some_and(|claim| claim.stop_requested())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(status) = self
            .child
            .as_mut()
            .expect("output child exists")
            .try_wait()?
        {
            bail!("FFmpeg output stopped ({status}): {}", self.errors.text());
        }
        self.input
            .as_mut()
            .ok_or_else(|| anyhow!("output is closed"))?
            .write_all(bytes)
            .with_context(|| format!("cannot write video output: {}", self.errors.text()))
    }
    fn finish(mut self) -> Result<()> {
        self.input.take();
        wait_output(
            self.child.as_mut().expect("output child exists"),
            &self.errors,
            Duration::from_secs(10),
        )
    }
}
impl Drop for PipeOutput {
    fn drop(&mut self) {
        self.input.take();
        let Some(mut child) = self.child.take() else {
            return;
        };
        if !kill_bounded(&mut child) {
            // A kernel-stuck child retains its claim until the reaper observes its actual exit.
            let claim = self.claim.take();
            thread::spawn(move || {
                let _ = child.wait();
                drop(claim);
            });
        }
    }
}

/// Kill and wait at most 2 s; `false` means the child is still alive (stuck in uninterruptible kernel
/// I/O, which ignores SIGKILL). Callers must keep owning such a child instead of blocking on it.
fn kill_bounded(child: &mut Child) -> bool {
    let _ = child.kill();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) => return false,
            Ok(Some(_)) | Err(_) => return true,
        }
    }
}

/// Run a helper in its own process group with a deadline. On timeout the group is killed and the
/// leader is handed to a reaper thread; this call never blocks on a stuck process.
fn run_bounded(
    mut command: Command,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, Vec<u8>, Vec<u8>)> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = command.spawn()?;
    let read = |pipe: Option<Box<dyn Read + Send>>| {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(pipe) = pipe {
                let _ = pipe.take(1 << 20).read_to_end(&mut bytes);
            }
            let _ = sender.send(bytes);
        });
        receiver
    };
    let stdout = read(child.stdout.take().map(|p| Box::new(p) as _));
    let stderr = read(child.stderr.take().map(|p| Box::new(p) as _));
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = crate::process::stop_group(&mut child);
            thread::spawn(move || {
                let _ = child.wait();
            });
            bail!("did not finish within {} seconds", timeout.as_secs());
        }
        thread::sleep(Duration::from_millis(20));
    };
    let collect = |receiver: mpsc::Receiver<Vec<u8>>| {
        receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap_or_default()
    };
    Ok((status, collect(stdout), collect(stderr)))
}

fn wait_output(child: &mut Child, errors: &ErrorLog, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(
                status.success(),
                "FFmpeg output failed ({status}): {}",
                errors.text()
            );
            return Ok(());
        }
        if Instant::now() >= deadline {
            kill_bounded(child);
            bail!("FFmpeg did not finalize output in time: {}", errors.text());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

static NEXT_RECORDING: AtomicU64 = AtomicU64::new(0);
struct Recording {
    destination: PathBuf,
    file: Option<File>,
    directory: PathBuf,
    config: VideoConfig,
    audio: Option<String>,
    segment: Option<PipeOutput>,
    paused: bool,
    segments: usize,
    frames: u64,
    finished: bool,
}
impl Recording {
    fn start(path: &Path, config: &VideoConfig, audio: Option<String>) -> Result<Self> {
        container(path)?;
        let directory = std::env::temp_dir().join(format!(
            "link-camera-{}-{}",
            std::process::id(),
            NEXT_RECORDING.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .context("cannot create private recording directory")?;
        let file = match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => file,
            Err(error) => {
                let _ = fs::remove_dir(&directory);
                return Err(error).with_context(|| {
                    format!(
                        "cannot create {}; existing files are never overwritten",
                        path.display()
                    )
                });
            }
        };
        Ok(Self {
            destination: path.to_owned(),
            file: Some(file),
            directory,
            config: config.clone(),
            audio,
            segment: None,
            paused: false,
            segments: 0,
            frames: 0,
            finished: false,
        })
    }
    fn set_paused(&mut self, paused: bool) -> Result<()> {
        self.paused = paused;
        if paused {
            if let Some(segment) = self.segment.take() {
                segment.finish()?;
            }
        }
        Ok(())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if self.paused {
            return Ok(());
        }
        // Empty active spans must never become inputs to the final concat.
        if self.segment.is_none() {
            let path = self.directory.join(format!("{:04}.mkv", self.segments));
            self.segment = Some(PipeOutput::segment(
                &path,
                &self.config,
                self.audio.as_deref(),
            )?);
            self.segments += 1;
        }
        if let Some(segment) = self.segment.as_mut() {
            segment.write(bytes)?;
            self.frames += 1;
        }
        Ok(())
    }
    fn finish(mut self) -> Result<()> {
        if let Some(segment) = self.segment.take() {
            segment.finish()?;
        }
        ensure!(
            self.frames > 0,
            "recording contains no video frames; recovery files: {}",
            self.directory.display()
        );
        let manifest = (0..self.segments)
            .map(|index| format!("file '{index:04}.mkv'\n"))
            .collect::<String>();
        fs::write(self.directory.join("segments.txt"), manifest)?;
        let format = container(&self.destination)?;
        let mut command = ffmpeg();
        command.current_dir(&self.directory).args([
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            "segments.txt",
            "-map",
            "0",
            "-c",
            "copy",
        ]);
        if format == "mp4" {
            command.args(["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]);
        }
        command
            .args(["-f", format, "pipe:1"])
            .stdout(Stdio::from(self.file.take().expect("reserved output file")));
        let mut child = command.spawn().context("cannot assemble recording")?;
        let errors = ErrorLog::drain(child.stderr.take().expect("piped concat stderr"));
        wait_output(&mut child, &errors, Duration::from_secs(30)).with_context(|| {
            format!(
                "recording segments preserved for recovery in {}",
                self.directory.display()
            )
        })?;
        self.finished = true;
        fs::remove_dir_all(&self.directory)
            .context("recording saved but temporary segments could not be removed")?;
        Ok(())
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        if !self.finished {
            eprintln!(
                "link-camera: unfinished recording; segments retained at {} (destination {})",
                self.directory.display(),
                self.destination.display()
            );
        }
    }
}
fn container(path: &Path) -> Result<&'static str> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" | "mov" => Ok("mp4"),
        "mkv" => Ok("matroska"),
        _ => bail!("recording filename must end in .mp4, .mov or .mkv"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capture_request(
        requests: &mpsc::Sender<Request>,
        make: impl FnOnce(Reply) -> Request,
    ) -> std::result::Result<(), String> {
        let (reply, result) = mpsc::channel();
        requests.send(make(reply)).unwrap();
        result
            .recv_timeout(Duration::from_secs(2))
            .expect("capture control blocked waiting for a frame")
    }
    #[test]
    fn idle_capture_handles_recording_controls_and_close_without_a_frame() {
        let config = VideoConfig {
            width: 64,
            height: 64,
            fps: 30,
        };
        let path = std::env::temp_dir().join(format!(
            "link-camera-idle-test-{}-{}.mkv",
            std::process::id(),
            NEXT_RECORDING.fetch_add(1, Ordering::Relaxed)
        ));
        let (reader, writer) = UnixStream::pair().unwrap();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let worker_shared = shared.clone();
        let (requests, receiver) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _ = done.send(capture_loop(
                reader,
                receiver,
                &worker_shared,
                Path::new("/unused-camera"),
                &config,
            ));
        });
        capture_request(&requests, |reply| {
            Request::Record(path.clone(), None, reply)
        })
        .unwrap();
        assert!(shared.lock().unwrap().status.recording);
        capture_request(&requests, |reply| Request::Pause(true, reply)).unwrap();
        assert!(shared.lock().unwrap().status.paused);
        capture_request(&requests, |reply| Request::Pause(false, reply)).unwrap();
        assert!(!shared.lock().unwrap().status.paused);
        let empty = capture_request(&requests, Request::StopRecord).unwrap_err();
        {
            let state = shared.lock().unwrap();
            assert!(!state.status.recording);
            assert!(!state.status.paused);
            assert!(state.frame.is_none());
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
        capture_request(&requests, |reply| Request::Virtual(None, reply)).unwrap();
        requests.send(Request::Close).unwrap();
        finished
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        worker.join().unwrap();
        drop(writer);
        fs::remove_file(path).unwrap();
        // Empty recordings deliberately retain recovery directories; clean only this error's path.
        let temporary = std::env::temp_dir();
        let prefix = format!("link-camera-{}-", std::process::id());
        for value in empty.split_whitespace() {
            let recovery = Path::new(value);
            if recovery.parent() == Some(temporary.as_path())
                && recovery
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.strip_prefix(&prefix)
                            .is_some_and(|id| id.parse::<u64>().is_ok())
                    })
            {
                fs::remove_dir(recovery).unwrap();
            }
        }
    }

    #[test]
    fn partial_capture_frames_survive_timeouts_and_effects_stay_a_publication_barrier() {
        let config = VideoConfig {
            width: 64,
            height: 64,
            fps: 30,
        };
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let worker_shared = shared.clone();
        let (requests, receiver) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _ = done.send(capture_loop(
                reader,
                receiver,
                &worker_shared,
                Path::new("/unused-camera"),
                &config,
            ));
        });
        let mut first = Vec::new();
        let mut second = Vec::new();
        for y in 0..64 {
            for x in 0..64 {
                first.extend_from_slice(&[x, y, 17]);
                second.extend_from_slice(&[x, y, 211]);
            }
        }
        writer.write_all(&first[..17]).unwrap();
        let (reply, effects) = mpsc::channel();
        requests
            .send(Request::Effects(
                Effects {
                    mirror: true,
                    ..Effects::default()
                },
                reply,
            ))
            .unwrap();
        capture_request(&requests, |reply| Request::Virtual(None, reply)).unwrap();
        assert!(matches!(
            effects.recv_timeout(Duration::from_millis(150)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(shared.lock().unwrap().frame.is_none());
        writer.write_all(&first[17..]).unwrap();
        writer.write_all(&second[..31]).unwrap();
        effects
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        {
            let state = shared.lock().unwrap();
            let frame = state.frame.as_ref().unwrap();
            assert_eq!(frame.sequence, 1);
            for (index, pixel) in frame.rgb.chunks_exact(3).enumerate() {
                assert_eq!(pixel, &[63 - (index % 64) as u8, (index / 64) as u8, 17]);
            }
        }
        let (reply, effects) = mpsc::channel();
        requests
            .send(Request::Effects(Effects::default(), reply))
            .unwrap();
        capture_request(&requests, |reply| Request::Virtual(None, reply)).unwrap();
        assert!(matches!(
            effects.recv_timeout(Duration::from_millis(150)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(shared.lock().unwrap().status.frames, 1);
        writer.write_all(&second[31..]).unwrap();
        effects
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        {
            let state = shared.lock().unwrap();
            let frame = state.frame.as_ref().unwrap();
            assert_eq!(frame.sequence, 2);
            assert_eq!(frame.rgb, second);
        }
        drop(requests);
        finished
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        worker.join().unwrap();
        drop(writer);
    }

    #[test]
    fn rejects_dimensions_that_break_rgb_or_video_encoding() {
        for config in [
            VideoConfig {
                width: 0,
                ..VideoConfig::default()
            },
            VideoConfig {
                width: 1279,
                ..VideoConfig::default()
            },
            VideoConfig {
                fps: 0,
                ..VideoConfig::default()
            },
            VideoConfig {
                height: 100_000,
                ..VideoConfig::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
        assert!(
            VideoConfig {
                width: 3840,
                height: 2160,
                fps: 30
            }
            .validate()
            .is_ok()
        );
    }
    #[test]
    fn does_not_treat_arbitrary_files_as_camera_devices() {
        assert!(video_node(Path::new("/dev/null")).is_err());
        assert!(container(Path::new("capture.png")).is_err());
    }
    #[test]
    fn enumerates_only_advertised_mjpeg_size_rate_pairs() {
        let modes = parse_formats(
            "[0]: 'H264'\nSize: Discrete 640x480\nInterval: Discrete 0.033s (30.000 fps)\n[1]: 'MJPG'\nSize: Discrete 1280x960\nInterval: Discrete 0.033s (30.000 fps)\nInterval: Discrete 0.040s (25.000 fps)\nSize: Discrete 1920x1080\nInterval: Discrete 0.033s (29.970 fps)\nInterval: Discrete 0.033s (30.000 fps)\n[2]: 'YUYV'\nSize: Discrete 320x240\nInterval: Discrete 0.033s (30.000 fps)",
        );
        assert_eq!(
            modes,
            vec![
                VideoConfig {
                    width: 1280,
                    height: 960,
                    fps: 30
                },
                VideoConfig {
                    width: 1280,
                    height: 960,
                    fps: 25
                },
                VideoConfig {
                    width: 1920,
                    height: 1080,
                    fps: 30
                },
            ]
        );
    }
    #[test]
    #[ignore = "requires FFmpeg with libx264; uses generated frames, no camera or audio device"]
    fn recording_omits_empty_spans_and_rejects_empty_recordings() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "link-camera-recording-test-{}-{}",
            std::process::id(),
            NEXT_RECORDING.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&directory)?;
        let config = VideoConfig {
            width: 64,
            height: 64,
            fps: 30,
        };
        let frame = vec![96; config.width as usize * config.height as usize * 3];
        let result = (|| -> Result<()> {
            for extension in ["mp4", "mov", "mkv"] {
                for pause_before_stop in [false, true] {
                    let path =
                        directory.join(format!("{extension}-{pause_before_stop}.{extension}"));
                    let mut recording = Recording::start(&path, &config, None)?;
                    ensure!(!recording.paused, "new recording reported paused");
                    recording.set_paused(true)?;
                    recording.write(&frame)?;
                    recording.set_paused(false)?;
                    recording.set_paused(true)?;
                    recording.set_paused(false)?;
                    ensure!(!recording.paused, "resumed recording reported paused");
                    recording.write(&frame)?;
                    recording.set_paused(true)?;
                    recording.set_paused(false)?;
                    recording.set_paused(true)?;
                    recording.set_paused(false)?;
                    recording.write(&frame)?;
                    recording.set_paused(true)?;
                    recording.set_paused(false)?;
                    if pause_before_stop {
                        recording.set_paused(true)?;
                    }
                    recording.finish()?;

                    let mut decode = ffmpeg();
                    decode.arg("-i").arg(&path).args([
                        "-map",
                        "0:v:0",
                        "-fps_mode",
                        "passthrough",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ]);
                    let (status, bytes, errors) = run_bounded(decode, Duration::from_secs(10))?;
                    ensure!(
                        status.success(),
                        "{} could not be decoded: {}",
                        path.display(),
                        String::from_utf8_lossy(&errors)
                    );
                    ensure!(
                        bytes.len() == frame.len() * 2,
                        "{} did not preserve both recorded frames",
                        path.display()
                    );
                }
            }
            for paused in [false, true] {
                let path = directory.join(format!("empty-{paused}.mkv"));
                let mut recording = Recording::start(&path, &config, None)?;
                if paused {
                    recording.set_paused(true)?;
                    recording.set_paused(false)?;
                    recording.set_paused(true)?;
                }
                let recovery = recording.directory.clone();
                let error = recording.finish().unwrap_err();
                ensure!(
                    error
                        .to_string()
                        .contains("recording contains no video frames"),
                    "empty recording failed unclearly: {error:#}"
                );
                ensure!(fs::metadata(path)?.len() == 0, "empty recording was saved");
                fs::remove_dir(recovery)?;
            }
            Ok(())
        })();
        fs::remove_dir_all(directory)?;
        result
    }
    #[test]
    #[ignore = "requires an idle live camera; sets its capture format, never its picture controls"]
    fn effects_acknowledge_a_processed_frame() {
        let directory = std::env::temp_dir().join(format!(
            "link-camera-effects-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .unwrap();
        let overlay = directory.join("opaque.png");
        let config = VideoConfig::default();
        image::RgbImage::from_pixel(config.width, config.height, image::Rgb([255, 0, 0]))
            .save(&overlay)
            .unwrap();
        let result = (|| -> Result<(bool, u64, u64)> {
            let device =
                std::env::var("LINK_CAMERA_TEST_DEVICE").unwrap_or_else(|_| "/dev/video0".into());
            let preview = Preview::start(&device, &config)?;
            let before = preview.wait_first_frame(Duration::from_secs(10))?;
            preview.set_effects(Effects {
                overlay_image: Some(overlay),
                ..Effects::default()
            })?;
            let after = preview
                .latest_frame()
                .ok_or_else(|| anyhow!("effects returned without a preview frame"))?;
            Ok((
                after.rgb.chunks_exact(3).all(|pixel| pixel == [255, 0, 0]),
                before.sequence,
                after.sequence,
            ))
        })();
        fs::remove_dir_all(directory).unwrap();
        let (processed, before, after) = result.unwrap();
        assert!(
            processed && after > before,
            "effects acknowledged a stale/unprocessed frame: before={before}, after={after}"
        );
    }
}
