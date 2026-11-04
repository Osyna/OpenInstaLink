<p align="center">
  <img alt="" src="assets/icons/link-logo.svg" width="72"/>
</p>

# Link Camera

<p align="center">
  Local controller for an Insta360 Link camera on Linux.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License GPL-3.0-only" src="https://img.shields.io/badge/license-GPL--3.0--only-blue"/></a>
  <a href="CHANGELOG.md"><img alt="version 0.6.1" src="https://img.shields.io/badge/version-0.6.1-lightgrey"/></a>
</p>

Native Rust controller for Insta360 Link cameras on Linux, with a Slint desktop interface and a headless CLI. The connected Link 2C is the hardware target. No Windows installation, Wine, account or cloud inference is required.

The GitHub repository is [Osyna/OpenInstaLink](https://github.com/Osyna/OpenInstaLink). The program and crate name is `link-camera`.

## Features

- Picture, focus, exposure, zoom and framing controls from the desktop app or the CLI, through the kernel `uvcvideo` driver.
- Settings are saved per physical camera and reapplied when that camera is used again, including by other programs. The helper does not open a video stream, and it does not turn privacy mode back on.
- PNG/JPEG snapshots and MKV/MP4/MOV recordings never overwrite an existing file. Virtual camera output uses a `v4l2loopback` node this app created.
- The person-matting model is bundled and checked by SHA-256 before use. Eight further vision models are not bundled; each is installed only by an explicit download, and inference stays on this machine.
- `link-camera serve` is a token-gated browser page for the same controls. It does not stream video. This repository does not include a Home Assistant integration or a TV app.

## Install

```sh
./target/release/link-camera            # Slint GUI
./target/release/link-camera --help     # CLI commands
```

To build from source (Rust 1.92 or newer):

```sh
cargo build --release --locked
cargo run --release --locked -- gui
```

A CLI-only build does not compile Slint:

```sh
cargo build --release --locked --no-default-features
./target/release/link-camera status
```

The first build downloads Cargo dependencies and the native ONNX Runtime build dependency. The bundled RVM person-matting model needs no runtime download. The eight optional vision models are not bundled and nothing in a build or a normal launch downloads them; each one is installed only by an explicit download (see [Local vision models](#local-vision-models)). The prebuilt ONNX Runtime targets x86-64 CPUs with AVX2.

### Dependencies

Runtime: `ffmpeg`, `v4l2-ctl` (v4l-utils), access to the camera's `/dev/videoN` capture node, and a desktop session for the GUI. File picking optionally uses `zenity` or `kdialog`; paths can also be entered directly. GUI microphone discovery uses `pactl` when available; recording without audio needs neither PulseAudio nor ALSA capture permission.

Only the features that use them need these extras:

- Microphone state, mute and volume: `pactl` (PulseAudio or PipeWire-Pulse).
- Virtual camera creation and removal: the `v4l2loopback` kernel module (installed for the running kernel), `v4l2loopback-utils` (`v4l2loopback-ctl` 0.13 or newer) and polkit's `pkexec`. See [Virtual camera](#virtual-camera).
- Local AI download: `uv`, or `python3.11` on `PATH`, plus `libseccomp2` (Arch/Fedora: `libseccomp`) for the offline network sandbox and `libegl1` and `libgles2` for MediaPipe. See [Local vision models](#local-vision-models).

Arch Linux build/runtime packages:

```sh
sudo pacman -S --needed rust base-devel pkgconf openssl fontconfig \
  libxkbcommon libxkbcommon-x11 libx11 wayland ffmpeg v4l-utils
```

Debian/Ubuntu build packages include `build-essential`, `pkg-config`, `libssl-dev`, `libfontconfig1-dev`, `libxkbcommon-dev`, `libx11-dev`, `libxcursor-dev`, `libxi-dev`, `libxrandr-dev`, `libudev-dev`, `libwayland-dev`, `libegl1-mesa-dev` and `libgl1-mesa-dev`, plus `ffmpeg` and `v4l-utils` at runtime.

Do not run the app as root. A normal active desktop session usually receives camera access through logind ACLs. If the renderer cannot initialize, an explicit software-rendering launch is available:

```sh
SLINT_BACKEND=winit-software ./target/release/link-camera
```

## Usage

Open the app with no arguments, or run a subcommand. Both talk to the connected Link camera through `uvcvideo`.

## Desktop

The app starts a local preview on opening and reuses the capture size, frame rate and local effects you last used. Stop preview to release capture to another application. The camera selector, capture size and frame-rate menus use the connected camera's reported MJPEG modes.

The light studio layout uses a labelled navigation rail, a large image stage and a scrollable inspector. Numeric camera edits commit before switching sections; keyboard Tab, Space and Enter work with the new buttons. While recording, Pause/Resume and Stop replace Record rather than pushing actions off the compact window. Errors offer **Details** for the full message.

- **Camera:** supported image, focus, white-balance, exposure, HDR, zoom/view, framing, gesture, privacy and audio-processing controls. Exact numeric entry and sliders share the same backend as the CLI. Range limits and defaults come from the camera's V4L2 metadata; a control that reports no default cannot be reset and says so. On/off controls are switches. Range sliders move with the mouse wheel and arrow keys (Page Up/Down 10 steps, Home/End to the ends). Double-click, Backspace/Delete or the reset button restores the camera's default. A plain click on the track commits after 500 ms, so a double-click reset never has a hardware write ahead of it.
- **Views:** local camera profiles plus on-camera slots and tracking zones when the firmware exposes them. On the tested Link 2C, on-camera preset slots return Unsupported; their save controls are disabled. Use local camera profiles instead. Pan/tilt controls do not imply a physical gimbal on a fixed camera.
- **Effects:** green-screen keying, image softening, mirror, image overlays and four-corner whiteboard correction, plus person-segmented blur/replacement once optional AI is enabled. Changes affect preview, snapshots, recordings and virtual output.
- **Output:** capture folders, recording container, the recording audio source with its activity state, mute and input volume, and the virtual camera controls (create, start/stop sending, remove, stop all). PNG/JPEG snapshots and MKV/MP4/MOV recordings never overwrite an existing file. Recordings support pause/resume and are finalized on Stop or window close.
- **Script:** editable, automatically saved teleprompter with playback, speed, text size and position. It is an on-screen reading aid, not part of snapshots, recordings or virtual output.
- **AI tools (off by default):** turn on **Enable optional AI features** in Settings first. **Model** explains each of the eight local models and manages its downloads; **Analyze** holds one-shot/live controls and model-specific options; **History** browses the latest 100 results across all models, with frame-matched renders, complete raw outputs and no-overwrite exports. Download and deletion each require confirmation. Inference does not block preview or camera controls.
- **Settings:** the optional-AI switch and what the app saves for you.

Tracking on the Link 2C family is Auto Framing: the camera reframes the picture and no motor moves. The tracking status line reports what the firmware says (off, on, needs a video stream, paused by privacy mode, or between modes). Modes only apply while a program is streaming. While Auto Framing or tracking drives the view, saved pan/tilt/zoom is not replayed over it.

Unreadable values are not written as defaults. Failed writes restore actual readback when communication is available; a lost USB transport clears controls instead of displaying stale settings. Controls can depend on current mode, frame format and firmware. Applying a profile is sequential, not transactional: it stops at the first rejected setting and earlier settings may already have changed. Saving an existing profile name replaces that named profile atomically.

### Saved camera settings

Camera controls are saved per physical camera after each successful change, from the actual readback rather than the requested value, and are restored when that camera is used again. A failed write saves nothing; a change the camera cannot confirm is reported as applied but not saved, and is never replayed automatically.

A background helper (`link-camera-restore.service`, a user service enabled after your first saved change) watches for cameras without opening a video stream and reapplies the saved settings when any application starts capturing, including Teams, Zoom and browsers. Picture controls are reapplied immediately; settings that the firmware only accepts while streaming wait for a stream. Privacy mode is never reopened automatically.

A per-device discovery or setup error is reported without terminating the watcher. A camera disappearing between enumeration and identity lookup does not prevent the remaining cameras from being scanned.

```sh
$APP restore apply      # apply saved settings once, without taking capture
$APP restore disable    # stop automatic restoration
$APP restore enable     # turn it back on
```

Other applications can still change standard V4L2 controls while they capture, and their own automatic adjustments win until the camera is used again. This app's effects (mirror, softening, backgrounds, overlays) reach other applications only through the virtual camera; the camera's own settings do not need one. Stop the preview here before an application that needs exclusive capture.

### Optional AI

Local AI features are disabled until you turn them on, in Settings or with `link-camera settings ai true`. They are the eight vision models and the neural person blur/replacement. While off, no model runs, nothing is downloaded and `models list`, `models download`, `models run` and `models watch` fail with that reason. `models delete` still works so a user can reclaim disk space with AI off. Enabling does not download or start anything by itself. Turning it off stops new analysis, cancels owned inference, unloads its worker and switches neural backgrounds off. Unreadable settings keep AI off rather than assuming it is on. Green-screen keying, softening, mirror, overlays and the camera's own HDR/framing/tracking do not depend on this switch.

### Camera connection errors

`Preview unavailable` with `/dev/videoN: Input/output error` is a capture/device failure, not a model error. The desktop runs camera controls in a private subprocess with a 15-second reply deadline. A timeout requests cancellation of that process group, clears the session and stops automatic refresh. It never automatically repeats a write. Cancellation cannot undo a USB request already issued in the kernel: a pending write has an unknown outcome, so read back the settings after recovery before explicitly retrying it.

Unplug the camera itself, then reconnect it directly to a working PC USB port; try another suitable cable if USB descriptor reads still time out. Do not reset a shared hub containing mounted storage or other peripherals. Use **Refresh devices** if the camera is missing, then **Reconnect** once it appears. The app does not reset USB or override camera privacy. If the old control process remains blocked in the kernel, it must exit before another is started.

The 15-second process deadline applies to desktop controls. Direct CLI camera operations still inherit kernel USB waits; do not repeatedly issue writes against an unresponsive device.

## CLI

```sh
APP=./target/release/link-camera

$APP list --json
$APP --device /dev/video0 status --json
$APP formats
$APP controls --group Picture

$APP set brightness 55
$APP set zoom 1.5
$APP set hdr true
$APP set mode auto-framing
$APP set focus-position 65
$APP set white-balance-temperature 5000

$APP profile save desk
$APP profile list
$APP profile load desk
$APP profile export camera-settings.json
$APP profile apply camera-settings.json

$APP settings show
$APP settings ai true              # local AI is off until you do this
$APP settings ai false
$APP restore apply

$APP mic sources                    # activity state, mute and volume per source
$APP mic status                     # the camera's own microphone, matched by USB path
$APP mic mute alsa_input.NAME true
$APP mic volume alsa_input.NAME 80  # 0-100


$APP snapshot photo.png
$APP snapshot blurred.png --background blur --blur 12
$APP snapshot replaced.png --background replace --background-image background.jpg
$APP snapshot overlay.png --overlay logo.png --overlay-opacity 0.8
$APP snapshot corrected.png --board 0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9

$APP record clip.mkv --duration 10
$APP record clip.mp4 --background blur             # Ctrl-C finalizes it
$APP record sound.mkv --duration 10 --audio pulse:default
```

On this machine the camera microphone was also exercised with `--audio alsa:plughw:I2C,0`. ALSA card names vary. The GUI can enumerate PulseAudio/PipeWire-Pulse sources with Rescan.

`set` accepts the keys, ranges and choice values reported by `controls`; negative values are accepted. Vendor settings need an active camera stream. The CLI starts a temporary capture when a write reports NotStreaming. Standard controls can be changed while another application captures video. Snapshot/record/virtual streaming require capture ownership: stop another preview first if the driver reports Device or resource busy.

`preset` and `area` expose typed firmware operations, not raw USB requests. Use their `--help` subcommands. Unsupported firmware operations fail rather than being simulated. `--json` is available on data-printing commands.

### Effects configuration

GUI and CLI use the same local processor. A JSON file can hold its configuration:

```json
{
  "background": "blur",
  "blur": 12,
  "soften": 0.15,
  "mirror": false,
  "overlay_opacity": 1
}
```

```sh
$APP record local-effects.mkv --effects effects.json
```

Additional fields: `background_image` (required for `replace`/`chroma`), `overlay_image`, and `board` (normalized corners in TL, TR, BR, BL order). Backgrounds are centre-cropped at source resolution before scaling to cover the frame, avoiding oversized intermediates for very wide or tall images. Overlay images preserve alpha, are centered and are not upscaled. Green-screen thresholds are fixed; this is not a blue-screen keyer. Whiteboard correction stretches the chosen quadrilateral to the output frame's aspect ratio.

Person blur/replacement use the separately licensed **Robust Video Matting** model, not Insta360's proprietary effects engine. Softening is local image processing, not a claim to reproduce vendor makeup. Missing or hash-mismatched model files fail explicitly. See [model provenance and lookup paths](assets/models/README.md).

### Virtual camera

A virtual camera is a v4l2loopback node that other applications (Teams, browsers, OBS) open like a webcam. Link Camera feeds it the processed picture with FFmpeg. The camera's own settings reach other applications without it; this app's effects (mirror, softening, backgrounds, overlays, whiteboard correction) reach Teams only through a virtual camera that is receiving frames, so start sending before joining the call.

```sh
$APP virtual-camera status                       # what is missing, as exact commands
$APP virtual-camera list                         # every v4l2loopback node: owned/unmanaged, producer
$APP virtual-camera create --label "Link Camera" # needs write access to /dev/v4l2loopback
$APP virtual-camera create --label "Link Camera" --authorize
$APP stream --virtual-output /dev/video10 --background blur
$APP virtual-camera stop /dev/video10            # stop this app's producer on one node
$APP virtual-camera stop-all                     # stop every producer this app owns
$APP virtual-camera remove /dev/video10 --authorize
$APP virtual-camera remove-all --authorize
```

Prerequisites are the `v4l2loopback` module for the running kernel, `v4l2loopback-ctl` 0.13 or newer and FFmpeg. `status` prints the missing pieces. Without `--authorize` a step that needs root fails with the exact command to run yourself. With `--authorize`, `create` may show a polkit prompt through `pkexec` and, if the installed module is not loaded, runs `modprobe v4l2loopback devices=0` first, which loads the driver without creating any device. `remove` and `remove-all` take `--authorize` for the same prompt. Nodes are added and deleted one at a time with `v4l2loopback-ctl`; the app never unloads the kernel module. Deletion runs as a short `sh` script that checks each node's card label again right before `v4l2loopback-ctl delete`, so the prompt for `remove` names `sh`, and a number that another application took while the prompt was open is left alone.

Created nodes use `exclusive_caps=1`, so a consumer such as Teams lists the camera only while a producer is active. An idle node can be missing from the application's camera list; start the preview and sending, then reopen the picker.

Ownership is recorded in `$XDG_STATE_HOME/link-camera/virtual-camera/` and checked against the node's exact label. `stop`, `stop-all`, `remove` and `remove-all` operate on recorded nodes and their FFmpeg producers from any Link Camera window or CLI process, without opening the physical camera. `stop` and `stop-all` work without a preview and leave the nodes installed. Unmanaged devices are not selected for removal. The driver's delete ioctl accepts only a device number, not an identity token: another administrative tool can still replace a node between the label check and deletion. Do not concurrently add or remove v4l2loopback devices using other tools while deleting them here. The app's state lock coordinates Link Camera instances, not external tools. A node takes one Link Camera producer at a time.

Removal reserves its target nodes before stopping their producers, including while an administrator prompt is open. Other current-version instances cannot claim or remove those nodes until the operation returns; unrelated nodes remain available. Reservations are released on completion or error and pruned when their owning process identity is no longer alive. Close older app versions before upgrading: they do not honor removal reservations.

If a cancelled producer is still starting or cleaning up, removal leaves that node installed and reports the pending cleanup instead of deleting it. Wait for the output to finish stopping, then retry removal. `remove-all` can still remove other eligible nodes.

`stream` and `record --virtual-output` still accept an existing output node such as `/dev/video10`; you can also create one yourself with `sudo modprobe v4l2loopback video_nr=10 card_label="Link Camera" exclusive_caps=1`. The development host has neither the v4l2loopback module nor `v4l2loopback-ctl`, so creation, the node and consumption by a third-party application have not been verified here.

### Microphone

`mic sources` lists capture sources from the sound server (never monitor sources) and `mic status` finds the camera's own microphone by the camera's USB device path. If no source is on that path, or several are, the result says so and nothing is chosen for you; pass an explicit `SOURCE` name to `mute` and `volume`. These commands do not open a capture stream and never change the system default source.

The state field is the sound server's report, not a level meter: `RUNNING` means something is capturing from the source, `IDLE` means the source is available with no active capture, and `SUSPENDED` means the sound server released the device after it sat unused. It says nothing about mute. A source can be `RUNNING` and muted, or `SUSPENDED` and unmuted. Mute and volume are separate controls, and a source whose server entry reports neither keeps those controls disabled.

Each `pactl` call has a five-second deadline covering process exit and output collection. A timeout requests termination without blocking the caller on cleanup. A timed-out mute or volume write has an unknown outcome and is never automatically retried; refresh the source before trying again.

### Local browser remote

```sh
$APP serve                         # loopback, fresh random token, prints URL
$APP serve --listen 0.0.0.0:9360 --token YOUR_RANDOM_TOKEN
```

The server starts a local capture for firmware controls. It serves camera controls, not a video stream. For LAN use, open the printed URL with the computer's LAN address instead of `0.0.0.0`. LAN binding requires an explicit token. Keep this off the public internet: it is plain HTTP, so the token is not encrypted on the network.

The API uses bearer authentication: `GET /api/status` and JSON `POST /api/set` with `{"key":"brightness","value":"55"}`. Cross-origin mutations and oversized/invalid bodies are rejected. Each connection carries one request, which must arrive within 10 seconds; at most 32 connections are served at once, so a slow or stalled client loses only its own connection. This is the app's own API, not a promise of compatibility with Insta360's remote protocol.

## Local vision models

Eight models run on this computer. Nothing is downloaded until you ask for it, and camera frames never leave the machine. There is no server, URL, token or HTTP client for inference.

### Runtime and install

Link Camera embeds a small Python worker. The worker runs in a persistent, app-owned Python 3.11 virtualenv under `$XDG_DATA_HOME/link-camera/local-ai` (default `~/.local/share/link-camera/local-ai`). The Rust side orchestrates it; inference itself is Python, not pure Rust.

- The first `models download` (or the GUI **Download**) builds that runtime and then fetches the chosen model's weights. The runtime installs pinned packages including PyTorch with its CUDA libraries, so the first download is several gigabytes. Later models reuse it and download only their own weights.
- Python 3.11 comes from `uv` (which fetches its own managed 3.11 when needed), from `python3.11` on `PATH`, or from an interpreter you prepared and name in `LINK_CAMERA_AI_PYTHON` (absolute path). `uv` is not installed for you.
- The installer chooses its own Python and ignores inherited `UV_PYTHON_PREFERENCE`, `UV_MANAGED_PYTHON` and `UV_NO_MANAGED_PYTHON` only in its subprocesses; your shell configuration is unchanged.
- System libraries: `libseccomp2` (required) and, for MediaPipe, `libegl1` and `libgles2`.
- The Linux runtime needs glibc 2.28 or newer. PyTorch 2.13, torchvision 0.28 and ONNX Runtime 1.28 share CUDA 13 and cuDNN 9 libraries; GPU execution requires NVIDIA driver 580.65.06 or newer and a Turing-or-newer GPU. The installer does not change your GPU driver. Older hardware can use the models that support CPU inference; Marigold still requires CUDA.
- Only an explicit download uses the network. `models list`, `run` and `watch` start no download. Inference runs with Hugging Face offline mode set and a seccomp filter that makes creating internet (IPv4/IPv6) and raw-packet sockets fail, including from native libraries such as MediaPipe's usage reporting. Unix sockets stay allowed for the GPU stack. If the filter cannot be installed and verified, the worker does not start. Downloads and deletes touch only the app's own model directories, never another application's model cache.
- MediaPipe Hand/Pose/Face run on the CPU. The other models use an NVIDIA GPU through CUDA when it works and otherwise the CPU, except Marigold, which requires CUDA and refuses to fall back. Depth Anything, SAM 2.1, Florence-2 and Marigold are heavy: on CPU, or on a modest GPU, a frame can take seconds, and live analysis is not guaranteed to run in real time.
- One model worker is resident per local model store, with one inference in flight on the newest frame; no frame queue builds up. Another Link Camera window or CLI cannot spawn a duplicate resident worker in that store. Results and the device used are those the worker reports.

Download sizes, licenses and sources are listed by `models list` and in the GUI before you confirm. The MediaPipe bundles, Depth Anything V2 Small and SAM 2.1 Tiny are Apache-2.0, Florence-2 Base is MIT, Marigold Depth v1.1 is CreativeML OpenRAIL++-M (use restrictions apply) and YOLO11n is AGPL-3.0 (Ultralytics; commercial terms apply for proprietary use). **Confirm download** in the GUI and `models download` on the command line are the explicit steps that fetch a model under the listed license; read it first.

| Model | Native result support | Options |
|---|---|---|
| `mediapipe-hand` | 21-point hands, handedness, scores and skeletons | `max_results`, `min_confidence` |
| `mediapipe-pose` | 33-point body skeletons, visibility and world landmarks | `max_results`, `min_confidence` |
| `mediapipe-face` | 478 face landmarks, expression scores and head-pose matrix | `max_results`, `min_confidence`, `blendshapes` |
| `depth-anything-v2-small` | PNG relative-depth visualization | `colormap`: `gray` or `inferno` |
| `marigold-depth-v1-1` | PNG relative-depth visualization | `steps` 1–10, `ensemble` 1–5, `colormap` |
| `sam2.1-tiny` | Up to three scored polygon outlines, boxes and area fractions | Up to 16 positive `points`, `box`, `multimask` |
| `florence-2-base` | Captions, detection boxes, OCR quadrilaterals and candidate regions | `task`: `caption`, `detect`, `ocr`, `regions` |
| `yolo11n` | Labelled/scored object boxes | `threshold` 0–1, `top_k` 1–300 |

Depth is **relative**, normalized independently for each image, not metres. MediaPipe's hand labels use its selfie convention; pose/face outputs are geometry, not person identification or reliable emotion classification. SAM returns a simplified outline of the largest connected region per mask, not a full-resolution alpha mask. The app does not invent missing mask pixels.

Inference is single-image for MediaPipe, not its stateful VIDEO/LIVE_STREAM tracking API. Face results contain the ten strongest non-neutral blendshapes and the head-pose matrix, not all 52 coefficients. Handedness is preserved as reported; check source mirroring before attaching meaning to Left/Right. Depth models return normalized PNGs, not raw floating-point depth tensors.

### GUI

Open **AI tools** after enabling AI in Settings. **Model** explains the selected model's purpose, best uses, outputs, performance and limitations, alongside its install state, size, license and device. Use **Download** then **Confirm download**, or **Delete** then **Confirm delete**. The refresh button rescans the local library; source and file details can be expanded.

With a model installed, open **Analyze**, configure its options and run once on the current preview or enable live analysis at the chosen interval. Availability and normal progress each have a permanently reserved two-line viewport, including when empty, so starting/completing a run does not move the controls below. Longer normal messages can scroll; actual errors wrap in full. Each result stays paired with the exact submitted frame: slow captions or masks never appear over a newer, unrelated live frame. The camera preview continues independently.

**Stop** ends live analysis, cancels current work and unloads this window's model, even if it is already idle. It returns to the view selected before analysis unless you explicitly chose another view meanwhile. It does not start a stopped camera preview or delete completed history. Run/Live activation is blocked while a request starts, runs or stops; Run once is also disabled while Live is on. **Stopping…** remains until cleanup is confirmed, after which a fresh run can start.

**History** retains the latest **100 completed results total, across all models**, newest first. **Browse**, **Render** and **Raw** each use the available reading space. Select an entry for its exact source-matched render, summary and full raw response; the old 200 KB raw-text limit is gone. Opening Raw, focusing the summary/raw reader, selecting an older result or scrolling back through Browse turns off **Follow latest** without stopping inference. New results preserve the scrolled list position. Turn Follow latest back on to resume updating the selected view. A selected result stays readable even after it leaves the latest 100.

In Render, save the selected result's raw JSON/PNG or annotated PNG; existing files are never overwritten. **Clear** requires confirmation, releases the history buffer and clears the selected view without deleting exported files or changing the Follow latest choice. New results can still arrive while live analysis is running.

History is held only in this session's RAM, not restored on the next launch. Its memory readout counts compressed source frames and raw output bytes, not model/GPU memory or decoded display frames. Full-resolution frames are compressed losslessly; noisy 4K frames can still make 100 entries use several GB. Compression and restoring an older render run off the UI thread. Disabling AI or closing the app discards the history.

SAM point selection uses a frozen source image, not a moving target. Points and boxes are pixel coordinates of that image; manual entry is also available. Camera, capture-geometry and effects changes discard obsolete frozen input. These changes and model changes cancel obsolete analysis, but keep completed history tied to its original source frames. Analysis images are inspection/export results; they do not silently replace recordings, snapshots or virtual-camera output.

### CLI

```sh
$APP settings ai true                           # required first
$APP models list --json                         # state, size, license; no camera needed
$APP models download yolo11n                    # first download also builds the runtime; Ctrl-C cancels
$APP models delete yolo11n                      # works with AI off; removes only this app's files
$APP models run mediapipe-hand --output hands.json
$APP models watch mediapipe-pose --interval 0.5 --duration 10
$APP models run mediapipe-face --input portrait.jpg \
  --options '{"blendshapes":true}' --output face.json --visualization face.png
$APP models run depth-anything-v2-small --input photo.jpg --output depth.png
$APP models run marigold-depth-v1-1 --input photo.jpg \
  --options '{"steps":4,"ensemble":1,"colormap":"inferno"}' --output detailed-depth.png
$APP models run sam2.1-tiny --input photo.jpg \
  --options '{"points":["320,240"],"box":"100,80,500,400","multimask":true}' \
  --output masks.json --visualization outlines.png
$APP models run florence-2-base --input photo.jpg --options '{"task":"ocr"}'
$APP models run yolo11n --input photo.jpg --options '{"threshold":0.4}' \
  --output detections.json --visualization boxes.png
$APP models watch depth-anything-v2-small --duration 10 --output-dir depth-frames
```

Model ids are `mediapipe-hand`, `mediapipe-pose`, `mediapipe-face`, `depth-anything-v2-small`, `marigold-depth-v1-1`, `sam2.1-tiny`, `florence-2-base` and `yolo11n`. Without `--input`, `run` captures one camera frame; `watch` owns one capture session and samples only new frames. Stop another application's preview before CLI capture. `--input` uses bounded PNG/JPEG files, applies EXIF orientation and composites transparency onto white; it never opens the camera. JSON models print raw JSON when no output file is supplied. Depth models require a PNG output path, or an output directory for `watch`. Watch emits NDJSON; `--visualization` alongside `--output-dir` also saves annotated PNGs.

`--options` is a JSON object merged over that model's defaults; unknown keys, wrong types and out-of-image SAM coordinates fail before inference. `--timeout` (1-3600 seconds, default 300) bounds one inference. The commands have no URL or token argument. The optional camera remote-control `serve` command above is separate and unrelated to models.

### Scheduling, stop and private data

There is **one inference in flight per analysis engine/CLI process**, no growing frame queue, no automatic retry and no background inference before an explicit action. An exclusive resident-worker lease also prevents separate windows/CLI processes sharing the same model store from loading a second worker. The owning window's Stop or the CLI's cancellation releases it; another window never kills that owner. Stop/close the owning window or stop its CLI command, then use **Refresh** in the waiting window. Separate model stores are independent. Live sampling is inference-limited, not a promise of camera-rate tracking. Errors stop live submission rather than silently substituting raw results or another model.

When upgrading from an older version, close its windows and CLI model sessions before launching this build: older workers do not participate in the new resident-worker lease.

Cancelling an active run (Ctrl-C or Stop) terminates the owned worker process group, so a cancelled run publishes no output. In the GUI, Stop also unloads an idle resident model and restores the prior stage without deleting completed history. Disabling AI or closing the GUI unloads it and discards history. Cleanup runs off the UI thread; a genuinely kernel-stalled child retains its ownership lease and stopping state until it exits, rather than permitting a duplicate. Downloads are published only after they complete: an interrupted download leaves no half-installed model, and `.installed.json` is written last. Frames and history are held in memory and are not written unless you save an export.

The local-library, model and runtime state is under `$XDG_DATA_HOME/link-camera/local-ai`. Delete a model from the GUI or with `models delete`; removing the whole directory also removes the runtime.

## Data and limits

- Application settings (optional AI, last capture format, last effects): `$XDG_CONFIG_HOME/link-camera/settings.json` (default `~/.config/link-camera/settings.json`).
- Local AI runtime and downloaded models: `$XDG_DATA_HOME/link-camera/local-ai` (default `~/.local/share/link-camera/local-ai`).
- Virtual camera ownership: `$XDG_STATE_HOME/link-camera/virtual-camera/`.
- Saved per-camera hardware settings: `$XDG_STATE_HOME/link-camera/cameras` (default `~/.local/state/link-camera/cameras`), keyed by the camera's USB identity, so a changed `/dev/videoN` number keeps them.
- Camera profiles: `$XDG_STATE_HOME/link-camera/profiles`. These are named hardware-state profiles you apply explicitly; CLI effects files are separate.
- Teleprompter: `$XDG_CONFIG_HOME/link-camera/teleprompter.txt`.
- Recordings use private temporary segments so paused intervals are omitted. Failed finalization retains recovery segments and reports their directory instead of deleting them.
- No cloud AI recording, account, transcription or summary workflows are included.
- This is a working local controller, **not a certified 1:1 replacement** for every vendor feature. Proprietary makeup/depth-bokeh algorithms, exposure curves, firmware flashing, reboot-only compatibility switches, system-wide hotkeys and tray integration are not implemented. No controls pretend those operations succeeded.
- Marker-based whiteboard/DeskView/framing behavior depends on camera firmware and the scene; those visual outcomes have not all been compared with the Windows reference.

## Verification

Both Slint and no-default-features release builds were exercised. The suite covers parser/range/auth/geometry and processing boundaries. A hardware-only regression is ignored by default:

```sh
cargo test --release --locked
cargo test --release --locked --no-default-features
cargo test --release --locked effects_acknowledge_a_processed_frame -- --ignored
```

The last command requires an idle live camera (`LINK_CAMERA_TEST_DEVICE`, default `/dev/video0`); it changes capture format but not picture controls.

The 0.1 controller checks on the connected Link 2C covered discovery, advertised formats, zoom/HDR/Auto Framing write/readback, profiles, 4K stills, local person blur/replacement, overlays, video/audio recording, GUI controls, pause/resume, close-during-recording and token-authenticated browser control. That task restored its stored-setting changes from an exported state.

For 0.2, all eight models ran through the compiled CLI against the real local Hearrr API. Positive cases returned two 21-point hands, a 33-point pose with 33 world landmarks, and a 478-point face with ten expression scores and a 4×4 pose matrix. All four Florence tasks, SAM point/box/multimask requests, both depth formats and optional face outputs were exercised. These used the [official hand sample](https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb) and [official face sample](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb), plus live camera frames and Hearrr's sample image. External test photos are not bundled.

0.4.0 release checks (the verification notes for 0.2 to 0.4 record the earlier Hearrr-server integration that 0.5.0 replaced):

| Release configuration | Tests passed | Hardware-only tests ignored |
|---|---:|---:|
| Slint GUI | 78 | 1 |
| CLI only | 69 | 1 |

The suites overlap; their totals must not be added. Both release builds and formatting checks passed. Real file-inference cancellation drained and deleted its job without publishing output; invalid options, out-of-image prompts, nonlocal URLs and an offline API failed visibly. A deadline regression failed before the fix when a 1.45-second result escaped a one-second budget; it passes with shared upload/poll/download budgeting. A real one-second Marigold ensemble request also timed out with a retained-job warning; the owned test job was removed after it finished, with no misleading output created. All pre-existing server jobs were preserved and test jobs were removed.

The actual Slint surface was exercised at 1280×800 and 900×640: all eight option forms, offline recovery, live YOLO, frame-matched results, native JSON/PNG export and frozen-frame SAM point/box picking. Letterbox clicks were rejected. Closing during a real catalog request with deliberately injected latency kept the window responsive and closed after the request drained.

For 0.2.1, a failing-before/passing-after regression proves that a USB transport timeout aborts the snapshot after its first failed query. An actual 1280×800 native-window check stopped only the owned camera-control subprocess: the 15-second deadline cleared “Working…”, the Effects tab remained interactive, manual device refresh reaped/replaced the dead worker, and a normal window-close request exited successfully. The private worker also returned a missing-device error without retaining a stale camera snapshot; CLI discovery/status reported the absent device accurately.

For 0.3.0, the opt-in and saving behavior was exercised on the connected Link 2C with a local HTTP recorder in front of the real Hearrr API. A fresh profile started with AI off; the blocked `models list` produced no request at all, corrupt settings failed closed, and no analysis thread existed. Enabling in the native Settings tab persisted without connecting, and an explicit connection then listed all eight models. Disabling during a real in-flight catalog request kept the window responsive, drained the owned worker and left no job. Neural blur was refused while off, and green-screen keying stayed available.

A CLI `set brightness` wrote a private 0600 per-camera file; after an external `v4l2-ctl` change, `restore apply` reported `waiting-for-stream` and reapplied the saved value, and a GUI restart did the same. A corrupt saved file blocked the next write and left the hardware untouched. With the helper running and no preview, an independent `ffmpeg` capture of 436 frames triggered restoration within ten seconds while that application kept streaming, and the camera was released afterwards. `restore disable` survived a later camera change until `restore enable`. The generated user service passed `systemd-analyze --user verify`. Choosing 1920×1080 and Mirror in the real window brought both back on the next launch. The camera's original settings were restored from an exported state afterwards.

For 0.4.0, the redesigned native interface was exercised at 1440×900, 1280×800 and 900×640 on a private X11 display. Checks covered every section, Tab/Space/Enter navigation, numeric commit when leaving a section, AI opt-in, real discovery of all eight models, all eight option forms at desktop and compact sizes, draft background selection with independent mirror changes, script persistence and frozen-frame point picking. Real H.264 recordings finalized after pause/resume/Stop and after window close; temporary footage was deleted. Compact form wrapping and full error details were corrected from the actual surface.

The camera stopped responding again late in this run: capture startup failed and a bounded attempt to restore the test brightness from 52 to its original 51 timed out. Hardware readback is therefore unknown. The user's saved brightness remains 51 and privacy remains unchanged; no USB reset was issued. Final layout checks used an explicitly missing-device launch, not a simulated live preview. All eight inference jobs, Wayland and fractional scaling were not rerun for this UI release.

0.5.0 local-runtime checks:

| Release configuration | Tests passed | Hardware-only tests ignored |
|---|---:|---:|
| Slint GUI | 90 | 1 |
| CLI only | 79 | 1 |

The suites overlap. Both release builds, formatting and Python compilation passed. Real on-machine inference passed for all eight models, including 21-point hands, 33-point pose/world coordinates, 478-point face geometry and matrix, person detection, variable relative-depth PNGs, SAM outlines and a caption. Additional requests exercised face blendshapes off, both depth colour maps, Marigold ensembling, SAM point/box/multimask and Florence detection/OCR/regions. A generated "HELLO 42" image produced a positive OCR result.

These are single warmed second calls on a 640×960 public sample, measured through the Rust client, including frame/result transfer. They are not averages or camera/render FPS; first loads take longer.

| Model | Device | Warmed call (ms) |
|---|---|---:|
| MediaPipe Hand | CPU | 17.5 |
| MediaPipe Pose | CPU | 14.5 |
| MediaPipe Face | CPU | 7.2 |
| Depth Anything V2 Small | RTX 3090 | 34.6 |
| Marigold Depth v1.1 | RTX 3090 | 337.1 |
| SAM 2.1 Tiny | RTX 3090 | 51.7 |
| Florence-2 Base (caption) | RTX 3090 | 237.6 |
| YOLO11n | RTX 3090 | 15.8 |

Cancellation of an actual warmed CUDA Marigold request returned in 383 ms and unloaded its worker. A real one-second model-startup deadline returned an error with no late result. Native X11 checks covered seven pages and all eight forms at 1280×800 and 900×640, slider click, drag, keyboard and wheel actions, exact 33 ms default reset on a 10 ms grid, switch click/Space, preference restoration, AI gating, library refresh after CLI installation, and confirmed download/delete. Deleting Hand preserved the other seven models and settings.

The Link 2C microphone reported its exact USB-matched source, suspended state, mute and 100% volume. A 99%/mute readback exercise restored the original source state. Regression tests cover automatic restoration not fighting firmware framing, owned virtual-producer shutdown, PID identity and model locks retained through duplicated descriptors.

This host has neither the v4l2loopback module nor `v4l2loopback-ctl`: prerequisite reporting and stop-all safety paths were checked, but no real virtual node or Teams video consumer was exercised. Live camera capture/tracking, Wayland, fractional scaling, physical touchpads and screen readers were not rechecked. No USB reset, privacy override or unrelated-process termination was issued. Evidence: `.local/verification/local-ai/summary.json`.

For 0.6.0, native private-X11 checks at 1280×800 and 900×640 used explicitly generated frames, not camera or neural-model inference. They covered all eight model guides, mixed-model history, the 100-entry bound, pinned selection after eviction, complete valid JSON over 220 KB, unchanged scrolled-list content after five prepends, follow/resume, focus/scroll pausing, model changes, selected-frame JSON/PNG exports, confirmed Clear and AI-off cleanup. Empty, short and wrapped normal status changes altered only their reserved text region; a multiline error displayed its final recovery instruction. Real app/camera preferences were unchanged. The history codec tests restore source frames, raw outputs and annotations for all eight result types. Evidence: `.local/verification/history/summary.json`. No camera capture, new neural-inference benchmark, Wayland or screen-reader check was performed for this UI change.

For 0.6.1, private-X11 checks used real MediaPipe Hand CPU inference on generated frames in a private copy of the model store. They verified cold-start cancellation, idle and live worker termination/reaping, normal and frozen prior-view restoration, no late refreeze, rejected duplicate Run/Live actions, same-store CLI rejection and successful CLI inference after Stop released ownership. At 1280×800 and 900×640, empty/analyzed/running/stopping text changed only the two reserved status viewports; every lower control remained pixel-identical. Tests passed 104 with GUI and 88 headless, with one hardware-only ignore each. Real app/camera preferences were unchanged. No physical camera capture, other-model/GPU benchmark, Wayland or screen-reader check was made. Evidence: `.local/verification/ai-stop/summary.json`.

**Hardware history and remaining limits:** capture and USB enumeration failed during the longer 0.2/0.2.1 runs, with repeated kernel USB timeouts and a blocked first UVC `ioctl`. No USB reset or privacy override was issued. The camera was available again for 0.3 and 0.4 live verification; the underlying cable/port/camera cause was not established. The previously blocked final CLI-watch/SAM-result/depth-GUI rechecks were not all repeated.

Measured evidence is in `.local/verification/models/summary.json`, `.local/verification/camera-io/summary.json`, `.local/verification/preferences/summary.json` and `.local/verification/studio/summary.json`. Temporary camera frames, raw inference payloads and test footage were removed; retained screenshots contain only controls, redacted preview regions or the explicit no-camera test surface.

## License and references

Application: GPL-3.0-only. Slint and the bundled RVM model are used under GPL-compatible terms. Camera control uses the MIT-licensed [`sglre6355/insta360-link`](https://github.com/sglre6355/insta360-link), pinned in `Cargo.toml`/`Cargo.lock`. See [third-party notices](THIRD_PARTY.md).

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md), not a public issue.

The Windows installer, original binaries and extracted QML are private analysis references and are not in this repository. The original has not been executed or modified.
