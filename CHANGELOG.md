# Changelog

## 0.6.1 - 2026-11-02

- Fixed the remaining analysis-sidebar jump: availability text above the frame status also needs a permanently reserved viewport. Both normal message areas now keep their height, including when empty.
- AI Stop now asynchronously terminates the owned model worker, including an idle loaded model, and restores the stage selected before analysis. Explicit view changes take precedence; Stop never starts camera capture or deletes result history. Late inference/history-render completions cannot re-freeze the restored view.
- Blocked redundant Run/Live activation during startup, active inference, live sessions and cleanup. Stop stays usable while work is active; the UI reports Stopping until cleanup and catalog state are current.
- Added exclusive resident-worker ownership per local model store. Another window or CLI fails clearly before spawning a duplicate worker; Stop/death/drop releases ownership only after real exit. No unrelated processes are stopped.
- The managed Python runtime pins Pillow 13.0.0 instead of 11.3.0, which fixes the Pillow advisories against 11.3.0, including CVE-2026-112390. An existing runtime reports itself missing until a model download builds the new one; the old runtime directory is not deleted automatically.
- `serve` no longer uses tiny_http. A request that announced a body and never sent it could hold every request worker, and a huge announced length could abort the server, both without the token. Each connection now carries one request that must arrive within 10 seconds, with at most 32 connections at a time.
- Virtual camera removal checks each node's card label again inside the privileged step itself, so a `/dev/videoN` number that another application received while the administrator prompt was open is left alone.
- Virtual camera listing and removal now snapshot sysfs while holding the ownership-state lock, preserving cameras registered by another instance while a listing waits.
- Recording starts each FFmpeg segment on its first frame. Immediate pause/resume/stop sequences no longer add empty segments that can prevent an otherwise valid recording from being saved.
- Updated torch to 2.13.0 and its matching torchvision 0.28.0 and ONNX Runtime 1.28.0 to address CVE-2025-3000 and CVE-2025-3001. The shared GPU runtime is now CUDA 13, requiring NVIDIA driver 580.65.06+ and Turing-or-newer hardware.
- Virtual-camera producer shutdown uses libc's architecture-aware pidfd syscall constants instead of hard-coded Linux syscall numbers.
- Capture explicitly fixes the raw RGB output size, keeping frame boundaries aligned when the camera negotiates a different input resolution.
- Documented the remaining v4l2loopback deletion limit: its number-only ioctl cannot atomically verify identity, so other administrative tools must not add/remove devices concurrently with this app's deletion.
- Helper stdout and stderr are drained concurrently with bounded diagnostics, preventing large removal errors from filling a pipe and stalling the operation.
- Capture handles media controls between timed socket reads, so pause/stop and output shutdown remain responsive without another camera frame; partial frames and processed-frame acknowledgement are preserved.
- Microphone helper exit and output collection now share one deadline; inherited pipes cannot keep discovery blocked after the helper exits. Timed-out children are reaped off the caller thread without retrying writes.
- FFmpeg output cleanup uses safe child ownership instead of `ManuallyDrop`, while retaining virtual-camera reservations until the producer has actually exited.
- Background images are centre-cropped before resizing, preventing extreme aspect ratios from creating enormous intermediate images.
- Virtual-camera removal reserves its target nodes under the shared state lock before stopping producers, preventing another instance from claiming them during authorization or deletion. A cancelled startup must release its producer claim before deletion; otherwise the node is left installed with a cleanup error. Finished operations release reservations; stale process identities are pruned.
- The restoration watcher reports per-device discovery and setup failures without exiting, so a camera unplugged between enumeration and identity lookup cannot stop restoration for the remaining cameras.

## 0.6.0 - 2026-11-02

- Split the AI inspector into Model, Analyze and History. Each model explains its purpose, useful cases, outputs, performance requirements and limitations; source/file details remain available separately.
- Keep the latest 100 completed analyses across all models in a session-only history. Full raw outputs and losslessly compressed source frames are retained; only the selected old result is decoded and rendered, off the UI thread. The history shows its compressed-buffer memory use.
- Added full-height Browse, Render and Raw views, Follow latest, scroll-position preservation and confirmed Clear. Opening Raw, focusing a result or browsing older entries pauses following without stopping inference. A pinned result survives eviction from the recent 100, and exports use that selected result.
- Removed the raw viewer's 200 KB truncation. Model, camera, capture-format and effects changes cancel obsolete requests without discarding completed history. Disabling AI or closing the app discards history; saved exports remain untouched.
- Fixed normal analysis-status updates moving the sidebar. The status has a fixed two-line viewport, including when empty; errors instead wrap in full.

## 0.5.1 - 2026-11-02

- Fixed Python runtime creation when inherited uv Python-selection variables conflict with the app's managed-Python requirement. The installer clears only `UV_PYTHON_PREFERENCE`, `UV_MANAGED_PYTHON` and `UV_NO_MANAGED_PYTHON` in its subprocesses. This fixes the shared download setup for all eight models without changing shell configuration, camera settings or existing model files.

## 0.5.0 — 2026-11-02

- Replaced the Hearrr server integration with fully local inference: an embedded Python 3.11 worker in an app-owned runtime under `$XDG_DATA_HOME/link-camera/local-ai`, with explicit download and delete for MediaPipe Hand/Pose/Face, Depth Anything V2 Small, Marigold Depth v1.1, SAM 2.1 Tiny, Florence-2 Base and YOLO11n. Inference runs offline behind a seccomp network filter. The first download installs several GB of PyTorch/CUDA packages; MediaPipe uses the CPU, Marigold requires CUDA and the other models use CUDA when available.
- Added `models download` and `models delete`; `models list/run/watch` now work on the local library. Removed the AI server options (`--url`, `LINK_CAMERA_MODELS_URL`, `LINK_CAMERA_MODELS_TOKEN`). `--timeout` now bounds one local inference. `models delete` works with AI disabled.
- The AI tab shows model state, size, license and device, confirms downloads and deletions, and has a manual library refresh.
- Added `virtual-camera status|list|create|stop|stop-all|remove|remove-all` for app-owned v4l2loopback nodes. A step that needs root runs through pkexec only with an explicit `--authorize`, and otherwise fails with the exact command; the module is never unloaded and other nodes are never touched. Created nodes use `exclusive_caps=1`, so consumers see them only while a producer is active.
- Added `mic sources|status|mute|volume` and GUI controls. The camera's microphone is matched only by its USB path; the sound server's `RUNNING`/`IDLE`/`SUSPENDED` state is shown apart from mute.
- Link 2C tracking is reported honestly as Auto Framing with firmware state; saved pan/tilt/zoom is not replayed over camera-driven modes.
- Sliders respond to the wheel, keys, double-click and a reset button using the camera-reported default (reset is unavailable when none is reported); a plain click commits after 500 ms; toggles are switches.

The v4l2loopback module and `v4l2loopback-ctl` were absent on the development host, so no third-party consumption of a virtual camera has been verified.

## 0.4.0 — 2026-11-02

- Redesigned the desktop around a warm light studio palette, rounded surfaces, authored line icons and a labelled navigation rail, inspired by the supplied visual reference.
- Kept the image stage and capture actions visible beside seven task-specific sections; moved local profiles into Views.
- Added keyboard-operable custom buttons, focus-before-navigation numeric commits and persistent inspector state.
- Improved compact recording actions, full error details, stopped-camera hand-off messaging and explicit live/analysis/frozen stage titles.
- Kept AI opt-in, saved camera setup and the shared CLI unchanged; independent image effects remain usable while a background image choice is incomplete.
- Prevented overlapping AI exports and stopped teleprompter advancement over a frozen picking frame.

## 0.3.0 — 2026-11-02

- Optional AI (Hearrr models and neural person blur/replacement) is now off until it is enabled in the Settings tab or with `settings ai`. While off nothing is contacted and no model loads; turning it off drains a job already running instead of abandoning it.
- Camera settings are saved per physical camera from the actual readback and restored when the camera is used again, including by other applications. A user service reapplies them without opening a video stream.
- Added `settings show`, `settings ai <true|false>` and `restore apply|enable|disable`.
- The app now reopens with the capture format and local effects it was last used with.
- Writes that the camera confirms but cannot be saved are reported as applied-but-not-saved and are never replayed; saving and automatic restoration share one per-camera lock.

Privacy mode is never reopened automatically. Camera settings reach other applications; this app's own effects still need the virtual camera.

## 0.2.1 — 2026-11-01

- Abort camera snapshots on a lost USB transport instead of multiplying driver timeouts across every control; preserve optional-selector handling.
- Isolate desktop controls in an owned subprocess group with a 15-second reply deadline. Quarantine timed-out workers until they exit, discard stale sessions and report unknown write outcomes without replaying operations.
- Clear connection progress after preview startup fails and stop automatic control refresh after transport loss.
- Prevent overlapping reconnect/device-selection requests while controls are busy, while preserving access to Stop preview.
- Document physical USB recovery and the distinction between the desktop deadline and direct CLI driver waits.

## 0.2.0 — 2026-11-01

- Integrated eight local Hearrr models through a shared, bounded Rust client, typed result layer and single-flight analysis worker.
- Added native model-specific controls, live/one-shot analysis, frame-matched landmarks/boxes/outlines/depth, SAM source-frame point picking and raw/visual exports.
- Added headless `models list`, `models run` and NDJSON `models watch`, supporting image files or live camera frames.
- Restricted inference to loopback HTTP, with optional environment-only bearer authentication, bounded transfers, no automatic retries, owned-job cleanup and explicit timeout/retention warnings.
- Preserved normal capture/recording behavior while slow model jobs run independently.
- Enforced one deadline across upload, polling and result download; cleanup uses its own bounded budget. A failing-before/passing-after regression covers late-result publication.

Hearrr cannot abort queued/running inference; Stop drains the current job and discards unwanted results. Relative depth is not metric, and SAM's outline API is not a full alpha-mask API.

## 0.1.0 — 2026-11-01

- Added a native Slint GUI and CLI sharing the Linux camera backend.
- Added capability-based controls, advertised capture formats, local camera profiles, supported firmware views/zones and explicit unsupported-control errors.
- Added preview, snapshots, recording with optional local audio, pause/resume and safe close-time finalization.
- Added offline RVM person matting, background blur/replacement, green-screen keying, softening, mirror, image overlays and four-corner correction.
- Added existing-device virtual-camera output, token-authenticated local browser control and a persistent on-screen teleprompter.
- Added a live-camera regression for processed-frame acknowledgement; effects cannot acknowledge success while a snapshot would still read the preceding raw frame.

Cloud workflows and a blanket claim of vendor feature parity are not included. See README.md for verified behavior and prerequisites.
