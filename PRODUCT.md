# Link Camera

A local Linux controller for the Insta360 Link camera family, targeting the connected Link 2C. One Rust backend serves a Slint desktop app and command-line commands.

The user requested everything feasible locally: camera control, preview/capture/recording, local effects, working virtual camera output, microphone state/mute/volume, profiles, local remote control and teleprompter behavior. Cloud AI recording, cloud summaries, accounts and other service-backed workflows are excluded for now. Local replacements must not be presented as identical to proprietary algorithms.

Camera commands use the kernel's existing uvcvideo driver. No raw firmware flashing, driver detachment, automatic root operations or cloud inference. Unsupported controls and missing runtime prerequisites must report real errors, not success.

Local AI runs MediaPipe Hand/Pose/Face, Depth Anything V2 Small, Marigold Depth v1.1, SAM 2.1 Tiny, Florence-2 Base and YOLO11n on the user's machine. Rust orchestrates a persistent app-owned Python 3.11 worker; it is not pure-Rust ML. The runtime and each model are downloaded only by an explicit GUI confirmation or `models download`, into the app's own data directory, and can be deleted again. Inference is offline: the worker's environment is offline and a seccomp filter denies internet socket creation. AI is off by default and needs an explicit saved opt-in; the camera, capture and non-neural effects work without it. GUI and CLI share model contracts, results and scheduling: one resident model, one inference in flight on the newest frame. There is no inference server, URL, token or HTTP client, no cloud endpoint, no automatic camera action from an inference and no unrequested replacement of recorded/virtual video. MediaPipe runs on the CPU, the other models use CUDA where available, Marigold requires it, and the heavy models are not guaranteed to run in real time.

Virtual cameras are v4l2loopback nodes the app creates, starts, stops and removes itself. It validates recorded ownership before removal, never unloads the kernel module and never signals unrelated producer processes; administrator rights are requested only with explicit authorization. The driver's number-only deletion cannot be atomic against concurrent device replacement by other administrative tools. Microphone mute and volume use the sound server and never change the default source. Tracking on a fixed Link 2C is Auto Framing and is reported as firmware state, not as a physical gimbal.

The camera's own settings are the user's: they are saved per physical camera and restored when that camera is used again, including by other applications such as Teams. Restoration never opens a video stream, never reopens privacy mode and never replays a write whose outcome is unknown.

Effects reach other applications only through a virtual camera that is receiving frames; the camera's own hardware settings do not need one.

The extracted Windows installer, original resources and binaries under case/ are private reference evidence, not application assets. Local processing uses separately licensed components/models.
