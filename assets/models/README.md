# Person segmentation model

| | |
|---|---|
| File | `rvm_mobilenetv3_fp32.onnx` (14,975,696 bytes) |
| SHA-256 | `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828` |
| Model | Robust Video Matting, MobileNetV3 backbone, FP32, ONNX opset 12 |
| Source | <https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx> |
| Licence | GPL-3.0 (`RVM-LICENSE.txt`, copied from the upstream repository; the README there states the code and releases are GPL-3.0). Compatible with this application's GPL-3.0-only; anyone redistributing a binary with the model must also provide the corresponding source. |
| Paper | Shanchuan Lin et al., *Robust High-Resolution Video Matting with Temporal Guidance*, WACV 2022 |

Reproduce / repair:

```sh
curl -L -o rvm_mobilenetv3_fp32.onnx https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
echo "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828  rvm_mobilenetv3_fp32.onnx" | sha256sum -c -
```

The application looks for the file, in order, at `$LINK_CAMERA_RVM_MODEL` (exclusive when set),
`$XDG_DATA_HOME/link-camera/models/` (default `~/.local/share/link-camera/models/`), next to the
executable (`assets/models/`, `../share/link-camera/models/`, `../../assets/models/`), and this
directory. Every candidate is verified against the SHA-256 above before use; a missing or
mismatching model makes `blur`/`replace` fail with the list of searched paths and the commands
above. There is no silent fallback. Nothing is downloaded at run time.

## Interface (read from the model with onnxruntime)

Inputs: `src` `[1,3,H,W]` f32 RGB 0..1; `r1i..r4i` recurrent state, f32, start as zeros `[1,1,1,1]`;
`downsample_ratio` f32 `[1]`. Outputs: `fgr`, `pha` `[1,1,H,W]` alpha 0..1, `r1o..r4o` next state.
Any H/W works. The app runs it at the frame scaled to a 512 px long side with
`downsample_ratio = 0.5` (measured 7.6 ms per 512x288 frame, CPU, Python onnxruntime) and upsamples
`pha` bilinearly. The model matters only for people: on a frame without a person the alpha is
near zero and the whole frame counts as background.

## Runtime

`ort = 2.0.0-rc.14` with `download-binaries`: the build script fetches the pyke prebuilt ONNX
Runtime for the target once (cached in `~/.cache/ort.pyke.io`) and statically links it, so the
installed program needs no `libonnxruntime.so` or environment variables. Build-time needs: network
on first build, OpenSSL development files (`tls-native`). Run-time needs: x86-64 CPU with AVX2
(ort's prebuilt binaries), no network, no GPU. ONNX Runtime telemetry is disabled.

## Smoke test

There is no unit test that runs the model: a test that only checks buffer lengths would pass with a
no-op backend, and mask content depends on a person being in frame. Verify the real path end to end
after building: run the CLI `snapshot` with `--background blur` (or `replace`) against a live
camera with a person in view, and compare the written image with a plain snapshot. The background
must be blurred or replaced while the person stays sharp. A missing or mismatching model must fail
with the list of searched paths, never produce an unprocessed frame.
