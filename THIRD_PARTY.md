# Third-party components

The application does not include the extracted Insta360 QML, DLLs or proprietary effect models as runtime assets.

## Camera control: insta360-link

Source: https://github.com/sglre6355/insta360-link

Pinned revision: `389e5a7ba028cc45f4277bafd1faa55a5cd12ea2`.

The app uses the library through its public typed API; unsafe raw camera access is not enabled.

```text
MIT License

Copyright (c) 2026 sglre6355

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Slint

Slint 1.18.1: https://github.com/slint-ui/slint . This application's licensing selects GPL-3.0-compatible use. Slint also offers other license options; this notice does not grant those licenses.

## Local person matting

Robust Video Matting, MobileNetV3 ONNX release: https://github.com/PeterL1n/RobustVideoMatting . The bundled weights are used under GPL-3.0. Full model license, source, checksum and runtime information are under [assets/models](assets/models/README.md).

Inference uses `ort` (MIT OR Apache-2.0) and the native ONNX Runtime (MIT): https://github.com/pykeio/ort and https://github.com/microsoft/onnxruntime .

## Local AI models and runtime

The Rust binary embeds the worker and adapter sources in `assets/local_ai/`. The eight optional model weights and their Python runtime packages are downloaded only on request into `$XDG_DATA_HOME/link-camera/local-ai` and retain their upstream licenses. They are not relicensed under the application's GPL-3.0-only license. The pins below are recorded in `landmarks.py`, `detection.py` and `scene.py`.

| Model | Source and pin | Upstream license |
|---|---|---|
| MediaPipe Hand, Pose (lite), Face | Google `float16/1` task bundles under `https://storage.googleapis.com/mediapipe-models/`, size and SHA-256 pinned | Apache-2.0, stated on the model cards for [hand tracking](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf), [BlazePose GHUM 3D](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), [Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), [Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf) and [BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf); [MediaPipe code](https://github.com/google-ai-edge/mediapipe) is Apache-2.0 |
| Depth Anything V2 Small | [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) at `5426e4f0f36572d16453bbda7a8389317b1bef99` | Apache-2.0 (model card); [code](https://github.com/DepthAnything/Depth-Anything-V2) Apache-2.0 |
| Marigold Depth v1.1 | [`prs-eth/marigold-depth-v1-1`](https://huggingface.co/prs-eth/marigold-depth-v1-1) at `9571e7123e258cf052b4e54241f17971c290e9a8`, fp16 safetensors | CreativeML OpenRAIL++-M weights, fine-tuned from Stable Diffusion 2, with use-based restrictions; [Marigold code](https://github.com/prs-eth/Marigold) is Apache-2.0 |
| SAM 2.1 Tiny | [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny) at `de431c4043854a71d8101e17995dfe596bf101a5` | Apache-2.0 (model card); [code](https://github.com/facebookresearch/sam2) Apache-2.0 |
| Florence-2 Base | [`florence-community/Florence-2-base`](https://huggingface.co/florence-community/Florence-2-base) at `00921df66db728a9ceb750f5eca43e5c203a2051` | MIT (model card, which links Microsoft's [Florence-2 license](https://huggingface.co/microsoft/Florence-2-large/resolve/main/LICENSE)) |
| YOLO11n | `yolo11n.onnx` from [`ultralytics/assets` release v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0), SHA-256 pinned | AGPL-3.0 ([repository license](https://github.com/ultralytics/assets/blob/main/LICENSE)); Ultralytics also sells an [Enterprise license](https://www.ultralytics.com/license) for proprietary use |

The Hugging Face license fields were read from the model cards at the pinned revisions. The MediaPipe licenses are the ones printed on the linked model cards. The Marigold card links its OpenRAIL++-M text at <https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL>, which answered an unauthenticated request with HTTP 401, so read it while signed in to Hugging Face. OpenRAIL++-M restricts how the weights may be used and requires those restrictions to be passed on when the weights are redistributed. YOLO11n's AGPL-3.0 obligations (source offer, and network-service terms if you host it) fall on whoever redistributes or serves the weights; Link Camera only downloads them to the user's own disk. Check the terms before commercial use or redistribution.

The downloaded Python runtime is separate software with its own licenses, installed from the pins in `assets/local_ai/requirements.txt`. PyPI metadata for the pinned versions declares: BSD for numpy, scipy, torch (BSD-3-Clause) and torchvision; MIT for onnxruntime-gpu; MIT-CMU for Pillow; Apache-2.0 for mediapipe, opencv-contrib-python, transformers, tokenizers, diffusers, safetensors and huggingface_hub. torch's Linux x86-64 wheels depend on the CUDA 13 toolkit, cuDNN and other NVIDIA runtime packages, which are under NVIDIA's license terms. The environment is created with `uv` (MIT or Apache-2.0) when it is installed, otherwise with `python3.11`; a uv-fetched CPython 3.11 is under the Python Software Foundation license. Each installed package keeps its own license files in the environment.

### Scene, landmark and detection adapters

`assets/local_ai/scene.py`, `landmarks.py` and `detection.py` are adapted from Serverrr's `task_models/scene.py`, `task_models/landmarks.py` and `task_models/vision.py` (https://github.com/Osyna/Serverrr, Copyright 2026 Osyna, [Apache License 2.0](#apache-license-20-serverrr)). The Serverrr tree checked for this notice was commit `ba2cf85a171d317e09736b6d6471113ca66b84b4`; it ships a `LICENSE` and no `NOTICE` file, so there are no further attribution notices to carry over. The adapters were modified: they are standalone (no Serverrr import), download into and load offline from Link Camera's own model directories, and their module docstrings say they are adapted. `scene.py` keeps Serverrr's pinned Hugging Face commits, options and result geometry. These files are distributed under GPL-3.0-only with the rest of the application, which Apache-2.0 permits because the two are compatible; the Apache terms and the text below stay with them.

### Apache License 2.0 (Serverrr)

Full text of the license distributed with Serverrr:

```text
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

     (a) You must give any other recipients of the Work or Derivative Works a copy of this License; and

     (b) You must cause any modified files to carry prominent notices stating that You changed the files; and

     (c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and

     (d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.

     You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

To apply the Apache License to your work, attach the following boilerplate notice, with the fields enclosed by brackets "[]" replaced with your own identifying information. (Don't include the brackets!)  The text should be enclosed in the appropriate comment syntax for the file format. We also recommend that a file or class name and description of purpose be included on the same "printed page" as the copyright notice for easier identification within third-party archives.

Copyright 2026 Osyna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Other dependencies

Cargo dependencies and exact revisions are recorded in `Cargo.toml` and `Cargo.lock`; retain their applicable notices when redistributing builds. FFmpeg and v4l-utils are separately installed system programs, not bundled executables.
