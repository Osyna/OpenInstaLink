# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Osyna (Serverrr); full license in THIRD_PARTY.md.
# Modified for Link Camera: standalone downloads and resident local-worker inference.
"""YOLO11n object detection (official Ultralytics ONNX export) for the Camera local worker.

Adapted from Serverrr task_models/vision.py (standalone; no import from that project). The pinned yolo11n.onnx is
Ultralytics' own release asset (dynamic shape, no embedded NMS); pre/post-processing mirrors Ultralytics' predictor:
rectangular letterbox at 640, per-class NMS at IoU 0.7, boxes mapped back to the original image. download() is the
only network access; the session is kept resident, uses CUDA when onnxruntime can really run on it, else CPU.
"""
import ast
import hashlib
import json
import logging
import math
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"
_NAME, _SIZE = "yolo11n.onnx", 10930182
_SHA256 = "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b"
MODEL_IDS = ("yolo11n",)
_SIZE_PX, _STRIDE, _PAD, _IOU = 640, 32, 114, 0.7


def _check(model):
    if model not in MODEL_IDS:
        raise ValueError(f"unsupported model {model!r}")


def metadata(model):
    _check(model)
    return {"source": _URL, "revision": f"ultralytics/assets v8.4.0 sha256:{_SHA256}",
            "license": "AGPL-3.0 (Ultralytics; commercial terms apply for proprietary use)"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model, directory):
    _check(model)
    directory = Path(directory)
    part = directory / (_NAME + ".part")
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(_URL, timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if received > _SIZE:
                raise RuntimeError(f"{_NAME} is larger than the pinned {_SIZE} bytes")
            digest.update(chunk)
            out.write(chunk)
    if received != _SIZE or digest.hexdigest() != _SHA256:
        raise RuntimeError(f"{_NAME} does not match the pinned size/SHA-256")
    os.replace(part, directory / _NAME)


def _number(options, key, default, low, high, integer=False):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number")
    if integer:
        if value != int(value):
            raise ValueError(f"{key} must be an integer")
        value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def _letterbox_geometry(width, height):
    """(gain, left, top, new_w, new_h, canvas_w, canvas_h) as Ultralytics' LetterBox(auto=True)."""
    gain = min(_SIZE_PX / width, _SIZE_PX / height)
    new_w, new_h = max(1, round(width * gain)), max(1, round(height * gain))
    dw, dh = (_SIZE_PX - new_w) % _STRIDE / 2, (_SIZE_PX - new_h) % _STRIDE / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    right, bottom = round(dw + 0.1), round(dh + 0.1)
    return gain, left, top, new_w, new_h, new_w + left + right, new_h + top + bottom


def _nms(boxes, scores, classes, limit):
    """Class-aware greedy NMS on xyxy boxes; returns kept indices, best first, at most `limit`."""
    import numpy as np

    order = np.argsort(-scores, kind="stable")
    shifted = boxes + (classes.astype(np.float32) * 7680.0)[:, None]  # boxes of different classes never overlap
    areas = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    keep = []
    while order.size and len(keep) < limit:
        i, order = order[0], order[1:]
        keep.append(int(i))
        if not order.size:
            break
        ix1 = np.maximum(shifted[i, 0], shifted[order, 0])
        iy1 = np.maximum(shifted[i, 1], shifted[order, 1])
        ix2 = np.minimum(shifted[i, 2], shifted[order, 2])
        iy2 = np.minimum(shifted[i, 3], shifted[order, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        order = order[inter / (areas[i] + areas[order] - inter + 1e-9) <= _IOU]
    return keep


def _decode(raw, width, height, threshold, top_k):
    """raw: (84, N) YOLO output in letterbox pixels -> list of (class_id, score, x1, y1, x2, y2) in image pixels."""
    import numpy as np

    gain, left, top, *_ = _letterbox_geometry(width, height)
    rows = raw.T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), 4 + classes]
    mask = scores > threshold
    rows, classes, scores = rows[mask], classes[mask], scores[mask]
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
    kept = _nms(boxes, scores, classes, top_k)
    out = []
    for i in kept:
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = np.clip([(x1 - left) / gain, (x2 - left) / gain], 0, width)
        y1, y2 = np.clip([(y1 - top) / gain, (y2 - top) / gain], 0, height)
        out.append((int(classes[i]), float(scores[i]), float(x1), float(y1), float(x2), float(y2)))
    return out


def load(model, directory):
    _check(model)
    path = Path(directory) / _NAME
    if not path.is_file() or path.stat().st_size != _SIZE or _sha256(path) != _SHA256:
        raise RuntimeError(f"{_NAME} is missing or does not match its pinned SHA-256")
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.capi._pybind_state import (
        EngineError,
        EPFail,
        Fail,
        RuntimeException,
    )
    from onnxruntime.capi._pybind_state import (
        NotImplemented as OrtNotImplemented,
    )
    if hasattr(ort, "preload_dlls"):  # lets onnxruntime find pip-installed cuDNN/cuBLAS
        try:
            ort.preload_dlls()
        except (OSError, RuntimeError) as error:
            logger.warning("ONNX GPU library preloading failed: %s", str(error)[:600])
    from PIL import Image

    def session(providers):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        s = ort.InferenceSession(str(path), opts, providers=providers)
        s.run(None, {"images": np.zeros((1, 3, _SIZE_PX, _SIZE_PX), np.float32)})  # warm up; proves the provider runs
        return s

    sess = None
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            sess = session([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                            "CPUExecutionProvider"])
            if sess.get_providers()[0] != "CUDAExecutionProvider":
                sess = None  # onnxruntime silently fell back (e.g. missing cuDNN); report CPU honestly
        except (OSError, RuntimeError, EPFail, EngineError, Fail, OrtNotImplemented, RuntimeException) as error:
            logger.warning("ONNX CUDA initialization failed; using CPU: %s", str(error)[:600])
            sess = None
    if sess is None:
        sess = session(["CPUExecutionProvider"])
    device = "cuda" if sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
    names = {int(k): str(v) for k, v in ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"]).items()}
    if len(names) != 80:
        raise RuntimeError("yolo11n.onnx is not an 80-class COCO detector")

    def infer(image, options):
        unknown = set(options) - {"threshold", "top_k"}
        if unknown:
            raise ValueError(f"unsupported options: {', '.join(sorted(unknown))}")
        threshold = _number(options, "threshold", 0.25, 0.0, 1.0)
        top_k = _number(options, "top_k", 100, 1, 300, integer=True)
        width, height = image.size
        _, left, top, new_w, new_h, canvas_w, canvas_h = _letterbox_geometry(width, height)
        canvas = np.full((canvas_h, canvas_w, 3), _PAD, np.uint8)
        canvas[top:top + new_h, left:left + new_w] = np.asarray(image.resize((new_w, new_h), Image.BILINEAR))
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        raw = sess.run(None, {"images": tensor})[0][0]
        detections, counts = [], {}
        for class_id, score, x1, y1, x2, y2 in _decode(raw, width, height, threshold, top_k):
            label = names[class_id]
            counts[label] = counts.get(label, 0) + 1
            detections.append({"label": label, "class_id": class_id, "score": round(min(score, 1.0), 4),
                               "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1),
                                       "y2": round(y2, 1)}})
        text = ", ".join(f"{n} {label}" for label, n in counts.items()) or "no objects detected"
        payload = {"model": model, "text": text, "image": {"width": width, "height": height},
                   "detections": detections}
        return "application/json", json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()

    infer.device = device
    infer.close = lambda: None  # dropping the session releases it; nothing else is held
    return infer


if __name__ == "__main__":  # offline self-check of the geometry and NMS (no model file needed)
    import numpy as np

    assert _letterbox_geometry(1920, 1080) == (1 / 3, 0, 12, 640, 360, 640, 384)
    raw = np.zeros((84, 3), np.float32)
    raw[:4, 0] = (320, 192, 100, 100)  # person centred in the letterbox of a 1280x720 frame
    raw[4, 0] = 0.9
    raw[:4, 1] = (322, 192, 100, 100)  # near-duplicate: suppressed
    raw[4, 1] = 0.8
    raw[:4, 2] = (322, 192, 100, 100)  # same box, other class: kept
    raw[5, 2] = 0.7
    got = _decode(raw, 1280, 720, 0.25, 10)
    assert [g[0] for g in got] == [0, 1] and np.allclose(got[0][2:], (540, 260, 740, 460)), got
    assert len(_decode(raw, 1280, 720, 0.25, 1)) == 1 and not _decode(raw, 1280, 720, 0.95, 5)
    print("detection self-check ok")
