# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Osyna (Serverrr); full license in THIRD_PARTY.md.
# Modified for Link Camera: standalone downloads and resident local-worker inference.
"""MediaPipe Tasks hand / pose / face landmarkers for the Camera local worker.

Adapted from Serverrr task_models/landmarks.py (standalone; no import from that project). Official Google
`float16/1` task bundles, size + SHA-256 pinned. download() is the only network access; load() and inference are
offline on CPU. A landmarker is kept for the life of the returned callable and rebuilt only when its options change.
"""
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models/{0}/{1}/float16/1/{1}.task"
# model id -> (kind, url, file name, size, sha256)
_BUNDLES = {
    "mediapipe-hand": ("hand", _BASE.format("hand_landmarker", "hand_landmarker"), "hand_landmarker.task", 7819105,
                       "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    "mediapipe-pose": ("pose", _BASE.format("pose_landmarker", "pose_landmarker_lite"), "pose_landmarker_lite.task",
                       5777746, "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "mediapipe-face": ("face", _BASE.format("face_landmarker", "face_landmarker"), "face_landmarker.task", 3758596,
                       "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
}
MODEL_IDS = tuple(_BUNDLES)

_FINGERS = ("THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP") + tuple(
    f"{finger}_{joint}" for finger in ("INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY")
    for joint in ("MCP", "PIP", "DIP", "TIP"))
HAND_NAMES = ("WRIST",) + _FINGERS
POSE_NAMES = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
    "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")


def _bundle(model):
    try:
        return _BUNDLES[model]
    except KeyError:
        raise ValueError(f"unsupported model {model!r}") from None


def metadata(model):
    _, url, _, _, sha256 = _bundle(model)
    return {"source": url, "revision": f"float16/1 sha256:{sha256}",
            "license": "Apache-2.0 (Google MediaPipe; model bundle per its MediaPipe model card)"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model, directory):
    _, url, name, size, sha256 = _bundle(model)
    directory = Path(directory)
    part = directory / (name + ".part")
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if received > size:
                raise RuntimeError(f"{name} is larger than the pinned {size} bytes")
            digest.update(chunk)
            out.write(chunk)
    if received != size or digest.hexdigest() != sha256:
        raise RuntimeError(f"{name} does not match the pinned size/SHA-256")
    os.replace(part, directory / name)


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


def _prob(value):
    return min(1.0, max(0.0, float(value)))


def _points(landmarks, names, width, height, world=False):
    out = []
    for i, p in enumerate(landmarks):
        point = ({"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5)} if world else
                 {"x": round(p.x * width, 1), "y": round(p.y * height, 1), "z": round(p.z, 4)})
        if names:
            point["name"] = names[i]
        for attr in ("visibility", "presence"):  # pose only; None elsewhere
            if getattr(p, attr, None) is not None:
                point[attr] = round(_prob(getattr(p, attr)), 3)
        out.append(point)
    return out


def _count(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'} detected" if n else f"no {noun}s detected"


def load(model, directory):
    kind, _, name, size, sha256 = _bundle(model)
    path = Path(directory) / name
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != sha256:
        raise RuntimeError(f"{name} is missing or does not match its pinned SHA-256")
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker_class, options_class = {
        "hand": (vision.HandLandmarker, vision.HandLandmarkerOptions),
        "pose": (vision.PoseLandmarker, vision.PoseLandmarkerOptions),
        "face": (vision.FaceLandmarker, vision.FaceLandmarkerOptions)}[kind]
    allowed = {"max_results", "min_confidence"} | ({"blendshapes"} if kind == "face" else set())
    default_results = {"hand": 2, "pose": 1, "face": 1}[kind]
    base = BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU)
    state = {"key": None, "landmarker": None}

    def landmarker(max_results, confidence, blendshapes):
        key = (max_results, confidence, blendshapes)
        if state["key"] != key:  # creation parameters are immutable, so rebuild only on change
            close()
            settings = {
                "hand": {"num_hands": max_results, "min_hand_detection_confidence": confidence,
                         "min_hand_presence_confidence": confidence},
                "pose": {"num_poses": max_results, "min_pose_detection_confidence": confidence,
                         "min_pose_presence_confidence": confidence},
                "face": {"num_faces": max_results, "min_face_detection_confidence": confidence,
                         "min_face_presence_confidence": confidence, "output_face_blendshapes": blendshapes,
                         "output_facial_transformation_matrixes": blendshapes},
            }[kind]
            state["landmarker"] = landmarker_class.create_from_options(options_class(base_options=base, **settings))
            state["key"] = key
        return state["landmarker"]

    def close():
        if state["landmarker"] is not None:
            state["landmarker"].close()
        state["landmarker"] = state["key"] = None

    def infer(image, options):
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unsupported options: {', '.join(sorted(unknown))}")
        max_results = _number(options, "max_results", default_results, 1, 10, integer=True)
        confidence = _number(options, "min_confidence", 0.5, 0.1, 0.9)
        blendshapes = options.get("blendshapes", True) if kind == "face" else False
        if not isinstance(blendshapes, bool):
            raise TypeError("blendshapes must be true or false")
        width, height = image.size
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(image)))
        result = landmarker(max_results, confidence, blendshapes).detect(frame)

        payload = {"model": model, "image": {"width": width, "height": height}}
        if kind == "hand":
            hands = [{"handedness": handed[0].category_name, "score": round(_prob(handed[0].score), 4),
                      "landmarks": _points(points, HAND_NAMES, width, height)}
                     for points, handed in zip(result.hand_landmarks, result.handedness)]
            payload.update(text=_count(len(hands), "hand"), hands=hands)
        elif kind == "pose":
            poses = [{"landmarks": _points(points, POSE_NAMES, width, height),
                      "world_landmarks": _points(world, POSE_NAMES, width, height, world=True)}
                     for points, world in zip(result.pose_landmarks, result.pose_world_landmarks)]
            payload.update(text=_count(len(poses), "pose"), poses=poses)
        else:
            faces = []
            for i, points in enumerate(result.face_landmarks):
                face = {"landmarks": _points(points, None, width, height)}
                if blendshapes:
                    ranked = sorted((c for c in result.face_blendshapes[i] if c.category_name != "_neutral"),
                                    key=lambda c: -c.score)[:10]
                    face["blendshapes"] = [{"name": c.category_name, "score": round(_prob(c.score), 4)}
                                           for c in ranked]
                    face["transformation_matrix"] = np.round(result.facial_transformation_matrixes[i], 5).tolist()
                faces.append(face)
            payload.update(text=_count(len(faces), "face"), faces=faces)
        return "application/json", json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()

    infer.device = "cpu"
    infer.close = close
    return infer
