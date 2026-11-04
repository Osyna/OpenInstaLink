# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Osyna (Serverrr); full license in THIRD_PARTY.md.
# Modified for Link Camera: standalone downloads and resident local-worker inference.
"""Scene-understanding plugin for the Link Camera local AI worker.

Adapted (standalone, no serverrr import) from Serverrr task_models/scene.py: same pinned Hugging Face commits, same
options, same result geometry. Every checkpoint is safetensors only and loaded with trust_remote_code off.

  depth-anything-v2-small  relative depth PNG, near = bright (Apache-2.0)
  marigold-depth-v1-1      diffusion depth PNG, CUDA fp16 only (CreativeML OpenRAIL++-M)
  sam2.1-tiny              promptable segmentation: positive points / box / multimask -> outlines (Apache-2.0)
  florence-2-base          caption / detect / ocr / regions (MIT)

Contract: download(model, directory) is the only function that touches the network; load(model, directory) is offline.
"""
import io
import json
import shutil
from pathlib import Path

MAX_SIDE = 2048  # SAM 2 downscales to 1024 internally; larger inputs only cost memory
MAX_POINTS = 16
MAX_POLYGON_POINTS = 200
FLORENCE_TASKS = {"caption": "<MORE_DETAILED_CAPTION>", "detect": "<OD>", "ocr": "<OCR_WITH_REGION>",
                  "regions": "<REGION_PROPOSAL>"}

_OPENRAIL_USE = ("CreativeML OpenRAIL++-M; use restrictions apply, "
                 "see https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL")

# model id -> source repo, pinned commit, license, files fetched (safetensors + small configs only: no .bin/.pt)
_SPECS = {
    "depth-anything-v2-small": {
        "source": "depth-anything/Depth-Anything-V2-Small-hf",
        "revision": "5426e4f0f36572d16453bbda7a8389317b1bef99", "license": "Apache-2.0",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"]},
    "marigold-depth-v1-1": {
        "source": "prs-eth/marigold-depth-v1-1",
        "revision": "9571e7123e258cf052b4e54241f17971c290e9a8", "license": _OPENRAIL_USE,
        "files": ["model_index.json", "*/*.json", "*/*.txt", "*.fp16.safetensors"]},
    "sam2.1-tiny": {
        "source": "facebook/sam2.1-hiera-tiny",
        "revision": "de431c4043854a71d8101e17995dfe596bf101a5", "license": "Apache-2.0",
        "files": ["config.json", "preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json", "model.safetensors"]},
    "florence-2-base": {
        "source": "florence-community/Florence-2-base",
        "revision": "00921df66db728a9ceb750f5eca43e5c203a2051", "license": "MIT",
        "files": ["*.json", "*.txt", "model.safetensors"]},
}
MODEL_IDS = tuple(_SPECS)

# matplotlib's inferno sampled every 1/8: (r, g, b); interpolated to a 256-entry table so matplotlib is not needed
_INFERNO = ((0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85), (227, 89, 51), (249, 140, 10),
            (249, 201, 50), (252, 255, 164))


def _spec(model):
    try:
        return _SPECS[model]
    except KeyError:
        raise ValueError(f"Unknown scene model {model!r}; available: {', '.join(MODEL_IDS)}") from None


def metadata(model):
    spec = _spec(model)
    return {"source": f"https://huggingface.co/{spec['source']}", "revision": spec["revision"], "license": spec["license"]}


def download(model, directory):
    from huggingface_hub import snapshot_download

    spec = _spec(model)
    directory = Path(directory)
    snapshot_download(spec["source"], revision=spec["revision"], local_dir=directory, allow_patterns=spec["files"])
    shutil.rmtree(directory / ".cache", ignore_errors=True)  # hub download bookkeeping, not model data
    unsafe = [p.name for p in directory.rglob("*") if p.suffix in (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".py")]
    if unsafe or not any(directory.rglob("*.safetensors")):
        raise RuntimeError(f"{model}: download is not safetensors-only ({unsafe or 'no weights'})")


def load(model, directory):
    _spec(model)
    return {"depth-anything-v2-small": _load_depth_anything, "marigold-depth-v1-1": _load_marigold,
            "sam2.1-tiny": _load_sam2, "florence-2-base": _load_florence}[model](str(directory))


# ---- shared helpers ---------------------------------------------------------------------------

def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _options(options, allowed):
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise TypeError("options must be an object")
    extra = sorted(str(k) for k in options if k not in allowed)
    if extra:
        raise ValueError(f"Unsupported options: {', '.join(extra)}; allowed: {', '.join(sorted(allowed))}")
    return options


def _integer(options, key, default, low, high):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer between {low} and {high}")
    return value


def _colormap(opts):
    colormap = opts.get("colormap", "gray")
    if colormap not in ("gray", "inferno"):
        raise ValueError("colormap must be 'gray' or 'inferno'")
    return colormap


def _png(array):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    return "image/png", buffer.getvalue()


def _depth_png(depth, colormap):
    """Normalise a float depth map (larger = nearer) to 8 bit, near = bright, optionally through the inferno table."""
    import numpy as np

    if not np.isfinite(depth).all():
        raise RuntimeError("model produced a non-finite depth value")
    low, high = float(depth.min()), float(depth.max())
    grey = np.zeros(depth.shape, np.uint8) if high <= low else np.rint((depth - low) / (high - low) * 255).astype(np.uint8)
    if colormap == "inferno":
        table = np.rint(np.stack([np.interp(np.arange(256) / 255, np.linspace(0, 1, len(_INFERNO)), [c[i] for c in _INFERNO])
                                  for i in range(3)], axis=1)).astype(np.uint8)
        return _png(table[grey])
    return _png(grey)


def _json(payload):
    try:
        return "application/json", json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError as e:
        raise RuntimeError("model produced a non-finite value") from e


def _clip(value, high):
    return min(max(float(value), 0.0), float(high))


def _box(x1, y1, x2, y2, width, height):
    return {"x1": round(_clip(x1, width), 1), "y1": round(_clip(y1, height), 1),
            "x2": round(_clip(x2, width), 1), "y2": round(_clip(y2, height), 1)}


# ---- Depth Anything V2 ------------------------------------------------------------------------

def _load_depth_anything(path):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = _device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                                        trust_remote_code=False, dtype=dtype).to(device).eval()

    def infer(image, options):
        colormap = _colormap(_options(options, {"colormap"}))
        inputs = processor(images=image, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            depth = model(**inputs).predicted_depth  # relative inverse depth: larger = nearer
            depth = torch.nn.functional.interpolate(depth[:, None].float(), size=(image.height, image.width),
                                                    mode="bicubic", align_corners=False)[0, 0]
        return _depth_png(depth.cpu().numpy(), colormap)

    infer.device = device
    return infer


# ---- Marigold depth ---------------------------------------------------------------------------

def _load_marigold(path):
    import torch
    from diffusers import MarigoldDepthPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("marigold-depth-v1-1 requires CUDA (fp16 diffusion); refusing to fall back to CPU")
    pipe = MarigoldDepthPipeline.from_pretrained(path, variant="fp16", dtype=torch.float16, use_safetensors=True,
                                                 local_files_only=True, trust_remote_code=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def infer(image, options):
        opts = _options(options, {"steps", "ensemble", "colormap"})
        steps = _integer(opts, "steps", 4, 1, 10)
        ensemble = _integer(opts, "ensemble", 1, 1, 5)
        colormap = _colormap(opts)
        with torch.inference_mode():
            result = pipe(image, num_inference_steps=steps, ensemble_size=ensemble, match_input_resolution=True,
                          output_type="np")
        # prediction is 0 = near .. 1 = far; flip so near is bright like the other depth model
        return _depth_png(1.0 - result.prediction[0, ..., 0], colormap)

    infer.device = "cuda"
    return infer


# ---- SAM 2.1 ----------------------------------------------------------------------------------

def _pair(value, count, name):
    """'x,y' / 'x1,y1,x2,y2' -> list of finite floats."""
    try:
        numbers = [float(part) for part in value.split(",")] if isinstance(value, str) else None
    except ValueError:
        numbers = None
    if numbers is None or len(numbers) != count or not all(abs(n) < 1e9 for n in numbers):
        raise ValueError(f"{name} must be {count} comma-separated numbers, e.g. {','.join(['120'] * count)}")
    return numbers


def _sam_prompt(opts, width, height):
    """Validated (points, box, multimask, centred); no prompt at all means one positive point at the image centre."""
    points, box, multimask = opts.get("points") or [], opts.get("box") or None, opts.get("multimask", False)
    if not isinstance(multimask, bool):
        raise TypeError("multimask must be true or false")
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ValueError(f"points must be a list of at most {MAX_POINTS} 'x,y' strings")
    points = [_pair(p, 2, "each point") for p in points]
    if box is not None:
        box = _pair(box, 4, "box")
    for x, y in points:
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"point {x:g},{y:g} is outside the {width}x{height} image")
    if box is not None and not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"box must be x1,y1,x2,y2 with x1<x2, y1<y2, inside the {width}x{height} image")
    centred = not points and box is None
    if centred:
        points = [[width / 2, height / 2]]
    return points, box, multimask, centred


def _polygon(mask):
    """Outline of the largest connected region, simplified to at most MAX_POLYGON_POINTS vertices."""
    import cv2

    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > MAX_POLYGON_POINTS:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def _load_sam2(path):
    import numpy as np
    import torch
    from transformers import Sam2Model, Sam2Processor

    device = _device()
    processor = Sam2Processor.from_pretrained(path, local_files_only=True)
    model = Sam2Model.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                      trust_remote_code=False).to(device).eval()

    def infer(full, options):
        opts = _options(options, {"points", "box", "multimask"})
        width, height = full.size
        points, box, multimask, centred = _sam_prompt(opts, width, height)
        scale = min(1.0, MAX_SIDE / max(width, height))
        image = full.resize((max(1, round(width * scale)), max(1, round(height * scale)))) if scale < 1 else full
        kwargs = {}
        if points:
            kwargs.update(input_points=[[[[x * scale, y * scale] for x, y in points]]], input_labels=[[[1] * len(points)]])
        if box is not None:
            kwargs["input_boxes"] = [[[v * scale for v in box]]]
        inputs = processor(images=image, return_tensors="pt", **kwargs).to(device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=multimask)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        scores = outputs.iou_scores.float().cpu().numpy().reshape(-1)
        results = []
        for index in np.argsort(-scores):
            mask = masks[index]
            ys, xs = np.nonzero(mask)
            results.append({
                "score": round(_clip(scores[index], 1), 4), "area_fraction": round(float(mask.mean()), 4),
                "bbox": _box(xs.min() / scale, ys.min() / scale, (xs.max() + 1) / scale, (ys.max() + 1) / scale,
                             width, height) if len(xs) else None,
                "polygon": [[round(_clip(x / scale, width), 1), round(_clip(y / scale, height), 1)]
                            for x, y in _polygon(mask)]})
        count = len(results)
        text = f"{count} mask{'s' if count != 1 else ''}, " + " / ".join(f"{r['area_fraction'] * 100:.1f}%" for r in results) + " of the image"
        if centred:
            text += " (no prompt given: segmented the object at the image centre)"
        return _json({"text": text, "masks": results, "image": {"width": width, "height": height},
                      "prompt": {"points": points, "box": box, "default_centre": centred}})

    infer.device = device
    return infer


# ---- Florence-2 -------------------------------------------------------------------------------

def _load_florence(path):
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Florence2ForConditionalGeneration.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                                              trust_remote_code=False, dtype=dtype).to(device).eval()

    def infer(image, options):
        task = _options(options, {"task"}).get("task", "caption")
        if task not in FLORENCE_TASKS:
            raise ValueError(f"task must be one of: {', '.join(FLORENCE_TASKS)}")
        prompt = FLORENCE_TASKS[task]
        w, h = image.size
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=512, num_beams=3, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)[prompt]
        out = {"task": task, "image": {"width": w, "height": h}}
        if task == "caption":
            out["text"] = parsed.strip()
        elif task == "detect":
            out["detections"] = [{"label": label.strip(), "box": _box(*box, w, h)}
                                 for box, label in zip(parsed["bboxes"], parsed["labels"])]
            out["text"] = ", ".join(d["label"] for d in out["detections"])
        elif task == "ocr":
            out["regions"] = []
            for quad, text in zip(parsed["quad_boxes"], parsed["labels"]):
                xs, ys = quad[0::2], quad[1::2]
                out["regions"].append({"text": text.replace("</s>", "").strip(),
                                       "box": _box(min(xs), min(ys), max(xs), max(ys), w, h),
                                       "quad": [[round(_clip(x, w), 1), round(_clip(y, h), 1)] for x, y in zip(xs, ys)]})
            out["text"] = "\n".join(r["text"] for r in out["regions"] if r["text"])
        else:
            out["regions"] = [{"box": _box(*box, w, h)} for box in parsed["bboxes"]]
            out["text"] = f"{len(out['regions'])} candidate regions"
        return _json(out)

    infer.device = device
    return infer
