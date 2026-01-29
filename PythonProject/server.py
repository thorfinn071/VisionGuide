import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import base64
import time
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ====== SETTINGS ======
CONF_THRESHOLD = 0.35
MIN_BOX_AREA_RATIO = 0.004

MODE_STREET = "street"
MODE_CANE = "cane"

STREET_OBJECTS = frozenset({
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "dog", "cat", "pole"
})

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_DOWNSCALE = 0.45
DEPTH_INVERT_DEFAULT = False

FLOOR_GRAD_STEP = 0.12
FLOOR_GRAD_DROP = 0.14

POS_TRANSLATE_EN = {
    "left": "on your left",
    "right": "on your right",
    "center": "in front of you"
}

def decode_b64_image(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bad image: cannot decode")
    return img

def get_horizontal_position(cx: float, w: int) -> str:
    if cx < w * 0.33:
        return "left"
    elif cx > w * 0.66:
        return "right"
    return "center"

def is_vehicle(name: str) -> bool:
    return name in {"car", "bus", "truck", "motorcycle", "bicycle"}

def dist_rank(dist: str) -> int:
    return 0 if dist == "very close" else 1 if dist == "close" else 2

def danger_rank(name: str, pos: str, dist: str):
    d = dist_rank(dist)
    center_bonus = -3 if (pos == "center" and d <= 1) else 0
    vehicle_bonus = -2 if is_vehicle(name) else 0
    person_bonus = -1 if name == "person" else 0
    return (d, center_bonus, vehicle_bonus, person_bonus)

def plural_en(name: str, n: int) -> str:
    if n == 1:
        return name
    irregular = {"person": "persons"}
    if name in irregular:
        return irregular[name]
    if name.endswith("s"):
        return name
    return name + "s"

def build_single_description_en(name: str, pos: str, dist: str, count: int = 1) -> str:
    where = POS_TRANSLATE_EN.get(pos, "near you")
    who = f"{count} {plural_en(name, count)}" if count > 1 else name
    if dist == "very close":
        return f"{who} very close {where}"
    if dist == "close":
        return f"{who} close {where}"
    return f"{who} {where}"

def compute_depth_fast(frame_bgr: np.ndarray, processor, depth_model, device, downscale: float) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    s = float(downscale)
    h2 = max(160, int(H * s))
    w2 = max(160, int(W * s))

    small = cv2.resize(frame_bgr, (w2, h2), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = depth_model(**inputs)
        else:
            out = depth_model(**inputs)

    d = out.predicted_depth[0].detach().float().cpu().numpy()
    d_up = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return d_up

def depth_distance_category(d_map: np.ndarray, x1, y1, x2, y2, invert: bool,
                            box_area_ratio: Optional[float] = None,
                            height_ratio: Optional[float] = None) -> Optional[str]:
    if d_map is None:
        return None

    H, W = d_map.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W, int(x2)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = d_map[y1:y2, x1:x2]
    if roi.size < 80:
        return None

    med = float(np.median(roi))
    lo, hi = np.percentile(d_map, 5), np.percentile(d_map, 95)
    if hi - lo < 1e-6:
        return None

    dn = float(np.clip((med - lo) / (hi - lo), 0.0, 1.0))
    if invert:
        dn = 1.0 - dn

    if dn <= 0.18:
        dist = "very close"
    elif dn <= 0.38:
        dist = "close"
    else:
        dist = "far"

    if box_area_ratio is not None:
        if dist == "very close" and box_area_ratio < 0.06:
            dist = "close"
        if dist in ("very close", "close") and box_area_ratio < 0.025:
            dist = "far"

    if height_ratio is not None:
        if dist == "very close" and height_ratio < 0.35:
            dist = "close"
        if dist in ("very close", "close") and height_ratio < 0.20:
            dist = "far"

    return dist

def floor_hazard_from_depth(d_map: np.ndarray, invert: bool) -> Optional[str]:
    if d_map is None:
        return None

    H, W = d_map.shape[:2]
    y0 = int(H * 0.62)
    y1 = int(H * 0.98)
    x0 = int(W * 0.32)
    x1 = int(W * 0.68)

    roi = d_map[y0:y1, x0:x1]
    if roi.size < 500:
        return None

    prof = np.median(roi, axis=1).astype(np.float32)
    lo, hi = np.percentile(prof, 5), np.percentile(prof, 95)
    if hi - lo < 1e-6:
        return None

    pn = np.clip((prof - lo) / (hi - lo), 0, 1)
    if invert:
        pn = 1.0 - pn

    if len(pn) >= 7:
        k = 7
        pn = np.convolve(pn, np.ones(k, dtype=np.float32) / k, mode="same")

    g = pn[1:] - pn[:-1]
    max_pos = float(np.max(g))
    min_neg = float(np.min(g))

    if max_pos >= FLOOR_GRAD_DROP:
        return "drop"
    if min_neg <= -FLOOR_GRAD_STEP:
        return "step"
    return None

# ====== LOAD MODELS ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO("yolov8n.pt")
try:
    if device.type == "cuda":
        yolo.to("cuda")
except Exception:
    pass

processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME, use_fast=True)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(device)
depth_model.eval()

app = FastAPI(title="BlindAssistant Vision Server")

class DetectRequest(BaseModel):
    image_b64: str
    mode: str = MODE_STREET
    mirror: bool = True
    use_depth: bool = False          # начни с False! потом включишь
    depth_invert: bool = DEPTH_INVERT_DEFAULT
    want_floor: bool = False         # начни с False! потом включишь
    max_objects: int = 10

class ObjOut(BaseModel):
    name: str
    pos: str
    dist: str
    conf: float
    bbox: List[int]
    count: int = 1

class DetectResponse(BaseModel):
    ts: float
    text: str
    floor: Optional[str] = None      # "step"/"drop"/None
    objects: List[ObjOut]
    primary: Optional[ObjOut] = None

@app.get("/ping")
def ping():
    return {"ok": True, "device": str(device)}

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    frame = decode_b64_image(req.image_b64)
    if req.mirror:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]
    frame_area = H * W
    min_box_area = frame_area * MIN_BOX_AREA_RATIO

    d_map = None
    if req.use_depth:
        d_map = compute_depth_fast(frame, processor, depth_model, device, DEPTH_DOWNSCALE)

    half_flag = (device.type == "cuda")
    results = yolo(frame, imgsz=416, verbose=False, half=half_flag)

    dets = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        xyxy = boxes.xyxy.detach().cpu().numpy()

        for i, conf in enumerate(confidences):
            conf = float(conf)
            if conf < CONF_THRESHOLD:
                continue

            cls = int(classes[i])
            name = yolo.names.get(cls, "object")

            if req.mode == MODE_STREET and name not in STREET_OBJECTS:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            box_area = bw * bh
            if box_area < min_box_area:
                continue

            box_area_ratio = box_area / frame_area
            height_ratio = bh / H

            cx = 0.5 * (x1 + x2)
            pos = get_horizontal_position(cx, W)

            dist = None
            if d_map is not None:
                dist = depth_distance_category(
                    d_map, x1, y1, x2, y2,
                    invert=req.depth_invert,
                    box_area_ratio=box_area_ratio,
                    height_ratio=height_ratio
                )

            if dist is None:
                bottom_ratio = (y2 / H)
                score = box_area_ratio * 0.50 + height_ratio * 0.25 + bottom_ratio * 0.25
                if score > 0.18:
                    dist = "very close"
                elif score > 0.09:
                    dist = "close"
                else:
                    dist = "far"

            dets.append({
                "name": name,
                "pos": pos,
                "dist": dist,
                "conf": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    buckets: Dict[tuple, Dict[str, Any]] = {}
    for d in dets:
        k = (d["name"], d["pos"], d["dist"])
        if k not in buckets:
            buckets[k] = {**d, "count": 1}
        else:
            buckets[k]["count"] += 1
            buckets[k]["conf"] = max(buckets[k]["conf"], d["conf"])

    objects = list(buckets.values())
    objects.sort(key=lambda d: danger_rank(d["name"], d["pos"], d["dist"]))

    primary = objects[0] if objects else None

    floor = None
    if req.want_floor and req.use_depth and d_map is not None:
        floor = floor_hazard_from_depth(d_map, invert=req.depth_invert)

    if primary:
        text = build_single_description_en(primary["name"], primary["pos"], primary["dist"], primary.get("count", 1))
    else:
        text = "Path is clear."

    out_objs = [ObjOut(**d) for d in objects[: max(1, req.max_objects)]]
    out_primary = ObjOut(**primary) if primary else None

    return DetectResponse(
        ts=time.time(),
        text=text,
        floor=floor,
        objects=out_objs,
        primary=out_primary
    )
