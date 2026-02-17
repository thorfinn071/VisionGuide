import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from ultralytics import YOLO
import cv2
import win32com.client
import pythoncom
import threading
import queue
import time
import winsound
import speech_recognition as sr
import numpy as np
import torch
import itertools
from collections import deque, Counter

try:
    cv2.setNumThreads(1)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# ================= SETTINGS =================

UNMIRROR_CAMERA = True
SPEAK_USER_PERSPECTIVE = True

DIST_VERY_CLOSE_T = 0.26
DIST_CLOSE_T = 0.13

FAR_AREA_MAX_FOR_NEAR = 0.030
FAR_HEIGHT_MAX_FOR_NEAR = 0.22
ABS_FAR_AREA = 0.018
ABS_FAR_HEIGHT = 0.16

CONF_THRESHOLD = 0.35
MIN_BOX_AREA_RATIO = 0.004

YOLO_INTERVAL = 0.08

last_danger_state = False
danger_just_announced = False

USE_ROI_INFERENCE = True
ROI_WIDTH_RATIO = 0.6
ROI_HEIGHT_RATIO = 0.65
FULL_FRAME_REFRESH_SEC = 1.5

ADAPTIVE_ROI = True
ROI_WIDE_WIDTH_RATIO = 0.85
SIDE_ACTIVITY_HOLD_SEC = 1.2
FORCE_FULL_ON_VERY_CLOSE = True

SPEECH_COOLDOWN = 2.5
PERSON_COOLDOWN = 7.0
FREE_PATH_COOLDOWN = 4.0

CLOSE_CONFIRM_FRAMES = 2
VERY_CLOSE_CONFIRM_FRAMES = 2

PRIORITY_CRITICAL = 0
PRIORITY_IMPORTANT = 1
PRIORITY_INFO = 2

CRITICAL_MIN_GAP = 1.4
_last_critical_text = ""
_last_critical_time = 0.0

APPROACH_HIST_LEN = 6
APPROACH_MIN_DT = 0.35
VEH_APPROACH_AREA_RATE_T = 0.22
VEH_APPROACH_HEIGHT_RATE_T = 0.14
APPROACH_THREAT_BOOST_MAX = 0.75
APPROACH_SAY_COOLDOWN = 1.4
last_approach_say_time = 0.0

WALKWAY_ENABLE = True
WALKWAY_INTERVAL_SEC = 0.35
WALKWAY_Y_START_RATIO = 0.55
WALKWAY_X_MARGIN_RATIO = 0.10
WALKWAY_SMOOTH_WIN = 31
WALKWAY_SHIFT_T = 0.12
WALKWAY_SAY_COOLDOWN = 1.8
WALKWAY_DRAW_OVERLAY = True

_last_walkway_time = 0.0
_last_walkway_say = 0.0
_last_walkway_hint = "forward"
_last_walkway_x = None

MODE_STREET = "street"
MODE_CANE = "cane"
MODE_SCAN = "scan"

current_mode = MODE_STREET
paused = False

STREET_OBJECTS = frozenset({
    "person", "dog", "cat",
    "car", "bus", "truck", "motorcycle", "bicycle",
    "traffic light", "stop sign", "fire hydrant",
    "parking meter", "bench",
    "backpack", "handbag", "suitcase",
    "umbrella",
})

STATE_CLEAR = "clear"
STATE_CAUTION = "caution"
STATE_DANGER = "danger"

CLEAR_HOLD_SEC = 1.2
CAUTION_HOLD_SEC = 0.25
DANGER_HOLD_SEC = 0.12

YOLO_STALE_SEC = 2.0
clear_announced = False

BEEP_MIN_INTERVAL = 0.12
BEEP_INTERVAL_VERY_CLOSE = 0.20
BEEP_INTERVAL_CLOSE = 0.45
BEEP_INTERVAL_FAR = 0.85
last_beep_time = 0.0

FRAME_SKIP_COUNT = 2

DISPLAY_W = 960
DISPLAY_H = 720
DISPLAY_INTERP = cv2.INTER_LINEAR
DISPLAY_EVERY_N = 2
display_frame = None
display_counter = 0

TRACK_MAX_AGE = 8
TRACK_MATCH_DIST = 70.0
TRACK_CONFIRM_FRAMES = 2
TRACK_DIST_HISTORY = 6

# ================= FILTER SETTINGS =================

FRONT_ONLY_DEFAULT = True
IGNORE_FAR_OBJECTS_DEFAULT = True
CENTER_BAND = (0.35, 0.65)
ALLOW_SIDE_CLOSE = True

# ===== Scene / Threat / Guidance =====

ZONE_COUNT = 5
GUIDE_MIN_CENTER_THREAT = 2.2
GUIDE_IMPROVEMENT_RATIO = 1.45
GUIDE_IMPROVEMENT_ABS = 1.2
SUMMARY_COOLDOWN = 4.0
GUIDE_COOLDOWN = 2.0
MAX_SUMMARY_ITEMS = 3

CLASS_WEIGHT = {
    "person": 1.0,
    "car": 1.6,
    "bus": 1.8,
    "truck": 2.0,
    "motorcycle": 1.5,
    "bicycle": 1.2,
    "dog": 1.2,
    "traffic light": 0.6,
    "stop sign": 0.6,
    "bench": 0.8,
    "fire hydrant": 0.9,
    "parking meter": 0.7,
    "backpack": 0.7,
    "handbag": 0.7,
    "suitcase": 0.8,
    "umbrella": 0.7,
}
DEFAULT_WEIGHT = 1.0

# ================= HELPERS =================

POS_EN = {
    "left": "on your left",
    "right": "on your right",
    "center": "in front of you"
}

def clamp(x, a, b):
    return a if x < a else (b if x > b else x)

def compute_nav_state(detections):
    if not detections:
        return STATE_CLEAR
    has_very_close = any(d.get("dist") == "very close" for d in detections)
    if has_very_close:
        return STATE_DANGER
    has_close = any(d.get("dist") == "close" for d in detections)
    if has_close:
        return STATE_CAUTION
    return STATE_CLEAR

def get_horizontal_position(cx, w):
    lo, hi = CENTER_BAND
    if cx < w * lo:
        side = "left"
    elif cx > w * hi:
        side = "right"
    else:
        side = "center"
    if SPEAK_USER_PERSPECTIVE and UNMIRROR_CAMERA and side != "center":
        return "right" if side == "left" else "left"
    return side

def dist_rank(dist: str) -> int:
    return 0 if dist == "very close" else 1 if dist == "close" else 2

def is_vehicle(name: str) -> bool:
    return name in {"car", "bus", "truck", "motorcycle", "bicycle"}

def danger_rank_tuple(name: str, pos: str, dist: str):
    d = dist_rank(dist)
    center_bonus = -3 if (pos == "center" and d <= 1) else 0
    vehicle_bonus = -2 if is_vehicle(name) else 0
    person_bonus = -1 if name == "person" else 0
    return (d, center_bonus, vehicle_bonus, person_bonus)

def labels_get(labels, cls_id: int) -> str:
    try:
        if isinstance(labels, dict):
            return labels.get(cls_id, "object")
        if isinstance(labels, (list, tuple)) and 0 <= cls_id < len(labels):
            return labels[cls_id]
    except Exception:
        pass
    return "object"

def norm_dist_for_speech(dist: str) -> str:
    return "near" if dist in ("very close", "close") else "far"

def vibrate_by_distance(dist, now):
    global last_beep_time
    if dist == "very close":
        interval = BEEP_INTERVAL_VERY_CLOSE
        freq, dur = 1600, 110
    elif dist == "close":
        interval = BEEP_INTERVAL_CLOSE
        freq, dur = 1100, 90
    else:
        interval = BEEP_INTERVAL_FAR
        freq, dur = 700, 70

    if now - last_beep_time >= max(BEEP_MIN_INTERVAL, interval):
        last_beep_time = now
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass

def box_center(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def estimate_distance_by_box(box_area_ratio: float, height_ratio: float, bottom_ratio: float) -> str:
    if box_area_ratio < ABS_FAR_AREA and height_ratio < ABS_FAR_HEIGHT:
        return "far"
    score = box_area_ratio * 0.70 + height_ratio * 0.25 + bottom_ratio * 0.05
    if box_area_ratio < FAR_AREA_MAX_FOR_NEAR and height_ratio < FAR_HEIGHT_MAX_FOR_NEAR:
        return "far"
    if score >= DIST_VERY_CLOSE_T:
        return "very close"
    if score >= DIST_CLOSE_T:
        return "close"
    return "far"

def should_keep_detection(mode: str, pos: str, dist: str) -> bool:
    if mode == MODE_SCAN:
        return True
    if dist == "very close":
        return True
    if mode == MODE_STREET and pos != "center":
        return False
    if mode == MODE_CANE:
        if ALLOW_SIDE_CLOSE and dist == "close":
            return True
    if FRONT_ONLY_DEFAULT and pos != "center":
        return False
    if IGNORE_FAR_OBJECTS_DEFAULT and dist == "far":
        return False
    return True

def most_common_dist(hist_deque):
    if not hist_deque:
        return "far"
    c = Counter(hist_deque)
    return c.most_common(1)[0][0]

def area_ratio_from_bbox(bbox, frame_area):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1) / float(frame_area)

def threat_score(name: str, pos: str, dist: str, area_ratio: float) -> float:
    w = CLASS_WEIGHT.get(name, DEFAULT_WEIGHT)
    if dist == "very close":
        dist_factor = 2.2
    elif dist == "close":
        dist_factor = 1.5
    else:
        dist_factor = 1.0
    size_factor = 0.7 + 2.2 * (area_ratio ** 0.5)
    center_factor = 1.25 if pos == "center" else 1.0
    return w * dist_factor * size_factor * center_factor

def build_scene_summary(detections):
    if not detections:
        return "Path is clear."
    top = sorted(detections, key=lambda d: d.get("threat", 0.0), reverse=True)[:MAX_SUMMARY_ITEMS]
    parts = [f'{d["name"]} {POS_EN.get(d["pos"], "near you")}' for d in top]
    return "Ahead: " + ", ".join(parts) + "."

def best_direction_hint(detections, w, frame_area):
    if not detections or ZONE_COUNT < 3:
        return "forward", 0.0, 0.0

    zone_threat = [0.0] * ZONE_COUNT
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cx = 0.5 * (x1 + x2)
        cx_norm = clamp(cx / float(w), 0.0, 0.9999)
        zi = int(cx_norm * ZONE_COUNT)

        ar = d.get("area_ratio")
        if ar is None:
            ar = area_ratio_from_bbox(d["bbox"], frame_area)

        t = d.get("threat")
        if t is None:
            t = threat_score(d["name"], d["pos"], d["dist"], ar)

        zone_threat[zi] += t

    best_i = min(range(ZONE_COUNT), key=lambda i: zone_threat[i])
    mid_i = ZONE_COUNT // 2

    best_threat = float(zone_threat[best_i])
    center_threat = float(zone_threat[mid_i])

    zone_center_cx = (best_i + 0.5) * (w / float(ZONE_COUNT))
    best_pos = get_horizontal_position(zone_center_cx, w)

    if best_i == mid_i or best_pos == "center":
        return "forward", center_threat, best_threat

    if best_pos == "left":
        hint = "left" if best_i == 0 else "slightly left"
    elif best_pos == "right":
        hint = "right" if best_i == (ZONE_COUNT - 1) else "slightly right"
    else:
        hint = "forward"

    return hint, center_threat, best_threat

def compute_growth_rate(hist):
    if hist is None or len(hist) < 2:
        return 0.0
    t0, v0 = hist[0]
    t1, v1 = hist[-1]
    dt = float(t1 - t0)
    if dt <= 1e-6:
        return 0.0
    return float(v1 - v0) / dt

def walkway_hint_from_frame(frame_bgr):
    try:
        h, w = frame_bgr.shape[:2]
        y0 = int(h * WALKWAY_Y_START_RATIO)
        x_margin = int(w * WALKWAY_X_MARGIN_RATIO)

        roi = frame_bgr[y0:h, x_margin:w - x_margin]
        if roi.size == 0:
            return "forward", 0.5, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        col = edges.sum(axis=0).astype(np.float32)

        win = int(WALKWAY_SMOOTH_WIN)
        if win < 5:
            win = 5
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        col_s = np.convolve(col, kernel, mode="same")

        idx = int(np.argmin(col_s))
        x_corr = (idx + x_margin) / float(w)

        med = float(np.median(col_s))
        mn = float(col_s[idx])
        conf = 0.0
        if med > 1e-6:
            conf = clamp((med - mn) / (med + 1e-6), 0.0, 1.0)

        if x_corr < 0.5 - WALKWAY_SHIFT_T:
            return "left", x_corr, conf
        elif x_corr > 0.5 + WALKWAY_SHIFT_T:
            return "right", x_corr, conf
        else:
            return "forward", x_corr, conf

    except Exception:
        return "forward", 0.5, 0.0

# ================= SPEECH =================

def pick_english_voice(spk):
    try:
        voices = spk.GetVoices()
        preferred = ["Aria", "Jenny", "Guy", "Zira", "David", "Mark", "Ryan", "Susan", "Hazel", "George"]
        best = None
        best_score = -1
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = ""
            try:
                desc = v.GetDescription()
            except Exception:
                pass
            dlow = (desc or "").lower()

            score = 0
            if "english" in dlow or "en-" in dlow:
                score += 3
            for p in preferred:
                if p.lower() in dlow:
                    score += 5
                    break

            if score > best_score:
                best_score = score
                best = v

        if best is not None:
            spk.Voice = best
    except Exception:
        pass

speech_queue = queue.PriorityQueue(maxsize=25)
_msg_counter = itertools.count()
speech_running = True
speech_interrupt = threading.Event()

last_enqueued_text = ""
last_enqueued_time = 0.0
ENQUEUE_DEBOUNCE = 1.0
MAX_INFO_CHARS = 90
MAX_IMPORTANT_CHARS = 120
MAX_CRITICAL_CHARS = 140

def _clear_queue():
    try:
        while not speech_queue.empty():
            _ = speech_queue.get_nowait()
            speech_queue.task_done()
    except Exception:
        pass

def _drop_pending_info():
    try:
        temp = []
        while not speech_queue.empty():
            item = speech_queue.get_nowait()
            p = item[0]
            if p >= PRIORITY_INFO:
                speech_queue.task_done()
                continue
            temp.append(item)
            speech_queue.task_done()
        for it in temp:
            try:
                speech_queue.put_nowait(it)
            except Exception:
                break
    except Exception:
        pass

def _truncate_for_priority(text: str, priority: int) -> str:
    if not text:
        return text
    if priority >= PRIORITY_INFO:
        limit = MAX_INFO_CHARS
    elif priority == PRIORITY_IMPORTANT:
        limit = MAX_IMPORTANT_CHARS
    else:
        limit = MAX_CRITICAL_CHARS
    if len(text) <= limit:
        return text
    return text[:limit-3].rstrip() + "..."

def say(text, priority=PRIORITY_INFO, purge=False):
    global last_enqueued_text, last_enqueued_time
    global _last_critical_text, _last_critical_time

    now = time.time()
    text = _truncate_for_priority(text, priority)

    if priority == PRIORITY_CRITICAL:
        if text == _last_critical_text and (now - _last_critical_time) < CRITICAL_MIN_GAP:
            return
        _last_critical_text = text
        _last_critical_time = now

    if priority >= PRIORITY_INFO:
        try:
            if speech_queue.qsize() >= 12:
                return
        except Exception:
            pass

    if priority != PRIORITY_CRITICAL:
        if text == last_enqueued_text and (now - last_enqueued_time) < ENQUEUE_DEBOUNCE:
            return

    last_enqueued_text = text
    last_enqueued_time = now

    try:
        if priority == PRIORITY_CRITICAL:
            speech_interrupt.set()
            purge = True
        elif priority == PRIORITY_IMPORTANT:
            _drop_pending_info()

        if purge:
            _clear_queue()

        if speech_queue.full():
            temp = []
            removed = False
            try:
                while not speech_queue.empty():
                    item = speech_queue.get_nowait()
                    p = item[0]
                    if (not removed) and p >= PRIORITY_INFO:
                        removed = True
                        speech_queue.task_done()
                        continue
                    temp.append(item)
                    speech_queue.task_done()
            except Exception:
                pass

            for it in temp:
                try:
                    speech_queue.put_nowait(it)
                except Exception:
                    break

            if speech_queue.full() and priority >= PRIORITY_INFO:
                return

        speech_queue.put_nowait((priority, next(_msg_counter), text))
    except Exception:
        pass

def speech_worker():
    pythoncom.CoInitialize()
    speaker = None
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        pick_english_voice(speaker)

        while speech_running:
            if speech_interrupt.is_set():
                try:
                    speaker.Speak("", 2)
                except Exception:
                    pass
                speech_interrupt.clear()

            try:
                priority, _, text = speech_queue.get(timeout=0.2)
                try:
                    speaker.Speak(text, 0)
                except Exception:
                    pass
                speech_queue.task_done()
            except queue.Empty:
                pass
    finally:
        try:
            if speaker is not None:
                speaker.Speak("", 2)
        except Exception:
            pass
        pythoncom.CoUninitialize()

threading.Thread(target=speech_worker, daemon=True).start()

# ================= VOICE COMMANDS =================

request_whats_around = False
request_sector = None

def voice_command_worker():
    global current_mode, paused
    global request_whats_around, request_sector

    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic:
        try:
            r.adjust_for_ambient_noise(mic, duration=0.8)
        except Exception:
            pass

    while True:
        try:
            with mic:
                audio = r.listen(mic, phrase_time_limit=3)

            text = r.recognize_google(audio, language="en-US").lower().strip()

            if "pause" in text:
                paused = True
                say("Paused.", priority=PRIORITY_IMPORTANT)
            elif "resume" in text or "continue" in text:
                paused = False
                say("Resumed.", priority=PRIORITY_IMPORTANT)

            elif "street mode" in text:
                current_mode = MODE_STREET
                say("Street mode.", priority=PRIORITY_IMPORTANT)
            elif "cane mode" in text:
                current_mode = MODE_CANE
                say("Cane mode.", priority=PRIORITY_IMPORTANT)
            elif "scan mode" in text or "overview mode" in text:
                current_mode = MODE_SCAN
                say("Scan mode.", priority=PRIORITY_IMPORTANT)

            elif "what's around" in text or "what is around" in text or "around me" in text:
                request_whats_around = True
                request_sector = None
            elif "what's on the left" in text or "what is on the left" in text:
                request_whats_around = True
                request_sector = "left"
            elif "what's on the right" in text or "what is on the right" in text:
                request_whats_around = True
                request_sector = "right"
            elif "what's in front" in text or "what is in front" in text:
                request_whats_around = True
                request_sector = "center"

        except Exception:
            pass

threading.Thread(target=voice_command_worker, daemon=True).start()

# ================= DEVICE =================

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ================= YOLO =================

yolo = YOLO("yolov8n.pt")
labels = yolo.names
if use_cuda:
    try:
        yolo.to("cuda")
        try:
            dummy = np.zeros((416, 416, 3), dtype=np.uint8)
            _ = yolo(dummy, imgsz=416, verbose=False, half=True)
        except Exception:
            pass
    except Exception:
        pass

# ================= CAMERA =================

def open_camera():
    for idx in range(0, 4):
        c = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if c.isOpened():
            return c
        c.release()
    for idx in range(0, 4):
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened():
            return c
        c.release()
    for idx in range(0, 4):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            return c
        c.release()
    return None

def configure_camera(c):
    try:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    c.set(cv2.CAP_PROP_FPS, 30)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap = open_camera()
if cap is None:
    raise RuntimeError("Camera not found or busy (Zoom/Discord/browser).")
configure_camera(cap)

window_name = "BlindAssistant"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 720)

latest_frame = None
latest_frame_id = 0
last_cam_publish_time = 0.0
frame_lock = threading.Lock()

last_detections = []
detections_lock = threading.Lock()

camera_running = True
yolo_running = True

CAM_STALL_SEC = 1.5

cap_lock = threading.Lock()

def camera_worker():
    global latest_frame, latest_frame_id, last_cam_publish_time, camera_running, cap
    while camera_running:
        try:
            with cap_lock:
                local_cap = cap
                if local_cap is None:
                    time.sleep(0.05)
                    continue
                for _ in range(FRAME_SKIP_COUNT):
                    local_cap.grab()
                ret, frame = local_cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if UNMIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            with frame_lock:
                latest_frame = frame
                latest_frame_id += 1

            last_cam_publish_time = time.time()
        except Exception:
            time.sleep(0.02)

# ================= TRACKING =================

tracks = {}
next_track_id = 1
last_yolo_time = 0.0
_processed_frame_id = 0
last_full_frame_time = 0.0
last_side_activity_time = 0.0

yolo_last_ms = 0.0
yolo_fps = 0.0
_yolo_fps_counter = 0
_yolo_fps_last_t = time.time()

last_yolo_publish_time = 0.0

def match_tracks(dets, existing_tracks):
    if not existing_tracks or not dets:
        return [], list(range(len(dets)))

    track_ids = list(existing_tracks.keys())
    track_centers = np.array([existing_tracks[tid]["center"] for tid in track_ids], dtype=np.float32)
    det_centers = np.array([d["center"] for d in dets], dtype=np.float32)

    pairs = []
    for di in range(len(dets)):
        dc = det_centers[di]
        dists = np.linalg.norm(track_centers - dc, axis=1)
        for tj in range(len(track_ids)):
            tid = track_ids[tj]
            if existing_tracks[tid]["name"] != dets[di]["name"]:
                continue
            dist = float(dists[tj])
            if dist <= TRACK_MATCH_DIST:
                pairs.append((dist, di, tid))

    pairs.sort(key=lambda x: x[0])

    matches = []
    used_tracks = set()
    used_dets = set()

    for dist, di, tid in pairs:
        if di in used_dets or tid in used_tracks:
            continue
        matches.append((di, tid))
        used_dets.add(di)
        used_tracks.add(tid)

    new_det_indices = [i for i in range(len(dets)) if i not in used_dets]
    return matches, new_det_indices

def decay_tracks():
    to_del = []
    for tid, t in tracks.items():
        t["age"] += 1
        if t["age"] > TRACK_MAX_AGE:
            to_del.append(tid)
    for tid in to_del:
        del tracks[tid]

def yolo_worker():
    global last_yolo_time, last_detections, _processed_frame_id
    global next_track_id, yolo_running
    global yolo_last_ms, yolo_fps, _yolo_fps_counter, _yolo_fps_last_t
    global last_full_frame_time, last_side_activity_time
    global last_yolo_publish_time

    while yolo_running:
        now = time.time()
        mode_now = current_mode

        if now - last_yolo_time < YOLO_INTERVAL:
            time.sleep(0.003)
            continue

        with frame_lock:
            lf = latest_frame
            fid = latest_frame_id

        if lf is None or fid == _processed_frame_id:
            time.sleep(0.003)
            continue

        _processed_frame_id = fid
        last_yolo_time = now

        frame_yolo = lf
        h, w = frame_yolo.shape[:2]
        frame_area = h * w

        run_full_frame = False
        use_wide_roi = False

        if not USE_ROI_INFERENCE:
            run_full_frame = True
        else:
            if now - last_full_frame_time > FULL_FRAME_REFRESH_SEC:
                run_full_frame = True
                last_full_frame_time = now

        if ADAPTIVE_ROI and (not run_full_frame) and USE_ROI_INFERENCE:
            if now - last_side_activity_time <= SIDE_ACTIVITY_HOLD_SEC:
                use_wide_roi = True

        roi_offset_x = 0
        roi_offset_y = 0

        if run_full_frame or (not USE_ROI_INFERENCE):
            roi_frame = frame_yolo
        else:
            wr = ROI_WIDE_WIDTH_RATIO if use_wide_roi else ROI_WIDTH_RATIO
            roi_w = int(w * wr)
            roi_h = int(h * ROI_HEIGHT_RATIO)

            x1r = (w - roi_w) // 2
            y1r = h - roi_h
            x2r = x1r + roi_w
            y2r = h

            roi_frame = frame_yolo[y1r:y2r, x1r:x2r]
            roi_offset_x = x1r
            roi_offset_y = y1r

        half_flag = bool(use_cuda)

        try:
            t0 = time.perf_counter()
            results = yolo(roi_frame, imgsz=416, verbose=False, half=half_flag)
            t1 = time.perf_counter()

            yolo_last_ms = (t1 - t0) * 1000.0

            _yolo_fps_counter += 1
            now2 = time.time()
            if now2 - _yolo_fps_last_t >= 1.0:
                yolo_fps = _yolo_fps_counter / (now2 - _yolo_fps_last_t)
                _yolo_fps_counter = 0
                _yolo_fps_last_t = now2

        except Exception:
            continue

        min_box_area = frame_area * MIN_BOX_AREA_RATIO
        dets_for_tracking = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            xyxy = boxes.xyxy.detach().cpu().numpy()

            for i, conf in enumerate(confidences):
                if conf < CONF_THRESHOLD:
                    continue

                cls = classes[i]
                name = labels_get(labels, cls)

                if mode_now == MODE_STREET and name not in STREET_OBJECTS:
                    continue

                x1, y1, x2, y2 = xyxy[i].astype(int)

                x1 += roi_offset_x
                x2 += roi_offset_x
                y1 += roi_offset_y
                y2 += roi_offset_y

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                box_area = bw * bh
                if box_area < min_box_area:
                    continue

                box_area_ratio = box_area / frame_area
                height_ratio = bh / h
                bottom_ratio = (y2 / h)

                cx, cy = box_center(x1, y1, x2, y2)
                pos = get_horizontal_position(cx, w)

                dist_raw = estimate_distance_by_box(box_area_ratio, height_ratio, bottom_ratio)

                if pos != "center":
                    last_side_activity_time = now

                if FORCE_FULL_ON_VERY_CLOSE and dist_raw == "very close":
                    last_full_frame_time = now - (FULL_FRAME_REFRESH_SEC + 0.01)

                if not should_keep_detection(mode_now, pos, dist_raw):
                    continue

                dets_for_tracking.append({
                    "name": name,
                    "pos": pos,
                    "dist": dist_raw,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy)
                })

        decay_tracks()
        matches, new_det_indices = match_tracks(dets_for_tracking, tracks)

        for di, tid in matches:
            d = dets_for_tracking[di]
            t = tracks[tid]
            t["age"] = 0
            t["name"] = d["name"]
            t["pos"] = d["pos"]
            t["bbox"] = d["bbox"]
            t["center"] = d["center"]

            t["seen_count"] = min(9999, t.get("seen_count", 0) + 1)
            dh = t.get("dist_hist")
            if dh is None:
                dh = deque(maxlen=TRACK_DIST_HISTORY)
                t["dist_hist"] = dh

            dh.append(d["dist"])
            t["dist"] = most_common_dist(dh)

            new_dist = t["dist"]
            prev_raw = t.get("last_dist_raw", "far")
            t["last_dist_raw"] = new_dist

            if new_dist == "very close":
                t["near_confirm"] = min(999, t.get("near_confirm", 0) + 1)
            elif new_dist == "close":
                if prev_raw in ("close", "very close"):
                    t["near_confirm"] = min(999, t.get("near_confirm", 0) + 1)
                else:
                    t["near_confirm"] = 1
            else:
                t["near_confirm"] = 0

        for di in new_det_indices:
            d = dets_for_tracking[di]
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = {
                "age": 0,
                "name": d["name"],
                "pos": d["pos"],
                "bbox": d["bbox"],
                "center": d["center"],
                "seen_count": 1,
                "dist_hist": deque([d["dist"]], maxlen=TRACK_DIST_HISTORY),
                "dist": d["dist"],
                "near_confirm": 0,
                "last_dist_raw": d["dist"],
            }

        active = []
        for tid, t in tracks.items():
            if t["age"] != 0:
                continue

            if mode_now in (MODE_STREET, MODE_CANE) and t.get("seen_count", 0) < TRACK_CONFIRM_FRAMES:
                continue

            x1, y1, x2, y2 = t["bbox"]
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            ar = (bw * bh) / float(frame_area) if frame_area > 0 else 0.0
            hr = (bh / float(h)) if h > 0 else 0.0

            ah = t.get("area_hist")
            if ah is None:
                ah = deque(maxlen=APPROACH_HIST_LEN)
                t["area_hist"] = ah
            hh = t.get("h_hist")
            if hh is None:
                hh = deque(maxlen=APPROACH_HIST_LEN)
                t["h_hist"] = hh

            ah.append((now, ar))
            hh.append((now, hr))

            approaching = False
            area_rate = 0.0
            h_rate = 0.0

            if len(ah) >= 2 and len(hh) >= 2:
                dt = ah[-1][0] - ah[0][0]
                if dt >= APPROACH_MIN_DT and is_vehicle(t.get("name", "")):
                    area_rate = compute_growth_rate(ah)
                    h_rate = compute_growth_rate(hh)
                    if (area_rate >= VEH_APPROACH_AREA_RATE_T) or (h_rate >= VEH_APPROACH_HEIGHT_RATE_T):
                        approaching = True

            t["approaching"] = approaching
            t["area_rate"] = float(area_rate)
            t["h_rate"] = float(h_rate)

            thr = threat_score(t["name"], t["pos"], t["dist"], ar)

            if approaching:
                score = max(
                    area_rate / max(VEH_APPROACH_AREA_RATE_T, 1e-6),
                    h_rate / max(VEH_APPROACH_HEIGHT_RATE_T, 1e-6),
                )
                boost = min(APPROACH_THREAT_BOOST_MAX, 0.25 * score)
                thr *= (1.0 + boost)

            active.append({"id": tid, **t, "area_ratio": ar, "threat": thr})

        with detections_lock:
            last_detections = active
        last_yolo_publish_time = time.time()

# ================= SUMMARIES =================

def summarize_objects(detections, sector=None):
    def pluralize(noun: str, n: int) -> str:
        if n == 1:
            return noun
        if noun == "person":
            return "people"
        parts = noun.split()
        last = parts[-1]
        if last.endswith(("s", "x", "z", "ch", "sh")):
            last_pl = last + "es"
        elif last.endswith("y") and len(last) > 1 and last[-2].lower() not in "aeiou":
            last_pl = last[:-1] + "ies"
        else:
            last_pl = last + "s"
        parts[-1] = last_pl
        return " ".join(parts)

    def where_phrase(sec):
        if sec == "left":
            return "on your left"
        if sec == "right":
            return "on your right"
        if sec == "center":
            return "in front of you"
        return "in view"

    if not detections:
        if sector is None:
            return "I don't see anything important."
        if sector == "left":
            return "I don't see anything on your left."
        if sector == "right":
            return "I don't see anything on your right."
        if sector == "center":
            return "I don't see anything in front of you."
        return f"I don't see anything {where_phrase(sector)}."

    filtered = detections if sector is None else [d for d in detections if d.get("pos") == sector]
    if not filtered:
        if sector == "left":
            return "I don't see anything on your left."
        if sector == "right":
            return "I don't see anything on your right."
        if sector == "center":
            return "I don't see anything in front of you."
        return "I don't see anything important."

    counts = Counter(d["name"] for d in filtered)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]

    if len(items) == 1:
        name, n = items[0]
        return f"{n} {pluralize(name, n)} {where_phrase(sector)}."

    prefix = {"left": "On the left: ", "right": "On the right: ", "center": "In front: ", None: "In view: "}.get(sector, "In view: ")
    parts = [f"{n} {pluralize(name, n)}" for name, n in items]
    return prefix + ", ".join(parts) + "."

def build_short_alert_en(name, pos, dist):
    where = POS_EN.get(pos, "near you")
    if dist == "very close":
        return f"{name} very close {where}."
    if dist == "close":
        return f"{name} close {where}."
    return f"{name} {where}."

def aggregate_for_regular_speech(tracks_list):
    if not tracks_list:
        return []
    buckets = {}
    for d in tracks_list:
        k = (d["name"], d["pos"], d["dist"])
        buckets[k] = buckets.get(k, 0) + 1
    items = [(name, pos, dist, cnt) for (name, pos, dist), cnt in buckets.items()]
    items.sort(key=lambda it: danger_rank_tuple(it[0], it[1], it[2]))

    chosen = []
    used_names = set()
    for it in items:
        name = it[0]
        if name in used_names and len(chosen) >= 1:
            continue
        chosen.append(it)
        used_names.add(name)
        if len(chosen) >= 2:
            break
    return chosen

# ================= STATE =================

last_spoken = {}
last_free_time = 0.0
nav_state = STATE_CLEAR
_state_candidate = STATE_CLEAR
_state_candidate_since = 0.0

EMPTY_SCENE_CONFIRM_SEC = 1.2
CLEAR_MIN_GAP_AFTER_OBJECT = 1.4

empty_since = None
last_object_seen_time = 0.0

last_summary_time = 0.0
last_guide_time = 0.0
last_guide_text = ""

last_fps_time = time.time()
fps_counter = 0
fps_value = 0.0

def start_threads():
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=yolo_worker, daemon=True).start()

start_threads()

# ================= MAIN LOOP =================

try:
    TARGET_UI_FPS = 60
    _frame_dt = 1.0 / TARGET_UI_FPS
    _last_ui_t = time.time()
    last_danger_state = False

    while True:
        now = time.time()

        if last_cam_publish_time > 0 and (now - last_cam_publish_time) > CAM_STALL_SEC:
            say("Camera stalled. Restarting camera.", priority=PRIORITY_IMPORTANT)
            with cap_lock:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = open_camera()
                if cap is not None:
                    configure_camera(cap)
            last_cam_publish_time = time.time()

        with frame_lock:
            lf = latest_frame
            cid = latest_frame_id

        if lf is None:
            time.sleep(0.01)
            continue

        display_counter += 1
        if display_frame is None or (display_counter % DISPLAY_EVERY_N == 0):
            display_frame = cv2.resize(lf, (DISPLAY_W, DISPLAY_H), interpolation=DISPLAY_INTERP)

        frame = display_frame.copy()

        h0, w0 = lf.shape[:2]
        frame_area = h0 * w0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            say("Paused." if paused else "Resumed.", priority=PRIORITY_IMPORTANT)
        elif key == ord("1"):
            current_mode = MODE_STREET
            say("Street mode.", priority=PRIORITY_IMPORTANT)
        elif key == ord("2"):
            current_mode = MODE_CANE
            say("Cane mode.", priority=PRIORITY_IMPORTANT)
        elif key == ord("3"):
            current_mode = MODE_SCAN
            say("Scan mode.", priority=PRIORITY_IMPORTANT)
        elif key == ord("w"):
            request_whats_around = True
            request_sector = None
        elif key == ord("a"):
            request_whats_around = True
            request_sector = "left"
        elif key == ord("s"):
            request_whats_around = True
            request_sector = "center"
        elif key == ord("d"):
            request_whats_around = True
            request_sector = "right"

        fps_counter += 1
        if now - last_fps_time >= 1.0:
            fps_value = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        if paused:
            cv2.putText(frame, "PAUSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_value:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            try:
                qsz = speech_queue.qsize()
            except Exception:
                qsz = -1

            cam_age = (time.time() - last_cam_publish_time) if last_cam_publish_time > 0 else 999.0
            cv2.putText(frame, f"CAM id:{cid} age:{cam_age:.2f}s", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, f"YOLO: {yolo_fps:.1f} fps  {yolo_last_ms:.1f} ms", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Speech queue: {qsz}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            continue

        with detections_lock:
            detections = last_detections
            if detections:
                clear_announced = False

        if detections:
            last_object_seen_time = now
            empty_since = None
        else:
            if empty_since is None:
                empty_since = now

        if WALKWAY_ENABLE and (now - _last_walkway_time) >= WALKWAY_INTERVAL_SEC:
            _last_walkway_time = now
            hint_w, x_norm, conf = walkway_hint_from_frame(lf)
            _last_walkway_hint = hint_w
            _last_walkway_x = x_norm

            has_close = any(d.get("dist") in ("close", "very close") for d in detections) if detections else False
            yolo_fresh = (now - last_yolo_publish_time) <= YOLO_STALE_SEC

            if detections and (nav_state != STATE_CLEAR) and (not has_close) and yolo_fresh:
                if conf >= 0.30 and (now - _last_walkway_say) >= WALKWAY_SAY_COOLDOWN:
                    if hint_w == "left":
                        say("Keep slightly left.", priority=PRIORITY_INFO)
                        _last_walkway_say = now
                    elif hint_w == "right":
                        say("Keep slightly right.", priority=PRIORITY_INFO)
                        _last_walkway_say = now

        if now - last_yolo_publish_time > YOLO_STALE_SEC:
            desired_state = STATE_CAUTION
        else:
            desired_state = compute_nav_state(detections)

        if desired_state != _state_candidate:
            _state_candidate = desired_state
            _state_candidate_since = now

        hold = (now - _state_candidate_since)

        need_switch = False
        if _state_candidate == STATE_CLEAR and hold >= CLEAR_HOLD_SEC:
            need_switch = True
        elif _state_candidate == STATE_CAUTION and hold >= CAUTION_HOLD_SEC:
            need_switch = True
        elif _state_candidate == STATE_DANGER and hold >= DANGER_HOLD_SEC:
            need_switch = True

        if need_switch and nav_state != _state_candidate:
            nav_state = _state_candidate

        danger_just_announced = False

        if nav_state == STATE_DANGER:
            if not last_danger_state:
                say("Warning.", priority=PRIORITY_CRITICAL, purge=True)
                danger_just_announced = True
            last_danger_state = True
        else:
            last_danger_state = False

        if nav_state == STATE_CLEAR:
            yolo_fresh = (now - last_yolo_publish_time) <= YOLO_STALE_SEC

            empty_ok = (empty_since is not None) and ((now - empty_since) >= EMPTY_SCENE_CONFIRM_SEC)
            gap_ok = (now - last_object_seen_time) >= CLEAR_MIN_GAP_AFTER_OBJECT

            if yolo_fresh and (not detections) and (not clear_announced) and empty_ok and gap_ok:
                say("Path is clear.", priority=PRIORITY_INFO)
                clear_announced = True
                last_free_time = now

        handled_request = False

        if current_mode == MODE_CANE:
            if detections:
                det_sorted = sorted(detections, key=lambda d: danger_rank_tuple(d["name"], d["pos"], d["dist"]))
                first = det_sorted[0]
                vibrate_by_distance(first["dist"], now)

        elif current_mode == MODE_SCAN:
            if request_whats_around:
                request_whats_around = False
                handled_request = True
                say(summarize_objects(detections, sector=request_sector), priority=PRIORITY_IMPORTANT)

        else:
            if request_whats_around:
                request_whats_around = False
                handled_request = True
                say(summarize_objects(detections, sector=request_sector), priority=PRIORITY_IMPORTANT)

            if detections and (now - last_approach_say_time) >= APPROACH_SAY_COOLDOWN:
                approaching_vehicles = [d for d in detections if d.get("approaching") and is_vehicle(d.get("name", ""))]
                if approaching_vehicles:
                    topv = max(approaching_vehicles, key=lambda d: d.get("threat", 0.0))
                    if topv.get("dist") != "very close":
                        say(f"Vehicle approaching {POS_EN.get(topv.get('pos','center'),'in front of you')}.",
                            priority=PRIORITY_IMPORTANT)
                        last_approach_say_time = now

            if detections and (not handled_request) and (now - last_summary_time > SUMMARY_COOLDOWN):
                has_very_close = any(d["dist"] == "very close" for d in detections)
                if not has_very_close:
                    say(build_scene_summary(detections), priority=PRIORITY_INFO)
                    last_summary_time = now

            if detections and (now - last_guide_time > GUIDE_COOLDOWN):
                has_very_close = any(d["dist"] == "very close" for d in detections)
                if not has_very_close:
                    hint, center_th, best_th = best_direction_hint(detections, w0, frame_area)

                    need_guidance = (
                        center_th >= GUIDE_MIN_CENTER_THREAT and
                        (center_th - best_th >= GUIDE_IMPROVEMENT_ABS) and
                        (center_th >= best_th * GUIDE_IMPROVEMENT_RATIO)
                    )

                    if need_guidance and hint != "forward":
                        guide_text = f"Move {hint}."
                        if guide_text != last_guide_text:
                            say(guide_text, priority=PRIORITY_IMPORTANT)
                            last_guide_text = guide_text
                            last_guide_time = now

            if detections:
                filtered_for_speech = []
                for d in detections:
                    nc = int(d.get("near_confirm", 0))
                    if d["dist"] == "very close":
                        if nc >= VERY_CLOSE_CONFIRM_FRAMES:
                            filtered_for_speech.append(d)
                    elif d["dist"] == "close":
                        if nc >= CLOSE_CONFIRM_FRAMES:
                            filtered_for_speech.append(d)
                    else:
                        filtered_for_speech.append(d)

                chosen = aggregate_for_regular_speech(filtered_for_speech)
                if chosen:
                    parts = [build_short_alert_en(n, p, dist) for (n, p, dist, c) in chosen]
                    text = " ".join(parts)

                    key_say = tuple((n, p, norm_dist_for_speech(dist), min(c, 3)) for (n, p, dist, c) in chosen)
                    cooldown = PERSON_COOLDOWN if (len(chosen) == 1 and chosen[0][0] == "person") else SPEECH_COOLDOWN

                    if (key_say not in last_spoken) or (now - last_spoken[key_say] > cooldown):
                        has_vc = any((dist == "very close") for (_, _, dist, _) in chosen)
                        has_c = any((dist == "close") for (_, _, dist, _) in chosen)

                        if has_vc:
                            if not danger_just_announced:
                                say(text, priority=PRIORITY_CRITICAL, purge=True)
                        elif has_c:
                            say(text, priority=PRIORITY_IMPORTANT)
                        else:
                            say(text, priority=PRIORITY_INFO)

                        last_spoken[key_say] = now
                        last_free_time = now

        cv2.putText(frame, f"FPS: {fps_value:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"MODE: {current_mode} (1=street,2=cane,3=scan)", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Keys: w=scan all, a=left, s=front, d=right, p=pause, q=quit", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        try:
            qsz = speech_queue.qsize()
        except Exception:
            qsz = -1

        cam_age = (time.time() - last_cam_publish_time) if last_cam_publish_time > 0 else 999.0
        cv2.putText(frame, f"CAM id:{cid} age:{cam_age:.2f}s", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"YOLO: {yolo_fps:.1f} fps  {yolo_last_ms:.1f} ms", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Speech queue: {qsz}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if WALKWAY_DRAW_OVERLAY and _last_walkway_x is not None:
            xpix = int(_last_walkway_x * DISPLAY_W)
            cv2.line(frame, (xpix, int(DISPLAY_H * 0.55)), (xpix, DISPLAY_H), (0, 255, 0), 2)
            cv2.putText(frame, f"Walkway: {_last_walkway_hint}", (20, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        now_ui = time.time()
        dt = now_ui - _last_ui_t
        if dt < _frame_dt:
            time.sleep(_frame_dt - dt)
        _last_ui_t = time.time()

        cv2.imshow(window_name, frame)

except Exception as e:
    print("FATAL ERROR:", repr(e))
    raise

finally:
    camera_running = False
    yolo_running = False
    speech_running = False

    with cap_lock:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    cv2.destroyAllWindows()
