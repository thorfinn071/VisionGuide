import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from ultralytics import YOLO
import cv2
import win32com.client
import threading
import queue
import time
import winsound
import speech_recognition as sr
import numpy as np
import torch
from collections import deque, Counter

# ================= SETTINGS =================

# Camera is mirrored by driver -> we unmirror in code (no toggle needed)
UNMIRROR_CAMERA = True

# Distance thresholds (tune here)
DIST_VERY_CLOSE_T = 0.26
DIST_CLOSE_T = 0.13

# Hard limits to prevent false "close"
FAR_AREA_MAX_FOR_NEAR = 0.030
FAR_HEIGHT_MAX_FOR_NEAR = 0.22
ABS_FAR_AREA = 0.018
ABS_FAR_HEIGHT = 0.16

CONF_THRESHOLD = 0.35
MIN_BOX_AREA_RATIO = 0.004

YOLO_INTERVAL = 0.12

# Speech
SPEECH_COOLDOWN = 2.5
PERSON_COOLDOWN = 7.0
FREE_PATH_COOLDOWN = 4.0

# Modes
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

CLEAR_CONFIRM_COUNT = 10
clear_counter = 0

# Cane beeps (3 levels)
BEEP_MIN_INTERVAL = 0.12
BEEP_INTERVAL_VERY_CLOSE = 0.20
BEEP_INTERVAL_CLOSE = 0.45
BEEP_INTERVAL_FAR = 0.85
last_beep_time = 0.0

FRAME_SKIP_COUNT = 2

# Tracking
TRACK_MAX_AGE = 8
TRACK_MATCH_DIST = 70.0
TRACK_CONFIRM_FRAMES = 2
TRACK_DIST_HISTORY = 6

# ================= FILTER SETTINGS =================
FRONT_ONLY_DEFAULT = True
IGNORE_FAR_OBJECTS_DEFAULT = True
CENTER_BAND = (0.35, 0.65)

ALLOW_SIDE_CLOSE = True

# ================= HELPERS =================

POS_EN = {
    "left": "on your left",
    "right": "on your right",
    "center": "in front of you"
}

def get_horizontal_position(cx, w):
    lo, hi = CENTER_BAND
    if cx < w * lo:
        return "right"   # swapped
    elif cx > w * hi:
        return "left"    # swapped
    return "center"


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

# ================= SPEECH =================

speaker = win32com.client.Dispatch("SAPI.SpVoice")

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

pick_english_voice(speaker)

speech_queue = queue.Queue(maxsize=1)
speech_running = True

last_enqueued_text = ""
last_enqueued_time = 0.0
ENQUEUE_DEBOUNCE = 1.0

def say(text, urgent=False):
    global last_enqueued_text, last_enqueued_time
    now = time.time()
    if text == last_enqueued_text and (now - last_enqueued_time) < ENQUEUE_DEBOUNCE:
        return
    last_enqueued_text = text
    last_enqueued_time = now

    try:
        if urgent:
            while not speech_queue.empty():
                try:
                    _ = speech_queue.get_nowait()
                    speech_queue.task_done()
                except Exception:
                    break
        else:
            if speech_queue.full():
                try:
                    _ = speech_queue.get_nowait()
                    speech_queue.task_done()
                except Exception:
                    pass

        speech_queue.put_nowait((text, urgent))
    except Exception:
        pass

def speech_worker():
    while speech_running:
        try:
            text, urgent = speech_queue.get(timeout=0.2)
            try:
                flags = 2 if urgent else 0
                speaker.Speak(text, flags)
            except Exception:
                pass
            speech_queue.task_done()
        except queue.Empty:
            pass

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
                say("Paused.", urgent=True)
            elif "resume" in text or "continue" in text:
                paused = False
                say("Resumed.", urgent=True)

            elif "street mode" in text:
                current_mode = MODE_STREET
                say("Street mode.", urgent=True)
            elif "cane mode" in text:
                current_mode = MODE_CANE
                say("Cane mode.", urgent=True)
            elif "scan mode" in text or "overview mode" in text:
                current_mode = MODE_SCAN
                say("Scan mode.", urgent=True)

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
    except Exception:
        pass

# ================= CAMERA =================

def open_camera():
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

cap = open_camera()
if cap is None:
    raise RuntimeError("Camera not found or busy (Zoom/Discord/browser).")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

window_name = "BlindAssistant"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 720)

latest_frame = None
frame_lock = threading.Lock()
camera_running = True

def camera_worker():
    global latest_frame, camera_running
    while camera_running:
        for _ in range(FRAME_SKIP_COUNT):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame

threading.Thread(target=camera_worker, daemon=True).start()

# ================= TRACKING =================

tracks = {}
next_track_id = 1

def match_tracks(dets, existing_tracks):
    if not existing_tracks:
        return [], list(range(len(dets)))

    track_ids = list(existing_tracks.keys())
    track_centers = np.array([existing_tracks[tid]["center"] for tid in track_ids], dtype=np.float32)
    det_centers = np.array([d["center"] for d in dets], dtype=np.float32)

    matches = []
    used_tracks = set()
    used_dets = set()

    for di in range(len(dets)):
        dc = det_centers[di]
        dists = np.linalg.norm(track_centers - dc, axis=1)
        j = int(np.argmin(dists))
        best_tid = track_ids[j]
        best_dist = float(dists[j])

        if best_dist <= TRACK_MATCH_DIST and existing_tracks[best_tid]["name"] == dets[di]["name"] and best_tid not in used_tracks:
            matches.append((di, best_tid))
            used_tracks.add(best_tid)
            used_dets.add(di)

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
        return f"Warning. {name} very close {where}."
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
last_yolo_time = 0.0
last_detections = []

last_fps_time = time.time()
fps_counter = 0
fps_value = 0.0

# ================= MAIN LOOP =================

try:
    while True:
        with frame_lock:
            lf = latest_frame

        if lf is None:
            time.sleep(0.01)
            continue

        frame = lf.copy()

        # IMPORTANT: unmirror camera BEFORE any logic (fixes left/right)
        if UNMIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            say("Paused." if paused else "Resumed.", urgent=True)
        elif key == ord("1"):
            current_mode = MODE_STREET
            say("Street mode.", urgent=True)
        elif key == ord("2"):
            current_mode = MODE_CANE
            say("Cane mode.", urgent=True)
        elif key == ord("3"):
            current_mode = MODE_SCAN
            say("Scan mode.", urgent=True)
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

        now = time.time()

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
            cv2.imshow(window_name, frame)
            continue

        detections = last_detections
        if now - last_yolo_time > YOLO_INTERVAL:
            last_yolo_time = now
            half_flag = bool(use_cuda)

            results = yolo(frame, imgsz=416, verbose=False, half=half_flag)

            h, w = frame.shape[:2]
            frame_area = h * w
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

                    if current_mode == MODE_STREET and name not in STREET_OBJECTS:
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
                    height_ratio = bh / h
                    bottom_ratio = (y2 / h)

                    cx, cy = box_center(x1, y1, x2, y2)
                    pos = get_horizontal_position(cx, w)

                    dist_raw = estimate_distance_by_box(box_area_ratio, height_ratio, bottom_ratio)

                    if not should_keep_detection(current_mode, pos, dist_raw):
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
                }

            active = []
            for tid, t in tracks.items():
                if t["age"] == 0:
                    if current_mode in (MODE_STREET, MODE_CANE) and t.get("seen_count", 0) < TRACK_CONFIRM_FRAMES:
                        continue
                    active.append({"id": tid, **t})

            detections = active
            last_detections = detections

        if current_mode == MODE_CANE:
            if detections:
                det_sorted = sorted(detections, key=lambda d: danger_rank_tuple(d["name"], d["pos"], d["dist"]))
                first = det_sorted[0]
                vibrate_by_distance(first["dist"], now)

        elif current_mode == MODE_SCAN:
            if request_whats_around:
                request_whats_around = False
                say(summarize_objects(detections, sector=request_sector), urgent=True)

        else:
            if request_whats_around:
                request_whats_around = False
                say(summarize_objects(detections, sector=request_sector), urgent=True)

            if detections:
                clear_counter = 0
                chosen = aggregate_for_regular_speech(detections)
                if chosen:
                    parts = [build_short_alert_en(n, p, d) for (n, p, d, c) in chosen]
                    text = " ".join(parts)

                    key_say = tuple((n, p, norm_dist_for_speech(d), min(c, 3)) for (n, p, d, c) in chosen)
                    cooldown = PERSON_COOLDOWN if (len(chosen) == 1 and chosen[0][0] == "person") else SPEECH_COOLDOWN

                    if (key_say not in last_spoken) or (now - last_spoken[key_say] > cooldown):
                        urgent = any((d == "very close") for (_, _, d, _) in chosen)
                        say(text, urgent=urgent)
                        last_spoken[key_say] = now
            else:
                clear_counter += 1
                if clear_counter >= CLEAR_CONFIRM_COUNT:
                    if now - last_free_time > FREE_PATH_COOLDOWN:
                        say("Path is clear.")
                        last_free_time = now
                        clear_counter = CLEAR_CONFIRM_COUNT // 2

        cv2.putText(frame, f"FPS: {fps_value:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"MODE: {current_mode} (1=street,2=cane,3=scan)", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Keys: w=scan all, a=left, s=front, d=right, p=pause, q=quit", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

except Exception as e:
    print("FATAL ERROR:", repr(e))
    raise

finally:
    camera_running = False
    speech_running = False

    try:
        try:
            speaker.Speak("", 2)
        except Exception:
            pass
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()