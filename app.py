from flask import Flask, render_template, jsonify, request
import cv2
import time
import os
import threading
from ultralytics import YOLO
from collections import deque, Counter
import numpy as np
import base64
import pytesseract
try:
    from paddleocr import PaddleOCR  # type: ignore[reportMissingImports]
except Exception:
    PaddleOCR = None

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

ACTION_HISTORY = deque(maxlen=10)  # last ~10 frames of actions

def make_json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    # fallback for anything else (numpy types, arrays, etc.)
    return str(obj)

app = Flask(__name__)

# --- CONFIG ---
CUSTOM_MODEL_NAME = "models/yolo11n.pt"
POSE_MODEL_NAME = "models/yolo11n-pose.pt"

CONF_THRESH = 0.30
DIST_CALIB = 1500
SPEECH_DELAY = 8
URGENT_SPEECH_COOLDOWN = 4
PERSON_HAZARD_MIN_STREAK = 3
PERSON_MIN_WIDTH_RATIO_HAZARD = 0.14
PERSON_MIN_HEIGHT_RATIO_HAZARD = 0.32

DIST_TIER_THRESHOLDS = {
    "very_close": 2.0,
    "close": 4.0,
    "near": 7.0
}

# Person distance feels better when using frame coverage + rough estimate together.
# Tuned so ~1-2m usually lands in close/near (not very_close).
PERSON_TIER_WIDTH_RATIO = {
    "very_close": 0.52,
    "close": 0.34,
    "near": 0.20
}
PERSON_TIER_HEIGHT_RATIO = {
    "very_close": 0.88,
    "close": 0.62,
    "near": 0.38
}

HIGH_RISK_LABELS = {
    "person", "bicycle", "motorcycle", "car", "bus", "truck", "train"
}

# --- INITIALIZE MODELS ---
print("Waking up the brains...")
custom_brain = YOLO(CUSTOM_MODEL_NAME)
pose_brain = YOLO(POSE_MODEL_NAME)

# --- OCR ENGINE SETUP ---
paddle_ocr = None
if PaddleOCR is not None:
    try:
        paddle_ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        print("✅ PaddleOCR loaded")
    except Exception as e:
        paddle_ocr = None
        print("⚠️ PaddleOCR unavailable, falling back to Tesseract:", repr(e))
else:
    print("⚠️ PaddleOCR not installed, using Tesseract")

# --- FEATURE MATCHER SETUP (learned items from .jpg files in folder) ---
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

learned_items = {}
for file in os.listdir():
    if file.lower().endswith(".jpg"):
        name = os.path.splitext(file)[0]
        img = cv2.imread(file, 0)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                learned_items[name] = {"kp": kp, "des": des}

# --- SHARED STATE (for /detections) ---
state_lock = threading.Lock()
latest_state = {
    "timestamp": 0,
    "speech": "",
    "summary_speech": "",
    "urgent_speech": "",
    "learned": None,
    "hazards": [],
    "furniture": [],
    "people": [],
    "table_items": [],
    "other_items": []
}
last_summary_time = 0
last_urgent_events = {}
person_hazard_streaks = {}


def draw_skeleton(frame, kps):
    connections = [(5,6), (5,7), (7,9), (6,8), (8,10),
                   (5,11), (6,12), (11,12), (11,13), (13,15),
                   (12,14), (14,16)]
    for kp in kps:
        if kp[2] > 0.5:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0,0,255), -1)
    for s, e in connections:
        if kps[s][2] > 0.5 and kps[e][2] > 0.5:
            cv2.line(frame,
                     (int(kps[s][0]), int(kps[s][1])),
                     (int(kps[e][0]), int(kps[e][1])),
                     (0,255,0), 2)

def kp_ok(kps, idx, conf=0.5):
    """Return True if keypoint idx exists and has confidence >= conf."""
    return kps is not None and kps.shape[0] > idx and kps[idx][2] >= conf


def has_torso_keypoints(kps, conf=0.45):
    # Shoulders: 5,6 and hips: 11,12 in COCO keypoint order.
    return (
        kp_ok(kps, 5, conf) and kp_ok(kps, 6, conf)
        and kp_ok(kps, 11, conf) and kp_ok(kps, 12, conf)
    )

def infer_action(kps, w, h):
    """
    Returns ONE action string: fallen / sitting / waving / standing
    Uses stricter rules + confidence checks to reduce false positives.
    """

    # If no keypoints, fall back to just standing
    if kps is None:
        return "standing"

    # Convenience getter
    def xy(i):
        return kps[i][0], kps[i][1]

    # --- Waving (wrist clearly above shoulder by a margin) ---
    waving = False
    margin = 0.18 * h  # bigger margin = fewer false positives
    if kp_ok(kps, 5) and kp_ok(kps, 9):   # left shoulder & left wrist
        _, sy = xy(5)
        _, wy = xy(9)
        if wy < sy - margin:
            waving = True
    if kp_ok(kps, 6) and kp_ok(kps, 10):  # right shoulder & right wrist
        _, sy = xy(6)
        _, wy = xy(10)
        if wy < sy - margin:
            waving = True

    # --- Sitting (hips and knees roughly same height BUT knees must be below hips a bit) ---
    sitting = False
    # left hip (11) left knee (13), right hip (12) right knee (14)
    if kp_ok(kps, 11) and kp_ok(kps, 13):
        _, hy = xy(11)
        _, ky = xy(13)
        # knee should be below hip, and not too far below
        if (ky > hy) and abs(hy - ky) < 0.22 * h:
            sitting = True
    if kp_ok(kps, 12) and kp_ok(kps, 14):
        _, hy = xy(12)
        _, ky = xy(14)
        if (ky > hy) and abs(hy - ky) < 0.22 * h:
            sitting = True

    # --- Fallen (much wider than tall + shoulders/hips are not vertically separated much) ---
    fallen = False
    aspect = w / max(h, 1e-6)

    if aspect > 1.6 and kp_ok(kps, 5) and kp_ok(kps, 6) and kp_ok(kps, 11) and kp_ok(kps, 12):
        # compare avg shoulder y vs avg hip y
        _, ls_y = xy(5)
        _, rs_y = xy(6)
        _, lh_y = xy(11)
        _, rh_y = xy(12)
        shoulder_y = (ls_y + rs_y) / 2
        hip_y = (lh_y + rh_y) / 2

        # if shoulders and hips are too close vertically, person is likely horizontal
        if abs(shoulder_y - hip_y) < 0.30 * h:
            fallen = True

    # --- PRIORITY: fallen > sitting > waving > standing ---
    if fallen:
        return "fallen"
    if sitting:
        return "sitting"
    if waving:
        return "waving"
    return "standing"


def smooth_action(new_action):
    """Push action into history and return majority vote."""
    ACTION_HISTORY.append(new_action)
    return Counter(ACTION_HISTORY).most_common(1)[0][0]


def distance_tier(dist_meters):
    if dist_meters <= DIST_TIER_THRESHOLDS["very_close"]:
        return "very_close"
    if dist_meters <= DIST_TIER_THRESHOLDS["close"]:
        return "close"
    if dist_meters <= DIST_TIER_THRESHOLDS["near"]:
        return "near"
    return "far"


def person_distance_tier(
    dist_meters,
    person_width_px,
    person_height_px,
    frame_width_px,
    frame_height_px,
    box_xyxy=None
):
    width_ratio = person_width_px / max(frame_width_px, 1)
    height_ratio = person_height_px / max(frame_height_px, 1)
    close_size_match = (
        width_ratio >= PERSON_TIER_WIDTH_RATIO["close"]
        and height_ratio >= PERSON_TIER_HEIGHT_RATIO["close"]
    )
    near_size_match = (
        width_ratio >= PERSON_TIER_WIDTH_RATIO["near"]
        or height_ratio >= PERSON_TIER_HEIGHT_RATIO["near"]
    )

    # Require strong width+height coverage for very_close to avoid false alarms.
    if (
        dist_meters <= 0.9
        and width_ratio >= PERSON_TIER_WIDTH_RATIO["very_close"]
        and height_ratio >= PERSON_TIER_HEIGHT_RATIO["very_close"]
    ):
        return "very_close"

    # If a person box is clipped, width-only distance can be unstable.
    # Use vertical edge touch (top/bottom) for promotion; side clipping alone is too noisy.
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        edge_touch_x = x1 <= 2 or x2 >= (frame_width_px - 2)
        edge_touch_y = y1 <= 2 or y2 >= (frame_height_px - 2)
        if edge_touch_y and (height_ratio >= 0.86 or (height_ratio >= 0.72 and width_ratio >= 0.50)):
            return "very_close"
        if edge_touch_y and (height_ratio >= 0.50 or width_ratio >= 0.34):
            return "close"
        # Side-only clipping often happens during occlusion; avoid promoting to very_close.
        if edge_touch_x and not edge_touch_y and dist_meters <= 2.1 and close_size_match:
            return "close"

    # Close should need stronger evidence than a single noisy cue.
    # Keep this conservative so normal seated framing is usually near, not close.
    if (
        dist_meters <= 1.45
        or (dist_meters <= 2.15 and close_size_match and height_ratio >= 0.56)
        or (height_ratio >= 0.84 and width_ratio >= 0.34)
    ):
        return "close"

    if dist_meters <= 5.2 or near_size_match:
        return "near"
    return "far"


def direction_from_x(x_mid, frame_width):
    ratio = x_mid / max(frame_width, 1)
    if ratio < 0.33:
        return "left"
    if ratio < 0.67:
        return "center"
    return "right"


def hazard_level_for(label, tier):
    base = 0
    if label in HIGH_RISK_LABELS:
        base = {
            "person": 4,
            "bicycle": 4,
            "motorcycle": 5,
            "car": 5,
            "bus": 5,
            "truck": 5,
            "train": 5
        }.get(label, 3)
    elif tier in {"very_close", "close"}:
        # Treat close unknown obstacles as lower-priority hazards.
        base = 2

    tier_bonus = {
        "very_close": 3,
        "close": 2,
        "near": 1,
        "far": 0
    }[tier]
    return base + tier_bonus


def maybe_build_urgent_speech(hazards, now):
    """Return urgent speech for new/worsened hazards with cooldown."""
    stale_before = now - 30
    for event_key, t in list(last_urgent_events.items()):
        if t < stale_before:
            del last_urgent_events[event_key]

    if not hazards:
        return ""

    top = hazards[0]
    if top["hazard_level"] < 6:
        return ""

    key = f"{top['label']}|{top['direction']}|{top['distance_tier']}"
    last_spoken = last_urgent_events.get(key, 0.0)
    if (now - last_spoken) < URGENT_SPEECH_COOLDOWN:
        return ""

    last_urgent_events[key] = now
    return (
        f"Warning. {top['label']} on your {top['direction']}, "
        f"{top['distance_tier'].replace('_', ' ')}."
    )


def _normalize_text(text):
    return " ".join((text or "").split()).strip()


def run_paddle_ocr(frame):
    if paddle_ocr is None:
        return ""

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = paddle_ocr.ocr(rgb, cls=True)
    except Exception as e:
        print("⚠️ PaddleOCR runtime error:", repr(e))
        return ""

    entries = []

    def collect(node):
        if isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
                and isinstance(node[1][0], str)
            ):
                text = node[1][0]
                try:
                    score = float(node[1][1])
                except Exception:
                    score = 0.0
                entries.append((text, score))
                return
            for child in node:
                collect(child)

    collect(result)
    if not entries:
        return ""

    # Keep strong detections; fallback to softer threshold if needed.
    strong = [t for t, s in entries if s >= 0.45 and len(_normalize_text(t)) >= 2]
    if strong:
        return _normalize_text(" ".join(strong))

    soft = [t for t, s in entries if s >= 0.25 and len(_normalize_text(t)) >= 2]
    return _normalize_text(" ".join(soft))


def run_tesseract_ocr(frame):
    # Build multiple image variants; scene text quality varies a lot per frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )
    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def extract_text_with_conf(img, psm, min_conf=42):
        config = f"--oem 3 --psm {psm} -l eng"
        data = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        words = []
        conf_sum = 0.0
        conf_count = 0
        for i, raw in enumerate(data.get("text", [])):
            token = (raw or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0

            # Keep moderate-confidence words to avoid dropping real text.
            if conf >= min_conf:
                words.append(token)
                conf_sum += conf
                conf_count += 1

        text = " ".join(words).strip()
        avg_conf = (conf_sum / conf_count) if conf_count else 0.0
        return text, avg_conf

    candidates = []
    for img in (adaptive, otsu, denoised):
        for psm in (6, 11):
            text, avg_conf = extract_text_with_conf(img, psm, min_conf=42)
            if text:
                candidates.append((text, avg_conf))

    # Last-resort fallback with no confidence gating.
    if not candidates:
        raw_config = "--oem 3 --psm 11 -l eng"
        raw_text = pytesseract.image_to_string(denoised, config=raw_config)
        raw_text = _normalize_text(raw_text)
        if raw_text:
            candidates.append((raw_text, 0.0))

    # Prefer longer text first, then confidence.
    if candidates:
        return max(candidates, key=lambda c: (len(c[0]), c[1]))[0]
    return ""


def process_frame(frame):
    global last_summary_time, latest_state, person_hazard_streaks

    fh, fw, _ = frame.shape

    # 1) RUN BOTH BRAINS
    custom_results = custom_brain(frame, conf=CONF_THRESH, imgsz=960, verbose=False)
    pose_results = pose_brain(frame, conf=CONF_THRESH, imgsz=960, verbose=False)

    # 2) FEATURE MATCHING (learned items)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = orb.detectAndCompute(gray, None)

    learned_alert = None
    if des_f is not None:
        for name, data in learned_items.items():
            matches = bf.match(data["des"], des_f)
            if len(matches) > 45:
                learned_alert = name
                cv2.putText(frame, f"LEARNED: {name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)
                break

    table_box = None
    table_items = []
    people_found, furniture_found, other_items = [], [], []
    hazards, context_items = [], []

    # 3) PROCESS CUSTOM BRAIN (Furniture & Items)
    for r in custom_results:
        for box in r.boxes:
            c = box.xyxy[0].cpu().numpy()
            lbl = custom_brain.names[int(box.cls[0])]

            # Person uses pose model below.
            if lbl == "person":
                continue

            dist = round(DIST_CALIB / max((c[2] - c[0]), 1), 1)
            xm, ym = float((c[0] + c[2]) / 2), float((c[1] + c[3]) / 2)
            direction = direction_from_x(xm, fw)
            tier = distance_tier(dist)
            hazard_level = hazard_level_for(lbl, tier)
            is_on_table = (
                table_box is not None
                and (table_box[0] < xm < table_box[2])
                and (table_box[1] < ym < table_box[3])
            )
            is_hazard = (lbl in HIGH_RISK_LABELS) and (not is_on_table) and (hazard_level >= 4)

            if lbl in ["dining table", "desk", "bed", "chair", "couch"]:
                if lbl in ["dining table", "desk"]:
                    table_box = c

                furniture_found.append({
                    "label": lbl,
                    "dist": dist,
                    "distance_tier": tier,
                    "direction": direction,
                    "x": xm,
                    "hazard_level": hazard_level
                })
            else:
                if is_on_table:
                    table_items.append(lbl)
                else:
                    other_items.append({
                        "label": lbl,
                        "dist": dist,
                        "distance_tier": tier,
                        "direction": direction,
                        "x": xm,
                        "hazard_level": hazard_level
                    })

            if is_hazard:
                hazards.append({
                    "label": lbl,
                    "dist": dist,
                    "distance_tier": tier,
                    "direction": direction,
                    "hazard_level": hazard_level
                })
            else:
                context_items.append({
                    "label": lbl,
                    "dist": dist,
                    "distance_tier": tier,
                    "direction": direction
                })

            color = (0, 0, 255) if is_hazard else (0, 255, 255)
            label_prefix = "HAZARD" if is_hazard else "OBJ"
            draw_text = f"{label_prefix} {lbl} {direction} {tier}"
            cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color, 2)
            cv2.putText(
                frame,
                draw_text,
                (int(c[0]), int(c[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    # 4) PROCESS POSE BRAIN (Person & Actions)
    seen_person_dirs = set()
    for r in pose_results:
        for i, box in enumerate(r.boxes):
            if pose_brain.names[int(box.cls[0])] != "person":
                continue

            c = box.xyxy[0].cpu().numpy()
            w, h = (c[2] - c[0]), (c[3] - c[1])
            kps = None
            if r.keypoints is not None:
                kps = r.keypoints.data[i].cpu().numpy()
                draw_skeleton(frame, kps)

            raw_action = infer_action(kps, w, h)
            act_str = smooth_action(raw_action)

            dist = round(DIST_CALIB / max(w, 1), 1)
            xm = float((c[0] + c[2]) / 2)
            direction = direction_from_x(xm, fw)
            tier = person_distance_tier(dist, w, h, fw, fh, c)
            hazard_level = hazard_level_for("person", tier)
            width_ratio = w / max(fw, 1)
            height_ratio = h / max(fh, 1)
            person_candidate = (
                width_ratio >= PERSON_MIN_WIDTH_RATIO_HAZARD
                and height_ratio >= PERSON_MIN_HEIGHT_RATIO_HAZARD
                and has_torso_keypoints(kps)
            )
            seen_person_dirs.add(direction)
            if person_candidate:
                person_hazard_streaks[direction] = person_hazard_streaks.get(direction, 0) + 1
            else:
                person_hazard_streaks[direction] = 0
            person_stable = person_hazard_streaks[direction] >= PERSON_HAZARD_MIN_STREAK

            person_obj = {
                "label": "person",
                "action": act_str,
                "dist": dist,
                "distance_tier": tier,
                "direction": direction,
                "x": xm,
                "hazard_level": hazard_level
            }
            people_found.append(person_obj)
            # Seated people at a table are often close in frame but usually not an immediate collision hazard.
            person_is_hazard = (
                person_stable
                and tier in {"very_close", "close"}
                and act_str != "sitting"
            )
            if person_is_hazard:
                hazards.append({
                    "label": "person",
                    "action": act_str,
                    "dist": dist,
                    "distance_tier": tier,
                    "direction": direction,
                    "hazard_level": hazard_level
                })
                color = (0, 0, 255)
                label = f"HAZARD person {direction} {tier}"
            else:
                context_items.append({
                    "label": "person",
                    "dist": dist,
                    "distance_tier": tier,
                    "direction": direction
                })
                color = (0, 255, 255)
                label = f"OBJ person {direction} {tier}"

            cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color, 2)
            cv2.putText(
                frame,
                label,
                (int(c[0]), int(c[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    # Decay streaks for directions not seen this frame.
    for d in list(person_hazard_streaks.keys()):
        if d not in seen_person_dirs:
            person_hazard_streaks[d] = 0

    hazards.sort(key=lambda x: (-x["hazard_level"], x["dist"]))

    # 5) SPEECH LOGIC (urgent first, summary on cadence)
    now = time.time()
    urgent_speech = maybe_build_urgent_speech(hazards, now)
    summary_speech = ""
    if (now - last_summary_time) > SPEECH_DELAY:
        parts = []
        if learned_alert:
            parts.append(f"I found your {learned_alert}")

        if hazards:
            h = hazards[0]
            parts.append(
                f"hazard {h['label']} on your {h['direction']}, "
                f"{h['distance_tier'].replace('_', ' ')}"
            )

        if people_found:
            p = people_found[0]
            parts.append(
                f"person is {p['action']} on your {p['direction']}, "
                f"{p['distance_tier'].replace('_', ' ')}"
            )

        if table_items:
            parts.append(f"table has {table_items[0]} on it")

        if context_items:
            c = context_items[0]
            parts.append(
                f"{c['label']} on your {c['direction']}, "
                f"{c['distance_tier'].replace('_', ' ')}"
            )

        if parts:
            summary_speech = "I see " + ", and ".join(parts) + "."
            last_summary_time = now

    # 6) SUMMARY OVERLAY ON VIDEO
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (fw - 10, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 38
    cv2.putText(frame, "SAFETY VIEW:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 26

    if hazards:
        for hz in hazards[:3]:
            hz_line = (
                f"HAZARD {hz['label']} {hz['direction']} "
                f"{hz['distance_tier'].replace('_', ' ')}"
            )
            cv2.putText(frame, hz_line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2)
            y += 20
    else:
        cv2.putText(frame, "No immediate hazards", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 200, 255), 2)
        y += 20

    for ctx_item in context_items[:2]:
        ctx_line = (
            f"{ctx_item['label']} {ctx_item['direction']} "
            f"{ctx_item['distance_tier'].replace('_', ' ')}"
        )
        cv2.putText(frame, ctx_line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2)
        y += 19

    # Update global state for /detections
    with state_lock:
        latest_state = {
            "timestamp": time.time(),
            "speech": summary_speech,
            "summary_speech": summary_speech,
            "urgent_speech": urgent_speech,
            "learned": learned_alert,
            "hazards": hazards,
            "furniture": furniture_found,
            "people": people_found,
            "table_items": table_items,
            "other_items": other_items,
            "context_items": context_items
        }

    return frame

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/detections")
def detections():
    try:
        with state_lock:
            safe = make_json_safe(latest_state)
        return jsonify(safe)
    except Exception as e:
        print("❌ /detections error:", repr(e))
        # Always return something valid so the frontend doesn't break
        return jsonify({"error": str(e), "timestamp": time.time()}), 200
    
@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/infer", methods=["POST"])
def infer():
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame provided"}), 400

    img_bytes = file.read()
    npbuf = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    # IMPORTANT: process_frame returns the annotated frame
    annotated = process_frame(frame)

    ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    annotated_b64 = ""
    if ok:
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    with state_lock:
        safe = make_json_safe(latest_state)

    safe["annotated_jpg"] = annotated_b64
    return jsonify(safe)

@app.route("/ocr", methods=["POST"])
def ocr():
    print("✅ /ocr called")
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame provided"}), 400

    img_bytes = file.read()
    npbuf = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    text = run_paddle_ocr(frame)
    engine = "paddleocr"

    if not text:
        text = run_tesseract_ocr(frame)
        engine = "tesseract"

    print(f"OCR ENGINE: {engine} | OCR TEXT:", text)
    return jsonify({"text": text, "engine": engine})

if __name__ == "__main__":
    # Use 0.0.0.0 if you want to open from your phone on the same Wi-Fi
    app.run(host="0.0.0.0", port=5050, debug=True)
