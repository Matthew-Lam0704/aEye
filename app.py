from flask import Flask, Response, render_template, jsonify, request
import cv2
import time
import os
import threading
from ultralytics import YOLO
from collections import deque, Counter
import numpy as np
import base64
import pytesseract
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

# --- INITIALIZE MODELS ---
print("Waking up the brains...")
custom_brain = YOLO(CUSTOM_MODEL_NAME)
pose_brain = YOLO(POSE_MODEL_NAME)

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
    "learned": None,
    "furniture": [],
    "people": [],
    "table_items": [],
    "other_items": []
}
last_summary_time = 0


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


def process_frame(frame):
    global last_summary_time, latest_state

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
    table_items, people_found, furniture_found, other_items = [], [], [], []

    # 3) PROCESS CUSTOM BRAIN (Furniture & Items)
    for r in custom_results:
        for box in r.boxes:
            c = box.xyxy[0].cpu().numpy()
            lbl = custom_brain.names[int(box.cls[0])]

        # 🚫 Skip people here — pose model handles them
            if lbl == "person":
                continue

            dist = round(DIST_CALIB / max((c[2] - c[0]), 1), 1)

            if lbl in ["dining table", "desk", "bed", "chair", "couch"]:
                if lbl in ["dining table", "desk"]:
                    table_box = c

                furniture_found.append({
                    "label": lbl,
                    "dist": dist,
                    "x": float((c[0] + c[2]) / 2)
                })

                cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255,0,0), 2)
                cv2.putText(frame, f"{lbl} {int(dist)}m",
                            (int(c[0]), int(c[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            else:
                xm, ym = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2

                if table_box is not None and (table_box[0] < xm < table_box[2]) and (table_box[1] < ym < table_box[3]):
                    table_items.append(lbl)
                else:
                    other_items.append({"label": lbl, "dist": dist, "x": float(xm)})

                cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,255,255), 2)
                cv2.putText(frame, lbl,
                            (int(c[0]), int(c[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # 4) PROCESS POSE BRAIN (Person & Actions)
    for r in pose_results:
        for i, box in enumerate(r.boxes):
            if pose_brain.names[int(box.cls[0])] == "person":
                c = box.xyxy[0].cpu().numpy()
                w, h = (c[2]-c[0]), (c[3]-c[1])
                kps = None
                if r.keypoints is not None:
                    kps = r.keypoints.data[i].cpu().numpy()
                    draw_skeleton(frame, kps)

                raw_action = infer_action(kps, w, h)
                act_str = smooth_action(raw_action)

                people_found.append({"action": act_str, "dist": round(DIST_CALIB/max(w,1),1), "x": float((c[0]+c[2])/2)})

                cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,255,0), 2)
                cv2.putText(frame, f"Person {act_str} {int(round(DIST_CALIB/max(w,1),1))}m",
                            (int(c[0]), int(c[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 5) SPEECH LOGIC (send to browser; browser speaks)
    now = time.time()
    speech = ""
    if (now - last_summary_time) > SPEECH_DELAY:
        parts = []
        if learned_alert:
            parts.append(f"I found your {learned_alert}!")

        if furniture_found:
            f = furniture_found[0]
            parts.append(f"a {f['label']} {int(f['dist'])} meters away")

        if people_found:
            p = people_found[0]
            parts.append(f"a person who is {p['action']} {int(p['dist'])} meters away")

        if table_items:
            parts.append(f"a table with a {table_items[0]} on it")

        if other_items:
            items_by_distance = {}
            for item in other_items:
                d = item["dist"]
                items_by_distance.setdefault(d, []).append(item["label"])

            for d, items in items_by_distance.items():
                if len(items) == 1:
                    parts.append(f"a {items[0]} {int(d)} meters away")
                else:
                    items_str = ", ".join(items[:-1]) + f" and {items[-1]}"
                    parts.append(f"{items_str} {int(d)} meters away")

        if parts:
            speech = "I see " + ", and ".join(parts) + "."
            last_summary_time = now

    # 6) SUMMARY OVERLAY ON VIDEO
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (fw-10, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 40
    cv2.putText(frame, "DETECTIONS:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y += 25

    if learned_alert:
        cv2.putText(frame, f"• Learned Item: {learned_alert}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y += 20

    for f in furniture_found[:3]:
        cv2.putText(frame, f"• {f['label']} ({int(f['dist'])}m)", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        y += 20

    for p in people_found[:2]:
        cv2.putText(frame, f"• Person {p['action']} ({int(p['dist'])}m)", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 20

    for item in table_items[:2]:
        cv2.putText(frame, f"• Item on table: {item}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y += 20

    for item in other_items[:3]:
        cv2.putText(frame, f"• {item['label']} ({int(item['dist'])}m)", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y += 20

    # Update global state for /detections
    with state_lock:
        latest_state = {
            "timestamp": time.time(),
            "speech": speech,
            "learned": learned_alert,
            "furniture": furniture_found,
            "people": people_found,
            "table_items": table_items,
            "other_items": other_items
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    text = pytesseract.image_to_string(gray)
    print("OCR TEXT:", text)
    
    # clean it
    text = " ".join(text.split())
    return jsonify({"text": text})

if __name__ == "__main__":
    # Use 0.0.0.0 if you want to open from your phone on the same Wi-Fi
    app.run(host="0.0.0.0", port=5050, debug=True)
