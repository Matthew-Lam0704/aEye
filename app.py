from flask import Flask, Response, render_template, jsonify
import cv2
import time
import os
import threading
from ultralytics import YOLO

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
CUSTOM_MODEL_NAME = "yolo26n.pt"
POSE_MODEL_NAME = "yolo26n-pose.pt"

CONF_THRESH = 0.60
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

# --- CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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


def process_frame(frame):
    global last_summary_time, latest_state

    fh, fw, _ = frame.shape

    # 1) RUN BOTH BRAINS
    custom_results = custom_brain(frame, conf=CONF_THRESH, verbose=False)
    pose_results = pose_brain(frame, conf=CONF_THRESH, verbose=False)

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
            dist = round(DIST_CALIB / max((c[2] - c[0]), 1), 1)

            if lbl in ["dining table", "desk", "bed", "chair", "couch"]:
                if lbl in ["dining table", "desk"]:
                    table_box = c
                furniture_found.append({"label": lbl, "dist": dist, "x": float((c[0]+c[2])/2)})
                cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255,0,0), 2)
                cv2.putText(frame, f"{lbl} {int(dist)}m", (int(c[0]), int(c[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            elif lbl != "person":
                xm, ym = (c[0]+c[2])/2, (c[1]+c[3])/2
                if table_box is not None and (table_box[0] < xm < table_box[2]) and (table_box[1] < ym < table_box[3]):
                    table_items.append(lbl)
                else:
                    other_items.append({"label": lbl, "dist": dist, "x": float(xm)})

                cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,255,255), 2)
                cv2.putText(frame, f"{lbl}", (int(c[0]), int(c[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # 4) PROCESS POSE BRAIN (Person & Actions)
    for r in pose_results:
        for i, box in enumerate(r.boxes):
            if pose_brain.names[int(box.cls[0])] == "person":
                c = box.xyxy[0].cpu().numpy()
                w, h = (c[2]-c[0]), (c[3]-c[1])
                actions = []

                if w > (h * 1.3):
                    actions.append("fallen")

                if r.keypoints is not None:
                    kps = r.keypoints.data[i].cpu().numpy()
                    draw_skeleton(frame, kps)

                    if abs(kps[11][1] - kps[13][1]) < h * 0.15:
                        actions.append("sitting")
                    if (kps[9][1] < kps[5][1]) or (kps[10][1] < kps[6][1]):
                        actions.append("waving")

                act_str = " and ".join(actions) if actions else "standing"
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


def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = process_frame(frame)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

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

if __name__ == "__main__":
    # Use 0.0.0.0 if you want to open from your phone on the same Wi-Fi
    app.run(host="0.0.0.0", port=5050, debug=True)
