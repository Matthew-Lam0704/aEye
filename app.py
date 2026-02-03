from flask import Flask, Response, render_template, jsonify
import cv2
import time
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIG ---
CUSTOM_MODEL_NAME = "yolo26n.pt"
POSE_MODEL_NAME = "yolo26n-pose.pt"
CONF_THRESH = 0.60
DIST_CALIB = 1500
SPEECH_DELAY = 8

# --- LOAD MODELS ---
print("Loading models...")
custom_brain = YOLO(CUSTOM_MODEL_NAME)
pose_brain = YOLO(POSE_MODEL_NAME)

# --- CAMERA ---
cap = cv2.VideoCapture(0)

latest_state = {
    "speech": "",
    "people": [],
    "items": []
}
last_summary_time = 0


def draw_skeleton(frame, kps):
    connections = [
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),
        (13,15),(12,14),(14,16)
    ]
    for x, y, c in kps:
        if c > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)
    for s, e in connections:
        if kps[s][2] > 0.5 and kps[e][2] > 0.5:
            cv2.line(
                frame,
                (int(kps[s][0]), int(kps[s][1])),
                (int(kps[e][0]), int(kps[e][1])),
                (0,255,0), 2
            )


def process_frame(frame):
    global latest_state, last_summary_time

    results = custom_brain(frame, conf=CONF_THRESH, verbose=False)
    pose_results = pose_brain(frame, conf=CONF_THRESH, verbose=False)

    people = []
    items = []

    for r in results:
        for box in r.boxes:
            label = custom_brain.names[int(box.cls[0])]
            if label != "person":
                items.append(label)

    for r in pose_results:
        for i, box in enumerate(r.boxes):
            if pose_brain.names[int(box.cls[0])] == "person":
                kps = r.keypoints.data[i].cpu().numpy()
                draw_skeleton(frame, kps)
                people.append("person")

    now = time.time()
    speech = ""
    if now - last_summary_time > SPEECH_DELAY:
        parts = []
        if people:
            parts.append("a person")
        if items:
            parts.append(f"a {items[0]}")
        if parts:
            speech = "I see " + " and ".join(parts)
            last_summary_time = now

    latest_state = {
        "speech": speech,
        "people": people,
        "items": items
    }

    return frame


def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = process_frame(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return "<h1>aEye backend running</h1><p>Go to /video</p>"


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detections")
def detections():
    return jsonify(latest_state)


if __name__ == "__main__":
    app.run(debug=True)
