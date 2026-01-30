import os
import cv2
import numpy as np
import threading
import subprocess
import time
import queue
from collections import Counter
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_NAME = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.6
DISTANCE_CALIBRATION = 1500  # Adjust based on your camera
SPEECH_DELAY = 5             # Seconds between regular summaries
HAZARD_THRESHOLD = 1.5       # Meters for emergency warnings
HAZARD_COOLDOWN = 2          # Seconds between emergency warnings

# --- INITIALIZATION ---
model = YOLO(MODEL_NAME)
tts_queue = queue.Queue()
tts_stop_event = threading.Event()

def tts_worker():
    """Processes the speech queue. Priority is always given to the newest message."""
    while not tts_stop_event.is_set():
        try:
            # If queue is backed up, skip old messages to stay 'live'
            if tts_queue.qsize() > 1:
                while not tts_queue.empty():
                    tts_queue.get_nowait()
                continue

            text = tts_queue.get(timeout=0.2)
            if text is None: break
            
            # macOS native speech command
            subprocess.run(['say', text])
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS Thread Error: {e}")

def get_position(x_center, frame_width):
    """Splits the frame into three vertical zones."""
    if x_center < frame_width / 3:
        return "on your left"
    elif x_center > (2 * frame_width / 3):
        return "on your right"
    else:
        return "straight ahead"

def speak_group(detections):
    """Formats list of (label, dist, pos) into a clean, human sentence."""
    if not detections:
        return

    # Count (label, position) pairs and average their distances
    group_counts = Counter([(d[0], d[2]) for d in detections])
    avg_dists = {}
    for key in group_counts:
        dists = [d[1] for d in detections if (d[0], d[2]) == key]
        avg_dists[key] = sum(dists) / len(dists)

    parts = []
    # Priority: Mention people first
    sorted_keys = sorted(group_counts.keys(), key=lambda x: x[0] != 'person')

    for (label, pos) in sorted_keys:
        count = group_counts[(label, pos)]
        dist = int(avg_dists[(label, pos)])
        
        if count > 1:
            name = "people" if label == "person" else f"{label}s"
            item_str = f"{count} {name}"
        else:
            item_str = f"a person" if label == "person" else f"a {label}"
        
        parts.append(f"{item_str} {pos}, about {dist} meters away")

    if len(parts) == 1:
        sentence = "I see " + parts[0] + "."
    else:
        sentence = "I see " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

    tts_queue.put(sentence)

def check_for_hazards(detections):
    """Detects if any object is dead-ahead and too close."""
    for label, dist, pos in detections:
        if pos == "straight ahead" and dist < HAZARD_THRESHOLD:
            return f"Warning: {label} directly ahead, only {int(dist)} meters."
    return None

# --- MAIN EXECUTION ---

# Start Audio Thread
threading.Thread(target=tts_worker, daemon=True).start()

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("aEye Assistant", cv2.WINDOW_NORMAL)
last_sentence_time = 0
last_hazard_time = 0

print("SYSTEM ONLINE. Press 'q' to quit.")

try:
    while True:
        success, frame = cap.read()
        if not success: break
        
        h, w, _ = frame.shape
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)
        
        frame_detections = []

        for r in results:
            for box in r.boxes:
                # Extract data
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                label = model.names[int(box.cls[0])]
                
                # Logic
                x_center = (x1 + x2) / 2
                position = get_position(x_center, w)
                w_px = max(x2 - x1, 1)
                dist = round(DISTANCE_CALIBRATION / w_px, 1)

                frame_detections.append((label, dist, position))

                # Visual Feedback
                is_hazard = (position == "straight ahead" and dist < HAZARD_THRESHOLD)
                color = (0, 0, 255) if is_hazard else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {dist}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Zone Dividers
        cv2.line(frame, (int(w/3), 0), (int(w/3), h), (255, 255, 255), 1)
        cv2.line(frame, (int(2*w/3), 0), (int(2*w/3), h), (255, 255, 255), 1)

        now = time.time()
        
        # 1. Check for Emergency Hazards
        hazard_msg = check_for_hazards(frame_detections)
        if hazard_msg and (now - last_hazard_time) > HAZARD_COOLDOWN:
            tts_queue.put(hazard_msg)
            last_hazard_time = now
            last_sentence_time = now # Delay the regular summary
            
        # 2. Regular Scene Summary
        elif frame_detections and (now - last_sentence_time) > SPEECH_DELAY:
            speak_group(frame_detections)
            last_sentence_time = now

        cv2.imshow("aEye Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Cleaning up...")
    tts_stop_event.set()
    cap.release()
    cv2.destroyAllWindows()