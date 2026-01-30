import os
import cv2
import numpy as np
import pyttsx3 as pyt
import threading
import subprocess
import time
import queue
from ultralytics import YOLO

# --- INITIALIZATION ---
try:
    engine = pyt.init(driverName='nsss')
    # Slow down the speech rate for clarity
    engine.setProperty('rate', 175) 
except Exception as e:
    print(f"TTS Init Warning: {e}")
    engine = None

model = YOLO('yolov8n.pt')

tts_queue = queue.Queue()
tts_stop_event = threading.Event()
tts_enabled = True  
TTS_THROTTLE_SECONDS = 3

# State Variables
debug_mode = False
PAUSE_MODE = False
bad_frame_count = 0
BAD_FRAME_THRESHOLD = 50 

# --- HELPER FUNCTIONS ---

def tts_worker():
    last_msg = None
    last_time = 0
    while not tts_stop_event.is_set():
        try:
            # 1. Check how many messages are waiting
            q_size = tts_queue.qsize()
            
            # 2. If the queue is backed up, skip the old ones to catch up!
            if q_size > 1:
                while not tts_queue.empty():
                    try: tts_queue.get_nowait()
                    except: break
                continue

            text = tts_queue.get(timeout=0.2)
            if text is None: break
            
            now = time.time()
            # 3. Throttle: Don't repeat the same thing too quickly
            if text == last_msg and (now - last_time) < 4: 
                continue
            
            last_msg = text
            last_time = now
            
            # macOS Direct Speech
            subprocess.run(['say', text])
            
        except queue.Empty:
            continue

def speak(text):
    if tts_enabled:
        # Only add to queue if it's empty to prevent lag
        if tts_queue.empty():
            try:
                tts_queue.put_nowait(text)
            except:
                pass

def find_camera():
    for i in [0, 1, 2]:
        c = cv2.VideoCapture(i)
        if c.isOpened():
            return c, i
        c.release()
    return cv2.VideoCapture(0), 0

# --- START THREADS ---
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

cap, cam_idx = find_camera()
cv2.namedWindow("aEye Assistant", cv2.WINDOW_NORMAL)

print("SYSTEM READY. Press 'q' to quit.")

# --- MAIN LOOP ---
# ... (Keep your imports and tts_worker the same) ...

# --- MAIN LOOP ---
try:
    while True:
        success, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF 

        if key == ord('q'): break
        
        if not success or frame is None:
            continue

        # 1. RUN INFERENCE (Increased confidence to 0.6 to avoid fake 'chairs')
        try:
            results = model(frame, conf=0.6, iou=0.45, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    
                    # 2. FILTER OUT FALSE POSITIVES
                    # If it keeps seeing chairs that aren't there, we ignore them
                    if label in ['chair', 'dining table']:
                        continue

                    w_px = max(x2 - x1, 1)
                    # Calibration: Adjust 1500 if the meters feel wrong
                    dist = round(1500 / w_px, 1)
                    
                    # 3. DRAW
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {dist}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 4. SPEAK (Now including distance again)
                    # We use a short format so the audio doesn't lag
                    speak(f"{label}, {int(dist)} meters") 

        except Exception as e:
            print(f"Loop Error: {e}")

        cv2.imshow("aEye Assistant", frame)

finally:
    tts_stop_event.set()
    cap.release()
    cv2.destroyAllWindows()