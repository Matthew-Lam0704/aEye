import cv2
import numpy as np
import pyttsx3 as pyt
import threading
import subprocess
import time
import queue
from ultralytics import YOLO

#Initialize TTS engine
engine = pyt.init(driverName='nsss')
model = YOLO('yolov8n.pt') #laptop can handle 'yolov8s.pt' for better accuracy

# Queue-based TTS worker to avoid spawning many threads and allow clean shutdown
tts_queue = queue.Queue()
tts_stop_event = threading.Event()

# Configurable TTS options
tts_enabled = True  # set False to start muted
TTS_THROTTLE_SECONDS = 3

def tts_worker():
    last_msg = None
    last_time = 0
    while not tts_stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            break
        if not tts_enabled:
            # drop queued messages while muted
            continue
        now = time.time()
        # throttle duplicate messages for configured seconds
        if text == last_msg and (now - last_time) < TTS_THROTTLE_SECONDS:
            continue
        last_msg = text
        last_time = now
        try:
            if not engine.isBusy():
                engine.say(text)
                engine.runAndWait()
            else:
                engine.say(text)
                engine.runAndWait()
        except Exception:
            # Fallback to macOS 'say' command
            try:
                p = subprocess.Popen(['say', text])
                p.wait()
            except Exception:
                pass

# start worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    # respect global toggle
    if not tts_enabled:
        return
    # enqueue text for the worker (non-blocking)
    try:
        tts_queue.put_nowait(text)
    except Exception:
        pass
        
def find_camera(max_idx=4):
    for i in range(max_idx):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            return c, i
        c.release()
    return None, None

cap, cam_idx = find_camera(4)
if cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(0)
    cam_idx = 0

# Make the window visible and resizable
cv2.namedWindow("aEye Assistant", cv2.WINDOW_NORMAL)
cv2.resizeWindow("aEye Assistant", 800, 600)
cv2.moveWindow("aEye Assistant", 100, 100)

try:
    while True:
        if not cap.isOpened():
            # show placeholder while camera unavailable
            frame = 255 * np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available. Press 'r' to retry or 'q' to quit.", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("aEye Assistant", frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('r'):
                try:
                    cap.release()
                except Exception:
                    pass
                cap, cam_idx = find_camera(4)
                continue
            elif key == ord('q'):
                break
            else:
                continue

        success, frame = cap.read()
        if not success:
            # show placeholder and allow retry
            frame = 255 * np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No frame from camera. Press 'r' to retry", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            cv2.imshow("aEye Assistant", frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('r'):
                try:
                    cap.release()
                except Exception:
                    pass
                cap, cam_idx = find_camera(4)
                continue
            elif key == ord('q'):
                break
            else:
                continue

        #Run YOLO detection
        results = model(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                #Safe access to xy and class
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                w_px = x2 - x1
                if w_px <=0:
                    continue

                cls_idx = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                label = model.names[cls_idx]

                dist = round(1500 / w_px, 1) #Calibration constant
                msg = f"{label} at {dist} meters"

                "Draw on screen for your portfolio demo"
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Enqueue message for the TTS worker (avoids spawning many threads)
                speak(msg)

        # Show TTS status and throttle on the frame
        status_text = f"TTS: {'ON' if tts_enabled else 'OFF'} | Throttle: {TTS_THROTTLE_SECONDS}s"
        try:
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # toggle TTS
            tts_enabled = not tts_enabled
            if tts_enabled:
                # brief spoken confirmation
                try:
                    speak("Text to speech enabled")
                except Exception:
                    pass
        elif key == ord('['):
            # decrease throttle window
            TTS_THROTTLE_SECONDS = max(0, TTS_THROTTLE_SECONDS - 1)
            try:
                speak(f"Throttle {TTS_THROTTLE_SECONDS} seconds")
            except Exception:
                pass
        elif key == ord(']'):
            # increase throttle window
            TTS_THROTTLE_SECONDS = TTS_THROTTLE_SECONDS + 1
            try:
                speak(f"Throttle {TTS_THROTTLE_SECONDS} seconds")
            except Exception:
                pass
finally:
    # Clean shutdown
    tts_stop_event.set()
    try:
        tts_queue.put(None)  # signal worker to exit
    except Exception:
        pass
    try:
        engine.stop()
    except Exception:
        pass
    try:
        tts_thread.join(timeout=2)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
                