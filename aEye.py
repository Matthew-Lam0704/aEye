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

# Debug / UI behavior
debug_mode = False  # press 'd' to toggle verbose prints
last_frame = None
last_frame_time = 0
STALE_FRAME_MAX_SECONDS = 5  # show last frame up to this many seconds if camera reads fail

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
            now = time.time()
            if last_frame is not None and (now - last_frame_time) < STALE_FRAME_MAX_SECONDS:
                disp = last_frame.copy()
                cv2.putText(disp, "Camera unavailable — showing last frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                disp = 255 * np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(disp, "Camera not available. Press 'r' to retry or 'q' to quit.", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("aEye Assistant", disp)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('r'):
                try:
                    cap.release()
                except Exception:
                    pass
                cap, cam_idx = find_camera(4)
                if debug_mode:
                    print(f"Retrying camera scan, got index: {cam_idx}")
                continue
            elif key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
                continue
            else:
                continue

        success, frame = cap.read()
        if not success or frame is None:
            # show last valid frame if recent, else placeholder and allow retry
            now = time.time()
            if last_frame is not None and (now - last_frame_time) < STALE_FRAME_MAX_SECONDS:
                disp = last_frame.copy()
                cv2.putText(disp, "No frame (showing last). Press 'r' to retry.", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imshow("aEye Assistant", disp)
            else:
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
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
                continue
            else:
                continue

        #Run YOLO detection (protected)
        try:
            results = model(frame, conf=0.5, verbose=False)
            inference_error = None
        except Exception as e:
            results = []
            inference_error = str(e)
            if debug_mode:
                print(f"YOLO inference error: {e}")

        # record last good frame for fallback
        if frame is not None:
            last_frame = frame.copy()
            last_frame_time = time.time()

        for r in results:
            for box in r.boxes:
                #Safe access to xy and class
                try:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                except Exception:
                    continue
                w_px = x2 - x1
                if w_px <=0:
                    continue

                cls_idx = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                label = model.names[cls_idx]

                dist = round(1500 / w_px, 1) #Calibration constant
                msg = f"{label} at {dist} meters"

                # Draw on screen
                cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Enqueue message for the TTS worker (avoids spawning many threads)
                speak(msg)

        # Show inference error overlay if present
        if inference_error is not None:
            cv2.putText(frame, "Inference error - see console" , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

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
                