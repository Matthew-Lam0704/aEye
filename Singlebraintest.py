import cv2
import numpy as np
import threading
import subprocess
import time
import queue
from ultralytics import YOLO

# --- CONFIG ---
MODEL_NAME = 'yolov8n-pose.pt' 
CONF_THRESH = 0.25  # Even lower to ensure the table is caught
DIST_CALIB = 1500   
SPEECH_DELAY = 6 

model = YOLO(MODEL_NAME)
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        subprocess.run(['say', text]) 
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# Helper to draw the skeleton
def draw_skeleton(frame, kps):
    # Connections for a human skeleton (Simplified)
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body
        (5, 11), (6, 12), (11, 12),              # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)   # Lower body
    ]
    for kp in kps:
        x, y, conf = kp
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
    for start, end in connections:
        if kps[start][2] > 0.5 and kps[end][2] > 0.5:
            cv2.line(frame, (int(kps[start][0]), int(kps[start][1])), 
                     (int(kps[end][0]), int(kps[end][1])), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)
last_summary_time = 0

try:
    while True:
        success, frame = cap.read()
        if not success: break
        fh, fw, _ = frame.shape
        
        results = model(frame, conf=CONF_THRESH, verbose=False)

        table_box = None
        table_items = []
        people_data = []

        for r in results:
            if r.boxes:
                for i, box in enumerate(r.boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    conf = box.conf[0].cpu().numpy()
                    label = model.names[int(box.cls[0])]
                    dist = round(DIST_CALIB / max((x2 - x1), 1), 1)

                    # 1. PERSON + SKELETON + SITTING/WAVING
                    if label == 'person':
                        actions = []
                        if r.keypoints is not None:
                            kps = r.keypoints.data[i].cpu().numpy()
                            draw_skeleton(frame, kps) # DRAW THE SKELETON
                            
                            # Sitting Logic: Hip (11) and Knee (13) relative height
                            if abs(kps[11][1] - kps[13][1]) < (y2 - y1) * 0.15:
                                actions.append("sitting")
                            # Waving Logic: Wrist (9/10) above Shoulder (5/6)
                            if (kps[9][1] < kps[5][1]) or (kps[10][1] < kps[6][1]):
                                actions.append("waving")
                        
                        action_str = " and ".join(actions) if actions else "standing"
                        people_data.append(f"a person who is {action_str} {dist} meters away")
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"PERSON: {action_str} {dist}m", (x1, y1-10), 1, 1, (0, 255, 0), 2)

                    # 2. TABLE DETECTION
                    elif label in ['dining table', 'desk', 'couch']:
                        table_box = coords
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(frame, f"TABLE {dist}m", (x1, y1-10), 1, 1, (255, 0, 0), 2)

                    # 3. OTHER ITEMS
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        # Check if inside table
                        if table_box is not None:
                            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                            if (table_box[0] < xm < table_box[2]) and (table_box[1] < ym < table_box[3]):
                                table_items.append(label)
                                cv2.putText(frame, "ON TABLE", (x1, y2+15), 1, 0.8, (0, 255, 255), 2)

        # SPEECH
        now = time.time()
        if (now - last_summary_time) > SPEECH_DELAY:
            msg = ""
            if people_data: msg += f"I see {people_data[0]}. "
            if table_box is not None:
                item_list = ", ".join(set(table_items)) if table_items else "nothing"
                msg += f"There is a table with {item_list} on it."
            
            if msg:
                tts_queue.put(msg)
                last_summary_time = now

        cv2.imshow("aEye Skeletal & Table View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()