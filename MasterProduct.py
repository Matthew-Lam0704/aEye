import cv2
import threading
import subprocess
import time
import queue
import os
from ultralytics import YOLO

# --- CONFIG ---
# Your custom-trained model for items/furniture
CUSTOM_MODEL_NAME = 'yolo26n.pt' 
# Standard lightweight pose model for skeleton/actions
POSE_MODEL_NAME = 'yolo26n-pose.pt' 

CONF_THRESH = 0.25 
DIST_CALIB = 1500   
SPEECH_DELAY = 8 

# --- INITIALIZE MODELS ---
print("Waking up the brains...")
custom_brain = YOLO(CUSTOM_MODEL_NAME)
pose_brain = YOLO(POSE_MODEL_NAME)
tts_queue = queue.Queue()

# --- FEATURE MATCHER SETUP (For your learned items) ---
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
learned_items = {}
for file in os.listdir():
    if file.endswith(".jpg"):
        name = file.split(".")[0]
        img = cv2.imread(file, 0)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None: 
                learned_items[name] = {"kp": kp, "des": des}

def tts_worker():
    while True:
        text = tts_queue.get()
        if text: 
            subprocess.run(['say', text]) 
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def draw_skeleton(frame, kps):
    connections = [(5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
    for kp in kps:
        if kp[2] > 0.5: 
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0,0,255), -1)
    for s, e in connections:
        if kps[s][2] > 0.5 and kps[e][2] > 0.5:
            cv2.line(frame, (int(kps[s][0]), int(kps[s][1])), (int(kps[e][0]), int(kps[e][1])), (0,255,0), 2)

cap = cv2.VideoCapture(0)
last_summary_time = 0

try:
    while True:
        success, frame = cap.read()
        if not success: 
            break
        fh, fw, _ = frame.shape
        
        # 1. RUN BOTH BRAINS
        custom_results = custom_brain(frame, conf=CONF_THRESH, verbose=False)
        pose_results = pose_brain(frame, conf=CONF_THRESH, verbose=False)
        
        # 2. FEATURE MATCHING (For "100% confidence" items)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = orb.detectAndCompute(gray, None)
        learned_alert = None
        if des_f is not None:
            for name, data in learned_items.items():
                matches = bf.match(data["des"], des_f)
                if len(matches) > 45: 
                    learned_alert = name
                    # Add visual indicator for learned item
                    cv2.putText(frame, f"LEARNED: {name}", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)

        table_box = None
        table_items, people_found, furniture_found = [], [], []

        # 3. PROCESS CUSTOM BRAIN (Furniture & Items)
        for r in custom_results:
            for box in r.boxes:
                c = box.xyxy[0].cpu().numpy()
                lbl = custom_brain.names[int(box.cls[0])]
                dist = round(DIST_CALIB / max((c[2] - c[0]), 1), 1)
                
                if lbl in ['dining table', 'desk', 'bed', 'chair', 'couch']:
                    if lbl in ['dining table', 'desk']: 
                        table_box = c
                    furniture_found.append({'label': lbl, 'dist': dist, 'x': (c[0]+c[2])/2})
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255,0,0), 2)
                    # Add label text for furniture
                    label_text = f"{lbl} {int(dist)}m"
                    cv2.putText(frame, label_text, (int(c[0]), int(c[1])-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                elif lbl != 'person':
                    # Check if on table
                    if table_box is not None:
                        xm, ym = (c[0]+c[2])/2, (c[1]+c[3])/2
                        if (table_box[0] < xm < table_box[2]) and (table_box[1] < ym < table_box[3]):
                            table_items.append(lbl)
                    # Add label text for items
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,255,255), 2)
                    label_text = f"{lbl}"
                    cv2.putText(frame, label_text, (int(c[0]), int(c[1])-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 4. PROCESS POSE BRAIN (Person & Actions)
        for r in pose_results:
            for i, box in enumerate(r.boxes):
                if pose_brain.names[int(box.cls[0])] == 'person':
                    c = box.xyxy[0].cpu() .numpy()
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
                    people_found.append({'action': act_str, 'dist': round(DIST_CALIB/max(w,1),1), 'x': (c[0]+c[2])/2})
                    cv2.rectangle(frame, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,255,0), 2)
                    # Add label text for person with actions
                    label_text = f"Person {act_str} {int(round(DIST_CALIB/max(w,1),1))}m"
                    cv2.putText(frame, label_text, (int(c[0]), int(c[1])-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 5. SPEECH LOGIC
        now = time.time()
        if (now - last_summary_time) > SPEECH_DELAY:
            parts = []
            if learned_alert: 
                parts.append(f"I found your {learned_alert}!")
            if furniture_found and people_found:
                f, p = furniture_found[0], people_found[0]
                side = "to its left" if p['x'] < f['x'] else "to its right"
                parts.append(f"a {f['label']} {int(f['dist'])} meters away, and a person {side} who is {p['action']}")
            elif people_found:
                p = people_found[0]
                parts.append(f"a person who is {p['action']} {int(p['dist'])} meters away")
            elif furniture_found:
                parts.append(f"a {furniture_found[0]['label']}")
            
            if table_items: 
                parts.append(f"a table with a {table_items[0]} on it")

            if parts:
                tts_queue.put("I see " + ", and ".join(parts) + ".")
                last_summary_time = now

        # 6. ADD SUMMARY OVERLAY
        # Create a semi-transparent overlay for summary information
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (fw-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display current detections summary
        y_offset = 40
        cv2.putText(frame, "DETECTIONS:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25
        
        if learned_alert:
            cv2.putText(frame, f"• Learned Item: {learned_alert}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20
        
        if furniture_found:
            for furniture in furniture_found:
                cv2.putText(frame, f"• {furniture['label']} ({int(furniture['dist'])}m)", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += 20
        
        if people_found:
            for person in people_found:
                cv2.putText(frame, f"• Person {person['action']} ({int(person['dist'])}m)", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 20
        
        if table_items:
            for item in table_items:
                cv2.putText(frame, f"• Item on table: {item}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 20

        cv2.imshow("aEye Universal", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()