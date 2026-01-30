import cv2
import pyttsx3 as pyt
import threading 
from ultralytics import YOLO

#Initialize TTS engine
engine = pyt.init(driverName='nsss')
model = YOLO('yolov8n.pt') #laptop can handle 'yolov8s.pt' for better accuracy

def speak(text):
    """Function to run in a separate thread to prevent lag"""
    if not engine._isBusy():
        engine.say(text)
        engine.runAndWait()
        
cap = cv2.VideoCapture(1) # 0 is usually the integrated laptop webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
    
    #Run YOLO detection
    results = model(frame, conf=0.6, verbose=False)
    
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
            
            #Speak in the background without stopping the video
            if not threading.active_count() > 1:
                threading.Thread(target=speak, args=(msg,), daemon=True).start()
            
    cv2.imshow("Vision Assistant Demo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
                