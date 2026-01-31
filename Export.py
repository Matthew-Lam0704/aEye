from ultralytics import YOLO

# 1. Convert your custom model (yolo26n.pt)
model_custom = YOLO('yolo26n.pt')
model_custom.export(format='coreml', nms=True, imgsz=640)

# 2. Convert the pose model for skeleton detection
model_pose = YOLO('yolo26n-pose.pt')
model_pose.export(format='coreml', nms=True, imgsz=640)