import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

# Create a synthetic smartphone image with a face inside
img = np.full((480, 640, 3), 30, dtype=np.uint8)
px, py, pw, ph = 200, 40, 240, 400
# Phone body
cv2.rectangle(img, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
# Bezel
cv2.rectangle(img, (px, py), (px + pw, py + ph), (200, 200, 200), 3)
# Screen
cv2.rectangle(img, (px + 8, py + 8), (px + pw - 8, py + ph - 8), (90, 90, 90), -1)
# Face
cv2.circle(img, (320, 240), 50, (180, 160, 140), -1)

results = model.predict(img, classes=[67, 62, 63], conf=0.18, imgsz=640, verbose=False)
print("YOLO detections found:", len(results[0].boxes))
for box in results[0].boxes:
    cls_id = int(box.cls[0].cpu().numpy())
    conf = float(box.conf[0].cpu().numpy())
    print(f"Class: {cls_id} ({model.names[cls_id]}), Conf: {conf:.2f}")
