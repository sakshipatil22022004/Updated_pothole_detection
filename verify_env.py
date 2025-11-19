# verify_env.py
# Simple environment check for the pothole_detection project

import torch
import cv2
import geopy
import geocoder
from ultralytics import YOLO

print("✅ PyTorch version:", torch.__version__)
print("✅ OpenCV version:", cv2.__version__)
print("✅ Geopy version:", geopy.__version__)
print("✅ Geocoder version:", geocoder.__version__)

try:
    model = YOLO("yolov8n.pt")
    print("✅ YOLOv8 model loaded successfully!")
except Exception as e:
    print("❌ YOLOv8 load failed:", e)
