from ultralytics import YOLO

# Create YOLO model
model = YOLO('yolov8n.pt')

# Use absolute path (make sure forward slashes are used)
data_path = 'C:/pothole_detection/yolov8_pothole.yaml'

# Train the model
model.train(
    data=data_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name='pothole_detector3'
)
