MODEL: YOLOv8 (custom trained)
FORMAT: ONNX
INPUT SHAPE: 1x3x640x640 (BCHW)
CLASS ID:
0 = pothole

ANDROID INTEGRATION:
- Use ONNX Runtime Mobile
- Preprocess camera frame to 640x640 RGB
- Run inference on best.onnx
- Parse output (8400 boxes, 5 numbers each: x, y, w, h, conf)
- Draw bounding boxes
- Use Android LocationManager/Google FusedLocation API for GPS
