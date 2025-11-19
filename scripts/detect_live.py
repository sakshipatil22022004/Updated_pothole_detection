from ultralytics import YOLO
import cv2
import geocoder
import time
import os

# Configuration
MODEL_PATH = '../models/best.pt'
SAVE_OUTPUT = True                   # Save output video
OUTPUT_FILE = '../runs/detect/live_output.avi'
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CONF_THRESHOLD = 0.5


# Load YOLO model
print("Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# Setup video writer (optional)
if SAVE_OUTPUT:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
    print(f"💾 Saving live detections to: {os.path.abspath(OUTPUT_FILE)}")

# Fetch GPS once per session (avoid repeated API calls)
g = geocoder.ip('me')
if g.ok:
    lat, lon = g.latlng
    print(f"🌍 GPS Location: Latitude = {lat:.5f}, Longitude = {lon:.5f}")
else:
    lat, lon = None, None
    print("⚠️ Unable to fetch GPS coordinates. Continuing without location data.")

# Start live detection loop
print("🎥 Starting live detection. Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera disconnected or not detected.")
        break

    # Run YOLO inference
    results = model(frame, stream=True, conf=CONF_THRESHOLD)
    detection_found = False

    for r in results:
        for box in r.boxes:
            detection_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Pothole {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add GPS overlay (if available)
            if lat and lon:
                cv2.putText(frame, f"Lat: {lat:.5f}, Lon: {lon:.5f}",
                            (10, FRAME_HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

    if not detection_found:
        cv2.putText(frame, "No potholes detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Pothole Detection - Live", frame)

    # Save video
    if SAVE_OUTPUT:
        out.write(frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting live detection...")
        break

# Clean up
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
print("✅ Live detection ended successfully.")
