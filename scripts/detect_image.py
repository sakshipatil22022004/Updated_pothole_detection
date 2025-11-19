from ultralytics import YOLO
import cv2
import os

# Load your trained model
model = YOLO('../models/best.pt')

# Path to images
source_path = 'C:/pothole_detection/dataset/test/images'

# Run detection without displaying each image
results = model.predict(source=source_path, show=False, save=True, conf=0.5)

print("\n✅ Detection complete!")
print(f"Results are saved in: {os.path.abspath('runs/detect/predict')}")

# OPTIONAL: Preview one sample image result safely
sample_image = os.path.join('runs/detect/predict', os.listdir('runs/detect/predict')[0])
img = cv2.imread(sample_image)
cv2.imshow("Sample Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
