from ultralytics import YOLO

# Load trained model
model = YOLO('models/best.pt')

# Run inference on your test images
results = model.predict(
    source='dataset/test/images',  # Path to your test folder
    conf=0.25,                     # Confidence threshold
    save=True,                     # Save output images
    show=False                     # Don't open windows
)

print("✅ Detection complete. Check 'runs/detect/predict' folder for results.")
