# 🚗 Pothole Detection API (YOLOv8)

A lightweight, mobile‑friendly pothole detection system built using **YOLOv8**, designed for integration with Android/iOS applications. This project provides a clean API server, modular scripts, and optimized model files suitable for real‑time pothole detection.

---

## 🌟 Key Features

* **YOLOv8‑based pothole detection** (custom‑trained model)
* **REST API server** for mobile/IoT integration
* **Upload image → Get detection result**
* **Bounding boxes + confidence scores**
* **Supports Android live camera input**
* **GPS tagging support**
* **ONNX model available for on‑device inference**

---

## 📁 Project Structure

```
pothole_detection/
│
├── scripts/
│   ├── api_server.py          # Flask API server
│   ├── detect_image.py        # Test image detection
│   ├── detect_live.py         # Webcam detection (desktop)
│   ├── test_inference.py      # Quick model test
│   ├── get_location.py        # GPS utility
│   └── train.py               # YOLO training script
│
├── pothole_model_mobile/
│   ├── best.onnx              # ONNX model for Android
│   ├── classes.txt            # Model labels
│   └── README_for_android.txt # Android integration guide
│
├── test_sampless/             # Sample images
├── .gitignore
├── requirements.txt           # API dependencies
├── verify_env.py              # Environment test
└── yolov8_pothole.yaml        # Dataset configuration
```

---

## ⚙️ Installation

### 1️⃣ Create & activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API Server

```bash
python scripts/api_server.py
```

The server starts at:

```
http://127.0.0.1:5000
```

### Endpoints

| Endpoint  | Method | Description                            |
| --------- | ------ | -------------------------------------- |
| `/`       | GET    | Web UI for uploading image             |
| `/detect` | POST   | Accepts image → returns detected image |

---

## 📱 Mobile App Integration

Your Android/iOS app can:

1. Capture an image from live camera
2. Send it to `/detect` as multipart form‑data
3. Receive processed image with bounding boxes
4. Display or store GPS location (optional)

For ONNX‑based on‑device inference, refer to:

```
pothole_model_mobile/README_for_android.txt
```

---

## 📦 What You Should Provide to Mobile Developer

* `pothole_model_mobile/best.onnx`
* `pothole_model_mobile/classes.txt`
* API URL (if using server inference)
* Example request code (included in Android README)

---

## 🧠 Model Training

Model is trained using YOLOv8 with custom pothole dataset. Training script:

```
scripts/train.py
```

Dataset configuration file:

```
yolov8_pothole.yaml
```

---

## 👩‍💻 Author

**Sakshi Patil**
Pothole Detection System · YOLOv8 · API Deployment

For any issue, create a GitHub Issue in the repository.
