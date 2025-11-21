from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import base64
import geocoder

app = Flask(__name__)

# Load YOLO model
MODEL_PATH = r"C:\pothole_detection\scripts\best.pt"

model = YOLO(MODEL_PATH)

# ----------------- MODERN UI TEMPLATE ------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>Pothole Detection</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #eef2f3;
    padding: 30px;
    text-align: center;
}
h1 {
    color: #0056d6;
}
#drop-area {
    border: 3px dashed #007bff;
    padding: 40px;
    width: 60%;
    margin: 30px auto;
    background: #ffffff;
    border-radius: 15px;
}
#drop-area.hover {
    background-color: #e7f1ff;
}
button {
    margin-top: 15px;
    padding: 12px 25px;
    background: #007bff;
    color: white;
    border: none;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
}
button:hover {
    background: #0056b3;
}
#preview {
    margin-top: 20px;
    max-width: 70%;
    border-radius: 10px;
}
#output {
    margin-top: 30px;
    max-width: 70%;
    border-radius: 10px;
    display: none;
}
.loader {
    display: none;
    margin: 20px auto;
    border: 8px solid #f3f3f3;
    border-top: 8px solid #007bff;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>

</head>

<body>

<h1> Pothole Detection (YOLO + GPS)</h1>

<div id="drop-area">
    <h3>Drag & Drop Image Here</h3>
    <p>OR</p>
    <input type="file" id="file-input" accept="image/*">
    <br>
    <button onclick="uploadImage()">Detect Potholes</button>
</div>

<img id="preview">
<div class="loader" id="loader"></div>

<h2 id="gps-info"></h2>

<img id="output">

<script>
let selectedFile = null;

// Drag and drop handling
const dropArea = document.getElementById("drop-area");

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("hover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("hover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("hover");
    selectedFile = e.dataTransfer.files[0];
    previewImage(selectedFile);
});

// File input handling
document.getElementById("file-input").addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    previewImage(selectedFile);
});

// Preview function
function previewImage(file) {
    if (!file) return;
    let reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById("preview").src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Upload to server
function uploadImage() {
    if (!selectedFile) {
        alert("Please select an image first!");
        return;
    }

    document.getElementById("loader").style.display = "block";

    let formData = new FormData();
    formData.append("image", selectedFile);

    fetch("/detect", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        document.getElementById("loader").style.display = "none";

        // Show image result
        document.getElementById("output").src = "data:image/jpeg;base64," + data.image;
        document.getElementById("output").style.display = "block";

        // GPS Info
        document.getElementById("gps-info").innerText = data.gps;
    })
    .catch(err => {
        alert("Error during detection");
        console.error(err);
        document.getElementById("loader").style.display = "none";
    });
}s
</script>

</body>
</html>
"""

# ----------------- DETECTION BACKEND ------------------

@app.route("/detect", methods=["POST"])
def detect_pothole():

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to YOLO-compatible RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model(img_rgb, conf=0.25)

    # Draw boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            conf = float(box.conf[0])

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_bgr, f"Pothole {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    # Get GPS
    g = geocoder.ip("me")
    gps_text = "GPS: Unknown"
    if g.ok:
        gps_text = f"Lat: {g.latlng[0]:.5f}, Lon: {g.latlng[1]:.5f}"

    cv2.putText(img_bgr, gps_text, (10, img_bgr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Convert output to Base64 so browser can show it
    _, buffer = cv2.imencode(".jpg", img_bgr)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": img_base64, "gps": gps_text})


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


if __name__ == "__main__":
    print("🔥 Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
