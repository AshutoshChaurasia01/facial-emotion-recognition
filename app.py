from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import base64
import logging

# -----------------------------------------------------------------------------
# App + logging setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("emotion-app")

# -----------------------------------------------------------------------------
# Load Keras model ONCE at module level.
# With `gunicorn --preload` this happens a single time and is shared via
# copy-on-write with all workers, drastically cutting memory.
# -----------------------------------------------------------------------------
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

try:
    model = tf.keras.models.load_model("emotion_model.h5")
    # Build the inference function once so TF graph is fixed & warm.
    log.info("Keras model loaded successfully.")
except Exception as e:
    log.error("Failed to load model: %s", e)
    model = None

# -----------------------------------------------------------------------------
# Face detector: OpenCV Haar cascade.
# Replaces MTCNN which was the #1 cause of the 502 crashes:
#   * MTCNN runs 3 CNN stages -> ~2-10s per frame on a 0.1 vCPU free tier
#   * MTCNN allocates large TF graphs per call -> OOM on 512MB Render free tier
# Haar cascade is a classic Viola-Jones detector shipped with OpenCV:
#   * ~5-20ms per frame on the same hardware
#   * negligible extra memory
# -----------------------------------------------------------------------------
_haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(_haar_path)
if face_cascade.empty():
    log.error("Failed to load Haar cascade from %s", _haar_path)
else:
    log.info("Haar cascade loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"emotion": "Model Not Loaded", "confidence": 0}), 500

        data = request.json.get("image")
        if not data:
            return jsonify({"emotion": "No Image", "confidence": 0})

        # Decode base64 data URL -> OpenCV BGR frame
        img_bytes = base64.b64decode(data.split(",")[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"emotion": "No Frame", "confidence": 0})

        original_h, original_w = frame.shape[:2]

        # --- Face detection (Haar) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for speed; Haar is fast even at full res, but 320 wide is plenty.
        scale = 320.0 / original_w if original_w > 320 else 1.0
        small_gray = cv2.resize(gray, (int(original_w * scale), int(original_h * scale))) if scale != 1.0 else gray

        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        log.info("Faces detected: %d", len(faces))

        if len(faces) == 0:
            return jsonify({"emotion": "No Face", "confidence": 0})

        # Take the largest face (most likely the subject)
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # Map back to original coordinates if we downscaled
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, original_w - x)
        h = min(h, original_h - y)

        face = gray[y:y + h, x:x + w]
        if face.size == 0:
            return jsonify({"emotion": "No Face", "confidence": 0})

        # --- Preprocess for the model: 48x48 grayscale, normalized ---
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1).astype(np.float32)

        preds = model.predict(face, verbose=0)
        emotion_index = int(np.argmax(preds))
        emotion = labels[emotion_index]
        confidence = float(np.max(preds)) * 100
        log.info("Predicted: %s (%.2f%%)", emotion, confidence)

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "box": [
                x / original_w,
                y / original_h,
                w / original_w,
                h / original_h
            ]
        })

    except Exception as e:
        log.exception("Predict error")
        return jsonify({"emotion": "Error", "confidence": 0, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
