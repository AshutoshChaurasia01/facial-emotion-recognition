from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import logging
from ai_edge_litert.interpreter import Interpreter

# -----------------------------------------------------------------------------
# App + logging setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("emotion-app")

# -----------------------------------------------------------------------------
# Emotion labels
# -----------------------------------------------------------------------------
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------------------------------------------------------
# Load TFLite model ONCE at module level.
#
# WHY TFLite instead of the original Keras .h5?
#   * The .h5 model was saved with Keras 3.12.0. TF <= 2.15 ships Keras 2 and
#     CANNOT load it — that was crashing the gunicorn master during --preload.
#   * TensorFlow itself imports ~400-500 MB of RAM. On Render's 512 MB free
#     tier this leaves almost nothing for inference -> OOM-kill -> 502.
#   * tflite-runtime is a ~5 MB package with a tiny memory footprint.
#     The .tflite model is 360 KB (vs 4.1 MB .h5) and inference needs < 20 MB.
#
# The .tflite file is produced from the original .h5 by:
#   converter = tf.lite.TFLiteConverter.from_keras_model(model)
#   converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   open("emotion_detection_model.tflite", "wb").write(converter.convert())
# -----------------------------------------------------------------------------
try:
    interpreter = Interpreter(model_path="emotion_detection_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log.info("TFLite model loaded. Input: %s, Output: %s",
             input_details[0]['shape'], output_details[0]['shape'])
except Exception as e:
    log.error("Failed to load TFLite model: %s", e)
    interpreter = None
    input_details = output_details = None

# -----------------------------------------------------------------------------
# Face detector: OpenCV Haar cascade.
# Replaces MTCNN which was too heavy for Render's free tier (3 CNN stages,
# 2-10s per frame, large TF graphs per call -> OOM).
# Haar is ~5-20ms per frame and uses negligible extra memory.
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
        if interpreter is None:
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

        # --- TFLite inference ---
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        emotion_index = int(np.argmax(preds[0]))
        emotion = labels[emotion_index]
        confidence = float(np.max(preds[0])) * 100
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
