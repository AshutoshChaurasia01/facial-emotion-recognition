from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import base64
from mtcnn import MTCNN

app = Flask(__name__)


model = tf.keras.models.load_model("emotion_model.h5")

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

detector = MTCNN()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"emotion": "No Frame", "confidence": 0})

        original_h, original_w = frame.shape[:2]

        small_frame = cv2.resize(frame, (320, 240))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        
        faces = detector.detect_faces(rgb_small)
        print("Faces detected:", len(faces))

        if len(faces) == 0:
            return jsonify({"emotion": "No Face", "confidence": 0})

        x, y, w, h = faces[0]["box"]
        x, y = abs(x), abs(y)

        
        scale_x = original_w / 320
        scale_y = original_h / 240

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            return jsonify({"emotion": "No Face", "confidence": 0})

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        
        preds = model.predict(face, verbose=0)
        emotion_index = int(np.argmax(preds))
        emotion = labels[emotion_index]
        confidence = float(np.max(preds)) * 100

        print(f"Predicted: {emotion} ({confidence:.2f}%)")

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
        print("ERROR:", e)
        return jsonify({"emotion": "Error", "confidence": 0})

if __name__ == "__main__":
    app.run(debug=False)
