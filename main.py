import base64
import io
import os
import urllib.request

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

IMAGE_SIZE   = 224
ROI_FRAC     = 0.60
RESIZE_SHORT = int(round(IMAGE_SIZE / ROI_FRAC))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/model.onnx")
MODEL_URL  = os.environ.get("MODEL_URL", "")

_session = None

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL not set and model.onnx not found")
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

def get_session():
    global _session
    if _session is None:
        download_model()
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session

def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = RESIZE_SHORT / min(w, h)
    new_w = max(IMAGE_SIZE, round(w * scale))
    new_h = max(IMAGE_SIZE, round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - IMAGE_SIZE) // 2
    top  = (new_h - IMAGE_SIZE) // 2
    img  = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    arr  = (arr - MEAN) / STD
    arr  = arr.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(arr, dtype=np.float32)

def l2_normalize(v):
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True, silent=True) or {}
    b64 = data.get("image", "")
    if not b64:
        return jsonify({"error": "image field missing"}), 400
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        sess = get_session()
    except Exception as e:
        return jsonify({"error": f"model load failed: {e}"}), 500
    inp  = sess.get_inputs()[0].name
    out  = sess.get_outputs()[0].name
    raw  = sess.run([out], {inp: preprocess(img)})[0][0]
    return jsonify({"embedding": l2_normalize(raw).tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
