import os
import io
import numpy as np
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tifffile as tiff
import tensorflow as tf

# ----------------------
# 1. Config
# ----------------------
UPLOAD_TIFF = os.path.join("static", "uploads")
UPLOAD_RGB = os.path.join("static", "rgb")
UPLOAD_PRED = os.path.join("static", "predictions")

for folder in [UPLOAD_TIFF, UPLOAD_RGB, UPLOAD_PRED]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {"tif", "tiff"}

# ----------------------
# 2. Model
# ----------------------
model = tf.keras.models.load_model("best_pre_model.keras", compile=False)
IMG_SIZE = (128, 128)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_input(image_path):
    img = tiff.imread(image_path).astype("float32") / 10000.0
    img_resized = tf.image.resize(img, IMG_SIZE, method="bilinear").numpy()
    img_resized = np.expand_dims(img_resized, axis=0)  # (1,128,128,channels)
    return img_resized

def model_predict(image_path):
    input_tensor = preprocess_input(image_path)
    pred_mask = model.predict(input_tensor)[0]
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return mask.squeeze()

# ----------------------
# 3. Flask App
# ----------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    files = request.files.getlist("file")
    results = []

    for file in files:
        if file.filename == "":
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # --- Save original TIFF ---
            tiff_path = os.path.join(UPLOAD_TIFF, filename)
            file.save(tiff_path)

            # --- Convert to RGB preview ---
            img_array = tiff.imread(tiff_path)

            # Use bands 4,3,2 as RGB if available
            if img_array.ndim == 3 and img_array.shape[-1] >= 4:
                rgb = np.stack([
                    img_array[:, :, 3],  # Red
                    img_array[:, :, 2],  # Green
                    img_array[:, :, 1],  # Blue
                ], axis=-1)
            else:
                # fallback grayscale
                rgb = img_array

            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            rgb_uint8 = (rgb_norm * 255).astype("uint8")

            rgb_name = os.path.splitext(filename)[0] + "_rgb.png"
            rgb_path = os.path.join(UPLOAD_RGB, rgb_name)
            Image.fromarray(rgb_uint8).save(rgb_path)

            # --- Prediction ---
            prediction = model_predict(tiff_path)
            pred_name = os.path.splitext(filename)[0] + "_pred.png"
            pred_path = os.path.join(UPLOAD_PRED, pred_name)
            Image.fromarray(prediction.astype("uint8")).save(pred_path)

            # --- Collect for HTML ---
            results.append((rgb_name, pred_name, filename))

    return render_template("result.html", results=results)

# ----------------------
# 4. Run
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
