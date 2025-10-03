# WaterSegNet Deployment

**Flask-based web application for deploying the WaterSegNet model.**  
This app allows users to upload multispectral TIFF satellite images, visualize RGB components, and generate water segmentation masks using a trained U-Net model.  

---

## Features

- **Upload TIFF images** for prediction.  
- **Automatic RGB extraction** saved in `static/uploads_rgb/`.  
- **Original TIFFs** stored in `static/uploads_tiff/`.  
- **Predicted masks** stored in `static/predictions/` (saved as `.png`).  
- **Interactive results page**:
  - Shows RGB version of uploaded image.
  - Displays model prediction.
- Simple navigation (return to upload page after predictions).

---

## Requirements

- Python 3.9+
- Flask
- TensorFlow / Keras
- NumPy
- tifffile
- Matplotlib


##  Running the App

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/WaterSegNet-Deployment.git
cd WaterSegNet-Deployment
````

2. **Place your trained model** in the project root (e.g., `best_pre_model.keras`).

3. **Start the Flask server**:

```bash
python app.py
```

4. **Open your browser** and go to:

```
http://127.0.0.1/
```

5. **Upload TIFF images** to view:

   * Original RGB preview
   * Predicted water segmentation mask
   * Download links for predictions

