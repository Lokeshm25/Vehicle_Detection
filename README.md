
# Vehicle Detection System (YOLOv3)

A modular **vehicle detection and counting system** built using **YOLOv3** and **OpenCV DNN**.
The system processes a video, detects vehicles, counts them inside a defined polygon region, and saves an annotated output video.

---

## 🚗 Features

* YOLOv3-based object detection
* Vehicle classes supported:

  * Car
  * Bus
  * Truck
  * Motorbike
* Polygon-based counting zone
* Video input → annotated video output
* Clean, modular Python codebase
* Works locally and in Google Colab

---

## 📁 Project Structure

```
vehicle-detection/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── config.py
    ├── download_assets.py
    ├── yolo_loader.py
    ├── detector.py
    ├── video_processor.py
    └── cli.py
```

### Folder Description

* **config.py** – default constants (paths, thresholds, vehicle classes)
* **download_assets.py** – downloads YOLOv3 weights, config, and COCO names
* **yolo_loader.py** – loads YOLO network and output layers
* **detector.py** – frame-level detection logic
* **video_processor.py** – video loop, drawing, counting, saving output
* **cli.py** – command-line entry point

---

## 🚀 Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Download YOLOv3 Model Files

Run **once** to download required files:

```bash
python -m src.download_assets
```

This downloads:

* `yolov3.cfg`
* `yolov3.weights` (~200 MB)
* `coco.names`

> ⚠️ The weights file is large. Do not commit it to GitHub.

---

## ▶️ Run Vehicle Detection

Basic usage:

```bash
python -m src.cli --input vehicle_video.mp4 --output output.mp4
```

---

## 🔺 Polygon-Based Counting Zone

To count vehicles **inside a specific region**, provide polygon coordinates:

```bash
python -m src.cli \
  --input vehicle_video.mp4 \
  --output output.mp4 \
  --polygon "472,2119 3384,2103 2604,735 1436,711"
```

Coordinates are in **pixel space** relative to the video resolution.

---

## 🖥️ Show Live Processing (Optional)

To display frames while processing:

```bash
python -m src.cli --input vehicle_video.mp4 --show
```

Press **q** to stop early.

---

## 📌 Notes & Limitations

* YOLOv3 is CPU-friendly but slower than modern models.
* Accuracy depends on:

  * Camera angle
  * Lighting
  * Occlusion
* Polygon coordinates must be adjusted per video.
* For better speed and accuracy, consider YOLOv8 + GPU.

---

## 💡 Possible Improvements

* Add DeepSort for vehicle ID tracking
* Export per-frame vehicle counts to CSV
* Switch to YOLOv8 / Ultralytics
* Add a web UI (Streamlit / Gradio)
* Real-time camera input support

---

## 📄 License

YOLOv3 configuration and weights are provided by the original YOLO authors.
Check their license before redistributing weights.

This project code is released under the **MIT License**.

