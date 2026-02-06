# Vehicle Detection (YOLOv3) — Modular Project

Detect and count vehicles in a video using YOLOv3 + OpenCV DNN. This modular version splits the functionality into small, testable modules.

## Project layout

vehicle-detection/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
├── init.py
├── config.py
├── download_assets.py
├── yolo_loader.py
├── detector.py
├── video_processor.py
└── cli.py


## Quick start

1. Install dependencies:
```bash
pip install -r requirements.txt
Download YOLOv3 assets (only once):

python -m src.download_assets
Run detector:

python -m src.cli --input vehicle_video.mp4 --output out.mp4 \
  --polygon "472,2119 3384,2103 2604,735 1436,711"
Add --show to display frames while processing (press q to quit).

Notes
yolov3.weights is ~200MB; download once and keep locally.

Polygon coordinates are pixel coordinates relative to video resolution; use the first frame to find suitable points.

For speed / accuracy improvements consider upgrading to YOLOv8 or enabling OpenCV CUDA backend (requires CUDA-enabled OpenCV build).

To log counts per frame, add a logger module (optional enhancement).

Next steps (optional)
Add DeepSort for persistent tracking (ID per vehicle)

Output per-frame CSV logs for analytics

Migrate to YOLOv8 for faster inference on GPUs

Add a small web UI (Flask/Streamlit/Gradio) for uploading videos and previewing outputs


---

## How to run (summary)

1. Put a video file (e.g., `vehicle_video.mp4`) in the project root or pass full path.
2. If you haven't already, download assets:
   ```bash
   python -m src.download_assets
Run CLI:

python -m src.cli --input vehicle_video.mp4 --output out.mp4 --polygon "472,2119 3384,2103 2604,735 1436,711"
