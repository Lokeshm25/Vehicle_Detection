# src/config.py
"""
Default configuration constants for vehicle detection.
Modify values here or pass alternatives via CLI args.
"""

DEFAULT_CFG = "yolov3.cfg"
DEFAULT_WEIGHTS = "yolov3.weights"
DEFAULT_NAMES = "coco.names"

# Classes considered as vehicles
VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike"]

# Default polygon (example). Replace with your video coordinates if needed.
DEFAULT_POLYGON = [
    [472, 2119],
    [3384, 2103],
    [2604, 735],
    [1436, 711],
]

# Default thresholds
DEFAULT_CONF_THRES = 0.5
DEFAULT_NMS_THRES = 0.4
