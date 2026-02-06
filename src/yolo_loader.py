# src/yolo_loader.py
"""
Utilities to load YOLOv3 network and COCO class names.
"""

import os
import cv2
import numpy as np


def load_yolo(cfg_path: str, weights_path: str, names_path: str):
    """
    Load YOLO network and return (net, output_layers, classes_list)
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cfg file not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Names file not found: {names_path}")

    net = cv2.dnn.readNet(weights_path, cfg_path)

    # Determine output layer names with compatibility for different OpenCV versions
    layer_names = net.getLayerNames()
    try:
        out_layers_idx = net.getUnconnectedOutLayers()
        # out_layers_idx may be Nx1 or 1D array; flatten safely
        out_layers_idx = [int(x) for x in np.array(out_layers_idx).flatten()]
        output_layers = [layer_names[i - 1] for i in out_layers_idx]
    except Exception:
        # fallback: handle case where getUnconnectedOutLayers returns integers directly
        out_layers_idx = [int(x) for x in np.array(net.getUnconnectedOutLayers()).flatten()]
        output_layers = [layer_names[i - 1] for i in out_layers_idx]

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes
