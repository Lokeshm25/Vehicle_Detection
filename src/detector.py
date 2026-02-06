# src/detector.py
"""
Frame-level detection utilities.
"""

import numpy as np
import cv2
from typing import List, Tuple


def detect_on_frame(net, output_layers, frame, conf_thres=0.5, nms_thres=0.4):
    """
    Run YOLOv3 detection on a single frame.
    Returns: (boxes, confidences, class_ids, indices)
      - boxes: list of [x,y,w,h]
      - confidences: list of floats
      - class_ids: list of ints (indices into classes list)
      - indices: flattened list of selected indices after NMS
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_thres:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS
    if len(boxes) > 0:
        try:
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
            # idxs can be a list of lists or array; flatten to plain list of ints
            if isinstance(idxs, (list, tuple, np.ndarray)) and len(idxs) > 0:
                # flatten nested arrays
                idxs_flat = []
                for it in idxs:
                    if isinstance(it, (list, tuple, np.ndarray)):
                        idxs_flat.append(int(it[0]))
                    else:
                        idxs_flat.append(int(it))
                indices = idxs_flat
            else:
                indices = []
        except Exception:
            indices = []
    else:
        indices = []

    return boxes, confidences, class_ids, indices
