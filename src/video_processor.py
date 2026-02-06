# src/video_processor.py
"""
High-level video processing: open video, run detection, draw overlays, write output.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from .detector import detect_on_frame


def is_inside_polygon(polygon: np.ndarray, x: int, y: int) -> bool:
    """
    polygon: np.ndarray shaped (-1,1,2) or Nx2
    """
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0


def process_video(
    input_path: str,
    output_path: str,
    net,
    output_layers,
    classes: List[str],
    polygon_coords: Optional[List[List[int]]] = None,
    conf_thres: float = 0.5,
    nms_thres: float = 0.4,
    show: bool = False,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read frames from video")

    height, width = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pts = None
    if polygon_coords:
        pts = np.array(polygon_coords, np.int32).reshape((-1, 1, 2))

    VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}

    frame_idx = 0
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        boxes, confidences, class_ids, indices = detect_on_frame(net, output_layers, frame, conf_thres, nms_thres)

        # draw polygon zone
        if pts is not None:
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        vehicle_counts = {k: 0 for k in VEHICLE_CLASSES}

        for i in indices:
            try:
                i = int(i)
            except Exception:
                continue
            if i < 0 or i >= len(boxes):
                continue
            x, y, w, h = boxes[i]
            cls_idx = class_ids[i]
            if cls_idx < 0 or cls_idx >= len(classes):
                continue
            cls_name = classes[cls_idx]
            if cls_name in VEHICLE_CLASSES:
                cx = x + w // 2
                cy = y + h // 2
                inside = True if pts is None else is_inside_polygon(pts, cx, cy)
                if inside:
                    vehicle_counts[cls_name] += 1

                # draw
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{cls_name} {confidences[i]:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # overlay counts
        y_off = 40
        for k, v in vehicle_counts.items():
            cv2.putText(frame, f"{k.capitalize()}: {v}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_off += 30

        out.write(frame)

        if show:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output to: {output_path}")
