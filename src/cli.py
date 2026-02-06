# src/cli.py
"""
Command-line entrypoint for vehicle detection.
Usage examples:
    python -m src.cli --input vehicle_video.mp4 --output out.mp4
    python -m src.cli --input vehicle_video.mp4 --output out.mp4 --polygon "472,2119 3384,2103 2604,735 1436,711"
"""

import argparse
from .yolo_loader import load_yolo
from .video_processor import process_video
from .config import DEFAULT_CFG, DEFAULT_WEIGHTS, DEFAULT_NAMES, DEFAULT_CONF_THRES, DEFAULT_NMS_THRES


def parse_polygon(s: str):
    if not s:
        return None
    parts = s.strip().split()
    coords = []
    for p in parts:
        x, y = p.split(",")
        coords.append([int(x), int(y)])
    return coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input video path")
    p.add_argument("--output", "-o", default="output.mp4", help="Output video path")
    p.add_argument("--cfg", default=DEFAULT_CFG, help="YOLO cfg file path")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights file path")
    p.add_argument("--names", default=DEFAULT_NAMES, help="coco.names file path")
    p.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES, help="Confidence threshold")
    p.add_argument("--nms-thres", type=float, default=DEFAULT_NMS_THRES, help="NMS threshold")
    p.add_argument("--polygon", type=str, default="", help="Polygon as 'x1,y1 x2,y2 x3,y3 ...'")
    p.add_argument("--show", action="store_true", help="Show frames during processing")
    args = p.parse_args()

    polygon_coords = parse_polygon(args.polygon) if args.polygon else None

    print("Loading YOLO...")
    net, output_layers, classes = load_yolo(args.cfg, args.weights, args.names)
    print("Starting processing...")
    process_video(
        input_path=args.input,
        output_path=args.output,
        net=net,
        output_layers=output_layers,
        classes=classes,
        polygon_coords=polygon_coords,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        show=args.show,
    )


if __name__ == "__main__":
    main()
