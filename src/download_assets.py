# src/download_assets.py
"""
Download YOLOv3 assets (cfg, weights, coco.names) if they are not present.
Usage:
    python -m src.download_assets
"""

import os
import sys

ASSETS = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}


def download_file(url, out_path):
    try:
        # prefer wget if available
        import wget
        wget.download(url, out=out_path)
        print()
    except Exception:
        # fallback to requests
        import requests
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {out_path}")


def download(path=".", force=False):
    os.makedirs(path, exist_ok=True)
    for fname, url in ASSETS.items():
        out_path = os.path.join(path, fname)
        if os.path.exists(out_path) and not force:
            print(f"[skip] {fname} already exists")
            continue
        print(f"[download] {fname} -> {out_path}")
        download_file(url, out_path)


if __name__ == "__main__":
    target_dir = "."
    force_flag = False
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] in ("-f", "--force"):
        force_flag = True
    download(path=target_dir, force=force_flag)
