# src/__init__.py
# package initializer for vehicle-detection src

from . import config, download_assets, yolo_loader, detector, video_processor, cli

__all__ = [
    "config",
    "download_assets",
    "yolo_loader",
    "detector",
    "video_processor",
    "cli",
]
