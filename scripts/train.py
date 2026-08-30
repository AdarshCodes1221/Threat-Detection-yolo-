#!/usr/bin/env python3
"""Reconstruct a YOLO11 training run from thesis-recorded settings."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


VARIANTS = {
    "yolo11n": {"model": "yolo11n.pt", "patience": 35},
    "yolo11s": {"model": "yolo11s.pt", "patience": 35},
    "yolo11s_preprocessing": {
        "model": "yolo11s.pt",
        "patience": 35,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "mosaic": 0.7,
        "mixup": 0.05,
    },
    "yolo11m": {"model": "yolo11m.pt", "patience": 40},
    "yolo11l": {"model": "yolo11l.pt", "patience": 40},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a recorded YOLO11 experiment configuration."
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--data", default="config/data.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = dict(VARIANTS[args.variant])
    model_name = selected.pop("model")
    model = YOLO(model_name)
    model.train(
        data=args.data,
        epochs=150,
        batch=8,
        imgsz=832,
        optimizer="auto",
        device=args.device,
        project=args.project,
        name=args.name or args.variant,
        **selected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
