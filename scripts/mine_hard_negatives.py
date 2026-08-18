"""Automated Hard Negative Mining Script for YOLO Weapon Detection.

Scans background / non-weapon images, identifies frames where the model incorrectly detects
false positive weapons (pistol, knife, rifle), and automatically saves them as hard negative
training samples into the dataset with empty 0-byte label files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
from uuid import uuid4

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Dataset" / "weapon_dataset"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "detect" / "train-gpu-negatives" / "weights" / "best.pt"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine hard negative samples automatically.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to folder containing non-weapon background images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for catching false positives (default: 0.30).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
        help="Dataset split to add hard negatives (default: train).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source)
    weights_path = Path(args.weights)

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    target_img_dir = DEFAULT_DATASET / args.split / "images"
    target_lbl_dir = DEFAULT_DATASET / args.split / "labels"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- HARD NEGATIVE MINING ---")
    print(f"Scanning images in: {source_dir}")
    print(f"Using model weights: {weights_path}")
    print(f"Detection confidence threshold: {args.conf}\n")

    mined_count = 0
    scanned_count = 0

    for item in sorted(source_dir.iterdir()):
        if not item.is_file() or item.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue

        scanned_count += 1
        results = model.predict(str(item), conf=args.conf, verbose=False)[0]

        # Check if the model incorrectly detected any boxes on this negative image
        if results.boxes is not None and len(results.boxes) > 0:
            labels = [results.names[int(box.cls[0].item())] for box in results.boxes]
            confs = [round(float(box.conf[0].item()), 2) for box in results.boxes]

            stamp = uuid4().hex[:6]
            dest_img_name = f"neg_mined_{stamp}_{item.name}"
            dest_lbl_name = f"neg_mined_{stamp}_{item.stem}.txt"

            shutil.copy2(item, target_img_dir / dest_img_name)
            (target_lbl_dir / dest_lbl_name).write_text("", encoding="utf-8")

            mined_count += 1
            print(f"[FALSE POSITIVE MINED #{mined_count}] {item.name} -> Detected {labels} (conf: {confs})")

    print(f"\nScanned {scanned_count} images.")
    print(f"Mined {mined_count} hard negative sample(s) with false positive detections.")
    print(f"Hard negatives saved to: {target_img_dir}")


if __name__ == "__main__":
    main()
