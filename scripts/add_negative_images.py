"""Utility for adding negative (background / no-weapon) images to the YOLO dataset.

Negative images contain normal background scenes (e.g., holding a phone, mug, wallet,
or empty room) without any weapons. They teach the YOLO model to eliminate false positives.
"""

from __future__ import annotations

import argparse

import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Dataset" / "weapon_dataset"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add negative (background) images to YOLO dataset.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a file or folder of negative background images.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
        help="Dataset split to add negative images to (default: train).",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Interactively capture negative background images from your webcam.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to merged weapon dataset root.",
    )
    return parser


def add_negative_file(image_path: Path, target_img_dir: Path, target_lbl_dir: Path) -> str:
    stamp = uuid4().hex[:8]
    dest_img_name = f"neg_{stamp}_{image_path.name}"
    dest_lbl_name = f"neg_{stamp}_{image_path.stem}.txt"

    shutil.copy2(image_path, target_img_dir / dest_img_name)
    (target_lbl_dir / dest_lbl_name).write_text("", encoding="utf-8")
    return dest_img_name


def capture_negatives_from_webcam(target_img_dir: Path, target_lbl_dir: Path) -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0). Make sure no other process is using the camera.")

    print("\n--- WEBCAM NEGATIVE IMAGE CAPTURE ---")
    print("Press SPACEBAR to capture a negative background frame.")
    print("Press ESC or 'q' to stop capturing.\n")

    captured_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Captured: {captured_count} neg images | SPACE: capture | ESC/q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture Negative Images", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:4]}"
                dest_img_name = f"neg_webcam_{stamp}.jpg"
                dest_lbl_name = f"neg_webcam_{stamp}.txt"

                cv2.imwrite(str(target_img_dir / dest_img_name), frame)
                (target_lbl_dir / dest_lbl_name).write_text("", encoding="utf-8")

                captured_count += 1
                print(f"Captured negative frame {captured_count}: {dest_img_name}")

            elif key in (27, ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_count


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    target_img_dir = dataset_root / args.split / "images"
    target_lbl_dir = dataset_root / args.split / "labels"

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    added_count = 0

    if args.webcam:
        added_count = capture_negatives_from_webcam(target_img_dir, target_lbl_dir)
    elif args.source:
        source_path = Path(args.source)
        if source_path.is_file() and source_path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            add_negative_file(source_path, target_img_dir, target_lbl_dir)
            added_count = 1
        elif source_path.is_dir():
            for item in source_path.iterdir():
                if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                    add_negative_file(item, target_img_dir, target_lbl_dir)
                    added_count += 1
        else:
            raise FileNotFoundError(f"Source not found or invalid image: {source_path}")
    else:
        print("Please specify either --source <folder_or_file> or --webcam")
        return

    print(f"\nSuccessfully added {added_count} negative background image(s) to '{args.split}' split.")
    print(f"Images location: {target_img_dir}")
    print(f"Labels location: {target_lbl_dir}")


if __name__ == "__main__":
    main()
