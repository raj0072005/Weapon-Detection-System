from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from inference_utils import DEFAULT_DATA, DEFAULT_WEIGHTS, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the merged weapon dataset.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Checkpoint to continue from.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to the merged dataset YAML file.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or a CUDA device id like 0.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Training output directory.",
    )
    parser.add_argument("--name", type=str, default="train-5", help="Run name.")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation this many epochs before the end.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.15,
        help="Mixup augmentation probability.",
    )
    parser.add_argument(
        "--copy-paste",
        type=float,
        default=0.1,
        help="Copy-paste augmentation probability.",
    )
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="Classification loss gain weight.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = YOLO(str(weights_path))
    device = resolve_device(args.device)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=False,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        cls=args.cls_weight,
        workers=args.workers,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()