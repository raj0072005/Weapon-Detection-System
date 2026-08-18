from __future__ import annotations

import argparse
from pathlib import Path

from inference_utils import (
    DEFAULT_WEIGHTS,
    load_model,
    make_output_dir,
    process_image_source,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 weapon detection on images.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(Path("Dataset") / "weapon_dataset" / "test" / "images"),
        help="Image file or folder to analyze.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained YOLOv8 weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or a CUDA device id like 0.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="outputs/detect_image",
        help="Directory where annotated images are saved.",
    )
    parser.add_argument("--name", type=str, default=None, help="Optional run name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = load_model(args.weights)
    output_dir = make_output_dir("detect_image", args.name, args.project)
    output_paths = process_image_source(
        model=model,
        source=args.source,
        output_dir=output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
    )

    print(f"Saved {len(output_paths)} annotated image(s) to {output_dir}")


if __name__ == "__main__":
    main()