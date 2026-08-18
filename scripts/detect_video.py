from __future__ import annotations

import argparse

from inference_utils import (
    DEFAULT_WEIGHTS,
    load_model,
    make_output_dir,
    process_video_source,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 weapon detection on a video file.")
    parser.add_argument("--source", type=str, required=True, help="Path to the input video.")
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
    parser.add_argument("--no-show", action="store_true", help="Disable the preview window.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the annotated output video.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name for the output folder.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = load_model(args.weights)
    output_dir = make_output_dir("detect_video", args.name, "outputs")
    output_path = process_video_source(
        model=model,
        source=args.source,
        output_dir=output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        show=not args.no_show,
        save=not args.no_save,
        window_name="Weapon Video Detection",
    )

    if output_path is not None:
        print(f"Saved annotated video to {output_path}")
    else:
        print("Video processed without saving an output file.")


if __name__ == "__main__":
    main()