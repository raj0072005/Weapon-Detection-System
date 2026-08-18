from __future__ import annotations

import argparse
import os

from alerting import AlertManager, CameraDetails

from inference_utils import (
    DEFAULT_WEIGHTS,
    load_model,
    make_output_dir,
    process_video_source,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 weapon detection on a webcam feed.")
    parser.add_argument("--source", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--camera-id", required=True, help="Registered camera identifier.")
    parser.add_argument("--camera-location", required=True, help="Physical camera location for responders.")
    parser.add_argument("--camera-ip", required=True, help="Registered camera IP address.")
    parser.add_argument("--telegram", action="store_true", help="Send alert text and evidence image to the configured Telegram chat.")
    parser.add_argument("--alert-frames", type=int, default=5, help="Consecutive detection frames needed before an alert.")
    parser.add_argument("--alert-cooldown", type=float, default=60.0, help="Minimum seconds between alerts for this camera.")
    parser.add_argument("--alert-min-conf", type=float, default=0.95, help="Minimum pistol/rifle confidence accepted for an alert.")
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
        help="Disable saving the annotated webcam video.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="webcam",
        help="Optional run name for the output folder.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN") if args.telegram else None
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID") if args.telegram else None
    if args.telegram and (not telegram_token or not telegram_chat_id):
        raise SystemExit("--telegram requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
    device = resolve_device(args.device)
    model = load_model(args.weights)
    output_dir = make_output_dir("detect_webcam", args.name, "outputs")
    alert_manager = AlertManager(
        camera=CameraDetails(args.camera_id, args.camera_location, args.camera_ip),
        evidence_dir=output_dir / "alerts",
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        required_frames=args.alert_frames,
        cooldown_seconds=args.alert_cooldown,
        min_confidence=args.alert_min_conf,
    )
    output_path = process_video_source(
        model=model,
        source=args.source,
        output_dir=output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        show=not args.no_show,
        save=not args.no_save,
        window_name="Weapon Webcam Detection",
        alert_manager=alert_manager,
    )

    if output_path is not None:
        print(f"Saved annotated webcam video to {output_path}")
    else:
        print("Webcam processed without saving an output file.")


if __name__ == "__main__":
    main()
