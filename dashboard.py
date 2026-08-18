"""Local dashboard for reviewing weapon detections and testing image uploads."""

from __future__ import annotations

from datetime import datetime
import base64
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from ultralytics import YOLO

from scripts.alerting import AlertManager, CameraDetails


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
app = Flask(__name__)
# Prevent accidental uploads that would exhaust the local dashboard process.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
_model: YOLO | None = None
_webcam_alerts: AlertManager | None = None


def inference_device() -> str | int:
    """Use the GPU when it is available, otherwise keep the dashboard usable on CPU."""
    return 0 if torch.cuda.is_available() else "cpu"


def latest_weights() -> Path:
    preferred = ROOT / "runs" / "detect" / "runs" / "detect" / "train-gpu-negatives" / "weights" / "best.pt"
    fallback = ROOT / "runs" / "detect" / "runs" / "detect" / "train-5-2" / "weights" / "best.pt"
    return preferred if preferred.exists() else fallback


def webcam_alert_manager() -> AlertManager | None:
    """Create the local-webcam notifier once with permanent default Telegram credentials."""
    global _webcam_alerts
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or "8935972088:AAEhJmbqSzP96HFOReNQSbLB_TKexU3vUQU"
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or "1394876861"
    if not token or not chat_id:
        return None
    if _webcam_alerts is None:
        _webcam_alerts = AlertManager(
            camera=CameraDetails(
                os.environ.get("DASHBOARD_CAMERA_ID", "LOCAL-WEBCAM"),
                os.environ.get("DASHBOARD_CAMERA_LOCATION", "Local dashboard webcam"),
                os.environ.get("DASHBOARD_CAMERA_IP", "127.0.0.1"),
            ),
            evidence_dir=OUTPUT_ROOT / "dashboard" / "alerts",
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
            required_frames=1,
            cooldown_seconds=10,
            min_confidence=0.55,
            alert_classes={"pistol", "rifle", "knife", "gun"},
        )
    return _webcam_alerts


def get_model() -> YOLO:
    global _model
    if _model is None:
        weights = latest_weights()
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        _model = YOLO(str(weights))
    return _model


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/status")
def status() -> Any:
    weights = latest_weights()
    return jsonify(
        {
            "model_ready": weights.exists(),
            "weights": str(weights.relative_to(ROOT)) if weights.exists() else str(weights),
            "telegram_configured": bool(
                os.environ.get("TELEGRAM_BOT_TOKEN")
                and os.environ.get("TELEGRAM_CHAT_ID")
            ),
        }
    )


@app.get("/api/alerts")
def alerts() -> Any:
    events: list[dict[str, Any]] = []
    for event_path in OUTPUT_ROOT.glob("**/alerts/**/event.json"):
        try:
            event = json.loads(event_path.read_text(encoding="utf-8"))
            evidence = Path(event.get("evidence_path", ""))
            if evidence.exists() and evidence.is_relative_to(OUTPUT_ROOT):
                event["evidence_url"] = f"/api/evidence/{evidence.relative_to(OUTPUT_ROOT).as_posix()}"
            events.append(event)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    events.sort(key=lambda item: item.get("detected_at_utc", ""), reverse=True)
    return jsonify(events[:20])


@app.get("/api/evidence/<path:relative_path>")
def evidence(relative_path: str) -> Any:
    file_path = (OUTPUT_ROOT / relative_path).resolve()
    if not file_path.is_relative_to(OUTPUT_ROOT.resolve()) or not file_path.is_file():
        return jsonify({"error": "Evidence file not found"}), 404
    suffix = file_path.suffix.lower()
    mimetype = "video/mp4" if suffix == ".mp4" else "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png" if suffix == ".png" else None
    return send_file(file_path, mimetype=mimetype)


@app.post("/api/detect-image")
def detect_image() -> Any:
    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "Choose an image first."}), 400

    suffix = Path(image_file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return jsonify({"error": "Use a JPG, PNG, BMP, or WEBP image."}), 400

    try:
        confidence = float(request.form.get("confidence", "0.55"))
        if not 0.01 <= confidence <= 1.0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Confidence must be between 0.01 and 1.0."}), 400

    image_bytes = image_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "The selected file could not be read as an image."}), 400

    result = get_model().predict(image, conf=confidence, imgsz=640, device=inference_device(), verbose=False)[0]
    annotated = result.plot()
    output_dir = OUTPUT_ROOT / "dashboard" / datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"analysis_{uuid4().hex}.jpg"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), annotated)

    manager = webcam_alert_manager()
    if manager is not None:
        manager.consider(result, annotated)

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "label": str(result.names[class_id]),
                    "confidence": round(float(box.conf[0].item()), 4),
                }
            )
    return jsonify(
        {
            "detections": detections,
            "image_url": f"/api/evidence/{output_path.relative_to(OUTPUT_ROOT).as_posix()}",
        }
    )


@app.post("/api/detect-video")
def detect_video() -> Any:
    """Run local detection over an uploaded video and return an annotated copy."""
    video_file = request.files.get("video")
    if video_file is None or not video_file.filename:
        return jsonify({"error": "Choose a video first."}), 400

    suffix = Path(video_file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify({"error": "Use an MP4, AVI, MOV, MKV, or WEBM video."}), 400

    try:
        confidence = float(request.form.get("confidence", "0.55"))
        if not 0.01 <= confidence <= 1.0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Confidence must be between 0.01 and 1.0."}), 400

    output_dir = OUTPUT_ROOT / "dashboard" / datetime.now().strftime("%Y%m%d") / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"upload{suffix}"
    output_path = output_dir / "annotated.mp4"
    video_file.save(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return jsonify({"error": "The selected video could not be opened."}), 400

    writer = None
    frames = 0
    detection_counts: dict[str, int] = {}
    manager = webcam_alert_manager()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = get_model().predict(frame, conf=confidence, imgsz=640, device=inference_device(), verbose=False)[0]
            annotated = result.plot()

            if manager is not None:
                manager.consider(result, annotated)

            if writer is None:
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                # Try H.264 (avc1) first for HTML5 web compatibility, fall back to mp4v if needed
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    return jsonify({"error": "This computer could not create an annotated video."}), 500
            writer.write(annotated)
            frames += 1
            if result.boxes is not None:
                for box in result.boxes:
                    label = str(result.names[int(box.cls[0].item())])
                    detection_counts[label] = detection_counts.get(label, 0) + 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not frames:
        return jsonify({"error": "No readable frames were found in the selected video."}), 400
    return jsonify({
        "video_url": f"/api/evidence/{output_path.relative_to(OUTPUT_ROOT).as_posix()}",
        "frames": frames,
        "detections": detection_counts,
    })


@app.post("/api/detect-frame")
def detect_frame() -> Any:
    """Detect objects in one browser-webcam frame without saving it to disk."""
    frame_file = request.files.get("frame")
    if frame_file is None:
        return jsonify({"error": "Webcam frame is missing."}), 400

    try:
        confidence = float(request.form.get("confidence", "0.55"))
        if not 0.01 <= confidence <= 1.0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Confidence must be between 0.01 and 1.0."}), 400

    frame = cv2.imdecode(np.frombuffer(frame_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "The webcam frame could not be read."}), 400

    result = get_model().predict(frame, conf=confidence, imgsz=640, device=inference_device(), verbose=False)[0]
    ok, encoded = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return jsonify({"error": "Could not render the detection result."}), 500

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({"label": str(result.names[class_id]), "confidence": round(float(box.conf[0].item()), 4)})
    alert_event = None
    manager = webcam_alert_manager()
    if manager is not None:
        alert_event = manager.consider(result, result.plot())
    return jsonify({
        "detections": detections,
        "annotated_image": base64.b64encode(encoded).decode("ascii"),
        "alert_created": alert_event is not None,
    })


DEFAULT_DATASET = ROOT / "Dataset" / "weapon_dataset"


@app.post("/api/add-negative-frame")
def add_negative_frame() -> Any:
    """Save the current webcam frame as a hard negative sample (no weapon) to train set."""
    frame_file = request.files.get("frame")
    if frame_file is None:
        return jsonify({"error": "Webcam frame is missing."}), 400

    frame = cv2.imdecode(np.frombuffer(frame_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "The webcam frame could not be read."}), 400

    target_img_dir = DEFAULT_DATASET / "train" / "images"
    target_lbl_dir = DEFAULT_DATASET / "train" / "labels"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:4]}"
    img_name = f"neg_hard_webcam_{stamp}.jpg"
    lbl_name = f"neg_hard_webcam_{stamp}.txt"

    cv2.imwrite(str(target_img_dir / img_name), frame)
    (target_lbl_dir / lbl_name).write_text("", encoding="utf-8")

    return jsonify({"success": True, "filename": img_name})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
