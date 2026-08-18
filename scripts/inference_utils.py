from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import TYPE_CHECKING

import os

import cv2
import torch
from ultralytics import YOLO

if TYPE_CHECKING:
    from alerting import AlertManager


ROOT = Path(__file__).resolve().parents[1]
# Keep every entry point on the same selected model.  Fall back to train-4
# only when the newer weights are unavailable.
PREFERRED_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "detect" / "train-gpu-negatives" / "weights" / "best.pt"
FALLBACK_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "detect" / "train-5-2" / "weights" / "best.pt"
DEFAULT_WEIGHTS = PREFERRED_WEIGHTS if PREFERRED_WEIGHTS.exists() else FALLBACK_WEIGHTS
DEFAULT_DATA = ROOT / "Dataset" / "weapon_dataset" / "data.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_device(device: str | int | None) -> str | int:
    if device is None:
        return 0 if torch.cuda.is_available() else "cpu"

    if isinstance(device, str):
        device = device.strip().lower()
        if device == "auto":
            return 0 if torch.cuda.is_available() else "cpu"
        if device.isdigit():
            return int(device)

    return device


def load_model(weights: str | Path = DEFAULT_WEIGHTS) -> YOLO:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    return YOLO(str(weights_path))


def make_output_dir(task: str, run_name: str | None = None, project_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = _slugify(run_name or stamp)
    output_dir = Path(project_root) / task / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    if source.isdigit():
        return int(source)
    return source


def build_capture_backends(source: str | int | None = None) -> list[int]:
    normalized_source = normalize_source(source) if source is not None else 0
    if isinstance(normalized_source, int):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_video_capture(source: str | int, backend_priority: list[int] | None = None) -> tuple[cv2.VideoCapture | None, int | None]:
    backends = backend_priority or build_capture_backends(source)
    normalized_source = normalize_source(source)

    if isinstance(normalized_source, int):
        index_candidates = []
        for index in [normalized_source, 0, 1, 2, -1]:
            if index not in index_candidates:
                index_candidates.append(index)
        for index in index_candidates:
            for backend in backends:
                capture = cv2.VideoCapture(index, backend)
                if capture.isOpened():
                    if index != normalized_source:
                        print(f"Opened camera source {index} (requested {source}) using backend {backend}")
                    return capture, backend
        return None, None

    source_str = str(normalized_source)
    for backend in backends:
        capture = cv2.VideoCapture(source_str, backend)
        if capture.isOpened():
            if backend is not None:
                print(f"Opened video source {source} using backend {backend}")
            return capture, backend

    return None, None


def collect_image_paths(source: str | Path) -> list[Path]:
    source_path = Path(source)
    if source_path.is_dir():
        return sorted(
            path for path in source_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Image source not found: {source_path}")
    return [source_path]


def process_image_source(
    model: YOLO,
    source: str | Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str | int,
) -> list[Path]:
    output_paths: list[Path] = []
    for image_path in collect_image_paths(source):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        result = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        annotated = result.plot()
        output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        output_paths.append(output_path)

    return output_paths


def process_video_source(
    model: YOLO,
    source: str | int,
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str | int,
    show: bool = True,
    save: bool = True,
    window_name: str = "Weapon Detection",
    alert_manager: "AlertManager | None" = None,
) -> Path | None:
    normalized_source = normalize_source(source)
    if isinstance(normalized_source, str):
        source_path = Path(normalized_source)
        if not source_path.exists() and "://" not in normalized_source:
            raise FileNotFoundError(f"Video source not found: {source_path}")

    capture, backend_used = open_video_capture(normalized_source)
    if capture is None:
        raise RuntimeError(f"Unable to open video source: {source}")
    if backend_used is not None:
        print(f"Opened video source {source} with backend {backend_used}")

    output_path = output_dir / f"{_source_name(source)}.mp4"
    writer = None
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            annotated = result.plot()
            if alert_manager is not None:
                event = alert_manager.consider(result, annotated)
                if event is not None:
                    print(
                        "ALERT CREATED: "
                        f"{event['event_id']} ({event['camera']['location']}) - "
                        f"delivery: {event['delivery']}"
                    )

            if save and writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 1:
                    fps = 30.0
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_count += 1

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return output_path if save and frame_count > 0 else None


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def _source_name(source: str | int) -> str:
    if isinstance(source, int):
        return f"webcam_{source}"

    source_path = Path(source)
    if source_path.exists():
        return source_path.stem

    return _slugify(source.split("://", 1)[-1])
