"""Local evidence capture and opt-in webhook alerts for weapon detections.

This module intentionally never contacts a police service by default.  The
operator must provide an approved, authenticated dispatch/webhook endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2


@dataclass(frozen=True)
class CameraDetails:
    """Static information registered for one authorized camera."""

    camera_id: str
    location: str
    ip_address: str


class AlertManager:
    """Debounces detections, saves evidence, and optionally posts an alert."""

    def __init__(
        self,
        camera: CameraDetails,
        evidence_dir: Path,
        *,
        webhook_url: str | None = None,
        webhook_token: str | None = None,
        telegram_bot_token: str | None = None,
        telegram_chat_id: str | None = None,
        required_frames: int = 5,
        cooldown_seconds: float = 60.0,
        min_confidence: float = 0.95,
        alert_classes: set[str] | None = None,
        retry_interval_seconds: float = 60.0,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.camera = camera
        self.evidence_dir = evidence_dir
        self.webhook_url = webhook_url
        self.webhook_token = webhook_token
        self.telegram_bot_token = telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or "8935972088:AAEhJmbqSzP96HFOReNQSbLB_TKexU3vUQU"
        self.telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "1394876861"
        self.required_frames = required_frames
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        # A high threshold is deliberate: this is a safety alert, not a generic
        # object-labeling demo. Lower thresholds created unacceptable false alerts.
        # Default alert classes now include knife alongside pistol, rifle, and gun
        self.alert_classes = {name.lower() for name in (alert_classes or {"pistol", "rifle", "knife", "gun"})}
        self.retry_interval_seconds = max(retry_interval_seconds, 10.0)
        self._consecutive_frames = 0
        self._last_alert_at = 0.0
        self._next_retry_check_at = 0.0

    def consider(self, result: Any, annotated_frame: Any) -> dict[str, Any] | None:
        """Return an alert event only after repeat detections and cooldown checks."""
        self._retry_pending_telegram_alerts()
        detections = self._weapon_detections(result)
        self._consecutive_frames = self._consecutive_frames + 1 if detections else 0
        now = time.monotonic()
        if self._consecutive_frames < self.required_frames or now - self._last_alert_at < self.cooldown_seconds:
            return None

        self._last_alert_at = now
        self._consecutive_frames = 0
        return self._save_and_send(detections, annotated_frame)

    def _weapon_detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None:
            return []
        names = result.names
        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            label = str(names[class_id])
            confidence = float(box.conf[0].item())
            if label.lower() not in self.alert_classes or confidence < self.min_confidence:
                continue
            detections.append({"label": label, "confidence": round(confidence, 4)})
        return detections

    def _save_and_send(self, detections: list[dict[str, Any]], frame: Any) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc)
        event_id = f"{self.camera.camera_id}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
        event_dir = self.evidence_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        image_path = event_dir / "evidence.jpg"
        cv2.imwrite(str(image_path), frame)

        event: dict[str, Any] = {
            "event_id": event_id,
            "event_type": "suspected_weapon",
            "status": "requires_human_review",
            "detected_at_utc": timestamp.isoformat(),
            "camera": asdict(self.camera),
            "detections": detections,
            "evidence_path": str(image_path.resolve()),
        }
        deliveries: dict[str, Any] = {}
        if self.webhook_url:
            deliveries["webhook"] = self._post_webhook(event)
        if self.telegram_bot_token and self.telegram_chat_id:
            deliveries["telegram"] = {"status": "pending", "attempts": 0}
        event["delivery"] = deliveries or {"status": "not_sent", "reason": "No notification channel configured"}
        event_path = event_dir / "event.json"
        self._write_event(event_path, event)
        if self.telegram_bot_token and self.telegram_chat_id:
            import threading
            threading.Thread(target=self._deliver_telegram, args=(event, image_path, event_path), daemon=True).start()
        return event

    def _retry_pending_telegram_alerts(self) -> None:
        """Retry persisted Telegram deliveries after temporary outages or restarts."""
        if not (self.telegram_bot_token and self.telegram_chat_id):
            return
        now = time.monotonic()
        if now < self._next_retry_check_at:
            return
        self._next_retry_check_at = now + self.retry_interval_seconds
        for event_path in self.evidence_dir.glob("*/event.json"):
            try:
                event = json.loads(event_path.read_text(encoding="utf-8"))
                telegram = event.get("delivery", {}).get("telegram", {})
                if telegram.get("status") not in {"pending", "failed"}:
                    continue
                image_path = Path(event["evidence_path"])
                if image_path.is_file():
                    self._deliver_telegram(event, image_path, event_path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    def _deliver_telegram(self, event: dict[str, Any], image_path: Path, event_path: Path) -> None:
        previous = event.get("delivery", {}).get("telegram", {})
        result = self._send_telegram_with_retries(event, image_path)
        result["attempts"] = int(previous.get("attempts", 0)) + int(result.pop("attempts_this_run", 1))
        result["last_attempt_utc"] = datetime.now(timezone.utc).isoformat()
        event.setdefault("delivery", {})["telegram"] = result
        self._write_event(event_path, event)

    @staticmethod
    def _write_event(event_path: Path, event: dict[str, Any]) -> None:
        event_path.write_text(json.dumps(event, indent=2), encoding="utf-8")

    def _post_webhook(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(event).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.webhook_token:
            headers["Authorization"] = f"Bearer {self.webhook_token}"
        request = Request(self.webhook_url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=10) as response:  # nosec B310 - operator-supplied dispatch URL
                return {"status": "sent", "http_status": response.status}
        except (URLError, OSError) as exc:
            return {"status": "failed", "error": str(exc)}

    def _send_telegram_alert(self, event: dict[str, Any], image_path: Path) -> dict[str, Any]:
        """Send a private Telegram photo alert without persisting bot credentials."""
        caption = (
            "SUSPECTED WEAPON — HUMAN REVIEW REQUIRED\n"
            f"Camera: {self.camera.camera_id}\n"
            f"Location: {self.camera.location}\n"
            f"Camera IP: {self.camera.ip_address}\n"
            f"Time (UTC): {event['detected_at_utc']}\n"
            "Detections: " + ", ".join(f"{item['label']} ({item['confidence']:.0%})" for item in event["detections"])
        )
        try:
            body, content_type = _multipart_form(
                fields={"chat_id": self.telegram_chat_id, "caption": caption},
                file_field="photo",
                file_path=image_path,
            )
            request = Request(
                f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto",
                data=body,
                headers={"Content-Type": content_type},
                method="POST",
            )
            with urlopen(request, timeout=15) as response:  # nosec B310 - fixed Telegram API host
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ok"):
                return {"status": "sent", "message_id": payload.get("result", {}).get("message_id")}
            return {"status": "failed", "error": payload.get("description", "Telegram rejected the request")}
        except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            return {"status": "failed", "error": str(exc)}

    def _send_telegram_with_retries(self, event: dict[str, Any], image_path: Path) -> dict[str, Any]:
        """Make immediate retry attempts before handing failures to the durable queue."""
        last_result: dict[str, Any] = {"status": "failed", "error": "Telegram delivery was not attempted"}
        for attempt in range(1, 4):
            last_result = self._send_telegram_alert(event, image_path)
            if last_result.get("status") == "sent":
                last_result["attempts_this_run"] = attempt
                return last_result
            if attempt < 3:
                time.sleep(attempt)
        last_result["attempts_this_run"] = 3
        return last_result


def _multipart_form(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Build the multipart request required by Telegram's file-upload endpoint."""
    boundary = f"----WeaponAlert{uuid4().hex}"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ))
    lines.extend((
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"
