# Weapon Detection and Alerting

This project uses the selected trained YOLO model in `runs/detect/runs/detect/train-5-2/weights/best.pt` (with an automatic fallback to `train-4`) to identify `knife`, `pistol`, and `rifle` in images or camera streams.

## Local dashboard

The dashboard supports image testing and an opt-in live webcam mode. It uses the GPU when available and automatically falls back to CPU otherwise.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dashboard.txt
python dashboard.py
```

If `py -0p` reports no installed Python version, install Python 3.11 or newer first and make sure **Add Python to PATH** is selected. The copied virtual environments in this folder point to Python installations that no longer exist, so recreate `.venv` after Python is installed.

Open `http://127.0.0.1:5000` in a browser and select the separate **Start live webcam detection** button to grant webcam access. The dashboard now starts at a 0.70 confidence threshold to reduce false positives; increase it further for testing normal scenes. Do not open the HTML template directly from the filesystem: it must be served by `dashboard.py` for the API and static assets to work.

## Camera alert workflow

The stream command can create an alert only after a pistol/rifle is detected in several consecutive frames. Every alert is marked `requires_human_review`, stores an annotated evidence image and JSON record locally, and includes the registered camera ID, physical location, and camera IP address. This reduces false alarms and provides responders with usable context.

Copy `camera_config.example.json` privately and use its registered values. Do not commit camera credentials or an internal camera inventory.

```powershell
.\venv\Scripts\python.exe scripts\detect_stream.py `
  --source "rtsp://username:password@10.20.30.40:554/stream1" `
  --camera-id "CITY-CAM-001" `
  --camera-location "Main Street and Station Road, Ward 4" `
  --camera-ip "10.20.30.40" `
  --conf 0.90 `
  --alert-min-conf 0.95 `
  --alert-frames 5 `
  --alert-cooldown 60
```

Add `--alert-webhook https://dispatch.example.gov/alerts` only after the receiving service is approved and authenticated. Set `WEAPON_ALERT_TOKEN` in the environment before running if it requires a bearer token. The webhook receives alert metadata and the local evidence path; a production dispatch integration should retrieve/upload the evidence through a protected internal service.

## Telegram alerts

Create a bot with `@BotFather`, start a private chat with it (or add it to an approved private group), then set its token and destination chat ID only in your terminal session:

```powershell
$env:TELEGRAM_BOT_TOKEN = "token-from-BotFather"
$env:TELEGRAM_CHAT_ID = "your-private-chat-or-group-id"
```

Add `--telegram` to the stream command. Telegram will receive the camera ID, location, IP address, detection confidences, and annotated evidence image. Treat the bot token and the chat as sensitive access: do not commit either value, and restrict group membership to authorized operators. Telegram's Bot API supports HTTPS requests and multipart uploads for files, which this integration uses. [Telegram Bot API](https://core.telegram.org/bots/api)

Before monitoring, verify both the bot token and the destination chat:

```powershell
.\.venv\Scripts\python.exe scripts\check_telegram.py
```

Each alert is saved locally before Telegram delivery starts. Failed notifications are retried three times immediately, then retried automatically every minute while the stream or webcam detector remains running. The event's `event.json` records the delivery state and total attempts. A network outage can delay delivery, so no external bot can guarantee instant delivery; keep the local event log and a human review process as the source of record.

The same alert arguments work with `scripts/detect_webcam.py`. Use `--camera-id`, `--camera-location`, `--camera-ip`, and `--telegram` when testing an authorized local webcam.

The dashboard webcam also sends Telegram alerts when the two `TELEGRAM_*` environment variables are set before starting `dashboard.py`. It requires three consecutive pistol/rifle predictions at 80% or higher, writes local evidence first, and then sends the image to Telegram.

The current validation report shows the detector is not reliable enough for automated enforcement (notably pistol precision is about 0.41). Keep a trained operator in the review loop, tune thresholds using local CCTV footage, audit false alerts, and comply with local privacy, retention, and police-integration requirements. For a production model, retrain with many normal camera frames labelled as **no weapon**; thresholding is only a temporary false-positive reduction.
