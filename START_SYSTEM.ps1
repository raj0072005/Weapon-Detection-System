# Weapon Detection System - One-Click PowerShell Launcher

$env:TELEGRAM_BOT_TOKEN = "8935972088:AAEhJmbqSzP96HFOReNQSbLB_TKexU3vUQU"
$env:TELEGRAM_CHAT_ID = "1394876861"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       WEAPON DETECTION SYSTEM - ONE-CLICK LAUNCH" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " [1/3] Setting Telegram Credentials & Environment..." -ForegroundColor Green
Write-Host " [2/3] Opening Web Dashboard at http://127.0.0.1:5000..." -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"
Write-Host " [3/3] Launching GPU Detection Server..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

& ".\.venv-new\Scripts\python.exe" dashboard.py
