@echo off
title Weapon Detection System - Dashboard Server
color 0A

echo ========================================================
echo       WEAPON DETECTION SYSTEM - ONE-CLICK LAUNCH
echo ========================================================
echo.
echo  [1/3] Setting Telegram Credentials & Environment...
set TELEGRAM_BOT_TOKEN=8935972088:AAEhJmbqSzP96HFOReNQSbLB_TKexU3vUQU
set TELEGRAM_CHAT_ID=1394876861

echo  [2/3] Opening Web Dashboard at http://127.0.0.1:5000 ...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000

echo  [3/3] Starting High-Speed GPU Detection Server...
echo.
echo System ready! Press Ctrl+C in this window to stop the server.
echo ========================================================
echo.

"%~dp0\.venv-new\Scripts\python.exe" "%~dp0\dashboard.py"

pause
