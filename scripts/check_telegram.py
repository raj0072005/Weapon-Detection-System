"""Verify that the configured Telegram bot can access its destination chat."""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen


def call(method: str, token: str, chat_id: str | None = None) -> dict:
    # Tokens and IDs are commonly copied from terminals; trim accidental
    # surrounding spaces/newlines so they cannot make an invalid request URL.
    url = f"https://api.telegram.org/bot{token.strip()}/{method}"
    if chat_id is not None:
        url += "?" + urlencode({"chat_id": chat_id.strip()})
    with urlopen(url, timeout=15) as response:  # nosec B310 - fixed Telegram API host
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before running this check.")
    try:
        bot = call("getMe", token)
        chat = call("getChat", token, chat_id)
    except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Telegram connectivity check failed: {exc}") from exc
    if not bot.get("ok"):
        raise SystemExit(f"Telegram bot token check failed: {bot.get('description', 'unknown error')}")
    if not chat.get("ok"):
        raise SystemExit(f"Telegram chat check failed: {chat.get('description', 'unknown error')}")
    print(f"Telegram notifier is ready: @{bot['result']['username']} can access chat {chat_id}.")


if __name__ == "__main__":
    main()
