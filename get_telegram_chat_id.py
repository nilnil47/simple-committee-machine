"""
Fetch your Telegram chat id from recent messages to the bot.

Usage:
  1. Open https://t.me/ShlichtBot and press START
  2. Send any message (e.g. "hi")
  3. Run: python get_telegram_chat_id.py
"""

from __future__ import annotations

import os
import sys

import requests

import prepare  # loads .env via prepare.py


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env first.")
        sys.exit(1)

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data.get("ok"):
        print("Telegram API error:", data)
        sys.exit(1)

    results = data.get("result", [])
    if not results:
        print("No messages found for your bot yet.\n")
        print("Do this:")
        print("  1. Open https://t.me/ShlichtBot")
        print("  2. Press START (or Send Message)")
        print("  3. Type anything, e.g. hi")
        print("  4. Run this script again immediately")
        print("\nIf you already did that, send a NEW message and retry.")
        sys.exit(1)

    seen: set[int] = set()
    print("Found chat id(s):\n")
    for update in results:
        message = update.get("message") or update.get("edited_message")
        if not message:
            continue
        chat = message["chat"]
        chat_id = chat["id"]
        if chat_id in seen:
            continue
        seen.add(chat_id)
        name = chat.get("first_name") or chat.get("title") or "unknown"
        username = chat.get("username", "")
        user_tag = f" @{username}" if username else ""
        print(f"  TELEGRAM_CHAT_ID={chat_id}   ({name}{user_tag})")

    print("\nCopy the line above into your .env file.")


if __name__ == "__main__":
    main()
