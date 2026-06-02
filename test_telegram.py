"""
Send a test validation-loss plot to Telegram (no training run).

Usage:
    python test_telegram.py
"""

from __future__ import annotations

import os
import sys

from prepare import VAL_LOSS_PLOT_PATH, save_val_loss_plot, send_telegram_val_plot


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Missing Telegram credentials in .env:")
        if not token:
            print("  TELEGRAM_BOT_TOKEN is not set")
        if not chat_id:
            print("  TELEGRAM_CHAT_ID is not set")
        print("\nCopy .env.example to .env and fill in both values.")
        sys.exit(1)

    # Fake val curve (high plateau → drop) so the plot looks realistic
    val_history = []
    for step in range(500, 10_001, 500):
        if step < 6000:
            val_mse = 9.2 + 0.001 * (step / 500)
        else:
            val_mse = max(0.3, 9.2 - 0.015 * ((step - 6000) / 500))
        val_history.append((step, val_mse))

    plot_path = save_val_loss_plot(val_history, VAL_LOSS_PLOT_PATH)
    print(f"Saved test plot to {plot_path}")

    metrics = {
        "val_mse": val_history[-1][1],
        "train_mse": 0.05,
        "max_overlap": 0.72,
        "grok_epoch": 6500,
    }
    send_telegram_val_plot(
        plot_path,
        metrics,
        num_steps=10_000,
        training_seconds=0,
    )
    print("Telegram test OK — check your chat with ShlichtBot.")


if __name__ == "__main__":
    main()
