"""Quick W&B smoke test without a full training run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import prepare  # noqa: F401 — loads .env
from prepare import VAL_LOSS_PLOT_PATH, log_to_wandb, save_val_loss_plot


def main() -> None:
    if os.environ.get("WANDB_MODE") == "offline":
        print("WANDB_MODE=offline — run will stay local (wandb sync later).")
    val_history = [(500, 1.2), (1000, 0.8)]
    plot_path = save_val_loss_plot(val_history, VAL_LOSS_PLOT_PATH)
    metrics = {
        "val_mse": val_history[-1][1],
        "train_mse": 0.5,
        "train_val_gap": 0.3,
        "max_overlap": 0.1,
        "mean_overlap": 0.05,
        "grok_epoch": 1.0,
        "plateau_length": 0.0,
    }
    try:
        log_to_wandb(metrics, val_history, plot_path, 1000, 1.0)
    except Exception as exc:
        print(f"W&B test failed: {exc}")
        print(
            "Set WANDB_API_KEY in .env or run `wandb login`, "
            "then: python test_wandb.py"
        )
        sys.exit(1)
    print("W&B test OK — check your project dashboard.")


if __name__ == "__main__":
    main()
