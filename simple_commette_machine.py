# -*- coding: utf-8 -*-
"""simple_commette_machine.ipynb

### Minimal Teacher-Student Model (Knowledge Distillation) Implementation

The teacher is a fixed direction w_* with target He_3(z) = z^3 - 3z on z = w_* · x.

Offline setup: generate and cache 1000 train + 1000 test points once, then each run
uses the first k train / first k test samples (fixed w_* and data order across runs).
Training is full-batch plain GD (SGD, momentum=0) — one gradient step per epoch.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset

# --- Hyperparameters ---

DIMENSION = 10
N_HIDDEN = 32
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 10_000
SEED = 42

# Cached dataset size (generated once)
N_TRAIN_TOTAL = 1000
N_TEST_TOTAL = 1000

# How many of the cached points to use this run (first k, same order every epoch)
N_TRAIN_USED = 100
N_TEST_USED = 200

CACHE_DIR = Path.home() / ".cache" / "simple-committee-machine"


def hermite_teacher(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    z = torch.matmul(x, w)
    return (z**3 - 3 * z).squeeze()


def prepare_data() -> None:
    """Generate and cache train/test tensors and teacher weights (run once)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "w_star": CACHE_DIR / "w_star.pt",
        "x_train": CACHE_DIR / "x_train.pt",
        "y_train": CACHE_DIR / "y_train.pt",
        "x_test": CACHE_DIR / "x_test.pt",
        "y_test": CACHE_DIR / "y_test.pt",
    }
    if all(p.exists() for p in paths.values()):
        print(f"Data already cached at {CACHE_DIR}")
        return

    print(
        f"Preparing data (d={DIMENSION}, train={N_TRAIN_TOTAL}, test={N_TEST_TOTAL})..."
    )
    generator = torch.Generator().manual_seed(SEED)
    w_star = torch.randn(DIMENSION, 1, generator=generator)
    w_star = w_star / torch.norm(w_star)

    x_train = torch.randn(N_TRAIN_TOTAL, DIMENSION, generator=generator)
    y_train = hermite_teacher(x_train, w_star)
    x_test = torch.randn(N_TEST_TOTAL, DIMENSION, generator=generator)
    y_test = hermite_teacher(x_test, w_star)

    torch.save(w_star, paths["w_star"])
    torch.save(x_train, paths["x_train"])
    torch.save(y_train, paths["y_train"])
    torch.save(x_test, paths["x_test"])
    torch.save(y_test, paths["y_test"])
    print(f"Cached data to {CACHE_DIR}")


def load_data(
    n_train_used: int,
    n_test_used: int,
) -> tuple[torch.Tensor, DataLoader, torch.Tensor, torch.Tensor]:
    """Load cached data, slice first k train/test points, return full-batch loader."""
    if not (CACHE_DIR / "w_star.pt").exists():
        raise FileNotFoundError(
            f"Cached data not found at {CACHE_DIR}. Run prepare_data() first."
        )
    if not (1 <= n_train_used <= N_TRAIN_TOTAL):
        raise ValueError(f"n_train_used must be in [1, {N_TRAIN_TOTAL}], got {n_train_used}")
    if not (1 <= n_test_used <= N_TEST_TOTAL):
        raise ValueError(f"n_test_used must be in [1, {N_TEST_TOTAL}], got {n_test_used}")

    w_star = torch.load(CACHE_DIR / "w_star.pt", weights_only=True)
    x_train = torch.load(CACHE_DIR / "x_train.pt", weights_only=True)[:n_train_used]
    y_train = torch.load(CACHE_DIR / "y_train.pt", weights_only=True)[:n_train_used]
    x_test = torch.load(CACHE_DIR / "x_test.pt", weights_only=True)[:n_test_used]
    y_test = torch.load(CACHE_DIR / "y_test.pt", weights_only=True)[:n_test_used]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=n_train_used,
        shuffle=False,
    )
    return w_star, train_loader, x_test, y_test


class CommitteeStudent(nn.Module):
    def __init__(self, d: int, n_hidden: int) -> None:
        super().__init__()
        self.readout_scale = 1.0 / math.sqrt(n_hidden)
        self.W = nn.Parameter(torch.empty(n_hidden, d))
        nn.init.normal_(self.W, mean=0.0, std=1.0 / math.sqrt(d))
        with torch.no_grad():
            current_norm = torch.norm(self.W)
            self.W.data = self.W.data / current_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.erf(x @ self.W.T)
        return self.readout_scale * h.sum(dim=-1)


def main() -> None:
    prepare_data()
    w_star, train_loader, x_test, y_test = load_data(N_TRAIN_USED, N_TEST_USED)

    wandb.init(
        project="hermite-distillation",
        config={
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD",
            "architecture": "CommitteeStudent",
            "dataset": "offline cached, first-k subset",
            "epochs": TRAIN_EPOCHS,
            "n_train_total": N_TRAIN_TOTAL,
            "n_test_total": N_TEST_TOTAL,
            "n_train_used": N_TRAIN_USED,
            "n_test_used": N_TEST_USED,
            "dimension": DIMENSION,
            "n_hidden": N_HIDDEN,
            "batch_size": N_TRAIN_USED,
            "activation": "erf",
            "seed": SEED,
        },
    )

    checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pth")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(
        f"Offline full-batch GD: train={N_TRAIN_USED}/{N_TRAIN_TOTAL}, "
        f"test={N_TEST_USED}/{N_TEST_TOTAL}, lr={LEARNING_RATE}"
    )

    student = CommitteeStudent(DIMENSION, N_HIDDEN)
    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0)
    criterion = nn.MSELoss()

    for epoch in range(TRAIN_EPOCHS):
        student.train()
        for x_batch, y_batch in train_loader:
            with torch.no_grad():
                z = torch.matmul(x_batch, w_star).squeeze()
                avg_z = z.mean().item()

            y_pred = student(x_batch)
            train_loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        student.eval()
        with torch.no_grad():
            y_test_pred = student(x_test)
            test_loss = criterion(y_test_pred, y_test).item()
            current_norm = torch.norm(student.W).item()

        wandb.log(
            {
                "epoch": epoch,
                "loss": train_loss.item(),
                "test_loss": test_loss,
                "weight_norm": current_norm,
                "avg_projection_z": avg_z,
            }
        )

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch + 1}/{TRAIN_EPOCHS}] | "
                f"Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss:.4f}"
            )

    wandb.config.update(
        {
            "w_star": w_star.flatten().tolist(),
            "activation": "erf",
        }
    )
    torch.save(student.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print("Successfully updated configuration and logged metrics to W&B.")
    wandb.finish()


if __name__ == "__main__":
    main()
