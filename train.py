"""
Committee machine training script — the only file the autoresearch agent edits.

Offline learning: train on a fixed cached dataset with epoch-wise mini-batch GD
(no per-step resampling of inputs). Optimizer: plain gradient descent (SGD, momentum=0).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from prepare import (
    BATCH_SIZE,
    DIMENSION,
    EVAL_EVERY_STEPS,
    N_HIDDEN,
    TRAINING_SECONDS,
    compute_mse,
    evaluate,
    load_data,
    report_run,
)

LEARNING_RATE = 0.15


class CommitteeStudent(nn.Module):
    def __init__(self, d: int, n_hidden: int) -> None:
        super().__init__()
        self.readout_scale = 1.0 / math.sqrt(n_hidden)
        self.W = nn.Parameter(torch.empty(n_hidden, d))
        nn.init.normal_(self.W, mean=0.0, std=1.0 / math.sqrt(d))
        with torch.no_grad():
            self.W.data = self.W.data / torch.norm(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.erf(x @ self.W.T)
        return self.readout_scale * h.sum(dim=-1)


def main() -> None:
    total_t0 = time.time()

    w_star, train_loader, val_loader, device = load_data()
    w_star = w_star.to(device)

    student = CommitteeStudent(DIMENSION, N_HIDDEN).to(device)
    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0)
    criterion = nn.MSELoss()

    val_history: list[tuple[int, float]] = []
    num_steps = 0
    training_t0 = time.time()

    print(
        f"Offline GD for {TRAINING_SECONDS}s on {device} "
        f"(d={DIMENSION}, n_hidden={N_HIDDEN}, batch={BATCH_SIZE}, lr={LEARNING_RATE})"
    )

    # Fixed dataset: each epoch is a full pass over cached train tensors (shuffle=True).
    while time.time() - training_t0 < TRAINING_SECONDS:
        for x_batch, y_batch in train_loader:
            if time.time() - training_t0 >= TRAINING_SECONDS:
                break

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = student(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_steps += 1

            if num_steps % EVAL_EVERY_STEPS == 0:
                val_mse = compute_mse(student, val_loader, device)
                val_history.append((num_steps, val_mse))
                elapsed = time.time() - training_t0
                print(
                    f"step {num_steps:6d} | val_mse={val_mse:.6f} | "
                    f"train_batch_loss={loss.item():.6f} | {elapsed:.0f}s"
                )

    training_seconds = time.time() - training_t0
    metrics = evaluate(student, train_loader, val_loader, w_star, device, val_history)
    total_seconds = time.time() - total_t0

    report_run(metrics, val_history, training_seconds, total_seconds, num_steps)


if __name__ == "__main__":
    main()
