"""
Committee machine training script — the only file the autoresearch agent edits.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from prepare import (
    DIMENSION,
    EVAL_EVERY_STEPS,
    N_HIDDEN,
    TRAINING_SECONDS,
    compute_mse,
    evaluate,
    load_data,
    print_summary,
)


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
    total_t0 = time.time()

    w_star, train_loader, val_loader, device = load_data()
    w_star = w_star.to(device)

    student = CommitteeStudent(DIMENSION, N_HIDDEN).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    val_history: list[tuple[int, float]] = []
    num_steps = 0
    training_t0 = time.time()

    print(f"Training for {TRAINING_SECONDS}s on {device} (d={DIMENSION}, n_hidden={N_HIDDEN})")

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

    print_summary(metrics, training_seconds, total_seconds, num_steps)


if __name__ == "__main__":
    main()
