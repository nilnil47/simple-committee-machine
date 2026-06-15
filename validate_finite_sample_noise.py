"""
Validation A: finite-sample noise in the empirical cross-gradient

Professor's note:
    (1/P) sum_p grad f(x_p) y(x_p)  has std ~ 1/sqrt(P)

We measure std across independent dataset redraws at each P (fixed student + teacher).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# --- experiment knobs ---
DIMENSION = 30
N_HIDDEN = 32
INIT_SEED = 42
P_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
REPEATS_PER_P = 80
TASK = "hermite"

OUTPUT_DIR = Path("simple-committee-machine")
PLOT_PATH = OUTPUT_DIR / "validation_finite_sample_noise.png"


def hermite_teacher(x: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
    z = (x @ w_star).squeeze(-1)
    return z**3 - 3 * z


class CommitteeStudent(nn.Module):
    def __init__(self, d: int, n_hidden: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(n_hidden)
        self.W = nn.Parameter(torch.empty(n_hidden, d))
        nn.init.normal_(self.W, 0.0, 1.0 / math.sqrt(d))
        with torch.no_grad():
            self.W /= torch.norm(self.W, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.erf(x @ self.W.T).sum(dim=-1)


def empirical_cross_gradient(
    student: CommitteeStudent, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Gradient of mean(y * f(x)) = (1/P) sum_p y_p grad f(x_p)."""
    student.zero_grad()
    loss = (y * student(x)).mean()
    loss.backward()
    return student.W.grad.detach().clone()


def init_student_and_teacher(d: int, n_hidden: int, seed: int) -> tuple[CommitteeStudent, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    w_star = torch.randn(d, 1, generator=gen)
    w_star /= torch.norm(w_star)

    torch.manual_seed(seed)
    student = CommitteeStudent(d, n_hidden)
    return student, w_star


def std_across_redraws(
    P: int,
    student_template: CommitteeStudent,
    student_state: dict[str, torch.Tensor],
    w_star: torch.Tensor,
    repeats: int,
) -> dict[str, float]:
    d = w_star.shape[0]
    w_star_flat = w_star.squeeze()
    grad_norms: list[float] = []
    teacher_proj: list[float] = []

    for repeat in range(repeats):
        gen = torch.Generator().manual_seed(P * 1_000_003 + repeat)
        x = torch.randn(P, d, generator=gen)
        y = hermite_teacher(x, w_star)

        student_template.load_state_dict(student_state)
        grad = empirical_cross_gradient(student_template, x, y)
        grad_norms.append(grad.norm().item())
        teacher_proj.append((grad @ w_star_flat).sum().item())

    return {
        "std_grad_norm": float(np.std(grad_norms)),
        "std_teacher_projection": float(np.std(teacher_proj)),
        "mean_teacher_projection": float(np.mean(teacher_proj)),
    }


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
  """Fit log(y) = slope * log(x) + intercept; return slope."""
  log_x = np.log(x)
  log_y = np.log(y)
  slope, _ = np.polyfit(log_x, log_y, 1)
  return float(slope)


def main() -> None:
    student, w_star = init_student_and_teacher(DIMENSION, N_HIDDEN, INIT_SEED)
    student_state = {k: v.clone() for k, v in student.state_dict().items()}

    results: list[dict[str, float | int]] = []
    print(f"Validation A: d={DIMENSION}, n_hidden={N_HIDDEN}, repeats={REPEATS_PER_P}")
    print("Measuring std of empirical cross-gradient across dataset redraws...\n")

    for P in P_VALUES:
        stats = std_across_redraws(P, student, student_state, w_star, REPEATS_PER_P)
        row = {"P": P, **stats}
        results.append(row)
        print(
            f"P={P:5d} | std(||grad||)={stats['std_grad_norm']:.6f} "
            f"| std(proj)={stats['std_teacher_projection']:.6f} "
            f"| mean(proj)={stats['mean_teacher_projection']:.6f}"
        )

    P_arr = np.array([r["P"] for r in results], dtype=float)
    std_norm = np.array([r["std_grad_norm"] for r in results])
    std_proj = np.array([r["std_teacher_projection"] for r in results])

    slope_norm = fit_loglog_slope(P_arr, std_norm)
    slope_proj = fit_loglog_slope(P_arr, std_proj)

    print("\n--- log-log slopes (expect ~ -0.5 for 1/sqrt(P) std scaling) ---")
    print(f"std(||grad||_F):     slope = {slope_norm:.3f}")
    print(f"std(teacher proj):  slope = {slope_proj:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y, label, slope in zip(
        axes,
        [std_norm, std_proj],
        ["std(||grad||_F)", "std(teacher projection)"],
        [slope_norm, slope_proj],
    ):
        ax.loglog(P_arr, y, "o-", linewidth=1.5, label="measured")
        ref = y[0] * (P_arr / P_arr[0]) ** -0.5
        ax.loglog(P_arr, ref, "--", color="gray", alpha=0.8, label="P^{-0.5} reference")
        ax.set_xlabel("P (training samples)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs P (slope={slope:.3f})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Validation A: finite-sample cross-gradient noise (d={DIMENSION}, He$_3$ teacher)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
