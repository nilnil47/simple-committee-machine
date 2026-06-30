"""Classic erf committee on teacher y = erf(x_1) - 2 erf(x_1/2). d=20, N=100, P=100 samples."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

DIMENSION = 1
N = 16

W_STAR = torch.zeros(DIMENSION)
W_STAR[0] = 1.0

LR = 1e-4
EPOCHS = 10_000
SEED = 42
INIT_SEED = 42
INIT_MEAN = 0.0
INIT_VAR = 1.0 / DIMENSION  # std = sqrt(INIT_VAR); use a small value for a narrow init

N_TRAIN_TOTAL = 10_000
N_TEST_TOTAL = 5_000
P = 100
N_TEST_USED = 100

CACHE = Path(".cache").parent / "simple-committee-machine-erf-combo"
INIT_WEIGHTS_PATH = CACHE / "student_init.pt"
TRAINED_WEIGHTS_PATH = CACHE / "student_trained.pt"
LOSS_LOGLOG_PLOT_PATH = CACHE / "loss_loglog.png"
PRED_VS_THEORY_PLOT_PATH = CACHE / "pred_vs_theory.png"
THEORY_CURVES_PLOT_PATH = CACHE / "theory_curves.png"
WEIGHT_DISTRIBUTION_PLOT_PATH = CACHE / "weights_distribution.png"
CHECKPOINT_DIR = CACHE / "checkpoints"

WEIGHT_DIST_BINS = 30

THEORY_GRID_POINTS = 500

# 1-indexed epochs at which to save student weights; None = final checkpoint only
SAVE_EPOCHS: list[int] | None = None
# SAVE_EPOCHS: list[int] | None = [500]

# None = start from saved init; "trained" = load TRAINED_WEIGHTS_PATH; or any .pt path
LOAD_FROM = CACHE / "student_init.pt"
# LOAD_FROM = TRAINED_WEIGHTS_PATH
# LOAD_FROM = None


def sample_init_W(n: int, d: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(INIT_SEED)
    return INIT_MEAN + math.sqrt(INIT_VAR) * torch.randn(n, d, generator=g)


class CommitteeStudent(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n)
        self.W = nn.Parameter(sample_init_W(n, d))

    def forward(self, x):
        return self.scale * torch.erf(x @ self.W.T).sum(dim=-1)


def teacher_erf_combo(x: torch.Tensor) -> torch.Tensor:
    """Theoretical target y(x) = erf(x_1) - 2 erf(x_1/2).

    This is the supervisor's teacher. It has no linear term at the origin:
    y'(0) = 0, and the Taylor expansion starts at O(x^3).
    """
    x1 = x[:, 0]
    return torch.erf(x1) - 2.0 * torch.erf(0.5 * x1)


def mse_vs_teacher(pred: torch.Tensor, x: torch.Tensor) -> float:
    y = teacher_erf_combo(x)
    return ((pred - y) ** 2).mean().item()


def make_theory_grid(
    x_min: float = -3.0, x_max: float = 3.0, n_points: int = THEORY_GRID_POINTS
) -> torch.Tensor:
    xs = torch.linspace(x_min, x_max, n_points)
    grid = torch.zeros(n_points, DIMENSION)
    grid[:, 0] = xs
    return grid


def save_loss_loglog_plot(
    epochs: list[int],
    train_losses: list[float],
    test_losses: list[float],
    path: Path,
    nngp_test_mse: float | None = None,
) -> Path:
    eps = 1e-12
    x = [e + 1 for e in epochs]
    train_y = [max(v, eps) for v in train_losses]
    test_y = [max(v, eps) for v in test_losses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, train_y, label="train loss", linewidth=1.5)
    ax.loglog(x, test_y, label="test loss", linewidth=1.5)
    if nngp_test_mse is not None:
        ax.axhline(
            max(nngp_test_mse, eps),
            color="C2",
            linestyle="--",
            linewidth=1.5,
            label=f"NNGP test MSE ({nngp_test_mse:.4f})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_theory_stages_plot(
    x: torch.Tensor,
    y_teacher: torch.Tensor,
    y_student_init: torch.Tensor,
    y_student_trained: torch.Tensor,
    path: Path,
) -> Path:
    xs = x[:, 0].numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(xs, y_teacher.numpy(), label="teacher y(x)", linewidth=2.0)
    axes[0].plot(
        xs, y_student_init.numpy(), label="initial student", linewidth=1.5, alpha=0.85
    )
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, y_teacher.numpy(), label="teacher y(x)", linewidth=2.0)
    axes[1].plot(
        xs,
        y_student_trained.numpy(),
        label="trained student",
        linewidth=1.5,
        alpha=0.85,
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def student_hidden_weights(student: CommitteeStudent) -> torch.Tensor:
    """Hidden weight vectors w_p for each unit, shape (N, d)."""
    return student.W.detach()


def project_onto_w_star(w: torch.Tensor) -> torch.Tensor:
    """Project each hidden weight row onto w*; returns shape (N,)."""
    return w @ W_STAR


def save_weight_distribution_plot(
    w_init: torch.Tensor,
    w_trained: torch.Tensor,
    path: Path,
) -> Path:
    init_proj = project_onto_w_star(w_init).numpy()
    trained_proj = project_onto_w_star(w_trained).numpy()

    lo = min(init_proj.min(), trained_proj.min())
    hi = max(init_proj.max(), trained_proj.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bins = WEIGHT_DIST_BINS
    hist_range = (lo, hi)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].hist(init_proj, bins=bins, range=hist_range, color="C0", alpha=0.85)
    axes[0].set_ylabel("count")
    axes[0].set_title("initial")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(trained_proj, bins=bins, range=hist_range, color="C1", alpha=0.85)
    axes[1].set_xlabel(r"$w \cdot w^*$")
    axes[1].set_ylabel("count")
    axes[1].set_title("trained")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_pred_vs_theory_plot(
    y_teacher: torch.Tensor, y_student: torch.Tensor, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_teacher.numpy(), y_student.numpy(), s=8, alpha=0.5)
    lo = min(y_teacher.min().item(), y_student.min().item())
    hi = max(y_teacher.max().item(), y_student.max().item())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
    ax.set_xlabel("teacher y")
    ax.set_ylabel("student f")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def analyze_convergence_to_theory(
    student: CommitteeStudent,
    x_test: torch.Tensor,
    y_student_init: torch.Tensor,
) -> dict[str, float]:
    """Compare trained student to the erf-combo teacher."""
    student.eval()
    with torch.no_grad():
        y_test = teacher_erf_combo(x_test)
        y_student = student(x_test)
        trained_test_mse = mse_vs_teacher(y_student, x_test)

        x_grid = make_theory_grid()
        y_teacher_grid = teacher_erf_combo(x_grid)
        y_student_grid = student(x_grid)
        trained_grid_mse = mse_vs_teacher(y_student_grid, x_grid)

    curves_path = save_theory_stages_plot(
        x_grid,
        y_teacher_grid,
        y_student_init,
        y_student_grid,
        THEORY_CURVES_PLOT_PATH,
    )
    scatter_path = save_pred_vs_theory_plot(y_test, y_student, PRED_VS_THEORY_PLOT_PATH)

    w_trained = project_onto_w_star(student_hidden_weights(student))
    print("Teacher comparison:")
    print(f"  trained_test_mse={trained_test_mse:.6f}")
    print(f"  trained_grid_mse={trained_grid_mse:.6f}")
    print(f"  saved theory curves to {curves_path}")
    print(f"  saved pred-vs-theory scatter to {scatter_path}")
    print(
        f"  trained w·w*: mean={w_trained.mean().item():.4f} "
        f"std={w_trained.std().item():.4f} "
        f"min={w_trained.min().item():.4f} max={w_trained.max().item():.4f}"
    )

    return {
        "trained_test_mse": trained_test_mse,
        "trained_grid_mse": trained_grid_mse,
        "trained_w_mean": w_trained.mean().item(),
        "trained_w_std": w_trained.std().item(),
    }


def _cached_data_valid(path: Path, min_rows: int) -> bool:
    if not path.exists():
        return False
    x = torch.load(path, weights_only=True)
    return x.shape[0] >= min_rows and x.shape[1] == DIMENSION


def data_cache_valid() -> bool:
    return _cached_data_valid(CACHE / "x_train.pt", N_TRAIN_TOTAL) and _cached_data_valid(
        CACHE / "x_test.pt", N_TEST_TOTAL
    )


def ensure_init_weights() -> None:
    """Sample init once from N(INIT_MEAN, INIT_VAR), cache for all runs."""
    if INIT_WEIGHTS_PATH.exists():
        state = torch.load(INIT_WEIGHTS_PATH, weights_only=True)
        if (
            state["W"].shape == (N, DIMENSION)
            and state.get("init_seed") == INIT_SEED
            and state.get("init_mean") == INIT_MEAN
            and state.get("init_var") == INIT_VAR
        ):
            return
        print("Regenerating init weights (cached init mismatch)")
    student = CommitteeStudent(DIMENSION, N)
    torch.save(
        {
            "W": student.W.detach().clone(),
            "init_seed": INIT_SEED,
            "init_mean": INIT_MEAN,
            "init_var": INIT_VAR,
        },
        INIT_WEIGHTS_PATH,
    )
    print(f"Saved init weights to {INIT_WEIGHTS_PATH}")


def resolve_checkpoint(load_from: str | Path | None) -> Path:
    if load_from is None:
        return INIT_WEIGHTS_PATH
    if load_from == "trained":
        return TRAINED_WEIGHTS_PATH
    return Path(load_from)


def load_student(load_from: str | Path | None) -> tuple[CommitteeStudent, Path]:
    ensure_init_weights()
    student = CommitteeStudent(DIMENSION, N)
    path = resolve_checkpoint(load_from)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, weights_only=True)
    if "init_seed" in state:
        student.load_state_dict({"W": state["W"]})
    else:
        student.load_state_dict(state)
    print(f"Loaded weights from {path}")
    return student, path


def checkpoint_path(epoch: int) -> Path:
    """Path for a 1-indexed training epoch checkpoint."""
    return CHECKPOINT_DIR / f"student_epoch_{epoch:06d}.pt"


def save_student_checkpoint(
    student: CommitteeStudent, epoch: int, path: Path | None = None
) -> Path:
    """Save student weights at the given 1-indexed epoch."""
    path = path or checkpoint_path(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), path)
    print(f"Saved checkpoint at epoch {epoch} to {path}")
    return path


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if data_cache_valid():
        x_train = torch.load(CACHE / "x_train.pt", weights_only=True)
        x_test = torch.load(CACHE / "x_test.pt", weights_only=True)
    else:
        print("Regenerating data cache (missing or dimension mismatch)")
        g = torch.Generator().manual_seed(SEED)
        x_train = torch.randn(N_TRAIN_TOTAL, DIMENSION, generator=g)
        x_test = torch.randn(N_TEST_TOTAL, DIMENSION, generator=g)
        torch.save(x_train, CACHE / "x_train.pt")
        torch.save(x_test, CACHE / "x_test.pt")

    x_train = x_train[:P]
    x_test = x_test[:N_TEST_USED]
    y_train = teacher_erf_combo(x_train)
    y_test = teacher_erf_combo(x_test)

    student, loaded_from = load_student(LOAD_FROM)

    student.eval()
    with torch.no_grad():
        x_grid = make_theory_grid()
        y_teacher_grid = teacher_erf_combo(x_grid)
        y_student_init = student(x_grid)
        init_grid_mse = mse_vs_teacher(y_student_init, x_grid)
    print(f"init_grid_mse={init_grid_mse:.6f}")

    w_init = student_hidden_weights(student).clone()

    wandb.init(
        project="committee-student",
        name="erf_combo_d20",
        config={
            "task": "erf_combo",
            "dimension": DIMENSION,
            "N": N,
            "P": P,
            "student": "(1/sqrt(N)) sum_p erf(w_p·x)",
            "lr": LR,
            "optimizer": "adam",
            "epochs": EPOCHS,
            "n_test_used": N_TEST_USED,
            "init_seed": INIT_SEED,
            "init_mean": INIT_MEAN,
            "init_var": INIT_VAR,
            "loaded_from": str(loaded_from),
            "save_epochs": SAVE_EPOCHS,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("loss", step_metric="epoch")
    wandb.define_metric("grad_norm", step_metric="epoch")
    save_epochs = set(SAVE_EPOCHS or ())
    optimizer = optim.Adam(student.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    epochs_hist: list[int] = []
    train_loss_hist: list[float] = []
    test_loss_hist: list[float] = []

    for epoch in range(EPOCHS):
        student.train()
        loss = loss_fn(student(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = student.W.grad.pow(2).sum().item()
        optimizer.step()

        with torch.no_grad():
            test_loss = loss_fn(student(x_test), y_test).item()

        epochs_hist.append(epoch)
        train_loss_hist.append(loss.item())
        test_loss_hist.append(test_loss)
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "test_loss": test_loss,
                "grad_norm": grad_norm,
            }
        )

        epoch_num = epoch + 1

        if epoch_num in save_epochs:
            save_student_checkpoint(student, epoch_num)

        if epoch_num % 1000 == 0:
            print(f"epoch {epoch_num}: train={loss.item():.4f} test={test_loss:.4f}")

    plot_path = save_loss_loglog_plot(
        epochs_hist,
        train_loss_hist,
        test_loss_hist,
        LOSS_LOGLOG_PLOT_PATH,
    )
    print(f"Saved log-log loss plot to {plot_path}")

    torch.save(student.state_dict(), TRAINED_WEIGHTS_PATH)
    print(f"Saved trained weights to {TRAINED_WEIGHTS_PATH}")

    theory_metrics = analyze_convergence_to_theory(student, x_test, y_student_init)

    w_trained = student_hidden_weights(student)
    weight_plot_path = save_weight_distribution_plot(
        w_init, w_trained, WEIGHT_DISTRIBUTION_PLOT_PATH
    )
    print(f"Saved weight distribution plot to {weight_plot_path}")
    w_init_proj = project_onto_w_star(w_init)
    w_trained_proj = project_onto_w_star(w_trained)
    print(
        f"  init w·w*: mean={w_init_proj.mean().item():.4f} "
        f"std={w_init_proj.std().item():.4f}"
    )
    print(
        f"  trained w·w*: mean={w_trained_proj.mean().item():.4f} "
        f"std={w_trained_proj.std().item():.4f}"
    )

    wandb.run.summary.update({"init_grid_mse": init_grid_mse, **theory_metrics})
    wandb.log(
        {
            "weights_distribution": wandb.Image(str(weight_plot_path)),
            "pred_vs_theory": wandb.Image(str(PRED_VS_THEORY_PLOT_PATH)),
            "theory_curves": wandb.Image(str(THEORY_CURVES_PLOT_PATH)),
        }
    )
    wandb.finish()
