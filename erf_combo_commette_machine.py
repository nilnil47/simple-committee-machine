"""Classic erf committee on teacher y = erf(x_1) - 2 erf(x_1/2). d=10, N=256, P=100 samples."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

DIMENSION = 10
N = 256
# N = 100

W_STAR = torch.zeros(DIMENSION)
W_STAR[0] = 1.0

LR = 1e-4
EPOCHS = 100_000

# Optimizer: "adam" or "gd" (plain gradient descent, no momentum)
OPTIMIZER = "adam"
# OPTIMIZER = "gd"
SEED = 43
INIT_SEED = 43
# Number of networks in the ensemble (seeds INIT_SEED .. INIT_SEED + ENSEMBLE_SIZE - 1).
# Set to 1 for a single run. Ignored if ENSEMBLE_SEEDS is set explicitly.
ENSEMBLE_SIZE = 1
# Explicit seed list overrides ENSEMBLE_SIZE when not None, e.g. [0, 7, 42]
ENSEMBLE_SEEDS: list[int] | None = None
# None = auto group name from hyperparams
WANDB_GROUP: str | None = None
INIT_MEAN = 0.0
# INIT_VAR = 0.1 / DIMENSION  # std = sqrt(INIT_VAR); use a small value for a narrow init
INIT_VAR = 1.0 / DIMENSION  # std = sqrt(INIT_VAR); use a small value for a narrow init

# Init mode: "gaussian" or "manual"
# INIT_MODE = "gaussian"
INIT_MODE = "manual"

# Manual init: one entry per hidden unit (length N).
# With scale 1/sqrt(N), sqrt(N) units at w=1 and 2*sqrt(N) at w=-1/2
# reproduce y = erf(x_1) - 2 erf(x_1/2); remaining units start at 0.
# d=1: each entry is a scalar w_p, e.g. [1.0, -0.5, -0.5, 0.0, ...]
# d>1: each entry is a length-d list for that unit's weight row
_N_SQRT = int(round(math.sqrt(N)))
if _N_SQRT * _N_SQRT != N:
    raise ValueError(f"Manual erf-combo init requires perfect-square N, got N={N}")
_N_ONES = _N_SQRT
_N_HALVES = 2 * _N_SQRT
_N_ZEROS = N - _N_ONES - _N_HALVES
INIT_W_MANUAL: list[float] | list[list[float]] | None = (
    [[1.0] + [0.0] * (DIMENSION - 1)] * _N_ONES
    + [[-0.5] + [0.0] * (DIMENSION - 1)] * _N_HALVES
    + [[0.0] * DIMENSION] * _N_ZEROS
)
# Gaussian noise on manual init: std = sqrt(INIT_MANUAL_NOISE_VAR); 0 for exact manual values
INIT_MANUAL_NOISE_VAR = 0.1 / DIMENSION
# INIT_MANUAL_NOISE_VAR = 0.00
# INIT_MODE = "manual"

N_TRAIN_TOTAL = 100_000
N_TEST_TOTAL = 5_000
P = 100
N_TEST_USED = 100

# Grokking triage thresholds (logged in wandb config + used for summary)
GROK_TRAIN_THRESH = 1e-3
GROK_TEST_THRESH = 1e-2

CACHE = Path(".cache").parent / "simple-committee-machine-erf-combo"
CHECKPOINT_DIR = CACHE / "checkpoints"

WEIGHT_DIST_BINS = 30

THEORY_GRID_POINTS = 500

# 1-indexed epochs at which to save student weights; None = final checkpoint only
SAVE_EPOCHS: list[int] | None = None
# SAVE_EPOCHS: list[int] | None = [500]

# Single-seed only: None = per-seed init cache; "trained" = that seed's trained weights; or any .pt path.
# Ignored when training more than one ensemble seed.
LOAD_FROM: str | Path | None = None
# LOAD_FROM = "trained"


def resolve_ensemble_seeds() -> list[int]:
    if ENSEMBLE_SEEDS is not None:
        return list(ENSEMBLE_SEEDS)
    if ENSEMBLE_SIZE < 1:
        raise ValueError(f"ENSEMBLE_SIZE must be >= 1, got {ENSEMBLE_SIZE}")
    return list(range(INIT_SEED, INIT_SEED + ENSEMBLE_SIZE))


def resolve_wandb_group() -> str:
    if WANDB_GROUP is not None:
        return WANDB_GROUP
    if INIT_MODE == "manual":
        return f"erf_combo_{INIT_MODE}_P{P}_noise{INIT_MANUAL_NOISE_VAR}"
    return f"erf_combo_{INIT_MODE}_P{P}_var{INIT_VAR}"


def init_weights_path(seed: int) -> Path:
    return CACHE / f"student_init_seed{seed}.pt"


def trained_weights_path(seed: int) -> Path:
    return CACHE / f"student_trained_seed{seed}.pt"


def seed_plot_dir(seed: int) -> Path:
    return CACHE / "plots" / f"seed{seed}"


def seed_checkpoint_dir(seed: int) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}"


def sample_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(init_seed)
    return INIT_MEAN + math.sqrt(INIT_VAR) * torch.randn(n, d, generator=g)


def manual_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    if INIT_W_MANUAL is None:
        raise ValueError("INIT_MODE='manual' requires INIT_W_MANUAL")
    if len(INIT_W_MANUAL) != n:
        raise ValueError(
            f"INIT_W_MANUAL must have length N={n}, got {len(INIT_W_MANUAL)}"
        )
    if d == 1:
        W = torch.tensor(INIT_W_MANUAL, dtype=torch.float32).reshape(n, d)
    else:
        rows: list[list[float]] = []
        for i, row in enumerate(INIT_W_MANUAL):
            if isinstance(row, (int, float)):
                raise ValueError(
                    f"INIT_W_MANUAL[{i}] must be a length-{d} list when DIMENSION > 1"
                )
            if len(row) != d:
                raise ValueError(
                    f"INIT_W_MANUAL[{i}] must have length d={d}, got {len(row)}"
                )
            rows.append([float(v) for v in row])
        W = torch.tensor(rows, dtype=torch.float32)
    if INIT_MANUAL_NOISE_VAR > 0:
        g = torch.Generator().manual_seed(init_seed)
        W = W + math.sqrt(INIT_MANUAL_NOISE_VAR) * torch.randn(n, d, generator=g)
    return W


def make_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    if INIT_MODE == "gaussian":
        return sample_init_W(n, d, init_seed)
    if INIT_MODE == "manual":
        return manual_init_W(n, d, init_seed)
    raise ValueError(f"Unknown INIT_MODE: {INIT_MODE!r}")


class CommitteeStudent(nn.Module):
    def __init__(self, d, n, init_seed: int = INIT_SEED):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n)
        self.W = nn.Parameter(make_init_W(n, d, init_seed))

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
    theory_curves_path: Path,
    pred_vs_theory_path: Path,
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
        theory_curves_path,
    )
    scatter_path = save_pred_vs_theory_plot(y_test, y_student, pred_vs_theory_path)

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


def compute_grok_summaries(
    train_losses: list[float],
    test_losses: list[float],
) -> dict[str, float | int | bool]:
    """Summary fields for sorting/filtering grokking across many seeds in W&B."""
    if not train_losses or not test_losses:
        raise ValueError("loss histories must be non-empty")
    min_test_loss = min(test_losses)
    min_test_loss_epoch = test_losses.index(min_test_loss) + 1
    train_fit_epoch = -1
    grok_epoch = -1
    for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        epoch_num = i + 1
        if train_fit_epoch < 0 and train_loss < GROK_TRAIN_THRESH:
            train_fit_epoch = epoch_num
        if (
            grok_epoch < 0
            and train_loss < GROK_TRAIN_THRESH
            and test_loss < GROK_TEST_THRESH
        ):
            grok_epoch = epoch_num
    return {
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "min_test_loss": min_test_loss,
        "min_test_loss_epoch": min_test_loss_epoch,
        "train_fit_epoch": train_fit_epoch,
        "grok_epoch": grok_epoch,
        "grokked": grok_epoch >= 0,
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


def _init_cache_matches(state: dict, init_seed: int) -> bool:
    if state["W"].shape != (N, DIMENSION):
        return False
    if state.get("init_mode") != INIT_MODE:
        return False
    if INIT_MODE == "gaussian":
        return (
            state.get("init_seed") == init_seed
            and state.get("init_mean") == INIT_MEAN
            and state.get("init_var") == INIT_VAR
        )
    if INIT_MODE == "manual":
        return (
            state.get("init_seed") == init_seed
            and state.get("init_manual_noise_var") == INIT_MANUAL_NOISE_VAR
            and torch.allclose(state["W"], make_init_W(N, DIMENSION, init_seed))
        )
    return False


def ensure_init_weights(init_seed: int) -> Path:
    """Cache init weights for this seed (Gaussian or manual)."""
    path = init_weights_path(init_seed)
    if path.exists():
        state = torch.load(path, weights_only=True)
        if _init_cache_matches(state, init_seed):
            return path
        print(f"Regenerating init weights for seed {init_seed} (cached init mismatch)")
    student = CommitteeStudent(DIMENSION, N, init_seed)
    payload: dict = {
        "W": student.W.detach().clone(),
        "init_mode": INIT_MODE,
    }
    if INIT_MODE == "gaussian":
        payload.update(
            {
                "init_seed": init_seed,
                "init_mean": INIT_MEAN,
                "init_var": INIT_VAR,
            }
        )
    if INIT_MODE == "manual":
        payload.update(
            {
                "init_seed": init_seed,
                "init_manual_noise_var": INIT_MANUAL_NOISE_VAR,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved init weights to {path}")
    return path


def make_optimizer(
    params, lr: float, optimizer: str = OPTIMIZER
) -> optim.Optimizer:
    if optimizer == "adam":
        return optim.Adam(params, lr=lr)
    if optimizer == "gd":
        return optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown OPTIMIZER: {optimizer!r} (use 'adam' or 'gd')")


def resolve_checkpoint(
    load_from: str | Path | None, init_seed: int
) -> Path:
    if load_from is None:
        return ensure_init_weights(init_seed)
    if load_from == "trained":
        return trained_weights_path(init_seed)
    return Path(load_from)


def load_student(
    init_seed: int, load_from: str | Path | None
) -> tuple[CommitteeStudent, Path]:
    ensure_init_weights(init_seed)
    student = CommitteeStudent(DIMENSION, N, init_seed)
    path = resolve_checkpoint(load_from, init_seed)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, weights_only=True)
    if "init_mode" in state or "init_seed" in state:
        student.load_state_dict({"W": state["W"]})
    else:
        student.load_state_dict(state)
    print(f"Loaded weights from {path}")
    return student, path


def checkpoint_path(seed: int, epoch: int) -> Path:
    """Path for a 1-indexed training epoch checkpoint."""
    return seed_checkpoint_dir(seed) / f"student_epoch_{epoch:06d}.pt"


def save_student_checkpoint(
    student: CommitteeStudent, seed: int, epoch: int, path: Path | None = None
) -> Path:
    """Save student weights at the given 1-indexed epoch."""
    path = path or checkpoint_path(seed, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), path)
    print(f"Saved checkpoint at epoch {epoch} to {path}")
    return path


def train_one_seed(
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    group: str,
    ensemble_size: int,
    load_from: str | Path | None,
) -> None:
    plot_dir = seed_plot_dir(seed)
    loss_loglog_path = plot_dir / "loss_loglog.png"
    pred_vs_theory_path = plot_dir / "pred_vs_theory.png"
    theory_curves_path = plot_dir / "theory_curves.png"
    weight_dist_path = plot_dir / "weights_distribution.png"
    trained_path = trained_weights_path(seed)

    student, loaded_from = load_student(seed, load_from)

    student.eval()
    with torch.no_grad():
        x_grid = make_theory_grid()
        y_student_init = student(x_grid)
        init_grid_mse = mse_vs_teacher(y_student_init, x_grid)
    print(f"[seed {seed}] init_grid_mse={init_grid_mse:.6f}")

    w_init = student_hidden_weights(student).clone()

    wandb.init(
        project="committee-student",
        group=group,
        job_type="seed",
        name=f"seed_{seed}",
        config={
            "task": "erf_combo",
            "dimension": DIMENSION,
            "N": N,
            "P": P,
            "student": "(1/sqrt(N)) sum_p erf(w_p·x)",
            "lr": LR,
            "optimizer": OPTIMIZER,
            "epochs": EPOCHS,
            "n_test_used": N_TEST_USED,
            "init_seed": seed,
            "ensemble_size": ensemble_size,
            "ensemble_group": group,
            "init_mean": INIT_MEAN,
            "init_var": INIT_VAR,
            "init_mode": INIT_MODE,
            "init_w_manual": INIT_W_MANUAL,
            "init_manual_noise_var": INIT_MANUAL_NOISE_VAR,
            "loaded_from": str(loaded_from),
            "save_epochs": SAVE_EPOCHS,
            "grok_train_thresh": GROK_TRAIN_THRESH,
            "grok_test_thresh": GROK_TEST_THRESH,
        },
        reinit=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("loss", step_metric="epoch")
    wandb.define_metric("grad_norm", step_metric="epoch")
    save_epochs = set(SAVE_EPOCHS or ())
    optimizer = make_optimizer(student.parameters(), LR)
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
            save_student_checkpoint(student, seed, epoch_num)

        if epoch_num % 1000 == 0:
            print(
                f"[seed {seed}] epoch {epoch_num}: "
                f"train={loss.item():.4f} test={test_loss:.4f}"
            )

    plot_path = save_loss_loglog_plot(
        epochs_hist,
        train_loss_hist,
        test_loss_hist,
        loss_loglog_path,
    )
    print(f"[seed {seed}] Saved log-log loss plot to {plot_path}")

    torch.save(student.state_dict(), trained_path)
    print(f"[seed {seed}] Saved trained weights to {trained_path}")

    theory_metrics = analyze_convergence_to_theory(
        student,
        x_test,
        y_student_init,
        theory_curves_path,
        pred_vs_theory_path,
    )

    w_trained = student_hidden_weights(student)
    weight_plot_path = save_weight_distribution_plot(w_init, w_trained, weight_dist_path)
    print(f"[seed {seed}] Saved weight distribution plot to {weight_plot_path}")
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

    grok_metrics = compute_grok_summaries(train_loss_hist, test_loss_hist)
    print(
        f"[seed {seed}] grokked={grok_metrics['grokked']} "
        f"grok_epoch={grok_metrics['grok_epoch']} "
        f"train_fit_epoch={grok_metrics['train_fit_epoch']} "
        f"final_test_loss={grok_metrics['final_test_loss']:.6f}"
    )

    wandb.run.summary.update(
        {
            "init_grid_mse": init_grid_mse,
            **theory_metrics,
            **grok_metrics,
        }
    )
    wandb.log(
        {
            "loss_loglog": wandb.Image(str(plot_path)),
            "weights_distribution": wandb.Image(str(weight_plot_path)),
            "pred_vs_theory": wandb.Image(str(pred_vs_theory_path)),
            "theory_curves": wandb.Image(str(theory_curves_path)),
        }
    )
    wandb.finish()


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

    seeds = resolve_ensemble_seeds()
    group = resolve_wandb_group()
    # Multi-seed: always start from per-seed init so LOAD_FROM cannot collapse seeds.
    load_from = LOAD_FROM if len(seeds) == 1 else None
    print(
        f"Ensemble: {len(seeds)} seed(s), group={group!r}, "
        f"load_from={load_from!r}"
    )

    for seed in seeds:
        train_one_seed(
            seed,
            x_train,
            y_train,
            x_test,
            y_test,
            group=group,
            ensemble_size=len(seeds),
            load_from=load_from,
        )
