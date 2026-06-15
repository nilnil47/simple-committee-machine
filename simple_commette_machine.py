"""Committee student on linear teacher w*·x. Offline data, full-batch SGD."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

DIMENSION = 10
N_HIDDEN = 32
LR = 0.001
EPOCHS = 100_000
SEED = 42
INIT_SEED = 42

N_TRAIN_TOTAL = 1000
N_TEST_TOTAL = 1000
N_TRAIN_USED = 100
N_TEST_USED = 200

CACHE = Path(".cache").parent / "simple-committee-machine"
INIT_WEIGHTS_PATH = CACHE / "student_init.pt"
TRAINED_WEIGHTS_PATH = CACHE / "student_trained.pt"
LOSS_LOGLOG_PLOT_PATH = CACHE / "loss_loglog.png"
CHECKPOINT_DIR = CACHE / "checkpoints"

# 1-indexed epochs at which to save student weights; None = final checkpoint only
SAVE_EPOCHS: list[int] | None = [500]

# Teacher target: "linear" (w*·x) or "hermite" (He_3(w*·x))
# TASK = "linear"
TASK = "hermite"

# None = start from saved init; "trained" = load TRAINED_WEIGHTS_PATH; or any .pt path
LOAD_FROM = CACHE / "student_init.pt"
# LOAD_FROM = CACHE / "student_trained_linear.pt"
# LOAD_FROM = CACHE / "checkpoints" / "student_epoch_00500.pt"
# LOAD_FROM = None


def teacher(x, w):
    z = x @ w
    if TASK == "linear":
        return z.squeeze()
    if TASK == "hermite":
        return (z**3 - 3 * z).squeeze()
    raise ValueError(f"Unknown task: {TASK!r} (expected 'linear' or 'hermite')")


class CommitteeStudent(nn.Module):
    def __init__(self, d, n_hidden):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n_hidden)
        self.W = nn.Parameter(torch.empty(n_hidden, d))
        nn.init.normal_(self.W, 0.0, 1.0 / math.sqrt(d))
        with torch.no_grad():
            self.W /= torch.norm(self.W, dim=1, keepdim=True)

    def forward(self, x):
        return self.scale * torch.erf(x @ self.W.T).sum(dim=-1)


def make_teacher_weights(dimension: int, generator: torch.Generator) -> torch.Tensor:
    w_star = torch.randn(dimension, 1, generator=generator)
    return w_star / torch.norm(w_star)


def make_student(dimension: int, n_hidden: int, seed: int) -> CommitteeStudent:
    torch.manual_seed(seed)
    return CommitteeStudent(dimension, n_hidden)


def empirical_cross_gradient(
    student: CommitteeStudent, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Gradient of mean(y * f(x)) = (1/P) sum_p y_p grad f(x_p)."""
    student.zero_grad()
    loss = (y * student(x)).mean()
    loss.backward()
    return student.W.grad.detach().clone()


def ensure_init_weights() -> None:
    """Sample init once from N(0, 1/sqrt(d)), per-row unit normalize, cache for all runs."""
    if INIT_WEIGHTS_PATH.exists():
        return
    torch.manual_seed(INIT_SEED)
    student = CommitteeStudent(DIMENSION, N_HIDDEN)
    torch.save(student.state_dict(), INIT_WEIGHTS_PATH)
    print(f"Saved init weights to {INIT_WEIGHTS_PATH}")


def resolve_checkpoint(load_from: str | None) -> Path:
    if load_from is None:
        return INIT_WEIGHTS_PATH
    if load_from == "trained":
        return TRAINED_WEIGHTS_PATH
    return Path(load_from)


def load_student(load_from: str | None) -> tuple[CommitteeStudent, Path]:
    ensure_init_weights()
    student = CommitteeStudent(DIMENSION, N_HIDDEN)
    path = resolve_checkpoint(load_from)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    student.load_state_dict(torch.load(path, weights_only=True))
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


def save_loss_loglog_plot(
    epochs: list[int],
    train_losses: list[float],
    test_losses: list[float],
    path: Path,
) -> Path:
    eps = 1e-12
    x = [e + 1 for e in epochs]
    train_y = [max(v, eps) for v in train_losses]
    test_y = [max(v, eps) for v in test_losses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, train_y, label="train loss", linewidth=1.5)
    ax.loglog(x, test_y, label="test loss", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "x_train.pt").exists():
        w_star = torch.load(CACHE / "w_star.pt", weights_only=True)
        x_train = torch.load(CACHE / "x_train.pt", weights_only=True)
        x_test = torch.load(CACHE / "x_test.pt", weights_only=True)
    else:
        g = torch.Generator().manual_seed(SEED)
        w_star = make_teacher_weights(DIMENSION, g)
        x_train = torch.randn(N_TRAIN_TOTAL, DIMENSION, generator=g)
        x_test = torch.randn(N_TEST_TOTAL, DIMENSION, generator=g)
        torch.save(w_star, CACHE / "w_star.pt")
        torch.save(x_train, CACHE / "x_train.pt")
        torch.save(x_test, CACHE / "x_test.pt")

    x_train = x_train[:N_TRAIN_USED]
    x_test = x_test[:N_TEST_USED]
    y_train = teacher(x_train, w_star)
    y_test = teacher(x_test, w_star)

    student, loaded_from = load_student(LOAD_FROM)

    wandb.init(
        project="hermite-distillation",
        config={
            "task": TASK,
            "dimension": DIMENSION,
            "n_hidden": N_HIDDEN,
            "lr": LR,
            "epochs": EPOCHS,
            "n_train_used": N_TRAIN_USED,
            "n_test_used": N_TEST_USED,
            "init_seed": INIT_SEED,
            "loaded_from": str(loaded_from),
            "save_epochs": SAVE_EPOCHS,
        },
    )
    save_epochs = set(SAVE_EPOCHS or ())
    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0)
    loss_fn = nn.MSELoss()
    epochs_hist: list[int] = []
    train_loss_hist: list[float] = []
    test_loss_hist: list[float] = []

    for epoch in range(EPOCHS):
        student.train()
        loss = loss_fn(student(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_loss = loss_fn(student(x_test), y_test).item()

        epochs_hist.append(epoch)
        train_loss_hist.append(loss.item())
        test_loss_hist.append(test_loss)
        wandb.log({"epoch": epoch, "loss": loss.item(), "test_loss": test_loss})

        epoch_num = epoch + 1
        if epoch_num in save_epochs:
            save_student_checkpoint(student, epoch_num)

        if epoch_num % 1000 == 0:
            print(f"epoch {epoch_num}: train={loss.item():.4f} test={test_loss:.4f}")

    plot_path = save_loss_loglog_plot(
        epochs_hist, train_loss_hist, test_loss_hist, LOSS_LOGLOG_PLOT_PATH
    )
    print(f"Saved log-log loss plot to {plot_path}")
    wandb.log({"loss_loglog": wandb.Image(str(plot_path))})

    torch.save(student.state_dict(), TRAINED_WEIGHTS_PATH)
    print(f"Saved trained weights to {TRAINED_WEIGHTS_PATH}")
    wandb.finish()
