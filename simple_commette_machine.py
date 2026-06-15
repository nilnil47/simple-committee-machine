"""Committee student on linear teacher w*·x. Offline data, full-batch SGD."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

DIMENSION = 10
N_HIDDEN = 32
LR = 0.001
EPOCHS = 10_000
SEED = 42
INIT_SEED = 42

N_TRAIN_TOTAL = 1000
N_TEST_TOTAL = 1000
N_TRAIN_USED = 100
N_TEST_USED = 200

CACHE = Path.home() / ".cache" / "simple-committee-machine"
INIT_WEIGHTS_PATH = CACHE / "student_init.pt"
TRAINED_WEIGHTS_PATH = CACHE / "student_trained.pt"

# None = start from saved init; "trained" = load TRAINED_WEIGHTS_PATH; or any .pt path
LOAD_FROM = None


def teacher(x, w):
    z = x @ w
    return z.squeeze()
    # return (z**3 - 3 * z).squeeze()


class CommitteeStudent(nn.Module):
    def __init__(self, d, n_hidden):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n_hidden)
        self.W = nn.Parameter(torch.empty(n_hidden, d))
        nn.init.normal_(self.W, 0.0, 1.0 / math.sqrt(d))
        with torch.no_grad():
            self.W /= torch.norm(self.W)

    def forward(self, x):
        return self.scale * torch.erf(x @ self.W.T).sum(dim=-1)


def ensure_init_weights() -> None:
    """Sample init once from N(0, 1/sqrt(d)), Frobenius-normalize, cache for all runs."""
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


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "x_train.pt").exists():
        w_star = torch.load(CACHE / "w_star.pt", weights_only=True)
        x_train = torch.load(CACHE / "x_train.pt", weights_only=True)
        x_test = torch.load(CACHE / "x_test.pt", weights_only=True)
    else:
        g = torch.Generator().manual_seed(SEED)
        w_star = torch.randn(DIMENSION, 1, generator=g)
        w_star /= torch.norm(w_star)
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
            "dimension": DIMENSION,
            "n_hidden": N_HIDDEN,
            "lr": LR,
            "epochs": EPOCHS,
            "n_train_used": N_TRAIN_USED,
            "n_test_used": N_TEST_USED,
            "init_seed": INIT_SEED,
            "loaded_from": str(loaded_from),
        },
    )
    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        student.train()
        loss = loss_fn(student(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_loss = loss_fn(student(x_test), y_test).item()
        wandb.log({"epoch": epoch, "loss": loss.item(), "test_loss": test_loss})

        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1}: train={loss.item():.4f} test={test_loss:.4f}")

    torch.save(student.state_dict(), TRAINED_WEIGHTS_PATH)
    print(f"Saved trained weights to {TRAINED_WEIGHTS_PATH}")
    wandb.finish()
