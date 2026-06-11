"""
Fixed constants, data prep, and evaluation harness for committee autoresearch.
Do not modify during autonomous experiments — the agent only edits train.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Empty placeholder in .env must not block credentials from `wandb login`.
if not (os.environ.get("WANDB_API_KEY") or "").strip():
    os.environ.pop("WANDB_API_KEY", None)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- Fixed experiment constants (not agent-editable) ---

DIMENSION = 128
N_HIDDEN = 64
TRAIN_SAMPLES = 512
VAL_SAMPLES = 10_000
BATCH_SIZE = 256
TRAINING_SECONDS = 300  # 5 min wall-clock training budget
EVAL_EVERY_STEPS = 500
SEED = 42

CACHE_DIR = Path.home() / ".cache" / "committee-autoresearch"
WANDB_PROJECT = "committee-autoresearch"
VAL_LOSS_PLOT_PATH = Path("val_loss_plot.png")


def hermite_teacher(x: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
    """Third Hermite polynomial He_3(z) = z^3 - 3z on projection z = w* · x."""
    z = x @ w_star
    return (z**3 - 3 * z).squeeze(-1)


def make_teacher_weights(dimension: int, generator: torch.Generator) -> torch.Tensor:
    w_star = torch.randn(dimension, 1, generator=generator)
    return w_star / torch.norm(w_star)


def compute_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            total_loss += torch.sum((y_pred - y_batch) ** 2).item()
            total_samples += y_batch.numel()
    model.train()
    return total_loss / total_samples


def compute_overlaps(
    model: torch.nn.Module,
    w_star: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_overlap, mean_overlap) of student hidden weights with teacher."""
    w_star_flat = w_star.squeeze(-1)
    overlaps = model.W @ w_star_flat
    return overlaps.abs().max().item(), overlaps.abs().mean().item()


def evaluate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    w_star: torch.Tensor,
    device: torch.device,
    val_history: list[tuple[int, float]] | None = None,
) -> dict[str, float]:
    """
    Final evaluation after training.

    val_history: optional list of (step, val_mse) from periodic eval during training,
    used to compute grok_epoch and plateau_length.
    """
    train_mse = compute_mse(model, train_loader, device)
    val_mse = compute_mse(model, val_loader, device)
    max_overlap, mean_overlap = compute_overlaps(model, w_star.to(device))

    grok_epoch = 0.0
    plateau_length = 0.0

    if val_history and len(val_history) > 0:
        initial_val_mse = val_history[0][1]
        threshold = 0.5 * initial_val_mse

        # grok_epoch: first step after warmup where val drops below 50% of initial
        warmup_steps = 100
        for step, v_mse in val_history:
            if step >= warmup_steps and v_mse < threshold:
                grok_epoch = float(step)
                break

        # plateau_length: steps where val stayed within 5% of its running max
        running_max = val_history[0][1]
        plateau_count = 0
        for _, v_mse in val_history:
            running_max = max(running_max, v_mse)
            if v_mse >= 0.95 * running_max:
                plateau_count += 1
        plateau_length = float(plateau_count)

    return {
        "val_mse": val_mse,
        "train_mse": train_mse,
        "train_val_gap": train_mse - val_mse,
        "max_overlap": max_overlap,
        "mean_overlap": mean_overlap,
        "grok_epoch": grok_epoch,
        "plateau_length": plateau_length,
    }


def print_summary(
    metrics: dict[str, float],
    training_seconds: float,
    total_seconds: float,
    num_steps: int,
) -> None:
    print("---")
    print(f"val_mse:          {metrics['val_mse']:.6f}")
    print(f"train_mse:        {metrics['train_mse']:.6f}")
    print(f"train_val_gap:    {metrics['train_val_gap']:.6f}")
    print(f"max_overlap:      {metrics['max_overlap']:.6f}")
    print(f"mean_overlap:     {metrics['mean_overlap']:.6f}")
    print(f"grok_epoch:       {metrics['grok_epoch']:.0f}")
    print(f"plateau_length:   {metrics['plateau_length']:.0f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_steps:        {num_steps}")


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_val_loss_plot(val_history: list[tuple[int, float]]) -> plt.Figure:
    steps, val_mses = zip(*val_history)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, val_mses, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Validation MSE vs Training Step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_val_loss_plot(val_history: list[tuple[int, float]], path: Path) -> Path:
    fig = make_val_loss_plot(val_history)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ensure_wandb_auth() -> None:
    """Use WANDB_API_KEY from .env when set; otherwise rely on `wandb login`."""
    import wandb

    api_key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if api_key:
        wandb.login(key=api_key, relogin=True)
        return
    if os.environ.get("WANDB_MODE") == "offline":
        return
    if wandb.api.api_key:
        return
    raise RuntimeError(
        "W&B is not authenticated. Set WANDB_API_KEY in .env or run: wandb login"
    )


def log_to_wandb(
    metrics: dict[str, float],
    val_history: list[tuple[int, float]],
    plot_path: Path,
    num_steps: int,
    training_seconds: float,
) -> None:
    import wandb

    _ensure_wandb_auth()
    commit = _git_commit_short()
    project_name = os.environ.get("WANDB_PROJECT", WANDB_PROJECT)
    run = wandb.init(
        project=project_name,
        config={
            "dimension": DIMENSION,
            "n_hidden": N_HIDDEN,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "training_seconds_budget": TRAINING_SECONDS,
            "git_commit": commit,
        },
        reinit=True,
    )
    wandb.log(
        {
            **metrics,
            "num_steps": num_steps,
            "training_seconds": training_seconds,
            "val_loss_curve": wandb.Image(str(plot_path)),
        }
    )
    if val_history:
        val_table = wandb.Table(
            columns=["step", "val_mse"],
            data=[[step, val_mse] for step, val_mse in val_history],
        )
        wandb.log({"val_mse_curve": wandb.plot.line(val_table, "step", "val_mse")})
    run.finish()
    print(f"Logged run to W&B project '{project_name}'")


def send_telegram_val_plot(
    plot_path: Path,
    metrics: dict[str, float],
    num_steps: int,
    training_seconds: float,
) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    commit = _git_commit_short()
    caption = (
        f"Committee autoresearch run ({commit})\n"
        f"val_mse: {metrics['val_mse']:.4f}\n"
        f"train_mse: {metrics['train_mse']:.4f}\n"
        f"max_overlap: {metrics['max_overlap']:.4f}\n"
        f"grok_epoch: {int(metrics['grok_epoch'])}\n"
        f"steps: {num_steps} | time: {training_seconds:.0f}s"
    )
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with plot_path.open("rb") as plot_file:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": plot_file},
            timeout=30,
        )
    response.raise_for_status()
    print("Sent validation loss plot to Telegram")


def report_run(
    metrics: dict[str, float],
    val_history: list[tuple[int, float]],
    training_seconds: float,
    total_seconds: float,
    num_steps: int,
) -> None:
    """Print summary, save plot, upload to W&B, and notify Telegram."""
    print_summary(metrics, training_seconds, total_seconds, num_steps)

    if not val_history:
        print("No validation history to plot; skipping W&B and Telegram.")
        return

    plot_path = save_val_loss_plot(val_history, VAL_LOSS_PLOT_PATH)
    print(f"Saved validation loss plot to {plot_path}")

    try:
        log_to_wandb(metrics, val_history, plot_path, num_steps, training_seconds)
    except Exception as exc:
        print(f"W&B logging failed: {exc}")
        print(
            "Fix: add WANDB_API_KEY to .env (https://wandb.ai/authorize) "
            "or run `wandb login`, then re-run train.py."
        )

    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        try:
            send_telegram_val_plot(plot_path, metrics, num_steps, training_seconds)
        except Exception as exc:
            print(f"Telegram notification failed: {exc}")
    else:
        print(
            "Telegram skipped: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable."
        )


def prepare_data() -> None:
    """Download/generate and cache train/val tensors and teacher weights."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "w_star": CACHE_DIR / "w_star.pt",
        "x_train": CACHE_DIR / "x_train.pt",
        "y_train": CACHE_DIR / "y_train.pt",
        "x_val": CACHE_DIR / "x_val.pt",
        "y_val": CACHE_DIR / "y_val.pt",
    }

    if all(p.exists() for p in paths.values()):
        print(f"Data already cached at {CACHE_DIR}")
        return

    print(f"Preparing data (d={DIMENSION}, train={TRAIN_SAMPLES}, val={VAL_SAMPLES})...")
    generator = torch.Generator().manual_seed(SEED)

    w_star = make_teacher_weights(DIMENSION, generator)
    x_train = torch.randn(TRAIN_SAMPLES, DIMENSION, generator=generator)
    y_train = hermite_teacher(x_train, w_star)
    x_val = torch.randn(VAL_SAMPLES, DIMENSION, generator=generator)
    y_val = hermite_teacher(x_val, w_star)

    torch.save(w_star, paths["w_star"])
    torch.save(x_train, paths["x_train"])
    torch.save(y_train, paths["y_train"])
    torch.save(x_val, paths["x_val"])
    torch.save(y_val, paths["y_val"])
    print(f"Cached data to {CACHE_DIR}")


def load_data() -> tuple[torch.Tensor, DataLoader, DataLoader, torch.device]:
    """Load cached data and return (w_star, train_loader, val_loader, device)."""
    if not (CACHE_DIR / "w_star.pt").exists():
        raise FileNotFoundError(
            f"Cached data not found at {CACHE_DIR}. Run: python prepare.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w_star = torch.load(CACHE_DIR / "w_star.pt", weights_only=True)
    x_train = torch.load(CACHE_DIR / "x_train.pt", weights_only=True)
    y_train = torch.load(CACHE_DIR / "y_train.pt", weights_only=True)
    x_val = torch.load(CACHE_DIR / "x_val.pt", weights_only=True)
    y_val = torch.load(CACHE_DIR / "y_val.pt", weights_only=True)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return w_star, train_loader, val_loader, device


if __name__ == "__main__":
    prepare_data()
