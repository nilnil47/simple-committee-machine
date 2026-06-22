# -*- coding: utf-8 -*-
"""Langevin training for a small FCN on synthetic cubic regression (Colab origin).

Original notebook:
    https://colab.research.google.com/drive/1tSHdWsKMiiqmil2SRoxr14eEK3hoerB6

Run (from repo root, in your existing Python env with torch + wandb):

    pip install -r requirements.txt
    python noa_training_code.py

Uses the original Colab hyperparameters by default (5000 epochs, [20,20,1] erf FCN,
n_train=300, lr0=1e-3, Langevin optimizer). W&B logging is added on top.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv

load_dotenv()
if not (os.environ.get("WANDB_API_KEY") or "").strip():
    os.environ.pop("WANDB_API_KEY", None)

from prepare import _ensure_wandb_auth, _git_commit_short


def resolve_device() -> str:
    # Original notebook used cuda:0; fall back to cpu when no GPU.
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


DTYPE = torch.float32
DEVICE = resolve_device()
criterion = nn.MSELoss(reduction="sum")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_data(d, n, seed, target_func):
    np.random.seed(seed)
    x = torch.tensor(np.random.normal(loc=0, scale=1.0, size=(n, d))).to(dtype=DTYPE)
    return x, target_func(x)


def get_train_test_data(d, n, n_test, train_seed, test_seed, target_func):
    x_train, y_train = get_data(d, n, train_seed, target_func)
    x_test, y_test = get_data(d, n_test, test_seed, target_func)
    return x_train, y_train, x_test, y_test


def prep_train_test(d, n, n_test, train_seed, test_seed, my_target):
    x_train, y_train, x_test, y_test = get_train_test_data(
        d, n, n_test, train_seed, test_seed, my_target
    )
    train_data = torch.utils.data.TensorDataset(
        x_train.to(dtype=DTYPE), y_train.to(dtype=DTYPE)
    )
    test_data = torch.utils.data.TensorDataset(
        x_test.to(dtype=DTYPE), y_test.to(dtype=DTYPE)
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=n, shuffle=False, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=n_test, shuffle=False, num_workers=1
    )
    return x_train, x_test, y_train, y_test, train_loader, test_loader


def calc_run_time(start, end):
    seconds = int(end - start)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Optimizer and model
# ---------------------------------------------------------------------------


class LangevinSimpleL(optim.Optimizer):
    """Langevin (GD + noise) optimizer; noise is standard normal."""

    def __init__(self, model: nn.Module, learning_rate, weight_decays, temperature):
        defaults = {
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "temperature": temperature,
        }
        groups = []
        for i, layer in enumerate(model.layer_funcs):
            group = {
                "params": layer.parameters(),
                "learning_rate": learning_rate,
                "weight_decay": weight_decays[i],
                "temperature": temperature,
            }
            groups.append(group)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            learning_rate = group["learning_rate"]
            weight_decay = group["weight_decay"]
            temperature = group["temperature"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                d_p = torch.randn_like(parameter) * (2 * learning_rate * temperature) ** 0.5
                d_p.add_(parameter.grad, alpha=-learning_rate)
                d_p.add_(parameter, alpha=-learning_rate * weight_decay)
                parameter.add_(d_p)


class FCN_L_layers(nn.Module):
    def __init__(self, cs, inits, activation, init_seed=None):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.layer_funcs = nn.ModuleList()
        for i in range(len(cs) - 1):
            layer = nn.Linear(cs[i], cs[i + 1], bias=False)
            nn.init.normal_(layer.weight, 0, inits[i] ** 0.5)
            self.layer_funcs.append(layer)

        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layer_funcs):
            x = layer(x)
            if i < len(self.layer_funcs) - 1:
                if self.activation == "relu":
                    x = torch.nn.ReLU()(x)
                elif self.activation == "erf":
                    x = torch.erf(x)
                elif self.activation == "lin":
                    pass
                else:
                    raise ValueError(f"Unsupported activation: {self.activation}")
        return x.squeeze(1)


class MyNetwork:
    def __init__(
        self,
        layer_widths,
        n_train,
        n_test,
        max_epochs,
        sigma_2,
        sigma_l2s,
        fl_scale,
        seeds,
        activation,
        save_path,
        lr_list,
        target,
        device: str,
    ):
        self.Ns = layer_widths
        self.n = int(n_train)
        self.n_test = int(n_test)
        self.activation = activation
        self.sl2s, self.s2 = sigma_l2s, sigma_2
        self.FL_scale = fl_scale
        self.device = device

        self.sl2s_MF_scale, self.sigma_2_MF_scale = sigma_l2s, self.s2 * fl_scale / self.Ns[-2]
        self.sl2s_MF_scale[-1] *= fl_scale / self.Ns[-2]

        inits = [self.sl2s_MF_scale[i] / self.Ns[i] for i in range(len(self.sl2s_MF_scale))]
        self.net = FCN_L_layers(self.Ns, inits, self.activation).to(device)

        self.lr_list = [(lr_list[i][0], lr_list[i][1] / self.n) for i in range(len(lr_list))]
        self.max_epochs = max_epochs
        # Match original notebook: start with unscaled lr0/2, scheduler uses n-scaled values.
        self.lr = lr_list[0][1]
        self.temperature = 2 * self.sigma_2_MF_scale
        self.wds = [self.temperature * 1 / inits[i] for i in range(len(inits))]

        self.train_seed, self.test_seed = seeds
        (
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
            self.train_loader,
            self.test_loader,
        ) = prep_train_test(
            self.Ns[0], self.n, self.n_test, self.train_seed, self.test_seed, target
        )
        for data in self.train_loader:
            inputs, labels = data
            self.inputs, self.labels = inputs.to(device), labels.to(device)
        for data_test in self.test_loader:
            inputs_test, labels_test = data_test
            self.inputs_test, self.labels_test = inputs_test.to(device), labels_test.to(
                device
            )

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.saved_networks_state_dicts = []

    def _save_current_state_to_memory(self):
        self.saved_networks_state_dicts.append(copy.deepcopy(self.net.state_dict()))

    def one_epoch(self, optimizer):
        self.net.train()
        optimizer.zero_grad()
        outputs = self.net(self.inputs)
        loss = criterion(outputs, self.labels)
        outputs_test_full = self.net(self.inputs_test)
        loss_test = criterion(outputs_test_full, self.labels_test)
        loss.backward()
        optimizer.step()
        return loss.item() / len(self.Y_train), loss_test.item() / len(self.Y_test)

    def train_net(self, log_every: int = 1) -> str:
        start = time.time()
        optimizer = LangevinSimpleL(self.net, self.lr, self.wds, self.temperature)
        lr_ind = 0
        self.losses_train, self.losses_test = [], []
        self.epochs = []

        for epoch in range(self.max_epochs):
            train_loss, test_loss = self.one_epoch(optimizer)
            self.losses_train.append(train_loss)
            self.losses_test.append(test_loss)
            self.epochs.append(epoch)

            if (epoch + 1) % 500 == 0 or epoch == self.max_epochs - 1:
                self._save_current_state_to_memory()

            if epoch == self.lr_list[lr_ind][0]:
                self.lr = self.lr_list[lr_ind][1]
                print("lr updated")
                print(self.lr)
                if lr_ind < len(self.lr_list) - 1:
                    lr_ind += 1

            if wandb.run is not None and (epoch % log_every == 0 or epoch == self.max_epochs - 1):
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "lr": self.lr,
                        "train_test_gap": test_loss - train_loss,
                    }
                )

            if (epoch + 1) % 1000 == 0:
                print(
                    f"epoch {epoch + 1}: train={train_loss:.6f} test={test_loss:.6f}"
                )

            optimizer = LangevinSimpleL(self.net, self.lr, self.wds, self.temperature)

        run_time = calc_run_time(start, time.time())
        print(f"training time: {run_time}")
        self.training_seconds = time.time() - start

        self.losses_test = np.asarray(self.losses_test)
        self.losses_train = np.asarray(self.losses_train)
        return run_time


def save_loss_plots(net: MyNetwork, plot_dir: Path) -> tuple[Path, Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs_to_plot = np.concatenate((net.epochs[:100], net.epochs[100::100]))
    losses_test = np.concatenate((net.losses_test[:100], net.losses_test[100::100]))
    losses_train = np.concatenate((net.losses_train[:100], net.losses_train[100::100]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(epochs_to_plot, losses_test)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("test loss")
    axes[0].set_title("Test loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_to_plot, losses_train)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train loss")
    axes[1].set_title("Train loss")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    linear_path = plot_dir / "loss_curves.png"
    fig.savefig(linear_path, dpi=150)
    plt.close(fig)

    loglog_path = plot_dir / "loss_loglog.png"
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.loglog(net.epochs, net.losses_train, label="train", alpha=0.8)
    ax2.loglog(net.epochs, net.losses_test, label="test", alpha=0.8)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(loglog_path, dpi=150)
    plt.close(fig2)

    return linear_path, loglog_path


def save_prediction_scatter(net: MyNetwork, plot_dir: Path) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        y_pred_test = net.net(net.X_test.to(net.device)).detach().cpu().numpy()
        y_pred_train = net.net(net.X_train.to(net.device)).detach().cpu().numpy()
    y_test = net.Y_test.detach().cpu().numpy()
    y_train = net.Y_train.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred_test, y_test, alpha=0.5, label="test", s=12)
    ax.scatter(y_pred_train, y_train, alpha=0.5, label="train", s=12)
    ax.set_xlabel("prediction")
    ax.set_ylabel("target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = plot_dir / "pred_vs_target.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    return scatter_path


# ---------------------------------------------------------------------------
# Original Colab experiment settings (Training and Network settings cell)
# ---------------------------------------------------------------------------

INIT_SEED = 222
TEST_SEED = 10
TARGET = lambda x: x[:, 0] ** 3 - 3 * x[:, 0]
SAVE_PATH = "/home/noa/feature_united_icml_25/trained_networks/"
LOCAL_SAVE_PATH = Path(__file__).resolve().parent / "noa_runs" / "trained_networks"

MAX_EPOCHS = 5000
LR0 = 1e-3
LR_LIST = [(MAX_EPOCHS, LR0 / 2), (int(1.2 * MAX_EPOCHS), 0)]
N_TRAIN = 300
N_TEST = 400
NUM_NETS = 1
SL2S = [1.0, 1.0]
S2 = 1.0
WIDTHS = [20, 20, 1]
FL_SCALE = 1.0
ACTIVATION = "erf"


def resolve_save_path(requested: Path | None = None) -> Path:
    """Use original Colab path when available; otherwise repo-local fallback."""
    candidates = (
        [requested]
        if requested is not None
        else [Path(SAVE_PATH), LOCAL_SAVE_PATH]
    )
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if requested is None and path != Path(SAVE_PATH):
                print(f"Original save path unavailable ({SAVE_PATH}); using {path}")
            return path
        except OSError:
            continue
    raise RuntimeError(f"Could not create save directory (tried: {candidates})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original Noa Langevin FCN setup with W&B logging"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=f"Checkpoint/plot directory (default: {SAVE_PATH} or {LOCAL_SAVE_PATH})",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT_NOA")
        or os.environ.get("WANDB_PROJECT", "noa-langevin-fcn"),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics to W&B every N epochs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Number of training epochs (default: {MAX_EPOCHS})",
    )
    parser.add_argument("--offline", action="store_true", help="Use WANDB_MODE=offline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    max_epochs = args.epochs
    lr_list = [(max_epochs, LR0 / 2), (int(1.2 * max_epochs), 0)]

    save_path = resolve_save_path(args.save_path)
    seeds = np.random.randint(0, 3000, NUM_NETS)

    print(f"Device: {DEVICE}")
    print(
        "Original setup: "
        f"epochs={max_epochs}, n_train={N_TRAIN}, n_test={N_TEST}, "
        f"widths={WIDTHS}, activation={ACTIVATION}, lr0={LR0}, num_nets={NUM_NETS}"
    )
    print(f"Train seeds: {seeds.tolist()}, test_seed={TEST_SEED}")

    _ensure_wandb_auth()

    for k, train_seed in enumerate(seeds):
        # Original notebook: fresh DNN init each net, data seed from seeds[k].
        torch.seed()
        np.random.seed()

        run = wandb.init(
            project=args.project,
            name=f"net{k}_seed{train_seed}",
            config={
                "init_seed": INIT_SEED,
                "task": "cubic_regression_x0",
                "target": "x0^3 - 3*x0",
                "widths": WIDTHS,
                "activation": ACTIVATION,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "max_epochs": max_epochs,
                "lr0": LR0,
                "lr_list": lr_list,
                "initial_lr": LR0 / 2,
                "sl2s": SL2S,
                "sigma_2": S2,
                "fl_scale": FL_SCALE,
                "train_seed": int(train_seed),
                "test_seed": TEST_SEED,
                "optimizer": "LangevinSimpleL",
                "device": DEVICE,
                "save_path": str(save_path),
                "git_commit": _git_commit_short(),
            },
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("test_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("train_test_gap", step_metric="epoch")

        net = MyNetwork(
            WIDTHS,
            N_TRAIN,
            N_TEST,
            max_epochs,
            S2,
            SL2S,
            FL_SCALE,
            [int(train_seed), TEST_SEED],
            ACTIVATION,
            str(save_path),
            lr_list,
            TARGET,
            DEVICE,
        )

        run_time = net.train_net(log_every=args.log_every)
        linear_plot, loglog_plot = save_loss_plots(net, save_path)
        scatter_plot = save_prediction_scatter(net, save_path)

        wandb.log(
            {
                "loss_curves": wandb.Image(str(linear_plot)),
                "loss_loglog": wandb.Image(str(loglog_plot)),
                "pred_vs_target": wandb.Image(str(scatter_plot)),
            }
        )
        wandb.run.summary.update(
            {
                "final_train_loss": float(net.losses_train[-1]),
                "final_test_loss": float(net.losses_test[-1]),
                "final_train_test_gap": float(net.losses_test[-1] - net.losses_train[-1]),
                "training_seconds": float(net.training_seconds),
                "run_time_hms": run_time,
                "snapshots_taken": len(net.saved_networks_state_dicts),
            }
        )

        print("--- Training Summary ---")
        print(f"Net {k} (train_seed={train_seed})")
        print(f"Total Epochs Completed: {len(net.epochs)}")
        print(f"Initial Learning Rate:  {LR0 / 2}")
        print(f"Final Train Loss:       {net.losses_train[-1]:.6f}")
        print(f"Final Test Loss:        {net.losses_test[-1]:.6f}")
        print(f"Snapshots taken:        {len(net.saved_networks_state_dicts)}")
        print(f"Saved plots to:         {save_path}")
        print(f"Logged to W&B project:  {args.project}")
        print(f"done with {k}")

        run.finish()


if __name__ == "__main__":
    main()
