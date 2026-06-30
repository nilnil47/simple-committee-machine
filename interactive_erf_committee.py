"""Interactive continued training with keyboard weight injection.

Load a trained checkpoint, keep training, and hard-snap hidden weight rows to
preset targets on keypress while watching loss, prediction curves, and w·w*.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from erf_combo_commette_machine import (
    CACHE,
    DIMENSION,
    LR,
    N,
    N_TEST_TOTAL,
    N_TEST_USED,
    N_TRAIN_TOTAL,
    OPTIMIZER,
    P,
    SEED,
    TRAINED_WEIGHTS_PATH,
    CommitteeStudent,
    data_cache_valid,
    load_student,
    make_optimizer,
    make_theory_grid,
    mse_vs_teacher,
    project_onto_w_star,
    student_hidden_weights,
    teacher_erf_combo,
)

# key -> (hidden unit index 0..N-1, target weight vector length d)
WEIGHT_PRESETS: dict[str, tuple[int, list[float]]] = {
    "1": (0, [1.0] + [0.0] * (DIMENSION - 1)),
    "2": (1, [1.0] + [0.0] * (DIMENSION - 1)),
    "3": (2, [1.0] + [0.0] * (DIMENSION - 1)),
    "4": (3, [1.0] + [0.0] * (DIMENSION - 1)),
    "5": (4, [-0.5] + [0.0] * (DIMENSION - 1)),
    "6": (5, [-0.5] + [0.0] * (DIMENSION - 1)),
    "7": (6, [-0.5] + [0.0] * (DIMENSION - 1)),
    "8": (7, [-0.5] + [0.0] * (DIMENSION - 1)),
    "9": (8, [-0.5] + [0.0] * (DIMENSION - 1)),
    "0": (9, [-0.5] + [0.0] * (DIMENSION - 1)),
    "a": (10, [-0.5] + [0.0] * (DIMENSION - 1)),
    "b": (11, [-0.5] + [0.0] * (DIMENSION - 1)),
    "c": (12, [0.0] * DIMENSION),
    "d": (13, [0.0] * DIMENSION),
    "e": (14, [0.0] * DIMENSION),
    "f": (15, [0.0] * DIMENSION),
}

CHECKPOINT = TRAINED_WEIGHTS_PATH
EPOCHS_PER_UI_TICK = 10
# None = train until you close the plot window or press q; set e.g. 10_000 for a fixed cap
MAX_EPOCHS: int | None = 10_000
PAUSE_ON_START = False
UI_REFRESH_MS = 100


def validate_weight_presets(
    presets: dict[str, tuple[int, list[float]]], n: int, d: int
) -> None:
    seen_units: set[int] = set()
    for key, (unit_idx, target) in presets.items():
        if unit_idx < 0 or unit_idx >= n:
            raise ValueError(
                f"WEIGHT_PRESETS[{key!r}] unit index {unit_idx} out of range [0, {n})"
            )
        if len(target) != d:
            raise ValueError(
                f"WEIGHT_PRESETS[{key!r}] target must have length d={d}, got {len(target)}"
            )
        seen_units.add(unit_idx)


def print_key_map(presets: dict[str, tuple[int, list[float]]]) -> None:
    print("Weight injection key map (hard snap on keypress):")
    for key in sorted(presets.keys(), key=_sort_preset_keys):
        unit_idx, target = presets[key]
        preview = _format_vector_preview(target)
        print(f"  [{key}] -> unit {unit_idx}: {preview}")
    print("Controls: [space] pause/resume  [q] quit")


def _sort_preset_keys(key: str) -> tuple[int, str]:
    if key.isdigit():
        return (0, key)
    return (1, key)


def _format_vector_preview(target: list[float], max_dims: int = 4) -> str:
    head = ", ".join(f"{v:.2g}" for v in target[:max_dims])
    if len(target) > max_dims:
        head += ", ..."
    return f"[{head}]"


def reset_adam_row(
    optimizer: optim.Optimizer, param: nn.Parameter, unit_idx: int
) -> None:
    """Zero Adam momentum buffers for one row of a 2D parameter tensor."""
    state = optimizer.state.get(param)
    if state is None:
        return
    for buf_name in ("exp_avg", "exp_avg_sq"):
        buf = state.get(buf_name)
        if buf is not None and buf.ndim == 2:
            buf[unit_idx].zero_()


def apply_injection(
    student: CommitteeStudent,
    unit_idx: int,
    target: list[float],
    optimizer: optim.Optimizer,
) -> None:
    with torch.no_grad():
        student.W.data[unit_idx] = torch.as_tensor(
            target, dtype=student.W.dtype, device=student.W.device
        )
    if isinstance(optimizer, optim.Adam):
        reset_adam_row(optimizer, student.W, unit_idx)


@dataclass
class TrainingState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    injection_queue: queue.Queue[tuple[int, list[float]]] = field(
        default_factory=queue.Queue
    )
    paused: bool = PAUSE_ON_START
    stop: bool = False
    epoch: int = 0
    train_loss: float = 0.0
    test_loss: float = 0.0
    epochs_hist: list[int] = field(default_factory=list)
    train_loss_hist: list[float] = field(default_factory=list)
    test_loss_hist: list[float] = field(default_factory=list)
    last_injected_unit: int | None = None
    status_message: str = "running"
    stop_reason: str = ""


def load_training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return x_train, y_train, x_test, y_test


def drain_injections(
    student: CommitteeStudent,
    optimizer: optim.Optimizer,
    state: TrainingState,
) -> None:
    while True:
        try:
            unit_idx, target = state.injection_queue.get_nowait()
        except queue.Empty:
            break
        apply_injection(student, unit_idx, target, optimizer)
        state.last_injected_unit = unit_idx
        state.status_message = f"injected unit {unit_idx}"


def training_loop(
    student: CommitteeStudent,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    state: TrainingState,
) -> None:
    while not state.stop:
        with state.lock:
            if state.paused:
                pass
            else:
                for _ in range(EPOCHS_PER_UI_TICK):
                    if state.stop:
                        break
                    if MAX_EPOCHS is not None and state.epoch >= MAX_EPOCHS:
                        state.stop = True
                        state.stop_reason = f"reached MAX_EPOCHS={MAX_EPOCHS}"
                        state.status_message = state.stop_reason
                        break
                    drain_injections(student, optimizer, state)

                    student.train()
                    loss = loss_fn(student(x_train), y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        test_loss = loss_fn(student(x_test), y_test).item()

                    state.epoch += 1
                    state.train_loss = loss.item()
                    state.test_loss = test_loss
                    state.epochs_hist.append(state.epoch)
                    state.train_loss_hist.append(state.train_loss)
                    state.test_loss_hist.append(state.test_loss)

        threading.Event().wait(0.01)


class InteractiveDashboard:
    def __init__(
        self,
        student: CommitteeStudent,
        state: TrainingState,
        x_grid: torch.Tensor,
        y_teacher_grid: torch.Tensor,
        y_student_start: torch.Tensor,
    ) -> None:
        self.student = student
        self.state = state
        self.x_grid = x_grid
        self.y_teacher_grid = y_teacher_grid
        self.y_student_start = y_student_start
        self.xs = x_grid[:, 0].numpy()

        self.fig, self.axes = plt.subplots(3, 1, figsize=(9, 10))
        self._setup_axes()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.timer = self.fig.canvas.new_timer(interval=UI_REFRESH_MS)
        self.timer.add_callback(self._refresh)
        self.timer.start()

    def _setup_axes(self) -> None:
        ax_loss, ax_curve, ax_weights = self.axes

        (self.train_line,) = ax_loss.plot([], [], label="train loss", color="C0")
        (self.test_line,) = ax_loss.plot([], [], label="test loss", color="C1")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("MSE")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        self.status_text = ax_loss.text(
            0.02,
            0.98,
            "",
            transform=ax_loss.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "wheat", "alpha": 0.8, "pad": 4},
        )

        ax_curve.plot(
            self.xs,
            self.y_teacher_grid.numpy(),
            label="teacher",
            color="C2",
            linewidth=2.0,
        )
        (self.start_curve,) = ax_curve.plot(
            self.xs,
            self.y_student_start.numpy(),
            label="start (checkpoint)",
            color="C0",
            linewidth=1.0,
            alpha=0.5,
            linestyle="--",
        )
        (self.student_curve,) = ax_curve.plot(
            [], [], label="student", color="C0", linewidth=1.5
        )
        ax_curve.set_xlabel(r"$x_1$")
        ax_curve.set_ylabel("f(x)")
        ax_curve.legend(loc="upper left")
        ax_curve.grid(True, alpha=0.3)

        unit_labels = [str(i) for i in range(N)]
        self.weight_bars = ax_weights.bar(
            range(N),
            project_onto_w_star(student_hidden_weights(self.student)).numpy(),
            color="C0",
            alpha=0.85,
        )
        ax_weights.set_xticks(range(N))
        ax_weights.set_xticklabels(unit_labels)
        ax_weights.set_xlabel("hidden unit")
        ax_weights.set_ylabel(r"$w_p \cdot w^*$")
        ax_weights.set_title("weight projections")
        ax_weights.grid(True, axis="y", alpha=0.3)

        self.fig.tight_layout()

    def _on_key_press(self, event) -> None:
        if event.key is None:
            return

        key = event.key
        if key == " ":
            with self.state.lock:
                self.state.paused = not self.state.paused
                self.state.status_message = (
                    "paused" if self.state.paused else "running"
                )
            return

        if key == "q":
            self.state.stop = True
            self.state.stop_reason = "quit key"
            self.state.status_message = "quitting"
            plt.close(self.fig)
            return

        preset = WEIGHT_PRESETS.get(key)
        if preset is not None:
            unit_idx, target = preset
            self.state.injection_queue.put((unit_idx, target))
            self.state.status_message = f"queued unit {unit_idx}"

    def _refresh(self) -> None:
        if not plt.fignum_exists(self.fig.number):
            self.state.stop = True
            self.state.stop_reason = "plot window closed"
            return

        with self.state.lock:
            epochs = list(self.state.epochs_hist)
            train_losses = list(self.state.train_loss_hist)
            test_losses = list(self.state.test_loss_hist)
            status = self.state.status_message
            paused = self.state.paused
            last_unit = self.state.last_injected_unit
            epoch = self.state.epoch
            train_loss = self.state.train_loss
            test_loss = self.state.test_loss

        self.train_line.set_data(epochs, train_losses)
        self.test_line.set_data(epochs, test_losses)
        if epochs:
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        pause_suffix = " [PAUSED]" if paused else ""
        self.status_text.set_text(
            f"epoch={epoch}  train={train_loss:.4f}  test={test_loss:.4f}"
            f"  {status}{pause_suffix}"
        )

        self.student.eval()
        with torch.no_grad():
            y_student = self.student(self.x_grid).numpy()
        self.student_curve.set_data(self.xs, y_student)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        w_proj = project_onto_w_star(student_hidden_weights(self.student)).numpy()
        for bar, value in zip(self.weight_bars, w_proj):
            bar.set_height(value)
        if last_unit is not None:
            for i, bar in enumerate(self.weight_bars):
                bar.set_color("C3" if i == last_unit else "C0")
                bar.set_alpha(1.0 if i == last_unit else 0.85)
        y_lo = min(w_proj.min(), 0.0)
        y_hi = max(w_proj.max(), 0.0)
        if y_lo == y_hi:
            y_lo -= 0.5
            y_hi += 0.5
        self.axes[2].set_ylim(y_lo, y_hi)

        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    validate_weight_presets(WEIGHT_PRESETS, N, DIMENSION)
    print_key_map(WEIGHT_PRESETS)

    x_train, y_train, x_test, y_test = load_training_data()
    student, loaded_from = load_student(CHECKPOINT)
    print(f"Loaded checkpoint: {loaded_from}")

    x_grid = make_theory_grid()
    y_teacher_grid = teacher_erf_combo(x_grid)
    student.eval()
    with torch.no_grad():
        y_student_start = student(x_grid).clone()
        init_test_mse = mse_vs_teacher(student(x_test), x_test)
    print(f"checkpoint test MSE={init_test_mse:.6f}")
    if MAX_EPOCHS is None:
        print(
            "Training runs until you close the plot window or press q "
            "(no epoch limit — keep the window open)."
        )
    else:
        print(f"Training stops after MAX_EPOCHS={MAX_EPOCHS} or when the window closes.")

    optimizer = make_optimizer(student.parameters(), LR, OPTIMIZER)
    loss_fn = nn.MSELoss()
    state = TrainingState(paused=PAUSE_ON_START)

    train_thread = threading.Thread(
        target=training_loop,
        args=(student, optimizer, loss_fn, x_train, y_train, x_test, y_test, state),
        daemon=True,
    )
    train_thread.start()

    dashboard = InteractiveDashboard(
        student, state, x_grid, y_teacher_grid, y_student_start
    )
    dashboard.show()

    state.stop = True
    train_thread.join(timeout=2.0)
    reason = state.stop_reason or "plot window closed"
    print(f"Stopped at epoch {state.epoch} ({reason})")


if __name__ == "__main__":
    main()
