"""Interactive continued training with keyboard weight injection.

Load a trained checkpoint, keep training, and hard-snap hidden weight rows to
preset targets on keypress while watching loss, curves, and w·w* in the terminal.
"""

from __future__ import annotations

import curses
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from committee_network import (
    DIMENSION,
    INIT_MANUAL_NOISE_VAR,
    INIT_MEAN,
    INIT_MODE,
    INIT_SEED,
    INIT_VAR,
    INIT_W_MANUAL,
    N,
    CommitteeStudent,
    teacher_erf_combo,
)
from erf_combo_commette_machine import (
    CACHE,
    LR,
    N_TEST_TOTAL,
    N_TEST_USED,
    N_TRAIN_TOTAL,
    OPTIMIZER,
    P,
    SEED,
    data_cache_valid,
    load_student,
    make_optimizer,
    make_theory_grid,
    mse_vs_teacher,
    project_onto_w_star,
    save_loss_loglog_plot,
    save_pred_vs_theory_plot,
    save_theory_stages_plot,
    save_weight_distribution_plot,
    student_hidden_weights,
    trained_weights_path,
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

CHECKPOINT = trained_weights_path(INIT_SEED)
EPOCHS_PER_UI_TICK = 10
# None = train until q; set e.g. 10_000 for a fixed cap
MAX_EPOCHS: int | None = None
PAUSE_ON_START = False
UI_REFRESH_MS = 100

WANDB_PROJECT = "committee-student"
WANDB_RUN_NAME = "erf_combo_interactive"
# False = no W&B upload (wandb runs in disabled mode)
USE_WANDB = True

INTERACTIVE_PLOT_DIR = CACHE / "interactive"
LOSS_LOGLOG_PLOT_PATH = INTERACTIVE_PLOT_DIR / "loss_loglog.png"
PRED_VS_THEORY_PLOT_PATH = INTERACTIVE_PLOT_DIR / "pred_vs_theory.png"
THEORY_CURVES_PLOT_PATH = INTERACTIVE_PLOT_DIR / "theory_curves.png"
WEIGHT_DISTRIBUTION_PLOT_PATH = INTERACTIVE_PLOT_DIR / "weights_distribution.png"

# Sample x1 values shown in the terminal curve panel
CURVE_SAMPLE_X1 = [-2.0, -1.0, 0.0, 1.0, 2.0]


def validate_weight_presets(
    presets: dict[str, tuple[int, list[float]]], n: int, d: int
) -> None:
    for key, (unit_idx, target) in presets.items():
        if unit_idx < 0 or unit_idx >= n:
            raise ValueError(
                f"WEIGHT_PRESETS[{key!r}] unit index {unit_idx} out of range [0, {n})"
            )
        if len(target) != d:
            raise ValueError(
                f"WEIGHT_PRESETS[{key!r}] target must have length d={d}, got {len(target)}"
            )


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


def sparkline(values: list[float], width: int = 50) -> str:
    if not values:
        return "(no data yet)"
    sample = values[-width:]
    lo = min(sample)
    hi = max(sample)
    if hi - lo < 1e-12:
        return "▁" * len(sample)
    levels = " ▁▂▃▄▅▆▇█"
    out: list[str] = []
    for value in sample:
        t = (value - lo) / (hi - lo)
        idx = min(len(levels) - 1, int(t * (len(levels) - 1)))
        out.append(levels[idx])
    return "".join(out)


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 16) -> str:
    if vmax - vmin < 1e-12:
        filled = width // 2
    else:
        filled = int((value - vmin) / (vmax - vmin) * width)
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


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
    grid_mse: float = 0.0
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


def make_curve_sample_grid(x1_values: list[float]) -> torch.Tensor:
    grid = torch.zeros(len(x1_values), DIMENSION)
    grid[:, 0] = torch.tensor(x1_values, dtype=torch.float32)
    return grid


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
    x_grid: torch.Tensor,
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
                    grad_norm = student.W.grad.pow(2).sum().item()
                    optimizer.step()

                    with torch.no_grad():
                        test_loss = loss_fn(student(x_test), y_test).item()
                        grid_mse = mse_vs_teacher(student(x_grid), x_grid)

                    epoch_idx = state.epoch
                    state.epoch += 1
                    state.train_loss = loss.item()
                    state.test_loss = test_loss
                    state.grid_mse = grid_mse
                    state.epochs_hist.append(epoch_idx)
                    state.train_loss_hist.append(state.train_loss)
                    state.test_loss_hist.append(state.test_loss)

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "epoch": epoch_idx,
                                "loss": state.train_loss,
                                "test_loss": state.test_loss,
                                "grad_norm": grad_norm,
                            }
                        )

        threading.Event().wait(0.01)


class TerminalDashboard:
    def __init__(
        self,
        student: CommitteeStudent,
        state: TrainingState,
        x_grid: torch.Tensor,
        y_teacher_grid: torch.Tensor,
        y_student_start: torch.Tensor,
        curve_sample_grid: torch.Tensor,
        y_teacher_sample: torch.Tensor,
        start_grid_mse: float,
    ) -> None:
        self.student = student
        self.state = state
        self.x_grid = x_grid
        self.y_teacher_grid = y_teacher_grid
        self.y_student_start = y_student_start
        self.curve_sample_grid = curve_sample_grid
        self.y_teacher_sample = y_teacher_sample
        self.start_grid_mse = start_grid_mse

    def run(self) -> None:
        curses.wrapper(self._main)

    def _handle_key(self, key: int) -> None:
        if key == ord(" "):
            with self.state.lock:
                self.state.paused = not self.state.paused
                self.state.status_message = (
                    "paused" if self.state.paused else "running"
                )
            return

        if key in (ord("q"), ord("Q")):
            self.state.stop = True
            self.state.stop_reason = "quit key"
            self.state.status_message = "quitting"
            return

        if key == curses.KEY_RESIZE:
            return

        ch = chr(key) if 32 <= key < 127 else None
        if ch is None:
            return

        preset = WEIGHT_PRESETS.get(ch)
        if preset is not None:
            unit_idx, target = preset
            self.state.injection_queue.put((unit_idx, target))
            self.state.status_message = f"queued unit {unit_idx}"

    def _put_line(self, stdscr, row: int, text: str) -> None:
        height, width = stdscr.getmaxyx()
        if row >= height:
            return
        stdscr.move(row, 0)
        stdscr.clrtoeol()
        stdscr.addstr(row, 0, text[: max(0, width - 1)])

    def _draw(self, stdscr) -> None:
        with self.state.lock:
            epoch = self.state.epoch
            train_loss = self.state.train_loss
            test_loss = self.state.test_loss
            grid_mse = self.state.grid_mse
            status = self.state.status_message
            paused = self.state.paused
            last_unit = self.state.last_injected_unit
            train_hist = list(self.state.train_loss_hist)
            test_hist = list(self.state.test_loss_hist)

        pause_tag = " [PAUSED]" if paused else ""
        row = 0
        self._put_line(stdscr, row, "Interactive committee training (terminal)")
        row += 1
        self._put_line(
            stdscr,
            row,
            f"epoch={epoch}  train={train_loss:.6f}  test={test_loss:.6f}  "
            f"grid_mse={grid_mse:.6f}  {status}{pause_tag}",
        )
        row += 1
        self._put_line(
            stdscr,
            row,
            "Keys: 1-9,0,a-f inject | space pause | q quit",
        )
        row += 1
        self._put_line(stdscr, row, "")
        row += 1

        self._put_line(stdscr, row, "Train loss (recent)")
        row += 1
        self._put_line(stdscr, row, sparkline(train_hist))
        row += 1
        self._put_line(stdscr, row, "Test loss (recent)")
        row += 1
        self._put_line(stdscr, row, sparkline(test_hist))
        row += 1
        self._put_line(stdscr, row, "")
        row += 1

        self.student.eval()
        with torch.no_grad():
            y_student_sample = self.student(self.curve_sample_grid)
            current_grid_mse = mse_vs_teacher(self.student(self.x_grid), self.x_grid)

        self._put_line(
            stdscr,
            row,
            f"Theory grid MSE: start={self.start_grid_mse:.6f}  now={current_grid_mse:.6f}",
        )
        row += 1
        header = "x1     " + " ".join(f"{x:>7.1f}" for x in CURVE_SAMPLE_X1)
        self._put_line(stdscr, row, header)
        row += 1
        teacher_vals = self.y_teacher_sample.tolist()
        student_vals = y_student_sample.tolist()
        self._put_line(
            stdscr,
            row,
            "teacher" + "".join(f"{v:>8.3f}" for v in teacher_vals),
        )
        row += 1
        self._put_line(
            stdscr,
            row,
            "student" + "".join(f"{v:>8.3f}" for v in student_vals),
        )
        row += 1
        self._put_line(stdscr, row, "")
        row += 1

        w_proj = project_onto_w_star(student_hidden_weights(self.student)).tolist()
        w_min = min(w_proj)
        w_max = max(w_proj)
        if abs(w_max - w_min) < 1e-12:
            w_min -= 0.5
            w_max += 0.5

        self._put_line(stdscr, row, "Weight projections w_p · w*  (* = last injected)")
        row += 1
        height, _ = stdscr.getmaxyx()
        for unit_idx, value in enumerate(w_proj):
            if row >= height - 1:
                break
            mark = "*" if unit_idx == last_unit else " "
            bar = ascii_bar(value, w_min, w_max)
            self._put_line(
                stdscr,
                row,
                f"u{unit_idx:02d} [{bar}] {value:+.4f}{mark}",
            )
            row += 1

        stdscr.refresh()

    def _main(self, stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(False)
        stdscr.timeout(UI_REFRESH_MS)

        while not self.state.stop:
            key = stdscr.getch()
            if key != -1:
                self._handle_key(key)
            self._draw(stdscr)


def init_wandb(loaded_from: Path) -> None:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode="online" if USE_WANDB else "disabled",
        config={
            "task": "erf_combo_interactive",
            "mode": "interactive",
            "dimension": DIMENSION,
            "N": N,
            "P": P,
            "student": "(1/sqrt(N)) sum_p erf(w_p·x)",
            "lr": LR,
            "optimizer": OPTIMIZER,
            "max_epochs": MAX_EPOCHS,
            "epochs_per_ui_tick": EPOCHS_PER_UI_TICK,
            "n_test_used": N_TEST_USED,
            "init_seed": INIT_SEED,
            "init_mean": INIT_MEAN,
            "init_var": INIT_VAR,
            "init_mode": INIT_MODE,
            "init_w_manual": INIT_W_MANUAL,
            "init_manual_noise_var": INIT_MANUAL_NOISE_VAR,
            "checkpoint": str(CHECKPOINT),
            "loaded_from": str(loaded_from),
            "weight_presets": WEIGHT_PRESETS,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("loss", step_metric="epoch")
    wandb.define_metric("grad_norm", step_metric="epoch")


def finalize_wandb_run(
    student: CommitteeStudent,
    x_test: torch.Tensor,
    y_student_init: torch.Tensor,
    w_init: torch.Tensor,
    state: TrainingState,
    start_grid_mse: float,
) -> None:
    if wandb.run is None:
        return

    INTERACTIVE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_path = save_loss_loglog_plot(
        state.epochs_hist,
        state.train_loss_hist,
        state.test_loss_hist,
        LOSS_LOGLOG_PLOT_PATH,
    )
    print(f"Saved log-log loss plot to {plot_path}")

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
    scatter_path = save_pred_vs_theory_plot(
        y_test, y_student, PRED_VS_THEORY_PLOT_PATH
    )

    w_trained = project_onto_w_star(student_hidden_weights(student))
    theory_metrics = {
        "trained_test_mse": trained_test_mse,
        "trained_grid_mse": trained_grid_mse,
        "trained_w_mean": w_trained.mean().item(),
        "trained_w_std": w_trained.std().item(),
    }

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

    w_trained_weights = student_hidden_weights(student)
    weight_plot_path = save_weight_distribution_plot(
        w_init, w_trained_weights, WEIGHT_DISTRIBUTION_PLOT_PATH
    )
    print(f"Saved weight distribution plot to {weight_plot_path}")

    w_init_proj = project_onto_w_star(w_init)
    w_trained_proj = project_onto_w_star(w_trained_weights)
    print(
        f"  session-start w·w*: mean={w_init_proj.mean().item():.4f} "
        f"std={w_init_proj.std().item():.4f}"
    )
    print(
        f"  trained w·w*: mean={w_trained_proj.mean().item():.4f} "
        f"std={w_trained_proj.std().item():.4f}"
    )

    wandb.run.summary.update(
        {
            "init_grid_mse": start_grid_mse,
            "interactive_epochs": state.epoch,
            "stop_reason": state.stop_reason,
            **theory_metrics,
        }
    )
    wandb.log(
        {
            "weights_distribution": wandb.Image(str(weight_plot_path)),
            "pred_vs_theory": wandb.Image(str(scatter_path)),
            "theory_curves": wandb.Image(str(curves_path)),
        }
    )
    wandb.finish()


def main() -> None:
    validate_weight_presets(WEIGHT_PRESETS, N, DIMENSION)
    print_key_map(WEIGHT_PRESETS)

    x_train, y_train, x_test, y_test = load_training_data()
    student, loaded_from = load_student(CHECKPOINT)
    print(f"Loaded checkpoint: {loaded_from}")

    x_grid = make_theory_grid()
    y_teacher_grid = teacher_erf_combo(x_grid)
    curve_sample_grid = make_curve_sample_grid(CURVE_SAMPLE_X1)
    y_teacher_sample = teacher_erf_combo(curve_sample_grid)

    student.eval()
    with torch.no_grad():
        y_student_start = student(x_grid).clone()
        init_test_mse = mse_vs_teacher(student(x_test), x_test)
        start_grid_mse = mse_vs_teacher(y_student_start, x_grid)
    w_init = student_hidden_weights(student).clone()
    print(f"checkpoint test MSE={init_test_mse:.6f}")
    if MAX_EPOCHS is None:
        print("Training runs until you press q (no epoch limit).")
    else:
        print(f"Training stops after MAX_EPOCHS={MAX_EPOCHS} or when you press q.")
    print("Starting terminal UI...")

    init_wandb(loaded_from)

    optimizer = make_optimizer(student.parameters(), LR, OPTIMIZER)
    loss_fn = nn.MSELoss()
    state = TrainingState(paused=PAUSE_ON_START)

    train_thread = threading.Thread(
        target=training_loop,
        args=(
            student,
            optimizer,
            loss_fn,
            x_train,
            y_train,
            x_test,
            y_test,
            x_grid,
            state,
        ),
        daemon=True,
    )
    train_thread.start()

    dashboard = TerminalDashboard(
        student,
        state,
        x_grid,
        y_teacher_grid,
        y_student_start,
        curve_sample_grid,
        y_teacher_sample,
        start_grid_mse,
    )
    dashboard.run()

    state.stop = True
    train_thread.join(timeout=2.0)
    reason = state.stop_reason or "terminal UI exited"
    print(f"Stopped at epoch {state.epoch} ({reason})")

    finalize_wandb_run(
        student,
        x_test,
        y_student_start,
        w_init,
        state,
        start_grid_mse,
    )


if __name__ == "__main__":
    main()
