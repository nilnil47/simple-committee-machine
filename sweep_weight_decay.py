"""Multicore CPU sweep over weight decay for GD (SGD) training."""

from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from committee_network import teacher_erf_combo
from erf_combo_commette_machine import (
    ALPHA,
    CACHE,
    LOAD_FROM,
    P,
    N_TEST_USED,
    ensure_data_cache,
    resolve_ensemble_seeds,
    train_one_seed,
)

WD_START = 0.0
WD_STOP = 0.2
WD_STEP = 0.005
MAX_WORKERS: int | None = 20
LOG_EVERY = 100
WANDB_GROUP = "weight_decay_sweep"
USE_WANDB = False


def build_wd_list(
    start: float = WD_START,
    stop: float = WD_STOP,
    step: float = WD_STEP,
) -> list[float]:
    return [round(float(wd), 6) for wd in np.arange(start, stop + 1e-12, step)]


def _train_wd(weight_decay: float, log_every: int, use_wandb: bool) -> tuple[float, float]:
    """Train all ensemble seeds for one weight-decay value.

    Returns (weight_decay, mean final test loss) on success.
    """
    torch.set_num_threads(1)

    import erf_combo_commette_machine as harness

    harness.USE_WANDB = use_wandb
    harness.WEIGHT_DECAY = weight_decay

    x_train, x_test = ensure_data_cache()
    x_train = x_train[:P]
    x_test = x_test[:N_TEST_USED]
    y_train = teacher_erf_combo(x_train)
    y_test = teacher_erf_combo(x_test)

    seeds = resolve_ensemble_seeds()
    load_from = LOAD_FROM if len(seeds) == 1 else None
    print(
        f"[wd {weight_decay:.4g}] alpha={ALPHA:.2f}, P={P}, n_test={N_TEST_USED}, "
        f"optimizer={harness.OPTIMIZER}, "
        f"{len(seeds)} seed(s), load_from={load_from!r}"
    )

    final_test_losses: list[float] = []
    for seed in seeds:
        final_test_loss = train_one_seed(
            seed,
            x_train,
            y_train,
            x_test,
            y_test,
            group=WANDB_GROUP,
            ensemble_size=len(seeds),
            load_from=load_from,
            weight_decay=weight_decay,
            log_every=log_every,
        )
        final_test_losses.append(final_test_loss)

    mean_final_test_loss = float(np.mean(final_test_losses))
    return weight_decay, mean_final_test_loss


def save_final_test_loss_vs_wd_plot(
    results: list[tuple[float, float]],
    path: Path,
) -> Path:
    """Plot final test loss as a function of weight decay."""
    results = sorted(results, key=lambda item: item[0])
    wds = [wd for wd, _ in results]
    losses = [loss for _, loss in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wds, losses, marker="o", linewidth=1.5)
    ax.set_xlabel("weight decay")
    ax.set_ylabel("final test loss")
    ax.set_title(f"Final test loss vs weight decay (alpha={ALPHA:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep GD weight decay in parallel on CPU."
    )
    parser.add_argument("--wd-start", type=float, default=WD_START)
    parser.add_argument("--wd-stop", type=float, default=WD_STOP)
    parser.add_argument("--wd-step", type=float, default=WD_STEP)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Max parallel workers (default: CPU count).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY,
        help="W&B logging interval in epochs.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload for this sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_every = args.log_every
    use_wandb = not args.no_wandb

    wd_values = build_wd_list(args.wd_start, args.wd_stop, args.wd_step)
    workers = args.workers or os.cpu_count() or 1

    print(
        f"Weight-decay sweep: {len(wd_values)} values from {wd_values[0]:.4g} "
        f"to {wd_values[-1]:.4g} (step {args.wd_step}), alpha={ALPHA:.2f}, "
        f"workers={workers}, log_every={log_every}"
    )

    ensure_data_cache()

    started = time.monotonic()
    results: list[tuple[float, float]] = []
    failed: dict[float, str] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_train_wd, wd, log_every, use_wandb): wd
            for wd in wd_values
        }
        for future in as_completed(futures):
            wd = futures[future]
            try:
                done_wd, final_test_loss = future.result()
                results.append((done_wd, final_test_loss))
                print(
                    f"[done] wd={done_wd:.4g} final_test_loss={final_test_loss:.6f} "
                    f"({len(results)}/{len(wd_values)})"
                )
            except Exception:
                failed[wd] = traceback.format_exc()
                print(f"[failed] wd={wd:.4g}\n{failed[wd]}")

    elapsed = time.monotonic() - started
    print(
        f"\nSweep finished in {elapsed:.1f}s: "
        f"{len(results)} completed, {len(failed)} failed"
    )
    if failed:
        print("Failed weight-decay values:")
        for wd in sorted(failed):
            print(f"  wd={wd:.4g}")

    if results:
        plot_path = save_final_test_loss_vs_wd_plot(
            results,
            CACHE / "plots" / "final_test_loss_vs_weight_decay.png",
        )
        print(f"Saved final test loss vs weight decay plot to {plot_path}")


if __name__ == "__main__":
    main()
