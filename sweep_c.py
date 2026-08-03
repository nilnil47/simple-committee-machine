"""Multicore CPU sweep over C for manual-init noise variance (C / DIMENSION)."""

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
    apply_manual_noise_c,
    ensure_data_cache,
    resolve_ensemble_seeds,
    train_one_seed,
)

C_START = 0.1
C_STOP = 0.5
C_STEP = 0.02
MAX_WORKERS: int | None = 20
LOG_EVERY = 100
WANDB_GROUP = "c_sweep"
USE_WANDB = True


def build_c_list(
    start: float = C_START,
    stop: float = C_STOP,
    step: float = C_STEP,
) -> list[float]:
    return [round(c, 2) for c in np.arange(start, stop + 1e-9, step)]


def _train_c(c: float, log_every: int, use_wandb: bool) -> tuple[float, float]:
    """Train all ensemble seeds for one C value.

    Returns (C, mean final test loss) on success.
    """
    torch.set_num_threads(1)

    import erf_combo_commette_machine as harness

    harness.USE_WANDB = use_wandb
    apply_manual_noise_c(c)

    x_train, x_test = ensure_data_cache()
    x_train = x_train[:P]
    x_test = x_test[:N_TEST_USED]
    y_train = teacher_erf_combo(x_train)
    y_test = teacher_erf_combo(x_test)

    seeds = resolve_ensemble_seeds()
    load_from = LOAD_FROM if len(seeds) == 1 else None
    noise_var = c / harness.DIMENSION
    print(
        f"[C {c:.2f}] alpha={ALPHA:.2f}, P={P}, n_test={N_TEST_USED}, "
        f"init_manual_noise_var={noise_var:.6f}, "
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
            c=c,
            log_every=log_every,
        )
        final_test_losses.append(final_test_loss)

    mean_final_test_loss = float(np.mean(final_test_losses))
    return c, mean_final_test_loss


def save_final_test_loss_vs_c_plot(
    results: list[tuple[float, float]],
    path: Path,
) -> Path:
    """Plot final test loss as a function of C."""
    results = sorted(results, key=lambda item: item[0])
    cs = [c for c, _ in results]
    losses = [loss for _, loss in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, losses, marker="o", linewidth=1.5)
    ax.set_xlabel("C")
    ax.set_ylabel("final test loss")
    ax.set_title(f"Final test loss vs C (alpha={ALPHA:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep manual-init noise scale C in parallel on CPU."
    )
    parser.add_argument("--c-start", type=float, default=C_START)
    parser.add_argument("--c-stop", type=float, default=C_STOP)
    parser.add_argument("--c-step", type=float, default=C_STEP)
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

    c_values = build_c_list(args.c_start, args.c_stop, args.c_step)
    workers = args.workers or os.cpu_count() or 1

    print(
        f"C sweep: {len(c_values)} values from {c_values[0]:.2f} to {c_values[-1]:.2f} "
        f"(step {args.c_step}), alpha={ALPHA:.2f}, workers={workers}, "
        f"log_every={log_every}"
    )

    ensure_data_cache()

    started = time.monotonic()
    results: list[tuple[float, float]] = []
    failed: dict[float, str] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_train_c, c, log_every, use_wandb): c for c in c_values
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                done_c, final_test_loss = future.result()
                results.append((done_c, final_test_loss))
                print(
                    f"[done] C={done_c:.2f} final_test_loss={final_test_loss:.6f} "
                    f"({len(results)}/{len(c_values)})"
                )
            except Exception:
                failed[c] = traceback.format_exc()
                print(f"[failed] C={c:.2f}\n{failed[c]}")

    elapsed = time.monotonic() - started
    print(
        f"\nSweep finished in {elapsed:.1f}s: "
        f"{len(results)} completed, {len(failed)} failed"
    )
    if failed:
        print("Failed C values:")
        for c in sorted(failed):
            print(f"  C={c:.2f}")

    if results:
        plot_path = save_final_test_loss_vs_c_plot(
            results,
            CACHE / "plots" / "final_test_loss_vs_c.png",
        )
        print(f"Saved final test loss vs C plot to {plot_path}")


if __name__ == "__main__":
    main()
