"""Multicore CPU sweep over alpha for the erf committee training harness."""

from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from committee_network import teacher_erf_combo
from erf_combo_commette_machine import (
    LOAD_FROM,
    compute_sample_counts,
    ensure_data_cache,
    resolve_ensemble_seeds,
    train_one_seed,
)

ALPHA_START = 0.05
ALPHA_STOP = 1.0
ALPHA_STEP = 0.01
MAX_WORKERS: int | None = 20
LOG_EVERY = 100
WANDB_GROUP = "alpha_sweep"
USE_WANDB = True


def build_alpha_list(
    start: float = ALPHA_START,
    stop: float = ALPHA_STOP,
    step: float = ALPHA_STEP,
) -> list[float]:
    return [round(a, 2) for a in np.arange(start, stop + 1e-9, step)]


def _train_alpha(alpha: float, log_every: int, use_wandb: bool) -> float:
    """Train all ensemble seeds for one alpha value. Returns alpha on success."""
    torch.set_num_threads(1)

    import erf_combo_commette_machine as harness

    harness.USE_WANDB = use_wandb

    p, n_test = compute_sample_counts(alpha)
    x_train, x_test = ensure_data_cache()
    x_train = x_train[:p]
    x_test = x_test[:n_test]
    y_train = teacher_erf_combo(x_train)
    y_test = teacher_erf_combo(x_test)

    seeds = resolve_ensemble_seeds()
    load_from = LOAD_FROM if len(seeds) == 1 else None
    print(
        f"[alpha {alpha:.2f}] P={p}, n_test={n_test}, "
        f"{len(seeds)} seed(s), load_from={load_from!r}"
    )

    for seed in seeds:
        train_one_seed(
            seed,
            x_train,
            y_train,
            x_test,
            y_test,
            group=WANDB_GROUP,
            ensemble_size=len(seeds),
            load_from=load_from,
            alpha=alpha,
            log_every=log_every,
        )

    return alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep alpha in parallel on CPU.")
    parser.add_argument("--alpha-start", type=float, default=ALPHA_START)
    parser.add_argument("--alpha-stop", type=float, default=ALPHA_STOP)
    parser.add_argument("--alpha-step", type=float, default=ALPHA_STEP)
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

    alphas = build_alpha_list(args.alpha_start, args.alpha_stop, args.alpha_step)
    workers = args.workers or os.cpu_count() or 1

    print(
        f"Alpha sweep: {len(alphas)} values from {alphas[0]:.2f} to {alphas[-1]:.2f} "
        f"(step {args.alpha_step}), workers={workers}, log_every={log_every}"
    )

    ensure_data_cache()

    started = time.monotonic()
    completed: list[float] = []
    failed: dict[float, str] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_train_alpha, alpha, log_every, use_wandb): alpha
            for alpha in alphas
        }
        for future in as_completed(futures):
            alpha = futures[future]
            try:
                done_alpha = future.result()
                completed.append(done_alpha)
                print(f"[done] alpha={done_alpha:.2f} ({len(completed)}/{len(alphas)})")
            except Exception:
                failed[alpha] = traceback.format_exc()
                print(f"[failed] alpha={alpha:.2f}\n{failed[alpha]}")

    elapsed = time.monotonic() - started
    print(
        f"\nSweep finished in {elapsed:.1f}s: "
        f"{len(completed)} completed, {len(failed)} failed"
    )
    if failed:
        print("Failed alphas:")
        for alpha in sorted(failed):
            print(f"  alpha={alpha:.2f}")


if __name__ == "__main__":
    main()
