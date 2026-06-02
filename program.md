# committee-autoresearch

Autonomous research loop for the non-linear committee machine learning a Hermite teacher.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — repository context and quick start.
   - `theory_diffusion_scaling.tex` — analytical background on overlaps, plateau, and loss scaling.
   - `prepare.py` — fixed constants, data prep, evaluation harness. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/committee-autoresearch/` contains cached tensors. If not, run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs for a **fixed wall-clock training budget of 10 minutes** (`TRAINING_SECONDS = 600` in `prepare.py`). Launch with:

```bash
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` only — architecture, optimizer, hyperparameters, training loop, init, readout, weight constraints, Langevin noise, LR schedules, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only: fixed teacher, train/val splits, evaluation metrics, time budget, and data constants.
- Install new packages or add dependencies. Use only what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is ground truth.

## Goal

Unlike vanilla autoresearch (minimize val_bpb), this project targets **grokking / phase-transition dynamics**:

- Train loss drops early (memorization of the small training set).
- Validation MSE stays high on a plateau, then drops when student overlaps align with the teacher direction `w_*`.
- This matches the plateau story in `theory_diffusion_scaling.tex` (`m_i ~ O(1/√d)`, drift `Δm_i ~ O(1/d)`).

**Hard constraint for keeping changes:** `val_mse` must improve (lower is better). Generalization matters — we are not optimizing train loss alone.

**Secondary research signals** (log in description, use to guide ideas):
- `max_overlap` — higher means stronger alignment with teacher.
- `grok_epoch` — step when val_mse first dropped below 50% of initial (after warmup).
- `train_val_gap` — large gap early then convergence suggests grokking.
- `plateau_length` — longer plateau before escape is interesting dynamics.

**Simplicity criterion**: All else being equal, simpler is better. A small val_mse improvement that adds ugly complexity is probably not worth it. Removing code and getting equal or better results is a great outcome.

**The first run**: Always establish baseline by running `train.py` as-is.

## Output format

When training finishes, the script prints:

```
---
val_mse:          0.264100
train_mse:        0.001200
train_val_gap:    -0.262900
max_overlap:      0.842100
mean_overlap:     0.123456
grok_epoch:       8420
plateau_length:   120
training_seconds: 600.2
total_seconds:    612.5
num_steps:        48000
```

Extract key metrics:

```bash
grep "^val_mse:\|^max_overlap:\|^grok_epoch:" run.log
```

After each run, `prepare.py` automatically saves `val_loss_plot.png`, uploads metrics and the plot to **Weights & Biases**, and sends the plot to **Telegram** (requires `WANDB_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` in the environment). The agent does not need to configure this.

## Logging results

When an experiment is done, append to `results.tsv` (tab-separated, NOT comma-separated).

Header:

```
commit	val_mse	train_mse	max_overlap	grok_epoch	memory_gb	status	description
```

Columns:
1. git commit hash (short, 7 chars)
2. `val_mse` achieved — use `9.999999` for crashes
3. `train_mse` at end
4. `max_overlap` at end
5. `grok_epoch` (0 if never grokked)
6. peak memory in GB — use `0.0` for crashes (estimate from system if needed)
7. status: `keep`, `discard`, or `crash`
8. short description of what this experiment tried

Example:

```
commit	val_mse	train_mse	max_overlap	grok_epoch	memory_gb	status	description
a1b2c3d	2.450000	2.440000	0.050000	0	0.5	keep	baseline
b2c3d4e	1.820000	0.001200	0.620000	12000	0.5	keep	add weight decay 1e-4
c3d4e5f	2.600000	2.590000	0.040000	0	0.5	discard	switch to tanh activation
d4e5f6g	9.999999	9.999999	0.000000	0	0.0	crash	LR 10.0 diverged
```

Do not commit `results.tsv` — leave it untracked.

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/jun2`).

LOOP FOREVER:

1. Look at git state: current branch/commit.
2. Tune `train.py` with an experimental idea.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_mse:\|^max_overlap:\|^grok_epoch:" run.log`
6. If grep is empty, run crashed. `tail -n 50 run.log` for stack trace. Fix dumb bugs and retry; otherwise log `crash` and move on.
7. Record in `results.tsv`.
8. If `val_mse` improved (lower), keep the commit (branch advances).
9. If `val_mse` equal or worse, `git reset` back to previous best.

**Timeout**: Each experiment should take ~10 minutes (+ eval overhead). If a run exceeds 15 minutes, kill it and treat as failure.

**Crashes**: Fix typos and retry. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the loop begins, do not pause to ask the human to continue. Run until manually interrupted. If stuck, re-read `theory_diffusion_scaling.tex`, try Langevin noise, weight decay, LR schedules, Frobenius re-normalization each step, different readouts, or combine near-misses.

## Research angles to explore

- **Weight decay** vs none — affects plateau length and overlap dynamics.
- **Langevin noise** — theory doc discusses diffusion; add Gaussian noise to gradients with temperature T.
- **Learning rate** — too high diverges, too low extends plateau indefinitely within budget.
- **Frobenius constraint** — notebook normalizes W only at init; try re-normalizing each step vs free growth.
- **Optimizer** — SGD with momentum vs Adam; different betas.
- **Readout** — weighted sum vs mean; learnable readout weights.
- **Activation** — erf is fixed in problem but agent could try soft approximations (stay smooth).
- **Init scale** — smaller init may lengthen plateau; larger may escape faster but destabilize.

Remember: `dimension`, `TRAIN_SAMPLES`, and `VAL_SAMPLES` are in `prepare.py`. To change problem scale, the human must retune `prepare.py` between research campaigns — do not modify it yourself.
