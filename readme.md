# Committee Machine Autoresearch

Autonomous research loop for a non-linear committee machine learning a Hermite polynomial teacher. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Quick start

**Requirements:** Python 3.10+ with PyTorch.

```bash
pip install -r requirements.txt
python prepare.py
python train.py > run.log 2>&1
grep "^val_mse:" run.log
```

After each run, a validation loss plot is saved to `val_loss_plot.png`, uploaded to Weights & Biases, and sent to Telegram (if configured).

## W&B and Telegram setup

Copy the template and add your credentials:

```bash
cp .env.example .env
# edit .env with your keys
```

```bash
# .env contents:
WANDB_API_KEY=your-key              # or use `wandb login` instead
WANDB_PROJECT=committee-autoresearch

TELEGRAM_BOT_TOKEN=123456:ABC...    # from @BotFather
TELEGRAM_CHAT_ID=your-chat-id       # from @userinfobot
```

`prepare.py` loads `.env` automatically when you run `train.py`. The file is gitignored — never commit it.

Test integrations without a full training run:

```bash
python test_wandb.py
python test_telegram.py
```

If `.env` has `WANDB_API_KEY=` left blank, either remove that line or run `wandb login` — an empty key overrides saved login credentials.

## How it works

| File | Role |
|------|------|
| **`prepare.py`** | Fixed constants, data prep, evaluation, W&B/Telegram reporting |
| **`train.py`** | Model and training loop — **agent edits this only** |
| **`program.md`** | Instructions for the AI agent |

## Running the agent

```
Read program.md and kick off a new autoresearch experiment. Do the setup first.
```
