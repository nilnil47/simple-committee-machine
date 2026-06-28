import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from prepare import _ensure_wandb_auth

GRAD_NORM_PLOT_PATH = Path("groking_grad_norm.png")

# 1. Define a minimalist 2-Layer MLP Architecture
class GrokkingMLP(nn.Module):
    def __init__(self, p, d_model=128):
        super().__init__()
        # Learned embeddings for the two input operands
        self.embed = nn.Embedding(p, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, p)
        )
        
    def forward(self, x):
        # x shape: [batch, 2] -> tokens representing operand 'a' and 'b'
        emb_a = self.embed(x[:, 0])
        emb_b = self.embed(x[:, 1])
        # Concatenate the operand representations
        hidden = torch.cat([emb_a, emb_b], dim=1)
        return self.mlp(hidden)


def save_grad_norm_plot(
    epochs: list[int],
    grad_norms: list[float],
    train_losses: list[float],
    path: Path,
) -> Path:
    fig, ax_grad = plt.subplots(figsize=(8, 5))
    ax_loss = ax_grad.twinx()
    ax_grad.plot(epochs, grad_norms, color="C0", label="grad norm", linewidth=1.5)
    ax_loss.plot(epochs, train_losses, color="C1", label="train loss", linewidth=1.5, alpha=0.7)
    ax_grad.set_xlabel("epoch")
    ax_grad.set_ylabel("grad norm (squared L2)")
    ax_loss.set_ylabel("train loss")
    lines = ax_grad.get_lines() + ax_loss.get_lines()
    ax_grad.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax_grad.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# 2. Build the Modular Addition Dataset (p = 97)
p = 97
X = torch.tensor([[a, b] for a in range(p) for b in range(p)], dtype=torch.long)
Y = torch.tensor([(a + b) % p for a in range(p) for b in range(p)], dtype=torch.long)

# Shuffle and split: 40% Train (sparse memorization), 60% Validation (held-out)
torch.manual_seed(42) # Ensuring a reproducible phase transition trajectory
indices = torch.randperm(len(X))
split = int(0.4 * len(X))
train_idx, val_idx = indices[:split], indices[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# 3. Initialize Training Components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GrokkingMLP(p=p).to(device)

# Strong weight decay accelerates the contraction out of the memorization basin
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
criterion = nn.CrossEntropyLoss()

# 4. Optimization Loop
epochs = 6000
log_frequency = 100
train_split = 0.4

_ensure_wandb_auth()
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "grokking-arithmetic"),
    config={
        "p": p,
        "d_model": 128,
        "lr": 1e-3,
        "weight_decay": 1.0,
        "epochs": epochs,
        "train_split": train_split,
        "log_frequency": log_frequency,
        "grad_norm": "squared_l2",
        "seed": 42,
        "device": str(device),
    },
)
wandb.define_metric("epoch")
wandb.define_metric("train_loss", step_metric="epoch")
wandb.define_metric("grad_norm", step_metric="epoch")
wandb.define_metric("train_acc", step_metric="epoch")
wandb.define_metric("val_acc", step_metric="epoch")

print(f"Training on {device}... Watch for the validation accuracy phase transition.\n")
print(f"{'Epoch':<8} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Acc':<8}")
print("-" * 50)

epochs_hist: list[int] = []
grad_norm_hist: list[float] = []
train_loss_hist: list[float] = []

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Compute full-batch train metrics
    outputs = model(X_train.to(device))
    loss = criterion(outputs, Y_train.to(device))
    loss.backward()
    grad_norm = sum(
        p.grad.pow(2).sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    optimizer.step()

    train_loss = loss.item()
    epochs_hist.append(epoch)
    grad_norm_hist.append(grad_norm)
    train_loss_hist.append(train_loss)
    wandb.log(
        {
            "epoch": epoch,
            "grad_norm": grad_norm,
            "train_loss": train_loss,
        }
    )
    
    # Evaluate at regular intervals to observe the plateau and the sudden click
    if epoch % log_frequency == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            # Check training accuracy
            train_preds = outputs.argmax(dim=-1)
            train_acc = (train_preds == Y_train.to(device)).float().mean().item() * 100
            
            # Check validation accuracy
            val_outputs = model(X_val.to(device))
            val_preds = val_outputs.argmax(dim=-1)
            val_acc = (val_preds == Y_val.to(device)).float().mean().item() * 100
            
        print(f"{epoch:<8} | {train_loss:<10.4f} | {train_acc:<8.1f}% | {val_acc:<.1f}%")
        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        # Early stopping helper if you just want to see it cross the finish line
        if val_acc > 99.5:
            print("-" * 50)
            print(f"🎉 Grokking achieved at epoch {epoch}!")
            wandb.run.summary.update(
                {"grokking_epoch": epoch, "final_val_acc": val_acc}
            )
            break

plot_path = save_grad_norm_plot(
    epochs_hist,
    grad_norm_hist,
    train_loss_hist,
    GRAD_NORM_PLOT_PATH,
)
print(f"Saved grad norm plot to {plot_path}")
wandb.log({"grad_norm_plot": wandb.Image(str(plot_path))})
wandb.finish()