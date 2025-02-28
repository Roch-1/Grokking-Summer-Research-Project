import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global configuration
global_cfg = {
    "mod": 113,
    "val_frac": 0.2,
    "train_subset_frac": 0.4,
    "epochs": 2000,
}

# Create modular addition dataset
def create_dataset(mod):
    pairs = torch.cartesian_prod(torch.arange(mod), torch.arange(mod))
    labels = (pairs[:, 0] + pairs[:, 1]) % mod
    return pairs, labels

pairs, labels = create_dataset(global_cfg["mod"])
num_samples = pairs.size(0)
val_size = int(num_samples * global_cfg["val_frac"])
train_size = num_samples - val_size

# Split into train and validation sets
perm = torch.randperm(num_samples)
train_full_pairs = pairs[perm[:train_size]]
train_full_labels = labels[perm[:train_size]]
val_pairs = pairs[perm[train_size:]]
val_labels = labels[perm[train_size:]]

# Take a subset of training data
train_subset_size = int(train_size * global_cfg["train_subset_frac"])
train_pairs = train_full_pairs[:train_subset_size]
train_labels = train_full_labels[:train_subset_size]

# Define the Transformer model
class GrokTransformer(nn.Module):
    def __init__(self, mod, d_model=128, nhead=4, num_layers=2, init_orthogonal=False):
        super(GrokTransformer, self).__init__()
        self.mod = mod
        self.d_model = d_model
        self.token_emb = nn.Embedding(mod + 2, d_model)  # mod numbers + '+' and '='
        self.pos_emb = nn.Parameter(torch.zeros(1, 4, d_model))  # seq_len = 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, mod)
        if init_orthogonal:
            nn.init.orthogonal_(self.fc.weight)

    def forward(self, a, b):
        batch_size = a.size(0)
        seq = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
        seq[:, 0] = a  # First number
        seq[:, 1] = self.mod  # '+' token
        seq[:, 2] = b  # Second number
        seq[:, 3] = self.mod + 1  # '=' token
        x = self.token_emb(seq) + self.pos_emb
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # Predict based on '=' token
        return out

# StableMax functions
def stablemax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def stable_cross_entropy(logits, targets, use_stablemax):
    if use_stablemax:
        probs = stablemax(logits)
        return -torch.mean(torch.log(probs[range(len(targets)), targets] + 1e-10))
    else:
        return F.cross_entropy(logits, targets)

def train_model(model, train_pairs, train_labels, val_pairs, val_labels, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    num_train = train_pairs.size(0)
    batch_size = cfg["batch_size"]
    history = {"train_acc": [], "val_acc": []}

    # Grokfast buffers
    if cfg.get("use_ma", False):
        ma_buffer = {name: deque(maxlen=cfg["ma_window"]) for name, param in model.named_parameters()}
    if cfg.get("use_ema", False):
        ema_buffer = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for epoch in range(global_cfg["epochs"]):
        model.train()
        perm = torch.randperm(num_train)
        total_loss = 0
        for i in range(0, num_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            a_batch = train_pairs[batch_idx, 0].to(device)
            b_batch = train_pairs[batch_idx, 1].to(device)
            targets_batch = train_labels[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(a_batch, b_batch)
            loss = stable_cross_entropy(logits, targets_batch, cfg.get("use_stablemax", False))
            loss.backward()

            # Apply method-specific modifications
            if cfg.get("use_ma", False) or cfg.get("use_ema", False) or cfg.get("use_perpgrad", False):
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    grad = param.grad.clone()
                    if cfg.get("use_ma", False):
                        ma_buffer[name].append(grad)
                        if len(ma_buffer[name]) == cfg["ma_window"]:
                            avg_grad = torch.stack(list(ma_buffer[name])).mean(dim=0)
                            param.grad += cfg["ma_lambda"] * avg_grad
                    if cfg.get("use_ema", False):
                        ema_buffer[name] = cfg["alpha"] * ema_buffer[name] + (1 - cfg["alpha"]) * grad
                        param.grad += cfg["lamb"] * ema_buffer[name]
                    if cfg.get("use_perpgrad", False) and "fc.weight" in name:
                        w = param.data
                        g = param.grad
                        proj = (torch.sum(g * w) / (torch.sum(w * w) + 1e-10)) * w
                        param.grad = g - proj

            optimizer.step()
            total_loss += loss.item()

        # Evaluate after each epoch
        model.eval()
        with torch.no_grad():
            train_logits = model(train_pairs[:, 0].to(device), train_pairs[:, 1].to(device))
            train_acc = (train_logits.argmax(dim=-1).cpu() == train_labels).float().mean().item()
            val_logits = model(val_pairs[:, 0].to(device), val_pairs[:, 1].to(device))
            val_acc = (val_logits.argmax(dim=-1).cpu() == val_labels).float().mean().item()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Method: {cfg['name']}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Check stopping condition: stop if both train and val accuracies are >= 99%
        if train_acc >= 0.99 and val_acc >= 0.99:
            print(f"Stopping early at epoch {epoch} as both train and val accuracy reached 99%!")
            break

    return history


# Method configurations
methods_cfg = {
    "Baseline": {"name": "Baseline", "lr": 1e-3, "wd": 0.0, "batch_size": 128},
    "Grokfast_MA": {"name": "Grokfast_MA", "lr": 1e-3, "wd": 0.01, "batch_size": 128,
                    "use_ma": True, "ma_window": 100, "ma_lambda": 5.0},
    "Grokfast_EMA": {"name": "Grokfast_EMA", "lr": 1e-3, "wd": 0.005, "batch_size": 128,
                     "use_ema": True, "alpha": 0.98, "lamb": 2.0},
    "StableMax_Regular": {"name": "StableMax_Regular", "lr": 1e-3, "wd": 1e-3, "batch_size": 128,
                          "use_stablemax": True},
    "StableMax_Orthogonal": {"name": "StableMax_Orthogonal", "lr": 1e-3, "wd": 1e-3, "batch_size": 128,
                             "use_stablemax": True, "init_orthogonal": True},
    "StableMax_Perpendicular": {"name": "StableMax_Perpendicular", "lr": 1e-3, "wd": 1e-3, "batch_size": 128,
                                "use_stablemax": True, "use_perpgrad": True}
}

# Run experiments
results = {}
for method_name, cfg in methods_cfg.items():
    print(f"\nRunning {method_name}")
    model = GrokTransformer(
        mod=global_cfg["mod"],
        d_model=128,
        nhead=4,
        num_layers=2,
        init_orthogonal=cfg.get("init_orthogonal", False)
    ).to(device)
    history = train_model(model, train_pairs, train_labels, val_pairs, val_labels, cfg)
    results[method_name] = history

# Plot results
plt.figure(figsize=(12, 6))
for method_name, history in results.items():
    plt.plot(history["val_acc"], label=f"{method_name} (Val Acc)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.show()
