import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd


# =========================================================
# 0. Reproducibility & device
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =========================================================
# 1. Load & preprocess Adult dataset
# =========================================================
print("Loading Adult dataset from OpenML...")
adult = fetch_openml("adult", version=2, as_frame=True)
X_df: pd.DataFrame = adult.data
y_series: pd.Series = adult.target

# Convert target to 0/1
# Original labels are '>50K', '<=50K' (or similar)
y = (y_series.astype(str).str.contains(">50K")).astype(np.float32).values

# One-hot encode categoricals (simple but robust)
print("One-hot encoding categorical features...")
X_df = pd.get_dummies(X_df, drop_first=False)
X = X_df.values.astype(np.float32)
feature_dim = X.shape[1]
print(f"Feature dim after one-hot: {feature_dim}")

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standardize all features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Convert to tensors
X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
X_val_t   = torch.tensor(X_val,   device=device)
y_val_t   = torch.tensor(y_val,   device=device)
X_test_t  = torch.tensor(X_test,  device=device)
y_test_t  = torch.tensor(y_test,  device=device)

num_classes = 2
input_dim   = feature_dim


# =========================================================
# 2. Classifier MLP (for evaluation)
# =========================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def evaluate_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb).squeeze()
            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(trues, pred_labels)
    auc = roc_auc_score(trues, preds)
    return acc, auc


def train_classifier(X_train, y_train, X_val, y_val, epochs=20, seed=42):
    set_seed(seed)
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = MLPClassifier(input_dim).to(device)
    crit  = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb).squeeze()
            loss = crit(out, yb)
            loss.backward()
            opt.step()
        val_acc, val_auc = evaluate_loader(model, val_loader)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
        print(f"[Classifier] Epoch {epoch:02d} | Val AUC: {val_auc:.4f}")
    model.load_state_dict(best_state)
    return model, best_auc


def test_classifier(model, X_test, y_test):
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    return evaluate_loader(model, test_loader)


# =========================================================
# 3. Random embedding networks for DM (LayerNorm MLPs)
# =========================================================
class RandomMLPEmbedder(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def make_random_embedders(n_nets=10, seed=123):
    set_seed(seed)
    nets = []
    for _ in range(n_nets):
        net = RandomMLPEmbedder(input_dim=input_dim).to(device)
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    return nets


embed_nets = make_random_embedders(n_nets=10)


# =========================================================
# 4. Compute real feature stats per class & embedder
# =========================================================
@torch.no_grad()
def compute_real_stats(X_train, y_train, embed_nets):
    y_int = y_train.long()
    classes = list(range(num_classes))
    n_nets = len(embed_nets)

    sample_feat = embed_nets[0](X_train[:128])
    embed_dim = sample_feat.shape[1]

    mu_real  = torch.zeros(n_nets, num_classes, embed_dim, device=device)
    std_real = torch.zeros_like(mu_real)

    for ni, net in enumerate(embed_nets):
        net.eval()
        for c in classes:
            idx = (y_int == c)
            feats = net(X_train[idx])
            mu_real[ni, c]  = feats.mean(0)
            std_real[ni, c] = feats.std(0) + 1e-6
    return mu_real, std_real


print("Computing real feature stats for DM...")
mu_real, std_real = compute_real_stats(X_train_t, y_train_t, embed_nets)
embed_dim = mu_real.shape[-1]


# =========================================================
# 5. DM-style synthesis for Adult
# =========================================================
def dm_synthesize_tabular(ipc=10, iters=1500, lr=0.05, seed=999):
    """
    DM-style synthesis:
      - ipc synthetic samples per class
      - matches mean & std in random-MLP embedding spaces
    """
    set_seed(seed)
    # synthetic data: [num_classes, ipc, input_dim]
    syn_data = torch.randn(num_classes, ipc, input_dim, device=device, requires_grad=True)

    opt = torch.optim.Adam([syn_data], lr=lr)
    y_classes = torch.arange(num_classes, device=device)

    for it in range(1, iters + 1):
        opt.zero_grad()
        loss = 0.0

        for ni, net in enumerate(embed_nets):
            net.eval()
            for c in range(num_classes):
                feats_syn = net(syn_data[c])  # [ipc, embed_dim]
                mu_syn  = feats_syn.mean(0)
                std_syn = feats_syn.std(0) + 1e-6

                mu_r  = mu_real[ni, c]
                std_r = std_real[ni, c]

                loss_mu  = (mu_syn  - mu_r).pow(2).mean()
                loss_std = (std_syn - std_r).pow(2).mean()
                loss += loss_mu + loss_std

        loss = loss / (len(embed_nets) * num_classes)
        loss.backward()
        opt.step()

        if it % 100 == 0 or it == 1:
            print(f"[DM] iter {it:04d} | loss = {loss.item():.6f}")

    # Flatten synthetic data to [N_syn, input_dim]
    X_syn = syn_data.view(num_classes * ipc, input_dim).detach()
    y_syn = torch.repeat_interleave(y_classes, repeats=ipc).float()
    return X_syn, y_syn


# =========================================================
# 6. Run baselines + DM experiment
# =========================================================

# ---- Full-data baseline ----
print("\n=== FULL DATA BASELINE ===")
full_model, full_val_auc = train_classifier(X_train_t, y_train_t, X_val_t, y_val_t)
full_test_acc, full_test_auc = test_classifier(full_model, X_test_t, y_test_t)
print(f"[FULL] Val AUC:  {full_val_auc:.4f} | Test AUC: {full_test_auc:.4f}")

# ---- Random real IPC baseline ----
ipc = 10
print(f"\n=== RANDOM REAL BASELINE (IPC = {ipc} per class) ===")
set_seed(2024)
idx0 = np.where(y_train == 0)[0]
idx1 = np.where(y_train == 1)[0]
idx0_sel = np.random.choice(idx0, ipc, replace=False)
idx1_sel = np.random.choice(idx1, ipc, replace=False)
idx_sel  = np.concatenate([idx0_sel, idx1_sel])

X_ipc_real = torch.tensor(X_train[idx_sel], device=device)
y_ipc_real = torch.tensor(y_train[idx_sel], device=device)

rand_model, rand_val_auc = train_classifier(X_ipc_real, y_ipc_real, X_val_t, y_val_t)
rand_test_acc, rand_test_auc = test_classifier(rand_model, X_test_t, y_test_t)
print(f"[RAND IPC] Val AUC:  {rand_val_auc:.4f} | Test AUC: {rand_test_auc:.4f}")

# ---- DM synthetic IPC baseline ----
print(f"\n=== DM TABULAR SYNTHETIC (IPC = {ipc} per class) ===")
X_syn, y_syn = dm_synthesize_tabular(ipc=ipc, iters=1500, lr=0.05, seed=2025)
dm_model, dm_val_auc = train_classifier(X_syn, y_syn, X_val_t, y_val_t)
dm_test_acc, dm_test_auc = test_classifier(dm_model, X_test_t, y_test_t)
print(f"[DM SYN] Val AUC: {dm_val_auc:.4f} | Test AUC: {dm_test_auc:.4f}")
