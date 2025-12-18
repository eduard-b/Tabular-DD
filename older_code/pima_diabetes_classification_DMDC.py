import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# =====================================================
# 0. Reproducibility
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = "cpu"

# =====================================================
# 1. Load Pima dataset
# =====================================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
cols = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "label"
]
df = pd.read_csv(url, names=cols)

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
X_val_t   = torch.tensor(X_val,   device=device)
y_val_t   = torch.tensor(y_val,   device=device)
X_test_t  = torch.tensor(X_test,  device=device)
y_test_t  = torch.tensor(y_test,  device=device)

num_classes = 2
input_dim   = X_train.shape[1]

# =====================================================
# 2. Evaluation MLP (same as before)
# =====================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1),
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
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(trues, pred_labels)
    auc = roc_auc_score(trues, preds)
    return acc, auc

def train_classifier(X_train, y_train, X_val, y_val, epochs=40, seed=42):
    set_seed(seed)
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    model = MLPClassifier(input_dim=input_dim).to(device)
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
    model.load_state_dict(best_state)
    return model, best_auc

def test_classifier(model, X_test, y_test):
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    return evaluate_loader(model, test_loader)

# =====================================================
# 3. DM-style random feature extractors for tabular
# =====================================================
class RandomMLPEmbedder(nn.Module):
    def __init__(self, input_dim, hidden=128, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

def make_random_embedders(n_nets=5, seed=123):
    set_seed(seed)
    nets = []
    for _ in range(n_nets):
        net = RandomMLPEmbedder(input_dim=input_dim).to(device)
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    return nets

embed_nets = make_random_embedders(n_nets=5)

# =====================================================
# 4. Precompute real feature statistics per class & net
# =====================================================
@torch.no_grad()
def compute_real_stats(X_train, y_train, embed_nets):
    """
    Returns: mu_real, std_real with shape [n_nets, num_classes, embed_dim]
    """
    y_int = y_train.long()
    classes = list(range(num_classes))
    n_nets = len(embed_nets)

    # get embed_dim from a forward pass
    sample_feat = embed_nets[0](X_train[:4])
    embed_dim = sample_feat.shape[1]

    mu_real = torch.zeros(n_nets, num_classes, embed_dim, device=device)
    std_real = torch.zeros_like(mu_real)

    for ni, net in enumerate(embed_nets):
        net.train()  # use BN stats based on current batch
        for c in classes:
            idx = (y_int == c)
            feats = net(X_train[idx])
            mu_real[ni, c]  = feats.mean(0)
            std_real[ni, c] = feats.std(0) + 1e-6  # avoid zero
    return mu_real, std_real

mu_real, std_real = compute_real_stats(X_train_t, y_train_t, embed_nets)
embed_dim = mu_real.shape[-1]

# =====================================================
# 5. DM-style synthesis for tabular Pima
# =====================================================
def dm_synthesize_tabular(ipc=4, iters=1000, lr=0.1, seed=999):
    """
    ipc: synthetic samples per class
    Returns synthetic features (X_syn) and labels (y_syn)
    """
    set_seed(seed)
    # synthetic data: [num_classes, ipc, input_dim]
    syn_data = torch.randn(num_classes, ipc, input_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([syn_data], lr=lr)

    y_classes = torch.arange(num_classes, device=device)

    for it in range(1, iters + 1):
        opt.zero_grad()
        loss = 0.0
        # shape helpers
        # syn_data[c] -> [ipc, input_dim]
        for ni, net in enumerate(embed_nets):
            net.train()
            for c in range(num_classes):
                feats_syn = net(syn_data[c])  # [ipc, embed_dim]
                mu_syn  = feats_syn.mean(0)
                std_syn = feats_syn.std(0) + 1e-6

                mu_r = mu_real[ni, c]
                std_r = std_real[ni, c]

                loss_mu  = (mu_syn  - mu_r).pow(2).mean()
                loss_std = (std_syn - std_r).pow(2).mean()
                loss += loss_mu + loss_std

        loss = loss / (len(embed_nets) * num_classes)

        loss.backward()
        opt.step()

        if it % 100 == 0 or it == 1:
            print(f"[DM] iter {it:04d} | loss = {loss.item():.6f}")

    # flatten synthetic data to [N_syn, input_dim]
    X_syn = syn_data.view(num_classes * ipc, input_dim).detach()
    y_syn = torch.repeat_interleave(y_classes, repeats=ipc).float()
    return X_syn, y_syn

# =====================================================
# 6. Baselines + DM experiment
# =====================================================

# ---- Full-data baseline ----
print("\n=== FULL DATA BASELINE ===")
full_model, full_val_auc = train_classifier(X_train_t, y_train_t, X_val_t, y_val_t)
full_test_acc, full_test_auc = test_classifier(full_model, X_test_t, y_test_t)
print(f"Val AUC:  {full_val_auc:.4f} | Test AUC: {full_test_auc:.4f}")

# ---- Random real IPC baseline (same #points as DM) ----
ipc = 4
n_per_class = ipc
print(f"\n=== RANDOM REAL BASELINE (IPC = {ipc}) ===")
set_seed(1234)
idx0 = np.where(y_train == 0)[0]
idx1 = np.where(y_train == 1)[0]
idx0_sel = np.random.choice(idx0, n_per_class, replace=False)
idx1_sel = np.random.choice(idx1, n_per_class, replace=False)
idx_sel  = np.concatenate([idx0_sel, idx1_sel])

X_ipc_real = torch.tensor(X_train[idx_sel], device=device)
y_ipc_real = torch.tensor(y_train[idx_sel], device=device)

rand_model, rand_val_auc = train_classifier(X_ipc_real, y_ipc_real, X_val_t, y_val_t)
rand_test_acc, rand_test_auc = test_classifier(rand_model, X_test_t, y_test_t)
print(f"Val AUC:  {rand_val_auc:.4f} | Test AUC: {rand_test_auc:.4f}")

# ---- DM-synthesized IPC baseline ----
print(f"\n=== DM TABULAR SYNTHETIC (IPC = {ipc}) ===")
X_syn, y_syn = dm_synthesize_tabular(ipc=ipc, iters=1000, lr=0.1, seed=2025)
dm_model, dm_val_auc = train_classifier(X_syn, y_syn, X_val_t, y_val_t)
dm_test_acc, dm_test_auc = test_classifier(dm_model, X_test_t, y_test_t)
print(f"Val AUC:  {dm_val_auc:.4f} | Test AUC: {dm_test_auc:.4f}")
