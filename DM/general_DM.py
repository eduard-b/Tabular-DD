import os
import json
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

from prepare_database import prepare_db


# =========================================================
# Utility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_random_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    sel0 = np.random.choice(idx0, ipc, replace=False)
    sel1 = np.random.choice(idx1, ipc, replace=False)

    idx = np.concatenate([sel0, sel1])

    return (
        torch.tensor(X[idx], device=device, dtype=torch.float32),
        torch.tensor(y[idx], device=device, dtype=torch.float32),
    )


from sklearn.cluster import KMeans

def make_herding_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    X_herd = []
    y_herd = []

    for cls in [0, 1]:
        Xc = X[y == cls]
        km = KMeans(n_clusters=ipc, random_state=seed)
        km.fit(Xc)
        centers = km.cluster_centers_
        X_herd.append(centers)
        y_herd.append(np.full(ipc, cls))

    X_herd = np.vstack(X_herd)
    y_herd = np.concatenate(y_herd)

    return (
        torch.tensor(X_herd, device=device, dtype=torch.float32),
        torch.tensor(y_herd, device=device, dtype=torch.float32),
    )



# =========================================================
# Classifier MLP
# =========================================================
class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(X_train, y_train, X_val, y_val, input_dim, hidden, epochs=20, seed=42, device="cpu"):
    set_seed(seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=1024, shuffle=False)

    model = ClassifierMLP(input_dim=input_dim, hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.BCELoss()

    best_auc = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb).squeeze()
            loss = crit(out, yb)
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb).squeeze()
                preds.append(out.cpu().numpy())
                trues.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        auc = roc_auc_score(trues, preds)
        print(f"[Classifier] Epoch {ep:02d} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_auc


# =========================================================
# Random Embedders for DM
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


def build_embedders(n, input_dim, hidden, embed_dim, seed=42, device="cpu"):
    set_seed(seed)
    nets = []
    for _ in range(n):
        net = RandomMLPEmbedder(input_dim=input_dim, hidden=hidden, embed_dim=embed_dim).to(device)
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    return nets


# =========================================================
# DM Precomputation
# =========================================================
@torch.no_grad()
def compute_real_stats(X_train, y_train, embed_nets, num_classes):
    y_int = y_train.long()
    n_nets = len(embed_nets)

    sample_feat = embed_nets[0](X_train[:256])
    embed_dim = sample_feat.shape[1]

    mu_real  = torch.zeros(n_nets, num_classes, embed_dim, device=X_train.device)
    std_real = torch.zeros_like(mu_real)

    for i, net in enumerate(embed_nets):
        for c in range(num_classes):
            idx = (y_int == c)
            feats = net(X_train[idx])
            mu_real[i, c]  = feats.mean(0)
            std_real[i, c] = feats.std(0) + 1e-6

    return mu_real, std_real


# =========================================================
# DM Synthesis
# =========================================================
def dm_synthesize(X_train, y_train, mu_real, std_real, embed_nets,
                  ipc, iters, lr, seed, input_dim, num_classes, device):

    set_seed(seed)

    syn_data = torch.randn(num_classes, ipc, input_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([syn_data], lr=lr)

    for it in range(1, iters + 1):
        opt.zero_grad()
        loss = 0.0

        for ei, net in enumerate(embed_nets):
            for c in range(num_classes):
                feats_syn = net(syn_data[c])
                mu_s = feats_syn.mean(0)
                std_s = feats_syn.std(0) + 1e-6

                mu_r = mu_real[ei, c]
                std_r = std_real[ei, c]

                loss += (mu_s - mu_r).pow(2).mean()
                loss += (std_s - std_r).pow(2).mean()

        loss = loss / (len(embed_nets) * num_classes)
        loss.backward()
        opt.step()

        if it % 100 == 0 or it == 1:
            print(f"[DM] iter {it:04d} | loss: {loss.item():.6f}")

    X_syn = syn_data.view(num_classes * ipc, input_dim).detach()
    y_syn = torch.repeat_interleave(torch.arange(num_classes, device=device), ipc).float()
    return X_syn, y_syn


# =========================================================
# MAIN EXPERIMENT ENGINE
# =========================================================
def run_experiment(config):

    print("===== RUNNING DM-TABULAR EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    data = prepare_db(config, name=config['dataset_name'])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"],   data["y_val"]
    X_test, y_test   = data["X_test"],  data["y_test"]
    input_dim        = data["input_dim"]
    num_classes      = data["num_classes"]

    # -----------------------------------------
    # Full-data baseline
    # -----------------------------------------
    print("\n--- FULL DATA BASELINE ---")
    model_full, val_auc_full = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    test_acc_full, test_auc_full = evaluate_classifier(model_full, X_test, y_test, device)

    # -----------------------------------------
    # Random IPC baseline
    # -----------------------------------------
    print(f"\n--- RANDOM REAL BASELINE (IPC = {config['ipc']}) ---")
    X_rand, y_rand = make_random_ipc_subset(
        X_train, y_train,
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_rand, val_auc_rand = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    test_acc_rand, test_auc_rand = evaluate_classifier(model_rand, X_test, y_test, device)

    # -----------------------------------------
    # Herding IPC baseline
    # -----------------------------------------
    print(f"\n--- HERDING BASELINE (IPC = {config['ipc']}) ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train,
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_herd, val_auc_herd = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    test_acc_herd, test_auc_herd = evaluate_classifier(model_herd, X_test, y_test, device)


    # -----------------------------------------
    # Build embedders
    # -----------------------------------------
    embed_nets = build_embedders(
        n=config["num_embedders"],
        input_dim=input_dim,
        hidden=config["embed_hidden"],
        embed_dim=config["embed_dim"],
        seed=config["random_seed"],
        device=device
    )

    mu_real, std_real = compute_real_stats(X_train, y_train, embed_nets, num_classes)

    # -----------------------------------------
    # DM synthesis
    # -----------------------------------------
    print("\n--- DM SYNTHESIS ---")
    X_syn, y_syn = dm_synthesize(
        X_train, y_train,
        mu_real, std_real,
        embed_nets,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr=config["dm_lr"],
        seed=config["dm_seed"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )

    # Save synthetic dataset
    save_path = os.path.join(save_dir, f"{config['dataset_name']}syn_ipc{config['ipc']}_seed{config['dm_seed']}.npz")
    np.savez(save_path, X_syn=X_syn.cpu().numpy(), y_syn=y_syn.cpu().numpy(), config=config)
    print(f"Saved synthetic dataset to {save_path}")

    # -----------------------------------------
    # Train classifier on synthetic DM data
    # -----------------------------------------
    print("\n--- CLASSIFIER ON DM SYNTHETIC DATA ---")
    model_dm, val_auc_dm = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    test_acc_dm, test_auc_dm = evaluate_classifier(model_dm, X_test, y_test, device)

    # -----------------------------------------
    # Save experiment results
    # -----------------------------------------

    results = {
        "full_data_test_auc": float(test_auc_full),

        "random_ipc_val_auc": float(val_auc_rand),
        "random_ipc_test_auc": float(test_auc_rand),

        "herding_ipc_val_auc": float(val_auc_herd),
        "herding_ipc_test_auc": float(test_auc_herd),

        "dm_test_auc": float(test_auc_dm),
        "config": config
    }


    with open(os.path.join(save_dir, f"results_{config['dataset_name']}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== EXPERIMENT COMPLETE =====")

    print("Full-data AUC :", test_auc_full)
    print("Random IPC AUC:", test_auc_rand)
    print("Herding IPC AUC:", test_auc_herd)
    print("DM Synthetic AUC:", test_auc_dm)



# =========================================================
# Evaluate support
# =========================================================
def evaluate_classifier(model, X, y, device):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=2048, shuffle=False)
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





# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    config = {
        "dataset_name": "bank",
        "save_dir": "./results/",
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # experiment knobs:
        "ipc": 1,
        "dm_iters": 1500,
        "dm_lr": 0.05,
        "dm_seed": 2025,

        # embedders:
        "num_embedders": 10,
        "embed_hidden": 256,
        "embed_dim": 64,

        # classifier:
        "classifier_hidden": [128, 64],
        "classifier_epochs": 20,

        # reproducibility:
        "random_seed": 42,
    }

    run_experiment(config)
