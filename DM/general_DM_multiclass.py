import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from prepare_database import prepare_db    # you will add multiclass datasets here


# =========================================================
# Utilities
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


# =========================================================
# Multi-class Classifier
# =========================================================

class MLPClassifierMulti(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=[256, 128]):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, num_classes))  # logits, no sigmoid/softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # return logits


def train_classifier(
        X_train, y_train,
        X_val, y_val,
        input_dim, num_classes,
        hidden, epochs=20, seed=42, device="cpu"):

    set_seed(seed)

    y_train = y_train.long()
    y_val   = y_val.long()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    val_loader   = DataLoader(
        TensorDataset(X_val, y_val), batch_size=1024, shuffle=False
    )

    model = MLPClassifierMulti(input_dim, num_classes, hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    best_f1 = -1
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        f1 = f1_score(trues, preds, average="macro")
        print(f"[Classifier] Epoch {ep:02d} | Val Macro-F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    if best_state is None:
        print("[WARNING] No valid F1 found. Using last epoch.")
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_classifier(model, X_test, y_test, device):
    model.eval()

    y_test = y_test.long()
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=2048, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro")
    return acc, f1


# =========================================================
# DM-related components
# =========================================================

class RandomMLPEmbedder(nn.Module):
    def __init__(self, input_dim, hidden=512, embed_dim=128):
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
        net = RandomMLPEmbedder(input_dim, hidden, embed_dim).to(device)
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    return nets


@torch.no_grad()
def compute_real_stats(X_train, y_train, embed_nets, num_classes):
    y_int = y_train.long()
    n_nets = len(embed_nets)

    # infer embedding dim
    embed_dim = embed_nets[0](X_train[:64]).shape[1]

    mu_real  = torch.zeros(n_nets, num_classes, embed_dim, device=X_train.device)
    std_real = torch.zeros_like(mu_real)

    for ni, net in enumerate(embed_nets):
        for c in range(num_classes):
            feats = net(X_train[y_int == c])
            mu_real[ni, c]  = feats.mean(0)
            std_real[ni, c] = feats.std(0) + 1e-6

    return mu_real, std_real


def dm_synthesize(
        X_train, y_train,
        mu_real, std_real, embed_nets,
        ipc, iters, lr, seed,
        input_dim, num_classes, device):

    set_seed(seed)

    syn = torch.randn(num_classes, ipc, input_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([syn], lr=lr)

    for it in range(1, iters + 1):
        opt.zero_grad()
        loss = 0.0

        for ei, net in enumerate(embed_nets):
            for c in range(num_classes):
                feats_syn = net(syn[c])       # [ipc, embed_dim]
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
            print(f"[DM] iter {it:04d} | loss={loss.item():.6f}")

    X_syn = syn.view(num_classes * ipc, input_dim).detach()
    y_syn = torch.repeat_interleave(
        torch.arange(num_classes, device=device), ipc
    ).long()

    return X_syn, y_syn


# =========================================================
# IPC baselines (Random + Herding)
# =========================================================

def make_random_ipc_subset(X_train, y_train, ipc, num_classes, seed, device):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    samples = []
    labels  = []

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        sel = np.random.choice(idx, ipc, replace=False)
        samples.append(X[sel])
        labels.append(np.full(ipc, c))

    samples = np.vstack(samples).astype(np.float32)
    labels  = np.concatenate(labels).astype(np.int64)

    return (
        torch.tensor(samples, device=device),
        torch.tensor(labels, device=device)
    )


from sklearn.cluster import KMeans

def make_herding_ipc_subset(X_train, y_train, ipc, num_classes, seed, device):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    samples = []
    labels  = []

    for c in range(num_classes):
        Xc = X[y == c]
        km = KMeans(n_clusters=ipc, random_state=seed)
        km.fit(Xc)
        centers = km.cluster_centers_
        samples.append(centers)
        labels.append(np.full(ipc, c))

    samples = np.vstack(samples).astype(np.float32)
    labels  = np.concatenate(labels).astype(np.int64)

    return (
        torch.tensor(samples, device=device),
        torch.tensor(labels, device=device)
    )


# =========================================================
# Main experiment engine
# =========================================================

def run_experiment(config):

    print("\n===== RUNNING MULTI-CLASS DM EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # Load dataset
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"],   data["y_val"]
    X_test, y_test   = data["X_test"],  data["y_test"]
    input_dim        = data["input_dim"]
    num_classes      = data["num_classes"]

    # ====================================================
    # Full data baseline
    # ====================================================
    print("\n--- FULL DATA BASELINE ---")
    model_full, f1_full_val = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    acc_full, f1_full = evaluate_classifier(
        model_full, X_test, y_test, device
    )
    print(f"[FULL] Test Acc={acc_full:.4f} | Macro-F1={f1_full:.4f}")

    # ====================================================
    # Random IPC baseline
    # ====================================================
    print(f"\n--- RANDOM IPC BASELINE (IPC={config['ipc']}) ---")
    X_rand, y_rand = make_random_ipc_subset(
        X_train, y_train,
        ipc=config["ipc"],
        num_classes=num_classes,
        seed=config["random_seed"],
        device=device
    )
    model_rand, _ = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim, num_classes,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    acc_rand, f1_rand = evaluate_classifier(
        model_rand, X_test, y_test, device
    )
    print(f"[RAND IPC] Test Acc={acc_rand:.4f} | Macro-F1={f1_rand:.4f}")

    # ====================================================
    # Herding IPC baseline
    # ====================================================
    print(f"\n--- HERDING BASELINE (IPC={config['ipc']}) ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train,
        ipc=config["ipc"],
        num_classes=num_classes,
        seed=config["random_seed"],
        device=device
    )
    model_herd, _ = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim, num_classes,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    acc_herd, f1_herd = evaluate_classifier(
        model_herd, X_test, y_test, device
    )
    print(f"[HERD IPC] Test Acc={acc_herd:.4f} | Macro-F1={f1_herd:.4f}")

    # ====================================================
    # DM synthesis
    # ====================================================
    print("\n--- DM SYNTHESIS ---")

    embed_nets = build_embedders(
        config["num_embedders"],
        input_dim=input_dim,
        hidden=config["embed_hidden"],
        embed_dim=config["embed_dim"],
        seed=config["random_seed"],
        device=device
    )

    mu_real, std_real = compute_real_stats(
        X_train, y_train, embed_nets, num_classes
    )

    X_syn, y_syn = dm_synthesize(
        X_train, y_train,
        mu_real, std_real, embed_nets,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr=config["dm_lr"],
        seed=config["dm_seed"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )

    # Save synthetic set
    save_path = os.path.join(
        save_dir,
        f"{config['dataset_name']}_syn_ipc{config['ipc']}_seed{config['dm_seed']}.npz"
    )
    np.savez(save_path, X_syn=X_syn.cpu().numpy(), y_syn=y_syn.cpu().numpy(), config=config)
    print(f"Saved synthetic dataset to {save_path}")

    # Evaluate DM
    print("\n--- CLASSIFIER ON DM SYNTHETIC DATA ---")
    model_dm, _ = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim, num_classes,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device
    )
    acc_dm, f1_dm = evaluate_classifier(model_dm, X_test, y_test, device)
    print(f"[DM] Test Acc={acc_dm:.4f} | Macro-F1={f1_dm:.4f}")

    # Save summary
    results = {
        "full_data": {"acc": float(acc_full), "f1": float(f1_full)},
        "rand_ipc": {"acc": float(acc_rand), "f1": float(f1_rand)},
        "herd_ipc": {"acc": float(acc_herd), "f1": float(f1_herd)},
        "dm_ipc":   {"acc": float(acc_dm),   "f1": float(f1_dm)},
        "config": config
    }

    with open(os.path.join(save_dir, f"results_{config['dataset_name']}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== EXPERIMENT COMPLETE =====")
    print(json.dumps(results, indent=2))


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    config = {
        "dataset_name": "drybean",
        "save_dir": "./results_multiclass/",
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Experiment knobs
        "ipc": 10,
        "dm_iters": 1500,
        "dm_lr": 0.05,
        "dm_seed": 2025,

        # Embedders
        "num_embedders": 10,
        "embed_hidden": 512,
        "embed_dim": 128,

        # Classifier
        "classifier_hidden": [256, 128],
        "classifier_epochs": 20,

        # Reproducibility
        "random_seed": 42,
    }

    run_experiment(config)
