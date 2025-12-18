import os, json, time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score

from prepare_database import prepare_db

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64], num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h

        # Output logits only
        if num_classes == 2:
            layers.append(nn.Linear(prev, 1))
        else:
            layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        if self.num_classes == 2:
            return logits.view(-1)  # shape [N]
        else:
            return logits            # shape [N, C]


def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    hidden,
    epochs,
    seed,
    device,
    num_classes,
):
    set_seed(seed)

    X_train = X_train.to(device).float()
    X_val   = X_val.to(device).float()
    y_train = y_train.to(device)
    y_val   = y_val.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=1024, shuffle=False)

    model = ClassifierMLP(input_dim, hidden, num_classes).to(device)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):

        # -------------- TRAIN ----------------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            if num_classes == 2:
                loss = criterion(logits, yb.float())
            else:
                loss = criterion(logits, yb.long())

            loss.backward()
            opt.step()

        # -------------- VALID ----------------
        model.eval()
        probs_all = []
        trues_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device)

                logits = model(xb)
                if num_classes == 2:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

                probs_all.append(probs.cpu().numpy())
                trues_all.append(yb.cpu().numpy())

        probs_all = np.concatenate(probs_all)
        trues_all = np.concatenate(trues_all)

        if num_classes == 2:
            val_auc = roc_auc_score(trues_all, probs_all)
        else:
            val_auc = roc_auc_score(trues_all, probs_all, multi_class="ovr", average="macro")

        print(f"[Classifier] Epoch {ep:02d} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_auc

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_classifier(model, X, y, device, num_classes=2):
    model.eval()
    X = X.to(device).float()
    y = y.to(device)

    loader = DataLoader(TensorDataset(X, y), batch_size=2048, shuffle=False)
    probs_all, trues_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            if num_classes == 2:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

            probs_all.append(probs.cpu().numpy())
            trues_all.append(yb.cpu().numpy())

    probs_all = np.concatenate(probs_all)
    trues_all = np.concatenate(trues_all)

    if num_classes == 2:
        pred = (probs_all > 0.5).astype(int)
        acc = accuracy_score(trues_all, pred)
        auc = roc_auc_score(trues_all, probs_all)
    else:
        pred = np.argmax(probs_all, axis=1)
        acc = accuracy_score(trues_all, pred)
        auc = roc_auc_score(trues_all, probs_all, multi_class="ovr", average="macro")

    return acc, auc

# ----------------------------------------------------------------------
# IPC baselines
# ----------------------------------------------------------------------

def make_random_ipc_subset(X, y, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_sel, y_sel = [], []

    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        if len(idx) <= ipc:
            chosen = idx
        else:
            chosen = np.random.choice(idx, ipc, replace=False)
        X_sel.append(X_np[chosen])
        y_sel.append(np.full(len(chosen), cls))

    return (
        torch.tensor(np.vstack(X_sel), device=device).float(),
        torch.tensor(np.concatenate(y_sel), device=device).long(),
    )

# ---------------- Vector Quantization (KMeans centroids) ----------------

def make_kmeans_vq_ipc_subset(X, y, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y_np):
        Xc = X_np[y_np == cls]
        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue

        km = KMeans(n_clusters=ipc, random_state=seed)
        km.fit(Xc)
        X_out.append(km.cluster_centers_)
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device).float(),
        torch.tensor(np.concatenate(y_out), device=device).long(),
    )

# ---------------- Voronoi-restricted (real samples) ----------------

def make_voronoi_ipc_subset(X, y, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y_np):
        Xc = X_np[y_np == cls]
        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue

        km = KMeans(n_clusters=ipc, random_state=seed)
        labels = km.fit_predict(Xc)
        centers = km.cluster_centers_

        for k in range(ipc):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            d = np.linalg.norm(Xc[idx] - centers[k], axis=1)
            chosen = idx[np.argmin(d)]
            X_out.append(Xc[chosen:chosen+1])
            y_out.append(np.array([cls]))

    return (
        torch.tensor(np.vstack(X_out), device=device).float(),
        torch.tensor(np.concatenate(y_out), device=device).long(),
    )

# ---------------- Gonzalez (farthest-first) ----------------

def make_gonzalez_ipc_subset(X, y, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y_np):
        Xc = X_np[y_np == cls]
        n = len(Xc)

        if n <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(n, cls))
            continue

        idx = [np.random.randint(0, n)]
        dist = np.full(n, np.inf)

        for _ in range(ipc - 1):
            last = Xc[idx[-1]]
            dist = np.minimum(dist, np.linalg.norm(Xc - last, axis=1))
            idx.append(np.argmax(dist))

        X_out.append(Xc[idx])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device).float(),
        torch.tensor(np.concatenate(y_out), device=device).long(),
    )

# ---------------- True Herding (identity features) ----------------

def make_true_herding_ipc_subset(X, y, ipc, seed=0, device="cpu"):
    set_seed(seed)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y_np):
        Xc = X_np[y_np == cls]
        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue

        mu = Xc.mean(axis=0)
        w = mu.copy()
        chosen = []

        for _ in range(ipc):
            scores = Xc @ w
            i = np.argmax(scores)
            chosen.append(i)
            w = w + mu - Xc[i]

        X_out.append(Xc[chosen])
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device).float(),
        torch.tensor(np.concatenate(y_out), device=device).long(),
    )

# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def run_all_ipc_baselines(config):

    device = config["device"]
    ensure_dir(config["save_dir"])

    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]

    input_dim   = data["input_dim"]
    num_classes = data["num_classes"]

    results = {}

    # ---------------- Full ----------------
    model, val_auc = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    acc, auc = evaluate_classifier(model, X_test, y_test, device, num_classes)
    results["full"] = {"acc": acc, "auc": auc}

    # ---------------- IPC methods ----------------
    methods = {
        "random":   make_random_ipc_subset,
        "vq":       make_kmeans_vq_ipc_subset,
        "voronoi":  make_voronoi_ipc_subset,
        "gonzalez": make_gonzalez_ipc_subset,
        "herding":  make_true_herding_ipc_subset,
    }

    for name, fn in methods.items():
        print(f"\n--- {name.upper()} IPC ---")
        Xs, ys = fn(X_train, y_train, config["ipc"],
                    seed=config["random_seed"], device=device)

        model, val_auc = train_classifier(
            Xs, ys, X_val, y_val,
            input_dim=input_dim,
            hidden=config["classifier_hidden"],
            epochs=config["classifier_epochs"],
            seed=config["random_seed"],
            device=device,
            num_classes=num_classes,
        )
        acc, auc = evaluate_classifier(model, X_test, y_test, device, num_classes)
        results[name] = {"acc": acc, "auc": auc}

    # Save
    path = os.path.join(
        config["save_dir"],
        f"{config['dataset_name']}_ipc{config['ipc']}_baseline.json"
    )
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {path}")

    return results

def save_results_to_csv(all_results, csv_path):
    # Prepare data for DataFrame
    rows = []
    methods = ["full", "random", "vq", "voronoi", "gonzalez", "herding"]

    for db, res in all_results.items():
        row = {"dataset": db}
        for method in methods:
            acc = res.get(method, {}).get("acc", None)
            auc = res.get(method, {}).get("auc", None)
            row[f"{method}_acc"] = acc
            row[f"{method}_auc"] = auc
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":

    DB_LIST = [
        "drybean",
        "adult",
        "bank",
        "credit",
        "covertype",
        "airlines",
        "higgs",
    ]

    RESULTS_DIR = "./results_coreset/"
    ensure_dir(RESULTS_DIR)

    all_results = {}

    for db in DB_LIST:

        print("\n" + "="*90)
        print(f"Running embedder ablation for dataset: {db.upper()}")
        print("="*90)

        config = {
            "dataset_name": db,
            "save_dir": os.path.join(RESULTS_DIR, db),
            "device": "cuda" if torch.cuda.is_available() else "cpu",

            # DM hyperparameters
            "ipc": 10,
            "dm_iters": 2000,
            "dm_lr": 0.05,
            "dm_batch_real": 256,
            "dm_seed": 2025,

            # Classifier
            "classifier_hidden": [128, 64],
            "classifier_epochs": 20,

            # Reproducibility
            "random_seed": 42,
        }

        ensure_dir(config["save_dir"])

        # Run experiments and collect results
        results = run_all_ipc_baselines(config)

        # Store the results for this dataset
        all_results[db] = results

    # Save all results into a single CSV
    save_results_to_csv(all_results, os.path.join(RESULTS_DIR, "all_results.csv"))
    print("===== EXPERIMENTS FINISHED =====")