import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans

from prepare_database import prepare_db


# ============================================================================
# Utilities
# ============================================================================
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


# ============================================================================
# Random IPC (multiclass-safe)
# ============================================================================
def make_random_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    """
    Select ipc samples per class uniformly at random (no replacement).
    Works for any number of classes.
    """
    set_seed(seed)
    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    classes = np.unique(y)
    X_sel = []
    y_sel = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        sel = np.random.choice(idx, ipc, replace=False)
        X_sel.append(X[sel])
        y_sel.append(np.full(ipc, cls))

    X_sel = np.vstack(X_sel)
    y_sel = np.concatenate(y_sel)

    return (
        torch.tensor(X_sel, device=device, dtype=torch.float32),
        torch.tensor(y_sel, device=device, dtype=torch.long),
    )


# ============================================================================
# Herding IPC (multiclass-safe)
# ============================================================================
def make_herding_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    """
    KMeans herding: produce ipc synthetic samples per class.
    Multiclass-safe.
    """
    set_seed(seed)
    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    classes = np.unique(y).astype(int)

    X_herd = []
    y_herd = []

    for cls in classes:
        Xc = X[y == cls]

        if len(Xc) <= ipc:
            X_herd.append(Xc[:ipc])
            y_herd.append(np.full(min(ipc, len(Xc)), cls))
            continue

        km = KMeans(n_clusters=ipc, random_state=seed)
        km.fit(Xc)

        X_herd.append(km.cluster_centers_)
        y_herd.append(np.full(ipc, cls))

    X_herd = np.vstack(X_herd)
    y_herd = np.concatenate(y_herd)

    return (
        torch.tensor(X_herd, dtype=torch.float32, device=device),
        torch.tensor(y_herd, dtype=torch.long, device=device),
    )


# ============================================================================
# Classifier MLP (correct logits-only version)
# ============================================================================
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


# ============================================================================
# Train classifier
# ============================================================================
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


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_classifier(model, X, y, device, num_classes=2):
    model.eval()
    X = X.to(device).float()
    y = y.to(device)

    loader = DataLoader(TensorDataset(X, y), batch_size=2048, shuffle=False)
    probs_all = []
    trues_all = []

    with torch.no_grad():
        for xb, yb in loader:
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
        pred = (probs_all > 0.5).astype(int)
        acc = accuracy_score(trues_all, pred)
        auc = roc_auc_score(trues_all, probs_all)
    else:
        pred = np.argmax(probs_all, axis=1)
        acc = accuracy_score(trues_all, pred)
        auc = roc_auc_score(trues_all, probs_all, multi_class="ovr", average="macro")

    return acc, auc


# ============================================================================
# DM Tabular Embedder (configurable LN/BN)
# ============================================================================
class DMTabularEmbedder(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128, use_bn=False):
        super().__init__()

        if use_bn:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed_dim),
            )

    def forward(self, x):
        return self.net(x)


def sample_random_embedder(input_dim, hidden, embed_dim, use_bn, device):
    seed = int(time.time() * 1000) % 100000
    torch.manual_seed(seed)

    net = DMTabularEmbedder(input_dim, hidden, embed_dim, use_bn=use_bn).to(device)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net


# ============================================================================
# TRUE DM Synthesis (tabular)
# ============================================================================
def dm_true_synthesize(
    X_train,
    y_train,
    ipc,
    iters,
    lr_img,
    batch_real,
    input_dim,
    num_classes,
    embed_hidden,
    embed_dim,
    use_bn,
    device,
):

    set_seed(0)
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()

    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    # initialize synthetic = random real samples
    syn_data = torch.randn((num_classes * ipc, input_dim), device=device, requires_grad=True)
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc).long()

    optimizer_img = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    def get_real_batch(c, n):
        idx = indices_class[c]
        if len(idx) < n:
            idx_sel = np.random.choice(idx, n, replace=True)
        else:
            idx_sel = np.random.choice(idx, n, replace=False)
        return X_train[idx_sel]

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    # -------- DM optimization --------
    for it in range(iters + 1):
        embed_net = sample_random_embedder(
            input_dim, embed_hidden, embed_dim, use_bn, device
        )

        optimizer_img.zero_grad()
        loss = 0.0

        for c in range(num_classes):
            real_batch = get_real_batch(c, batch_real)
            syn_batch  = syn_data[c*ipc:(c+1)*ipc]

            feat_real = embed_net(real_batch).detach()
            feat_syn  = embed_net(syn_batch)

            mu_real = feat_real.mean(0)
            mu_syn  = feat_syn.mean(0)

            loss += ((mu_real - mu_syn)**2).sum()

        loss.backward()
        optimizer_img.step()

        if it % 50 == 0:
            print(f"[DM] iter {it:04d} | loss {(loss.item()/num_classes):.6f}")

    return syn_data.detach(), label_syn.detach()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_dm_true_experiment(config):

    print("===== RUNNING TRUE DM-TABULAR EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device   = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # ---------------- Load dataset ----------------
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    input_dim        = data["input_dim"]
    num_classes      = data["num_classes"]

    # ---------------- Full baseline ----------------
    print("\n--- FULL-DATA BASELINE ---")
    model_full, val_auc_full = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_full, test_auc_full = evaluate_classifier(
        model_full, X_test, y_test, device, num_classes
    )

    # ---------------- Random IPC baseline ----------------
    print(f"\n--- RANDOM IPC BASELINE (IPC={config['ipc']}) ---")
    X_rand, y_rand = make_random_ipc_subset(X_train, y_train, config["ipc"],
                                           seed=config["random_seed"],
                                           device=device)

    model_rand, val_auc_rand = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_rand, test_auc_rand = evaluate_classifier(
        model_rand, X_test, y_test, device, num_classes
    )

    # ---------------- Herding IPC baseline ----------------
    print(f"\n--- HERDING BASELINE (IPC={config['ipc']}) ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train, config["ipc"],
        seed=config["random_seed"],
        device=device,
    )
    model_herd, val_auc_herd = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_herd, test_auc_herd = evaluate_classifier(
        model_herd, X_test, y_test, device, num_classes
    )

    # ---------------- TRUE DM Synthesis ----------------
    print("\n--- TRUE DM SYNTHESIS ---")
    X_syn, y_syn = dm_true_synthesize(
        X_train, y_train,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr_img=config["dm_lr"],
        batch_real=config["dm_batch_real"],
        input_dim=input_dim,
        num_classes=num_classes,
        embed_hidden=config["dm_embed_hidden"],
        embed_dim=config["dm_embed_dim"],
        use_bn=config["dm_use_batchnorm"],
        device=device,
    )

    syn_path = os.path.join(
        save_dir,
        f"{config['dataset_name']}_trueDM_ipc{config['ipc']}_seed{config['dm_seed']}.pt"
    )
    torch.save({"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}, syn_path)
    print(f"Saved synthetic data to {syn_path}")

    # ---------------- Train classifier on synthetic ----------------
    print("\n--- CLASSIFIER ON TRUE DM SYNTHETIC DATA ---")
    model_dm, val_auc_dm = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_dm, test_auc_dm = evaluate_classifier(
        model_dm, X_test, y_test, device, num_classes
    )

    # ----------------------------------------------------------------------
    # Save JSON with all results
    # ----------------------------------------------------------------------
    run_id = time.strftime("%Y%m%d-%H%M%S")
    embedder_type = "BatchNorm" if config["dm_use_batchnorm"] else "LayerNorm"

    result = {
        "run_id": run_id,
        "dataset": config["dataset_name"],
        "method": "True DM (Tabular)",
        "ipc": config["ipc"],
        "embedder_type": embedder_type,
        "embed_hidden": config["dm_embed_hidden"],
        "embed_dim": config["dm_embed_dim"],
        "dm_iters": config["dm_iters"],
        "dm_lr": config["dm_lr"],
        "dm_batch_real": config["dm_batch_real"],
        "results": {
            "full":   {"val_auc": val_auc_full, "test_acc": test_acc_full, "test_auc": test_auc_full},
            "random": {"val_auc": val_auc_rand, "test_acc": test_acc_rand, "test_auc": test_auc_rand},
            "herd":   {"val_auc": val_auc_herd, "test_acc": test_acc_herd, "test_auc": test_auc_herd},
            "dm":     {"val_auc": val_auc_dm,  "test_acc": test_acc_dm,  "test_auc": test_auc_dm},
        }
    }

    fname = (
        f"{config['dataset_name']}_trueDM_"
        f"ipc{config['ipc']}_{embedder_type}_"
        f"h{config['dm_embed_hidden']}_e{config['dm_embed_dim']}_"
        f"it{config['dm_iters']}_lr{config['dm_lr']}_"
        f"{run_id}.json"
    )

    json_path = os.path.join(save_dir, fname)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to {json_path}")
    print("===== EXPERIMENT FINISHED =====")


# ============================================================================
# MAIN LOOP FOR ALL DATABASES
# ============================================================================
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

    for db in DB_LIST:
        print("\n" + "="*80)
        print(f"Running TRUE DM experiment on dataset: {db.upper()}")
        print("="*80)

        config = {
            "dataset_name": db,
            "save_dir": "./results_trueDM/",
            "device": "cuda" if torch.cuda.is_available() else "cpu",

            # DM hyperparameters
            "ipc": 10,
            "dm_iters": 2000,
            "dm_lr": 0.05,
            "dm_batch_real": 256,
            "dm_seed": 2025,

            # Embedder architecture
            # ----------------------------------------------------------
            # 🔥 switch between LayerNorm and BatchNorm here:
            # True → BatchNorm
            # False → LayerNorm
            # ----------------------------------------------------------
            "dm_use_batchnorm": True,
            "dm_embed_hidden": 256,
            "dm_embed_dim": 128,

            # Classifier
            "classifier_hidden": [128, 64],
            "classifier_epochs": 20,

            # Reproducibility
            "random_seed": 42,
        }

        run_dm_true_experiment(config)
