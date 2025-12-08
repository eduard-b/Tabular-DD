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
# Tabular Embedders for Ablation Studies
# ============================================================================

# ------------------------------------------------------
# 1. Simple LN embedder (baseline)
# ------------------------------------------------------
class EmbedderLN(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
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


# ------------------------------------------------------
# 2. Simple BN embedder (baseline)
# ------------------------------------------------------
class EmbedderBN(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# 3. Deeper BN embedder (3 BN blocks)
# ------------------------------------------------------
class EmbedderBNDeep(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# 4. Wide BN embedder (1024 → 512 → 256 → embed_dim)
# ------------------------------------------------------
class EmbedderBNWide(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# 5. Residual BN embedder
# ------------------------------------------------------
class BNResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        out = self.bn1(self.fc1(x))
        out = F.relu(out)
        out = self.bn2(self.fc2(out))
        out = out + identity
        return F.relu(out)


class EmbedderBNRes(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128, num_blocks=2):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(BNResBlock(hidden))
        self.blocks = nn.Sequential(*blocks)

        self.fc_out = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x


# ------------------------------------------------------
# 6. BN Cascade Embedder (heavy BN stack)
# ------------------------------------------------------
class EmbedderBNCascade(nn.Module):
    def __init__(self, input_dim, hidden=512, embed_dim=128, depth=4):
        super().__init__()

        layers = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            prev = hidden

        layers.append(nn.Linear(hidden, embed_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ln(x))

# -----------------------------------------------
# 1. LN (2-layer MLP)
# -----------------------------------------------
class EmbedderLN(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 2. LNDeep (3-layer)
# -----------------------------------------------
class EmbedderLNDeep(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 3. LNWide (wider)
# -----------------------------------------------
class EmbedderLNWide(nn.Module):
    def __init__(self, input_dim, hidden=512, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 4. LNRes (Residual MLP)
# -----------------------------------------------
class EmbedderLNRes(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.act = nn.ReLU()

        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.act(self.ln1(self.fc1(x)))
        h = h + self.act(self.ln2(self.fc2(h)))   # residual connection
        return self.out(h)

# -----------------------------------------------
# 5. LNCascade (maximal depth)
# -----------------------------------------------
class EmbedderLNCascade(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(6):  # 6 LN blocks
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            prev = hidden

        layers.append(nn.Linear(hidden, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sample_random_embedder(embedder_type, input_dim, hidden, embed_dim, device):
    seed = int(time.time() * 1000) % 100000
    torch.manual_seed(seed)

    if embedder_type == "LN":
        net = EmbedderLN(input_dim, hidden, embed_dim)
    elif embedder_type == "BN":
        net = EmbedderBN(input_dim, hidden, embed_dim)
    elif embedder_type == "BNDeep":
        net = EmbedderBNDeep(input_dim, hidden, embed_dim)
    elif embedder_type == "BNWide":
        net = EmbedderBNWide(input_dim, embed_dim)
    elif embedder_type == "BNRes":
        net = EmbedderBNRes(input_dim, hidden, embed_dim)
    elif embedder_type == "BNCascade":
        net = EmbedderBNCascade(input_dim, hidden, embed_dim)
    elif embedder_type == "LNDeep":
        net = EmbedderLNDeep(input_dim, hidden, embed_dim)
    elif embedder_type == "LNWide":
        net = EmbedderLNWide(input_dim, hidden*2, embed_dim)
    elif embedder_type == "LNRes":
        net = EmbedderLNRes(input_dim, hidden, embed_dim)
    elif embedder_type == "LNCascade":
        net = EmbedderLNCascade(input_dim, hidden, embed_dim)
    else:
        raise ValueError(f"Unknown embedder: {embedder_type}")

    net = net.to(device)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net

    for p in net.parameters():
        p.requires_grad = False

    net = net.to(device)
    net.eval()
    return net


# ============================================================================
# TRUE DM Synthesis (tabular)
# ============================================================================
def dm_true_synthesize_old(
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

def dm_true_synthesize_old_correct_BN_LN_split(
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
    use_bn,              # NEW: BN flag
    device,
):

    set_seed(0)
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()

    # group indices by class
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    # initialize synthetic data as random real samples
    syn_data = torch.randn(
        (num_classes * ipc, input_dim),
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc).long()

    def get_real_batch(c, n):
        idx = indices_class[c]
        if len(idx) < n:
            idx_sel = np.random.choice(idx, n, replace=True)
        else:
            idx_sel = np.random.choice(idx, n, replace=False)
        return X_train[idx_sel]

    # init with real samples
    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    optimizer_img = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    # -------------------------
    # DM optimization
    # -------------------------
    for it in range(iters + 1):

        embed_net = sample_random_embedder(
            config["dm_embedder_type"],
            input_dim,
            embed_hidden,
            embed_dim,
            device
        )

        optimizer_img.zero_grad()

        # ----------------------------------------
        # CASE 1: NON-BN → class-by-class forward
        # ----------------------------------------
        if not use_bn:

            loss = 0.0
            for c in range(num_classes):
                real_b = get_real_batch(c, batch_real)           # (B, d)
                syn_b  = syn_data[c*ipc:(c+1)*ipc]               # (ipc, d)

                feat_real = embed_net(real_b).detach()
                feat_syn  = embed_net(syn_b)

                mu_real = feat_real.mean(0)
                mu_syn  = feat_syn.mean(0)

                loss += ((mu_real - mu_syn)**2).sum()

        # ----------------------------------------
        # CASE 2: BN → concatenate all classes
        # ----------------------------------------
        else:
            # gather all real
            real_batches = []
            syn_batches  = []

            for c in range(num_classes):
                real_batches.append(get_real_batch(c, batch_real))
                syn_batches.append(syn_data[c*ipc:(c+1)*ipc])

            real_big = torch.cat(real_batches, dim=0)     # (C*B, d)
            syn_big  = torch.cat(syn_batches , dim=0)     # (C*ipc, d)

            # BN forward (two separate forward passes)
            feat_real_big = embed_net(real_big).detach()  # BN stats on real only
            feat_syn_big  = embed_net(syn_big)            # BN stats on syn only

            # reshape back into class groups
            feat_real = feat_real_big.reshape(num_classes, batch_real, -1)
            feat_syn  = feat_syn_big.reshape(num_classes, ipc, -1)

            # compute loss over per-class means
            mu_real = feat_real.mean(dim=1)   # (C, d)
            mu_syn  = feat_syn.mean(dim=1)    # (C, d)
            loss = ((mu_real - mu_syn)**2).sum()

        loss.backward()
        optimizer_img.step()

        if it % 50 == 0:
            print(f"[DM] iter {it:04d} | loss {(loss.item()/num_classes):.6f}")

    return syn_data.detach(), label_syn.detach()

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
    embedder_type,     # NEW: replaces use_bn
    device,
):

    set_seed(0)
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()

    # group indices by class
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    # initialize synthetic data
    syn_data = torch.randn(
        (num_classes * ipc, input_dim),
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc).long()

    def get_real_batch(c, n):
        idx = indices_class[c]
        if len(idx) < n:
            idx_sel = np.random.choice(idx, n, replace=True)
        else:
            idx_sel = np.random.choice(idx, n, replace=False)
        return X_train[idx_sel]

    # init synthetic with real samples
    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    optimizer_img = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    # ===========================
    # DM optimization
    # ===========================
    for it in range(iters + 1):

        embed_net = sample_random_embedder(
            embedder_type,
            input_dim,
            embed_hidden,
            embed_dim,
            device
        )

        optimizer_img.zero_grad()

        # Detect if embedder contains BatchNorm
        embed_has_bn = any(isinstance(m, nn.BatchNorm1d) for m in embed_net.modules())

        # -------------------------
        # CASE 1: NO BN (LN-only)
        # -------------------------
        if not embed_has_bn:

            loss = 0.0
            for c in range(num_classes):
                real_b = get_real_batch(c, batch_real)
                syn_b  = syn_data[c*ipc:(c+1)*ipc]

                feat_real = embed_net(real_b).detach()
                feat_syn  = embed_net(syn_b)

                mu_real = feat_real.mean(0)
                mu_syn  = feat_syn.mean(0)

                loss += ((mu_real - mu_syn)**2).sum()

        # -------------------------
        # CASE 2: BN MODE
        # -------------------------
        else:
            real_batches = []
            syn_batches  = []

            for c in range(num_classes):
                real_batches.append(get_real_batch(c, batch_real))
                syn_batches.append(syn_data[c*ipc:(c+1)*ipc])

            real_big = torch.cat(real_batches, dim=0)
            syn_big  = torch.cat(syn_batches, dim=0)

            # BN forward (two separate forward passes)
            feat_real_big = embed_net(real_big).detach()
            feat_syn_big  = embed_net(syn_big)

            # reshape into class groups
            feat_real = feat_real_big.reshape(num_classes, batch_real, -1)
            feat_syn  = feat_syn_big.reshape(num_classes, ipc, -1)

            mu_real = feat_real.mean(dim=1)
            mu_syn  = feat_syn.mean(dim=1)

            loss = ((mu_real - mu_syn)**2).sum()

        # optimization step
        loss.backward()
        optimizer_img.step()

        if it % 50 == 0:
            print(f"[DM] iter {it:04d} | loss {(loss.item()/num_classes):.6f}")

    return syn_data.detach(), label_syn.detach()

# ======================================================
# Collect BN running mean / var from a trained classifier
# ======================================================
def collect_bn_running_stats(model):
    means = []
    vars_ = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            means.append(m.running_mean.detach().clone())
            vars_.append(m.running_var.detach().clone())
    return means, vars_

# ======================================================
# Register forward hooks to capture BN activations
# ======================================================
def register_bn_hooks(model):
    activations = []

    def make_hook():
        def hook(module, input, output):
            activations.append(output)
        return hook
    
    handles = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            handles.append(m.register_forward_hook(make_hook()))

    return activations, handles

# ======================================================
# Compute BN stats loss
# ======================================================
def bn_stats_loss(real_means, real_vars, syn_acts):
    loss = 0.0
    for rm, rv, sa in zip(real_means, real_vars, syn_acts):
        mu_syn = sa.mean(dim=0)
        var_syn = sa.var(dim=0, unbiased=False)
        loss += (rm - mu_syn).pow(2).sum()
        loss += (rv - var_syn).pow(2).sum()
    return loss


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
        embedder_type=config["dm_embedder_type"],   # <-- NEW
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
    embedder_type = config["dm_embedder_type"]

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

# ======================================================
# DM-BN synthesis (BatchNorm-statistics matching only)
# ======================================================
def dm_bn_synthesize(
    model_full,
    X_train,
    y_train,
    ipc,
    iters,
    lr_img,
    batch_real,
    input_dim,
    num_classes,
    device,
):

    set_seed(0)
    model_full.eval()

    # extract BN stats once
    real_means, real_vars = collect_bn_running_stats(model_full)

    # prepare data
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).long()

    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        if len(idx) < n:
            idx_sel = np.random.choice(idx, n, replace=True)
        else:
            idx_sel = np.random.choice(idx, n, replace=False)
        return X_train[idx_sel]

    # init synthetic data
    syn_data = torch.randn(
        (num_classes * ipc, input_dim),
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc).long()

    # init syn with real samples (better than random)
    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    optimizer_img = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    # -------------------------
    # Optimization loop
    # -------------------------
    for it in range(iters + 1):

        optimizer_img.zero_grad()

        # gather synthetic batches per class
        syn_batches = []
        for c in range(num_classes):
            syn_batches.append(syn_data[c*ipc:(c+1)*ipc])
        syn_big = torch.cat(syn_batches, dim=0)

        # forward synthetic data through teacher and capture BN activations
        syn_acts, handles = register_bn_hooks(model_full)
        _ = model_full(syn_big)  # forward pass
        for h in handles:
            h.remove()

        # compute BN stats loss
        loss_bn = bn_stats_loss(real_means, real_vars, syn_acts)

        loss = loss_bn + 1e-3 * (syn_data**2).mean()


        loss.backward()
        optimizer_img.step()

        with torch.no_grad():
            syn_data.clamp_(-10, 10)


        if it % 50 == 0:
            print(f"[DM-BN] iter {it:04d} | loss {loss_bn.item():.6f}")

    return syn_data.detach(), label_syn.detach()

def run_dm_bn_experiment(config):

    print("===== RUNNING DM-BN EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device   = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # ---------------- Load dataset ----------------
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    input_dim   = data["input_dim"]
    num_classes = data["num_classes"]

    # ---------------- Full baseline ----------------
    print("\n--- FULL-DATA BASELINE ---")
    model_full, _ = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    full_acc, full_auc = evaluate_classifier(
        model_full, X_test, y_test, device, num_classes
    )

    # ---------------- Random IPC baseline ----------------
    print("\n--- RANDOM IPC BASELINE ---")
    X_rand, y_rand = make_random_ipc_subset(
        X_train, y_train,
        config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_rand, _ = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    rand_acc, rand_auc = evaluate_classifier(
        model_rand, X_test, y_test, device, num_classes
    )

    # ---------------- Herding IPC baseline ----------------
    print("\n--- HERDING IPC BASELINE ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train,
        config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_herd, _ = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    herd_acc, herd_auc = evaluate_classifier(
        model_herd, X_test, y_test, device, num_classes
    )

    # ---------------- DM-BN synthesis ----------------
    print("\n--- DM-BN SYNTHESIS ---")
    X_syn, y_syn = dm_bn_synthesize(
        model_full,
        X_train, y_train,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr_img=config["dm_lr"],
        batch_real=config["dm_batch_real"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
    )

    # save synthetic dataset
    syn_path = os.path.join(
        save_dir,
        f"{config['dataset_name']}_dmBN_ipc{config['ipc']}.pt"
    )
    torch.save({"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}, syn_path)
    print(f"Saved DM-BN synthetic data to {syn_path}")

    # ---------------- Train classifier on DM-BN synthetic ----------------
    print("\n--- CLASSIFIER ON DM-BN SYNTHETIC DATA ---")
    model_dm, _ = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    dm_acc, dm_auc = evaluate_classifier(
        model_dm, X_test, y_test, device, num_classes
    )

    # ---------------- Save result JSON ----------------
    result = {
        "dataset": config["dataset_name"],
        "method": "DM-BN",
        "ipc": config["ipc"],
        "results": {
            "full":  {"test_acc": full_acc,  "test_auc": full_auc},
            "random":{"test_acc": rand_acc,  "test_auc": rand_auc},
            "herd":  {"test_acc": herd_acc,  "test_auc": herd_auc},
            "dm_bn": {"test_acc": dm_acc,    "test_auc": dm_auc},
        }
    }

    fname = f"{config['dataset_name']}_dmBN_ipc{config['ipc']}.json"
    json_path = os.path.join(save_dir, fname)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved DM-BN results to {json_path}")
    print("===== DM-BN EXPERIMENT FINISHED =====")


# ============================================================================
# MAIN LOOP FOR ALL DATABASES
# ============================================================================
# if __name__ == "__main__":

#     DB_LIST = [
#         "drybean",
#         "adult",
#         "bank",
#         "credit",
#         "covertype",
#         "airlines",
#         "higgs",
#     ]

#     for db in DB_LIST:
#         print("\n" + "="*80)
#         print(f"Running TRUE DM experiment on dataset: {db.upper()}")
#         print("="*80)

#         config = {
#             "dataset_name": db,
#             "save_dir": "./results_trueDM_BN_2/",
#             "device": "cuda" if torch.cuda.is_available() else "cpu",

#             # DM hyperparameters
#             "ipc": 10,
#             "dm_iters": 2000,
#             "dm_lr": 0.05,
#             "dm_batch_real": 256,
#             "dm_seed": 2025,

#             # Embedder architecture
#             # ----------------------------------------------------------
#             # switch between LayerNorm and BatchNorm here:
#             # True → BatchNorm
#             # False → LayerNorm
#             # ----------------------------------------------------------
#             "dm_use_batchnorm": True,
#             "dm_embed_hidden": 256,
#             "dm_embed_dim": 128,

#             # Classifier
#             "classifier_hidden": [128, 64],
#             "classifier_epochs": 20,

#             # Reproducibility
#             "random_seed": 42,
#         }

#         run_dm_true_experiment(config)

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

    EMBEDDER_LIST = [
        "LN",
        "LNDeep",
        "LNWide",
        "LNRes",
        "LNCascade",
    ]

    RESULTS_DIR = "./results_embedder_ablation_LN/"
    ensure_dir(RESULTS_DIR)

    for db in DB_LIST:

        print("\n" + "="*90)
        print(f"Running embedder ablation for dataset: {db.upper()}")
        print("="*90)

        for emb in EMBEDDER_LIST:

            print(f"\n>>> Running embedder: {emb}")

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

                # New embedder selection
                "dm_embedder_type": emb,
                "dm_embed_hidden": 256,
                "dm_embed_dim": 128,

                # (remove old "dm_use_batchnorm" flag!)
                # Classifier
                "classifier_hidden": [128, 64],
                "classifier_epochs": 20,

                # Reproducibility
                "random_seed": 42,
            }

            ensure_dir(config["save_dir"])
            run_dm_true_experiment(config)

