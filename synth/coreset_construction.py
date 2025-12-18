import torch
import numpy as np
from sklearn.cluster import KMeans

from utils.utils import set_seed


# ============================================================================
# Random IPC
# ============================================================================
def make_random_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    X_sel, y_sel = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, ipc, replace=len(idx) < ipc)
        X_sel.append(X[chosen])
        y_sel.append(np.full(len(chosen), cls))

    return (
        torch.tensor(np.vstack(X_sel), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_sel), device=device, dtype=torch.long),
    )


def random_ipc_synthesize(data, config):
    return make_random_ipc_subset(
        X_train=data["X_train"],
        y_train=data["y_train"],
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=config["device"],
    )


# ============================================================================
# Vector Quantization (KMeans centroids)
# ============================================================================
def make_vq_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y):
        Xc = X[y == cls]

        if len(Xc) <= ipc:
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
            continue

        km = KMeans(n_clusters=ipc, random_state=seed)
        km.fit(Xc)
        X_out.append(km.cluster_centers_)
        y_out.append(np.full(ipc, cls))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


def vq_synthesize(data, config):
    return make_vq_ipc_subset(
        X_train=data["X_train"],
        y_train=data["y_train"],
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=config["device"],
    )


# ============================================================================
# Voronoi-restricted (closest real samples to centroids)
# ============================================================================
def make_voronoi_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y):
        Xc = X[y == cls]

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

            dists = np.linalg.norm(Xc[idx] - centers[k], axis=1)
            chosen = idx[np.argmin(dists)]

            X_out.append(Xc[chosen:chosen+1])
            y_out.append(np.array([cls]))

    return (
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


def voronoi_synthesize(data, config):
    return make_voronoi_ipc_subset(
        X_train=data["X_train"],
        y_train=data["y_train"],
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=config["device"],
    )


# ============================================================================
# Gonzalez farthest-first traversal
# ============================================================================
def make_gonzalez_ipc_subset(X_train, y_train, ipc, seed=0, device="cpu"):
    set_seed(seed)

    X = X_train.cpu().numpy()
    y = y_train.cpu().numpy()

    X_out, y_out = [], []

    for cls in np.unique(y):
        Xc = X[y == cls]
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
        torch.tensor(np.vstack(X_out), device=device, dtype=torch.float32),
        torch.tensor(np.concatenate(y_out), device=device, dtype=torch.long),
    )


def gonzalez_synthesize(data, config):
    return make_gonzalez_ipc_subset(
        X_train=data["X_train"],
        y_train=data["y_train"],
        ipc=config["ipc"],
        seed=config["random_seed"],
        device=config["device"],
    )
