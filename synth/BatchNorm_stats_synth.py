import torch
import torch.nn as nn
import json

from utils.utils import set_seed
import numpy as np

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

def batchnorm_stats_synthesize(data, config):
    """
    Teacher-based synthesis via BatchNorm statistics matching.
    Requires a trained classifier with BN layers.
    """

    device = config["device"]

    # ---- required inputs ----
    model_full = data["teacher"]
    model_full.eval()

    X_train = data["X_train"].to(device).float()
    y_train = data["y_train"].to(device).long()

    ipc         = config["ipc"]
    iters       = config["dm_iters"]
    lr_img      = config["dm_lr"]
    batch_real  = config["dm_batch_real"]
    input_dim   = data["input_dim"]
    num_classes = data["num_classes"]

    # ---- extract real BN stats once ----
    real_means, real_vars = collect_bn_running_stats(model_full)

    # ---- group indices by class ----
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        idx_sel = np.random.choice(idx, n, replace=len(idx) < n)
        return X_train[idx_sel]

    # ---- initialize synthetic data ----
    syn_data = torch.randn(
        (num_classes * ipc, input_dim),
        device=device,
        requires_grad=True,
    )

    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    # init from real samples
    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    optimizer = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    # ---- optimization loop ----
    for it in range(iters + 1):

        optimizer.zero_grad()

        syn_big = torch.cat(
            [syn_data[c*ipc:(c+1)*ipc] for c in range(num_classes)],
            dim=0
        )

        syn_acts, handles = register_bn_hooks(model_full)
        _ = model_full(syn_big)
        for h in handles:
            h.remove()

        loss_bn = bn_stats_loss(real_means, real_vars, syn_acts)
        loss_reg = 1e-3 * (syn_data ** 2).mean()
        loss = loss_bn + loss_reg

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            syn_data.clamp_(-10, 10)

        if it % 50 == 0:
            print(f"[BN-STATS] iter {it:04d} | loss {loss_bn.item():.6f}")

    return syn_data.detach(), label_syn.detach()