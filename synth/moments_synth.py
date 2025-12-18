import torch
import numpy as np

from models.embedders import sample_random_embedder


def dm_moment_synthesize(data, config):
    """
    Distribution Matching with explicit moment matching
    up to max_moment (1–4) in LN embedding space.
    """

    device = config["device"]

    X_train = data["X_train"].to(device).float()
    y_train = data["y_train"].to(device).long()

    # ---- hyperparameters ----
    ipc          = config["ipc"]
    iters        = config["dm_iters"]
    lr           = config["dm_lr"]
    batch_real   = config["dm_batch_real"]
    input_dim    = data["input_dim"]
    num_classes  = data["num_classes"]

    embed_hidden = config["dm_embed_hidden"]
    embed_dim    = config["dm_embed_dim"]
    embedder     = config["dm_embedder_type"]   # LN-only
    max_moment   = config["max_moment"]         # 1–4

    lambda3      = config.get("lambda3", 1e-3)
    lambda4      = config.get("lambda4", 1e-4)
    eps          = config.get("moment_eps", 1e-6)
    grad_clip    = config.get("grad_clip", 10.0)

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

    optimizer = torch.optim.SGD([syn_data], lr=lr, momentum=0.5)

    # ===========================
    # DM optimization
    # ===========================
    for it in range(iters + 1):

        embed_net = sample_random_embedder(
            embedder, input_dim, embed_hidden, embed_dim, device
        )

        optimizer.zero_grad()
        loss = 0.0

        for c in range(num_classes):

            real_b = get_real_batch(c, batch_real)
            syn_b  = syn_data[c*ipc:(c+1)*ipc]

            feat_real = embed_net(real_b).detach()
            feat_syn  = embed_net(syn_b)

            # ---- mean ----
            mu_real = feat_real.mean(0)
            mu_syn  = feat_syn.mean(0)
            loss += ((mu_real - mu_syn) ** 2).sum()

            if max_moment >= 2:
                # ---- variance ----
                var_real = feat_real.var(0, unbiased=False)
                var_syn  = feat_syn.var(0, unbiased=False)
                loss += ((var_real - var_syn) ** 2).sum()

            if max_moment >= 3:
                std_real = torch.sqrt(var_real + eps)
                std_syn  = torch.sqrt(var_syn + eps)

                z_real = (feat_real - mu_real) / std_real
                z_syn  = (feat_syn  - mu_syn)  / std_syn

                skew_real = (z_real ** 3).mean(0)
                skew_syn  = (z_syn  ** 3).mean(0)

                loss += lambda3 * ((skew_real - skew_syn) ** 2).sum()

            if max_moment >= 4:
                kurt_real = (z_real ** 4).mean(0)
                kurt_syn  = (z_syn  ** 4).mean(0)

                loss += lambda4 * ((kurt_real - kurt_syn) ** 2).sum()

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_([syn_data], grad_clip)
            optimizer.step()
        else:
            print("NaN/Inf loss detected — skipping step")

        if it % 50 == 0:
            print(
                f"[DM-M{max_moment}] iter {it:04d} | "
                f"loss {(loss.item() / num_classes):.6f}"
            )

    return syn_data.detach(), label_syn.detach()
