import torch
import numpy as np

from models.embedders import sample_random_embedder


def dm_moment_synthesize_cov2(data, config):
    """
    Distribution Matching with explicit moment matching:
      - 1st order: mean
      - 2nd order: FULL covariance matrix (not just per-dim variance)

    Matches the structure/style of your existing dm_moment_synthesize.
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
    embedder     = config["dm_embedder_type"]  # e.g. "ln", "node", etc.

    grad_clip    = config.get("grad_clip", 10.0)
    eps          = config.get("moment_eps", 1e-6)

    # Optional: weight the covariance term (keeps behavior tunable without changing structure)
    cov_weight   = config.get("cov_weight", 1.0)

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
            syn_data[c * ipc : (c + 1) * ipc] = get_real_batch(c, ipc)

    optimizer = torch.optim.SGD([syn_data], lr=lr, momentum=0.5)

    def cov_matrix(z: torch.Tensor):
        """
        z: (n, d)
        returns: (d, d) covariance estimate (centered).
        Uses 1/n scaling (unbiased=False equivalent for stability).
        """
        n = z.shape[0]
        mu = z.mean(0, keepdim=True)
        zc = z - mu
        # (d,d) = (d,n) @ (n,d)
        cov = (zc.T @ zc) / max(n, 1)
        return mu.squeeze(0), cov

    # ===========================
    # DM optimization
    # ===========================
    for it in range(iters + 1):
        embed_net = sample_random_embedder(
            embedder, input_dim, embed_hidden, embed_dim, device
        )

        optimizer.zero_grad()
        loss = torch.zeros((), device=device)

        for c in range(num_classes):
            real_b = get_real_batch(c, batch_real)
            syn_b  = syn_data[c * ipc : (c + 1) * ipc]

            feat_real = embed_net(real_b).detach()
            feat_syn  = embed_net(syn_b)

            # 1st + 2nd order (full cov)
            mu_r, cov_r = cov_matrix(feat_real)
            mu_s, cov_s = cov_matrix(feat_syn)

            loss = loss + ((mu_r - mu_s) ** 2).sum()

            # Frobenius norm of covariance difference (squared)
            # Optional small eps stabilizer if your embedder outputs are extremely low-variance
            # (usually not needed, but harmless)
            diff = (cov_r - cov_s)
            loss = loss + cov_weight * (diff * diff).sum()

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_([syn_data], grad_clip)
            optimizer.step()
        else:
            print("NaN/Inf loss detected — skipping step")

        if it % 50 == 0:
            print(
                f"[DM-M1+FullCov] iter {it:04d} | "
                f"loss {(loss.item() / num_classes):.6f}"
            )

    return syn_data.detach(), label_syn.detach()
