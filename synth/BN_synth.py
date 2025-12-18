import torch
import torch.nn as nn
import numpy as np
from models.embedders import sample_random_embedder

def dm_bn_synthesize(data, config):
    """
    Distribution Matching with BN embedders.
    """

    X_train = data["X_train"].to(config["device"]).float()
    y_train = data["y_train"].to(config["device"]).long()

    ipc          = config["ipc"]
    iters        = config["dm_iters"]
    lr_img       = config["dm_lr"]
    batch_real   = config["dm_batch_real"]
    input_dim    = data["input_dim"]
    num_classes  = data["num_classes"]
    embed_hidden = config["dm_embed_hidden"]
    embed_dim    = config["dm_embed_dim"]
    embedder     = config["dm_embedder_type"]
    device       = config["device"]

    # group indices
    y_np = y_train.cpu().numpy()
    indices_class = [np.where(y_np == c)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c]
        idx_sel = np.random.choice(idx, n, replace=len(idx) < n)
        return X_train[idx_sel]

    syn_data = torch.randn(
        (num_classes * ipc, input_dim),
        device=device,
        requires_grad=True,
    )
    label_syn = torch.arange(num_classes, device=device).repeat_interleave(ipc)

    with torch.no_grad():
        for c in range(num_classes):
            syn_data[c*ipc:(c+1)*ipc] = get_real_batch(c, ipc)

    optimizer = torch.optim.SGD([syn_data], lr=lr_img, momentum=0.5)

    for it in range(iters + 1):

        embed_net = sample_random_embedder(
            embedder, input_dim, embed_hidden, embed_dim, device
        )

        optimizer.zero_grad()

        real_batches, syn_batches = [], []
        for c in range(num_classes):
            real_batches.append(get_real_batch(c, batch_real))
            syn_batches.append(syn_data[c*ipc:(c+1)*ipc])

        real_big = torch.cat(real_batches, dim=0)
        syn_big  = torch.cat(syn_batches, dim=0)

        feat_real = embed_net(real_big).detach()
        feat_syn  = embed_net(syn_big)

        feat_real = feat_real.view(num_classes, batch_real, -1)
        feat_syn  = feat_syn.view(num_classes, ipc, -1)

        mu_real = feat_real.mean(1)
        mu_syn  = feat_syn.mean(1)

        loss = ((mu_real - mu_syn) ** 2).sum()
        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(f"[DM-BN] iter {it:04d} | loss {(loss.item()/num_classes):.6f}")

    return syn_data.detach(), label_syn.detach()
