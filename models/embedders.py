import torch.nn as nn
import time
import torch

# ------------------------------------------------------
# 1. Simple BN embedder (baseline)
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
# 2. Deeper BN embedder (3 BN blocks)
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
# 3. Wide BN embedder (1024 → 512 → 256 → embed_dim)
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
# 4. Residual BN embedder
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
# 5. BN Cascade Embedder (heavy BN stack)
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
# 6. LN (2-layer MLP)
# -----------------------------------------------
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

# -----------------------------------------------
# 8. LNDeep (3-layer)
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
# 9. LNWide (wider)
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
# 10. LNRes (Residual MLP)
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
# 11. LNCascade (maximal depth)
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

EMBEDDER_REGISTRY = {
    # ------------------
    # BatchNorm embedders
    # ------------------
    "bn": EmbedderBN,
    "bn_deep": EmbedderBNDeep,
    "bn_wide": EmbedderBNWide,
    "bn_res": EmbedderBNRes,
    "bn_cascade": EmbedderBNCascade,

    # ------------------
    # LayerNorm embedders
    # ------------------
    "ln": EmbedderLN,
    "ln_deep": EmbedderLNDeep,
    "ln_wide": EmbedderLNWide,
    "ln_res": EmbedderLNRes,
    "ln_cascade": EmbedderLNCascade,
}

def build_embedder(name: str, **kwargs):
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. "
            f"Available embedders: {list(EMBEDDER_REGISTRY.keys())}"
        )
    return EMBEDDER_REGISTRY[name](**kwargs)

def sample_random_embedder(
    embedder_type: str,
    input_dim: int,
    hidden: int,
    embed_dim: int,
    device: str,
):
    seed = int(time.time() * 1000) % 100000
    torch.manual_seed(seed)

    name = embedder_type.lower()

    # ---- constructor argument normalization ----
    if name in {"bn_wide"}:
        kwargs = dict(
            input_dim=input_dim,
            embed_dim=embed_dim,
        )

    elif name in {"ln_wide"}:
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden * 2,
            embed_dim=embed_dim,
        )

    else:
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden,
            embed_dim=embed_dim,
        )

    # ---- build via registry ----
    net = build_embedder(name, **kwargs)

    # ---- freeze + eval ----
    for p in net.parameters():
        p.requires_grad_(False)

    net = net.to(device)
    net.eval()
    return net