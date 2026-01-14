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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 12. LN ResMLP: deeper + wider
# ----------------------------

class _LNResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * expansion
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class EmbedderLNResXL(nn.Module):
    """
    Deeper + wider residual MLP embedder:
      x -> proj -> [ResBlocks]*depth -> LN -> out_proj
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        embed_dim: int,
        depth: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [_LNResBlock(hidden, expansion=expansion, dropout=dropout) for _ in range(depth)]
        )
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_ln(x)
        x = self.out_proj(x)
        return x


# -----------------------------------------
# 13. NODE-style embedder: soft oblivious trees
# -----------------------------------------

def _leaf_bit_matrix(depth: int, device=None):
    """
    Returns bits matrix of shape (2^depth, depth) with entries in {0,1}
    where row i is the binary representation of i over 'depth' bits.
    """
    n_leaves = 2 ** depth
    ar = torch.arange(n_leaves, device=device).unsqueeze(1)  # (L,1)
    bits = (ar >> torch.arange(depth, device=device)) & 1    # (L,depth) little-endian
    return bits.float()


class ObliviousTreeEnsemble(nn.Module):
    """
    Differentiable oblivious trees (NODE-ish):
    - Each depth chooses a (soft) feature via softmax over input dims
    - Uses learnable thresholds and temperatures (alpha)
    - Computes leaf probabilities and mixes leaf values -> (B, tree_dim)
    """
    def __init__(
        self,
        input_dim: int,
        num_trees: int,
        depth: int,
        tree_dim: int,
        alpha_init: float = 5.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim

        # Feature selection logits: (T, D, input_dim)
        self.feature_logits = nn.Parameter(torch.zeros(num_trees, depth, input_dim))

        # Thresholds per tree+depth: (T, D)
        self.thresholds = nn.Parameter(torch.zeros(num_trees, depth))

        # Temperature / sharpness per tree+depth: (T, D), constrained positive via softplus
        self.alpha_unconstrained = nn.Parameter(torch.full((num_trees, depth), math.log(math.exp(alpha_init) - 1.0)))

        # Leaf values: (T, 2^D, tree_dim)
        self.leaf_values = nn.Parameter(torch.zeros(num_trees, 2 ** depth, tree_dim))

        # Small init helps stability
        nn.init.normal_(self.leaf_values, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (B, input_dim)
        returns: (B, tree_dim)
        """
        B, Din = x.shape
        assert Din == self.input_dim

        device = x.device
        bits = _leaf_bit_matrix(self.depth, device=device)  # (L, D)
        L = bits.shape[0]

        # Soft feature selection
        sel = F.softmax(self.feature_logits, dim=-1)  # (T, D, Din)

        # Selected feature value per tree+depth: (B, T, D)
        # einsum: (B,Din) x (T,D,Din) -> (B,T,D)
        x_sel = torch.einsum("bi,tdi->btd", x, sel)

        # Compute decision probs p in (0,1): (B,T,D)
        alpha = F.softplus(self.alpha_unconstrained) + 1e-6
        thr = self.thresholds
        p = torch.sigmoid((x_sel - thr.unsqueeze(0)) * alpha.unsqueeze(0))

        # Leaf probs for each tree: (B,T,L)
        # For leaf with bit=1 use p, else use (1-p)
        # Expand to (B,T,1,D) and (1,1,L,D)
        p_exp = p.unsqueeze(2)                 # (B,T,1,D)
        bits_exp = bits.view(1, 1, L, self.depth)
        probs = bits_exp * p_exp + (1.0 - bits_exp) * (1.0 - p_exp)  # (B,T,L,D)
        leaf_prob = probs.prod(dim=-1)         # (B,T,L)

        # Mix leaf values: (B,T,L) @ (T,L,tree_dim) -> (B,T,tree_dim)
        out = torch.einsum("btl,tld->btd", leaf_prob, self.leaf_values)

        # Sum over trees -> (B, tree_dim)
        return out.sum(dim=1)


class EmbedderNODE(nn.Module):
    """
    A NODE-style embedder built as stacked oblivious-tree ensembles with residuals.
    Produces (B, embed_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,      # not strictly needed; kept for compatibility with your factory
        embed_dim: int,
        num_layers: int = 4,
        num_trees: int = 64,
        depth: int = 6,
        tree_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if tree_dim is None:
            tree_dim = max(16, embed_dim // 2)

        self.in_proj = nn.Linear(input_dim, tree_dim)
        self.layers = nn.ModuleList([
            ObliviousTreeEnsemble(
                input_dim=tree_dim,
                num_trees=num_trees,
                depth=depth,
                tree_dim=tree_dim,
            )
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(tree_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(tree_dim, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)  # (B, tree_dim)
        for layer in self.layers:
            # residual update (NODE-ish stacking)
            h = h + self.dropout(layer(self.ln(h)))
        return self.out_proj(self.ln(h))



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

    # ------------------
    # New embedders
    # ------------------
    "ln_res_xl": EmbedderLNResXL,
    "node": EmbedderNODE,
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

    elif name in {"ln_res_xl"}:
        # wider + deeper than your baseline ln_res
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden * 2,
            embed_dim=embed_dim,
            depth=10,
            expansion=4,
            dropout=0.0,
        )

    elif name in {"node"}:
        # NODE-ish defaults; hidden kept for signature compat
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden,
            embed_dim=embed_dim,
            num_layers=2,
            num_trees=64,
            depth=6,
            tree_dim=max(16, embed_dim // 2),
            dropout=0.0,
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
