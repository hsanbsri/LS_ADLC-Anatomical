"""latent_ops.py — generic latent-space augmentation operators.

Imported by train.py for its generic latent-augmentation option. The LS-ADLC
configuration used in the paper (gate "adlc", synth_mode "noise") does NOT use
these operators: its class-scaled Gaussian synthesis is implemented directly
in train.py (Eq. 4 of the paper).

Each operator works vectorized over a batch of latent vectors Z (B, D).
Application order:
    gaussian_noise -> feature_scale -> feature_dim_dropout -> structured_dropout
"""
import torch

OP_ORDER = ["gaussian_noise", "feature_scale", "feature_dim_dropout", "structured_dropout"]


def _bern(B, p, g, dev):
    return torch.rand(B, device=dev, generator=g) < p


def gaussian_noise(z, cfg, g):
    """+ N(0, sigma_ratio * std(z)); per-sample standard deviation over the vector."""
    p, sr = float(cfg["p"]), float(cfg["sigma_ratio"])
    if p <= 0 or sr <= 0:
        return z
    hit = _bern(z.shape[0], p, g, z.device)
    sd = z.detach().double().std(dim=-1, keepdim=True).to(z.dtype)
    noise = torch.randn(z.shape, device=z.device, generator=g, dtype=z.dtype)
    return torch.where(hit[:, None], z + noise * (sr * sd), z)


def feature_scale(z, cfg, g):
    """x U(lo, hi), one scalar per window."""
    p = float(cfg["p"]); lo, hi = cfg["range"]
    if p <= 0 or hi - lo <= 0:
        return z
    hit = _bern(z.shape[0], p, g, z.device)
    s = torch.rand(z.shape[0], 1, device=z.device, generator=g, dtype=z.dtype) * (hi - lo) + lo
    return torch.where(hit[:, None], z * s, z)


def feature_dim_dropout(z, cfg, g):
    """Zero k = max(1, round(ratio*D)) randomly SCATTERED dimensions per sample."""
    p, r = float(cfg["p"]), float(cfg["ratio"])
    B, D = z.shape
    k = min(D, max(1, int(round(r * D))))
    if p <= 0 or r <= 0:
        return z
    hit = _bern(B, p, g, z.device)
    # k random indices without repetition per row = argsort of random keys
    idx = torch.rand(B, D, device=z.device, generator=g).argsort(dim=1)[:, :k]
    mask = torch.ones(B, D, device=z.device, dtype=z.dtype)
    mask.scatter_(1, idx, 0.0)
    mask = torch.where(hit[:, None], mask, torch.ones_like(mask))
    return z * mask


def structured_dropout(z, cfg, g):
    """Zero one CONTIGUOUS block of dimensions; L ~ U{1..max(1,int(max_ratio*D))}."""
    p, mr = float(cfg["p"]), float(cfg["max_ratio"])
    B, D = z.shape
    Lmax = max(1, int(mr * D))
    if p <= 0 or mr <= 0:
        return z
    hit = _bern(B, p, g, z.device)
    L = torch.randint(1, Lmax + 1, (B, 1), device=z.device, generator=g)
    s = (torch.rand(B, 1, device=z.device, generator=g) * (D - L + 1).to(torch.float32)
         ).floor().to(torch.long)
    ar = torch.arange(D, device=z.device)[None, :]
    blk = (ar >= s) & (ar < s + L)
    mask = torch.where(blk, torch.zeros(1, device=z.device, dtype=z.dtype),
                       torch.ones(1, device=z.device, dtype=z.dtype))
    mask = torch.where(hit[:, None], mask, torch.ones_like(mask))
    return z * mask


OPS = {"gaussian_noise": gaussian_noise, "feature_scale": feature_scale,
       "feature_dim_dropout": feature_dim_dropout, "structured_dropout": structured_dropout}


def apply_chain(z, cfg, g, ops=None):
    """Apply the operator chain to Z (B, D). cfg: {op_name: {params}}."""
    for name in (ops or OP_ORDER):
        z = OPS[name](z, cfg[name], g)
    return z


# ---- encoding/decoding of an 8-dimensional search position (not used in the paper) ----
SPACE = [("gaussian_noise.p", 0.0, 1.0), ("gaussian_noise.sigma_ratio", 0.0, 0.10),
         ("feature_scale.p", 0.0, 1.0), ("feature_scale.width", 0.0, 0.40),
         ("feature_dim_dropout.p", 0.0, 1.0), ("feature_dim_dropout.ratio", 0.0, 0.60),
         ("structured_dropout.p", 0.0, 1.0), ("structured_dropout.max_ratio", 0.0, 0.60)]


def decode(pos):
    return {"gaussian_noise": {"p": float(pos[0]), "sigma_ratio": float(pos[1])},
            "feature_scale": {"p": float(pos[2]),
                              "range": [1 - float(pos[3]), 1 + float(pos[3])]},
            "feature_dim_dropout": {"p": float(pos[4]), "ratio": float(pos[5])},
            "structured_dropout": {"p": float(pos[6]), "max_ratio": float(pos[7])}}


NO_AUG = {"gaussian_noise": {"p": 0.0, "sigma_ratio": 0.0},
          "feature_scale": {"p": 0.0, "range": [1.0, 1.0]},
          "feature_dim_dropout": {"p": 0.0, "ratio": 0.0},
          "structured_dropout": {"p": 0.0, "max_ratio": 0.0}}
