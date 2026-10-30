"""precompute_feats.py: DWT subband features, computed once per (split, arm) -> .npy.

Normalization is deterministic, so precomputing the features is identical to
computing them on the fly, without the per-epoch cost.
"""
import argparse, os, sys, time
import h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config
from features import (normalize, dwt_features, periodicity_features,
                      laterality_features, feat_dim, glob_dim)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--split", required=True, choices=["train", "eval"])
ap.add_argument("--arm", required=True, choices=["A", "B"])
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--cache-tag", default="")
a = ap.parse_args()

cfg = load_config(a.config)
tag = f"_{a.cache_tag}" if a.cache_tag else ""
src = os.path.join(cfg["paths"]["cache_dir"], f"win_{a.split}{tag}.h5")
out = os.path.join(cfg["paths"]["cache_dir"], f"feat_{a.split}{tag}_{a.arm}.npy")
outg = os.path.join(cfg["paths"]["cache_dir"], f"glob_{a.split}{tag}_{a.arm}.npy")
t0 = time.time()
with h5py.File(src, "r") as h:
    N, C, T = h["X"].shape
    F = np.empty((N, C, feat_dim(cfg)), dtype=np.float32)
    G = np.empty((N, glob_dim(cfg)), dtype=np.float32)
    nd = cfg["wavelet"]["n_subbands"] * len(cfg["wavelet"]["stats"])
    for s in range(0, N, a.chunk):
        e = min(s + a.chunk, N)
        Z = normalize(h["X"][s:e], h["rec_med"][s:e], h["rec_scale"][s:e], a.arm, cfg)
        Fd = dwt_features(Z, cfg)
        F[s:e, :, :nd] = Fd
        if cfg["periodicity"]["enabled"]:
            F[s:e, :, nd:] = periodicity_features(Z, cfg)
        if cfg["laterality"]["enabled"]:
            G[s:e] = laterality_features(Fd, cfg)
        if (s // a.chunk) % 5 == 0:
            print(f"  {e}/{N}  {time.time()-t0:.0f}s", flush=True)
np.save(out, F)
if glob_dim(cfg):
    np.save(outg, G)
    print(f"-> {outg}  shape={G.shape}  finite={np.isfinite(G).all()}")
print(f"-> {out}  shape={F.shape}  {os.path.getsize(out)/1e6:.0f} MB  {time.time()-t0:.0f}s")
print(f"   finite={np.isfinite(F).all()}  mean={F.mean():.3f}  std={F.std():.3f}")
