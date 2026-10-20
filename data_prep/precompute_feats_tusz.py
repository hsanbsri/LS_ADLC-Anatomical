"""precompute_feats_tusz.py — DWT subband features of the TUSZ cache (arm A).

Uses the SAME functions as the TUEV pipeline (features.normalize + dwt_features).
Arm A is a per-window z-score, so no per-recording statistics are needed.
The float16 cache is cast to float32 before normalization.

Only the 21 DWT dimensions (7 subbands x 3 statistics) are computed; the
periodicity and laterality features are disabled in the paper for both corpora.

Usage: python data_prep/precompute_feats_tusz.py --config configs/config_tusz.yaml --split train
Reads <cache_dir>/win_<split>.h5, writes <cache_dir>/feat_<split>_A.npy.
"""
import argparse, os, sys, time
import h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config
from features import normalize, dwt_features

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None, help="configs/config_tusz.yaml")
ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

cfg = load_config(a.config)
d = cfg["paths"]["cache_dir"]
src = f"{d}/win_{a.split}.h5"
out = f"{d}/feat_{a.split}_A.npy"
if os.path.exists(out):
    print(f"{out} already exists, skipped"); sys.exit(0)

nd = cfg["wavelet"]["n_subbands"] * len(cfg["wavelet"]["stats"])
t0 = time.time()
with h5py.File(src, "r") as h:
    N, C, T = h["X"].shape
    F = np.lib.format.open_memmap(out + ".part", mode="w+",
                                  dtype=np.float32, shape=(N, C, nd))
    for s in range(0, N, a.chunk):
        e = min(s + a.chunk, N)
        Z = normalize(h["X"][s:e].astype(np.float32), None, None, "A", cfg)
        F[s:e] = dwt_features(Z, cfg)
        if (s // a.chunk) % 10 == 0:
            el = time.time() - t0
            print(f"  {e:,}/{N:,}  {el:.0f}s  (~{el/max(e,1)*(N-e)/60:.0f} min left)",
                  flush=True)
    F.flush(); del F
os.replace(out + ".part", out)
F = np.load(out, mmap_mode="r")
print(f"-> {out}  shape={F.shape}  {os.path.getsize(out)/1e6:.0f} MB  "
      f"{(time.time()-t0)/60:.1f} min")
sub = np.asarray(F[::max(1, F.shape[0] // 20000)])
print(f"   finite={np.isfinite(sub).all()}  mean={sub.mean():.3f}  std={sub.std():.3f}")
