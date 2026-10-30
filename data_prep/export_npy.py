"""export_npy.py: write X_<split>.npy next to win_<split>.h5 (optional, saves memory).

With runtime.mmap = true (default), src/train.py and src/oof_encoder.py read the
signal array X through np.memmap from <cache_dir>/X_<split>.npy when that file
exists, so all processes share one copy in the OS page cache instead of
loading the whole array into memory (about 9 GB for TUEV train, 20 GB for TUSZ
train). The .npy file is an exact copy of the HDF5 dataset (same dtype), so
results are identical with or without it.

Usage: python data_prep/export_npy.py [--config configs/config_tusz.yaml] --split train
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()

cfg = load_config(a.config)
d = cfg["paths"]["cache_dir"]
src, out = os.path.join(d, f"win_{a.split}.h5"), os.path.join(d, f"X_{a.split}.npy")
if os.path.exists(out):
    sys.exit(f"{out} already exists, skipped")
with h5py.File(src, "r") as h:
    X = h["X"]
    M = np.lib.format.open_memmap(out + ".part", mode="w+", dtype=X.dtype, shape=X.shape)
    for s in range(0, X.shape[0], a.chunk):
        M[s:s + a.chunk] = X[s:s + a.chunk]
    M.flush(); del M
os.replace(out + ".part", out)
print(f"-> {out}")
