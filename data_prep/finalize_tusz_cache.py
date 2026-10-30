"""finalize_tusz_cache.py: add rec_med / rec_scale placeholders to the TUSZ cache.

src/train.py (load_split) always reads rec_med/rec_scale, but normalization
arm A (per-window z-score, used in the paper) never uses their values. The
placeholders med = 0 / scale = 1 are therefore sufficient and exact for arm A;
re-reading every EDF for per-recording statistics is not needed.

Usage: python data_prep/finalize_tusz_cache.py --config configs/config_tusz.yaml --split train
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None, help="configs/config_tusz.yaml")
ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
a = ap.parse_args()

cfg = load_config(a.config)
path = os.path.join(cfg["paths"]["cache_dir"], f"win_{a.split}.h5")
with h5py.File(path, "a") as h:
    n, c = h["X"].shape[0], h["X"].shape[1]
    if "rec_med" in h and "rec_scale" in h:
        print(f"{path}: rec_med/rec_scale already present, nothing to do")
    else:
        h.create_dataset("rec_med", data=np.zeros((n, c), dtype=np.float32))
        h.create_dataset("rec_scale", data=np.ones((n, c), dtype=np.float32))
        print(f"{path}: added rec_med = 0, rec_scale = 1, shape ({n}, {c})")
