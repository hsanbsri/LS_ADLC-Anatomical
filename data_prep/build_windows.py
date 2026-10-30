"""build_windows.py: TUEV EDF + .rec  ->  HDF5 cache of 5-s windows, 6 classes.

One sample = one annotated event (channel, t0, t1) of a .rec file. TUEV
annotations are per channel, so the same time segment can appear more than
once with different labels.

Processing order per recording:
  1. read the EDF -> 22-channel bipolar TCP montage
  2. 60/120 Hz notch, then 0.5-80 Hz band-pass, zero-phase, on the WHOLE recording
  3. robust per-channel statistics (median, IQR) of the whole filtered recording
  4. cut a 5-s window centred on each event

The cache stores the FILTERED but NOT normalized signal plus the recording
statistics, so both normalization arms can be run from the same cache.
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import (load_config, sig_len, build_montage, build_filters,  # noqa: E402
                    apply_filters, recording_stats)


def read_rec(path, code_to_class):
    """-> list of (ch_idx, t0, t1, class_idx). Lines with an unknown code are dropped."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ch, t0, t1, code = line.split(",")
            code = int(code)
            if code not in code_to_class:
                continue
            out.append((int(ch), float(t0), float(t1), code_to_class[code]))
    return out


def process_file(rec_path, cfg):
    import mne
    mne.set_log_level("ERROR")
    L = sig_len(cfg)
    fs = cfg["windowing"]["fs"]
    w = cfg["windowing"]
    edf = rec_path[:-4] + ".edf"
    events = read_rec(rec_path, cfg["labels"]["code_to_class"])
    if not events:
        return None, {"file": rec_path, "reason": "no_events"}
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        if abs(raw.info["sfreq"] - fs) > 1e-6:
            raw.resample(fs, npad="auto")
        X = build_montage(raw.get_data(), raw.ch_names, cfg["channels"])
        del raw
    except Exception as e:
        return None, {"file": rec_path, "reason": f"{type(e).__name__}: {e}"}

    T = X.shape[1]
    if T < w["min_record_sec"] * fs:
        return None, {"file": rec_path, "reason": f"too_short ({T/fs:.1f}s)"}

    notches, bp = build_filters(cfg)
    Y = apply_filters(X, notches, bp)
    del X
    med, scale = recording_stats(Y, cfg)

    n = len(events)
    Xw = np.empty((n, cfg["channels"]["n_channels"], L), dtype=np.float32)
    meta = np.empty((n, 4), dtype=np.float64)   # ch, t0, t1, cls
    shifted = 0
    for i, (ch, t0, t1, cls) in enumerate(events):
        ct = 0.5 * (t0 + t1) if w["center_on_event"] else t0
        s = int(round(ct * fs)) - L // 2
        if s < 0 or s + L > T:
            if not w["shift_at_edges"]:
                continue
            s = min(max(s, 0), T - L)
            shifted += 1
        Xw[i] = Y[:, s:s + L].astype(np.float32)
        meta[i] = (ch, t0, t1, cls)
    del Y
    return (Xw, meta, med, scale, shifted), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True, choices=["train", "eval"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first n files only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-tag", default="", help="cache-name suffix, e.g. lp40")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="config override, e.g. filtering.bandpass.high_hz=40")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    if args.overrides:
        print("override:", args.overrides, flush=True)
    L = sig_len(cfg)
    C = cfg["channels"]["n_channels"]
    recs = sorted(glob.glob(os.path.join(cfg["paths"]["edf_root"], args.split, "*", "*.rec")))
    if args.limit:
        recs = recs[:args.limit]
    print(f"[{args.split}] {len(recs)} .rec files", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    results = Parallel(n_jobs=cfg["runtime"]["n_jobs"], verbose=5, batch_size=1)(
        delayed(process_file)(r, cfg) for r in recs)

    ok = [(r, res) for r, (res, err) in zip(recs, results) if res is not None]
    errs = [err for _, err in results if err is not None]
    total = sum(res[0].shape[0] for _, res in ok)
    print(f"processed {len(ok)}/{len(recs)} files, {total} windows, "
          f"{time.time()-t0:.0f}s, failed={len(errs)}", flush=True)
    for e in errs:
        print("  FAILED", e, flush=True)

    tag = f"_{args.cache_tag}" if args.cache_tag else ""
    out = args.out or os.path.join(cfg["paths"]["cache_dir"], f"win_{args.split}{tag}.h5")
    subjects = sorted({os.path.basename(os.path.dirname(r)) for r, _ in ok})
    subj_id = {s: i for i, s in enumerate(subjects)}
    files = [os.path.relpath(r, cfg["paths"]["edf_root"]) for r, _ in ok]

    with h5py.File(out, "w") as h:
        dX = h.create_dataset("X", (total, C, L), dtype="float32")
        dy = h.create_dataset("y", (total,), dtype="int8")
        dch = h.create_dataset("chan", (total,), dtype="int8")
        dsu = h.create_dataset("subject", (total,), dtype="int32")
        dfi = h.create_dataset("file", (total,), dtype="int32")
        dt = h.create_dataset("t0", (total,), dtype="float32")
        dmed = h.create_dataset("rec_med", (total, C), dtype="float32")
        dsc = h.create_dataset("rec_scale", (total, C), dtype="float32")
        p, n_shift = 0, 0
        for fi, (r, (Xw, meta, med, scale, sh)) in enumerate(ok):
            k = Xw.shape[0]
            dX[p:p + k] = Xw
            dy[p:p + k] = meta[:, 3].astype(np.int8)
            dch[p:p + k] = meta[:, 0].astype(np.int8)
            dt[p:p + k] = meta[:, 1].astype(np.float32)
            dsu[p:p + k] = subj_id[os.path.basename(os.path.dirname(r))]
            dfi[p:p + k] = fi
            dmed[p:p + k] = med
            dsc[p:p + k] = scale
            p += k
            n_shift += sh
        assert p == total, (p, total)
        h.create_dataset("subject_names", data=np.array(subjects, dtype=h5py.string_dtype()))
        h.create_dataset("file_names", data=np.array(files, dtype=h5py.string_dtype()))
        h.attrs["config"] = json.dumps(cfg, default=str)
        h.attrs["n_windows"] = total
        h.attrs["n_shifted_at_edge"] = n_shift
        h.attrs["fs"] = cfg["windowing"]["fs"]
        h.attrs["window_sec"] = cfg["windowing"]["window_sec"]

    y = np.array([m[3] for _, (_, m, *_ ) in ok for m in [m] ]) if False else None
    with h5py.File(out, "r") as h:
        y = h["y"][:]
        print(f"\nsaved -> {out}  ({os.path.getsize(out)/1e9:.2f} GB)")
        print(f"windows shifted at a recording edge: {n_shift} ({100*n_shift/total:.2f}%)")
        print(f"unique subjects: {len(subjects)}")
        for c, nm in enumerate(cfg["labels"]["class_names"]):
            print(f"  {c} {nm:5s} {np.sum(y==c):7d}  ({100*np.mean(y==c):5.2f}%)")


if __name__ == "__main__":
    main()
