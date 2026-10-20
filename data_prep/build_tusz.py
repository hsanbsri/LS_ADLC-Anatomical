"""build_tusz.py — TUSZ v2.0.5 EDF + .csv  ->  HDF5 cache of 5-s windows (binary task).

Scope: the WHOLE corpus.
  - all three montages. A bipolar montage cancels the reference (FP1-REF minus
    F7-REF = FP1 minus F7, the same for -LE), so 01_tcp_ar and 02_tcp_le are
    equivalent for the same electrode pairs and are pooled. 03_tcp_ar_a lacks
    A1/A2, so those two channels are filled with zeros (recorded in the
    `montage` array).
  - all subjects, including those without seizures (they contribute background).
  - the official subject-disjoint train / dev / eval pools of TUSZ.
  - preprocessing identical to TUEV: 60/120 Hz notch, 0.5-80 Hz band-pass,
    250 Hz, 5-s windows, 22-channel TCP montage.

Task: binary seizure vs. background, PER CHANNEL (TUSZ annotations are per channel).

Background windows. The TUSZ .csv annotations mark only SEIZURE periods; rows
labelled "bckg" are ~1-s gaps at segment edges near t = 0, so windows centred
on them fall outside the recording. Background is therefore sampled from the
NON-annotated part of each recording, at a safe distance (GUARD) from any
seizure on the same channel.

Usage (paths relative to the repository root):
  python data_prep/build_tusz.py --config configs/config_tusz.yaml \
      --tusz-root /path/to/TUSZ/v2.0.5/edf --split train
Writes <cache_dir>/win_<split>.h5 (cache_dir = data_tusz in config_tusz.yaml).
"""
import argparse, glob, os, sys, time
import h5py, numpy as np, mne, warnings
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config, build_montage, build_filters, apply_filters, montage_pairs

WIN = 5.0                      # seconds, as for TUEV
PER_EVENT = 3                  # windows per event-channel (long seizures -> several windows)
GUARD = 30.0                   # seconds between background windows and any seizure on the same channel


def read_ann(p):
    out = []
    try:
        fh = open(p, encoding="utf-8", errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            if line.startswith(("#", "channel")): continue
            q = line.strip().split(",")
            if len(q) < 5: continue
            try: a, b = float(q[1]), float(q[2])
            except ValueError: continue
            out.append((q[0], a, b, q[3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="configs/config_tusz.yaml")
    ap.add_argument("--tusz-root", required=True,
                    help="TUSZ edf/ directory containing train/, dev/ and eval/")
    ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    ap.add_argument("--max-subjects", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    D = a.tusz_root
    cfg = load_config(a.config); ch_cfg = cfg["channels"]
    pairs = montage_pairs(ch_cfg); names = [f"{x}-{y}" for x, y in pairs]
    idx_of = {n: i for i, n in enumerate(names)}
    fs = cfg["windowing"]["fs"]; L = int(WIN * fs)
    notches, bp = build_filters(cfg)

    files = sorted(glob.glob(f"{D}/{a.split}/*/*/*/*.edf"))
    bysub = {}
    for f in files: bysub.setdefault(f.split("/")[-4], []).append(f)
    keep = sorted(bysub)                       # ALL subjects
    if a.max_subjects: keep = keep[:a.max_subjects]
    import collections as _c
    mc = _c.Counter(f.split("/")[-2] for s in keep for f in bysub[s])
    print(f"{a.split}: {len(keep)} subjects, {sum(len(bysub[s]) for s in keep)} files "
          f"| montages {dict(mc)}", flush=True)

    rng = np.random.default_rng(20260924)
    NCH = len(names)
    import collections as _cc
    rej = _cc.Counter()
    out = f"{cfg['paths']['cache_dir']}/win_{a.split}.h5"
    tmp = out + ".part"
    hf = h5py.File(tmp, "w")
    dX = hf.create_dataset("X", shape=(0, NCH, L), maxshape=(None, NCH, L),
                           dtype=np.float16, chunks=(64, NCH, L))
    X = []; Y = []; C = []; S = []; M = []; t0 = time.time(); bad = 0

    def flush():
        """Move the buffer to disk; holding the whole corpus in RAM would need ~19 GB."""
        if not X: return
        n = dX.shape[0]; dX.resize(n + len(X), axis=0)
        dX[n:] = np.stack(X); X.clear()
    for si, s in enumerate(keep, 1):
        for f in bysub[s]:
            ann = read_ann(f[:-4] + ".csv")
            if not ann: continue
            mont = f.split("/")[-2]
            try:
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                raw.resample(fs, npad="auto")
                dat, cn = raw.get_data(), list(raw.ch_names)
                if mont == "03_tcp_ar_a":
                    # A1/A2 are not recorded in this variant -> channels A1-T3 and
                    # T4-A2 are filled with zeros; recorded in the `montage` array.
                    pref = cn[0].split()[0] + " " if cn[0].startswith("EEG") else ""
                    suf = "-LE" if cn[0].endswith("-LE") else "-REF"
                    for e in ("A1", "A2"):
                        if not any(e in c for c in cn):
                            cn.append(f"{pref}{e}{suf}")
                            dat = np.vstack([dat, np.zeros((1, dat.shape[1]))])
                sig = build_montage(dat, cn, ch_cfg)
                sig = apply_filters(sig, notches, bp)
            except Exception:
                bad += 1; continue
            T = sig.shape[1]
            if T < L:                 # recording shorter than one window
                bad += 1; continue
            # seizure spans per channel, used to keep background windows away from them
            sz_by_ch = {}
            for cname, st, sp, lab in ann:
                if lab != "bckg":
                    sz_by_ch.setdefault(idx_of.get(cname), []).append((st, sp))
            for cname, st, sp, lab in ann:
                if lab == "bckg":       # edge gap, not real background
                    continue
                ci = idx_of.get(cname)
                if ci is None: continue
                y = 0 if lab == "bckg" else 1
                span = sp - st
                n = 1 if span < WIN else min(PER_EVENT, int(span // WIN))
                for k in range(n):
                    mid = st + span * (k + 0.5) / n
                    b0 = int(mid * fs) - L // 2
                    if b0 < 0 or b0 + L > T: continue
                    w = sig[:, b0:b0 + L]
                    if w.shape != (NCH, L): rej[w.shape] += 1; continue
                    X.append((w * 1e6).astype(np.float16))
                    Y.append(y); C.append(ci); S.append(si - 1); M.append(mont)
            # ---- background: sampled from the NON-annotated part of the recording ----
            if not sz_by_ch:                       # subject/file without seizures
                sz_by_ch = {i: [] for i in range(len(names))}
            for ci, spans in sz_by_ch.items():
                if ci is None: continue
                taken = 0
                for _ in range(200):
                    if taken >= PER_EVENT * max(1, len(spans)): break
                    b0 = int(rng.integers(0, T - L + 1))
                    t_a, t_b = b0 / fs, (b0 + L) / fs
                    if any(t_a < e + GUARD and t_b > s0 - GUARD for s0, e in spans):
                        continue
                    w = sig[:, b0:b0 + L]
                    if w.shape != (NCH, L): rej[w.shape] += 1; continue
                    X.append((w * 1e6).astype(np.float16))
                    Y.append(0); C.append(ci); S.append(si - 1); M.append(mont); taken += 1
        flush()
        if si % 20 == 0:
            print(f"  {si}/{len(keep)} subjects | {dX.shape[0]:,} windows | "
                  f"{time.time()-t0:.0f}s", flush=True)

    flush()
    Y = np.array(Y, np.int8); C = np.array(C, np.int8); S = np.array(S, np.int32)
    assert dX.shape[0] == len(Y) == len(M), (dX.shape[0], len(Y), len(M))
    with hf as h:
        h.create_dataset("y", data=Y)
        h.create_dataset("chan", data=C); h.create_dataset("subject", data=S)
        h.create_dataset("subject_names", data=np.array(keep, dtype=object),
                         dtype=h5py.string_dtype())
        h.create_dataset("montage", data=np.array(M, dtype=object),
                         dtype=h5py.string_dtype())
        h.attrs["units"] = "microvolt"; h.attrs["dtype_note"] = "float16; multiply by 1e-6 for volts"
        h.attrs["task"] = "seizure vs bckg, per channel"; h.attrs["scope"] = "all montages, all subjects"
        h.attrs["bad_files"] = bad
        h.attrs["rejected_shapes"] = str(dict(rej))
    os.replace(tmp, out)                      # atomic: a partial cache never looks complete
    print(f"\n{out}  ({len(Y):,}, {NCH}, {L})  ({os.path.getsize(out)/1e9:.1f} GB)")
    if rej: print(f"  rejected shapes: {dict(rej)}")
    print(f"  seizure {int(Y.sum()):,} / background {int((Y==0).sum()):,} | "
          f"{len(keep)} subjects | {bad} failed files | {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
