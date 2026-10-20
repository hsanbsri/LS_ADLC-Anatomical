"""
fig1_examples.py
================
Fig. 1: one representative TUEV evaluation window for each of the six
classes, rendered from the ORIGINAL TUEV EDF recordings (paths.edf_root in configs/config.yaml) in a
clinical EEG-viewer style, each with a zoom box on the annotated channel.

Main trace of each panel: all 22 bipolar TCP channels, a 6-s page centred on
the annotated event, 1-s grid, fixed sensitivity without clipping, the
annotated 1-s interval (from the .rec file) shaded and the annotated channel
drawn in colour. Zoom box: the annotated channel from 0.5 s before to 0.5 s
after the annotated interval, enlarged.

Montage and filtering are identical to the model pipeline
(common.build_montage / build_filters / apply_filters).

Selection (data-driven, no hand picking): for each class, the evaluation
window with the highest seed-mean probability of its true class under the
LS-ADLC ensemble (confirmatory batch, seeds 100-124, fusion by Eq. 1), with
the six windows drawn from six different subjects (rarest class picks first)
and events at least 10 s from both ends of the recording. When the same
segment is annotated for the class on several channels, the annotated channel
with the largest peak-to-peak amplitude inside the annotated interval is shown.
Writes figures/output/fig_examples.{png,pdf}. Requires the raw TUEV EDF/.rec
files, data/win_eval.h5 and the batch-2 pLS *_evalprob.npy files.
"""
import os
import sys
import numpy as np
import h5py
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, ".."))
OUTS = [os.path.join(HERE, "output")]
DS = None                        # raw TUEV EDF root; set from the config in main()
plt.rcParams["font.family"] = "DejaVu Sans"
NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
FULL = ["spike-and-sharp-wave", "generalized periodic discharge", "lateralized periodic discharge",
        "eye movement", "artifact", "background"]
EDGE = 10.0                      # s; events this close to the start or end of a recording are skipped
COL = ["#a31515", "#b5651d", "#8e44ad", "#1f6f3a", "#1f4e8c", "#555555"]
GN = ["temporal", "central", "parasagittal"]
SEEDS = range(100, 125)
FS = 250.0
PAGE, SPACING = 6.0, 120.0       # seconds per page; microvolts between channel baselines


def seed_mean_prob(arm):
    acc = None
    for s in SEEDS:
        lp = np.mean([np.log(np.clip(np.load(f"{R}/results/{arm}_{g}_armA_seed{s}_evalprob.npy"), 1e-12, 1))
                      for g in GN], 0)
        p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        acc = p if acc is None else acc + p
    return acc / len(SEEDS)


def main():
    sys.path.insert(0, os.path.join(R, "src"))
    import mne
    from common import load_config, build_montage, build_filters, apply_filters
    mne.set_log_level("ERROR")
    cfg = load_config(os.path.join(R, "configs", "config.yaml"))
    global DS
    DS = cfg["paths"]["edf_root"]
    notches, bp = build_filters(cfg)
    labels = [f"{a}-{b}" for a, b in yaml.safe_load(open(f"{R}/configs/config.yaml"))["channels"]["montage"]]

    f = h5py.File(f"{R}/data/win_eval.h5", "r")
    y = f["y"][:].astype(int); ch = f["chan"][:].astype(int); subj = f["subject"][:]
    fnames = [x.decode() if isinstance(x, bytes) else x for x in f["file_names"][:]]
    P = seed_mean_prob("pLS")

    dur_cache = {}

    def duration(rec):
        if rec not in dur_cache:
            raw = mne.io.read_raw_edf(os.path.join(DS, rec[:-4] + ".edf"), preload=False, verbose="ERROR")
            dur_cache[rec] = raw.n_times / raw.info["sfreq"]
        return dur_cache[rec]

    # one window per class, highest true-class probability, six distinct subjects,
    # events at least EDGE s away from both ends of the recording (filter edge effects)
    chosen, used = {}, set()
    for c in np.argsort([np.sum(y == k) for k in range(6)]):          # rarest class picks first
        idx = np.where((y == c) & ~np.isin(subj, list(used)))[0]
        for w in idx[np.argsort(-P[idx, c], kind="stable")]:
            rec = fnames[f["file"][w]]; t0 = float(f["t0"][w])
            if t0 >= EDGE and t0 + 1.0 <= duration(rec) - EDGE:
                break
        chosen[int(c)] = int(w); used.add(subj[w])

    def page(w):
        rec = fnames[f["file"][w]]; t0 = float(f["t0"][w]); c = int(ch[w])
        line = [l for l in open(os.path.join(DS, rec)).read().split("\n")
                if l.startswith(f"{c},") and abs(float(l.split(",")[1]) - t0) < 1e-3][0]
        e0, e1 = float(line.split(",")[1]), float(line.split(",")[2])
        raw = mne.io.read_raw_edf(os.path.join(DS, rec[:-4] + ".edf"), preload=True, verbose="ERROR")
        if abs(raw.info["sfreq"] - FS) > 1e-6:
            raw.resample(FS, npad="auto")
        Y = apply_filters(build_montage(raw.get_data(), raw.ch_names, cfg["channels"]), notches, bp)
        n = int(PAGE * FS); mid = 0.5 * (e0 + e1)
        a = min(max(int(round((mid - PAGE / 2) * FS)), 0), Y.shape[1] - n)
        return Y[:, a:a + n] * 1e6, e0 - a / FS, e1 - a / FS, rec, line

    fig = plt.figure(figsize=(7.48, 7.6))
    xs = [0.085, 0.405, 0.725]; W = 0.255
    rows = [(0.605, 0.315, 0.52, 0.068), (0.125, 0.315, 0.04, 0.068)]   # (main y, main h, zoom y, zoom h)
    t0all = f["t0"][:]; fileall = f["file"][:]
    for k, c in enumerate(range(6)):
        w = chosen[c]
        # tie-break: the same segment is often annotated for the same class on several channels
        # (identical input, identical probability); show the annotated channel with the largest
        # peak-to-peak amplitude inside the annotated interval.
        ties = np.where((fileall == fileall[w]) & (np.abs(t0all - t0all[w]) < 1e-3) & (y == c))[0]
        if len(ties) > 1:
            seg0, a0, b0, _, _ = page(w)
            m0 = (np.arange(seg0.shape[1]) / FS >= a0) & (np.arange(seg0.shape[1]) / FS <= b0)
            w = int(ties[np.argmax([np.ptp(seg0[ch[t], m0]) for t in ties])])
        seg, s0, s1, rec, line = page(w)
        print(f"{NAMES[c]}: {rec} annotation '{line}' subj {subj[w]} p_true={P[w, c]:.2f}")
        col, (my, mh, zy, zh) = xs[k % 3], rows[k // 3]
        ax = fig.add_axes([col, my, W, mh])
        tt = np.arange(seg.shape[1]) / FS
        ax.axvspan(s0, s1, color="#f3e7b5", alpha=0.75, lw=0, zorder=0)
        for sec in range(1, int(PAGE)):
            ax.axvline(sec, color="#dddddd", lw=0.4, zorder=1)
        for n in range(22):
            hl = n == ch[w]
            ax.plot(tt, seg[n] - n * SPACING, lw=0.55 if hl else 0.32, color=COL[c] if hl else "black",
                    zorder=3 if hl else 2)
            if k % 3 == 0 or hl:
                ax.text(-0.06, -n * SPACING, labels[n], fontsize=4.6, ha="right", va="center",
                        color=COL[c] if hl else "#444444", fontweight="bold" if hl else "normal")
        yb = -22.6 * SPACING
        ax.plot([PAGE - 1.0] * 2, [yb, yb + 100], color="k", lw=0.8)
        ax.plot([PAGE - 1.0, PAGE], [yb, yb], color="k", lw=0.8)
        ax.text(PAGE - 1.05, yb + 50, "100 µV", fontsize=4.8, ha="right", va="center")
        ax.text(PAGE - 0.5, yb - 0.25 * SPACING, "1 s", fontsize=4.8, ha="center", va="top")
        ax.set_xlim(0, PAGE); ax.set_ylim(yb - 0.9 * SPACING, 1.5 * SPACING); ax.axis("off")
        fig.text(col + W / 2, my + mh + 0.006, f"{NAMES[c]} ({FULL[c]})", fontsize=6.3, ha="center", va="bottom",
                 color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL[c], lw=0.8))
        # zoom box on the annotated channel
        z0, z1 = max(s0 - 0.5, 0), min(s1 + 0.5, PAGE)
        az = fig.add_axes([col + 0.03, zy, W - 0.06, zh])
        m = (tt >= z0) & (tt <= z1); sig = seg[ch[w], m]
        az.axvspan(s0, s1, color="#f3e7b5", alpha=0.75, lw=0)
        az.plot(tt[m], sig, color=COL[c], lw=0.8)
        pad = 0.12 * (sig.max() - sig.min() + 1e-6)
        az.set_xlim(z0, z1); az.set_ylim(sig.min() - pad, sig.max() + pad)
        az.set_xticks([]); az.set_yticks([])
        for sp in az.spines.values():
            sp.set_edgecolor(COL[c]); sp.set_linewidth(0.8)
        span = sig.max() - sig.min()
        bar = 50 if span > 120 else (25 if span > 50 else 10)
        x0 = z1 + 0.025 * (z1 - z0)                 # scale bar just outside the zoom box
        az.plot([x0, x0], [sig.min(), sig.min() + bar], color="k", lw=0.7, clip_on=False)
        az.text(x0 + 0.02 * (z1 - z0), sig.min() + bar / 2, f"{bar} µV", fontsize=4.6, ha="left", va="center")
        az.text(z0, sig.max() + pad * 1.15, f"zoom: {labels[ch[w]]}", fontsize=4.8,
                ha="left", va="bottom", color=COL[c], fontweight="bold")
        # box on the main trace + connectors
        yc = -ch[w] * SPACING
        ax.add_patch(Rectangle((z0, yc - 0.9 * SPACING), z1 - z0, 1.8 * SPACING, fill=False,
                               ec=COL[c], lw=0.7, zorder=4))
        for xa in (z0, z1):
            fig.add_artist(ConnectionPatch(xyA=(xa, yc - 0.9 * SPACING), coordsA=ax.transData,
                                           xyB=(xa, sig.max() + pad), coordsB=az.transData,
                                           color=COL[c], lw=0.5, ls=(0, (2, 1.5))))
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"fig_examples.{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("-> fig_examples.png/.pdf")


if __name__ == "__main__":
    main()
