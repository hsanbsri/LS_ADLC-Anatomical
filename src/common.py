"""common.py: config loader, bipolar montage, filter chain, subject-disjoint split.

No absolute paths: project_root is derived from the location of this file.
"""
import os
import re
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(_HERE)
DEFAULT_CONFIG = os.path.join(DEFAULT_ROOT, "configs", "config.yaml")


def apply_overrides(cfg, overrides):
    """overrides: list of "a.b.c=value". Values are parsed as YAML (int/float/bool/str)."""
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for part in parts[:-1]:
            node = node[part]
        assert parts[-1] in node, f"unknown config key: {k}"
        val = yaml.safe_load(v)
        # YAML 1.1 parses "3e-4" as a STRING (it needs a dot: "3.0e-4").
        # Without this correction, lr/wd would reach the optimizer as str and fail.
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        node[parts[-1]] = val
    return cfg


def load_config(path=None, project_root=None, overrides=None):
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    root = os.path.abspath(project_root or cfg["paths"].get("project_root") or DEFAULT_ROOT)
    cfg["paths"]["project_root"] = root
    for k in ("dataset_root", "cache_dir", "logs_dir", "results_dir", "ckpt_dir"):
        v = cfg["paths"][k]
        cfg["paths"][k] = v if os.path.isabs(v) else os.path.abspath(os.path.join(root, v))
    cfg["paths"]["edf_root"] = os.path.join(cfg["paths"]["dataset_root"], cfg["paths"]["edf_subdir"])
    for k in ("cache_dir", "logs_dir", "results_dir", "ckpt_dir"):
        os.makedirs(cfg["paths"][k], exist_ok=True)
    cfg["labels"]["code_to_class"] = {int(k): int(v) for k, v in cfg["labels"]["code_to_class"].items()}
    return cfg


def sig_len(cfg):
    return int(round(cfg["windowing"]["window_sec"] * cfg["windowing"]["fs"]))


# ----------------------------------------------------------------- montage ---
def make_normalizer(ch):
    pre = re.compile(ch["strip_prefix_regex"], re.I)
    suf = re.compile(ch["strip_suffix_regex"], re.I)
    alias = {k.upper(): v.upper() for k, v in (ch.get("aliases") or {}).items()}

    def norm(name):
        s = suf.sub("", pre.sub("", name.strip().upper())).strip()
        return alias.get(s, s)
    return norm


def montage_pairs(ch):
    pairs = [(a.upper(), c.upper()) for a, c in ch["montage"]]
    assert len(pairs) == ch["n_channels"], f"{len(pairs)} != {ch['n_channels']}"
    return pairs


def build_montage(data, ch_names, ch_cfg):
    """data (n_edf_ch, T) -> (22, T) bipolar TCP. Raises if an electrode is missing."""
    norm = make_normalizer(ch_cfg)
    idx = {}
    for i, nm in enumerate(ch_names):
        e = norm(nm)
        if e not in idx:          # first occurrence wins
            idx[e] = i
    pairs = montage_pairs(ch_cfg)
    missing = sorted({e for p in pairs for e in p} - set(idx))
    if missing:
        raise KeyError(f"missing electrodes: {missing}")
    return np.stack([data[idx[a]] - data[idx[c]] for a, c in pairs], axis=0)


# ------------------------------------------------------------------ filter ---
def build_filters(cfg):
    """Return (list_of_notch_sos, bandpass_sos), ready to use at the configured fs."""
    from scipy.signal import butter, iirnotch, tf2sos
    fs = cfg["windowing"]["fs"]
    f = cfg["filtering"]
    notches = []
    if f["notch"]["enabled"]:
        assert f["notch"]["method"] == "iirnotch", f["notch"]["method"]
        for f0 in f["notch"]["freqs"]:
            if f0 >= fs / 2:                      # harmonic above Nyquist
                continue
            b, a = iirnotch(f0, f["notch"]["quality"], fs=fs)
            notches.append(tf2sos(b, a))
    bp = None
    if f["bandpass"]["enabled"]:
        assert f["bandpass"]["filter"] == "butter", f["bandpass"]["filter"]
        bp = butter(f["bandpass"]["order"],
                    [f["bandpass"]["low_hz"], f["bandpass"]["high_hz"]],
                    "bandpass", fs=fs, output=f["bandpass"]["output"])
    return notches, bp


def apply_filters(x, notches, bp):
    """x (C, T) float64 -> zero-phase filtered. Notch filters first, then band-pass."""
    from scipy.signal import sosfiltfilt
    y = np.asarray(x, dtype=np.float64)
    for sos in notches:
        y = sosfiltfilt(sos, y, axis=-1)
    if bp is not None:
        y = sosfiltfilt(bp, y, axis=-1)
    return y


def recording_stats(y, cfg):
    """Robust per-channel statistics over the WHOLE filtered recording -> (med, scale)."""
    rs = cfg["recording_stats"]
    med = np.median(y, axis=-1)
    q75, q25 = np.percentile(y, [75, 25], axis=-1)
    scale = (q75 - q25) / rs["iqr_to_sigma"] + rs["eps"]
    return med.astype(np.float32), scale.astype(np.float32)


# ------------------------------------------------------------------- split ---
def subject_val_split(subjects, labels, subj_of_window, cfg, n_classes=6):
    """Subject-disjoint split; the mode is chosen with cfg["split"]["mode"].

    mode="alphabetical_tail" (used in the paper): subject names are sorted
    alphabetically, the first 80% are used for gradient training and the last
    20% for internal validation.

    mode="iterative" (default) uses the iterative stratification below.
    """
    mode = cfg["split"].get("mode", "iterative")
    if mode == "alphabetical_tail":
        order = np.argsort(np.asarray(subjects).astype(str), kind="stable")
        n_tr = int(len(order) * (1.0 - cfg["split"]["val_fraction"]))
        return order[:n_tr], order[n_tr:]
    assert mode == "iterative", mode
    return _subject_val_split_iterative(subjects, labels, subj_of_window, cfg, n_classes)


def _subject_val_split_iterative(subjects, labels, subj_of_window, cfg, n_classes=6):
    """Iterative stratification (Sechidis et al. 2011).

    The granularity is the subject, not the window, so per-class quotas can
    never be met exactly. The algorithm minimizes the deviation by always
    handling the class with the fewest remaining instances first, which
    protects the rarest class.

    subjects: array of unique subject names. subj_of_window: subject index of each window.
    Returns (train_subject_idx, val_subject_idx).
    """
    rng = np.random.RandomState(cfg["split"]["seed"])
    n_subj = len(subjects)
    pres = np.zeros((n_subj, n_classes), dtype=np.float64)
    np.add.at(pres, (subj_of_window, labels), 1)

    frac = cfg["split"]["val_fraction"]
    need = np.stack([pres.sum(0) * (1 - frac), pres.sum(0) * frac])   # (2, n_classes)
    n_need = np.array([n_subj * (1 - frac), n_subj * frac])
    assign = np.full(n_subj, -1, dtype=int)
    left = np.ones(n_subj, dtype=bool)

    while left.any():
        rem = pres[left]
        cnt = rem.sum(0)
        pos = np.where(cnt > 0)[0]
        if len(pos) == 0:                      # remaining subjects have no class at all
            for s in np.where(left)[0]:
                k = int(np.argmax(n_need))
                assign[s] = k; n_need[k] -= 1; left[s] = False
            break
        c = pos[np.argmin(cnt[pos])]           # class with the fewest remaining instances
        cand = np.where(left & (pres[:, c] > 0))[0]
        # subjects contributing most are handled first (most constraining)
        cand = cand[np.lexsort((rng.rand(len(cand)), -pres[cand, c]))]
        for s in cand:
            k = int(np.argmax(need[:, c]))
            if need[0, c] == need[1, c]:       # tie -> use the global subject quota
                k = int(np.argmax(n_need))
            assign[s] = k
            need[k] -= pres[s]
            n_need[k] -= 1
            left[s] = False

    tr = np.where(assign == 0)[0]
    va = np.where(assign == 1)[0]
    return tr, va
