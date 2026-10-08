"""common.py — loader config, montase, filter chain, split subject-disjoint.

Tidak ada path absolut. project_root diturunkan dari lokasi file ini.
"""
import os
import re
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(_HERE)
DEFAULT_CONFIG = os.path.join(DEFAULT_ROOT, "configs", "config.yaml")


def apply_overrides(cfg, overrides):
    """overrides: list "a.b.c=nilai". Nilai diparse sebagai YAML (int/float/bool/str)."""
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for part in parts[:-1]:
            node = node[part]
        assert parts[-1] in node, f"kunci config tak dikenal: {k}"
        val = yaml.safe_load(v)
        # YAML 1.1: "3e-4" diparse sebagai STRING (butuh titik: "3.0e-4").
        # Tanpa koreksi ini, lr/wd masuk ke optimizer sebagai str dan gagal.
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


# ----------------------------------------------------------------- montase ---
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
    """data (n_edf_ch, T) -> (22, T) bipolar TCP. Raise bila elektroda kurang."""
    norm = make_normalizer(ch_cfg)
    idx = {}
    for i, nm in enumerate(ch_names):
        e = norm(nm)
        if e not in idx:          # kemunculan pertama menang
            idx[e] = i
    pairs = montage_pairs(ch_cfg)
    missing = sorted({e for p in pairs for e in p} - set(idx))
    if missing:
        raise KeyError(f"elektroda hilang: {missing}")
    return np.stack([data[idx[a]] - data[idx[c]] for a, c in pairs], axis=0)


# ------------------------------------------------------------------ filter ---
def build_filters(cfg):
    """Kembalikan (list_sos_notch, sos_bandpass) siap pakai pada fs config."""
    from scipy.signal import butter, iirnotch, tf2sos
    fs = cfg["windowing"]["fs"]
    f = cfg["filtering"]
    notches = []
    if f["notch"]["enabled"]:
        assert f["notch"]["method"] == "iirnotch", f["notch"]["method"]
        for f0 in f["notch"]["freqs"]:
            if f0 >= fs / 2:                      # harmonik di atas Nyquist
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
    """x (C, T) float64 -> difilter zero-phase. Notch dulu, baru bandpass."""
    from scipy.signal import sosfiltfilt
    y = np.asarray(x, dtype=np.float64)
    for sos in notches:
        y = sosfiltfilt(sos, y, axis=-1)
    if bp is not None:
        y = sosfiltfilt(bp, y, axis=-1)
    return y


def recording_stats(y, cfg):
    """Statistik robust per kanal pada rekaman PENUH terfilter -> (med, scale)."""
    rs = cfg["recording_stats"]
    med = np.median(y, axis=-1)
    q75, q25 = np.percentile(y, [75, 25], axis=-1)
    scale = (q75 - q25) / rs["iqr_to_sigma"] + rs["eps"]
    return med.astype(np.float32), scale.astype(np.float32)


# ------------------------------------------------------------------- split ---
def subject_val_split(subjects, labels, subj_of_window, cfg, n_classes=6):
    """Split subject-disjoint; mode dipilih lewat cfg["split"]["mode"].

    mode="alphabetical_tail" mereplikasi split RESMI CBraMod persis
    (preprocessing_tuev.py:207-214): nama subjek diurut alfabetis, 80% pertama
    latih, 20% terakhir val. Wajib dipakai bila angkanya akan dijejerkan dengan
    tabel CBraMod/CSBrain -- split yang berbeda membuat perbandingan tak sah.

    mode="iterative" (bawaan) memakai iterative stratification di bawah ini.
    """
    mode = cfg["split"].get("mode", "iterative")
    if mode == "alphabetical_tail":
        order = np.argsort(np.asarray(subjects).astype(str), kind="stable")
        n_tr = int(len(order) * (1.0 - cfg["split"]["val_fraction"]))
        return order[:n_tr], order[n_tr:]
    assert mode == "iterative", mode
    return _subject_val_split_iterative(subjects, labels, subj_of_window, cfg, n_classes)


def _subject_val_split_iterative(subjects, labels, subj_of_window, cfg, n_classes=6):
    """Iterative stratification (Sechidis dkk. 2011).

    Granularitasnya subjek, bukan window, jadi kuota per kelas tidak akan pernah
    tepat. Algoritma ini meminimalkan penyimpangan dengan selalu menangani kelas
    yang sisa instansinya paling sedikit lebih dulu — itu yang melindungi SPSW.

    subjects: array nama subjek unik. subj_of_window: indeks subjek per window.
    Kembalikan (idx_subjek_train, idx_subjek_val).
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
        if len(pos) == 0:                      # sisa subjek tanpa kelas apa pun
            for s in np.where(left)[0]:
                k = int(np.argmax(n_need))
                assign[s] = k; n_need[k] -= 1; left[s] = False
            break
        c = pos[np.argmin(cnt[pos])]           # kelas dengan sisa paling sedikit
        cand = np.where(left & (pres[:, c] > 0))[0]
        # subjek dengan kontribusi terbesar ditangani lebih dulu (paling mengikat)
        cand = cand[np.lexsort((rng.rand(len(cand)), -pres[cand, c]))]
        for s in cand:
            k = int(np.argmax(need[:, c]))
            if need[0, c] == need[1, c]:       # seri -> lihat kuota subjek global
                k = int(np.argmax(n_need))
            assign[s] = k
            need[k] -= pres[s]
            n_need[k] -= 1
            left[s] = False

    tr = np.where(assign == 0)[0]
    va = np.where(assign == 1)[0]
    return tr, va
