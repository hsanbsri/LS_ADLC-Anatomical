"""features.py — two normalization arms + DWT subband features.

Arm A  per_window_zscore     : (x - mean_w) / (std_w + eps), per channel and window.
                               Absolute amplitude is removed. Used in the paper.
Arm B  per_recording_robust  : (x - median_rec) / (IQR_rec/1.349 + eps).
                               Statistics come from the recording itself, without
                               any label, so there is no leakage. The amplitude of a
                               spike relative to the recording background is kept.

DWT features: db4, level 6 at 250 Hz -> 7 subbands
    A6 0-1.95 | D6 1.95-3.9 | D5 3.9-7.8 | D4 7.8-15.6
    D3 15.6-31.25 | D2 31.25-62.5 | D1 62.5-125 Hz
The 0.5-80 Hz band-pass keeps real signal in all seven subbands (with a 40 Hz
low-pass, D1 would be empty and D2 would only cover 31-40 Hz).
"""
import numpy as np
import pywt

SUBBANDS = ["A6", "D6", "D5", "D4", "D3", "D2", "D1"]


def normalize(X, med, scale, arm, cfg):
    """X (n, C, T) float32; med/scale (n, C). -> normalized, clipped float32."""
    mode = cfg["normalization"]["arms"][arm]["mode"]
    eps = cfg["recording_stats"]["eps"]
    if mode == "per_window_zscore":
        mu = X.mean(axis=-1, keepdims=True)
        sd = X.std(axis=-1, keepdims=True) + eps
        Z = (X - mu) / sd
    elif mode == "per_recording_robust":
        Z = (X - med[..., None]) / (scale[..., None] + eps)
    else:
        raise ValueError(f"unknown arm: {mode}")
    c = cfg["normalization"]["clip_sigma"]
    return np.clip(Z, -c, c, out=Z).astype(np.float32)


def dwt_features(Z, cfg):
    """Z (n, C, T) normalized -> (n, C, n_subband * n_stats) float32."""
    wv = cfg["wavelet"]
    eps = wv["eps"]
    n, C, T = Z.shape
    flat = Z.reshape(n * C, T).astype(np.float64)
    coeffs = pywt.wavedec(flat, wv["family"], level=wv["level"], axis=-1)
    assert len(coeffs) == wv["n_subbands"], (len(coeffs), wv["n_subbands"])

    feats = []
    for c in coeffs:
        col = []
        for st in wv["stats"]:
            if st == "log_energy":
                col.append(np.log((c ** 2).mean(-1) + eps))
            elif st == "log_max_abs":
                col.append(np.log(np.abs(c).max(-1) + eps))
            elif st == "kurtosis":
                mu = c.mean(-1, keepdims=True)
                d = c - mu
                m2 = (d ** 2).mean(-1)
                m4 = (d ** 4).mean(-1)
                col.append(m4 / (m2 ** 2 + eps) - 3.0)
            else:
                raise ValueError(f"unknown statistic: {st}")
        feats.append(np.stack(col, -1))            # (n*C, n_stats)
    F = np.concatenate(feats, -1)                   # (n*C, n_sub*n_stats)
    return np.nan_to_num(F.reshape(n, C, -1), nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


def periodicity_features(Z, cfg):
    """Z (n,C,T) normalized -> (n,C,3). Optional; disabled in the paper.

    The smoothed envelope |x| is mean-removed and its power spectrum taken. In
    the discharge-rate band (0.5-3 Hz) the peak power, peak frequency and
    periodicity index (peak power / total envelope power) are returned.
    """
    pc = cfg["periodicity"]
    fs = cfg["windowing"]["fs"]
    n, C, T = Z.shape
    k = max(1, int(pc["smooth_ms"] * fs / 1000))
    ker = np.ones(k, dtype=np.float32) / k
    env = np.abs(Z).reshape(n * C, T)
    # 'same' convolution of every row
    env = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), -1, env)
    env -= env.mean(-1, keepdims=True)
    P = np.abs(np.fft.rfft(env, axis=-1)) ** 2
    fr = np.fft.rfftfreq(T, 1 / fs)
    lo, hi = pc["band_hz"]
    band = (fr >= lo) & (fr <= hi)
    Pb = P[:, band]; frb = fr[band]
    pk = Pb.argmax(-1)
    peak = np.take_along_axis(Pb, pk[:, None], -1)[:, 0]
    tot = P[:, fr > 0].sum(-1) + 1e-12
    F = np.stack([np.log(peak + 1e-12), frb[pk], peak / tot], -1)
    return np.nan_to_num(F.reshape(n, C, 3), nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


def laterality_features(F_dwt, cfg):
    """GLOBAL features from the per-channel subband energies. Optional; disabled in the paper.

    F_dwt (n, C, n_sub*n_stats); the first statistic of each subband is log_energy.
    -> (n, n_sub + 2 + n_pairs): laterality index (LI) per subband, broadband LI,
       maximum |LI| over symmetric pairs, then the LI of each symmetric pair.
    """
    lc = cfg["laterality"]
    ns = cfg["wavelet"]["n_subbands"]; nst = len(cfg["wavelet"]["stats"])
    eps = lc["eps"]
    E = np.exp(F_dwt[:, :, 0::nst].astype(np.float64))      # (n, C, n_sub)
    L = E[:, lc["left"], :].sum(1); R = E[:, lc["right"], :].sum(1)
    li_band = (L - R) / (L + R + eps)                        # (n, n_sub)
    Eb = E.sum(-1)                                           # broadband per channel
    li_bb = ((Eb[:, lc["left"]].sum(1) - Eb[:, lc["right"]].sum(1))
             / (Eb[:, lc["left"]].sum(1) + Eb[:, lc["right"]].sum(1) + eps))
    pr = np.stack([(Eb[:, a] - Eb[:, b]) / (Eb[:, a] + Eb[:, b] + eps)
                   for a, b in lc["pairs"]], -1)             # (n, n_pairs)
    out = np.concatenate([li_band, li_bb[:, None], np.abs(pr).max(1, keepdims=True), pr], -1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feat_dim(cfg):
    d = cfg["wavelet"]["n_subbands"] * len(cfg["wavelet"]["stats"])
    if cfg.get("periodicity", {}).get("enabled"):
        d += len(cfg["periodicity"]["stats"])
    return d


def glob_dim(cfg):
    if not cfg.get("laterality", {}).get("enabled"):
        return 0
    return cfg["wavelet"]["n_subbands"] + 2 + len(cfg["laterality"]["pairs"])
