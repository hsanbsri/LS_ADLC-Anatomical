"""train.py — train one model (one normalization arm, one seed) on the cached windows.

Split discipline (cannot be overridden from the config):
  - the official evaluation pool is used ONCE, at the end, for reporting only.
  - the internal validation split is carved SUBJECT-DISJOINTLY from the training
    pool; all model selection (early stopping, best epoch) uses it.
  - no statistic of any kind flows from validation/evaluation into training.

Plain cross-entropy, no class weighting and no re-balancing. LS-ADLC (paper,
Algorithm 2) is enabled with --lsadlc adlc and the out-of-fold k-NN file of oof_encoder.py.
"""
import argparse, json, os, sys, time
import h5py, numpy as np, torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                             confusion_matrix, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, sig_len, subject_val_split
from features import feat_dim, glob_dim
from model import build_model


def metrics(y, p, names):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "cohen_kappa": float(cohen_kappa_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "accuracy": float((y == p).mean()),
        "per_class_f1": {n: float(v) for n, v in zip(
            names, f1_score(y, p, average=None, labels=range(len(names)), zero_division=0))},
        "per_class_sensitivity": {n: float(v) for n, v in zip(
            names, recall_score(y, p, average=None, labels=range(len(names)), zero_division=0))},
        "support": {n: int(v) for n, v in zip(names, np.bincount(y, minlength=len(names)))},
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(names))).tolist(),
    }


def load_split(cfg, split, arm, tag=""):
    d = cfg["paths"]["cache_dir"]
    tag = f"_{tag}" if tag else ""
    h = h5py.File(os.path.join(d, f"win_{split}{tag}.h5"), "r")
    npy = os.path.join(d, f"X_{split}{tag}.npy")
    if cfg["runtime"].get("mmap", False) and os.path.exists(npy):
        X = np.load(npy, mmap_mode="r")              # shared through the OS page cache
    else:
        X = h["X"][:]                                # full copy into memory
    out = dict(X=X, y=h["y"][:].astype(np.int64), subj=h["subject"][:].astype(np.int64),
               chan=h["chan"][:].astype(np.int64),
               med=h["rec_med"][:], scale=h["rec_scale"][:],
               subjects=[s.decode() for s in h["subject_names"][:]])
    h.close()
    out["F"] = np.load(os.path.join(d, f"feat_{split}{tag}_{arm}.npy"))
    assert len(out["F"]) == len(out["X"])
    # Feature files always contain the periodicity columns; when the periodicity
    # features are disabled, keep only the DWT columns.
    nd = cfg["wavelet"]["n_subbands"] * len(cfg["wavelet"]["stats"])
    if not cfg.get("periodicity", {}).get("enabled", False):
        out["F"] = np.ascontiguousarray(out["F"][:, :, :nd])
    gp = os.path.join(d, f"glob_{split}{tag}_{arm}.npy")
    out["G"] = (np.load(gp) if (os.path.exists(gp)
                and cfg.get("laterality", {}).get("enabled", False)) else None)
    assert out["F"].shape[2] == feat_dim(cfg), (out["F"].shape, feat_dim(cfg))
    return out


class GPUNorm:
    """Arm normalization on the GPU, per batch: low RAM use, deterministic."""
    def __init__(self, cfg, arm, dev):
        self.mode = cfg["normalization"]["arms"][arm]["mode"]
        self.clip = float(cfg["normalization"]["clip_sigma"])
        self.eps = float(cfg["recording_stats"]["eps"])
        self.dev = dev

    def __call__(self, x, med, scale):
        if self.mode == "per_window_zscore":
            z = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)
        else:
            z = (x - med[..., None]) / (scale[..., None] + self.eps)
        return z.clamp_(-self.clip, self.clip)


class ChannelDropout:
    """Zero n random channels per sample, in the signal AND in its DWT feature rows.

    Active during training only. The queried channel (channel-query variant, off
    in the paper) is protected when protect_query=true so that its label stays defined.
    """
    def __init__(self, cfg):
        c = cfg["training"]["augment"]["channel_dropout"]
        self.on = bool(c["enabled"])
        self.p = float(c["p"])
        self.n = int(c["n_channels"])
        self.protect = bool(c["protect_query"])

    def __call__(self, x, f, ch=None):
        if not self.on:
            return x, f
        B, C, _ = x.shape
        hit = torch.rand(B, device=x.device) < self.p
        if not hit.any():
            return x, f
        score = torch.rand(B, C, device=x.device)
        if self.protect and ch is not None:
            score.scatter_(1, ch[:, None], 2.0)       # highest score -> never selected
        drop = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        drop.scatter_(1, score.argsort(dim=1)[:, :self.n], True)
        drop &= hit[:, None]
        keep = (~drop).to(x.dtype)[:, :, None]
        return x * keep, f * keep



# ---- optional data cache in VRAM -------------------------------------
# Moves the large arrays to the GPU ONCE at the start, so the training loop no
# longer copies host->device every batch. The values read are IDENTICAL; only
# their memory location differs -> bit-identical results (verified).
_GPU_KEYS = ("X", "F", "med", "scale", "chan", "G")


def to_gpu_cache(D, dev, enable, reserve_gb=3.5):
    if not enable or not torch.cuda.is_available():
        return D, False
    need = sum(D[k].nbytes for k in _GPU_KEYS
               if D.get(k) is not None and hasattr(D[k], "nbytes"))
    free, _ = torch.cuda.mem_get_info()
    if need + reserve_gb * 1e9 > free:
        print(f"  gpu_cache SKIPPED: needs {need/1e9:.2f} GB + reserve "
              f"{reserve_gb:.1f} GB, free VRAM {free/1e9:.2f} GB")
        return D, False
    G = dict(D)
    for k in _GPU_KEYS:
        v = D.get(k)
        if v is not None and hasattr(v, "dtype"):
            G[k] = torch.as_tensor(np.ascontiguousarray(v)).to(dev)
    print(f"  gpu_cache ACTIVE: {need/1e9:.2f} GB of data placed in VRAM "
          f"(free before: {free/1e9:.2f} GB)")
    return G, True


def take(D, k, j, dev, jt):
    """Fetch a batch from D[k]; jt != None means D[k] is already a GPU tensor."""
    v = D.get(k)
    if v is None:
        return None
    if jt is not None:
        return v.index_select(0, jt)
    return torch.from_numpy(v[j]).to(dev, non_blocking=True)


@torch.no_grad()
def predict(model, data, idx, norm, bs, dev, amp, n_cls=6):
    model.eval()
    out = np.empty(len(idx), dtype=np.int64)
    prob = np.empty((len(idx), n_cls), dtype=np.float32)   # binary tasks need 2 columns, not 6
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        jt = (torch.as_tensor(j, device=dev)
              if torch.is_tensor(data["X"]) else None)
        x = take(data, "X", j, dev, jt)
        f = take(data, "F", j, dev, jt)
        m = take(data, "med", j, dev, jt)
        sc = take(data, "scale", j, dev, jt)
        c = take(data, "chan", j, dev, jt)
        gg = take(data, "G", j, dev, jt)
        with torch.autocast("cuda", dtype=amp):
            lo = model(norm(x, m, sc), f, ch=c, g=gg)
        p = lo.float().softmax(-1)
        prob[s:s + bs] = p.cpu().numpy()
        out[s:s + bs] = p.argmax(-1).cpu().numpy()
    return out, prob


@torch.no_grad()
def predict_gates(model, data, idx, norm, bs, dev, amp):
    """Per-sample gate weights (B, 3) of the optional chain_expert model (not used in the paper)."""
    model.eval()
    G = np.empty((len(idx), 3), dtype=np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        jt = (torch.as_tensor(j, device=dev) if torch.is_tensor(data["X"]) else None)
        x = take(data, "X", j, dev, jt)
        f = take(data, "F", j, dev, jt)
        m = take(data, "med", j, dev, jt)
        sc = take(data, "scale", j, dev, jt)
        with torch.autocast("cuda", dtype=amp):
            model(norm(x, m, sc), f)
        G[s:s + bs] = model.last_gate.float().cpu().numpy()
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--variant", default="baseline", choices=["baseline", "chanquery"])
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--adlc-synth", default=None,
                    help="npz of signal-space synthetic rows; added to the training split ONLY")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--oof", default="oof_encoder.npz",
                    help="OOF file of the alternative 'pair' gate (relative to cache_dir)")
    ap.add_argument("--allow-oof-mismatch", action="store_true",
                    help="allow an OOF file that does not cover every training row")
    ap.add_argument("--lsadlc", default=None, choices=["pair", "none", "adlc"],
                    help="enable LS-ADLC during training; selects the gate (paper: adlc)")
    ap.add_argument("--binary", action="store_true",
                    help="stage 1 of an optional two-stage scheme: binary epileptiform vs. not (not used in the paper)")
    ap.add_argument("--init-from", default=None,
                    help="stage 2: load stage-1 weights; the classification head is re-initialized")
    ap.add_argument("--monitor", default=None,
                    choices=["balanced_accuracy", "macro_f1", "cohen_kappa"])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, overrides=a.overrides)
    cfg["model"]["channel_query"]["enabled"] = (a.variant == "chanquery")
    if a.monitor:
        cfg["training"]["early_stop"]["monitor"] = a.monitor
    if a.lsadlc:
        cfg["lsadlc_train"]["enabled"] = True
        cfg["lsadlc_train"]["gate"] = a.lsadlc
    a.tag = a.tag or a.variant
    tr_cfg = cfg["training"]
    seed = a.seed if a.seed is not None else tr_cfg["seeds"][0]
    epochs = a.epochs or tr_cfg["epochs"]
    names = cfg["labels"]["class_names"]
    run = f"{a.tag}_arm{a.arm}_seed{seed}"
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}[tr_cfg["amp_dtype"]]

    torch.manual_seed(seed); np.random.seed(seed)
    if cfg["runtime"].get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    else:
        torch.backends.cudnn.benchmark = True

    t0 = time.time()
    print(f"[{run}] loading cache ...", flush=True)
    D = load_split(cfg, "train", a.arm, a.cache_tag)
    E = load_split(cfg, "eval", a.arm, a.cache_tag)
    print(f"  train {D['X'].shape}  eval {E['X'].shape}  {time.time()-t0:.0f}s", flush=True)

    # ---- split subject-disjoint train/val -----------------------------------
    s_tr, s_va = subject_val_split(np.array(D["subjects"]), D["y"], D["subj"], cfg,
                                    n_classes=cfg["model"]["n_classes"])
    in_val = np.isin(D["subj"], s_va)
    idx_tr = np.where(~in_val)[0]
    idx_va = np.where(in_val)[0]
    assert not (set(D["subj"][idx_tr]) & set(D["subj"][idx_va])), "SPLIT BOCOR"
    print(f"  subjects train {len(s_tr)} / val {len(s_va)} (disjointness verified)")
    n_cache_rows = len(D["y"])          # before any synthetic rows are added
    if a.adlc_synth:
        S = np.load(a.adlc_synth, allow_pickle=True)
        n0 = len(D["y"])
        for k in ("X", "F", "y", "chan", "med", "scale", "subj"):
            src = S["subject"] if k == "subj" else S[k]
            D[k] = np.concatenate([D[k], src.astype(D[k].dtype)])
        assert D.get("G") is None, "signal-space ADLC does not support global features yet"
        idx_tr = np.concatenate([idx_tr, np.arange(n0, len(D["y"]))])
        print(f"  signal-space ADLC: +{len(D['y'])-n0} synthetic rows (sigma={float(S['sigma'])}) "
              f"-> train {len(idx_tr)}")
    print(f"  window  train {len(idx_tr)} / val {len(idx_va)}")
    print(f"  seleksi model: val {tr_cfg['early_stop']['monitor']} "
          f"(min_epochs={tr_cfg['early_stop']['min_epochs']})")
    for c, n in enumerate(names):
        print(f"    {n:5s} train={np.sum(D['y'][idx_tr]==c):6d}  val={np.sum(D['y'][idx_va]==c):5d}")

    if a.binary:
        # epileptiform = SPSW(0)/GPED(1)/PLED(2) -> 1 ; EYEM/ARTF/BCKG -> 0
        D["y"] = (D["y"] < 3).astype(np.int64)
        E["y"] = (E["y"] < 3).astype(np.int64)
        cfg["model"]["n_classes"] = 2
        names = ["non-epi", "epi"]
        print(f"  STAGE 1 BINARY: positives {int(D['y'][idx_tr].sum())} / {len(idx_tr)} training rows",
              flush=True)

    D, _gpu_cached = to_gpu_cache(D, dev, cfg["runtime"].get("gpu_cache", False))
    gd = glob_dim(cfg) if D.get("G") is not None else 0
    model = build_model(cfg, cfg["channels"]["n_channels"], feat_dim(cfg), gd).to(dev)

    if a.init_from:
        # Stage 2: inherit the WHOLE encoder from the binary stage; ONLY the head is new.
        sd = torch.load(a.init_from, map_location="cpu")["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        miss = model.load_state_dict(sd, strict=False)
        assert all(k.startswith("head.") for k in miss.missing_keys), \
            f"keys other than the head are missing: {[k for k in miss.missing_keys if not k.startswith('head.')][:5]}"
        assert not miss.unexpected_keys, miss.unexpected_keys
        print(f"  STAGE 2: encoder loaded from {os.path.basename(a.init_from)}; "
              f"classification head re-initialized ({len(miss.missing_keys)} tensors)", flush=True)

    # ---- LS-ADLC during training (the encoder is NOT frozen) ------------------
    lat_cfg = cfg["lsadlc_train"]
    gate_full = lat_aug = None
    if lat_cfg["enabled"]:
        import latent_ops, lsadlc_v2 as L2
        assert lat_cfg["point"] == "B", lat_cfg["point"]
        if lat_cfg["gate"] == "pair":
            oof_path = os.path.join(cfg["paths"]["cache_dir"], a.oof)
            oof = np.load(oof_path)
            gp, _ = L2.gate_pair(torch.from_numpy(oof["prob"]).to(dev),
                                 torch.from_numpy(oof["y"].astype(np.int64)).to(dev))
            gate_full = np.zeros(len(D["y"]), dtype=bool)
            gate_full[oof["idx_in_train_cache"]] = gp
            assert (D["y"][oof["idx_in_train_cache"]] == oof["y"]).all(), "OOF file not aligned"
            # The OOF file MUST cover every training row. If it was computed on a
            # DIFFERENT split, some training rows would never have a score and the
            # gate would be blind to them; this is checked explicitly instead of
            # failing silently.
            # (Training rows without a score are reported below.)
            real = idx_tr[idx_tr < n_cache_rows]      # synthetic rows have no OOF score by design
            has = np.isin(real, oof["idx_in_train_cache"])
            if not has.all():
                msg = (f"OOF {a.oof} covers only {100*has.mean():.1f}% of the training rows "
                       f"({int((~has).sum())} rows without an OOF score). The OOF split does not "
                       f"match split.mode={cfg['split']['mode']}/"
                       f"val_fraction={cfg['split']['val_fraction']}.")
                if a.allow_oof_mismatch:
                    print(f"  PERINGATAN: {msg}")
                else:
                    raise AssertionError(msg + " Use --allow-oof-mismatch if this is intended.")
        if lat_cfg["gate"] == "adlc":
            ac = lat_cfg["adlc"]
            kf = np.load(os.path.join(cfg["paths"]["cache_dir"], ac["knn_file"]))
            kj = list(kf["knn_grid"]).index(int(ac["k"]))
            conf = (kf["knn_w"] if ac["weighted"] else kf["knn_prop"])[:, kj]
            ky = kf["y"].astype(np.int64); kidx = kf["idx_in_train_cache"]
            assert (D["y"][kidx] == ky).all(), "k-NN file not aligned"
            real = idx_tr[idx_tr < n_cache_rows]
            has = np.isin(real, kidx)
            assert has.all(), (f"k-NN file {ac['knn_file']} covers only "
                               f"{100*has.mean():.1f}% of the training rows -- split mismatch")
            lowv = np.full(len(ky), float(ac.get("threshold_low", 0.0)))
            nmin = int(ac.get("tau_low_min_borderline", 0))
            if nmin > 0:   # classes with few borderline rows: drop ONLY outliers (c = 0); off in the paper
                for cc in range(len(names)):
                    if cc in ac["exclude_classes"]: continue
                    nb = int(((conf >= lowv) & (conf < float(ac["threshold"])) & (ky == cc)).sum())
                    if nb < nmin:
                        lowv[ky == cc] = 1e-6
                        print(f"  V-TAU: class {names[cc]} borderline {nb} < {nmin} -> lower bound 1e-6")
            g_sel = ((conf < float(ac["threshold"])) & (conf >= lowv)
                     & ~np.isin(ky, ac["exclude_classes"]))
            if ac["control_random"]:
                # CONTROL (random gate): per-class counts IDENTICAL to the gate, rows drawn at random
                rs = np.random.RandomState(seed + 999); g_rand = np.zeros_like(g_sel)
                for cc in np.unique(ky[g_sel]):
                    pool = np.where(ky == cc)[0]
                    g_rand[rs.choice(pool, int((g_sel & (ky == cc)).sum()), replace=False)] = True
                g_sel = g_rand
            gate_full = np.zeros(len(D["y"]), dtype=bool)
            gate_full[kidx] = g_sel
            per_c = {names[cc]: int((gate_full & (D["y"] == cc)).sum()) for cc in range(len(names))}
            print(f"  ADLC gate ({'RANDOM control' if ac['control_random'] else 'OOF latent k-NN'}"
                  f", K={ac['k']}, threshold=[{ac.get('threshold_low', 0.0)},{ac['threshold']})): {per_c}")
        gen = torch.Generator(device=dev); gen.manual_seed(seed + 777)
        ops_cfg = lat_cfg["ops"]

        def make_lat_aug(mask_t):
            def fn(z):
                za = latent_ops.apply_chain(z, ops_cfg, gen)
                return za if mask_t is None else torch.where(mask_t[:, None], za, z)
            return fn
        n_gate = int(gate_full.sum()) if gate_full is not None else len(idx_tr)
        print(f"  LS-ADLC training: point {lat_cfg['point']}, gate={lat_cfg['gate']}, "
              f"{n_gate} gated rows ({100*n_gate/len(idx_tr):.2f}% of train)")
    nper = len(cfg["periodicity"]["stats"]) if cfg["periodicity"]["enabled"] else 0
    print(f"  features: {feat_dim(cfg)}/channel ({feat_dim(cfg)-nper} DWT"
          + (f" + {nper} periodisitas" if nper else "")
          + f") + {gd} global" + (" (lateralitas)" if gd else ""))
    norm = GPUNorm(cfg, a.arm, dev)
    aug = ChannelDropout(cfg)
    cd = cfg["training"]["augment"]["channel_dropout"]
    print(f"  channel_dropout augmentation: {'ON' if cd['enabled'] else 'off'}"
          + (f" (p={cd['p']}, n={cd['n_channels']}/{cfg['channels']['n_channels']})"
             if cd["enabled"] else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    bs = tr_cfg["batch_size"]
    steps = int(np.ceil(len(idx_tr) / bs))
    warm = tr_cfg["warmup_epochs"] * steps
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1)))))
    assert tr_cfg["loss"] == "ce", tr_cfg["loss"]
    crit = nn.CrossEntropyLoss(label_smoothing=tr_cfg["label_smoothing"])

    best = {"score": -1, "epoch": -1}
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], f"{run}.pt")
    hist, rng = [], np.random.RandomState(seed)
    adlc_on = lat_cfg["enabled"] and lat_cfg["gate"] == "adlc"
    if adlc_on:
        ac = lat_cfg["adlc"]; zdim = cfg["model"]["latent_dim"]; ncls = len(names)
        cls_sd = None                                  # epoch 1: global batch SD
        mix_on = ac.get("synth_mode", "noise") in ("mixup", "mixup_pre"); QCAP = 1024
        pre_on = ac.get("synth_mode", "noise") == "mixup_pre"   # optional: mixup in u (before to_latent)
        queues = [None] * ncls                         # same-class latents (detached) for mixup
        ratio_log = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(idx_tr)
        tot, nb, te = 0.0, 0, time.time()
        if adlc_on:
            # copy-ratio rule (Eq. 5; ADLC calculate_dynamic_augmentation), history = validation monitor
            mon_k = tr_cfg["early_stop"]["monitor"].replace("val_", "")
            fh = [h_[mon_k] for h_ in hist]
            mr = float(ac["max_ratio"])
            ratio = (mr if len(fh) < 2 else
                     max(float(ac["min_ratio"]), mr - 0.05) if fh[-1] >= fh[-2] else mr + 0.05)
            n_aug = int(int(ac["num_augmented"]) * ratio)
            if ep < int(ac.get("start_epoch", 1)):
                n_aug = 0                              # let the encoder form first
            s1 = torch.zeros(ncls, zdim, device=dev, dtype=torch.float64)
            s2 = torch.zeros_like(s1); cnt = torch.zeros(ncls, device=dev, dtype=torch.float64)
            n_syn_ep = 0
        for s in range(0, len(perm), bs):
            j = np.sort(perm[s:s + bs])
            jt = torch.as_tensor(j, device=dev) if torch.is_tensor(D["X"]) else None
            x = take(D, "X", j, dev, jt)
            f = take(D, "F", j, dev, jt)
            m = take(D, "med", j, dev, jt)
            sc = take(D, "scale", j, dev, jt)
            y = torch.from_numpy(D["y"][j]).to(dev)      # D["y"] stays numpy (used by metrics)
            c = take(D, "chan", j, dev, jt)
            gg = take(D, "G", j, dev, jt)
            xa, fa = aug(norm(x, m, sc), f, ch=c)
            if lat_cfg["enabled"] and not adlc_on:
                mk = (torch.from_numpy(gate_full[j]).to(dev)
                      if gate_full is not None else None)
                lat_aug = make_lat_aug(mk)
            if adlc_on:
                with torch.autocast("cuda", dtype=amp):
                    if pre_on:
                        lo, z, u = model(xa, fa, ch=c, g=gg, return_pre=True)
                    else:
                        lo, z = model(xa, fa, ch=c, g=gg, return_latent=True)
                zf = z.float(); zd = zf.detach().double()
                vf = u.float() if pre_on else zf               # space in which mixup is done
                oh = torch.nn.functional.one_hot(y, ncls).double()   # deterministic
                s1 += oh.T @ zd; s2 += oh.T @ (zd * zd); cnt += oh.sum(0)
                sel = torch.from_numpy(gate_full[j]).to(dev)
                if n_aug > 0 and sel.any():
                    zs = vf[sel].repeat_interleave(n_aug, 0)          # NEW copies (not detached)
                    ys = y[sel].repeat_interleave(n_aug, 0)
                    sd = None if pre_on else (cls_sd[ys] if cls_sd is not None
                          else zd.std(0, keepdim=True).float().expand_as(zs))
                    if not mix_on:
                        noise = torch.randn(zs.shape, device=dev, generator=gen)
                        zs = zs + noise * (float(ac["alpha"]) * sd)
                    else:   # optional mixup: z' = z + lam(q - z), q from the same-class queue
                        zs = zs.clone()
                        for cc in range(ncls):
                            rows = (ys == cc).nonzero(as_tuple=True)[0]
                            if len(rows) == 0: continue
                            if queues[cc] is not None and len(queues[cc]) > 0:
                                qi = torch.randint(0, len(queues[cc]), (len(rows),), device=dev, generator=gen)
                                lam_ = torch.rand(len(rows), 1, device=dev, generator=gen)
                                zs[rows] = zs[rows] + lam_ * (queues[cc][qi] - zs[rows])
                            elif not pre_on:   # empty queue -> Gaussian copy instead
                                nz = torch.randn((len(rows), zs.shape[1]), device=dev, generator=gen)
                                zs[rows] = zs[rows] + nz * (float(ac["alpha"]) * sd[rows])
                    with torch.autocast("cuda", dtype=amp):
                        ls = model.head(model.to_latent(zs) if pre_on else zs)
                    lo = torch.cat([lo, ls]); yl = torch.cat([y, ys]); n_syn_ep += len(ys)
                    lam = float(ac.get("syn_weight", 1.0))
                else:
                    yl = y; lam = 1.0
                if mix_on:
                    zq = vf.detach()
                    for cc in range(ncls):
                        m_ = y == cc
                        if m_.any():
                            q_ = zq[m_] if queues[cc] is None else torch.cat([queues[cc], zq[m_]])
                            queues[cc] = q_[-QCAP:]
                gam_f = float(ac.get("focal_gamma", 0.0))
                if gam_f > 0:   # optional focal loss (off in the paper)
                    w = torch.ones(len(yl), device=dev); w[len(y):] = lam
                    li = nn.functional.cross_entropy(lo.float(), yl, reduction="none")
                    li = (1 - torch.exp(-li)).pow(gam_f) * li
                    loss = (li * w).sum() / w.sum()
                elif lam == 1.0:
                    loss = crit(lo.float(), yl)
                else:                                  # synthetic rows weighted by lam
                    w = torch.ones(len(yl), device=dev); w[len(y):] = lam
                    li = nn.functional.cross_entropy(lo.float(), yl, reduction="none")
                    loss = (li * w).sum() / w.sum()
            else:
                with torch.autocast("cuda", dtype=amp):
                    loss = crit(model(xa, fa, ch=c, g=gg, lat_aug=lat_aug), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            opt.step(); sched.step()
            tot += loss.item(); nb += 1

        if adlc_on:
            mean_c = s1 / cnt.clamp(min=1)[:, None]
            var_c = (s2 / cnt.clamp(min=1)[:, None] - mean_c ** 2).clamp(min=0)
            glob = ((s2.sum(0) / cnt.sum()) - (s1.sum(0) / cnt.sum()) ** 2).clamp(min=0)
            var_c[cnt < 2] = glob                                  # classes without data
            cls_sd = var_c.sqrt().float()
            ratio_log.append({"epoch": ep, "ratio": ratio, "n_aug": n_aug, "n_synth": n_syn_ep})
            print(f"  [ADLC] ep{ep} ratio={ratio:.2f} copies/sample={n_aug} synthetic={n_syn_ep}")
        pv, _ = predict(model, D, idx_va, norm, bs * 2, dev, amp, len(names))
        mv = metrics(D["y"][idx_va], pv, names)
        hist.append({"epoch": ep, "train_loss": tot / nb, **{k: mv[k] for k in
                     ("balanced_accuracy", "cohen_kappa", "macro_f1", "weighted_f1")}})
        star = ""
        mon = tr_cfg["early_stop"]["monitor"].replace("val_", "")
        if ep >= tr_cfg["early_stop"]["min_epochs"] and mv[mon] > best["score"]:
            best = {"score": mv[mon], "epoch": ep, "val": mv}
            torch.save({"model": model.state_dict(), "epoch": ep, "cfg": cfg}, ckpt)
            star = "  *"
        print(f"  ep{ep:3d} loss={tot/nb:.4f} | val balAcc={mv['balanced_accuracy']*100:6.2f} "
              f"kappa={mv['cohen_kappa']*100:6.2f} macroF1={mv['macro_f1']*100:6.2f} "
              f"| {time.time()-te:.0f}s{star}", flush=True)
        if ep - best["epoch"] >= tr_cfg["early_stop"]["patience"]:
            print(f"  early stop (no improvement for {tr_cfg['early_stop']['patience']} epochs)")
            break

    # ---- official evaluation, ONCE, with the best checkpoint on validation ----
    model.load_state_dict(torch.load(ckpt)["model"])
    pe, prob = predict(model, E, np.arange(len(E["y"])), norm, bs * 2, dev, amp, len(names))
    me = metrics(E["y"], pe, names)
    if cfg["model"].get("chain_expert", {}).get("enabled", False):
        G = predict_gates(model, E, np.arange(len(E["y"])), norm, bs * 2, dev, amp)
        np.save(os.path.join(cfg["paths"]["results_dir"], f"{run}_gate.npy"), G)
    res = {"run": run, "arm": a.arm, "seed": seed, "best_epoch": best["epoch"],
           "monitor": tr_cfg["early_stop"]["monitor"],
           "deterministic": bool(cfg["runtime"].get("deterministic", False)),
           "lsadlc_train": cfg["lsadlc_train"],
           "oof_file": (a.oof if cfg["lsadlc_train"]["enabled"]
                        and cfg["lsadlc_train"]["gate"] == "pair" else None),
           "split_mode": cfg["split"]["mode"],
           "val_fraction": cfg["split"]["val_fraction"],
           "adlc_synth": a.adlc_synth,
           "adlc_ratio_log": (ratio_log if adlc_on else None),
           "augment": cfg["training"]["augment"],
           "n_train_windows": int(len(idx_tr)), "n_val_windows": int(len(idx_va)),
           "n_train_subjects": int(len(s_tr)), "n_val_subjects": int(len(s_va)),
           "val": best["val"], "eval": me, "history": hist,
           "minutes": (time.time() - t0) / 60}
    rp = os.path.join(cfg["paths"]["results_dir"], f"{run}.json")
    with open(rp, "w") as fh:
        json.dump(res, fh, indent=2)
    np.save(os.path.join(cfg["paths"]["results_dir"], f"{run}_evalprob.npy"), prob)

    print(f"\n=== {run} — EVALUATION (best epoch {best['epoch']}) ===")
    print(f"  balanced accuracy {me['balanced_accuracy']*100:6.2f}")
    print(f"  cohen kappa       {me['cohen_kappa']*100:6.2f}")
    print(f"  macro F1          {me['macro_f1']*100:6.2f}")
    print(f"  weighted F1       {me['weighted_f1']*100:6.2f}")
    print(f"  {'class':6s}{'n':>7s}{'F1':>8s}{'sens':>8s}")
    for n in names:
        print(f"  {n:6s}{me['support'][n]:7d}{me['per_class_f1'][n]*100:8.2f}"
              f"{me['per_class_sensitivity'][n]*100:8.2f}")
    print(f"  -> {rp}  ({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
