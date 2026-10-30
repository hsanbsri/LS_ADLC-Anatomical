"""oof_encoder.py: out-of-fold (OOF) k-NN confidence gate of LS-ADLC (Algorithm 1).

Why a dedicated encoder per fold: latents extracted from an encoder trained on
the WHOLE training pool make any latent-space confidence circular (the encoder
has already seen every window). Here the encoder is RETRAINED in each of the
subject-disjoint folds, so the held-out fold is never seen by the encoder
whose latent space is used to score it.

The number of epochs is fixed (no validation-based selection), so there is no
nested model selection inside the OOF procedure.

With --knn, for every gradient-training window the script stores the
distance-weighted share of same-label neighbours among its k nearest
out-of-fold neighbours (Eq. 2 of the paper) for a grid of k values; train.py
reads the column k=15 and flags windows with vote < tau (Eq. 3).
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, subject_val_split
from features import feat_dim, glob_dim
from model import build_model
from train import GPUNorm, load_split, ChannelDropout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15, help="fixed; no validation-based selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out", default="oof_encoder.npz")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--knn", action="store_true",
                    help="LS-ADLC: store the OOF k-NN confidence in latent space (grid of k)")
    ap.add_argument("--knn-grid", default="5,10,15,25,50,100")
    ap.add_argument("--plain", action="store_true",
                    help="plain backbone: no channel query and no channel dropout")
    ap.add_argument("--keep-cd", action="store_true",
                    help="setting of the paper: channel query off, channel dropout as configured")
    a = ap.parse_args()

    cfg = load_config(a.config, overrides=a.overrides)
    # Pipeline selection. --keep-cd is the setting used in the paper.
    if a.keep_cd:
        # channel query off; channel dropout taken FROM THE CONFIG (--set), not forced
        cfg["model"]["channel_query"]["enabled"] = False
    elif not a.plain:
        cfg["model"]["channel_query"]["enabled"] = True
        cfg["training"]["augment"]["channel_dropout"]["enabled"] = True
    else:
        cfg["model"]["channel_query"]["enabled"] = False
        cfg["training"]["augment"]["channel_dropout"]["enabled"] = False
    mode = "KEEP-CD (no channel query, channel dropout from config)" if a.keep_cd else (
        "PLAIN" if a.plain else "CHANNEL QUERY + CHANNEL DROPOUT")
    print(f"  OOF pipeline: {mode}")
    dev = "cuda"; amp = torch.bfloat16
    C = cfg["channels"]["n_channels"]
    tr_cfg = cfg["training"]; bs = tr_cfg["batch_size"]

    print("loading cache ...", flush=True)
    D = load_split(cfg, "train", a.arm)
    s_tr, s_va = subject_val_split(np.array(D["subjects"]), D["y"], D["subj"], cfg,
                                    n_classes=cfg["model"]["n_classes"])
    keep = np.where(~np.isin(D["subj"], s_va))[0]          # gradient-training rows ONLY
    subj = D["subj"][keep]; y = D["y"][keep]
    print(f"  train {len(keep)} windows, {len(np.unique(subj))} subjects", flush=True)

    from sklearn.model_selection import GroupKFold
    P = np.zeros((len(keep), cfg["model"]["n_classes"]), dtype=np.float32)
    KG = [int(k) for k in a.knn_grid.split(",")]
    # OOF k-NN confidence: [share of same-label neighbours, 1/distance-weighted share]
    CP = np.zeros((len(keep), len(KG)), dtype=np.float32)
    CW = np.zeros((len(keep), len(KG)), dtype=np.float32)
    norm = GPUNorm(cfg, a.arm, dev); aug = ChannelDropout(cfg)
    gd = glob_dim(cfg) if D.get("G") is not None else 0
    t0 = time.time()

    for fi, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(
            np.arange(len(keep)), y, groups=subj)):
        gtr, gte = keep[tr], keep[te]
        torch.manual_seed(a.seed + fi); np.random.seed(a.seed + fi)
        model = build_model(cfg, C, feat_dim(cfg), gd).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                                weight_decay=tr_cfg["weight_decay"])
        steps = int(np.ceil(len(gtr) / bs)); total = a.epochs * steps
        warm = tr_cfg["warmup_epochs"] * steps
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
            (s + 1) / max(warm, 1) if s < warm else
            0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1)))))
        crit = nn.CrossEntropyLoss()
        rng = np.random.RandomState(a.seed + fi)

        def batch(j):
            x = torch.from_numpy(D["X"][j]).to(dev)
            f = torch.from_numpy(D["F"][j]).to(dev)
            m = torch.from_numpy(D["med"][j]).to(dev)
            sc = torch.from_numpy(D["scale"][j]).to(dev)
            c = torch.from_numpy(D["chan"][j]).to(dev)
            g = torch.from_numpy(D["G"][j]).to(dev) if D.get("G") is not None else None
            return norm(x, m, sc), f, c, g

        for ep in range(a.epochs):
            model.train()
            perm = rng.permutation(gtr)
            for s in range(0, len(perm), bs):
                j = np.sort(perm[s:s + bs])
                x, f, c, g = batch(j)
                xa, fa = aug(x, f, ch=c)
                yb = torch.from_numpy(D["y"][j]).to(dev)
                with torch.autocast("cuda", dtype=amp):
                    loss = crit(model(xa, fa, ch=c, g=g), yb)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
                opt.step(); sch.step()

        model.eval()
        with torch.no_grad():
            for s in range(0, len(gte), 512):
                j = gte[s:s + 512]
                x, f, c, g = batch(j)
                with torch.autocast("cuda", dtype=amp):
                    lo = model(x, f, ch=c, g=g)
                P[te[s:s + 512]] = lo.float().softmax(-1).cpu().numpy()
        if a.knn:
            # Latents of the training folds (reference) and of the held-out fold
            # (queries) from the SAME encoder; queries were never seen by this
            # encoder, and a query can never be its own neighbour.
            def latents(idx):
                out = []
                with torch.no_grad():
                    for s0 in range(0, len(idx), 1024):
                        x, f, c, g = batch(idx[s0:s0 + 1024])
                        with torch.autocast("cuda", dtype=amp):
                            _, z = model(x, f, ch=c, g=g, return_latent=True)
                        out.append(z.float())
                return torch.cat(out)
            Zr, Zq = latents(gtr), latents(gte)
            mu, sd = Zr.mean(0, keepdim=True), Zr.std(0, keepdim=True) + 1e-6
            Zr, Zq = (Zr - mu) / sd, (Zq - mu) / sd          # per-dimension standardization
            yr = torch.from_numpy(D["y"][gtr]).to(dev); yq = torch.from_numpy(D["y"][gte]).to(dev)
            kmax = max(KG)
            for s0 in range(0, len(gte), 2048):
                d = torch.cdist(Zq[s0:s0 + 2048], Zr)                 # (b, n_ref), Euclidean
                dk, ik = d.topk(kmax, dim=1, largest=False)           # ascending
                same = (yr[ik] == yq[s0:s0 + 2048, None]).float()
                w = 1.0 / (dk + 1e-6)
                for kj, K in enumerate(KG):
                    CP[te[s0:s0 + 2048], kj] = same[:, :K].mean(1).cpu().numpy()
                    CW[te[s0:s0 + 2048], kj] = ((same[:, :K] * w[:, :K]).sum(1)
                                                / w[:, :K].sum(1)).cpu().numpy()
            del Zr, Zq
        acc = (P[te].argmax(1) == y[te]).mean()
        print(f"  fold {fi+1}/{a.folds}: {len(tr)} train / {len(te)} held out "
              f"| held-out accuracy {acc*100:.2f}% | {(time.time()-t0)/60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.path.join(cfg["paths"]["cache_dir"], a.out)
    np.savez(out, prob=P, y=y, subj=subj, idx_in_train_cache=keep,
             folds=a.folds, epochs=a.epochs,
             **({"knn_grid": np.array(KG), "knn_prop": CP, "knn_w": CW} if a.knn else {}))
    acc = (P.argmax(1) == y).mean()
    print(f"\n  OOF accuracy: {acc*100:.2f}%")
    mg = np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]
    print(f"  OOF margin: median {np.median(mg):.3f}, "
          f"share > 0.9 = {(mg>0.9).mean()*100:.1f}%")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
