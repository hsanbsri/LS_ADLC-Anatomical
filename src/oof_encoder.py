"""oof_encoder.py — probabilitas OOF SEJATI lewat k-fold subject-disjoint ENCODER.

Kenapa perlu: Z diekstrak dari encoder yang dilatih pada SELURUH train, sehingga
k-fold di ruang laten memberi akurasi OOF 99,58% — encoder-nya sudah bocor. Ini
persis catatan proyek ini sendiri bahwa confidence di ruang laten bersifat
SIRKULAR dan margin-nya jenuh.

Di sini encoder DILATIH ULANG per fold, sehingga fold tahan benar-benar tak
pernah dilihat. Rancangannya mengikuti lsadlc_config.yaml Anda: gerbang dari
logit OOF, geometri laten dari model penuh yang dibekukan.

Epoch dipatok (tanpa seleksi val) agar tidak ada seleksi bersarang di dalam OOF.
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
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15, help="dipatok; tanpa seleksi val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out", default="oof_encoder.npz")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--knn", action="store_true",
                    help="LS-ADLC: simpan confidence k-NN OOF di ruang laten (grid K)")
    ap.add_argument("--knn-grid", default="5,10,15,25,50,100")
    ap.add_argument("--plain", action="store_true",
                    help="base POLOS tesis: tanpa chanquery & channel_dropout")
    ap.add_argument("--keep-cd", action="store_true",
                    help="setelan pG/pE1: channel query mati, channel dropout dari config")
    a = ap.parse_args()

    cfg = load_config(overrides=a.overrides)
    # Dulu kedua baris ini di-HARDCODE true sehingga oof_encoder.npz selalu
    # berasal dari pipeline PAPER (chanquery + channel_dropout). Untuk tesis
    # dibutuhkan OOF dari base POLOS, jadi sekarang bisa dimatikan lewat --plain.
    if a.keep_cd:
        # pG/pE1: channel query mati, channel dropout DARI CONFIG (--set), bukan dipaksa
        cfg["model"]["channel_query"]["enabled"] = False
    elif not a.plain:
        cfg["model"]["channel_query"]["enabled"] = True
        cfg["training"]["augment"]["channel_dropout"]["enabled"] = True
    else:
        cfg["model"]["channel_query"]["enabled"] = False
        cfg["training"]["augment"]["channel_dropout"]["enabled"] = False
    mode = "KEEP-CD (pG/pE1: no chanquery, cd dari config)" if a.keep_cd else (
        "POLOS (tesis)" if a.plain else "PAPER (chanquery+cd)")
    print(f"  OOF pipeline: {mode}")
    dev = "cuda"; amp = torch.bfloat16
    C = cfg["channels"]["n_channels"]
    tr_cfg = cfg["training"]; bs = tr_cfg["batch_size"]

    print("memuat cache ...", flush=True)
    D = load_split(cfg, "train", a.arm)
    s_tr, s_va = subject_val_split(np.array(D["subjects"]), D["y"], D["subj"], cfg)
    keep = np.where(~np.isin(D["subj"], s_va))[0]          # HANYA bagian train
    subj = D["subj"][keep]; y = D["y"][keep]
    print(f"  train {len(keep)} window, {len(np.unique(subj))} subjek", flush=True)

    from sklearn.model_selection import GroupKFold
    P = np.zeros((len(keep), cfg["model"]["n_classes"]), dtype=np.float32)
    KG = [int(k) for k in a.knn_grid.split(",")]
    # confidence k-NN OOF: [proporsi tetangga sekelas, versi berbobot 1/jarak]
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
            # Laten fold-latih (referensi) & fold-tahan (kueri) dari encoder YANG
            # SAMA; kueri tak pernah dilihat encoder ini -> tidak sirkular, dan
            # kueri tak pernah jadi tetangga dirinya sendiri (cacat ADLC asli).
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
            Zr, Zq = (Zr - mu) / sd, (Zq - mu) / sd          # standarisasi per dimensi
            yr = torch.from_numpy(D["y"][gtr]).to(dev); yq = torch.from_numpy(D["y"][gte]).to(dev)
            kmax = max(KG)
            for s0 in range(0, len(gte), 2048):
                d = torch.cdist(Zq[s0:s0 + 2048], Zr)                 # (b, n_ref)
                dk, ik = d.topk(kmax, dim=1, largest=False)           # terurut naik
                same = (yr[ik] == yq[s0:s0 + 2048, None]).float()
                w = 1.0 / (dk + 1e-6)
                for kj, K in enumerate(KG):
                    CP[te[s0:s0 + 2048], kj] = same[:, :K].mean(1).cpu().numpy()
                    CW[te[s0:s0 + 2048], kj] = ((same[:, :K] * w[:, :K]).sum(1)
                                                / w[:, :K].sum(1)).cpu().numpy()
            del Zr, Zq
        acc = (P[te].argmax(1) == y[te]).mean()
        print(f"  fold {fi+1}/{a.folds}: {len(tr)} latih / {len(te)} tahan "
              f"| akurasi fold-tahan {acc*100:.2f}% | {(time.time()-t0)/60:.1f} mnt", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.path.join(cfg["paths"]["cache_dir"], a.out)
    np.savez(out, prob=P, y=y, subj=subj, idx_in_train_cache=keep,
             folds=a.folds, epochs=a.epochs,
             **({"knn_grid": np.array(KG), "knn_prop": CP, "knn_w": CW} if a.knn else {}))
    acc = (P.argmax(1) == y).mean()
    print(f"\n  akurasi OOF SEJATI: {acc*100:.2f}%  (bandingkan OOF ruang laten 99,58% "
          f"yang sirkular)")
    mg = np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]
    print(f"  margin OOF: median {np.median(mg):.3f}, "
          f"fraksi > 0,9 = {(mg>0.9).mean()*100:.1f}% (jenuh bila mendekati 100%)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
