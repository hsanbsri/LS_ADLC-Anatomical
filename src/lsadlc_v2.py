"""lsadlc_v2.py: alternative gates and augmentation modes (exploratory).

Imported by train.py for its alternative "pair" gate. The LS-ADLC configuration
used in the paper (gate "adlc": out-of-fold latent k-NN vote, Eqs. 2-3) does
NOT use this module; it reads the k-NN file written by oof_encoder.py.

GATES
  percentile  out-of-fold margin, lowest percentile WITHIN each class
  pair        both top-2 out-of-fold classes inside the epileptiform triad
              {SPSW, GPED, PLED}
AUGMENTATION
  isotropic   z + N(0, sigma)
  directional z + lam * (mu_b - mu_a), moving from class a towards the
              confusing class b while keeping label a
"""
import numpy as np
import torch

TRIAD = [0, 1, 2]          # SPSW, GPED, PLED


# ------------------------------------------------------------------- OOF ----
def oof_probs(Zt, yt, subj, hp, seed=42, folds=5, n_cls=6, log=print):
    """Out-of-fold probabilities via subject-disjoint k-fold in latent space."""
    from head_train import build_head, _pred
    from sklearn.model_selection import GroupKFold
    dev = Zt.device
    P = torch.zeros(len(yt), n_cls, device=dev)
    yn = yt.cpu().numpy()
    bs = hp["batch_size"]
    crit = torch.nn.CrossEntropyLoss()
    for i, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(
            np.arange(len(yn)), yn, groups=subj)):
        tr_t = torch.from_numpy(tr).to(dev); te_t = torch.from_numpy(te).to(dev)
        torch.manual_seed(seed + i)
        head = build_head(Zt.shape[1], n_cls, hp, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"],
                                weight_decay=hp["weight_decay"])
        g = torch.Generator(device=dev); g.manual_seed(seed + i)
        for ep in range(hp["epochs"]):
            head.train()
            perm = torch.randperm(len(tr_t), device=dev, generator=g)
            for s in range(0, len(perm), bs):
                j = tr_t[perm[s:s + bs]]
                loss = crit(head(Zt[j]), yt[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        P[te_t] = _pred(head, Zt[te_t])
        del head
    acc = float((P.argmax(1) == yt).float().mean())
    log(f"  OOF {folds}-fold subject-disjoint: accuracy {acc*100:.2f}%")
    return P


# ------------------------------------------------------------------ gates ----
def gate_percentile(P, y, pct, n_cls=6):
    """Lowest out-of-fold margin; the percentile is computed WITHIN each class."""
    top2 = P.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    yn = y.cpu().numpy()
    gate = np.zeros(len(yn), bool)
    for c in range(n_cls):
        m = yn == c
        if m.sum() == 0:
            continue
        thr = np.percentile(margin[m], pct)
        gate[m & (margin <= thr)] = True
    return gate


def gate_pair(P, y, triad=TRIAD):
    """Both top-2 out-of-fold classes inside the triad -> model split between epileptiform classes.

    Returns (gate, partner): partner[i] = the other, confusing class.
    """
    idx = P.topk(2, dim=1).indices.cpu().numpy()
    yn = y.cpu().numpy()
    in_tri = np.isin(idx, triad).all(1)
    gate = in_tri & np.isin(yn, triad)
    partner = np.where(idx[:, 0] == yn, idx[:, 1], idx[:, 0])
    return gate, partner


def gate_adlc_knn(P, thr=0.8):
    return (P.max(1).values < thr).cpu().numpy()


# ----------------------------------------------------------- augmentation ----
def class_means(Z, y, n_cls=6):
    return torch.stack([Z[y == c].mean(0) if (y == c).any()
                        else torch.zeros(Z.shape[1], device=Z.device)
                        for c in range(n_cls)])


def augment(Z, y, gate, mode, rng, n_synth=5, sigma=0.1, lam_max=0.3,
            partner=None, mu=None):
    """-> (extended Z, extended y, number of new rows). Synthetic label = anchor label."""
    idx = np.where(gate)[0]
    if len(idx) == 0 or n_synth <= 0:
        return Z, y, 0
    dev = Z.device
    ii = torch.from_numpy(np.repeat(idx, n_synth)).to(dev)
    base = Z[ii]
    if mode == "isotropic":
        noise = torch.randn(base.shape, device=dev, generator=rng) * sigma
        new = base + noise
    elif mode == "directional":
        assert partner is not None and mu is not None
        a = y[ii]
        b = torch.from_numpy(partner[np.repeat(idx, n_synth)]).to(dev)
        d = mu[b] - mu[a]                                   # direction towards the confusing class
        lam = torch.rand(len(ii), 1, device=dev, generator=rng) * lam_max
        new = base + lam * d
    else:
        raise ValueError(mode)
    return (torch.cat([Z, new]), torch.cat([y, y[ii]]), len(ii))
