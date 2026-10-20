"""model.py — two-branch CNN-Res-BiLSTM-Attention backbone.

Temporal branch : normalized signal (B, C, 1250)
                  -> Conv1d stem -> 4 residual blocks -> (B, 256, 40)
                  -> BiLSTM -> attention pooling (+ global max pooling) -> sequence vector
Spectral branch : DWT subband features (B, C, 21) -> MLP -> 64-d
Concatenation -> projection to the latent z (256-d) -> linear head -> class logits.

The spectral branch keeps amplitude information through log_energy/log_max_abs;
with per-window z-scoring (arm A) it is the only path that preserves the
within-window scale. Optional modules (Transformer, Mamba, channel query,
annotated-channel branch, global features) are switched off in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, p_drop, residual=True):
        super().__init__()
        self.residual = bool(residual)
        pad = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p_drop)
        # without the residual path, the shortcut is not created at all (no parameters counted)
        self.short = None if not self.residual else (
            nn.Identity() if (cin == cout and stride == 1) else
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(cout)))

    def forward(self, x):
        r = self.short(x) if self.residual else None
        x = F.gelu(self.b1(self.c1(x)))
        x = self.drop(x)
        x = self.b2(self.c2(x))
        return F.gelu(x + r) if self.residual else F.gelu(x)


class AttentionPool(nn.Module):
    """Additive (Bahdanau) attention over the time axis."""
    def __init__(self, d_in, d_att):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1, bias=False)

    def forward(self, h):                      # h (B, T, D)
        a = self.v(torch.tanh(self.W(h)))      # (B, T, 1)
        a = torch.softmax(a, dim=1)
        return (a * h).sum(1), a.squeeze(-1)


class TransformerSeq(nn.Module):
    """Optional Transformer encoder over the time axis (alternative to the BiLSTM).

    The sequence has only 40 tokens (dim 256), so 40x40 self-attention is cheap.
    The output width equals the input width, so the downstream modules
    (attention pooling, to_latent) are unchanged.
    """
    def __init__(self, d_model, n_layers, n_head, d_ff, p_drop, max_len=64):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                           dim_feedforward=d_ff, dropout=p_drop,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, h):                      # h (B, T, D)
        return self.enc(h + self.pos[:, :h.shape[1]])


class BiMamba(nn.Module):
    """Optional bidirectional Mamba: projection -> forward + backward Mamba -> concat (width 2*proj, like the BiLSTM)."""
    def __init__(self, d_in, proj, d_state, d_conv, expand):
        super().__init__()
        from mamba_ssm import Mamba
        self.proj = nn.Linear(d_in, proj)
        self.fwd = Mamba(d_model=proj, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=proj, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, h):                      # h (B, T, D)
        u = self.proj(h)
        return torch.cat([self.fwd(u), self.bwd(u.flip(1)).flip(1)], dim=-1)


class CNNResBiLSTMAttention(nn.Module):
    def __init__(self, cfg, n_ch, feat_dim, glob_dim=0):
        super().__init__()
        m = cfg["model"]
        t, sp, ls, at = m["temporal"], m["spectral"], m["lstm"], m["attention"]
        k = t["kernel_size"]

        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, t["stem_channels"], k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(t["stem_channels"]), nn.GELU())
        self.use_res = bool(t.get("residual", True))
        blocks, cin = [], t["stem_channels"]
        # stride per block (default: 2 for every block, i.e. 32x downsampling -> 128 ms per token)
        strides = t.get("block_strides", [2] * len(t["res_blocks"]))
        assert len(strides) == len(t["res_blocks"])
        for cout, st in zip(t["res_blocks"], strides):
            blocks.append(ResBlock(cin, cout, k, int(st), t["dropout"], residual=self.use_res))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        c_temporal = cin                      # CNN output width (cin is reused below)

        # ablation switches; the defaults (True) give the backbone used in the paper
        self.use_lstm = bool(ls.get("enabled", True))
        self.use_attn = bool(at.get("enabled", True))
        tf = m.get("transformer", {"enabled": False})
        self.use_trf = bool(tf.get("enabled", False))
        mb = m.get("mamba", {"enabled": False})
        self.use_mamba = bool(mb.get("enabled", False))
        assert int(self.use_lstm) + int(self.use_trf) + int(self.use_mamba) <= 1, \
            "choose ONE temporal module: lstm, transformer, OR mamba"
        if self.use_trf:
            self.trf = TransformerSeq(c_temporal, int(tf.get("layers", 1)),
                                      int(tf.get("heads", 4)),
                                      int(tf.get("ff_dim", 256)),
                                      float(tf.get("dropout", 0.1)),
                                      int(tf.get("max_len", 64)))
        if self.use_lstm:
            self.lstm = nn.LSTM(c_temporal, ls["hidden"], num_layers=ls["layers"],
                                batch_first=True, bidirectional=True,
                                dropout=ls["dropout"] if ls["layers"] > 1 else 0.0)
            d_seq = 2 * ls["hidden"]
        elif self.use_mamba:
            self.lstm = None
            self.mamba = BiMamba(c_temporal, int(mb.get("proj", 128)), int(mb.get("d_state", 16)),
                                 int(mb.get("d_conv", 4)), int(mb.get("expand", 2)))
            d_seq = 2 * int(mb.get("proj", 128))
        else:
            self.lstm = None
            d_seq = c_temporal              # the Transformer keeps the width
        self.pool = AttentionPool(d_seq, at["dim"]) if self.use_attn else None

        cq = m.get("channel_query", {"enabled": False})
        self.cq = bool(cq["enabled"])
        self.film = bool(cq.get("film", False)) and self.cq
        e = int(cq["emb_dim"]) if self.cq else 0
        if self.cq:
            # emb(k) = "which channel is being queried" (optional, off in the paper)
            self.ch_emb = nn.Embedding(n_ch, e)
            if self.film:
                self.film_fc = nn.Linear(e, 2 * d_seq)
        # spectral branch: features of ALL channels (+ queried channel and embedding if enabled)
        d_spec_in = n_ch * feat_dim + (feat_dim + e if self.cq else 0)
        # optional global branch: laterality (a property of the whole window)
        gcfg = m.get("global_feats", {"enabled": False})
        self.gdim = glob_dim if (gcfg.get("enabled") and glob_dim) else 0
        d_glob = 0
        if self.gdim:
            self.gnet = nn.Sequential(nn.Linear(self.gdim, gcfg["hidden"]), nn.GELU())
            d_glob = gcfg["hidden"]
        self.spec = nn.Sequential(
            nn.Linear(d_spec_in, sp["hidden"]), nn.GELU(),
            nn.Dropout(sp["dropout"]),
            nn.Linear(sp["hidden"], sp["out_dim"]), nn.GELU())

        # global max pooling over time, next to attention pooling (enabled in the paper)
        self.max_pool = bool(m.get("max_pool", {}).get("enabled", False))
        d_mp = d_seq if self.max_pool else 0
        # optional annotated-channel branch (off in the paper)
        ac = m.get("ann_channel_branch", {"enabled": False})
        self.ann = bool(ac.get("enabled", False))
        d_ann = 0
        if self.ann:
            ka = ac["kernel_size"]; lay = []; cin = 1
            for cout in ac["channels"]:
                lay += [nn.Conv1d(cin, cout, ka, stride=ac["stride"], padding=ka // 2,
                                  bias=False), nn.BatchNorm1d(cout), nn.GELU()]
                cin = cout
            self.annnet = nn.Sequential(*lay)
            d_ann = 2 * cin                      # max + mean over time
        self.to_latent = nn.Sequential(
            nn.Linear(d_seq + d_mp + sp["out_dim"] + d_glob + d_ann,
                      m["latent_dim"]), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(m["head_dropout"]),
                                  nn.Linear(m["latent_dim"], m["n_classes"]))

    def forward(self, x, f, ch=None, g=None, lat_aug=None, return_latent=False, return_pre=False):
        h = self.blocks(self.stem(x))          # (B, C, T')
        hseq = h.transpose(1, 2)               # (B, T', C)
        if self.use_lstm:
            hseq, _ = self.lstm(hseq)          # (B, T', 2H)
        elif self.use_trf:
            hseq = self.trf(hseq)              # (B, T', C)
        elif self.use_mamba:
            hseq = self.mamba(hseq)            # (B, T', 2*proj)
        if self.use_attn:
            h, att = self.pool(hseq)
        else:
            h, att = hseq.mean(dim=1), None    # mean pooling over time
        s_in = f.flatten(1)
        if self.cq:
            assert ch is not None, "the channel_query variant needs the channel index"
            e = self.ch_emb(ch)
            q = f[torch.arange(f.shape[0], device=f.device), ch]   # features of the queried channel
            s_in = torch.cat([s_in, q, e], dim=-1)
            if self.film:
                gam, bet = self.film_fc(e).chunk(2, dim=-1)   # not named g:
                h = h * (1 + gam) + bet                        # g = global features
        parts = [h]
        if self.max_pool:
            parts.append(hseq.max(dim=1).values)
        if self.ann:
            assert ch is not None, "the annotated-channel branch needs the channel index"
            xa = x[torch.arange(x.shape[0], device=x.device), ch].unsqueeze(1)
            aa = self.annnet(xa)
            parts.append(torch.cat([aa.max(-1).values, aa.mean(-1)], dim=-1))
        parts.append(self.spec(s_in))
        if self.gdim:
            assert g is not None, "the model needs the global feature vector (laterality)"
            parts.append(self.gnet(g))
        u = torch.cat(parts, dim=-1)           # joint post-attention representation
        z = self.to_latent(u)
        if lat_aug is not None:          # latent augmentation hook (encoder is trained jointly)
            z = lat_aug(z)
        logits = self.head(z)
        if return_pre:                         # also return the pre-latent representation u
            return logits, z, u
        return (logits, z) if return_latent else logits


class ChannelSubset(nn.Module):
    """One model per anatomical group: only the channels `ids` enter the backbone.

    No annotated-channel information is used (ch is ignored). Normalization and
    channel dropout are computed on all 22 channels by train.py; the group's
    channels are selected here.
    """
    def __init__(self, base, ids):
        super().__init__()
        self.base = base
        self.register_buffer("idx", torch.tensor(ids, dtype=torch.long), persistent=False)

    @property
    def head(self):
        return self.base.head

    @property
    def to_latent(self):
        return self.base.to_latent

    def forward(self, x, f, ch=None, g=None, lat_aug=None, return_latent=False, return_pre=False):
        return self.base(x.index_select(1, self.idx), f.index_select(1, self.idx), ch=None,
                         g=g, lat_aug=lat_aug, return_latent=return_latent, return_pre=return_pre)


def build_model(cfg, n_ch, feat_dim, glob_dim=0):
    # Unrelated experimental architecture from a separate project; not part of
    # this repository and never enabled by the configs shipped here.
    if cfg["model"].get("chain_expert", {}).get("enabled", False):
        from chain_expert import ChainExpert
        return ChainExpert(cfg, n_ch, feat_dim, glob_dim)
    ids = cfg["model"].get("channel_subset")
    if ids:   # one independently trained model per anatomical channel group
        ids = [int(i) for i in ids]
        base = CNNResBiLSTMAttention(cfg, len(ids), feat_dim, glob_dim)
        return ChannelSubset(base, ids)
    return CNNResBiLSTMAttention(cfg, n_ch, feat_dim, glob_dim)
