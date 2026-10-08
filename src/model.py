"""model.py — CNN-Res-BiLSTM-Attention dua cabang.

Cabang T (temporal) : sinyal ternormalisasi (B, 22, 1250)
                      -> stem Conv1d -> 4 blok residual -> (B, 256, 40)
                      -> BiLSTM -> attention pooling -> 256-d
Cabang S (spektral) : fitur subband DWT (B, 22, 21) -> MLP -> 64-d
Gabung -> proyeksi ke laten Z (256) -> head linear -> 6 kelas.

Cabang S menjaga informasi amplitudo lewat log_energy/log_max_abs; pada arm A
cabang itulah satu-satunya jalur yang mempertahankan skala di dalam window.
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
        # tanpa residual: jalur pintas tidak dibuat sama sekali (param tidak ikut terhitung)
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
    """Additive attention (Bahdanau) atas sumbu waktu."""
    def __init__(self, d_in, d_att):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1, bias=False)

    def forward(self, h):                      # h (B, T, D)
        a = self.v(torch.tanh(self.W(h)))      # (B, T, 1)
        a = torch.softmax(a, dim=1)
        return (a * h).sum(1), a.squeeze(-1)


class TransformerSeq(nn.Module):
    """Encoder Transformer atas sumbu waktu, pengganti BiLSTM.

    Sekuensnya hanya 40 token (dim 256), jadi self-attention 40x40 sangat murah.
    Dimensi keluaran = dimensi masukan, sehingga bagian hilir (attention pool,
    to_latent) tidak berubah dan perbandingannya adil.
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
    """Bi-Mamba (pilot resolusi): proyeksi -> Mamba maju + Mamba mundur -> concat (lebar 2*proj, seperti BiLSTM)."""
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
        # pilot resolusi: stride per blok (default 2 semua = perilaku lama, 32x -> 128 ms/token)
        strides = t.get("block_strides", [2] * len(t["res_blocks"]))
        assert len(strides) == len(t["res_blocks"])
        for cout, st in zip(t["res_blocks"], strides):
            blocks.append(ResBlock(cin, cout, k, int(st), t["dropout"], residual=self.use_res))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        c_temporal = cin                      # lebar keluaran CNN (cin dipakai ulang di bawah)

        # saklar ablasi — default True, jadi perilaku lama tidak berubah
        self.use_lstm = bool(ls.get("enabled", True))
        self.use_attn = bool(at.get("enabled", True))
        tf = m.get("transformer", {"enabled": False})
        self.use_trf = bool(tf.get("enabled", False))
        mb = m.get("mamba", {"enabled": False})
        self.use_mamba = bool(mb.get("enabled", False))
        assert int(self.use_lstm) + int(self.use_trf) + int(self.use_mamba) <= 1, \
            "pilih SALAH SATU modul temporal: lstm, transformer, ATAU mamba"
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
            d_seq = c_temporal              # Transformer mempertahankan dimensi
        self.pool = AttentionPool(d_seq, at["dim"]) if self.use_attn else None

        cq = m.get("channel_query", {"enabled": False})
        self.cq = bool(cq["enabled"])
        self.film = bool(cq.get("film", False)) and self.cq
        e = int(cq["emb_dim"]) if self.cq else 0
        if self.cq:
            # emb(k) = "kanal mana yang sedang ditanyakan"
            self.ch_emb = nn.Embedding(n_ch, e)
            if self.film:
                self.film_fc = nn.Linear(e, 2 * d_seq)
        # cabang spektral: fitur SEMUA kanal + fitur kanal yang DITANYA + emb
        d_spec_in = n_ch * feat_dim + (feat_dim + e if self.cq else 0)
        # cabang global: lateralitas (properti seluruh window, bukan per kanal)
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

        # (A) pooling maksimum atas waktu, di samping attention-mean
        self.max_pool = bool(m.get("max_pool", {}).get("enabled", False))
        d_mp = d_seq if self.max_pool else 0
        # (C) cabang kanal-teranotasi: 1 kanal, downsampling jauh lebih kecil
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
            d_ann = 2 * cin                      # max + mean atas waktu
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
            h, att = hseq.mean(dim=1), None    # mean pooling atas waktu
        s_in = f.flatten(1)
        if self.cq:
            assert ch is not None, "varian channel_query butuh indeks kanal"
            e = self.ch_emb(ch)
            q = f[torch.arange(f.shape[0], device=f.device), ch]   # fitur kanal ditanya
            s_in = torch.cat([s_in, q, e], dim=-1)
            if self.film:
                gam, bet = self.film_fc(e).chunk(2, dim=-1)   # jangan pakai nama g:
                h = h * (1 + gam) + bet                        # g = fitur global
        parts = [h]
        if self.max_pool:
            parts.append(hseq.max(dim=1).values)
        if self.ann:
            assert ch is not None, "cabang kanal-teranotasi butuh indeks kanal"
            xa = x[torch.arange(x.shape[0], device=x.device), ch].unsqueeze(1)
            aa = self.annnet(xa)
            parts.append(torch.cat([aa.max(-1).values, aa.mean(-1)], dim=-1))
        parts.append(self.spec(s_in))
        if self.gdim:
            assert g is not None, "model butuh vektor fitur global (lateralitas)"
            parts.append(self.gnet(g))
        u = torch.cat(parts, dim=-1)           # representasi gabungan pasca-attention
        z = self.to_latent(u)
        if lat_aug is not None:          # LS-ADLC di titik B, encoder IKUT DILATIH
            z = lat_aug(z)
        logits = self.head(z)
        if return_pre:                         # V-MIXA: mixup di u, bukan z
            return logits, z, u
        return (logits, z) if return_latent else logits


class ChannelSubset(nn.Module):
    """Satu model per kelompok anatomi (gaya ACDE): hanya kanal `ids` yang masuk ke backbone.

    Tanpa informasi kanal beranotasi (ch diabaikan). Normalisasi dan channel dropout
    tetap dihitung di 22 kanal oleh train.py, lalu dipotong di sini.
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
    # Unrelated experiment from a separate project; not part of this repo
    # and never enabled by any config in configs/config.yaml here.
    if cfg["model"].get("chain_expert", {}).get("enabled", False):
        from chain_expert import ChainExpert
        return ChainExpert(cfg, n_ch, feat_dim, glob_dim)
    ids = cfg["model"].get("channel_subset")
    if ids:   # one independently-trained model per anatomical channel group
        ids = [int(i) for i in ids]
        base = CNNResBiLSTMAttention(cfg, len(ids), feat_dim, glob_dim)
        return ChannelSubset(base, ids)
    return CNNResBiLSTMAttention(cfg, n_ch, feat_dim, glob_dim)
