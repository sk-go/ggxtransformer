"""
chroma/modell.py
Decoder-only-Transformer, dessen Basis eine Variable ist (Spec § 4.2).

farbe = "keine": architektonisch rotationsinvariant (A₀, A)
farbe = "alpha": nichtnegatives Embedding, Archetyp = 0,1 · softplus(W) (Bα, Farbe im Archetyp)
farbe = "beta":  jede Lesung des Streams geht durch softplus (Bβ, Farbe im Vollzug)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FARBEN = ("keine", "alpha", "beta")
ALPHA_SKALA = 0.1        # skalare Skalierung ändert die Basiseigenschaft nicht
ALPHA_INIT_MITTEL = -2.0  # sonst lägen alle softplus-Vektoren nahe ln 2 und wären fast gleich


@dataclass
class ChromaConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    farbe: str = "keine"

    def __post_init__(self):
        if self.farbe not in FARBEN:
            raise ValueError(f"farbe muss eine von {FARBEN} sein, nicht {self.farbe!r}")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd muss durch n_head teilbar sein")


class SkalarRMSNorm(nn.Module):
    """RMSNorm mit einem einzigen skalaren Gain: Das Maß zeichnet keine Richtung aus."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.eps = eps

    def forward(self, x):
        return self.gain * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Lesung(nn.Module):
    """Wie eine Teilschicht den Stream erblickt; bei farbe='beta' durch die farbige Linse softplus."""

    def __init__(self, farbe):
        super().__init__()
        self.farbig = farbe == "beta"
        self.norm = SkalarRMSNorm()

    def forward(self, x):
        return self.norm(F.softplus(x) if self.farbig else x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.les_1 = Lesung(cfg.farbe)
        self.attn = Attention(cfg)
        self.les_2 = Lesung(cfg.farbe)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.les_1(x))
        return x + self.mlp(self.les_2(x))


class ChromaModell(nn.Module):
    def __init__(self, cfg: ChromaConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.les_f = Lesung(cfg.farbe)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # entkoppelt: kein weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)  # isotrop
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if cfg.farbe == "alpha":
            nn.init.normal_(self.wte.weight, ALPHA_INIT_MITTEL, 1.0)

    def archetyp(self):
        """Die Embedding-Matrix, wie das Modell sie benutzt (bei alpha nichtnegativ)."""
        W = self.wte.weight
        return ALPHA_SKALA * F.softplus(W) if self.cfg.farbe == "alpha" else W

    def strom(self, idx):
        """Stream-Zustände: Eingang jedes Blocks, dann Ausgang des letzten (Länge n_layer + 1)."""
        T = idx.shape[1]
        if T > self.cfg.block_size:
            raise ValueError(f"Sequenz der Länge {T} überschreitet block_size {self.cfg.block_size}")
        x = F.embedding(idx, self.archetyp()) + self.wpe(torch.arange(T, device=idx.device))
        zustaende = [x]
        for block in self.h:
            x = block(x)
            zustaende.append(x)
        return zustaende

    def forward(self, idx):
        return self.lm_head(self.les_f(self.strom(idx)[-1]))
