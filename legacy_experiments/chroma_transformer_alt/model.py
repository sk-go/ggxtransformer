"""
chroma_transformer/model.py
===========================
Realisierung der Tripelstruktur des Sinns in einem minimalen, voll interpretierbaren Transformer (nanoGPT-Stil).

Dimensionen des Sinns:
- Art (kantische Inhärenz / Humes Ähnlichkeit): Q * K^T Resonanz
- Folge (kantische Sukzession / Kausalität): Positionskodierung (RoPE / Pos-Embedding)
- Zugleichsein (kantische Gemeinschaft): Value-Aggregation über die Aufmerksamkeitsmatrix
- Träger (kantische Substanz / Nietzscheanischer Täter): Der Residual Stream

Chromatischer Akkord:
- Jede Schicht projiziert den Residual Stream über eine Farbmatrix W_RGB in den RGB-Farbraum.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChromaConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.0
    bias: bool = True
    
    # Tripel-Ablationsschalter (Standard: T111 = voller Sinn)
    art_enabled: bool = True       # Q * K^T Resonanz (Gleichartigkeit)
    folge_enabled: bool = True     # Positional Encoding (Sukzession / Folge)
    zugleich_enabled: bool = True  # Value-Mixing über fremde Tokens (Gemeinschaft / Zugleichsein)


class CausalSelfAttention(nn.Module):
    """
    Attention-Modul mit expliziten Schaltern für Art, Folge und Zugleichsein.
    """
    def __init__(self, config: ChromaConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V Projektionen
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Ausgangs-Projektion
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Kausale Maske für die gerichtete Folge
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        art: Optional[bool] = None,
        zugleich: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        art_active = self.config.art_enabled if art is None else art
        zugleich_active = self.config.zugleich_enabled if zugleich is None else zugleich

        # Q, K, V berechnen
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # In Multi-Head umformen: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if not zugleich_active:
            # T--0: OHNE ZUGLEICHSEIN
            # Jedes Token interagiert nur mit sich selbst (Einheitsmatrix delta_ij).
            # Keine Gemeinschaft fremder Token-Inhalte.
            attn_weights = torch.eye(T, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, self.n_head, T, T)
            y = v # da Summe_j delta_ij * v_j = v_i
        else:
            if not art_active:
                # T-0-: OHNE ART (Inhaltsblinde Aufmerksamkeit)
                # Q * K^T wird genullt. Die Aufmerksamkeit verteilt sich rein gleichmäßig
                # über alle bisherigen Zeitschritte (uniforme kausale Gewichtung).
                scores = torch.zeros((B, self.n_head, T, T), device=x.device)
            else:
                # T-1-: MIT ART (Projektive Resonanz)
                scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # Kausale Maskierung der Zukunft
            scores = scores.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Zugleichsein stiftet den neuen Gegenwartsvektor
            y = attn_weights @ v

        # Köpfe wieder zusammenführen
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        return out, attn_weights


class MLP(nn.Module):
    """
    Kants 3. Synthesis: Rekognition im Begriff (Schlüssel-Wert-Subsumtion nach Geva/Meng).
    """
    def __init__(self, config: ChromaConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Ein Transformer-Block: Schreibt Schicht für Schicht in den Träger (Residual Stream).
    """
    def __init__(self, config: ChromaConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        art: Optional[bool] = None,
        zugleich: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Träger-Addition: Kants 'Ich denke' / Halten
        attn_out, weights = self.attn(self.ln_1(x), art=art, zugleich=zugleich)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class ChromaTransformer(nn.Module):
    """
    Der Chromatische Transformer:
    Vollständige Modellierung von Art, Folge, Zugleichsein und chromatischer Spektralanalyse.
    """
    def __init__(self, config: ChromaConfig):
        super().__init__()
        self.config = config

        # 1. Embedding = Der Archetyp (Disposition ohne Folge und Zugleichsein)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Positionskodierung = Die reine Anschauungsform Zeit (Folge)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Erwartungshorizont / Unembedding
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # Weight Tying (Erscheinung & Erwartung am selben Maß)

        # 3. Die Farbmatrix: Projektion des hochdimensionalen Raums auf RGB
        # Orthogonale Initialisierung zur optimalen Trennung dreier Hauptachsen
        rgb_proj = torch.randn(config.n_embd, 3)
        q_proj, _ = torch.linalg.qr(rgb_proj)
        self.register_buffer("W_RGB", q_proj)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def project_to_rgb(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Projiziert Vektoren aus dem Residual Stream in normierte RGB-Farben [0, 1].
        """
        coords = hidden_state @ self.W_RGB
        rgb = torch.sigmoid(coords)
        return rgb

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        art: Optional[bool] = None,
        folge: Optional[bool] = None,
        zugleich: Optional[bool] = None,
        return_color_trace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequenzlänge {t} übersteigt Blockgröße {self.config.block_size}"

        folge_active = self.config.folge_enabled if folge is None else folge

        # 1. Archetyp (Token-Embedding)
        tok_emb = self.wte(idx)

        # 2. Folge (Positionskodierung)
        if folge_active:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        else:
            # T-0-: OHNE FOLGE (Zeitlose Menge)
            x = tok_emb

        x = self.drop(x)

        # Farbspur initialisieren (Schicht 0 = Archetyp)
        color_traces = []
        if return_color_trace:
            color_traces.append(self.project_to_rgb(x))

        all_attentions = []
        # Durch die Schichten (Transzendentale Apperzeption / Träger)
        for block in self.h:
            x, weights = block(x, art=art, zugleich=zugleich)
            all_attentions.append(weights)
            if return_color_trace:
                color_traces.append(self.project_to_rgb(x))

        x_norm = self.ln_f(x)
        logits = self.lm_head(x_norm)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        result = {
            "logits": logits,
            "loss": loss,
            "attentions": all_attentions,
        }
        if return_color_trace:
            # Tensor der Form (n_layer + 1, B, T, 3)
            result["color_trace"] = torch.stack(color_traces, dim=0)

        return result
