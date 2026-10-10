"""
chroma/messung/metrik.py
Explorativ, sekundär. Der Kosinus ist unser Vorurteil (V2 § 6.1: Metrik ist Füllung).
G_QK misst mit der eigenen, durch Gewohnheit sedimentierten Metrik des Modells: der
Bilinearform W_Qᵀ W_K der Köpfe in Schicht 0. G_top ist das topologische Gegenstück:
Überlappung der nächsten Nachbarn statt metrischer Werte.
"""

import numpy as np
import torch


@torch.no_grad()
def qk_metrik(model):
    d = model.cfg.n_embd
    W = model.h[0].attn.c_attn.weight
    M = W[:d].T @ W[d:2 * d] / model.cfg.n_head  # Summe über Köpfe der Blöcke W_q,hᵀ W_k,h
    return (M + M.T) / 2


@torch.no_grad()
def qk_aehnlichkeit(model, wort_ids):
    """Ähnlichkeit in der Lesung von Block 0 (bei Bβ also durch die farbige Linse); Biases bleiben außen vor."""
    E = model.archetyp()[torch.as_tensor(list(wort_ids), device=model.wte.weight.device)]
    X = model.h[0].les_1(E).double()
    return (X @ qk_metrik(model).double() @ X.T).cpu().numpy()


def nachbar_ueberlappung(A, B, k=10):
    """Mittlere Jaccard-Überlappung der k nächsten Nachbarn (ohne das Wort selbst); NaN zählt als fern."""
    def nachbarn(M):
        M = np.where(np.isfinite(M), M, -np.inf).astype(np.float64)
        np.fill_diagonal(M, -np.inf)
        return np.argsort(-M, axis=1)[:, :k]

    na, nb = nachbarn(np.asarray(A)), nachbarn(np.asarray(B))
    j = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nb)]
    return float(np.mean(j))