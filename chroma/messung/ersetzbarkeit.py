"""
chroma/messung/ersetzbarkeit.py
Kernmaß (b): Gleichartig ist, was die Warte als gleich behandelt (Gleichsetzung des Ungleichen).
Man tauscht x gegen y und misst die Ent-Täuschung der Erwartung über die folgenden Positionen
(KL-Divergenz). G = Übereinstimmung von Anlage (Kosinus im Archetyp) und Vollzug (Ersetzbarkeit)
im mittleren Bereich, also in den Graden zwischen Gleichheit und Verschiedenheit (Spec § 4.5).
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def kosinus_matrix(E):
    E = np.asarray(E, dtype=np.float64)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return En @ En.T


def ziehe_paare(cos, n_paare, rng, n_schichten=10):
    """Paare (i < j), geschichtet nach den Kosinus-Quantilen dieses Modells."""
    iu = np.triu_indices(len(cos), 1)
    werte = cos[iu]
    grenzen = np.quantile(werte, np.linspace(0, 1, n_schichten + 1))
    schicht = np.clip(np.searchsorted(grenzen, werte, side="right") - 1, 0, n_schichten - 1)
    pro = n_paare // n_schichten
    wahl = []
    for s in range(n_schichten):
        idx = np.flatnonzero(schicht == s)
        wahl.append(rng.choice(idx, size=min(pro, len(idx)), replace=False))
    w = np.concatenate(wahl)
    return np.stack([iu[0][w], iu[1][w]], axis=1)


def finde_kontexte(ids, wort, n, fenster, nach, rng):
    """Startindizes von Fenstern, in denen `wort` an Position fenster − nach − 1 steht."""
    p = fenster - nach - 1
    treffer = np.flatnonzero(np.asarray(ids[p:len(ids) - nach]) == wort) + p
    if len(treffer) > n:
        treffer = np.sort(rng.choice(treffer, size=n, replace=False))
    return treffer - p


@torch.no_grad()
def _logprobs(model, fenster_ids, p, nach, device, batch):
    teile = []
    for i in range(0, len(fenster_ids), batch):
        x = torch.as_tensor(fenster_ids[i:i + batch], dtype=torch.long, device=device)
        teile.append(F.log_softmax(model(x)[:, p:p + nach].float(), dim=-1).cpu())
    return torch.cat(teile)


def gerichtete_divergenz(model, ids, gerichtet, n_kontexte=20, min_kontexte=5, fenster=64, nach=8,
                         device="cpu", seed=0, batch=256):
    """D[(x, y)] = mittlere KL(p_x ‖ p_{x→y}) über Kontexte und Folgepositionen.
    Quellen mit weniger als min_kontexte Kontexten werden übersprungen (anzahl hält sie fest)."""
    model.eval()
    rng = np.random.default_rng(seed)
    p = fenster - nach - 1
    D, anzahl = {}, {}
    for x in sorted(gerichtet):
        starts = finde_kontexte(ids, x, n_kontexte, fenster, nach, rng)
        anzahl[x] = int(len(starts))
        if len(starts) < min_kontexte:
            continue
        fen = np.stack([np.asarray(ids[s:s + fenster], dtype=np.int64) for s in starts])
        lp_x = _logprobs(model, fen, p, nach, device, batch)
        p_x = lp_x.exp()
        ziele = list(gerichtet[x])
        pro_gruppe = max(1, batch // len(fen))
        for j in range(0, len(ziele), pro_gruppe):
            gruppe = ziele[j:j + pro_gruppe]
            ers = np.repeat(fen[None], len(gruppe), axis=0)
            ers[:, :, p] = np.asarray(gruppe)[:, None]
            lp_y = _logprobs(model, ers.reshape(-1, fenster), p, nach, device, batch)
            lp_y = lp_y.view(len(gruppe), len(fen), nach, -1)
            kl = (p_x[None] * (lp_x[None] - lp_y)).sum(-1).mean(dim=(1, 2))
            for y, v in zip(gruppe, kl.tolist()):
                D[(x, int(y))] = float(v)
    return D, anzahl


def ersetzbarkeit(model, ids, woerter, paare, **kw):
    """S = −(D(x→y) + D(y→x)) / 2; NaN, wenn eine Richtung fehlt."""
    woerter = [int(w) for w in woerter]
    paar_ids = [(woerter[i], woerter[j]) for i, j in paare]
    gerichtet = defaultdict(list)
    for a, b in paar_ids:
        gerichtet[a].append(b)
        gerichtet[b].append(a)
    D, anzahl = gerichtete_divergenz(model, ids, gerichtet, **kw)
    S = np.array([-(D[(a, b)] + D[(b, a)]) / 2 if (a, b) in D and (b, a) in D else np.nan
                  for a, b in paar_ids])
    min_k = kw.get("min_kontexte", 5)
    info = {
        "woerter_ohne_kontexte": sorted(x for x, k in anzahl.items() if k < min_k),
        "paare_gueltig": int(np.isfinite(S).sum()),
        "paare_gesamt": int(len(paar_ids)),
    }
    return S, info


def ersetzbarkeit_matrix(model, ids, woerter, **kw):
    n = len(woerter)
    paare = np.array(list(combinations(range(n), 2)))
    S, info = ersetzbarkeit(model, ids, woerter, paare, **kw)
    M = np.full((n, n), np.nan)
    M[paare[:, 0], paare[:, 1]] = S
    M[paare[:, 1], paare[:, 0]] = S
    return M, info


def gleichartigkeit_G(cos_werte, S, mitte=0.8):
    cos_werte = np.asarray(cos_werte, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ok = np.isfinite(S) & np.isfinite(cos_werte)
    c, s = cos_werte[ok], S[ok]
    if len(c) < 3:
        return float("nan")
    rang = c.argsort().argsort() / (len(c) - 1)
    rand = (1 - mitte) / 2
    m = (rang >= rand) & (rang <= 1 - rand)
    if m.sum() < 3 or np.ptp(c[m]) == 0 or np.ptp(s[m]) == 0:
        return float("nan")
    return float(spearmanr(c[m], s[m]).statistic)
