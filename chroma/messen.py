"""
chroma/messen.py
Alle Maße für einen Checkpoint (Spec § 4.5): Kernmaß G, Basismaße, Verschachtelung,
explorativ G_QK, G_top, Cluster fürs Register, SimLex als Vergleich.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .korpus import lade_frequenz, lade_split, lade_tokenizer, wortliste
from .messung import basis, metrik, normen, verschachtelung
from .messung import ersetzbarkeit as ers
from .training import lade_modell


@dataclass
class MessConfig:
    n_woerter: int = 2000
    n_paare: int = 20000
    n_kontexte: int = 20
    min_kontexte: int = 5
    fenster: int = 64
    nach: int = 8
    n_anker: int = 150
    k_nachbarn: int = 10
    n_fenster_strom: int = 64
    schicht_strom: int = 3
    n_cluster: int = 100
    seed: int = 0
    batch: int = 256


@torch.no_grad()
def _strom_proben(model, ids, mc, device, rng):
    T = model.cfg.block_size
    starts = rng.integers(0, len(ids) - T - 1, size=mc.n_fenster_strom)
    x = torch.as_tensor(np.stack([np.asarray(ids[s:s + T], dtype=np.int64) for s in starts]), device=device)
    zust = [z.double().cpu().numpy() for z in model.strom(x)]
    return x.cpu().numpy(), zust


def messe_checkpoint(ckpt, korpus_ordner, aus_json, mc=MessConfig(), device="cpu", simlex_pfad=None):
    model = lade_modell(ckpt, device)
    farbe = model.cfg.farbe
    tok = lade_tokenizer(korpus_ordner)
    woerter = wortliste(tok.get_vocab(), lade_frequenz(korpus_ordner), mc.n_woerter)
    test = lade_split(korpus_ordner, "test")
    rng = np.random.default_rng(mc.seed)
    kw = dict(n_kontexte=mc.n_kontexte, min_kontexte=mc.min_kontexte, fenster=mc.fenster, nach=mc.nach,
              device=device, seed=mc.seed, batch=mc.batch)

    with torch.no_grad():
        archetyp = model.archetyp().double().cpu().numpy()
    E = archetyp[woerter]
    cos = ers.kosinus_matrix(E)

    # Kernmaß (b)
    paare = ers.ziehe_paare(cos, mc.n_paare, rng)
    S, info = ers.ersetzbarkeit(model, test, woerter, paare, **kw)
    G = ers.gleichartigkeit_G(cos[paare[:, 0], paare[:, 1]], S)

    # explorativ: eigene Metrik und Topologie
    qk = metrik.qk_aehnlichkeit(model, woerter)
    G_QK = ers.gleichartigkeit_G(qk[paare[:, 0], paare[:, 1]], S)
    anker = woerter[:mc.n_anker]
    S_anker, info_anker = ers.ersetzbarkeit_matrix(model, test, anker, **kw)
    G_top = metrik.nachbar_ueberlappung(cos[:len(anker), :len(anker)], S_anker, k=mc.k_nachbarn)

    # Basis: Kurtosis je Schicht, mit und ohne Position 0, und Drehungskontrolle
    x, zust = _strom_proben(model, test, mc, device, rng)
    d = model.cfg.n_embd
    kurt = [basis.exzess_kurtosis(z.reshape(-1, d)) for z in zust]
    kurt0 = [basis.exzess_kurtosis(z[:, 1:].reshape(-1, d)) for z in zust]
    kontrolle = [float(np.mean(basis.kurtosis_kontrolle(z.reshape(-1, d)[:4096], 3, rng))) for z in zust]

    # P2: Archetyp und Stream-Lesung in Schicht schicht_strom, eigene Basis gegen ICA-Basis
    pr_arch_eigen = float(basis.participation_ratio(E).mean())
    pr_arch_ica = float(basis.participation_ratio(E @ basis.ica_drehung(E, mc.seed)).mean()) if len(E) >= E.shape[1] else float("nan")
    l = min(mc.schicht_strom, model.cfg.n_layer)
    maske = np.isin(x, woerter)
    R = zust[l][maske]
    R_lesung = F.softplus(torch.from_numpy(R)).numpy() if farbe == "beta" else R
    pr_strom_eigen = float(basis.participation_ratio(R_lesung).mean())
    pr_strom_ica = float(basis.participation_ratio(R @ basis.ica_drehung(R, mc.seed)).mean()) if len(R) >= d else float("nan")

    # Verschachtelung und Cluster fürs Register
    koph = verschachtelung.kophenetische_korrelation(E)
    labels = verschachtelung.cluster_labels(E, mc.n_cluster)

    simlex = normen.simlex_vergleich(normen.lade_simlex(simlex_pfad), tok.get_vocab(), archetyp) if simlex_pfad else None

    erg = {
        "ckpt": str(ckpt), "farbe": farbe, "G": G, "G_QK": G_QK, "G_top": G_top,
        "kurtosis": kurt, "kurtosis_ohne_pos0": kurt0, "kurtosis_kontrolle": kontrolle,
        "pr_archetyp_eigen": pr_arch_eigen, "pr_archetyp_ica": pr_arch_ica,
        "pr_strom_eigen": pr_strom_eigen, "pr_strom_ica": pr_strom_ica,
        "kophenetisch": koph, "simlex": simlex,
        "info": {**info, "anker": info_anker, "strom_zeilen": int(len(R)), "messconfig": asdict(mc)},
    }
    aus = Path(aus_json)
    aus.parent.mkdir(parents=True, exist_ok=True)
    aus.write_text(json.dumps(erg, indent=2), encoding="utf-8")  # NaN bleibt NaN (gültig für json.loads)
    np.save(aus.with_name(aus.stem + "_cluster.npy"), labels)
    return erg
