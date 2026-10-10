"""
chroma/messung/normen.py
Vergleich (a): menschliche Ähnlichkeitsnormen (SimLex-999). Entscheidet nichts, denn es setzte
voraus, dass die Maschine eine perfekte Metapher menschlicher Erfahrung ist.
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"


def lade_simlex_herunter(ziel_ordner):
    ziel = Path(ziel_ordner)
    ziel.mkdir(parents=True, exist_ok=True)
    zip_pfad = ziel / "SimLex-999.zip"
    urllib.request.urlretrieve(SIMLEX_URL, zip_pfad)
    with zipfile.ZipFile(zip_pfad) as z:
        z.extract("SimLex-999/SimLex-999.txt", ziel)
    return ziel / "SimLex-999" / "SimLex-999.txt"


def lade_simlex(pfad):
    zeilen = Path(pfad).read_text(encoding="utf-8").splitlines()[1:]
    paare = []
    for z in zeilen:
        teile = z.split("\t")
        if len(teile) >= 4:
            paare.append((teile[0], teile[1], float(teile[3])))
    return paare


def simlex_vergleich(simlex, vocab, E):
    """Nur Paare, deren beide Wörter einzelne Ganzwort-Tokens sind; Abdeckung wird berichtet."""
    E = np.asarray(E, dtype=np.float64)
    treffer = [(vocab["Ġ" + a], vocab["Ġ" + b], s) for a, b, s in simlex
               if "Ġ" + a in vocab and "Ġ" + b in vocab]
    erg = {"n": len(treffer), "abdeckung": len(treffer) / max(len(simlex), 1), "rho": float("nan")}
    if len(treffer) >= 3:
        En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        cos = [float(En[i] @ En[j]) for i, j, _ in treffer]
        erg["rho"] = float(spearmanr(cos, [s for _, _, s in treffer]).statistic)
    return erg