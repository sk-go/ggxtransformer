"""
chroma/messung/register.py
Register „Begriffsbedarf“ (explorativ): Cluster, die über Biographien (Seeds) hinweg stabil sind.
Der Autor benennt sie; was sich nicht benennen lässt, verlangt einen Begriff (Asynchronizität
auf der Ebene der Methode).
"""

from pathlib import Path

import numpy as np


def _mengen(labels):
    labels = np.asarray(labels)
    return [frozenset(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def stabile_cluster(labels_je_seed, min_seeds=3, min_jaccard=0.5):
    mengen = [_mengen(l) for l in labels_je_seed]
    behalten = []
    for i, ms in enumerate(mengen):
        for m in ms:
            if any(_jaccard(m, frozenset(b["mitglieder"])) >= min_jaccard for b in behalten):
                continue
            seeds = 1 + sum(any(_jaccard(m, o) >= min_jaccard for o in mengen[j])
                            for j in range(len(mengen)) if j != i)
            if seeds >= min_seeds:
                behalten.append({"mitglieder": sorted(m), "seeds": int(seeds)})
    return behalten


def schreibe_register(cluster, woerter_text, pfad):
    zeilen = [
        "# Register „Begriffsbedarf“ (explorativ)",
        "",
        "Stabile Cluster über Seeds. Spalte „Begriff“ füllt der Autor; leer gebliebene Zeilen "
        "sind Phänomene, die einen Begriff verlangen.",
        "",
        "| Nr. | Seeds | Mitglieder | Begriff |",
        "|---|---|---|---|",
    ]
    for n, c in enumerate(cluster, 1):
        mitglieder = ", ".join(woerter_text[i].strip() for i in c["mitglieder"])
        zeilen.append(f"| {n} | {c['seeds']} | {mitglieder} | |")
    Path(pfad).write_text("\n".join(zeilen) + "\n", encoding="utf-8")