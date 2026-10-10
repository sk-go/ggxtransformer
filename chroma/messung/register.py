"""
chroma/messung/register.py
Register „Begriffsbedarf“ (explorativ): Cluster, die über Biographien (Seeds) hinweg stabil sind.
Der Autor benennt sie; was sich nicht benennen lässt, verlangt einen Begriff (Asynchronizität
auf der Ebene der Methode).
"""

from pathlib import Path

import numpy as np


def _mengen(labels, min_groesse=1):
    labels = np.asarray(labels)
    mengen = (frozenset(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels))
    return [m for m in mengen if len(m) >= min_groesse]


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def stabile_cluster(labels_je_seed, min_seeds=3, min_jaccard=0.5, min_groesse=2):
    """Cluster, die in mindestens min_seeds Seeds (Jaccard ≥ min_jaccard) wiederkehren.
    Cluster mit weniger als min_groesse Mitgliedern zählen nicht. Das Ergebnis hängt nicht von der
    Reihenfolge der Seeds ab: Alle Kandidaten werden in einer festen, inhaltlich bestimmten
    Reihenfolge (mehr Seeds zuerst, dann nach Mitgliedern) zu Repräsentanten zusammengefasst; jeder
    behaltene Cluster trägt die höchste Seed-Zahl aller zu ihm passenden Kandidaten."""
    mengen = [_mengen(l, min_groesse) for l in labels_je_seed]
    kandidaten = []
    for i, ms in enumerate(mengen):
        for m in ms:
            seeds = 1 + sum(any(_jaccard(m, o) >= min_jaccard for o in mengen[j])
                            for j in range(len(mengen)) if j != i)
            if seeds >= min_seeds:
                kandidaten.append((m, int(seeds)))
    kandidaten.sort(key=lambda k: (-k[1], sorted(k[0])))
    behalten = []
    for m, seeds in kandidaten:
        passend = [b for b in behalten if _jaccard(m, b[0]) >= min_jaccard]
        if passend:
            for b in passend:
                b[1] = max(b[1], seeds)
        else:
            behalten.append([m, seeds])
    return [{"mitglieder": sorted(m), "seeds": s} for m, s in behalten]


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