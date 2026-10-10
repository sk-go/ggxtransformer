"""
chroma/auswertung.py
Entscheidungsregeln der Hauptstudie (Spec § 4.6). Einheit der Inferenz ist der Seed
(verschiedene Biographien desselben Gattungswesens). δ ist eine Festsetzung.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from .arme import ARME


def lade_messungen(haupt):
    erg = {}
    for f in Path(haupt).glob("*/seed*/messung_stufe_*.json"):
        arm, seed = f.parent.parent.name, int(f.parent.name[4:])
        stufe = float(f.stem.rsplit("_", 1)[-1])
        erg[(arm, seed, stufe)] = json.loads(f.read_text(encoding="utf-8"))
    return erg


def gemeinsame_stufe(messungen, arme=tuple(ARME)):
    """Die niedrigste Loss-Stufe, die alle Seeds aller Arme erreicht haben; None, wenn es keine gibt."""
    pro = {}
    for arm, seed, stufe in messungen:
        pro.setdefault((arm, seed), set()).add(stufe)
    if not pro or any(a not in {k[0] for k in pro} for a in arme):
        return None
    gemeinsam = set.intersection(*pro.values())
    return min(gemeinsam) if gemeinsam else None


def _wert(m, schluessel):
    v = m.get(schluessel)
    if isinstance(v, list):
        v = float(np.mean(v)) if v else None
    return float("nan") if v is None else float(v)


def werte(messungen, arm, stufe, schluessel):
    alle = [_wert(m, schluessel) for (a, _, s), m in sorted(messungen.items()) if a == arm and s == stufe]
    w = np.array([v for v in alle if np.isfinite(v)])
    return w, len(alle) - len(w)


def entscheide_p1(g_a0, g_a, delta=0.05, alpha=0.05):
    erg = {"n_a0": int(len(g_a0)), "n_a": int(len(g_a)), "p_differenz": None, "p_tost": None}
    if len(g_a0) < 2 or len(g_a) < 2:
        return {**erg, "urteil": "nicht auswertbar"}
    p_diff = ttest_ind(g_a0, g_a, equal_var=False, alternative="less").pvalue
    p_unten = ttest_ind(g_a0 + delta, g_a, equal_var=False, alternative="greater").pvalue
    p_oben = ttest_ind(g_a0 - delta, g_a, equal_var=False, alternative="less").pvalue
    p_tost = max(p_unten, p_oben)
    urteil = "bestätigt" if p_diff < alpha else "widerlegt" if p_tost < alpha else "unentschieden"
    return {**erg, "urteil": urteil, "p_differenz": float(p_diff), "p_tost": float(p_tost)}


def entscheide_p2(pr_b, pr_a_ica, g_b, g_a, delta=0.05, alpha=0.05):
    if min(len(pr_b), len(pr_a_ica), len(g_b), len(g_a)) < 2:
        return {"lesbar": None, "p_lesbar": None, "preis_ok": None, "p_preis": None}
    p_les = ttest_ind(pr_b, pr_a_ica, equal_var=False, alternative="less").pvalue
    p_preis = ttest_ind(g_b + delta, g_a, equal_var=False, alternative="greater").pvalue
    return {"lesbar": bool(p_les < alpha), "p_lesbar": float(p_les),
            "preis_ok": bool(p_preis < alpha), "p_preis": float(p_preis)}


def bericht(haupt, delta=0.05, alpha=0.05, schwelle_a0=2.0, schwelle_a=10.0):
    haupt = Path(haupt)
    m = lade_messungen(haupt)
    stufe = gemeinsame_stufe(m)
    zeilen = ["# Auswertung Hauptstudie (Chroma)", "",
              f"δ = {delta} (Festsetzung), α = {alpha}, Einheit: Seed.", ""]
    erg = {"stufe": stufe}
    if stufe is None:
        zeilen.append("**Es gibt keine gemeinsame Stufe**, die alle Seeds aller Arme erreicht haben. "
                      "Keine Entscheidung möglich; als Einschränkung berichten.")
        (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
        return erg
    zeilen.append(f"Primäre Loss-Stufe: {stufe:.2f}")
    zeilen.append("")

    k_a0, _ = werte(m, "A0", stufe, "kurtosis")
    k_a, _ = werte(m, "A", stufe, "kurtosis")
    bestanden = bool(len(k_a0) and len(k_a) and k_a0.mean() < schwelle_a0 and k_a.mean() > schwelle_a)
    erg["manipulation"] = {"bestanden": bestanden, "kurtosis_a0": k_a0.tolist(), "kurtosis_a": k_a.tolist()}
    zeilen += ["## Manipulationsprüfung", "",
               f"Kurtosis A₀ Mittel {k_a0.mean() if len(k_a0) else float('nan'):.3f} (Schwelle < {schwelle_a0}), "
               f"A Mittel {k_a.mean() if len(k_a) else float('nan'):.3f} (Schwelle > {schwelle_a}): "
               f"**{'bestanden' if bestanden else 'NICHT bestanden – Vergleich nichtig'}**", ""]

    g = {arm: werte(m, arm, stufe, "G") for arm in ARME}
    erg["p1"] = entscheide_p1(g["A0"][0], g["A"][0], delta, alpha)
    zeilen += ["## P1 (Gerüstfrage)", "",
               f"G(A₀) = {g['A0'][0].round(4).tolist()} (verworfen: {g['A0'][1]}), "
               f"G(A) = {g['A'][0].round(4).tolist()} (verworfen: {g['A'][1]})",
               f"p Differenz = {erg['p1']['p_differenz']}, p TOST = {erg['p1']['p_tost']} → "
               f"**{erg['p1']['urteil']}**", ""]

    pr_a_arch, _ = werte(m, "A", stufe, "pr_archetyp_ica")
    pr_a_strom, _ = werte(m, "A", stufe, "pr_strom_ica")
    erg["p2"] = {
        "Ba": entscheide_p2(werte(m, "Ba", stufe, "pr_archetyp_eigen")[0], pr_a_arch, g["Ba"][0], g["A"][0], delta, alpha),
        "Bb": entscheide_p2(werte(m, "Bb", stufe, "pr_strom_eigen")[0], pr_a_strom, g["Bb"][0], g["A"][0], delta, alpha),
    }
    zeilen += ["## P2 (Sichtbarkeitsfrage)", ""]
    for arm, e in erg["p2"].items():
        zeilen.append(f"- {arm}: lesbar = {e['lesbar']} (p = {e['p_lesbar']}), "
                      f"Preisprüfung bestanden = {e['preis_ok']} (p = {e['p_preis']})")
    zeilen.append("")

    koph = {arm: werte(m, arm, stufe, "kophenetisch")[0] for arm in ARME}
    mittel = {arm: float(v.mean()) if len(v) else float("nan") for arm, v in koph.items()}
    erg["p3"] = mittel
    zeilen += ["## P3 (Verschachtelung, sekundär, gerichtet)", "",
               ", ".join(f"{a}: {v:.4f}" for a, v in mittel.items()), ""]

    zeilen += ["## Weitere Maße (explorativ, nicht entscheidend)", ""]
    for schl in ("G_QK", "G_top"):
        zeilen.append(f"- {schl}: " + ", ".join(
            f"{arm} {np.mean(werte(m, arm, stufe, schl)[0]):.4f}" for arm in ARME))
    (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    (haupt / "auswertung.json").write_text(json.dumps(erg, indent=2, default=str), encoding="utf-8")
    return erg
