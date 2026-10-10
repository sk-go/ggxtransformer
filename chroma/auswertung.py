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
    """Die niedrigste Loss-Stufe, die alle Seeds aller Arme erreicht haben; None, wenn es keine gibt.
    Ignoriert unbekannte Arme und berücksichtigt nur die angegebenen."""
    pro = {}
    for arm, seed, stufe in messungen:
        if arm in arme:
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

    # Prüfe auf degenerierte p-Werte (NaN bei Null-Varianz)
    if not np.isfinite(p_diff) or not np.isfinite(p_tost):
        return {**erg, "urteil": "nicht auswertbar", "p_differenz": float(p_diff), "p_tost": float(p_tost)}

    # Entscheidungsregel: nicht auswertbar → (beide signifikant) → bestätigt/widerlegt/unentschieden
    if p_diff < alpha and p_tost < alpha:
        urteil = "signifikant, aber innerhalb δ"
    elif p_diff < alpha:
        urteil = "bestätigt"
    elif p_tost < alpha:
        urteil = "widerlegt"
    else:
        urteil = "unentschieden"

    return {**erg, "urteil": urteil, "p_differenz": float(p_diff), "p_tost": float(p_tost)}


def entscheide_p2(pr_b, pr_a_ica, g_b, g_a, delta=0.05, alpha=0.05):
    if min(len(pr_b), len(pr_a_ica), len(g_b), len(g_a)) < 2:
        return {"lesbar": None, "p_lesbar": None, "preis_ok": None, "p_preis": None}
    p_les = ttest_ind(pr_b, pr_a_ica, equal_var=False, alternative="less").pvalue
    p_preis = ttest_ind(g_b + delta, g_a, equal_var=False, alternative="greater").pvalue

    # Prüfe auf degenerierte p-Werte
    lesbar = None if not np.isfinite(p_les) else bool(p_les < alpha)
    preis_ok = None if not np.isfinite(p_preis) else bool(p_preis < alpha)

    return {"lesbar": lesbar, "p_lesbar": float(p_les),
            "preis_ok": preis_ok, "p_preis": float(p_preis)}


def seed_status(haupt, messungen, stufe, seeds=None, arme=tuple(ARME)):
    """Je Arm: erwartete Seeds (aus den Trainingsordnern <haupt>/<arm>/seed*/, oder vorgegeben),
    abgebrochene (abbruch.json), unfertige (ohne fertig.json) und Seeds ohne Messung der primären Stufe
    (ohne gemeinsame Stufe: ohne jede Messung)."""
    haupt = Path(haupt)
    status = {}
    for arm in arme:
        if seeds is None:
            erwartet = sorted(int(d.name[4:]) for d in haupt.glob(f"{arm}/seed*")
                              if d.is_dir() and d.name[4:].isdigit())
        else:
            erwartet = sorted(int(s) for s in seeds)
        gemessen = {s for a, s, st in messungen if a == arm and (stufe is None or st == stufe)}
        ordner = {s: haupt / arm / f"seed{s}" for s in erwartet}
        status[arm] = {
            "erwartet": erwartet,
            "abbruch": [s for s in erwartet if (ordner[s] / "abbruch.json").exists()],
            "ohne_fertig": [s for s in erwartet if not (ordner[s] / "fertig.json").exists()],
            "ohne_messung": [s for s in erwartet if s not in gemessen],
        }
    return status


def _seed_status_zeilen(status, stufe):
    messung = "Messung der primären Stufe" if stufe is not None else "jede Messung"
    zeilen = ["## Seed-Status", ""]
    for arm, st in status.items():
        zeilen += [f"- {arm}: erwartet {st['erwartet']}",
                   f"  - {arm}: abgebrochen (abbruch.json): {st['abbruch']}",
                   f"  - {arm}: ohne fertig.json: {st['ohne_fertig']}",
                   f"  - {arm}: ohne {messung}: {st['ohne_messung']}"]
    return zeilen + [""]


def pruefe_konsistenz(messungen, stufe, arme=tuple(ARME)):
    """Alle Messungen einer Stufe müssen dieselbe Messkonfiguration haben und die Farbe ihres Arms tragen.
    Rückgabe: Liste von Gründen (leer = konsistent)."""
    gruende = []
    eintraege = [((a, s), m) for (a, s, st), m in sorted(messungen.items()) if a in arme and st == stufe]
    configs = {}
    for (arm, seed), m in eintraege:
        mc = (m.get("info") or {}).get("messconfig")
        if mc is None:
            gruende.append(f"Messkonfiguration fehlt: ({arm}, seed{seed})")
        else:
            configs.setdefault(json.dumps(mc, sort_keys=True), []).append(f"({arm}, seed{seed})")
        if m.get("farbe") != ARME[arm]["farbe"]:
            gruende.append(f"Farbe passt nicht zum Arm: ({arm}, seed{seed}) hat {m.get('farbe')!r}, "
                           f"erwartet {ARME[arm]['farbe']!r}")
    if len(configs) > 1:
        gruppen = sorted(configs.values(), key=len, reverse=True)
        abweichend = [x for g in gruppen[1:] for x in g]
        gruende.append(f"Messkonfiguration uneinheitlich ({len(configs)} Varianten); abweichend: "
                       + ", ".join(abweichend))
    return gruende


def _info(m, *pfad):
    v = m
    for k in pfad:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def bericht(haupt, delta=0.05, alpha=0.05, schwelle_a0=2.0, schwelle_a=10.0, seeds=None):
    haupt = Path(haupt)
    m = lade_messungen(haupt)

    # Sammle alle verfügbaren Stufen pro (Arm, Seed)
    stufen_pro_seed = {}
    for arm, seed, stufe in m:
        stufen_pro_seed.setdefault((arm, seed), set()).add(stufe)

    stufe = gemeinsame_stufe(m)
    zeilen = ["# Auswertung Hauptstudie (Chroma)", "",
              f"δ = {delta} (Festsetzung), α = {alpha}, Einheit: Seed.", ""]
    erg = {"stufe": stufe}

    # Berechne fehlende Stufen für ALLE Läufe (vor der Verzweigung)
    alle_stufen = sorted(set(s for _, _, s in m))
    erg["fehlende_stufen"] = {f"{arm}:{seed}": sorted(set(alle_stufen) - stufen_pro_seed.get((arm, seed), set()))
                              for arm, seed in stufen_pro_seed}

    # Seed-Vollständigkeit: Was fehlt, wird ausdrücklich genannt
    erg["seed_status"] = seed_status(haupt, m, stufe, seeds)

    if stufe is None:
        zeilen.append("**Es gibt keine gemeinsame Stufe**, die alle Seeds aller Arme erreicht haben. "
                      "Keine Entscheidung möglich; als Einschränkung berichten.")
        zeilen.append("")
        zeilen += _seed_status_zeilen(erg["seed_status"], stufe)

        # Berichte verfügbare und fehlende Stufen
        zeilen.append("## Verfügbare Loss-Stufen pro Arm und Seed")
        zeilen.append("")
        for arm in sorted(set(a for a, _, _ in m)):
            zeilen.append(f"**{arm}:**")
            for seed in sorted(set(s for a, s, _ in m if a == arm)):
                stufen = sorted(stufen_pro_seed.get((arm, seed), set()))
                zeilen.append(f"  - seed{seed}: {stufen}")
        zeilen.append("")
        zeilen.append("## Fehlende Stufen")
        zeilen.append("")
        for arm in sorted(set(a for a, _, _ in m)):
            for seed in sorted(set(s for a, s, _ in m if a == arm)):
                fehlend = erg["fehlende_stufen"].get(f"{arm}:{seed}", [])
                if fehlend:
                    zeilen.append(f"- ({arm}, seed{seed}): fehlen {fehlend}")

        (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
        return erg

    zeilen.append(f"Primäre Loss-Stufe: {stufe:.2f}")
    zeilen.append("")
    zeilen += _seed_status_zeilen(erg["seed_status"], stufe)

    # Konsistenz: gleiche Messkonfiguration, Farbe passt zum Arm
    erg["konsistenz"] = pruefe_konsistenz(m, stufe)
    zeilen += ["## Konsistenzprüfung", ""]
    if erg["konsistenz"]:
        zeilen += ["**Nicht konsistent – P1/P2 nicht auswertbar:**", ""] + [f"- {g}" for g in erg["konsistenz"]]
    else:
        zeilen.append("Alle Messungen dieser Stufe haben dieselbe Messkonfiguration und die Farbe ihres Arms.")
    zeilen.append("")

    # Manipulationsprüfung mit kurtosis_ohne_pos0 als Gate
    k_a0, verw_ka0_kurto = werte(m, "A0", stufe, "kurtosis")
    k_a, verw_ka_kurto = werte(m, "A", stufe, "kurtosis")
    k_a0_op, verw_ka0_op = werte(m, "A0", stufe, "kurtosis_ohne_pos0")
    k_a_op, verw_ka_op = werte(m, "A", stufe, "kurtosis_ohne_pos0")
    kontrolle = {arm: werte(m, arm, stufe, "kurtosis_kontrolle")[0] for arm in ARME}
    kontrolle_mittel = {arm: float(v.mean()) if len(v) else float("nan") for arm, v in kontrolle.items()}

    gate_daten = bool(len(k_a0_op) and len(k_a_op))
    bestanden = bool(k_a0_op.mean() < schwelle_a0 and k_a_op.mean() > schwelle_a) if gate_daten else None
    erg["manipulation"] = {
        "bestanden": bestanden,
        "kurtosis_a0": k_a0.tolist(),
        "kurtosis_a": k_a.tolist(),
        "kurtosis_ohne_pos0_a0": k_a0_op.tolist(),
        "kurtosis_ohne_pos0_a": k_a_op.tolist(),
        "kurtosis_kontrolle": kontrolle_mittel,
    }
    if bestanden is None:
        gate_text = "nicht auswertbar – Kurtosiswerte (ohne Pos. 0) fehlen für A₀ oder A"
    elif bestanden:
        gate_text = "bestanden"
    else:
        gate_text = "NICHT bestanden – Verdikt nichtig"
    zeilen += ["## Manipulationsprüfung", "",
               f"Kurtosis (ohne Pos0) A₀: Mittel {k_a0_op.mean() if len(k_a0_op) else float('nan'):.3f} (Schwelle < {schwelle_a0}), "
               f"A: Mittel {k_a_op.mean() if len(k_a_op) else float('nan'):.3f} (Schwelle > {schwelle_a})",
               f"Kurtosis (alle) A₀: Mittel {k_a0.mean() if len(k_a0) else float('nan'):.3f}, "
               f"A: Mittel {k_a.mean() if len(k_a) else float('nan'):.3f}: "
               f"**{gate_text}**",
               "Drehungskontrolle (Kurtosis nach zufälliger Drehung, dieselben Zeilen ohne Pos. 0, "
               "Mittel über Schichten): "
               + ", ".join(f"{a} {v:.3f}" if np.isfinite(v) else f"{a} –" for a, v in kontrolle_mittel.items()),
               ""]

    # Was die Urteile P1/P2 aufhebt: Inkonsistenz oder fehlendes Gate → nicht auswertbar; Gate gescheitert → nichtig
    if erg["konsistenz"]:
        aufgehoben = "nicht auswertbar"
    elif bestanden is None:
        aufgehoben = "nicht auswertbar"
    elif not bestanden:
        aufgehoben = "nichtig"
    else:
        aufgehoben = None
    erg["urteile_aufgehoben"] = aufgehoben

    # Zähle Ausfälle aller Maße
    erg["n"] = {}
    messgroessen = ["kurtosis", "kurtosis_ohne_pos0", "G", "pr_archetyp_eigen", "pr_archetyp_ica",
                    "pr_strom_eigen", "pr_strom_ica", "kophenetisch", "G_QK", "G_top"]
    for arm in ARME:
        erg["n"][arm] = {}
        for schl in messgroessen:
            w, verw = werte(m, arm, stufe, schl)
            erg["n"][arm][schl] = {"n": int(len(w)), "verworfen": int(verw)}

    # Stichprobengröße-Tabelle
    zeilen += ["## Stichprobengröße je Arm und Maß", ""]
    zeilen.append("| Maß | " + " | ".join(ARME) + " |")
    zeilen.append("|---|" + "|".join(["---"] * len(ARME)) + "|")
    for schl in messgroessen:
        zeile_eintraege = [schl]
        for arm in ARME:
            n = erg["n"][arm][schl]["n"]
            verw = erg["n"][arm][schl]["verworfen"]
            zeile_eintraege.append(f"{n} (v={verw})")
        zeilen.append("| " + " | ".join(zeile_eintraege) + " |")
    zeilen.append("")

    # Gültige Paare je Arm und Seed (Mindestquote, siehe MessConfig.min_quote_paare)
    zeilen += ["## Gültige Paare je Arm und Seed", "",
               "| Arm | Seed | Paare gültig/gesamt | Quote | Anker gültig/gesamt | G verworfen (Quote) |",
               "|---|---|---|---|---|---|"]
    erg["paare"] = {}
    for (arm, seed, st), mm in sorted(m.items()):
        if arm not in ARME or st != stufe:
            continue
        pg, pges = _info(mm, "info", "paare_gueltig"), _info(mm, "info", "paare_gesamt")
        ag, ages = _info(mm, "info", "anker", "paare_gueltig"), _info(mm, "info", "anker", "paare_gesamt")
        verw = _info(mm, "info", "G_verworfen_quote")
        erg["paare"][f"{arm}:{seed}"] = {"gueltig": pg, "gesamt": pges, "anker_gueltig": ag, "anker_gesamt": ages,
                                         "G_verworfen_quote": verw}
        quote = f"{pg / pges:.3f}" if pg is not None and pges else "–"
        zeilen.append(f"| {arm} | {seed} | {pg if pg is not None else '–'}/{pges if pges is not None else '–'} | "
                      f"{quote} | {ag if ag is not None else '–'}/{ages if ages is not None else '–'} | "
                      f"{'–' if verw is None else ('ja' if verw else 'nein')} |")
    zeilen.append("")

    # ICA-Konvergenz: Werte aus nicht konvergierten Drehungen zählen, P2 ggf. als eingeschränkt markieren
    erg["ica_nicht_konvergiert"] = {}
    for arm in ARME:
        zaehler = {"archetyp": 0, "strom": 0}
        for (a, _, st), mm in m.items():
            if a == arm and st == stufe:
                for teil in zaehler:
                    if _info(mm, "info", "ica_konvergiert", teil) is False:
                        zaehler[teil] += 1
        erg["ica_nicht_konvergiert"][arm] = zaehler
    # In P2 gehen nur die ICA-Werte von A ein
    erg["p2_eingeschraenkt"] = bool(sum(erg["ica_nicht_konvergiert"]["A"].values()) > 0)

    g = {arm: werte(m, arm, stufe, "G") for arm in ARME}
    erg["G"] = {arm: g[arm][0].tolist() for arm in ARME}
    erg["p1"] = entscheide_p1(g["A0"][0], g["A"][0], delta, alpha)
    if aufgehoben:
        erg["p1"]["urteil"] = aufgehoben

    zeilen += ["## Kernmaß G je Arm", ""]
    for arm in ARME:
        w, verw = g[arm]
        mittel_g = f"{w.mean():.4f}" if len(w) else "–"
        zeilen.append(f"- G({arm}) = {w.round(4).tolist()} (Mittel {mittel_g}, verworfen: {verw})")
    zeilen.append("")

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

    if aufgehoben:
        for arm in erg["p2"]:
            erg["p2"][arm]["lesbar"] = aufgehoben
            erg["p2"][arm]["preis_ok"] = aufgehoben

    zeilen += ["## P2 (Sichtbarkeitsfrage)", ""]
    if erg["p2_eingeschraenkt"]:
        nk = erg["ica_nicht_konvergiert"]["A"]
        zeilen += [f"**P2 eingeschränkt:** Bei A stammen {nk['archetyp']} Archetyp- und {nk['strom']} "
                   f"Strom-Werte aus nicht konvergierten ICA-Drehungen.", ""]
    for arm, e in erg["p2"].items():
        zeilen.append(f"- {arm}: lesbar = {e['lesbar']} (p = {e['p_lesbar']}), "
                      f"Preisprüfung bestanden = {e['preis_ok']} (p = {e['p_preis']})")
    zeilen.append("- ICA nicht konvergiert (Archetyp/Strom) je Arm: "
                  + ", ".join(f"{a} {v['archetyp']}/{v['strom']}" for a, v in erg["ica_nicht_konvergiert"].items()))
    zeilen.append("")

    koph = {arm: werte(m, arm, stufe, "kophenetisch")[0] for arm in ARME}
    mittel = {arm: float(v.mean()) if len(v) else float("nan") for arm, v in koph.items()}
    erg["p3"] = mittel

    # P3 Ordnung: prüfe mean(B) ≥ mean(A) ≥ mean(A0)
    erg["p3_ordnung"] = {}
    for b_arm in ("Ba", "Bb"):
        if all(np.isfinite(mittel[arm]) for arm in ["A0", "A", b_arm]):
            erg["p3_ordnung"][b_arm] = bool(mittel[b_arm] >= mittel["A"] >= mittel["A0"])
        else:
            erg["p3_ordnung"][b_arm] = None

    zeilen += ["## P3 (Verschachtelung, sekundär, gerichtet)", "",
               ", ".join(f"{a}: {v:.4f}" if np.isfinite(v) else f"{a}: –" for a, v in mittel.items()),
               f"Ordnung (B ≥ A ≥ A₀): Ba = {erg['p3_ordnung']['Ba']}, Bb = {erg['p3_ordnung']['Bb']}", ""]

    zeilen += ["## Weitere Maße (explorativ, nicht entscheidend)", ""]
    for schl in ("G_QK", "G_top"):
        werte_pro_arm = []
        for arm in ARME:
            w, _ = werte(m, arm, stufe, schl)
            if len(w):
                werte_pro_arm.append(f"{arm} {np.mean(w):.4f}")
            else:
                werte_pro_arm.append(f"{arm} –")
        zeilen.append(f"- {schl}: " + ", ".join(werte_pro_arm))
    zeilen.append("")

    # Fehlende Stufen auch hier berichten
    zeilen.append("## Fehlende Stufen")
    zeilen.append("")
    if any(erg["fehlende_stufen"].values()):
        for arm in sorted(set(a for a, _, _ in m)):
            for seed in sorted(set(s for a, s, _ in m if a == arm)):
                fehlend = erg["fehlende_stufen"].get(f"{arm}:{seed}", [])
                if fehlend:
                    zeilen.append(f"- ({arm}, seed{seed}): fehlen {fehlend}")
    else:
        zeilen.append("Keine fehlenden Stufen.")
    zeilen.append("")

    (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    (haupt / "auswertung.json").write_text(json.dumps(erg, indent=2, default=str), encoding="utf-8")
    return erg
