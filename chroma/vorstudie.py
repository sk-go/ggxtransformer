"""
chroma/vorstudie.py
Genealogie der Achsen (Spec § 3): Liegt im sedimentierten Archetyp von GPT-2 und Pythia schon
eine bevorzugte Basis, und entsteht sie durch Gewohnheit? Präregistriert: V1–V3 (PRAEREG_8).
Explorativ: E1 Nachträglichkeit, E2 fraktale Dimension, E3 Knotenlinie, E4 Reflex.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .korpus import wortliste
from .messung import basis

PYTHIA_STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000]
PYTHIA_DICHT = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000,
                16000, 20000, 25000, 33000, 43000, 53000, 66000, 83000, 100000, 120000, 143000]


def effektiver_rang(E):
    lam = np.linalg.svd(E - E.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def korrelationsdimension(P, n_skalen=8):
    """Lokale Steigung von log C(r) gegen log r (Grassberger-Procaccia), Sehnenabstand normierter Vektoren."""
    P = np.asarray(P, dtype=np.float64)
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    d = pdist(P)
    d = d[d > 0]
    r = np.quantile(d, np.geomspace(0.01, 0.5, n_skalen + 1))
    C = np.array([(d < ri).mean() for ri in r])
    ok = C > 0
    steigung = np.diff(np.log(C[ok])) / np.diff(np.log(r[ok]))
    return {"radien": r[ok].tolist(), "dimension": steigung.tolist()}


def _spearman(x, y):
    """Spearman-ρ; None, wenn weniger als 3 Paare oder eine Seite konstant ist (kein ConstantInputWarning)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return spearmanr(x, y)


def _endlich(o):
    """Nicht endliche Zahlen werden zu None, damit nie ungültiges JSON entsteht."""
    if isinstance(o, dict):
        return {k: _endlich(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_endlich(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _v3_e4(strom, tokens, freq, n_ausreisser):
    """V3 (Ausreißerdimensionen gegen Frequenz) und E4 (Participation Ratio gegen log-Frequenz) auf einem Strom."""
    dims = np.argsort(-np.abs(strom).mean(axis=0))[:n_ausreisser]
    betrag = np.abs(strom[:, dims]).sum(axis=1)
    typen = np.unique(tokens)
    mittel_betrag = np.array([betrag[tokens == t].mean() for t in typen])
    v3 = _spearman(mittel_betrag, freq[typen])
    pr = basis.participation_ratio(strom)
    haeufig = typen[freq[typen] >= 5]
    e4 = _spearman([pr[tokens == t].mean() for t in haeufig], np.log(freq[haeufig])) if len(haeufig) else None
    return dims, v3, e4


def _rho(r):
    return float(r.statistic) if r is not None else None


def _p(r):
    return float(r.pvalue) if r is not None else None


@torch.no_grad()
def messe_hf_modell(model, seqs, vocab, device="cpu", batch=8, n_drehungen=10, seed=0, n_ausreisser=3,
                    n_woerter=2000):
    rng = np.random.default_rng(seed)
    model = model.to(device).eval()
    n_layer, d = model.config.num_hidden_layers, model.config.hidden_size
    mitte = n_layer // 2
    seqs = np.asarray(seqs, dtype=np.int64)
    summe, summe0, zeilen, zeilen0 = np.zeros(n_layer), np.zeros(n_layer), 0, 0
    kontrolle = kontrolle_max = None
    mitte_strom = []
    for i in range(0, len(seqs), batch):
        x = torch.as_tensor(seqs[i:i + batch], device=device)
        hs = model(x, output_hidden_states=True).hidden_states[:n_layer]  # Eingang jedes Blocks
        for l, h in enumerate(hs):
            h = h.double().cpu().numpy()
            kz = basis.zeilen_kurtosis(h.reshape(-1, d)).reshape(h.shape[:2])
            summe[l] += kz.sum()
            summe0[l] += kz[:, 1:].sum()
            if l == mitte:
                mitte_strom.append(h.astype(np.float32))
        zeilen += x.numel()
        zeilen0 += x.shape[0] * (x.shape[1] - 1)
        if kontrolle is None:  # Drehungskontrolle auf dem ersten Batch
            ks = [np.asarray(basis.kurtosis_kontrolle(h.double().cpu().numpy().reshape(-1, d), n_drehungen, rng))
                  for h in hs]
            kontrolle = [float(k.mean()) for k in ks]
            kontrolle_max = [float(k.max()) for k in ks]
    strom_voll = np.concatenate(mitte_strom)  # (N, T, d)
    V = model.get_input_embeddings().weight.shape[0]
    freq = np.bincount(seqs.reshape(-1), minlength=V)  # Korpus-Häufigkeiten aller Token

    # V3/E4 primär ohne Position 0 (Sink-Position), zusätzlich mit
    dims, v3, e4 = _v3_e4(strom_voll[:, 1:].reshape(-1, d), seqs[:, 1:].reshape(-1), freq, n_ausreisser)
    dims_m, v3_m, e4_m = _v3_e4(strom_voll.reshape(-1, d), seqs.reshape(-1), freq, n_ausreisser)

    # E1/E2 (explorativ): Embedding der Wortliste
    W = model.get_input_embeddings().weight.double().cpu().numpy()
    woerter = wortliste(vocab, freq, n_woerter)
    E = W[woerter]
    erg = {
        "kurtosis": (summe / zeilen).tolist(),
        "kurtosis_ohne_pos0": (summe0 / zeilen0).tolist(),
        "kontrolle": kontrolle,
        "kontrolle_max": kontrolle_max,
        "ausreisser_dims": dims.tolist(),
        "ausreisser_dims_mit_pos0": dims_m.tolist(),
        "v3_rho": _rho(v3), "v3_p": _p(v3),
        "v3_rho_mit_pos0": _rho(v3_m), "v3_p_mit_pos0": _p(v3_m),
        "e4_rho": _rho(e4), "e4_rho_mit_pos0": _rho(e4_m),
        "effektiver_rang": effektiver_rang(W),
        "korrelationsdimension": korrelationsdimension(E) if len(E) >= 3 else None,
        "n_woerter": len(woerter),
    }
    return erg, E


def lauf_vorstudie(modell_name, revisionen, seqs_pfad, aus_ordner, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seqs = np.load(seqs_pfad)["A"]
    vocab = AutoTokenizer.from_pretrained(modell_name).get_vocab()
    ordner = Path(aus_ordner) / modell_name.replace("/", "__")
    ordner.mkdir(parents=True, exist_ok=True)
    for rev in revisionen:
        model = AutoModelForCausalLM.from_pretrained(modell_name, revision=rev, dtype=torch.float32)
        erg, E = messe_hf_modell(model, seqs, vocab, device=device)
        text = json.dumps(_endlich({**erg, "modell": modell_name, "revision": rev}), indent=2, allow_nan=False)
        (ordner / f"{rev}.json").write_text(text)
        np.save(ordner / f"{rev}_woerter.npy", E)
        del model


def _schritt(rev):
    return int(rev[4:]) if rev.startswith("step") else 10 ** 9


def vorstudie_bericht(aus_ordner, v_schwelle=1.0):
    aus = Path(aus_ordner)
    zeilen = ["# Vorstudie: Genealogie der Achsen", "",
              f"V-Schwelle (Festsetzung aus PRAEREG_8): {v_schwelle}", ""]
    for ordner in sorted(p for p in aus.iterdir() if p.is_dir()):
        laeufe = sorted((json.loads(f.read_text()) for f in ordner.glob("*.json")), key=lambda e: _schritt(e["revision"]))
        if not laeufe:
            continue
        zeilen += [f"## {ordner.name}", ""]
        n = len(laeufe[0]["kurtosis"])

        def delta(e):  # Abstand ohne Pos. 0 zur Kontrollobergrenze je Schicht
            kmax = e.get("kontrolle_max") or e["kontrolle"]
            return np.array(e["kurtosis_ohne_pos0"]) - np.array(kmax)

        for e in laeufe:
            ueber = delta(e) > v_schwelle
            zeilen += [f"### {e['revision']}", "",
                       "| Schicht | Kurtosis | ohne Pos. 0 | Kontrolle Mittel | Kontrolle Max | V1 |", "|---|---|---|---|---|---|"]
            kmax = e.get("kontrolle_max") or e["kontrolle"]
            for l in range(n):
                zeilen.append(f"| {l} | {e['kurtosis'][l]:.3f} | {e['kurtosis_ohne_pos0'][l]:.3f} | "
                              f"{e['kontrolle'][l]:.3f} | {kmax[l]:.3f} | {'ja' if ueber[l] else 'nein'} |")
            v3 = "–" if e["v3_rho"] is None else f"{e['v3_rho']:.3f} (p = {e['v3_p']:.3g})"
            zeilen += ["", f"V1: {int(ueber.sum())} von {n} Schichten", f"V3 ρ (ohne Position 0): {v3}", ""]
        if len(laeufe) > 1:
            auf_niveau = delta(laeufe[0]) <= v_schwelle
            anstieg = np.any([delta(e) > v_schwelle for e in laeufe[1:]], axis=0)
            v2 = auf_niveau & anstieg
            zeilen += [f"V2 (erster Checkpoint je Schicht auf Kontrollniveau, später Anstieg): "
                       f"{int(v2.sum())} von {n} Schichten (Schichten: {np.flatnonzero(v2).tolist()})"]
        zeilen += ["", "### Weitere Maße (explorativ, nicht präregistriert entscheidend)", ""]
        mats = [np.load(ordner / f"{e['revision']}_woerter.npy") for e in laeufe]
        for (e1, m1), (e2, m2) in zip(zip(laeufe, mats), zip(laeufe[1:], mats[1:])):
            rho = spearmanr(pdist(m1, "cosine"), pdist(m2, "cosine")).statistic
            zeilen.append(f"- E1 Nachträglichkeit {e1['revision']} → {e2['revision']}: ρ = {rho:.3f}; "
                          f"effektiver Rang {e1['effektiver_rang']:.1f} → {e2['effektiver_rang']:.1f}")
        for e in laeufe:
            kd = e["korrelationsdimension"]
            dims = ", ".join(f"{v:.2f}" for v in kd["dimension"]) if kd else "–"
            zeilen.append(f"- E2/E4 {e['revision']}: Korrelationsdimension je Skala [{dims}]; E4 ρ = {e['e4_rho']}")
        zeilen += ["- E3 Knotenlinie: Verlauf der Kurtosis-Spalte oben (dichtes Raster mit --dicht).", ""]
    text = "\n".join(zeilen) + "\n"
    (aus / "bericht.md").write_text(text, encoding="utf-8")
    return text
