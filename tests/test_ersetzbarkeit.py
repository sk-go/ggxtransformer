import numpy as np
import torch

from chroma.messung import ersetzbarkeit as ers
from chroma.modell import ChromaConfig, ChromaModell

KW = dict(n_kontexte=5, min_kontexte=3, fenster=12, nach=3, device="cpu", seed=0, batch=64)


def mini():
    torch.manual_seed(0)
    model = ChromaModell(ChromaConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)  # deutlich verschiedene Verteilungen, damit KL > 0 sicher messbar ist
    ids = np.random.default_rng(0).integers(0, 19, size=3000).astype(np.uint16)  # Token 19 kommt nie vor
    return model, ids


def test_G_monoton_ist_eins():
    c = np.random.default_rng(0).random(500)
    assert abs(ers.gleichartigkeit_G(c, c ** 3) - 1.0) < 1e-12


def test_G_entartet_ist_nan():
    c = np.random.default_rng(0).random(50)
    assert np.isnan(ers.gleichartigkeit_G(c, np.ones(50)))


def test_G_ignoriert_nan():
    c = np.random.default_rng(0).random(200)
    S = c.copy()
    S[:20] = np.nan
    assert abs(ers.gleichartigkeit_G(c, S) - 1.0) < 1e-12


def test_ziehe_paare():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((40, 8))
    paare = ers.ziehe_paare(ers.kosinus_matrix(E), 100, rng)
    assert paare.shape == (100, 2)
    assert (paare[:, 0] < paare[:, 1]).all()
    assert len({tuple(p) for p in paare}) == 100


def test_divergenz_null_bei_selbstersetzung_und_fehlende_kontexte():
    model, ids = mini()
    D, anzahl = ers.gerichtete_divergenz(model, ids, {3: [3, 4], 19: [3]}, **KW)
    assert abs(D[(3, 3)]) < 1e-6
    assert D[(3, 4)] > 0
    assert anzahl[19] == 0 and (19, 3) not in D


def test_ersetzbarkeit_meldet_woerter_ohne_kontexte():
    model, ids = mini()
    woerter = [3, 4, 19]
    S, info = ers.ersetzbarkeit(model, ids, woerter, np.array([[0, 1], [0, 2]]), **KW)
    assert np.isfinite(S[0]) and np.isnan(S[1])
    assert info["woerter_ohne_kontexte"] == [19]
    assert info["paare_gueltig"] == 1 and info["paare_gesamt"] == 2


def test_ersetzbarkeit_matrix_symmetrisch():
    model, ids = mini()
    M, _ = ers.ersetzbarkeit_matrix(model, ids, [3, 4, 5], **KW)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T, equal_nan=True)


def test_kontexte_je_wort_unabhaengig_von_anderen_woertern():
    """Dasselbe Quellwort bekommt dieselben Kontexte, egal welche Wörter sonst gemessen werden."""
    model, ids = mini()
    kw = {**KW, "n_kontexte": 3}  # weniger Kontexte als Treffer: die Auswahl hängt vom RNG ab
    D1, _ = ers.gerichtete_divergenz(model, ids, {3: [4]}, **kw)
    D2, _ = ers.gerichtete_divergenz(model, ids, {3: [4], 1: [6], 2: [7]}, **kw)
    assert D1[(3, 4)] == D2[(3, 4)]


def test_G_mitte_ueber_alle_paare():
    """Die Mittelbereichsmaske wird auf den Rängen ALLER Paare gesetzt, erst danach fallen NaN-S heraus."""
    n = 1000
    c = np.arange(n, dtype=float) / n   # Kosinus, aufsteigend
    S = c.copy()
    S[:300] = np.nan                     # die kleinsten 30 % haben kein S
    # Im oberen Bereich wird S oberhalb des 90. Perzentils aller Paare gegenläufig: das darf nicht zählen.
    S[905:] = -c[905:]
    # Richtig: Maske [10 %, 90 %] aller Paare → übrig sind Indizes 300..899, alle monoton → G = 1.
    # Falsch (Ränge nur über gültige Paare): Mitte von 300..999 reicht bis ~930 und enthält Gegenläufige.
    assert abs(ers.gleichartigkeit_G(c, S) - 1.0) < 1e-12
