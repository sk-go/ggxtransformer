import numpy as np
import torch

from chroma.messung import metrik, normen, register, verschachtelung
from chroma.modell import ChromaConfig, ChromaModell


def test_kophenetisch_verschachtelt_hoeher_als_zufall():
    rng = np.random.default_rng(0)
    gross = rng.normal(size=(3, 20)) * 3
    klein = gross[:, None] + rng.normal(size=(3, 3, 20))
    punkte = (klein[:, :, None] + 0.1 * rng.normal(size=(3, 3, 10, 20))).reshape(90, 20)
    c_nest = verschachtelung.kophenetische_korrelation(punkte)
    c_zufall = verschachtelung.kophenetische_korrelation(rng.normal(size=(90, 20)))
    assert c_nest > 0.8 and c_nest > c_zufall


def test_qk_metrik_symmetrisch():
    torch.manual_seed(0)
    m = ChromaModell(ChromaConfig(vocab_size=30, block_size=8, n_layer=1, n_head=2, n_embd=8, farbe="beta"))
    M = metrik.qk_metrik(m)
    assert M.shape == (8, 8) and torch.allclose(M, M.T)
    A = metrik.qk_aehnlichkeit(m, [1, 2, 3, 4])
    assert A.shape == (4, 4) and np.allclose(A, A.T, atol=1e-6)


def test_nachbar_ueberlappung():
    rng = np.random.default_rng(0)
    A = rng.random((20, 20))
    A = A + A.T
    assert metrik.nachbar_ueberlappung(A, A, k=5) == 1.0
    B = A.copy()
    B[0, 1] = B[1, 0] = np.nan
    assert 0.0 <= metrik.nachbar_ueberlappung(A, B, k=5) <= 1.0


def test_stabile_cluster():
    rng = np.random.default_rng(0)
    basis_labels = np.repeat(np.arange(10), 10)
    umbenannt = [(basis_labels + s) % 10 for s in range(5)]  # gleiche Mengen, andere Nummern
    assert len(register.stabile_cluster(umbenannt)) == 10
    zufall = [rng.integers(0, 10, 100) for _ in range(5)]
    assert len(register.stabile_cluster(zufall)) == 0


def test_register_datei(tmp_path):
    cl = [{"mitglieder": [0, 2], "seeds": 4}]
    register.schreibe_register(cl, [" haus", " baum", " stein"], tmp_path / "register.md")
    text = (tmp_path / "register.md").read_text()
    assert "haus" in text and "stein" in text and "explorativ" in text


def test_simlex(tmp_path):
    datei = tmp_path / "SimLex-999.txt"
    datei.write_text("word1\tword2\tPOS\tSimLex999\nhaus\tbaum\tN\t2.0\nhaus\tstein\tN\t5.0\n"
                     "baum\tstein\tN\t8.0\nhaus\tfehlt\tN\t1.0\n")
    simlex = normen.lade_simlex(datei)
    assert len(simlex) == 4
    vocab = {"Ġhaus": 0, "Ġbaum": 1, "Ġstein": 2}
    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    erg = normen.simlex_vergleich(simlex, vocab, E)
    assert erg["n"] == 3 and abs(erg["abdeckung"] - 0.75) < 1e-12
    assert np.isfinite(erg["rho"])
