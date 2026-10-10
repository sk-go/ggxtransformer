import json

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from chroma import vorstudie

VOCAB = {f"Ġ{w}": i for i, w in enumerate(["haus", "baum", "katze", "hund", "stein", "fluss", "berg", "licht",
                                            "farbe", "zeit", "art", "folge"])}


def mini_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100, n_positions=32,
                                  bos_token_id=0, eos_token_id=0)).eval()


def seqs():
    rng = np.random.default_rng(0)  # schiefe Verteilung: Häufigkeiten variieren
    w = 1.0 / np.arange(1, 101)
    return rng.choice(100, size=(6, 32), p=w / w.sum())


def test_messe_hf_modell_struktur():
    erg, E = vorstudie.messe_hf_modell(mini_gpt2(), seqs(), VOCAB, n_drehungen=3, n_woerter=10)
    for k in ("kurtosis", "kurtosis_ohne_pos0", "kontrolle", "v3_rho", "v3_p", "e4_rho", "effektiver_rang",
              "korrelationsdimension", "ausreisser_dims"):
        assert k in erg
    assert len(erg["kurtosis"]) == 2 and len(erg["kontrolle"]) == 2
    assert E.shape[1] == 32
    assert isinstance(erg["e4_rho"], float) and np.isfinite(erg["e4_rho"])
    assert len(erg["kontrolle_max"]) == 2
    assert all(m >= k for m, k in zip(erg["kontrolle_max"], erg["kontrolle"]))
    assert "v3_rho_mit_pos0" in erg


def test_eingepflanzte_achse_wird_gefunden():
    m = mini_gpt2()
    with torch.no_grad():
        m.transformer.wte.weight[:, 0] += 20.0
    erg, _ = vorstudie.messe_hf_modell(m, seqs(), VOCAB, n_drehungen=3, n_woerter=10)
    assert erg["kurtosis"][0] > erg["kontrolle"][0] + 1.0
    assert 0 in erg["ausreisser_dims"]


def test_bericht(tmp_path):
    ordner = tmp_path / "mini"
    ordner.mkdir()
    for rev in ("step0", "step1"):
        erg, E = vorstudie.messe_hf_modell(mini_gpt2(), seqs(), VOCAB, n_drehungen=2, n_woerter=10)
        (ordner / f"{rev}.json").write_text(json.dumps({**erg, "revision": rev}))
        np.save(ordner / f"{rev}_woerter.npy", E)
    text = vorstudie.vorstudie_bericht(tmp_path)
    assert "V1" in text and "explorativ" in text and "E1" in text
    assert (tmp_path / "bericht.md").exists()


def _bericht_fuer(m, tmp_path):
    erg, E = vorstudie.messe_hf_modell(m, seqs(), VOCAB, n_drehungen=3, n_woerter=10)
    ordner = tmp_path / "mini"
    ordner.mkdir(parents=True)
    (ordner / "step0.json").write_text(json.dumps({**erg, "revision": "step0"}))
    np.save(ordner / "step0_woerter.npy", E)
    return vorstudie.vorstudie_bericht(tmp_path)


def test_v1_je_schicht(tmp_path):
    # Die Achse in wte läuft über den Residualstrom in alle Blöcke: beide Schichten gelten als V1 "ja".
    m = mini_gpt2()
    with torch.no_grad():
        m.transformer.wte.weight[:, 0] += 20.0
    text = _bericht_fuer(m, tmp_path / "a")
    assert "V1: 2 von 2 Schichten" in text
    assert [z for z in text.splitlines() if z.startswith("| 0 |")][0].endswith("| ja |")
    # Ohne eingepflanzte Achse: keine Schicht
    text = _bericht_fuer(mini_gpt2(), tmp_path / "b")
    assert "V1: 0 von 2 Schichten" in text


def test_json_ohne_nan():
    assert vorstudie._endlich({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
