import json
from dataclasses import replace

import numpy as np
import pytest

from chroma import korpus
from chroma.messen import MessConfig, _strom_proben, messe_checkpoint
from chroma.training import TrainConfig, lade_modell, trainiere
from tests.test_korpus import synthetische_texte

SCHLUESSEL = {"G", "G_QK", "G_top", "kurtosis", "kurtosis_ohne_pos0", "kurtosis_kontrolle", "pr_archetyp_eigen",
              "pr_archetyp_ica", "pr_strom_eigen", "pr_strom_ica", "kophenetisch", "simlex", "info",
              "farbe", "ckpt"}

MC = MessConfig(n_woerter=10, n_paare=30, n_kontexte=4, min_kontexte=2, fenster=16, nach=4, n_anker=6,
                k_nachbarn=3, n_fenster_strom=16, schicht_strom=1, n_cluster=3, batch=64)


@pytest.fixture(scope="module")
def trainiert(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("messen")
    splits = {"train": synthetische_texte(600, 0), "val": synthetische_texte(80, 1), "test": synthetische_texte(300, 2)}
    korpus.baue_korpus(tmp_path, splits=splits, vocab_size=400, quelle="synthetisch")
    tc = TrainConfig(arm="Bb", seed=0, lr=3e-3, batch_size=8, block_size=32, n_layer=2, n_head=2, n_embd=16,
                     max_tokens=8 * 32 * 30, eval_alle=10, eval_batches=2)
    trainiere(tmp_path, tmp_path / "lauf", tc, log=lambda *a: None)
    return tmp_path


def test_messe_checkpoint(trainiert):
    tmp_path = trainiert
    erg = messe_checkpoint(tmp_path / "lauf" / "ende.pt", tmp_path, tmp_path / "lauf" / "messung_ende.json", MC)
    assert SCHLUESSEL <= set(erg)
    assert erg["farbe"] == "beta" and len(erg["kurtosis"]) == 3
    gespeichert = json.loads((tmp_path / "lauf" / "messung_ende.json").read_text())
    assert gespeichert["G"] == erg["G"] or (np.isnan(gespeichert["G"]) and np.isnan(erg["G"]))
    assert (tmp_path / "lauf" / "messung_ende_cluster.npy").exists()
    info = erg["info"]
    assert isinstance(info["G_verworfen_quote"], bool) and isinstance(info["G_top_verworfen_quote"], bool)
    assert isinstance(info["G_top_entfernt"], int)
    assert set(info["ica_konvergiert"]) == {"archetyp", "strom"}
    assert all(v is None or isinstance(v, bool) for v in info["ica_konvergiert"].values())
    assert len(erg["kurtosis_kontrolle"]) == 3


def test_strom_zeilen_ohne_position0_und_eigener_rng(trainiert):
    """Die Strom-Proben hängen nur von [seed, 1] ab (nicht von der Paarziehung), und R enthält keine Position 0."""
    from chroma.korpus import lade_frequenz, lade_split, lade_tokenizer, wortliste

    tmp_path = trainiert
    mc = replace(MC, n_paare=20)  # andere Paarziehung: darf die Strom-Proben nicht verändern
    erg = messe_checkpoint(tmp_path / "lauf" / "ende.pt", tmp_path, tmp_path / "lauf" / "m2.json", mc)
    model = lade_modell(tmp_path / "lauf" / "ende.pt", "cpu")
    woerter = wortliste(lade_tokenizer(tmp_path).get_vocab(), lade_frequenz(tmp_path), mc.n_woerter)
    x, _ = _strom_proben(model, lade_split(tmp_path, "test"), mc, "cpu", np.random.default_rng([mc.seed, 1]))
    assert erg["info"]["strom_zeilen"] == int(np.isin(x[:, 1:], woerter).sum())
    assert np.isin(x[:, 0], woerter).any()  # sonst prüfte der Test nichts


def test_mindestquote_verwirft_G(trainiert):
    tmp_path = trainiert
    mc = replace(MC, min_quote_paare=1.01)  # unerreichbar: alles wird verworfen
    erg = messe_checkpoint(tmp_path / "lauf" / "ende.pt", tmp_path, tmp_path / "lauf" / "m3.json", mc)
    assert np.isnan(erg["G"]) and np.isnan(erg["G_QK"]) and np.isnan(erg["G_top"])
    assert erg["info"]["G_verworfen_quote"] is True and erg["info"]["G_top_verworfen_quote"] is True
    assert erg["info"]["messconfig"]["min_quote_paare"] == 1.01
