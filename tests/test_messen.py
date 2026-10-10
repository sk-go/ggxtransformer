import json

import numpy as np

from chroma import korpus
from chroma.messen import MessConfig, messe_checkpoint
from chroma.training import TrainConfig, trainiere
from tests.test_korpus import synthetische_texte

SCHLUESSEL = {"G", "G_QK", "G_top", "kurtosis", "kurtosis_ohne_pos0", "kurtosis_kontrolle", "pr_archetyp_eigen",
              "pr_archetyp_ica", "pr_strom_eigen", "pr_strom_ica", "kophenetisch", "simlex", "info",
              "farbe", "ckpt"}


def test_messe_checkpoint(tmp_path):
    splits = {"train": synthetische_texte(600, 0), "val": synthetische_texte(80, 1), "test": synthetische_texte(300, 2)}
    korpus.baue_korpus(tmp_path, splits=splits, vocab_size=400, quelle="synthetisch")
    tc = TrainConfig(arm="Bb", seed=0, lr=3e-3, batch_size=8, block_size=32, n_layer=2, n_head=2, n_embd=16,
                     max_tokens=8 * 32 * 30, eval_alle=10, eval_batches=2)
    trainiere(tmp_path, tmp_path / "lauf", tc, log=lambda *a: None)
    mc = MessConfig(n_woerter=10, n_paare=30, n_kontexte=4, min_kontexte=2, fenster=16, nach=4, n_anker=6,
                    k_nachbarn=3, n_fenster_strom=16, schicht_strom=1, n_cluster=3, batch=64)
    erg = messe_checkpoint(tmp_path / "lauf" / "ende.pt", tmp_path, tmp_path / "lauf" / "messung_ende.json", mc)
    assert SCHLUESSEL <= set(erg)
    assert erg["farbe"] == "beta" and len(erg["kurtosis"]) == 3
    gespeichert = json.loads((tmp_path / "lauf" / "messung_ende.json").read_text())
    assert gespeichert["G"] == erg["G"] or (np.isnan(gespeichert["G"]) and np.isnan(erg["G"]))
    assert (tmp_path / "lauf" / "messung_ende_cluster.npy").exists()
