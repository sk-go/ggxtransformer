import json
import math

import numpy as np
import pytest

from chroma.arme import ARME
from chroma.training import TrainConfig, lade_modell, trainiere

KLEIN = dict(batch_size=8, block_size=16, n_layer=1, n_head=2, n_embd=16, eval_alle=10, eval_batches=2)


def mini_korpus(tmp_path, vocab=20, n=20000):
    k = tmp_path / "korpus"
    k.mkdir()
    muster = np.tile(np.arange(vocab, dtype=np.uint16), n // vocab)  # lernbare Zählfolge
    for name in ("train", "val", "test"):
        muster.tofile(k / f"{name}.bin")
    meta = {"vocab_size": vocab, "tokens": {"train": len(muster)}, "quelle": "test"}
    (k / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_lernt(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 80, **KLEIN)
    erg = trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert erg["kurve"][-1]["val_loss"] < math.log(20) - 1.0
    assert (tmp_path / "lauf" / "fertig.json").exists()
    assert lade_modell(tmp_path / "lauf" / "ende.pt").cfg.farbe == "keine"


def test_stufe_wird_gesichert_und_beendet(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 80, loss_stufen=(100.0,), **KLEIN)
    erg = trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert (tmp_path / "lauf" / "stufe_100.00.pt").exists()
    assert erg["erreicht"] == {"100.00": 10}
    assert erg["kurve"][-1]["schritt"] == 10


def test_deterministisch(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="Bb", seed=3, lr=3e-3, max_tokens=8 * 16 * 20, **KLEIN)
    a = trainiere(k, tmp_path / "l1", tc, log=lambda *a: None)
    b = trainiere(k, tmp_path / "l2", tc, log=lambda *a: None)
    assert a["kurve"] == b["kurve"]


def test_fertiger_lauf_wird_nicht_ueberschrieben(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 10, **KLEIN)
    trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    with pytest.raises(FileExistsError):
        trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)


def test_nicht_endlicher_loss_bricht_ab(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A0", seed=0, lr=1e30, max_tokens=8 * 16 * 40, **KLEIN)
    with pytest.raises(FloatingPointError):
        trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert (tmp_path / "lauf" / "abbruch.json").exists()


@pytest.mark.parametrize("arm", list(ARME))
def test_alle_arme_laufen(tmp_path, arm):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm=arm, seed=0, lr=1e-3, max_tokens=8 * 16 * 2, **KLEIN)
    erg = trainiere(k, tmp_path / arm, tc, log=lambda *a: None)
    assert np.isfinite(erg["kurve"][-1]["val_loss"])
