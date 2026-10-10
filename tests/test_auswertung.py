import json

import numpy as np

from chroma import auswertung as aw


def schreibe(haupt, arm, seed, stufe, **werte):
    d = haupt / arm / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    basis = {"G": 0.5, "kurtosis": [0.0], "pr_archetyp_eigen": 50.0, "pr_archetyp_ica": 50.0,
             "pr_strom_eigen": 50.0, "pr_strom_ica": 50.0, "kophenetisch": 0.5, "G_QK": 0.1, "G_top": 0.1}
    basis.update(werte)
    (d / f"messung_stufe_{stufe:.2f}.json").write_text(json.dumps(basis))


def test_p1_bestaetigt_widerlegt_unentschieden():
    assert aw.entscheide_p1(np.array([.10, .12, .11, .09, .13]), np.array([.50, .52, .51, .49, .53]))["urteil"] == "bestätigt"
    assert aw.entscheide_p1(np.array([.400, .402, .398, .401, .399]), np.array([.401, .399, .400, .402, .398]))["urteil"] == "widerlegt"
    assert aw.entscheide_p1(np.array([.3, .5, .1, .6, .2]), np.array([.35, .55, .15, .62, .3]))["urteil"] == "unentschieden"
    assert aw.entscheide_p1(np.array([.3]), np.array([.4, .5]))["urteil"] == "nicht auswertbar"


def test_gemeinsame_stufe_weicht_aus(tmp_path):
    for arm in ("A0", "A", "Ba", "Bb"):
        for seed in range(2):
            schreibe(tmp_path, arm, seed, 5.0)
            if arm != "A0":
                schreibe(tmp_path, arm, seed, 4.0)
    assert aw.gemeinsame_stufe(aw.lade_messungen(tmp_path)) == 5.0


def test_keine_gemeinsame_stufe(tmp_path):
    for arm in ("A", "Ba", "Bb"):
        schreibe(tmp_path, arm, 0, 4.0)
    assert aw.gemeinsame_stufe(aw.lade_messungen(tmp_path)) is None
    erg = aw.bericht(tmp_path)
    assert erg["stufe"] is None
    assert "keine gemeinsame Stufe" in (tmp_path / "bericht.md").read_text()


def test_nan_seed_wird_verworfen_und_gezaehlt(tmp_path):
    schreibe(tmp_path, "A0", 0, 5.0, G=float("nan"))
    schreibe(tmp_path, "A0", 1, 5.0, G=0.3)
    w, verworfen = aw.werte(aw.lade_messungen(tmp_path), "A0", 5.0, "G")
    assert list(w) == [0.3] and verworfen == 1


def test_bericht_vollstaendig(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(5):
        schreibe(tmp_path, "A0", seed, 5.0, G=0.1 + 0.01 * rng.random(), kurtosis=[0.5])
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(), kurtosis=[40.0], pr_archetyp_ica=80.0, pr_strom_ica=80.0)
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    assert erg["stufe"] == 5.0
    assert erg["manipulation"]["bestanden"] is True
    assert erg["p1"]["urteil"] == "bestätigt"
    assert erg["p2"]["Ba"]["lesbar"] and erg["p2"]["Bb"]["lesbar"]
    text = (tmp_path / "bericht.md").read_text()
    assert "P1" in text and "explorativ" in text
