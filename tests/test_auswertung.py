import json

import numpy as np

from chroma import auswertung as aw


def schreibe(haupt, arm, seed, stufe, **werte):
    d = haupt / arm / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    basis = {"G": 0.5, "kurtosis": [0.0], "kurtosis_ohne_pos0": [0.0], "pr_archetyp_eigen": 50.0, "pr_archetyp_ica": 50.0,
             "pr_strom_eigen": 50.0, "pr_strom_ica": 50.0, "kophenetisch": 0.5, "G_QK": 0.1, "G_top": 0.1}
    basis.update(werte)
    (d / f"messung_stufe_{stufe:.2f}.json").write_text(json.dumps(basis))


def test_p1_bestaetigt_widerlegt_unentschieden():
    assert aw.entscheide_p1(np.array([.10, .12, .11, .09, .13]), np.array([.50, .52, .51, .49, .53]))["urteil"] == "bestätigt"
    assert aw.entscheide_p1(np.array([.400, .402, .398, .401, .399]), np.array([.401, .399, .400, .402, .398]))["urteil"] == "widerlegt"
    assert aw.entscheide_p1(np.array([.3, .5, .1, .6, .2]), np.array([.35, .55, .15, .62, .3]))["urteil"] == "unentschieden"
    assert aw.entscheide_p1(np.array([.3]), np.array([.4, .5]))["urteil"] == "nicht auswertbar"


def test_p1_signifikant_aber_innerhalb_delta():
    """Wenn beide p_diff und p_tost signifikant sind, ist das Urteil 'signifikant, aber innerhalb δ'."""
    result = aw.entscheide_p1(np.array([.480, .481, .479, .480, .480]), np.array([.500, .501, .499, .500, .500]))
    assert result["urteil"] == "signifikant, aber innerhalb δ"


def test_p1_konstante_gruppen_nan():
    """Zwei Gruppen mit einer konstant und einer null-varianz führen zu NaN und Urteil 'nicht auswertbar'."""
    import warnings
    # Die scipy Warning über Präzisionsverlust ist erwartbar bei konstanten Gruppen und wird in
    # entscheide_p1 durch NaN-Checks korrekt behandelt; supprimieren sie hier für diesen Test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = aw.entscheide_p1(np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    assert result["urteil"] == "nicht auswertbar"


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


def test_manipulation_fehlgeschlagen_macht_urteile_nichtig(tmp_path):
    """Wenn Manipulationsprüfung fehlschlägt, werden P1 und P2 Urteile nichtig."""
    rng = np.random.default_rng(0)
    for seed in range(5):
        # A₀ mit hoher kurtosis_ohne_pos0 (> 2), sollte Manipulation scheitern lassen
        schreibe(tmp_path, "A0", seed, 5.0, G=0.1 + 0.01 * rng.random(),
                 kurtosis=[0.5 + 0.1 * rng.random()], kurtosis_ohne_pos0=[5.0 + 0.1 * rng.random()])
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(),
                 kurtosis=[40.0 + rng.random()], kurtosis_ohne_pos0=[40.0 + rng.random()], pr_archetyp_ica=80.0 + rng.random(), pr_strom_ica=80.0 + rng.random())
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    assert erg["manipulation"]["bestanden"] is False
    assert erg["p1"]["urteil"] == "nichtig"
    assert erg["p2"]["Ba"]["lesbar"] == "nichtig"
    assert erg["p2"]["Bb"]["preis_ok"] == "nichtig"


def test_bericht_vollstaendig(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(5):
        # Gebe kurtosis und kurtosis_ohne_pos0 Spread zum Vermeiden von RuntimeWarnings
        schreibe(tmp_path, "A0", seed, 5.0, G=0.1 + 0.01 * rng.random(),
                 kurtosis=[0.5 + 0.1 * rng.random()], kurtosis_ohne_pos0=[0.5 + 0.1 * rng.random()])
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(),
                 kurtosis=[40.0 + rng.random()], kurtosis_ohne_pos0=[40.0 + rng.random()],
                 pr_archetyp_ica=80.0 + rng.random(), pr_strom_ica=80.0 + rng.random())
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    assert erg["stufe"] == 5.0
    assert erg["manipulation"]["bestanden"] is True
    assert erg["p1"]["urteil"] == "bestätigt"
    assert erg["p2"]["Ba"]["lesbar"] and erg["p2"]["Bb"]["lesbar"]
    assert "n" in erg
    assert "p3_ordnung" in erg
    text = (tmp_path / "bericht.md").read_text()
    assert "P1" in text and "explorativ" in text
