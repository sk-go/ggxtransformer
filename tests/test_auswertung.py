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


def test_fehlende_stufen_auf_allen_laeufen(tmp_path):
    """Fehlende Stufen werden auf ALLEN Läufen berechnet und in erg['fehlende_stufen'] und bericht.md geschrieben."""
    rng = np.random.default_rng(0)
    # A0 hat nur Stufe 5.0; andere Arms haben 4.0 und 5.0
    schreibe(tmp_path, "A0", 0, 5.0, G=0.1 + 0.01 * rng.random(), kurtosis=[0.5 + 0.1 * rng.random()], kurtosis_ohne_pos0=[0.5 + 0.1 * rng.random()])
    schreibe(tmp_path, "A0", 1, 5.0, G=0.1 + 0.01 * rng.random(), kurtosis=[0.5 + 0.1 * rng.random()], kurtosis_ohne_pos0=[0.5 + 0.1 * rng.random()])
    for seed in range(2):
        schreibe(tmp_path, "A", seed, 4.0, G=0.5 + 0.01 * rng.random(), kurtosis=[40.0 + rng.random()], kurtosis_ohne_pos0=[40.0 + rng.random()], pr_archetyp_ica=80.0 + rng.random(), pr_strom_ica=80.0 + rng.random())
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(), kurtosis=[40.0 + rng.random()], kurtosis_ohne_pos0=[40.0 + rng.random()], pr_archetyp_ica=80.0 + rng.random(), pr_strom_ica=80.0 + rng.random())
        schreibe(tmp_path, "Ba", seed, 4.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 4.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    # Stufe 5.0 ist die gemeinsame Stufe
    assert erg["stufe"] == 5.0
    # erg["fehlende_stufen"] nutzt string keys jetzt
    assert "A0:0" in erg["fehlende_stufen"]
    assert erg["fehlende_stufen"]["A0:0"] == [4.0]  # A0 fehlt 4.0
    assert erg["fehlende_stufen"]["A0:1"] == [4.0]
    assert erg["fehlende_stufen"]["A:0"] == []  # A hat beide
    # Bericht sollte "Fehlende Stufen" Sektion enthalten
    text = (tmp_path / "bericht.md").read_text()
    assert "Fehlende Stufen" in text
    assert "(A0, seed0): fehlen [4.0]" in text


def test_stichprobengröße_in_bericht(tmp_path):
    """Stichprobengröße-Tabelle wird in bericht.md geschrieben und zeigt n und verworfen."""
    rng = np.random.default_rng(0)
    for seed in range(5):
        # Seed 0 von A0 hat NaN für G, sollte verworfen=1 zeigen
        g_val = float("nan") if seed == 0 else (0.1 + 0.01 * rng.random())
        schreibe(tmp_path, "A0", seed, 5.0, G=g_val,
                 kurtosis=[0.5 + 0.1 * rng.random()], kurtosis_ohne_pos0=[0.5 + 0.1 * rng.random()])
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(),
                 kurtosis=[40.0 + rng.random()], kurtosis_ohne_pos0=[40.0 + rng.random()],
                 pr_archetyp_ica=80.0 + rng.random(), pr_strom_ica=80.0 + rng.random())
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    # A0 sollte verworfen=1 für G zeigen
    assert erg["n"]["A0"]["G"]["verworfen"] == 1
    assert erg["n"]["A0"]["G"]["n"] == 4
    # Bericht sollte Stichprobengröße-Tabelle enthalten
    text = (tmp_path / "bericht.md").read_text()
    assert "Stichprobengröße" in text
    assert "v=1" in text  # mindestens ein Eintrag mit verworfen=1


def test_p2_konstante_identische_gruppen():
    """entscheide_p2 mit identischen konstanten Gruppen gibt lesbar=None (wegen NaN p-Wert)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # pr_b und pr_a_ica identisch konstant: p_lesbar wird NaN → lesbar=None
        result = aw.entscheide_p2(
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # pr_b konstant
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # pr_a_ica konstant (gleich pr_b)
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),  # g_b konstant (unterschiedlich)
            np.array([0.9, 0.9, 0.9, 0.9, 0.9])   # g_a konstant (gleich g_b)
        )
    # Identische konstante pr_b und pr_a_ica führen zu NaN → lesbar=None
    assert result["lesbar"] is None
    # g_b + delta [0.95...] vs g_a [0.9...] sind unterschiedliche Konstanten → p_preis wird 0.0 (finite)
    assert result["preis_ok"] is True  # 0.0 < 0.05 ist True


def test_gemeinsame_stufe_ignoriert_unbekannte_arme(tmp_path):
    """gemeinsame_stufe ignoriert unbekannte Arme (z.B. 'X') die nicht in arme Parameter sind."""
    # Schreibe Daten für A0, A, Ba, Bb mit Stufe 5.0, und X mit Stufe 4.0
    schreibe(tmp_path, "A0", 0, 5.0)
    schreibe(tmp_path, "A", 0, 5.0)
    schreibe(tmp_path, "Ba", 0, 5.0)
    schreibe(tmp_path, "Bb", 0, 5.0)
    schreibe(tmp_path, "X", 0, 4.0)  # Unbekannter Arm

    m = aw.lade_messungen(tmp_path)
    # Mit default ARME Parameter sollte X ignoriert werden
    stufe = aw.gemeinsame_stufe(m)
    # Ohne X wäre die gemeinsame Stufe 5.0, X stört nicht
    assert stufe == 5.0
    # Wenn wir X explizit fordern, sollte stufe None sein (5.0 und 4.0 passen nicht)
    stufe_mit_x = aw.gemeinsame_stufe(m, arme=("A0", "A", "Ba", "Bb", "X"))
    assert stufe_mit_x is None
