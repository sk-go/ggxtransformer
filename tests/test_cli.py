from chroma.__main__ import selbsttest


def test_selbsttest(tmp_path):
    selbsttest(tmp_path)
    assert (tmp_path / "haupt" / "bericht.md").exists()
    for arm in ("A0", "A", "Ba", "Bb"):  # ein Register je Arm
        assert (tmp_path / "haupt" / f"register_{arm}.md").exists()
    assert not (tmp_path / "haupt" / "register.md").exists()
    assert (tmp_path / "vorstudie" / "bericht.md").exists()


def _register_args(haupt, arm="A"):
    import argparse

    return argparse.Namespace(haupt=str(haupt), korpus=str(haupt), stufe="100", arm=arm)


def test_register_ohne_dateien(tmp_path):
    import pytest
    from chroma.__main__ import cmd_register

    with pytest.raises(SystemExit, match="--haupt"):
        cmd_register(_register_args(tmp_path))


def test_register_zu_wenige_seeds(tmp_path):
    import numpy as np
    import pytest
    from chroma.__main__ import cmd_register

    (tmp_path / "A" / "seed0").mkdir(parents=True)
    np.save(tmp_path / "A" / "seed0" / "messung_stufe_100.00_cluster.npy", np.zeros(5, dtype=int))
    with pytest.raises(SystemExit, match="mindestens 2 Seeds"):
        cmd_register(_register_args(tmp_path))


def test_register_ungleiche_laengen(tmp_path):
    import numpy as np
    import pytest
    from chroma.__main__ import cmd_register

    for s, n in ((0, 5), (1, 6)):
        d = tmp_path / "A" / f"seed{s}"
        d.mkdir(parents=True)
        np.save(d / "messung_stufe_100.00_cluster.npy", np.zeros(n, dtype=int))
    with pytest.raises(SystemExit, match="unterschiedliche Längen"):
        cmd_register(_register_args(tmp_path))


def test_register_nur_eigener_arm(tmp_path):
    """Cluster anderer Arme zählen nicht: zwei Seeds in Ba reichen nicht für A."""
    import numpy as np
    import pytest
    from chroma.__main__ import cmd_register

    for arm, seed in (("A", 0), ("Ba", 0), ("Ba", 1)):
        d = tmp_path / arm / f"seed{seed}"
        d.mkdir(parents=True)
        np.save(d / "messung_stufe_100.00_cluster.npy", np.zeros(5, dtype=int))
    with pytest.raises(SystemExit, match="mindestens 2 Seeds"):
        cmd_register(_register_args(tmp_path, "A"))


def test_register_arm_pflicht(capsys):
    import pytest
    import sys
    from chroma.__main__ import main

    alt = sys.argv
    sys.argv = ["chroma", "register", "--haupt", "x", "--korpus", "x", "--stufe", "100"]
    try:
        with pytest.raises(SystemExit) as e:
            main()
    finally:
        sys.argv = alt
    assert e.value.code == 2 and "--arm" in capsys.readouterr().err  # argparse: Pflichtargument fehlt
