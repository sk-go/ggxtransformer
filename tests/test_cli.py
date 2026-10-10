from chroma.__main__ import selbsttest


def test_selbsttest(tmp_path):
    selbsttest(tmp_path)
    assert (tmp_path / "haupt" / "bericht.md").exists()
    assert (tmp_path / "haupt" / "register.md").exists()
    assert (tmp_path / "vorstudie" / "bericht.md").exists()


def _register_args(haupt):
    import argparse

    return argparse.Namespace(haupt=str(haupt), korpus=str(haupt), stufe="100")


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
