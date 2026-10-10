import numpy as np

from chroma.messung import basis


def test_kurtosis_gauss_nahe_null():
    X = np.random.default_rng(0).standard_normal((2000, 256))
    assert abs(basis.exzess_kurtosis(X)) < 0.1


def test_ausreisserdimension_und_drehungskontrolle():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 64))
    X[:, 0] += 30.0  # eine bevorzugte Achse
    assert basis.exzess_kurtosis(X) > 10
    assert max(basis.kurtosis_kontrolle(X, 5, rng)) < 2


def test_participation_ratio():
    assert np.allclose(basis.participation_ratio(np.eye(5)), 1.0)
    assert np.allclose(basis.participation_ratio(np.ones((2, 7))), 7.0)


def test_ica_findet_versteckte_spaerliche_basis():
    rng = np.random.default_rng(2)
    S = rng.standard_normal((4000, 6)) * (rng.random((4000, 6)) < 0.1)
    Q = basis.zufallsdrehung(6, rng)
    X = S @ Q
    M = basis.ica_drehung(X)
    assert np.allclose(M @ M.T, np.eye(6), atol=1e-8)
    pr_vorher = basis.participation_ratio(X).mean()
    pr_nachher = basis.participation_ratio(X @ M).mean()
    assert pr_nachher < pr_vorher - 0.3
