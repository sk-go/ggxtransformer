"""
chroma/messung/basis.py
Wie stark eine Warte ihre Achsen unterscheidet: Exzess-Kurtosis über die Dimensionen
(homogene gegen qualitative Mannigfaltigkeit), Participation Ratio (auf wie vielen Achsen
ein Begriff liegt) und die ICA-Drehung als beste Basis, die der Betrachter nachträglich wählt.
"""

import numpy as np


def zeilen_kurtosis(X):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    m2 = (Xc ** 2).mean(axis=1)
    m4 = (Xc ** 4).mean(axis=1)
    return m4 / np.maximum(m2 ** 2, 1e-30) - 3.0


def exzess_kurtosis(X):
    return float(zeilen_kurtosis(X).mean())


def zufallsdrehung(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def kurtosis_kontrolle(X, n_drehungen, rng):
    """Dieselben Aktivierungen nach zufälligen Drehungen: Was hier bleibt, ist nicht der Achse eigen."""
    X = np.asarray(X, dtype=np.float64)
    return [exzess_kurtosis(X @ zufallsdrehung(X.shape[1], rng)) for _ in range(n_drehungen)]


def participation_ratio(V):
    V = np.asarray(V, dtype=np.float64)
    s2 = (V ** 2).sum(axis=1)
    s4 = (V ** 4).sum(axis=1)
    return s2 ** 2 / np.maximum(s4, 1e-30)


def ica_drehung(X, seed=0, rueckgabe_konvergenz=False, max_iter=2000, tol=1e-4):
    """Orthogonale Matrix M, die FastICA wählt: Hauptachsen U, dann ICA-Drehung im weißen Raum,
    auf die nächste orthogonale Matrix projiziert. Gedrehte Daten: X @ M.

    M = U·Wᵀ ist ein orthogonales Surrogat der ICA-Lösung: exakt nur bei orthogonaler Mischung
    (dann ist die ICA-Entmischung im weißen Raum selbst orthogonal). Sonst weicht M von der besten
    ICA-Basis ab; das begünstigt im P2-Vergleich (PR von B gegen PR von A nach Drehung) tendenziell B.

    Mit rueckgabe_konvergenz=True: Rückgabe (M, konvergiert), konvergiert = n_iter_ < max_iter.
    Die ConvergenceWarning von FastICA wird dabei nur lokal unterdrückt; die Meldung steht im Flag."""
    import warnings

    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < d:
        raise ValueError(f"ICA braucht mindestens so viele Zeilen ({n}) wie Dimensionen ({d})")
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt.T
    Z = (Xc @ U) / np.maximum(s / np.sqrt(n), 1e-12)
    ica = FastICA(whiten=False, random_state=seed, max_iter=max_iter, tol=tol)
    if rueckgabe_konvergenz:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ica.fit(Z)
    else:
        ica.fit(Z)
    u, _, vt = np.linalg.svd(ica.components_.T)
    M = U @ (u @ vt)
    if rueckgabe_konvergenz:
        return M, bool(ica.n_iter_ < max_iter)
    return M
