"""Data for ``test_notears_reference_parity.py``: linear-Gaussian SEMs.

Four to seven variables, 300 or 1,000 rows, each possible edge present
with probability 0.4 and a coefficient of absolute value 0.6 to 1.6,
Gaussian noise with unequal scales. :func:`make` returns the raw data
(rounded to eight decimals) and the standardised matrix that
``sp.notears`` builds from it, which is what the reference is given.
``SEEDS`` are the designs kept in the fixture: those where every weight
of the reference solution is at least 0.1 away from the 0.3 threshold, so
that the thresholded graph does not hinge on the last digits of an
iterative solver.
"""

from __future__ import annotations

import numpy as np

CANDIDATES = range(30)


def make(seed: int):
    r = np.random.default_rng(990000 + seed)
    d = int(r.integers(4, 8))
    n = int(r.choice([300, 1000]))
    perm = r.permutation(d)
    B = np.zeros((d, d))  # B[i, j]: i -> j
    for a in range(d):
        for b in range(a + 1, d):
            if r.random() < 0.4:
                B[perm[a], perm[b]] = r.choice([-1, 1]) * r.uniform(0.6, 1.6)
    E = r.normal(size=(n, d)) * r.uniform(0.6, 1.4, d)
    X = np.zeros((n, d))
    for j in perm:
        X[:, j] = X @ B[:, j] + E[:, j]
    X = np.round(X, 8)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, Xs
