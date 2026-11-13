"""Data for ``test_lingam_package_parity.py``: 40 linear non-Gaussian SEMs.

Three to seven variables, a random causal order, each possible edge
present with probability one half and a coefficient of absolute value 0.5
to 1.5, n in {300, 1000, 3000}; the disturbances are uniform, Laplace or
centred exponential in turn. Values are rounded to eight decimals. The
test and the reference script both import :func:`make`, and the fixture
stores a checksum of every data set so a change in the random stream
cannot go unnoticed.
"""

from __future__ import annotations

import numpy as np

N_CASES = 40


def make(seed: int):
    r = np.random.default_rng(seed)
    p = int(r.integers(3, 8))
    n = int(r.choice([300, 1000, 3000]))
    perm = r.permutation(p)
    B = np.zeros((p, p))
    for a in range(p):
        for b in range(a + 1, p):
            if r.random() < 0.5:
                B[perm[b], perm[a]] = r.choice([-1, 1]) * r.uniform(0.5, 1.5)
    kind = seed % 3
    if kind == 0:
        E = r.uniform(-1, 1, (n, p))
    elif kind == 1:
        E = r.laplace(size=(n, p))
    else:
        E = r.exponential(size=(n, p)) - 1
    E = E * r.uniform(0.5, 1.5, p)
    X = np.zeros((n, p))
    for j in perm:
        X[:, j] = X @ B[j] + E[:, j]
    return np.round(X, 8), B
