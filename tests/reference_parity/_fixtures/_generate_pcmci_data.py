"""Data for ``test_pcmci_tigramite_parity.py``: 24 stable linear VARs.

Three to five variables, 150 to 600 time points, lags up to three, with
autocorrelation in every series and a random sparse set of cross links;
Gaussian or Laplace innovations in turn. Values are rounded to eight
decimals. The test and the reference script both import :func:`make`,
and the fixture stores a checksum of every data set.
"""

from __future__ import annotations

import numpy as np

N_CASES = 24


def make(seed: int):
    r = np.random.default_rng(770000 + seed)
    d = int(r.integers(3, 6))
    T = int(r.choice([150, 300, 600]))
    tau_max = int(r.choice([2, 3]))
    pc_alpha = float(r.choice([0.05, 0.2]))
    A = np.zeros((4, d, d))  # A[lag, target, source]
    for j in range(d):
        A[1, j, j] = r.uniform(0.3, 0.7)
    for _ in range(d + 1):
        i, j = r.choice(d, 2, replace=False)
        lag = int(r.integers(1, 4))
        A[lag, j, i] = r.choice([-1, 1]) * r.uniform(0.2, 0.5)
    # shrink until the companion matrix is comfortably stable
    comp = np.zeros((3 * d, 3 * d))
    comp[d:, : 2 * d] = np.eye(2 * d)
    while True:
        comp[:d] = np.hstack([A[1], A[2], A[3]])
        if np.abs(np.linalg.eigvals(comp)).max() < 0.85:
            break
        A *= 0.9
    burn = 200
    E = r.normal(size=(T + burn, d)) if seed % 2 == 0 else r.laplace(size=(T + burn, d))
    X = np.zeros((T + burn, d))
    for t in range(3, T + burn):
        X[t] = E[t] + sum(A[lag] @ X[t - lag] for lag in range(1, 4))
    X = np.round(X[burn:], 8)
    if not np.all(np.isfinite(X)) or np.abs(X).max() > 1e3:
        raise RuntimeError(f"unstable VAR for seed {seed}")
    return X, tau_max, pc_alpha
