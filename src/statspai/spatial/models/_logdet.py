"""Log-determinant of (I - rho W) for spatial ML estimation.

Two paths:

- `log_det_exact`: dense eigendecomposition. Correct for any |rho| < 1 given
  eigenvalues of W. Use when n < 5000.
- `log_det_approx`: Barry-Pace / Chebyshev-style stochastic approximation via
  Hutchinson trace estimation of tr(W^k). Use when n is large and the exact
  path is too expensive.

  log det(I - rho W)  =  -sum_{k>=1} (rho^k / k) * tr(W^k)
  tr(W^k) approximated by (1/M) sum_m u_m^T W^k u_m with Rademacher u_m.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse


def _to_csr(W):
    if sparse.issparse(W):
        return W.tocsr()
    return sparse.csr_matrix(np.asarray(W, dtype=float))


def log_det_exact(W, rho: float) -> float:
    W = _to_csr(W)
    eigvals = np.linalg.eigvals(W.toarray())
    return float(np.sum(np.log(np.abs(1.0 - rho * eigvals))))


def log_det_approx(W, rho: float, n_draws: int = 50, order: int = 50,
                   seed: Optional[int] = None) -> float:
    W = _to_csr(W)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    # The k=1 term, tr(W), is known exactly from the diagonal — no need to
    # estimate it stochastically. For row-standardised spatial weights this
    # is identically zero (no self-loops), which removes the dominant source
    # of Hutchinson variance.
    diag_W = np.asarray(W.diagonal())
    tr_W = float(diag_W.sum())
    # Precompute probes as a single matrix for vectorised multiplies.
    U = rng.choice([-1.0, 1.0], size=(n_draws, n))
    V = U.copy()
    total = 0.0
    rho_pow = 1.0
    for k in range(1, order + 1):
        # V starts as U; each iter sets V <- V @ W^T so that row i becomes W @ u_i
        V = V @ W.T
        rho_pow *= rho
        if k == 1:
            tr_k = tr_W
        else:
            tr_k = float(np.mean(np.sum(U * V, axis=1)))
        total -= (rho_pow / k) * tr_k
    return total
