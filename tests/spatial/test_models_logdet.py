import numpy as np
from scipy import sparse
from statspai.spatial.models._logdet import log_det_exact, log_det_approx


def _random_row_stochastic(n, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < 0.1).astype(float)
    np.fill_diagonal(A, 0.0)
    # Ensure at least one neighbour per row so row-normalisation works.
    for i in range(n):
        if A[i].sum() == 0:
            j = (i + 1) % n
            A[i, j] = 1.0
    A = A / A.sum(axis=1, keepdims=True)
    return sparse.csr_matrix(A)


def test_exact_log_det_matches_numpy():
    W = _random_row_stochastic(40, seed=1)
    rho = 0.35
    I = np.eye(40)
    ref = np.linalg.slogdet(I - rho * W.toarray())[1]
    got = log_det_exact(W, rho)
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_approx_close_to_exact():
    W = _random_row_stochastic(60, seed=2)
    rho = 0.3
    exact = log_det_exact(W, rho)
    approx = log_det_approx(W, rho, n_draws=400, order=40, seed=2)
    assert abs(exact - approx) / max(abs(exact), 1.0) < 0.10


def test_approx_zero_rho_is_zero():
    W = _random_row_stochastic(30, seed=3)
    assert abs(log_det_approx(W, 0.0, n_draws=50, order=20, seed=0)) < 1e-8
