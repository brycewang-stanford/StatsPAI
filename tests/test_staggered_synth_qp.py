"""KKT certificates for the block-structured multisynth QP solver.

``sp.staggered_synth`` minimises ``0.5 z'Hz + c'z`` subject to ``z >= 0``
and one adding-up constraint per treated unit / cohort. The solver never
forms ``H``; these tests form it densely and check the optimality
conditions the returned weights must satisfy (primal feasibility,
stationarity on the support, non-negative reduced costs off it). The
end-to-end numbers against R ``augsynth::multisynth`` are pinned in
``tests/reference_parity/test_did_synth_synthvar_parity.py``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth._multisynth_qp import solve_multisynth_qp


def _problem(seed, J=6, n0=(9, 12, 7, 15, 10, 8), m=5):
    rng = np.random.default_rng(seed)
    ndims = [m - (j % 3) for j in range(J)]
    xs, Ms = [], []
    for j in range(J):
        M = rng.normal(size=(m, n0[j])).cumsum(axis=0)
        M[: m - ndims[j]] = 0.0  # zero padding, as multisynth does
        x = M @ rng.dirichlet(np.ones(n0[j])) + rng.normal(0, 0.3, m)
        x[: m - ndims[j]] = 0.0
        xs.append(x)
        Ms.append(M)
    n1 = [1 + (j % 2) for j in range(J)]
    return xs, Ms, n1, ndims


def _dense(xs, Ms, n1, ndims, nu, lam, n_lags, norm_pool, norm_sep):
    J = len(xs)
    cuts = np.cumsum([0] + [M.shape[1] for M in Ms])
    N = cuts[-1]
    c_sep = (1 - nu) / (norm_sep * J)
    c_pool = nu / (norm_pool * J**2 * n_lags)
    Mh = np.hstack(Ms)
    H = c_pool * Mh.T @ Mh + lam * np.eye(N)
    c = -c_pool * Mh.T @ np.sum(xs, axis=0)
    for j in range(J):
        sl = slice(cuts[j], cuts[j + 1])
        H[sl, sl] += c_sep * Ms[j].T @ Ms[j] / ndims[j]
        c[sl] -= c_sep * Ms[j].T @ xs[j] / ndims[j]
    return H, c, cuts


@pytest.mark.parametrize("nu", [0.0, 0.35, 0.9])
@pytest.mark.parametrize("lam", [0.0, 0.2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solution_satisfies_kkt(nu, lam, seed):
    xs, Ms, n1, ndims = _problem(seed)
    args = (xs, Ms, n1, ndims, nu, lam, 5, 0.7, 1.3)
    gam = solve_multisynth_qp(*args)
    H, c, cuts = _dense(*args)
    z = np.concatenate(gam)
    scale = max(1.0, np.abs(H).max(), np.abs(c).max())
    # primal feasibility
    assert z.min() >= 0.0
    for j, g in enumerate(gam):
        assert g.sum() == pytest.approx(n1[j], rel=1e-12)
    grad = H @ z + c
    for j in range(len(gam)):
        g_j = grad[cuts[j] : cuts[j + 1]]
        on = gam[j] > 1e-12 * n1[j]
        mu = -g_j[on].mean()
        # stationarity on the support: every positive weight has the same
        # gradient (the block's multiplier) ...
        np.testing.assert_allclose(g_j[on] + mu, 0.0, atol=1e-10 * scale)
        # ... and no zero weight could lower the objective by entering
        assert np.all(g_j[~on] + mu >= -1e-10 * scale)


def test_pooled_only_objective_is_attained():
    """``nu = 1`` without a ridge penalty has a continuum of optima (only the
    summed imbalance enters); the solver must reach the optimal value, here
    zero because each target lies in its donors' convex hull."""
    rng = np.random.default_rng(3)
    xs, Ms, n1, ndims = [], [], [], []
    for j in range(4):
        M = rng.normal(size=(4, 6))
        xs.append(M @ rng.dirichlet(np.ones(6)))
        Ms.append(M)
        n1.append(1)
        ndims.append(4)
    gam = solve_multisynth_qp(xs, Ms, n1, ndims, 1.0, 0.0, 4, 1.0, 1.0)
    pooled = sum(x - M @ g for x, M, g in zip(xs, Ms, gam))
    assert np.abs(pooled).max() < 1e-10


def test_large_panel_solves():
    """A 300-unit panel (about 17,000 stacked weights) solves exactly; the
    former dense active-set solver formed the 17,000 x 17,000 Hessian."""
    rng = np.random.default_rng(1)
    n, T = 300, 10
    g = rng.choice([0, 4, 6, 8], size=n)
    a = rng.normal(size=n)
    tt = np.tile(np.arange(1, T + 1), n)
    gg = np.repeat(g, T)
    d = ((gg > 0) & (tt >= gg)).astype(int)
    y = np.repeat(a, T) + 0.3 * tt + 2.0 * d + rng.normal(size=n * T)
    df = pd.DataFrame({"id": np.repeat(np.arange(n), T), "t": tt, "d": d, "y": y})
    r = sp.staggered_synth(df, "y", "id", "t", "d", placebo=False)
    W = r.model_info["weights"].to_numpy()
    # every treated unit's weights lie on the simplex over never-treated units
    never = np.flatnonzero(g == 0)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, rtol=1e-12)
    assert W.min() >= 0.0
    assert np.abs(np.delete(W, never, axis=0)).max() == 0.0
    assert r.model_info["n_treated_units"] == W.shape[1]
