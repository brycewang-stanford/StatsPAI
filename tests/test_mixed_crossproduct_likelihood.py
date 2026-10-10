"""
``sp.mixed`` evaluates the profiled (RE)ML criterion from per-group
cross-products instead of factorising every ``V_j``.

The reference in each test is the definition written out directly: build
``V_j = Z_j G Z_j' + sigma2 I`` densely, profile the fixed effects by GLS,
and add the Gaussian log-densities with ``scipy.stats``.  Nothing in the
reference shares code with the package.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.multilevel._core import (
    _group_blocks,
    _lmm_reduce_blocks,
    _n_cov_params,
    _prepare_frame,
    _unpack_G,
)
from statspai.multilevel.lmm import _profiled_nll, _three_level_nll, _three_level_reduce

# The two evaluations differ by rounding only; on these well-scaled designs
# that is ~1e-13 of a criterion of order 10^2-10^3.
RTOL = 1e-10


# ---------------------------------------------------------------------------
# Designs and the direct definition
# ---------------------------------------------------------------------------


def _unbalanced_design(seed: int, q: int) -> pd.DataFrame:
    """Singletons, pairs, and one large group; rows in shuffled order."""
    rng = np.random.default_rng(seed)
    sizes = np.array([1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 12, 40])
    g = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    n = len(g)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    u = rng.normal(size=(len(sizes), 3))
    y = 3.0 + 0.5 * x1 - 0.25 * x2 + u[g, 0] + rng.normal(size=n)
    if q >= 2:
        y = y + 0.7 * u[g, 1] * x1
    if q >= 3:
        y = y + 0.4 * u[g, 2] * x2
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def _direct_nll(blocks, G, sigma2, reml):
    """-(RE)ML log-likelihood from dense V_j, profiled over beta by GLS."""
    p = blocks[0].X.shape[1]
    V = [b.Z @ G @ b.Z.T + sigma2 * np.eye(b.n) for b in blocks]
    XtVinvX = sum(b.X.T @ np.linalg.solve(v, b.X) for b, v in zip(blocks, V))
    XtVinvy = sum(b.X.T @ np.linalg.solve(v, b.y) for b, v in zip(blocks, V))
    beta = np.linalg.solve(XtVinvX, XtVinvy)
    loglik = sum(
        stats.multivariate_normal.logpdf(b.y, mean=b.X @ beta, cov=v)
        for b, v in zip(blocks, V)
    )
    if reml:
        # Restricted likelihood: integrate beta out under a flat prior.
        loglik += 0.5 * p * np.log(2 * np.pi)
        loglik -= 0.5 * np.linalg.slogdet(XtVinvX)[1]
    return -float(loglik)


def _blocks_for(df, x_random):
    frame = _prepare_frame(df, "y", ["x1", "x2"], ["g"], x_random)
    blocks, _, _ = _group_blocks(frame, "y", ["x1", "x2"], x_random, "g")
    return blocks


# ---------------------------------------------------------------------------
# The likelihood function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
@pytest.mark.parametrize(
    "q, cov_type",
    [
        (1, "unstructured"),
        (1, "identity"),
        (2, "unstructured"),
        (2, "diagonal"),
        (2, "identity"),
        (3, "unstructured"),
        (3, "diagonal"),
    ],
)
def test_crossproduct_criterion_equals_dense_definition(q, cov_type, reml):
    df = _unbalanced_design(seed=10 * q + len(cov_type), q=q)
    x_random = ["x1", "x2"][: q - 1]
    blocks = _blocks_for(df, x_random)
    reduced = _lmm_reduce_blocks(blocks)
    k = _n_cov_params(q, cov_type)
    n = len(df)

    rng = np.random.default_rng(1)
    for _ in range(25):
        theta = rng.normal(scale=0.8, size=k + 1)
        G = _unpack_G(theta[:k], q, cov_type)
        expected = _direct_nll(blocks, G, float(np.exp(theta[k])), reml)
        got = _profiled_nll(theta, reduced, 3, q, n, reml, cov_type)
        assert got == pytest.approx(expected, rel=RTOL)
        # Passing the blocks themselves reduces them on the fly.
        assert _profiled_nll(theta, blocks, 3, q, n, reml, cov_type) == got


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_criterion_with_a_zero_variance_in_G(reml):
    """A singular G (zero slope variance) needs no inverse of G."""
    df = _unbalanced_design(seed=3, q=2)
    blocks = _blocks_for(df, ["x1"])
    reduced = _lmm_reduce_blocks(blocks)
    n = len(df)

    # exp(-800) underflows to exactly 0: G = diag(0.6, 0).
    theta = np.array([np.log(0.6), -800.0, np.log(1.3)])
    G = _unpack_G(theta[:2], 2, "diagonal")
    assert G[1, 1] == 0.0
    got = _profiled_nll(theta, reduced, 3, 2, n, reml, "diagonal")
    assert got == pytest.approx(_direct_nll(blocks, G, 1.3, reml), rel=RTOL)

    # With the slope variance at zero the model is the random-intercept one.
    ri_blocks = _blocks_for(df, [])
    theta_ri = np.array([np.log(0.6), np.log(1.3)])
    ri = _profiled_nll(theta_ri, ri_blocks, 3, 1, n, reml, "identity")
    assert got == pytest.approx(ri, rel=RTOL)

    # Both variances zero: the criterion is that of pooled least squares.
    theta0 = np.array([-800.0, -800.0, np.log(1.3)])
    pooled = _direct_nll(blocks, np.zeros((2, 2)), 1.3, reml)
    got0 = _profiled_nll(theta0, reduced, 3, 2, n, reml, "diagonal")
    assert got0 == pytest.approx(pooled, rel=RTOL)


def test_criterion_does_not_depend_on_the_level_of_the_outcome():
    """Shifting y by X b leaves the profiled criterion unchanged.

    The reduced form works with pooled-OLS residuals so that a large mean
    costs no digits in the residual quadratic form; a shift of 1e7 moves the
    criterion by less than 1e-9 relative.
    """
    df = _unbalanced_design(seed=5, q=2)
    shifted = df.assign(y=df["y"] + 1e7 + 1e5 * df["x1"])
    theta = np.array([-0.3, 0.2, -0.5, 0.1])
    values = []
    for frame in (df, shifted):
        reduced = _lmm_reduce_blocks(_blocks_for(frame, ["x1"]))
        values.append(
            _profiled_nll(theta, reduced, 3, 2, len(df), True, "unstructured")
        )
    assert values[1] == pytest.approx(values[0], rel=1e-9)


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_three_level_criterion_equals_dense_definition(reml):
    rng = np.random.default_rng(11)
    rows = []
    for school in range(7):
        for klass in range(int(rng.integers(1, 5))):
            for _ in range(int(rng.integers(1, 9))):
                rows.append((school, klass, rng.normal(), rng.normal()))
    df = pd.DataFrame(rows, columns=["school", "klass", "x", "y"])
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    df["y"] += 2.0 + 0.5 * df["x"] + rng.normal(size=7)[df["school"]]

    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    school = df["school"].to_numpy()
    # klass labels repeat across schools: a class is a (school, klass) pair.
    _, inner = np.unique(school * 10 + df["klass"].to_numpy(), return_inverse=True)
    outer_levels, outer = np.unique(school, return_inverse=True)
    outer_of = np.array([outer[inner == c][0] for c in range(inner.max() + 1)])
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    reduced = _three_level_reduce(y, X, inner, outer_of, beta0)

    for _ in range(25):
        theta = rng.normal(scale=0.8, size=3)
        s2_s, s2_c, s2_e = np.exp(theta)
        XtVinvX = np.zeros((2, 2))
        XtVinvy = np.zeros(2)
        V_list = []
        for s in range(len(outer_levels)):
            rows_s = np.flatnonzero(outer == s)
            same_class = inner[rows_s][:, None] == inner[rows_s][None, :]
            V = s2_e * np.eye(len(rows_s)) + s2_s + s2_c * same_class
            V_list.append((rows_s, V))
            XtVinvX += X[rows_s].T @ np.linalg.solve(V, X[rows_s])
            XtVinvy += X[rows_s].T @ np.linalg.solve(V, y[rows_s])
        beta = np.linalg.solve(XtVinvX, XtVinvy)
        loglik = sum(
            stats.multivariate_normal.logpdf(y[r], mean=X[r] @ beta, cov=V)
            for r, V in V_list
        )
        if reml:
            loglik += 0.5 * 2 * np.log(2 * np.pi)
            loglik -= 0.5 * np.linalg.slogdet(XtVinvX)[1]
        got = _three_level_nll(theta, reduced, 2, len(df), reml)
        assert got == pytest.approx(-float(loglik), rel=RTOL)


# ---------------------------------------------------------------------------
# Fitted quantities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_random", [None, ["x1"]], ids=["q1", "q2"])
def test_fit_reproduces_dense_gls_and_blup_formulas(x_random):
    """beta, Cov(beta), BLUPs and their SEs from the dense textbook formulas."""
    df = _unbalanced_design(seed=21, q=2 if x_random else 1)
    fit = sp.mixed(df, "y", ["x1", "x2"], "g", x_random=x_random)
    G, sigma2 = fit._G, fit._sigma2
    blocks = _blocks_for(df, x_random or [])

    V = [b.Z @ G @ b.Z.T + sigma2 * np.eye(b.n) for b in blocks]
    XtVinvX = sum(b.X.T @ np.linalg.solve(v, b.X) for b, v in zip(blocks, V))
    XtVinvy = sum(b.X.T @ np.linalg.solve(v, b.y) for b, v in zip(blocks, V))
    cov_beta = np.linalg.inv(XtVinvX)
    beta = cov_beta @ XtVinvy

    np.testing.assert_allclose(fit.fixed_effects.to_numpy(), beta, rtol=1e-10)
    np.testing.assert_allclose(fit._cov_fixed, cov_beta, rtol=1e-9, atol=1e-14)

    blups, blup_se = fit.ranef(conditional_se=True)
    assert list(blups.index) == [b.key for b in blocks]
    for b, v in zip(blocks, V):
        ZtVinv = np.linalg.solve(v, b.Z).T
        u = G @ ZtVinv @ (b.y - b.X @ beta)
        ZtVinvX = ZtVinv @ b.X
        cond = G - G @ ZtVinv @ b.Z @ G + G @ ZtVinvX @ cov_beta @ ZtVinvX.T @ G
        np.testing.assert_allclose(blups.loc[b.key].to_numpy(), u, atol=1e-10)
        np.testing.assert_allclose(fit.blups[b.key], u, atol=1e-10)
        np.testing.assert_allclose(
            blup_se.loc[b.key].to_numpy(), np.sqrt(np.diag(cond)), rtol=1e-8
        )

    # The reported log-likelihood is the criterion at the reported estimates.
    assert -fit.log_likelihood == pytest.approx(
        _direct_nll(blocks, G, sigma2, reml=True), rel=RTOL
    )


def test_balanced_one_way_layout_matches_anova_closed_form():
    """REML = ANOVA estimators when the layout is balanced."""
    rng = np.random.default_rng(8)
    J, m = 25, 6
    g = np.repeat(np.arange(J), m)
    y = 1.0 + rng.normal(scale=0.9, size=J)[g] + rng.normal(scale=1.1, size=J * m)
    # Row order must not matter.
    df = pd.DataFrame({"y": y, "g": g}).sample(frac=1.0, random_state=1)
    fit = sp.mixed(df, "y", [], "g")

    group_means = df.groupby("g")["y"].mean()
    msw = float(((df["y"] - df["g"].map(group_means)) ** 2).sum() / (J * (m - 1)))
    msb = float(m * ((group_means - y.mean()) ** 2).sum() / (J - 1))
    assert msb > msw  # interior solution
    assert fit.variance_components["var(Residual)"] == pytest.approx(msw, rel=1e-6)
    assert fit.variance_components["var(_cons)"] == pytest.approx(
        (msb - msw) / m, rel=1e-6
    )
    assert fit.fixed_effects["_cons"] == pytest.approx(y.mean(), rel=1e-10)
    assert fit.std_errors["_cons"] == pytest.approx(np.sqrt(msb / (J * m)), rel=1e-6)


def test_group_blocks_keep_first_appearance_order_and_row_positions():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "g": ["b", "a", "b", "c", "a", "b"],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    blocks = _blocks_for(df, ["x1"])
    assert [b.key for b in blocks] == ["b", "a", "c"]
    assert [b.row_idx.tolist() for b in blocks] == [[0, 2, 5], [1, 4], [3]]
    np.testing.assert_array_equal(blocks[0].y, [1.0, 3.0, 6.0])
    np.testing.assert_array_equal(blocks[1].X, [[1.0, 0.2, 0.0], [1.0, 0.5, 1.0]])
    np.testing.assert_array_equal(blocks[2].Z, [[1.0, 0.4]])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_many_groups_fit_is_fast():
    """n = 20,000 in 400 groups; the dense-V evaluation needed ~6 s here.

    The bound is generous (the fit takes ~0.05 s) so a loaded CI machine
    does not trip it, while a return to per-group n_j x n_j solves would.
    """
    rng = np.random.default_rng(0)
    n, J = 20_000, 400
    cl = rng.integers(0, J, size=n)
    d = rng.integers(0, 2, size=n).astype(float)
    x1 = rng.normal(size=n)
    u = rng.normal(size=(J, 2)) * [0.8, 0.4]
    y = 1.0 + 0.5 * d + 0.3 * x1 + u[cl, 0] + u[cl, 1] * x1 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "cl": cl})

    start = time.perf_counter()
    ri = sp.mixed(df, "y", ["d", "x1"], "cl")
    rs = sp.mixed(df, "y", ["d", "x1"], "cl", x_random=["x1"])
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0

    # The fits are the right ones, not just quick ones.
    assert rs.fixed_effects["d"] == pytest.approx(0.5, abs=4 * rs.std_errors["d"])
    assert rs.variance_components["var(_cons)"] == pytest.approx(0.64, rel=0.25)
    assert rs.variance_components["var(x1)"] == pytest.approx(0.16, rel=0.25)
    assert rs.variance_components["var(Residual)"] == pytest.approx(1.0, rel=0.05)
    # Dropping a real random slope must cost likelihood.
    assert rs.log_likelihood > ri.log_likelihood + 100.0
