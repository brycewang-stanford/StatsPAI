"""The exact inner solver of the nested synthetic-control fit.

With fewer predictors than donors the inner problem ``min ||y - X w||^2``
on the simplex is rank deficient, and it used to go to SLSQP on every one
of the few thousand evaluations of the outer search: 55 seconds for one
fit on the Proposition 99 data. When the treated unit lies outside the
donors' hull the minimiser is nevertheless unique, and
``_unique_simplex_lsq`` returns it exactly with a certificate. Where the
certificate fails (a treated unit inside the hull, a predictor weighted to
zero) every minimiser has the same fitted values, and the minimum-norm one
is returned: a property of the problem, where SLSQP's choice depended on its
starting point.
"""

import time
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.synth import _core


def _problem(seed, n=4, J=30, inside=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, J))
    if inside:
        w = rng.dirichlet(np.ones(J))
        return X @ w, X
    return X.mean(axis=1) + 6.0 * rng.normal(size=n), X


@pytest.mark.parametrize("seed", range(12))
def test_exact_solution_satisfies_the_optimality_conditions(seed):
    y, X = _problem(seed)
    w = _core._unique_simplex_lsq(y, X)
    assert w is not None
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0, abs=1e-12)
    grad = -2.0 * X.T @ (y - X @ w)
    support = w > 0
    assert support.sum() <= X.shape[0] + 1
    lam = grad[support].mean()
    scale = max(1.0, np.abs(grad).max())
    np.testing.assert_allclose(grad[support], lam, atol=1e-9 * scale)
    assert (grad - lam > -1e-9 * scale).all()


@pytest.mark.parametrize("seed", range(6))
def test_it_is_the_point_slsqp_approximates(seed):
    y, X = _problem(seed)
    exact = _core._unique_simplex_lsq(y, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy import optimize

        J = X.shape[1]
        res = optimize.minimize(
            lambda w: float(((y - X @ w) ** 2).sum()),
            np.full(J, 1 / J),
            jac=lambda w: -2.0 * X.T @ (y - X @ w),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * J,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    approx = np.clip(res.x, 0, None)
    approx /= approx.sum()
    # SLSQP stops about 1e-5 short in the weights; the exact loss is lower
    np.testing.assert_allclose(exact, approx, atol=5e-4)
    loss = lambda w: float(((y - X @ w) ** 2).sum())  # noqa: E731
    assert loss(exact) <= loss(approx) + 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_no_certificate_inside_the_hull(seed):
    # a target inside the donors' hull is matched by a whole face of weights
    y, X = _problem(seed, inside=True)
    assert _core._unique_simplex_lsq(y, X) is None
    # the public solver returns the minimum-norm point of that face
    w = _core.solve_simplex_weights(y, X)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(((y - X @ w) ** 2).sum()) < 1e-8


@pytest.mark.parametrize("seed", range(8))
def test_non_unique_problems_get_the_minimum_norm_weights(seed):
    from scipy import optimize

    y, X = _problem(seed, inside=True)
    J = X.shape[1]
    w = _core.solve_simplex_weights(y, X)
    A = np.vstack([X, np.ones((1, J))])
    b = np.append(y, 1.0)
    # exact fit, on the simplex
    assert np.abs(A @ w - b).max() < 1e-10 and w.min() >= 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = optimize.minimize(
            lambda v: float(v @ v),
            np.full(J, 1 / J),
            jac=lambda v: 2 * v,
            method="trust-constr",
            bounds=optimize.Bounds(0, 1),
            constraints=optimize.LinearConstraint(A, b, b),
            options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 5000},
        )
    # a generic QP solver finds the same point, and never a shorter one
    assert float(w @ w) <= float(ref.x @ ref.x) + 1e-10
    np.testing.assert_allclose(w, ref.x, atol=1e-4)


def test_the_selection_does_not_depend_on_donor_order():
    y, X = _problem(2, inside=True)
    perm = np.random.default_rng(0).permutation(X.shape[1])
    a = _core.solve_simplex_weights(y, X)
    b = _core.solve_simplex_weights(y, X[:, perm])
    np.testing.assert_allclose(a[perm], b, atol=1e-10)


def test_a_unique_representation_inside_the_hull_is_recovered_exactly():
    # three donors in general position span a triangle: a point inside it
    # has one set of weights, although the fit is exact and every reduced
    # gradient is zero (which is why the uniqueness certificate cannot fire)
    X = np.array([[0.0, 4.0, 1.0], [0.0, 0.0, 3.0]])
    truth = np.array([0.5, 0.3, 0.2])
    w = _core.solve_simplex_weights(X @ truth, X)
    np.testing.assert_allclose(w, truth, atol=1e-13)


def test_least_norm_handles_a_degenerate_vertex():
    # the feasible set is the single point w0: nothing to improve, no loop
    A = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    w0 = np.array([0.5, 0.5, 0.0])
    w = _core._least_norm_weights(A, A @ w0, w0)
    np.testing.assert_allclose(w, w0, atol=1e-12)


def test_no_certificate_when_a_row_is_weighted_out():
    # a zero row (a predictor with V = 0) leaves ties among the donors
    y, X = _problem(3)
    X = X.copy()
    X[1:] = 0.0
    y = y.copy()
    y[1:] = 0.0
    X[0, :5] = X[0].max()  # several donors equally good on the one live row
    y[0] = X[0].max() + 1.0
    assert _core._unique_simplex_lsq(y, X) is None


def test_single_best_donor():
    X = np.array([[0.0, 1.0, 5.0], [0.0, 1.0, 5.0]])
    y = np.array([9.0, 9.0])
    w = _core._unique_simplex_lsq(y, X)
    np.testing.assert_allclose(w, [0.0, 0.0, 1.0])


def test_full_rank_problems_are_untouched():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    a = _core.solve_simplex_weights(y, X)
    b = _core._eq_bounded_lsq(X, y, 0.0, 1.0)
    np.testing.assert_array_equal(a, b)


def test_nested_fit_is_fast_and_agrees_with_the_previous_value():
    df = sp.california_prop99()
    spec = [
        ("packspercapita", 1975, "mean"),
        ("packspercapita", 1980, "mean"),
        ("packspercapita", 1988, "mean"),
        ("packspercapita", slice(1970, 1974), "mean"),
    ]
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.synth(
            df,
            "packspercapita",
            "state",
            "year",
            "California",
            1989,
            method="classic",
            special_predictors=spec,
            v_method="nested",
            placebo=False,
        )
    elapsed = time.perf_counter() - start
    # -19.907271918 with SLSQP in the inner problem (55 s). The outer search
    # is the same; its inner solutions are now exact.
    assert fit.estimate == pytest.approx(-19.907271918, rel=1e-7)
    assert elapsed < 20
