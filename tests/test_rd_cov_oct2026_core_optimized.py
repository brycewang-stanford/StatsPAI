"""Shared RD primitives and ``sp.rd_optimized`` branches not reached elsewhere.

``rd/_core.py`` (kernel constants, sandwich variance, local-polynomial WLS,
covariate-rank guard), ``rd/_locrand_core.py`` (robust standard errors,
window sequences, Hotelling's test) and ``rd/optimized.py`` (worst-case bias
integral and the refusals of ``sp.rd_optimized``). Closed forms are written
out in the tests; nothing here is compared with itself.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.rd import _core as co
from statspai.rd import _locrand_core as lr
from statspai.rd import optimized as op

# ------------------------------------------------------------ _core


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
def test_kernel_constants_are_the_kernel_moments(kernel):
    const = co._kernel_constants(kernel)

    def k(u):
        return float(co._kernel_fn(np.array([u]), kernel)[0])

    mass = integrate.quad(k, -1, 1)[0]
    mu2 = integrate.quad(lambda u: u**2 * k(u), -1, 1, points=[0])[0] / mass
    nu0 = integrate.quad(lambda u: k(u) ** 2, -1, 1, points=[0])[0] / mass**2
    # tabulated constants are for the kernel normalised to integrate to one;
    # adaptive quadrature of a piecewise polynomial: 1e-10
    assert const["mu_2"] == pytest.approx(mu2, abs=1e-10)
    assert const["nu_0"] == pytest.approx(nu0, abs=1e-10)
    assert co._kernel_mse_constant(kernel) == const["C_K"]
    # an unknown kernel gets the triangular constant, the documented default
    assert co._kernel_mse_constant("nope") == co._kernel_constants("triangular")["C_K"]


def test_sandwich_variance_recovers_the_kernel_weight_from_the_design():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.uniform(0, 1, n)
    w = 1 - x  # triangular weights
    y = 1 + 2 * x + rng.normal(0, 0.3 + x, n)
    X = np.column_stack([np.ones(n), x])
    Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    resid = y - X @ beta
    given = co._sandwich_variance(Xw, yw, beta, resid, n, 2, weights=w)
    inferred = co._sandwich_variance(Xw, yw, beta, resid, n, 2)
    # Xw[:, 0]**2 is w up to one rounding of sqrt
    np.testing.assert_allclose(inferred, given, rtol=1e-12)
    # HC1 with the CCT meat X' W diag(e^2) W X
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    meat = X.T @ (X * (w**2 * resid**2)[:, None])
    np.testing.assert_allclose(given, n / (n - 2) * bread @ meat @ bread, rtol=1e-10)


def test_sandwich_variance_survives_an_exactly_singular_design():
    # A zero column makes X'X exactly singular (inv raises); the pseudo-inverse
    # then gives the variance of the identified coefficient and zero elsewhere.
    n = 30
    rng = np.random.default_rng(1)
    Xw = np.column_stack([np.ones(n), np.zeros(n)])
    y = 2.0 + rng.normal(0, 1, n)
    beta = np.array([y.mean(), 0.0])
    resid = y - y.mean()
    V = co._sandwich_variance(Xw, y, beta, resid, n, 2, weights=np.ones(n))
    assert np.all(np.isfinite(V))
    assert V[1, 1] == 0.0 and V[0, 1] == 0.0
    # HC1 variance of a mean with k = 2: sum e^2 / n^2 * n / (n - 2)
    assert V[0, 0] == pytest.approx(np.sum(resid**2) / n**2 * n / (n - 2), rel=1e-12)


def test_local_poly_wls_flags_a_window_too_thin_to_fit():
    x = np.array([0.1, 0.2, 0.3, 2.0, 3.0])
    beta, vcov, n_eff = co._local_poly_wls(x * 2, x, 0.5, 1, "uniform")
    # three points inside h for a two-parameter fit: needs k + 2 = 4
    assert n_eff == 0
    np.testing.assert_array_equal(beta, np.zeros(2))
    # a huge variance, so that any downstream test statistic is ~0
    np.testing.assert_array_equal(vcov, np.eye(2) * 1e10)


def test_local_poly_wls_falls_back_to_least_squares_when_singular():
    # every x identical inside the window: the slope is not identified, the
    # intercept-at-zero is not either, but the fitted value at x0 is the mean
    x = np.full(10, 0.25)
    y = np.arange(10.0)
    beta, vcov, n_eff = co._local_poly_wls(y, x, 1.0, 1, "uniform")
    assert n_eff == 10
    # minimum-norm solution reproduces the only identified quantity
    assert beta[0] + beta[1] * 0.25 == pytest.approx(y.mean(), rel=1e-10)
    assert vcov.shape == (2, 2)

    # a well-posed fit, for contrast: exact line recovered
    xs = np.linspace(0.05, 0.95, 20)
    beta, _, n_eff = co._local_poly_wls(1 + 3 * xs, xs, 1.0, 1, "triangular")
    assert n_eff == 20
    np.testing.assert_allclose(beta, [1.0, 3.0], atol=1e-10)


def test_complete_cases_with_nothing_to_check_returns_the_frame():
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    out, dropped = co._complete_cases(frame, [None, []])
    assert out is frame and dropped == 0
    out, dropped = co._complete_cases(frame, ["a", None])
    assert len(out) == 1 and dropped == 1


def test_covariate_rank_guard():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 50)
    z = rng.normal(size=50)
    # nothing to check, or too few rows to tell: silent
    assert co._check_covariate_rank(x, None, 1) is None
    assert co._check_covariate_rank(x, np.empty((50, 0)), 1) is None
    assert co._check_covariate_rank(x[:3], x[:3] ** 2, 2) is None
    # an independent covariate, given as a vector
    assert co._check_covariate_rank(x, z, 1) is None
    # x itself is in the polynomial basis
    with pytest.raises(ValueError, match="rank deficient") as err:
        co._check_covariate_rank(x, 2 * x + 1, 1, names=["twice_x"], where="unit")
    assert "unit:" in str(err.value) and "(twice_x)" in str(err.value)
    # x^2 is collinear only once the basis is quadratic
    assert co._check_covariate_rank(x, x**2, 1) is None
    with pytest.raises(ValueError, match="rank deficient"):
        co._check_covariate_rank(x, x**2, 2)


# ------------------------------------------------------------ _locrand_core


def test_canonical_kernel_refuses_unknown_names():
    assert lr.canonical_kernel("Triangular") == "triangular"
    with pytest.raises(MethodIncompatibility, match="Unknown kernel 'gauss'"):
        lr.canonical_kernel("gauss")


def test_hc_standard_errors_against_their_textbook_forms():
    rng = np.random.default_rng(4)
    n1, n0 = 18, 25
    t = np.r_[np.ones(n1), np.zeros(n0)]
    xc = np.r_[rng.uniform(0, 1, n1), rng.uniform(-1, 0, n0)]
    y = 1.0 * t + rng.normal(0, 1 + t, n1 + n0)
    w = np.ones(n1 + n0)
    y1, y0 = y[t == 1], y[t == 0]
    # p = 0, equal weights: HC2 is Welch's standard error ...
    welch = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
    assert lr.hc_se(y, xc, t, w, 0, vce="hc2") == pytest.approx(welch, rel=1e-12)
    assert lr.hc2_se(y, xc, t, w, 0) == lr.hc_se(y, xc, t, w, 0, vce="hc2")
    # ... HC1 is HC0 times n / (n - 2) ...
    hc0 = y1.var(ddof=0) / n1 + y0.var(ddof=0) / n0
    n = n1 + n0
    assert lr.hc_se(y, xc, t, w, 0, vce="hc1") == pytest.approx(
        np.sqrt(hc0 * n / (n - 2)), rel=1e-12
    )
    # ... and HC3 divides each squared residual by (1 - 1/n_s)^2.
    hc3 = (
        y1.var(ddof=0) / n1 / (1 - 1 / n1) ** 2
        + y0.var(ddof=0) / n0 / (1 - 1 / n0) ** 2
    )
    assert lr.hc_se(y, xc, t, w, 0, vce="hc3") == pytest.approx(np.sqrt(hc3), rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="vce must be 'hc1', 'hc2' or"):
        lr.hc_se(y, xc, t, w, 0, vce="hc0")


def test_hc_standard_errors_are_nan_when_not_defined():
    # a saturated side: two points for a line, leverage 1, so HC2 has 0/0
    t = np.array([1, 1, 0, 0, 0, 0.0])
    xc = np.array([0.2, 0.8, -0.1, -0.4, -0.6, -0.9])
    y = np.array([1.0, 2.0, 0.1, 0.3, 0.2, 0.5])
    w = np.ones(6)
    assert np.isnan(lr.hc_se(y, xc, t, w, 1, vce="hc2"))
    # hc1 needs n > 2 (p + 1): four points, four coefficients
    keep = np.array([True, True, True, True, False, False])
    assert np.isnan(lr.hc_se(y[keep], xc[keep], t[keep], w[keep], 1, vce="hc1"))


def test_linear_functional_refuses_an_unidentified_polynomial():
    t = np.array([1, 1, 1, 0, 0, 0.0])
    xc = np.array([0.5, 0.5, 0.5, -0.1, -0.4, -0.6])  # one support point right
    with pytest.raises(DataInsufficient, match="order 1 is not identified"):
        lr.linear_functional(xc, t, np.ones(6), 1)
    # p = 0 is fine there: g @ y is the difference in means
    g, _ = lr.linear_functional(xc, t, np.ones(6), 0)
    y = np.arange(6.0)
    assert g @ y == pytest.approx(y[:3].mean() - y[3:].mean(), rel=1e-12)


def test_asymptotic_power_is_the_two_sided_z_power():
    assert np.isnan(lr.asymptotic_power(1.0, 0.0, 0.05))
    assert np.isnan(lr.asymptotic_power(1.0, np.nan, 0.05))
    # no shift: the size of the test
    assert lr.asymptotic_power(0.0, 1.0, 0.05) == pytest.approx(0.05, abs=1e-12)
    # P(|N(d/se, 1)| > z) through the noncentral chi-square
    want = stats.ncx2.sf(stats.norm.ppf(0.975) ** 2, 1, (0.5 / 0.2) ** 2)
    assert lr.asymptotic_power(0.5, 0.2, 0.05) == pytest.approx(want, rel=1e-9)


def test_window_sequence_rules_and_refusals():
    x = np.r_[-np.arange(1, 31) / 30.0, np.arange(1, 31) / 30.0]
    with pytest.raises(MethodIncompatibility, match="at most one of wobs= and wstep="):
        lr.window_sequence(x, 0.0, nwindows=3, wobs=5, wstep=0.1)
    with pytest.raises(MethodIncompatibility, match="wasymmetric=True"):
        lr.window_sequence(x, 0.0, nwindows=3, wasymmetric=True, wmin=0.1)
    with pytest.raises(MethodIncompatibility, match="wmin must be positive"):
        lr.window_sequence(x, 0.0, nwindows=3, wmin=0.0)
    with pytest.raises(DataInsufficient, match="Fewer than obsmin=40"):
        lr.window_sequence(x, 0.0, nwindows=3, obsmin=40)

    # default growth: obsmin = 10 first, then 5 more observations a side
    wins = lr.window_sequence(x, 0.0, nwindows=3)
    np.testing.assert_allclose(wins, [(10 / 30,) * 2, (15 / 30,) * 2, (20 / 30,) * 2])
    # fixed steps from a fixed start
    wins = lr.window_sequence(x, 0.0, nwindows=3, wmin=0.2, wstep=0.1)
    np.testing.assert_allclose(wins, [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4)])
    # the sequence stops once the window holds every observation
    wins = lr.window_sequence(x, 0.0, nwindows=50, wmin=0.9, wstep=0.2)
    np.testing.assert_allclose(wins, [(0.9, 0.9), (1.1, 1.1)])
    # ... or when fewer than wobs further observations remain
    wins = lr.window_sequence(x, 0.0, nwindows=50, obsmin=25, wobs=4)
    np.testing.assert_allclose(wins, [(25 / 30,) * 2, (29 / 30,) * 2])


def test_mass_point_windows_need_both_sides():
    x = np.array([0.1, 0.2, 0.2, 0.5])
    with pytest.raises(DataInsufficient, match="both sides of the cutoff"):
        lr.mass_point_windows(x, 0.0, nwindows=3)
    both = np.r_[-x, x]
    np.testing.assert_allclose(
        lr.mass_point_windows(both, 0.0, nwindows=5),
        [(0.1, 0.1), (0.2, 0.2), (0.5, 0.5)],
    )


def test_hotelling_t2_matches_the_definition_and_its_f_reference():
    rng = np.random.default_rng(6)
    n, k = 40, 2
    Z = rng.normal(size=(n, k))
    lab = (np.arange(n) < 15).astype(float)
    Z[lab == 1] += 0.8
    t2 = lr.hotelling_t2(Z, lab)[0]
    z1, z0 = Z[lab == 1], Z[lab == 0]
    d = z1.mean(0) - z0.mean(0)
    S = ((15 - 1) * np.cov(z1.T) + (25 - 1) * np.cov(z0.T)) / (n - 2)
    want = 15 * 25 / n * d @ np.linalg.solve(S, d)
    assert t2 == pytest.approx(want, rel=1e-10)
    # T2 (n - k - 1) / ((n - 2) k) ~ F(k, n - k - 1)
    f = (n - k - 1) / ((n - 2) * k) * want
    assert lr.hotelling_pvalue_f(t2, n, k) == pytest.approx(
        stats.f.sf(f, k, n - k - 1), rel=1e-10
    )
    assert np.isnan(lr.hotelling_pvalue_f(np.nan, n, k))
    assert np.isnan(lr.hotelling_pvalue_f(3.0, 3, 2))  # no residual df

    # a constant covariate makes the pooled covariance singular: NaN, not a
    # crash; and a relabelling with an empty arm is NaN as well
    Zc = np.column_stack([Z[:, 0], np.ones(n)])
    assert np.isnan(lr.hotelling_t2(Zc, lab)).all()
    assert np.isnan(lr.hotelling_t2(Z, np.zeros(n))).all()


# ------------------------------------------------------------ optimized


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    d = (rng.uniform(size=n) < 0.2 + 0.6 * (x >= 0)).astype(float)
    return pd.DataFrame({"y": y, "x": x, "d": d})


def test_worst_case_bias_equals_the_closed_form_for_local_linear_weights():
    rng = np.random.default_rng(8)
    xc = np.sort(rng.uniform(-1, 1, 120))
    right = xc >= 0
    gamma = np.zeros_like(xc)
    for mask, sign in ((right, 1.0), (~right, -1.0)):
        X = np.column_stack([np.ones(mask.sum()), xc[mask]])
        k = 1 - np.abs(xc[mask])
        gamma[mask] = (
            sign * np.linalg.solve((X * k[:, None]).T @ X, (X * k[:, None]).T)[0]
        )
    M = 1.7
    # Armstrong-Kolesar: for boundary local-linear weights with a triangular
    # kernel the Holder worst case is attained by f(x) = -M x^2 / 2 sign(x),
    # giving -M/2 sum_i w_i x_i^2 on each side.
    closed = -M / 2 * (gamma[right] @ xc[right] ** 2 - gamma[~right] @ xc[~right] ** 2)
    # exact piecewise-linear integral vs a finite sum: rounding only
    assert op.worst_case_bias(xc, gamma, M) == pytest.approx(closed, rel=1e-9)
    # linear in M
    assert op.worst_case_bias(xc, gamma, 2 * M) == pytest.approx(2 * closed, rel=1e-12)
    # weights that do not kill a linear trend have unbounded bias
    broken = gamma.copy()
    broken[right] = 1.0 / right.sum()
    assert op.worst_case_bias(xc, broken, M) == float("inf")


def test_optimized_primitives_on_empty_input():
    assert op._side_integral(np.array([]), np.array([])) == 0.0
    np.testing.assert_array_equal(
        op._nn_deviation(np.array([1.0]), np.array([2.0])), [0.0]
    )
    # no support points: a single unit cell, so the design keeps its shape
    design, _ = op._side_design(np.array([]), 4)
    assert design.shape[0] == 0 and design.shape[1] >= 3
    with pytest.raises(DataInsufficient, match="fewer than two distinct values"):
        op._project(
            np.array([-1.0, -0.5, 0.5, 1.0]),
            np.ones(4),
            np.array([0.0, -1.0, 1.0, 0.0]),
        )


@pytest.mark.parametrize(
    "kwargs, exc, fragment",
    [
        (dict(alpha=1.5), MethodIncompatibility, r"alpha must be in \(0, 1\)"),
        (dict(x="nope"), MethodIncompatibility, "column 'nope' not found"),
        (dict(h=-1.0), MethodIncompatibility, "h must be positive"),
        (dict(h=0.01), DataInsufficient, "inside the window"),
    ],
)
def test_rd_optimized_refusals(frame, kwargs, exc, fragment):
    with pytest.raises(exc, match=fragment):
        sp.rd_optimized(frame, **{**dict(y="y", x="x", M=1.0), **kwargs})


def test_rd_optimized_refuses_an_outcome_without_variation(frame):
    with pytest.raises(DataInsufficient, match="no variation near the cutoff"):
        sp.rd_optimized(frame.assign(y=1.0), y="y", x="x", M=1.0)


def test_rd_optimized_estimates_M_when_not_given_and_validates_M_fuzzy(frame):
    res = sp.rd_optimized(frame, y="y", x="x")
    mi = res.model_info
    assert mi["M_estimated"] is True and mi["M"] > 0
    # the reported bias is the exact integral for the weights actually used
    order = np.argsort(frame["x"].to_numpy(), kind="stable")
    xs = frame["x"].to_numpy()[order]
    ys = frame["y"].to_numpy()[order]
    pos = {idx: i for i, idx in enumerate(frame.index.to_numpy()[order])}
    rows = np.array([pos[i] for i in mi["index"]])
    w = np.asarray(mi["weights"])
    assert res.estimate == pytest.approx(w @ ys[rows], rel=1e-10)
    assert mi["max_bias"] == pytest.approx(
        op.worst_case_bias(xs[rows], w, mi["M"]), rel=1e-8
    )
    # the four moment conditions, to rounding
    r = xs[rows] >= 0
    assert w[r].sum() == pytest.approx(1.0, abs=1e-10)
    assert w[~r].sum() == pytest.approx(-1.0, abs=1e-10)
    assert abs(w[r] @ xs[rows][r]) < 1e-10 and abs(w[~r] @ xs[rows][~r]) < 1e-10
    # true jump 1.0 on a linear mean: any M > 0 bounds the curvature
    assert res.ci[0] < 1.0 < res.ci[1]

    with pytest.raises(MethodIncompatibility, match="M_fuzzy must be a non-negative"):
        sp.rd_optimized(frame, y="y", x="x", M=1.0, fuzzy="d", M_fuzzy=-1.0)
