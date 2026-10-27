"""``sp.akm``: known truth, exact identities and edge cases.

The reference comparison with pytwoway is in
``tests/reference_parity/test_akm_pytwoway_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, MethodIncompatibility, StatsPAIWarning


def _design(seed=7, n_workers=900, n_firms=40, n_years=3, p_move=0.12):
    """Limited mobility with sorting: few movers per firm, so the plug-in
    variances are visibly biased."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(0, 0.25, n_firms)
    alpha = rng.normal(0, 0.4, n_workers)
    rank = np.argsort(np.argsort(alpha)) / n_workers
    rows = []
    for w in range(n_workers):
        centre = rank[w] * (n_firms - 1)
        p = np.exp(-0.5 * ((np.arange(n_firms) - centre) / 5.0) ** 2)
        p /= p.sum()
        f = rng.choice(n_firms, p=p)
        for t in range(n_years):
            if t and rng.uniform() < p_move:
                f = rng.choice(n_firms, p=p)
            rows.append((w, f, t))
    d = pd.DataFrame(rows, columns=["worker", "firm", "year"])
    d["a"] = alpha[d.worker]
    d["p"] = np.sort(psi)[d.firm]
    # heteroskedastic: noisier at high-psi firms
    d["sd"] = 0.2 + 0.6 * (d.p - d.p.min()) / np.ptp(d.p)
    return d


_DENSE_CACHE: dict = {}


def _dense_B(sample: pd.DataFrame):
    key = (len(sample), tuple(sample.index[:5]), float(sample.worker.sum()))
    if key not in _DENSE_CACHE:
        _DENSE_CACHE[key] = _dense_B_uncached(sample)
    return _DENSE_CACHE[key]


def _dense_B_uncached(sample: pd.DataFrame):
    """P_ii and the three B_ii from the dense (X'X)^{-1} of [D, F]
    (one firm dropped), independent of the sparse implementation."""
    wc, _ = pd.factorize(sample.worker)
    fc, _ = pd.factorize(sample.firm)
    n, nw, nf = len(sample), wc.max() + 1, fc.max() + 1
    D = np.zeros((n, nw))
    D[np.arange(n), wc] = 1
    F = np.zeros((n, nf))
    F[np.arange(n), fc] = 1
    F = F[:, 1:]
    X = np.hstack([D, F])
    Z = np.linalg.solve(X.T @ X, X.T)
    M1 = np.eye(n) - 1.0 / n
    za, zp = Z[:nw], Z[nw:]
    DM, FM = M1 @ D, M1 @ F
    Ba = np.einsum("ni,ni->i", DM @ za, DM @ za) / n
    Bp = np.einsum("ni,ni->i", FM @ zp, FM @ zp) / n
    Bc = np.einsum("ni,ni->i", DM @ za, FM @ zp) / n
    P = np.einsum("ij,ji->i", X, Z)
    return P, Ba, Bp, Bc


@pytest.fixture(scope="module")
def mc():
    """Fixed design, 300 error draws: KSS is unbiased for the in-sample
    variance components, the plug-in is biased by sum_i B_ii sigma_i^2.

    The first draw goes through ``sp.akm``; the leverages depend on the
    design only, so the other draws reuse them and refit the effects."""
    from statspai.panel import _akm_kss as k

    d = _design()
    rng = np.random.default_rng(11)
    first = sp.akm(
        d.assign(y=d.a + d.p + rng.normal(0, d.sd)),
        "y",
        "worker",
        "firm",
        leverage="exact",
    )
    s = d.loc[first.data.index]
    wc, _ = pd.factorize(s.worker, sort=True)
    fc, _ = pd.factorize(s.firm, sort=True)
    n_w, n_f = wc.max() + 1, fc.max() + 1
    lev = None
    draws = [first.decomposition]
    for _ in range(299):
        y = (s.a + s.p + rng.normal(0, s.sd)).to_numpy()
        fit = k.fit_twoway(wc, fc, y, n_w, n_f)
        if lev is None:
            lev = k.leverages_exact(fit)
        c = k.corrected_components(fit, lev)
        draws.append(
            pd.DataFrame({key: c[key] for key in ("plugin", "homoskedastic", "kss")})
        )
    return d, first, draws


def test_kss_is_unbiased_and_plugin_bias_is_the_analytic_trace(mc):
    d, first, draws = mc
    s = d.loc[first.data.index]
    a, p = s.a.to_numpy(), s.p.to_numpy()
    truth = {
        "var_alpha": np.var(a),
        "var_psi": np.var(p),
        "cov": np.mean((a - a.mean()) * (p - p.mean())),
    }
    P, Ba, Bp, Bc = _dense_B(s)
    sig2 = s.sd.to_numpy() ** 2
    bias = {"var_alpha": Ba @ sig2, "var_psi": Bp @ sig2, "cov": Bc @ sig2}
    for comp in truth:
        plug = np.array([x.loc[comp, "plugin"] for x in draws])
        kss = np.array([x.loc[comp, "kss"] for x in draws])
        R = len(draws)
        # E[plug-in] = theta + sum B_ii sigma_i^2 exactly (design fixed)
        assert abs(plug.mean() - truth[comp] - bias[comp]) <= 4 * plug.std(
            ddof=1
        ) / np.sqrt(R), comp
        # E[KSS] = theta up to the O(1/n) term of centring y at its mean
        assert abs(kss.mean() - truth[comp]) <= 4 * kss.std(ddof=1) / np.sqrt(R), comp
    # the design makes the bias material: at least 30 Monte Carlo SEs
    plug = np.array([x.loc["var_psi", "plugin"] for x in draws])
    assert bias["var_psi"] > 30 * plug.std(ddof=1) / np.sqrt(len(draws))


def test_homoskedastic_correction_has_its_analytic_expectation(mc):
    d, first, draws = mc
    s = d.loc[first.data.index]
    p = s.p.to_numpy()
    P, Ba, Bp, Bc = _dense_B(s)
    sig2 = s.sd.to_numpy() ** 2
    n = len(s)
    dof = n - (s.worker.nunique() + s.firm.nunique() - 1)
    e_sigma = ((1 - P) @ sig2) / dof  # E[RSS] / dof
    truth = np.var(p)
    expect = truth + Bp @ sig2 - e_sigma * Bp.sum()
    ho = np.array([x.loc["var_psi", "homoskedastic"] for x in draws])
    assert abs(ho.mean() - expect) <= 4 * ho.std(ddof=1) / np.sqrt(len(draws))


def test_leverages_and_weights_equal_the_dense_formulas():
    from statspai.panel import _akm_kss as k

    d = _design(seed=3, n_workers=300, n_firms=15)
    d["y"] = d.a + d.p + np.random.default_rng(0).normal(0, d.sd)
    r = sp.akm(d, "y", "worker", "firm")
    s = d.loc[r.data.index]
    P, Ba, Bp, Bc = _dense_B(s)
    wc, _ = pd.factorize(s.worker, sort=True)
    fc, _ = pd.factorize(s.firm, sort=True)
    fit = k.fit_twoway(wc, fc, s.y.to_numpy(), wc.max() + 1, fc.max() + 1)
    lev = k.leverages_exact(fit)
    # closed forms vs a dense inverse of the same matrix: rounding only
    np.testing.assert_allclose(lev.P, P, atol=1e-12)
    np.testing.assert_allclose(lev.B_alpha, Ba, atol=1e-14)
    np.testing.assert_allclose(lev.B_psi, Bp, atol=1e-14)
    np.testing.assert_allclose(lev.B_cov, Bc, atol=1e-14)
    np.testing.assert_allclose(r.data["P_ii"].to_numpy(), P, atol=1e-12)


def test_jla_reproduces_the_exact_correction():
    d = _design(seed=5, n_workers=600, n_firms=30)
    d["y"] = d.a + d.p + np.random.default_rng(1).normal(0, d.sd)
    ex = sp.akm(d, "y", "worker", "firm", leverage="exact").decomposition
    jl = sp.akm(d, "y", "worker", "firm", leverage="jla", n_draws=2000, seed=9)
    assert jl.leverage == {
        "method": "jla",
        "n_draws": 2000,
        "seed": 9,
        "min_P": jl.leverage["min_P"],
        "max_P": jl.leverage["max_P"],
    }
    for comp in ("var_alpha", "var_psi", "cov"):
        correction = abs(ex.loc[comp, "plugin"] - ex.loc[comp, "kss"])
        # 2000 Rademacher draws: each sketched quadratic form has relative
        # sd sqrt(2/2000) ~ 3%; the correction (a sum of them) is
        # reproduced to well within 10% of its size
        assert abs(jl.decomposition.loc[comp, "kss"] - ex.loc[comp, "kss"]) <= (
            0.1 * correction
        ), comp
    # same seed, same numbers
    again = sp.akm(d, "y", "worker", "firm", leverage="jla", n_draws=2000, seed=9)
    pd.testing.assert_frame_equal(again.decomposition, jl.decomposition)


def test_plugin_decomposition_adds_up_and_effects_fit():
    d = _design(seed=2, n_workers=200, n_firms=12)
    d["y"] = 1.0 + d.a + d.p + np.random.default_rng(2).normal(0, d.sd)
    r = sp.akm(d, "y", "worker", "firm", variance="plugin")
    dec = r.decomposition["plugin"]
    # residuals are orthogonal to the fitted effects and average zero
    assert dec["var_y"] == pytest.approx(
        dec["var_alpha"] + dec["var_psi"] + 2 * dec["cov"] + dec["var_resid"],
        abs=1e-12,
    )
    s = d.loc[r.data.index]
    fitted = s.worker.map(r.fixef()["worker"]) + s.firm.map(r.fixef()["firm"])
    np.testing.assert_allclose(fitted + r.data["resid"], s.y, atol=1e-10)
    # same residuals as dense least squares on worker and firm dummies
    D = pd.get_dummies(s.worker).to_numpy(float)
    F = pd.get_dummies(s.firm).to_numpy(float)[:, 1:]
    X = np.hstack([D, F])
    beta, *_ = np.linalg.lstsq(X, s.y.to_numpy(), rcond=None)
    np.testing.assert_allclose(r.data["resid"], s.y - X @ beta, atol=1e-10)


def test_kss_does_not_change_when_a_constant_is_added():
    d = _design(seed=4, n_workers=300, n_firms=15)
    d["y"] = d.a + d.p + np.random.default_rng(4).normal(0, d.sd)
    a = sp.akm(d, "y", "worker", "firm").decomposition
    b = sp.akm(d.assign(y=d.y + 7.0), "y", "worker", "firm").decomposition
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-10)


def test_covariates_are_partialled_out_with_the_full_model():
    d = _design(seed=6, n_workers=300, n_firms=15)
    rng = np.random.default_rng(6)
    d["x"] = rng.normal(size=len(d)) + d.a
    d["y"] = d.a + d.p + 0.5 * d.x + 0.1 * d.year + rng.normal(0, d.sd)
    r = sp.akm(d, "y", "worker", "firm", time="year", covariates=["x"])
    s = d.loc[r.data.index]
    full = sp.hdfe_ols("y ~ x + i.year | worker + firm", data=s)
    assert r.coefficients.loc["x", "coef"] == pytest.approx(full.params["x"], abs=1e-10)
    assert r.coefficients.loc["year::2", "coef"] == pytest.approx(
        full.params["year::2"], abs=1e-10
    )
    np.testing.assert_allclose(
        r.data["xb"],
        s.x * full.params["x"]
        + (s.year == 1) * full.params["year::1"]
        + (s.year == 2) * full.params["year::2"],
        atol=1e-10,
    )
    assert r.params["x"] == r.coefficients.loc["x", "coef"]


def test_connected_sets_and_counts():
    # two islands; worker 9 seen once (pendant); firm 5 linked by one row
    rows = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
        (2, 1, 0),
        (2, 0, 1),
        (3, 0, 0),
        (3, 0, 1),
        (9, 1, 0),
        (4, 1, 0),
        (4, 5, 1),
        (5, 2, 0),
        (5, 3, 1),
        (6, 3, 0),
        (6, 2, 1),
    ]
    d = pd.DataFrame(rows, columns=["worker", "firm", "year"])
    d["y"] = np.arange(len(d), dtype=float) * 0.1
    lg = sp.akm(d, "y", "worker", "firm", variance="plugin")
    assert lg.sample["n_firms"] == 3  # firms 0, 1, 5
    assert lg.sample["dropped"]["not_connected"] == 4
    lo = sp.akm(d, "y", "worker", "firm", variance="plugin", connected="leave_one_out")
    # bridges (P_ii = 1): worker 9's only row, and worker 4's two rows,
    # each the only link of its pair (firm 5 hangs on (4, 5); without it
    # worker 4 hangs on (4, 1)); then the larger island is kept
    kept = set(map(tuple, d.loc[lo.data.index, ["worker", "firm"]].to_numpy()))
    assert kept == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0)}
    assert lo.sample["dropped"]["not_leave_one_out"] == 3
    al = sp.akm(d, "y", "worker", "firm", variance="none", connected="all")
    assert al.sample["n_components"] == 2 and al.sample["n_obs"] == len(d)


def test_refusals_and_warnings():
    d = _design(seed=1, n_workers=60, n_firms=6)
    d["y"] = d.a + d.p
    with pytest.raises(MethodIncompatibility, match="leave_one_out"):
        sp.akm(d, "y", "worker", "firm", variance="kss", connected="largest")
    with pytest.raises(MethodIncompatibility):
        sp.akm(d, "y", "worker", "firm", variance="ols")
    with pytest.raises(ColumnNotFound):
        sp.akm(d, "y", "worker", "establishment")
    two = pd.concat([d, d.assign(firm=d.firm + 100, worker=d.worker + 1000)])
    with pytest.warns(StatsPAIWarning, match="connected sets"):
        r = sp.akm(two, "y", "worker", "firm", variance="plugin", connected="all")
    assert r.decomposition is None


def test_duplicate_index_labels_are_kept():
    d = _design(seed=8, n_workers=120, n_firms=8)
    d["y"] = d.a + d.p + np.random.default_rng(8).normal(0, d.sd)
    base = sp.akm(d, "y", "worker", "firm")
    dup = d.copy()
    dup.index = np.zeros(len(d), dtype=int)  # every label 0
    r = sp.akm(dup, "y", "worker", "firm")
    pd.testing.assert_frame_equal(r.decomposition, base.decomposition)
    assert (r.data.index == 0).all() and len(r.data) == len(base.data)
