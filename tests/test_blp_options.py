"""``sp.blp``: the outer loop, boundary inference, diversion and markups.

Every assertion is an identity or a known value; the cross-package numbers
are in ``tests/reference_parity/test_blp_pyblp_parity.py``, and the logit,
nested logit and instruments are covered in ``tests/test_blp_berry_logit.py``.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.structural._blp_post import diversion_matrix, firm_markups

_blp = importlib.import_module("statspai.structural.blp")


def _panel(seed=3, n_markets=60, n_products=5, sigma_x=0.0):
    """Logit (``sigma_x = 0``) or random-coefficient demand, endogenous price."""
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(400)
    rows = []
    for m in range(n_markets):
        x = rng.uniform(0, 2, n_products)
        xi = rng.normal(0, 0.3, n_products)
        cost = rng.uniform(0.5, 2, n_products)
        z2 = rng.uniform(0, 1, n_products)
        price = 0.8 * cost + 0.5 * z2 + 0.4 * xi + rng.uniform(0, 0.5, n_products)
        delta = 0.5 + 1.0 * x - 1.5 * price + xi
        v = delta[:, None] + sigma_x * x[:, None] * nu[None, :]
        e = np.exp(v)
        share = (e / (1 + e.sum(axis=0))).mean(axis=1)
        for j in range(n_products):
            rows.append(
                {
                    "market_id": m,
                    "product_id": j,
                    "firm": j // 2,
                    "share": share[j],
                    "price": price[j],
                    "x1": x[j],
                    "cost": cost[j],
                    "z2": z2[j],
                }
            )
    return pd.DataFrame(rows)


KW = dict(shares="share", prices="price", x_linear=["x1"], n_draws=40, seed=0)
IV = ["cost", "z2"]


@pytest.fixture(scope="module")
def df():
    return _panel()


@pytest.fixture(scope="module")
def logit(df):
    return sp.blp(df, x_random=[], instruments=IV, firm_id="firm", **KW)


# ---------------------------------------------------------------------------
# Plain logit: x_random=[]
# ---------------------------------------------------------------------------


def test_empty_x_random_is_two_step_gmm_logit(df, logit):
    """The plain logit is the linear two-step GMM, computed here by hand."""
    s0 = 1 - df.groupby("market_id")["share"].transform("sum")
    y = (np.log(df["share"]) - np.log(s0)).to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x1"], df["price"]])
    Z = np.column_stack([np.ones(len(df)), df["x1"], df["cost"], df["z2"]])
    N = len(df)

    def gmm(W):
        A = X.T @ Z @ W @ Z.T @ X
        return np.linalg.solve(A, X.T @ Z @ W @ Z.T @ y)

    b1 = gmm(np.linalg.inv(Z.T @ Z / N))
    u = y - X @ b1
    b2 = gmm(np.linalg.inv((Z * u[:, None]).T @ (Z * u[:, None]) / N))
    np.testing.assert_allclose(logit.linear_params.to_numpy(), b2, rtol=1e-10)
    assert len(logit.nonlinear_params) == 0
    assert logit.converged
    assert logit.hansen_df == 1
    assert logit.model_info == {}


def test_logit_diversion_is_share_proportional(df, logit):
    """Logit: ``D_jk = s_k / (1 - s_j)`` whatever the price coefficient."""
    m = 4
    s = df.loc[df.market_id == m, "share"].to_numpy()
    D = logit.diversion_ratios(m, include_outside=True)
    expect = np.tile(s, (len(s), 1)) / (1 - s)[:, None]
    np.fill_diagonal(expect, 0.0)
    np.testing.assert_allclose(D.drop(columns="outside").to_numpy(), expect, rtol=1e-9)
    np.testing.assert_allclose(D["outside"], (1 - s.sum()) / (1 - s), rtol=1e-9)
    assert list(logit.diversion_ratios(m).columns) == list(range(len(s)))


def test_logit_single_product_markup(df, logit):
    """Single-product Bertrand logit: ``p - mc = -1 / (alpha (1 - s))``."""
    a = logit.linear_params["price"]
    mk = logit.markups(firm_ids="single")
    np.testing.assert_allclose(
        mk.to_numpy(), -1 / (a * (1 - df["share"].to_numpy())), rtol=1e-9
    )
    mc = logit.marginal_costs(firm_ids="single")
    np.testing.assert_allclose(mc.to_numpy(), df["price"].to_numpy() - mk.to_numpy())


def test_multiproduct_markup_solves_the_first_order_conditions(df, logit):
    """``s_j + sum_{k in f(j)} (p_k - mc_k) ds_k/dp_j = 0`` for every product."""
    m = 7
    i = (df.market_id == m).to_numpy()
    mk = logit.markups(m)
    assert list(mk.index) == list(df.loc[i, "product_id"])
    np.testing.assert_allclose(mk.to_numpy(), logit.markups()[i].to_numpy())
    dS = logit._jacobian(m)
    firm = df.loc[i, "firm"].to_numpy()
    s = df.loc[i, "share"].to_numpy()
    foc = s + ((firm[:, None] == firm[None, :]) * dS.T) @ mk.to_numpy()
    np.testing.assert_allclose(foc, 0.0, atol=1e-14)
    # pricing two products jointly raises both markups; a firm with one
    # product prices it as before
    single = logit.markups(m, firm_ids="single").to_numpy()
    joint = pd.Series(firm).map(pd.Series(firm).value_counts()).to_numpy() > 1
    assert np.all(mk.to_numpy()[joint] > single[joint])
    np.testing.assert_allclose(mk.to_numpy()[~joint], single[~joint], rtol=1e-12)


def test_markups_need_firms(df):
    fit = sp.blp(df, x_random=[], instruments=IV, **KW)
    with pytest.raises(MethodIncompatibility, match="firm_id"):
        fit.markups()
    with pytest.raises(MethodIncompatibility, match="shape"):
        fit.markups(firm_ids=[0, 1])
    with pytest.raises(MethodIncompatibility, match="'single'"):
        fit.markups(firm_ids="monopoly")


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rc_df():
    return _panel(seed=5, n_markets=80, sigma_x=1.5)


@pytest.fixture(scope="module")
def rc_fit(rc_df):
    return sp.blp(rc_df, x_random=["x1"], instruments=IV + ["x1sq"], **_rc_kw(rc_df))


def _rc_kw(rc_df):
    if "x1sq" not in rc_df:
        g = rc_df.groupby("market_id")["x1"]
        rc_df["x1sq"] = g.transform(lambda v: (v**2).sum()) - rc_df["x1"] ** 2
    return KW


def test_analytic_gradient_matches_finite_differences(rc_df):
    kw = _rc_kw(rc_df)
    N = len(rc_df)
    s = rc_df["share"].to_numpy()
    mk = rc_df["market_id"].to_numpy()
    X = np.column_stack([np.ones(N), rc_df["x1"], rc_df["price"]])
    Xr = np.column_stack([np.ones(N), rc_df["x1"]])
    Z = np.column_stack([np.ones(N), rc_df[["x1", "cost", "z2", "x1sq"]]])
    W = np.linalg.inv(Z.T @ Z / N)
    draws = _blp._halton_sequence(kw["n_draws"], 2, seed=0)
    d0 = np.log(s) - np.log(1 - rc_df.groupby("market_id")["share"].transform("sum"))
    d0 = d0.to_numpy()

    def f(sig):
        return _blp._objective_and_gradient(
            np.asarray(sig, float), s, X, Xr, Z, W, draws, mk, d0, 1e-14, 5000
        )

    sig = np.array([0.7, 1.1])
    _, grad, _ = f(sig)
    h = 1e-6
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        fd = (f(sig + e)[0] - f(sig - e)[0]) / (2 * h)
        assert abs(grad[k] - fd) < 1e-6 * max(abs(fd), 1.0)


def test_accelerated_contraction_reaches_the_plain_fixed_point(rc_df):
    s = rc_df["share"].to_numpy()
    mk = rc_df["market_id"].to_numpy()
    mu = _blp._compute_mu(
        rc_df[["x1"]].to_numpy(), np.array([1.5]), _blp._halton_sequence(40, 1, seed=0)
    )
    d = np.zeros(len(s))
    fast, ok = _blp._contraction_mapping(s, d, mu, mk, 1e-13, 5000)
    assert ok
    plain = d.copy()
    for _ in range(5000):
        new = plain + np.log(s) - np.log(_blp._compute_market_shares(plain, mu, mk))
        done = np.max(np.abs(new - plain)) < 1e-13
        plain = new
        if done:
            break
    assert done
    np.testing.assert_allclose(fast, plain, rtol=0, atol=1e-11)
    np.testing.assert_allclose(_blp._compute_market_shares(fast, mu, mk), s, rtol=1e-11)


def test_optimisers_and_starts_agree(rc_df, rc_fit):
    kw = _rc_kw(rc_df)
    assert rc_fit.converged
    assert rc_fit.model_info["optimizer"] == "l-bfgs-b"
    assert len(rc_fit.model_info["starts"]) == 4
    assert rc_fit.model_info["projected_gradient_norm"] < 1e-5
    # the truth is 1.5; the estimate is interior and within three SEs
    sig = rc_fit.nonlinear_params["sigma_x1"]
    assert abs(sig - 1.5) < 3 * rc_fit.se_nonlinear["sigma_x1"]
    nm = sp.blp(
        rc_df,
        x_random=["x1"],
        instruments=IV + ["x1sq"],
        optimizer="nelder-mead",
        tol_outer=1e-10,
        maxiter=2000,
        **kw,
    )
    np.testing.assert_allclose(nm.nonlinear_params.to_numpy(), sig, rtol=1e-6)
    np.testing.assert_allclose(
        nm.linear_params.to_numpy(), rc_fit.linear_params.to_numpy(), rtol=1e-6
    )
    for s0 in ([2.5], {"x1": 0.1}):
        other = sp.blp(
            rc_df,
            x_random=["x1"],
            instruments=IV + ["x1sq"],
            sigma0=s0,
            n_starts=1,
            **kw,
        )
        assert len(other.model_info["starts"]) == 1
        np.testing.assert_allclose(other.nonlinear_params.to_numpy(), sig, rtol=1e-5)


def test_sigma0_and_optimizer_are_validated(df):
    kw = dict(x_random=["x1"], instruments=IV, **KW)
    with pytest.raises(MethodIncompatibility, match="sigma0 has 2"):
        sp.blp(df, sigma0=[0.5, 0.5], **kw)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.blp(df, sigma0=[-1.0], **kw)
    with pytest.raises(MethodIncompatibility, match="one value per x_random"):
        sp.blp(df, sigma0={"price": 1.0}, **kw)
    with pytest.raises(MethodIncompatibility, match="optimizer"):
        sp.blp(df, optimizer="bfgs", **kw)
    with pytest.raises(MethodIncompatibility, match="n_starts"):
        sp.blp(df, n_starts=0, **kw)
    with pytest.raises(MethodIncompatibility, match="twice"):
        sp.blp(df, x_random=["x1", "x1"], instruments=IV, **KW)
    with pytest.raises(ValueError, match="'const'"):
        sp.blp(df, x_random=["cost"], instruments=IV, **KW)


def test_sigma_at_the_bound_is_held_fixed_for_inference(df, logit):
    """The data are logit, so the standard deviation is estimated at zero.

    There ``d delta / d sigma`` vanishes (the utility deviation is odd in the
    draws), and the joint sandwich used to divide by it: standard errors of
    1e5 to 1e6 on this design. Holding the parameter at its bound gives the
    standard errors of the model without it, here the just-identified
    counterpart of the over-identified logit, so compare with that model.
    """
    fit = sp.blp(df, x_random=["x1"], instruments=IV, **KW)
    assert fit.nonlinear_params["sigma_x1"] == 0.0
    assert fit.model_info["sigma_at_bound"] == ["sigma_x1"]
    assert np.isnan(fit.se_nonlinear["sigma_x1"])
    np.testing.assert_allclose(
        fit.linear_params.to_numpy(), logit.linear_params.to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        fit.se_linear.to_numpy(), logit.se_linear.to_numpy(), rtol=1e-8
    )
    assert "At the zero bound" in fit.summary()


# ---------------------------------------------------------------------------
# Upward-sloping demand is reported, not returned as a number
# ---------------------------------------------------------------------------


def test_diversion_is_undefined_when_demand_slopes_up():
    dS = np.array([[-0.2, 0.05, 0.02], [0.05, 0.01, 0.03], [0.02, 0.03, -0.1]])
    D, outside, undefined = diversion_matrix(dS)
    assert undefined.tolist() == [False, True, False]
    assert np.isnan(D[1]).all() and np.isnan(outside[1])
    np.testing.assert_allclose(D[0], [0.0, 0.25, 0.1])
    np.testing.assert_allclose(outside[0], 0.65)
    np.testing.assert_allclose(
        firm_markups(dS[[0, 2]][:, [0, 2]], np.array([0.1, 0.2]), np.array([0, 1])),
        [0.5, 2.0],
    )


def test_positive_own_elasticities_warn():
    """A wide normal price coefficient: rich-tail consumers buy the dear good."""
    rng = np.random.default_rng(0)
    rows = []
    for m in range(40):
        x = rng.uniform(0, 1, 4)
        price = np.sort(rng.uniform(1, 12, 4))
        cost = price + rng.normal(0, 0.3, 4)
        share = np.array([0.2, 0.1, 0.05, 0.03]) * rng.uniform(0.8, 1.2, 4)
        for j in range(4):
            rows.append(
                dict(
                    market_id=m,
                    product_id=j,
                    share=share[j],
                    price=price[j],
                    x1=x[j],
                    cost=cost[j],
                    z2=rng.uniform(),
                )
            )
    d = pd.DataFrame(rows)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit = sp.blp(
            d,
            x_random=["price"],
            instruments=IV,
            sigma0=[3.0],
            n_starts=1,
            maxiter=0,
            **KW,
        )
    info = fit.model_info
    assert info["n_positive_own_elasticity"] == int((fit.own_elasticities > 0).sum())
    assert info["n_positive_own_elasticity"] > 0
    assert 0 < info["share_positive_price_coef"] < 1
    assert any("positive own-price elasticity" in str(x.message) for x in w)
    assert "positive own-price elasticity" in fit.summary()
    with pytest.warns(UserWarning, match="undefined"):
        m_bad = d.market_id[np.argmax(fit.own_elasticities.to_numpy() > 0)]
        fit.diversion_ratios(m_bad)


def test_nested_logit_diversion_and_markups(df):
    """Nested logit: the outside good is its own nest, so
    ``ds_0/dp_j = -alpha s_0 s_j`` and the diversion to it is
    ``s_0 / (1/(1-rho) - rho/(1-rho) s_{j|g} - s_j)``."""
    d = df.assign(nest=df["product_id"] // 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.blp(d, x_random=[], nest="nest", instruments=IV, firm_id="firm", **KW)
    rho = fit.rho
    m = 2
    i = (d.market_id == m).to_numpy()
    s = d.loc[i, "share"].to_numpy()
    within = s / d.loc[i].groupby("nest")["share"].transform("sum").to_numpy()
    D = fit.diversion_ratios(m, include_outside=True)
    expect = (1 - s.sum()) / (1 / (1 - rho) - rho / (1 - rho) * within - s)
    np.testing.assert_allclose(D["outside"].to_numpy(), expect, rtol=1e-10)
    np.testing.assert_allclose(D.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    mk = fit.markups(m).to_numpy()
    dS = fit._jacobian(m)
    firm = d.loc[i, "firm"].to_numpy()
    foc = s + ((firm[:, None] == firm[None, :]) * dS.T) @ mk
    np.testing.assert_allclose(foc, 0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Option scan, round 6 (2026-10-27)
# ---------------------------------------------------------------------------


def test_alpha_sets_the_confidence_level_of_the_summary(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.blp(df, x_random=[], instruments=IV, alpha=0.10, **KW)
    line = next(ln for ln in fit.summary().splitlines() if ln.strip().startswith("x1"))
    lo, hi = (float(v) for v in line.split()[-2:])
    b, se = fit.linear_params["x1"], fit.se_linear["x1"]
    z = 1.6448536269514722  # norm.ppf(0.95)
    assert lo == pytest.approx(round(b - z * se, 4), abs=1e-4)
    assert hi == pytest.approx(round(b + z * se, 4), abs=1e-4)
    assert "[0.05" in fit.summary() and "0.95]" in fit.summary()


def test_row_level_outputs_are_indexed_like_the_data(df):
    """Shuffled rows with non-default labels: ``markups()``, ``mean_utility``
    and ``own_elasticities`` line up with the data by label."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.blp(df, x_random=[], instruments=IV, firm_id="firm", **KW)
        shuffled = df.sample(frac=1.0, random_state=4)
        shuffled.index = shuffled.index + 500
        fit = sp.blp(shuffled, x_random=[], instruments=IV, firm_id="firm", **KW)
    for name in ("mean_utility", "own_elasticities"):
        s = getattr(fit, name)
        assert s.index.equals(shuffled.index)
        np.testing.assert_allclose(
            s.loc[df.index + 500].to_numpy(),
            getattr(base, name).to_numpy(),
            rtol=1e-10,
        )
    mk = fit.markups()
    assert mk.index.equals(shuffled.index)
    np.testing.assert_allclose(
        mk.loc[df.index + 500].to_numpy(), base.markups().to_numpy(), rtol=1e-10
    )
    mc = fit.marginal_costs()
    np.testing.assert_allclose(
        mc.to_numpy(), shuffled["price"].to_numpy() - mk.to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize("column", ["share", "price", "x1", "z2", "market_id"])
def test_missing_values_are_refused(df, column):
    from statspai.exceptions import DataInsufficient

    bad = df.copy()
    bad[column] = bad[column].astype(float)
    bad.loc[5, column] = np.nan
    with pytest.raises(DataInsufficient, match=column):
        sp.blp(bad, x_random=[], instruments=IV, **KW)
