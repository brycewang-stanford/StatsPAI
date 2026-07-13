"""Analytical parity: structural production-function family + BLP demand.

Proxy-variable production estimators (Olley-Pakes 1996, Levinsohn-Petrin
2003, Ackerberg-Caves-Frazer 2015, Wooldridge 2009) identify Cobb-Douglas
output elasticities from the control-function inversion

    y_it = beta_l*l_it + beta_k*k_it + omega_it + eta_it,
    omega_it = rho*omega_{i,t-1} + xi_it,   proxy = h(omega, k) monotone.

We simulate two panels that *satisfy* the timing assumptions with known
(beta_l, beta_k, rho):

* ``exog_panel`` — labor drawn independently of the productivity
  innovation (identified for OP / LP / Wooldridge; avoids the ACF
  functional-dependence critique);
* ``endog_panel`` — labor responds to contemporaneous omega, the case
  ACF's lagged-labor moments are built for.

Each estimator must recover the true elasticities within a tolerance
justified by the two-step semi-parametric bias (see per-test comments).
De Loecker-Warzynski markups are checked on a DGP where the true markup
is known by construction (input expenditure set to a constant fraction
``beta_l / mu`` of eta-corrected revenue, so mu_it = mu exactly at the
true elasticity), plus the exact internal identity
mu = theta_v / cost_share. BLP is checked on a plain-logit demand DGP
with endogenous prices and valid cost instruments, where the linear
price / characteristic coefficients have a closed-form 2SLS benchmark.
Analytical evidence tier — deterministic seeds throughout, no external
references.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

# True structural parameters shared by both production panels.
BETA_L, BETA_K = 0.60, 0.30
RHO, SIGMA_XI, SIGMA_ETA = 0.7, 0.20, 0.10
MU_TRUE = 1.25  # true De Loecker-Warzynski markup in the markup DGP


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _simulate_production_panel(
    seed: int,
    n_firms: int = 300,
    n_periods: int = 10,
    endog_labor: bool = False,
) -> pd.DataFrame:
    """Cobb-Douglas panel satisfying the proxy-variable timing assumptions.

    omega is stationary AR(1); capital is predetermined (evolves from
    last period's investment); materials m = h(omega, k) is strictly
    monotone in omega given k (invertible proxy); investment i > 0 in
    levels (OP proxy). With ``endog_labor`` labor responds to
    contemporaneous omega (the ACF case); otherwise labor is exogenous
    so OP / LP / Wooldridge are point-identified.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_firms):
        omega = rng.normal(0.0, SIGMA_XI / np.sqrt(1 - RHO**2))
        k = rng.normal(0.0, 0.5)
        for t in range(n_periods):
            omega = RHO * omega + rng.normal(0.0, SIGMA_XI)
            if endog_labor:
                ell = 0.5 * omega + 0.3 * k + rng.normal(0.0, 0.10)
            else:
                ell = rng.normal(0.5, 0.4)
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)
            inv = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))
            eta = rng.normal(0.0, SIGMA_ETA)
            y = BETA_L * ell + BETA_K * k + omega + eta
            rows.append(
                {
                    "id": fid,
                    "year": t,
                    "y": y,
                    "l": ell,
                    "k": k,
                    "m": m,
                    "i": inv,
                    "eta_true": eta,
                }
            )
            k = 0.9 * k + 0.1 * np.log(inv + 1e-6)
    df = pd.DataFrame(rows)
    # Markup construction (labor = flexible input, output price P = 1):
    # log wage bill = (y - eta) + log(beta_l / mu)  =>  the eta-corrected
    # labor cost share is exactly beta_l / mu, so the DLW markup
    # theta_l / share equals MU_TRUE when theta_l = beta_l.
    df["log_rev"] = df["y"]
    df["log_wage_bill"] = (df["y"] - df["eta_true"]) + np.log(BETA_L / MU_TRUE)
    return df


def _simulate_logit_demand(
    seed: int = 1,
    n_products: int = 5,
    n_markets: int = 50,
    alpha: float = -1.5,
    beta_x: float = 1.0,
) -> pd.DataFrame:
    """Plain multinomial-logit demand with an outside good.

    Price is endogenous (loads on the demand shock xi) but driven by two
    exogenous cost shifters that serve as excluded instruments, so the
    linear parameters (alpha, beta_x) are point-identified by IV-GMM and
    have a closed-form 2SLS benchmark after the exact logit inversion
    delta = log(s_j) - log(s_0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mkt in range(n_markets):
        x = rng.uniform(0, 2, n_products)
        xi = rng.normal(0, 0.2, n_products)  # demand shock -> endogeneity
        cost = rng.uniform(0.5, 2, n_products)
        z2 = rng.uniform(0, 1, n_products)
        price = 0.8 * cost + 0.5 * z2 + 0.4 * xi + rng.uniform(0, 0.5, n_products)
        delta = beta_x * x + alpha * price + xi
        ev = np.exp(delta)
        shares = ev / (1 + ev.sum())
        for j in range(n_products):
            rows.append(
                {
                    "market_id": mkt,
                    "product_id": j,
                    "shares": shares[j],
                    "prices": price[j],
                    "x1": x[j],
                    "cost": cost[j],
                    "z2": z2[j],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: each estimator fits once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exog_panel():
    return _simulate_production_panel(seed=0, endog_labor=False)


@pytest.fixture(scope="module")
def endog_panel():
    return _simulate_production_panel(seed=0, endog_labor=True)


@pytest.fixture(scope="module")
def op_fit(exog_panel):
    return sp.olley_pakes(
        exog_panel,
        output="y",
        free="l",
        state="k",
        proxy="i",
        panel_id="id",
        time="year",
    )


@pytest.fixture(scope="module")
def lp_fit(exog_panel):
    return sp.levinsohn_petrin(
        exog_panel,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
    )


@pytest.fixture(scope="module")
def acf_fit(endog_panel):
    return sp.ackerberg_caves_frazer(
        endog_panel,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
    )


@pytest.fixture(scope="module")
def wrdg_fit(exog_panel):
    return sp.wooldridge_prod(
        exog_panel,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )


@pytest.fixture(scope="module")
def blp_fit():
    df = _simulate_logit_demand()
    res = sp.blp(
        df,
        shares="shares",
        prices="prices",
        x_linear=["x1"],
        x_random=["x1"],
        instruments=["cost", "z2"],
        n_draws=30,
        maxiter=100,
        seed=0,
    )
    return df, res


# ---------------------------------------------------------------------------
# Elasticity recovery
# ---------------------------------------------------------------------------
# Tolerance rationale (shared): these are semi-parametric two-step
# estimators, so the leading error is the stage-1 polynomial
# approximation + stage-2 GMM finite-sample bias, not the O(1/sqrt(NT))
# sampling noise (~0.004 here with NT = 3000, sigma_eta = 0.10). On this
# seed the observed absolute errors are <= 0.03 for beta_l and <= 0.05
# for beta_k across OP/LP; we assert abs 0.08 (beta_l) / 0.10 (beta_k)
# to leave ~2-3x headroom while still rejecting any O(0.1)+ correctness
# regression (e.g. an OLS-basin solution has beta_l biased up by ~+0.3).


def test_olley_pakes_recovers_elasticities(op_fit):
    assert op_fit.coef["l"] == pytest.approx(BETA_L, abs=0.08)
    assert op_fit.coef["k"] == pytest.approx(BETA_K, abs=0.10)


def test_levinsohn_petrin_recovers_elasticities(lp_fit):
    assert lp_fit.coef["l"] == pytest.approx(BETA_L, abs=0.08)
    assert lp_fit.coef["k"] == pytest.approx(BETA_K, abs=0.10)


def test_acf_recovers_elasticities_with_endogenous_labor(acf_fit):
    # The ACF DGP: labor responds to contemporaneous omega, so OLS (and
    # the OP/LP moments) are inconsistent for beta_l here; the ACF
    # lagged-labor moment restores identification. Finite-sample GMM
    # bias is larger than in the exogenous-labor design (observed
    # errors across seeds 0-7 are <= 0.07), hence abs 0.10 on both.
    assert acf_fit.coef["l"] == pytest.approx(BETA_L, abs=0.10)
    assert acf_fit.coef["k"] == pytest.approx(BETA_K, abs=0.10)


def test_wooldridge_recovers_elasticities(wrdg_fit):
    # Wooldridge's stacked one-step NLS on the exogenous-labor panel:
    # observed errors ~0.01 (beta_l) / ~0.06 (beta_k); same 0.08 / 0.10
    # budget as the two-step estimators.
    assert wrdg_fit.coef["l"] == pytest.approx(BETA_L, abs=0.08)
    assert wrdg_fit.coef["k"] == pytest.approx(BETA_K, abs=0.10)


def test_prod_fn_dispatcher_is_identical_to_direct_call(exog_panel, lp_fit):
    # Internal identity: the dispatcher must route to the very same
    # estimator, so coefficients agree to machine precision.
    res = sp.prod_fn(
        exog_panel,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        method="lp",
    )
    assert res.coef["l"] == pytest.approx(lp_fit.coef["l"], abs=1e-12)
    assert res.coef["k"] == pytest.approx(lp_fit.coef["k"], abs=1e-12)
    assert res.method == "lp"


# ---------------------------------------------------------------------------
# Productivity process + internal sanity
# ---------------------------------------------------------------------------


def test_recovered_productivity_persistence(lp_fit, wrdg_fit):
    # omega_hat = phi_hat - X @ beta_hat; its fitted AR(1) coefficient
    # should recover rho = 0.7. Error inherits the elasticity error via
    # omega_hat contamination -> abs 0.10 (observed: LP 0.668,
    # Wooldridge 0.703).
    assert lp_fit.productivity_process["rho"] == pytest.approx(RHO, abs=0.10)
    assert wrdg_fit.productivity_process["rho"] == pytest.approx(RHO, abs=0.10)


def test_returns_to_scale_identity(lp_fit, op_fit):
    # Elasticities must sum near the true returns to scale
    # beta_l + beta_k = 0.90; budget = sum of the individual coefficient
    # tolerances is 0.18, we use 0.15 (observed deviation ~0.08).
    rts = BETA_L + BETA_K
    assert lp_fit.coef["l"] + lp_fit.coef["k"] == pytest.approx(rts, abs=0.15)
    assert op_fit.coef["l"] + op_fit.coef["k"] == pytest.approx(rts, abs=0.15)


def test_tfp_series_is_finite_and_aligned(lp_fit, acf_fit, wrdg_fit):
    for res in (lp_fit, acf_fit, wrdg_fit):
        tfp = np.asarray(res.tfp, dtype=float)
        assert np.all(np.isfinite(tfp))
        assert len(tfp) == len(res.sample)
        # Stage-1 control function must explain most output variance
        # (omega + inputs dominate the eta noise in this DGP).
        r2_key = "stage1_r2" if "stage1_r2" in res.diagnostics else None
        if r2_key:
            assert res.diagnostics[r2_key] > 0.5


# ---------------------------------------------------------------------------
# De Loecker-Warzynski markup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def markup_series(exog_panel, lp_fit):
    # markup() reads revenue / input-cost columns off result.sample;
    # attach them by (id, year) merge so alignment is explicit.
    merged = lp_fit.sample.merge(
        exog_panel[["id", "year", "log_rev", "log_wage_bill"]],
        on=["id", "year"],
        how="left",
        suffixes=("", "_x"),
    )
    lp_fit.sample["log_rev"] = merged["log_rev"].to_numpy()
    lp_fit.sample["log_wage_bill"] = merged["log_wage_bill"].to_numpy()
    mu = sp.markup(
        lp_fit,
        revenue="log_rev",
        input_cost="log_wage_bill",
        flexible_input="l",
    )
    return mu


def test_markup_recovers_true_markup(markup_series):
    # By construction the eta-corrected labor cost share is exactly
    # beta_l / MU_TRUE, so mu_hat = MU_TRUE * (theta_l_hat / beta_l)
    # up to the (estimated - true) eta correction, which is mean-zero.
    # The elasticity budget |theta_l - beta_l| <= 0.08 implies
    # |median(mu) - MU_TRUE| <= 1.25 * 0.08 / 0.60 ~= 0.17; we assert
    # 0.15 (observed median 1.298, deviation 0.05).
    mu = markup_series
    assert isinstance(mu, pd.Series)
    assert (mu > 0).all()
    assert float(mu.median()) == pytest.approx(MU_TRUE, abs=0.15)


def test_markup_internal_identity(markup_series, lp_fit):
    # Exact identity (machine precision): mu_it = theta_l / cost_share_it
    # with cost_share_it = exp(log_cost - (log_rev - eta_hat)).
    theta_l = float(lp_fit.coef["l"])
    cost_share = np.exp(
        lp_fit.sample["log_wage_bill"].to_numpy(dtype=float)
        - (
            lp_fit.sample["log_rev"].to_numpy(dtype=float)
            - lp_fit.sample["eta"].to_numpy(dtype=float)
        )
    )
    np.testing.assert_allclose(
        markup_series.to_numpy(dtype=float), theta_l / cost_share, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# BLP demand
# ---------------------------------------------------------------------------


def test_blp_recovers_linear_demand_parameters(blp_fit):
    # Pure-logit DGP: the linear (alpha, beta_x) are IV-identified; the
    # spurious random-coefficient sigma estimated on 30 Halton draws
    # perturbs them slightly relative to the exact 2SLS solution, and
    # the IV sampling noise with 250 obs / xi sd 0.2 is O(0.05). abs
    # 0.15 covers both (observed: alpha -1.539, beta_x 1.043).
    _, res = blp_fit
    assert res.linear_params["prices"] == pytest.approx(-1.5, abs=0.15)
    assert res.linear_params["x1"] == pytest.approx(1.0, abs=0.15)


def test_blp_matches_closed_form_logit_2sls(blp_fit):
    # Closed-form benchmark: exact logit inversion
    # delta_j = log(s_j) - log(s_0), then 2SLS of delta on [1, x, p]
    # with Z = [1, x, cost, z2]. sp.blp's linear step is this same GMM
    # projection evaluated at sigma_hat instead of sigma = 0, so its
    # linear parameters must sit within the sigma-induced perturbation
    # (abs 0.10 observed ~0.05) of the closed form.
    df, res = blp_fit
    s0 = 1.0 - df.groupby("market_id")["shares"].transform("sum")
    delta = np.log(df["shares"].to_numpy()) - np.log(s0.to_numpy())
    X = np.column_stack([np.ones(len(df)), df["x1"], df["prices"]])
    Z = np.column_stack([np.ones(len(df)), df["x1"], df["cost"], df["z2"]])
    W = np.linalg.inv(Z.T @ Z)
    b = np.linalg.solve(X.T @ Z @ W @ Z.T @ X, X.T @ Z @ W @ Z.T @ delta)
    assert res.linear_params["x1"] == pytest.approx(b[1], abs=0.10)
    assert res.linear_params["prices"] == pytest.approx(b[2], abs=0.10)


def test_blp_own_price_elasticities_negative_and_ses_finite(blp_fit):
    # Demand slopes down: every own-price elasticity is negative under
    # alpha < 0; GMM standard errors must be finite and positive.
    _, res = blp_fit
    own = np.asarray(res.own_elasticities, dtype=float)
    assert np.all(np.isfinite(own))
    assert np.all(own < 0)
    se = np.asarray(res.se_linear, dtype=float)
    assert np.all(np.isfinite(se))
    assert np.all(se > 0)
