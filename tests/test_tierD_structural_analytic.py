"""Tier D analytic special-case tests — proxy-variable production functions.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). The short aliases ``levpet`` / ``opreg`` were
graded ``untested`` by ``scripts/tierd_classify.py``. On a data-generating
process that *satisfies* the proxy-variable timing assumptions — labour with an
independent exogenous source of variation (so it is identified separately from
productivity), and a proxy monotone in (productivity, capital) — the
Levinsohn-Petrin and Olley-Pakes estimators recover the known Cobb-Douglas
output elasticities. Each alias must also dispatch identically to its underlying
estimator.

``sp.blp`` (the third structural P1 estimator) is now covered too: the Tier D
probe first surfaced a real bug (``_gmm_objective`` called with ``maxiter=``
instead of ``maxiter_inner=``, raising ``TypeError`` on every estimation path),
which was reported (``.tierd_campaign/BUG_blp_gmm_objective_maxiter.md``) and
then fixed by the maintainer per CLAUDE.md §12 (CHANGELOG + MIGRATION,
``⚠️ Functionality fix``). ``TestBLPAnalytic`` below is the regression guard.

Entry points covered:
    sp.levpet  -> sp.levinsohn_petrin (Levinsohn-Petrin 2003)
    sp.opreg   -> sp.olley_pakes      (Olley-Pakes 1996)
    sp.blp     -> random-coefficients logit demand (Berry-Levinsohn-Pakes 1995)

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

BETA_L, BETA_K = 0.60, 0.35


def _identified_panel(seed=0, n_firms=300, n_periods=15, rho=0.7):
    """Cobb-Douglas panel satisfying the LP/OP assumptions.

    Crucially labour ``l`` is drawn *exogenously* (independent of the
    productivity innovation), so it is identified separately from omega — this
    avoids the Ackerberg-Caves-Frazer collinearity that biases the estimators
    when labour responds to contemporaneous productivity.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_firms):
        omega = rng.normal(0.0, 0.2 / np.sqrt(1 - rho**2))
        k = rng.normal(0.0, 0.5)
        for t in range(n_periods):
            omega = rho * omega + rng.normal(0.0, 0.2)
            ell = rng.normal(0.5, 0.4)  # exogenous labour
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)  # materials proxy
            i = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))
            y = BETA_L * ell + BETA_K * k + omega + rng.normal(0.0, 0.10)
            rows.append(
                {"id": fid, "year": t, "y": y, "l": ell, "k": k, "m": m, "i": i}
            )
            k = 0.9 * k + 0.1 * np.log(i + 1e-6)
    return pd.DataFrame(rows)


class TestLevinsohnPetrinAnalytic:

    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.08)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_alias_equals_levinsohn_petrin(self):
        df = _identified_panel()
        a = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        b = sp.levinsohn_petrin(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert a.coef["l"] == b.coef["l"] and a.coef["k"] == b.coef["k"]

    def test_first_stage_fits(self):
        df = _identified_panel()
        res = sp.levpet(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
        assert res.diagnostics["stage1_r2"] > 0.5


class TestOlleyPakesAnalytic:

    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.opreg(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.08)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_alias_equals_olley_pakes(self):
        df = _identified_panel()
        a = sp.opreg(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        b = sp.olley_pakes(
            df, output="y", free="l", state="k", proxy="i", panel_id="id", time="year"
        )
        assert a.coef["l"] == b.coef["l"] and a.coef["k"] == b.coef["k"]


def _blp_logit_panel(seed=1, n_products=6, n_markets=120, alpha=-1.5, beta=1.0):
    """Well-conditioned BLP panel: a multinomial-logit DGP with an endogenous
    price driven by two cost shifters (the excluded instruments). With enough
    markets and instruments, ``Z'Z`` stays well-conditioned, so the GMM
    weight matrix is non-singular and the estimator recovers the true linear
    price/characteristic coefficients."""
    rng = np.random.default_rng(seed)
    rows = []
    for mkt in range(n_markets):
        x = rng.uniform(0, 2, n_products)
        xi = rng.normal(0, 0.2, n_products)  # demand shock (price endogeneity)
        cost = rng.uniform(0.5, 2, n_products)
        z2 = rng.uniform(0, 1, n_products)
        price = 0.8 * cost + 0.5 * z2 + 0.4 * xi + rng.uniform(0, 0.5, n_products)
        delta = beta * x + alpha * price + xi
        ev = np.exp(delta)
        shares = ev / (1 + ev.sum())  # logit shares with an outside good
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


class TestBLPAnalytic:
    """BLP random-coefficients logit. On a pure-logit DGP with endogenous
    price and valid cost instruments, the estimator recovers the known linear
    price and characteristic coefficients.

    Regression guard for the maxiter keyword fix (CHANGELOG Unreleased /
    v1.17.0, ``⚠️ Functionality fix``): before the fix ``sp.blp`` raised
    ``TypeError: _gmm_objective() got an unexpected keyword argument
    'maxiter'`` on every estimation path and produced no output at all.
    """

    def test_recovers_linear_price_and_characteristic(self):
        df = _blp_logit_panel()
        res = sp.blp(
            df,
            shares="shares",
            prices="prices",
            x_linear=["x1"],
            x_random=["x1"],
            instruments=["cost", "z2"],
            n_draws=80,
            seed=0,
        )
        assert res.linear_params["prices"] == pytest.approx(-1.5, abs=0.15)
        assert res.linear_params["x1"] == pytest.approx(1.0, abs=0.15)

    def test_runs_without_maxiter_typeerror(self):
        # Direct guard: the buggy call raised TypeError before reaching output.
        df = _blp_logit_panel(seed=2, n_markets=80)
        res = sp.blp(
            df,
            shares="shares",
            prices="prices",
            x_linear=["x1"],
            instruments=["cost", "z2"],
            n_draws=50,
            seed=0,
        )
        # Price enters utility negatively; own-price elasticities are negative.
        assert res.own_elasticities.mean() < 0
