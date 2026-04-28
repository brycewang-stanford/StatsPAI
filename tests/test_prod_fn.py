"""
Tests for proxy-variable production function estimators.

Strategy
--------
We build a synthetic DGP that satisfies the timing assumptions of
OP / LP / ACF / Wooldridge:

* output ``y = beta_l*l + beta_k*k + omega + eta``
* AR(1) productivity ``omega_t = rho*omega_{t-1} + xi_t``
* capital predetermined; investment & materials respond to (omega, k)
* labor free given omega (responds to omega but not to xi)

We then check that each estimator recovers the true (beta_l, beta_k)
within a tolerance commensurate with sample size and the chosen
polynomial basis.  We also exercise the dispatcher, markup helper, and
basic edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Synthetic DGP
# ---------------------------------------------------------------------------

def _simulate_panel(
    n_firms: int = 200,
    n_periods: int = 12,
    beta_l: float = 0.60,
    beta_k: float = 0.35,
    rho: float = 0.7,
    sigma_xi: float = 0.20,
    sigma_eta: float = 0.10,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []

    # Initial state per firm
    for fid in range(n_firms):
        omega = rng.normal(0.0, sigma_xi / np.sqrt(1 - rho ** 2))
        k = rng.normal(0.0, 0.5)  # log capital initial level
        for t in range(n_periods):
            xi = rng.normal(0.0, sigma_xi)
            omega = rho * omega + xi

            # Free input: labor responds to (omega, k); free => correlated
            # with omega but not xi (we use omega_{t-1} via persistence, so
            # current l_t correlates with omega_t through omega_{t-1} too).
            l = 0.5 * omega + 0.3 * k + rng.normal(0.0, 0.10)

            # Materials proxy m = h(omega, k)
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)

            # Investment proxy (strictly positive in levels)
            i_lvl = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))

            eta = rng.normal(0.0, sigma_eta)
            y = beta_l * l + beta_k * k + omega + eta

            rows.append({
                "id": fid,
                "year": t,
                "y": y,
                "l": l,
                "k": k,
                "m": m,
                "i": i_lvl,
                "omega": omega,
                "eta": eta,
            })

            # Capital evolves with investment (predetermined for next period)
            k = 0.9 * k + 0.1 * np.log(i_lvl + 1e-6)

    df = pd.DataFrame(rows)
    truth = {
        "beta_l": beta_l,
        "beta_k": beta_k,
        "rho": rho,
        "sigma_xi": sigma_xi,
        "sigma_eta": sigma_eta,
    }
    return df, truth


@pytest.fixture(scope="module")
def panel():
    df, truth = _simulate_panel()
    return df, truth


# ---------------------------------------------------------------------------
# Recovery tests (large tolerance — proxy-variable estimators are noisy)
# ---------------------------------------------------------------------------

def _close(coef: float, truth: float, tol: float) -> bool:
    return abs(coef - truth) < tol


def test_olley_pakes_runs(panel):
    """OP is known to be biased per ACF (2015) when labor responds to
    contemporaneous productivity — which our DGP enforces. We only test
    that OP runs end-to-end and produces a sensible TFP series.
    """
    df, truth = panel
    res = sp.olley_pakes(
        df, output="y", free="l", state="k", proxy="i",
        panel_id="id", time="year",
        polynomial_degree=3, productivity_degree=1,
    )
    assert "l" in res.coef and "k" in res.coef
    assert res.diagnostics["stage1_r2"] > 0.5
    assert res.diagnostics["stage2_converged"]
    assert len(res.tfp) == len(res.sample)


def test_levinsohn_petrin_runs(panel):
    """Same caveat as OP — LP β_l identification fails in the ACF DGP.
    We test that the estimator runs and produces output."""
    df, _ = panel
    res = sp.levinsohn_petrin(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=3, productivity_degree=1,
    )
    assert "l" in res.coef and "k" in res.coef
    assert res.diagnostics["stage1_r2"] > 0.5


def test_acf_recovers_params(panel):
    """ACF (2015) is the rigorous identification — under our DGP it
    should recover (beta_l, beta_k) close to truth."""
    df, truth = panel
    res = sp.ackerberg_caves_frazer(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=3, productivity_degree=1,
    )
    assert _close(res.coef["l"], truth["beta_l"], tol=0.10)
    assert _close(res.coef["k"], truth["beta_k"], tol=0.10)
    assert res.method == "acf"


def test_wooldridge_runs(panel):
    """Wooldridge (2009) one-step GMM is high-dimensional NLS; we test
    that it converges to a sensible neighborhood. The objective is
    nonconvex with many polynomial coefficients, so we check the
    elasticities are positive and bounded rather than tightly close
    to truth."""
    df, _ = panel
    res = sp.wooldridge_prod(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=2, productivity_degree=1,
    )
    assert res.coef["l"] > 0 and res.coef["l"] < 1.5
    assert res.coef["k"] > -0.5 and res.coef["k"] < 1.0
    assert res.method == "wrdg"


# ---------------------------------------------------------------------------
# Dispatcher + aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["op", "lp", "acf", "wrdg"])
def test_prod_fn_dispatcher(panel, method):
    df, _ = panel
    proxy = "i" if method == "op" else "m"
    res = sp.prod_fn(
        df, output="y", free="l", state="k", proxy=proxy,
        panel_id="id", time="year",
        method=method,
        polynomial_degree=2, productivity_degree=1,
    )
    assert hasattr(res, "coef")
    assert hasattr(res, "tfp")
    assert len(res.tfp) > 0


def test_aliases_match_canonical(panel):
    df, _ = panel
    res1 = sp.acf(df, output="y", free="l", state="k", proxy="m",
                  panel_id="id", time="year",
                  polynomial_degree=2, productivity_degree=1)
    res2 = sp.ackerberg_caves_frazer(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=2, productivity_degree=1,
    )
    assert np.isclose(res1.coef["l"], res2.coef["l"])
    assert np.isclose(res1.coef["k"], res2.coef["k"])


# ---------------------------------------------------------------------------
# Bootstrap SE
# ---------------------------------------------------------------------------

def test_bootstrap_produces_se(panel):
    df, _ = panel
    res = sp.acf(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=2, productivity_degree=1,
        boot_reps=20, seed=0,
    )
    assert np.isfinite(res.std_errors["l"]).all()
    assert np.isfinite(res.std_errors["k"]).all()
    assert res.std_errors["l"] > 0
    assert res.diagnostics["boot_reps_effective"] > 0


# ---------------------------------------------------------------------------
# Markup
# ---------------------------------------------------------------------------

def test_markup_runs(panel):
    df, _ = panel
    # Construct log revenue and log materials cost as if firms face common
    # input price and constant markup. Noise added so cost share varies.
    rng = np.random.default_rng(42)
    df = df.copy()
    df["log_rev"] = df["y"] + np.log(1.20) + rng.normal(0, 0.05, len(df))
    df["log_mat_cost"] = df["m"] + np.log(0.80) + rng.normal(0, 0.05, len(df))

    res = sp.acf(
        df, output="y", free="l", state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=2, productivity_degree=1,
    )

    # markup() expects revenue/cost columns inside result.sample.
    res.sample["log_rev"] = df.loc[res.sample.index, "log_rev"].to_numpy() if "log_rev" in df else df["log_rev"].to_numpy()[: len(res.sample)]
    res.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res.sample)]

    # The flexible input here must be one of the production-function
    # coefficients. ACF coefs are {l, k}, so we re-fit with m as a free
    # input to expose theta_m.
    res2 = sp.acf(
        df, output="y", free=["l", "m"], state="k", proxy="m",
        panel_id="id", time="year",
        polynomial_degree=2, productivity_degree=1,
    )
    res2.sample["log_rev"] = df["log_rev"].to_numpy()[: len(res2.sample)]
    res2.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res2.sample)]

    mu = sp.markup(res2, revenue="log_rev", input_cost="log_mat_cost",
                   flexible_input="m")
    assert isinstance(mu, pd.Series)
    assert (mu > 0).all()
    # Median markup positive and finite
    assert np.isfinite(mu.median())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_missing_column_raises(panel):
    df, _ = panel
    with pytest.raises(ValueError, match="Missing columns"):
        sp.olley_pakes(df.drop(columns=["i"]), output="y", free="l",
                       state="k", proxy="i", panel_id="id", time="year")


def test_too_few_obs_raises():
    df = pd.DataFrame({
        "y": [1.0, 1.1], "l": [0.5, 0.6], "k": [0.4, 0.4],
        "m": [0.3, 0.3], "id": [0, 0], "year": [0, 1],
    })
    with pytest.raises(ValueError, match="valid observations"):
        sp.acf(df, output="y", free="l", state="k", proxy="m",
               panel_id="id", time="year")


def test_op_drops_zero_investment(panel):
    df, _ = panel
    df = df.copy()
    df.loc[df.index[:5], "i"] = 0.0  # five rows with zero investment
    n_before = len(df)
    res = sp.olley_pakes(df, output="y", free="l", state="k", proxy="i",
                         panel_id="id", time="year",
                         polynomial_degree=2, productivity_degree=1)
    # Should drop the zero-i rows internally
    assert len(res.sample) < n_before


# ---------------------------------------------------------------------------
# Result object surface
# ---------------------------------------------------------------------------

def test_production_result_has_summary(panel):
    df, _ = panel
    res = sp.acf(df, output="y", free="l", state="k", proxy="m",
                 panel_id="id", time="year",
                 polynomial_degree=2, productivity_degree=1)
    s = res.summary()
    assert "Production function" in s or "Model" in s or hasattr(res, "params")
    assert isinstance(res.cite(), str)
    assert "tfp" in dir(res)
    assert len(res.tfp) == len(res.sample)


def test_in_function_registry():
    """prod_fn / acf / markup / olley_pakes / levinsohn_petrin should
    be discoverable via sp.list_functions() once registered."""
    fns = set(sp.list_functions())
    assert "prod_fn" in fns
    assert "olley_pakes" in fns
    assert "ackerberg_caves_frazer" in fns
    # Aliases for Stata/R users
    assert "acf" in fns
    assert "opreg" in fns
    assert "levpet" in fns
    assert "markup" in fns


def test_diagnostics_without_bootstrap(panel):
    """Ensure result.diagnostics is well-formed when no bootstrap is run
    (boot_reps_effective == 0, no NameError on missing bootstrap state)."""
    df, _ = panel
    res = sp.acf(df, output="y", free="l", state="k", proxy="m",
                 panel_id="id", time="year",
                 polynomial_degree=2, productivity_degree=1)
    assert res.diagnostics["boot_reps_effective"] == 0
    assert np.isnan(res.std_errors["l"])
    assert np.isnan(res.std_errors["k"])
    assert res.cov is None


# ---------------------------------------------------------------------------
# Translog functional form
# ---------------------------------------------------------------------------

def test_translog_runs_and_exposes_quadratic_terms(panel):
    """Translog fit on a Cobb-Douglas DGP should:
      (i) converge with all 5 expected coefficient keys,
      (ii) produce a firm-time elasticity panel,
      (iii) recover near-zero quadratic / cross terms (since the truth
            is CD), and
      (iv) keep linear elasticities in a CD-plausible neighborhood.
    """
    df, _ = panel
    res = sp.acf(df, output="y", free="l", state="k", proxy="m",
                 panel_id="id", time="year",
                 polynomial_degree=3, productivity_degree=1,
                 functional_form="translog")
    expected_keys = {"l", "k", "ll", "kk", "lk"}
    assert expected_keys <= set(res.coef)
    elasts = res.model_info["elasticities"]
    assert isinstance(elasts, pd.DataFrame)
    assert list(elasts.columns) == ["l", "k"]
    assert len(elasts) == len(res.sample)
    # Linear coefs roughly in CD truth neighborhood (loose — translog ACF
    # has higher SE than the CD ACF baseline).
    assert -0.5 < res.coef["l"] < 1.5
    assert -0.5 < res.coef["k"] < 1.5
    # Under CD truth, the quadratic / cross terms should shrink toward
    # zero in expectation.  Tolerance is wide (|β| < 0.6) because
    # finite-sample variance on translog higher-order coefficients is
    # genuinely large — instruments are polynomial transforms of the
    # same raw (k, l_lag) pair, so the moment system is near-singular
    # in finite samples (cf. dispatcher docstring caveat).
    assert abs(res.coef["ll"]) < 0.60
    assert abs(res.coef["kk"]) < 0.60
    assert abs(res.coef["lk"]) < 0.60


def test_translog_dispatcher(panel):
    """Dispatcher should pass functional_form through to OP/LP/ACF."""
    df, _ = panel
    res = sp.prod_fn(df, output="y", free="l", state="k", proxy="m",
                     panel_id="id", time="year",
                     method="lp",
                     polynomial_degree=2, productivity_degree=1,
                     functional_form="translog")
    assert "ll" in res.coef and "kk" in res.coef and "lk" in res.coef


def test_translog_markup_uses_firm_time_elasticities(panel):
    """sp.markup with translog should pick up firm-time θ_v_it from the
    elasticity panel rather than a constant."""
    df, _ = panel
    rng = np.random.default_rng(0)
    df = df.copy()
    df["log_rev"] = df["y"] + np.log(1.20) + rng.normal(0, 0.05, len(df))
    df["log_mat_cost"] = df["m"] + np.log(0.80) + rng.normal(0, 0.05, len(df))

    res = sp.acf(df, output="y", free=["l", "m"], state="k", proxy="m",
                 panel_id="id", time="year",
                 polynomial_degree=2, productivity_degree=1,
                 functional_form="translog")
    res.sample["log_rev"] = df["log_rev"].to_numpy()[: len(res.sample)]
    res.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res.sample)]

    mu = sp.markup(res, revenue="log_rev", input_cost="log_mat_cost",
                   flexible_input="m")
    assert isinstance(mu, pd.Series)
    assert (mu > 0).all()
    # Translog markup should vary across firm-times (not constant).
    assert mu.std() > 0


def test_wooldridge_translog_raises():
    """Wooldridge currently only supports Cobb-Douglas — translog must
    raise NotImplementedError, not silently return wrong numbers."""
    df, _ = _simulate_panel(n_firms=80, n_periods=8)
    with pytest.raises(NotImplementedError, match="cobb-douglas"):
        sp.wooldridge_prod(df, output="y", free="l", state="k", proxy="m",
                           panel_id="id", time="year",
                           functional_form="translog")


def test_unknown_functional_form_raises(panel):
    """Anything outside {cobb-douglas, translog} must raise immediately."""
    df, _ = panel
    with pytest.raises(ValueError, match="Unknown functional_form"):
        sp.acf(df, output="y", free="l", state="k", proxy="m",
               panel_id="id", time="year",
               functional_form="ces")


def test_time_gap_warning():
    """Firms with non-consecutive years trigger a UserWarning."""
    rng = np.random.default_rng(0)
    rows = []
    # 50 firms, but each skips year 5 (creating a 2-year gap)
    for fid in range(50):
        for t in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:  # gap at t=5
            rows.append({
                "id": fid, "year": t,
                "y": rng.normal(0, 1),
                "l": rng.normal(0, 1),
                "k": rng.normal(0, 1),
                "m": rng.normal(0, 1),
            })
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="non-consecutive"):
        sp.acf(df, output="y", free="l", state="k", proxy="m",
               panel_id="id", time="year",
               polynomial_degree=2, productivity_degree=1)
