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

import warnings

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
    persistent_wage: bool = False,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []

    # Initial state per firm
    for fid in range(n_firms):
        omega = rng.normal(0.0, sigma_xi / np.sqrt(1 - rho**2))
        k = rng.normal(0.0, 0.5)  # log capital initial level
        wage = 0.0
        for t in range(n_periods):
            xi = rng.normal(0.0, sigma_xi)
            omega = rho * omega + xi

            # Free input: labor responds to (omega, k); free => correlated
            # with omega but not xi (we use omega_{t-1} via persistence, so
            # current l_t correlates with omega_t through omega_{t-1} too).
            l = 0.5 * omega + 0.3 * k + rng.normal(0.0, 0.10)
            if persistent_wage:
                # A persistent wage shock makes lagged labor a relevant
                # instrument once (k, m) have absorbed omega (Wooldridge).
                wage = 0.8 * wage + rng.normal(0.0, 0.3)
                l -= 0.6 * wage  # noqa: E741

            # Materials proxy m = h(omega, k)
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)

            # Investment proxy (strictly positive in levels)
            i_lvl = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))

            eta = rng.normal(0.0, sigma_eta)
            y = beta_l * l + beta_k * k + omega + eta

            rows.append(
                {
                    "id": fid,
                    "year": t,
                    "y": y,
                    "l": l,
                    "k": k,
                    "m": m,
                    "i": i_lvl,
                    "omega": omega,
                    "eta": eta,
                }
            )

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
        df,
        output="y",
        free="l",
        state="k",
        proxy="i",
        panel_id="id",
        time="year",
        polynomial_degree=3,
        productivity_degree=1,
    )
    assert "l" in res.coef and "k" in res.coef
    assert res.diagnostics["stage1_r2"] > 0.5
    assert res.diagnostics["stage2_converged"]
    np.testing.assert_allclose(len(res.tfp), len(res.sample))
    assert len(res.tfp) == len(res.sample)


def test_levinsohn_petrin_runs(panel):
    """Same caveat as OP — LP β_l identification fails in the ACF DGP.
    We test that the estimator runs and produces output."""
    df, _ = panel
    res = sp.levinsohn_petrin(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=3,
        productivity_degree=1,
    )
    assert "l" in res.coef and "k" in res.coef
    assert res.diagnostics["stage1_r2"] > 0.5


def test_acf_recovers_params(panel):
    """ACF (2015) is the rigorous identification — under our DGP it
    should recover (beta_l, beta_k) close to truth."""
    df, truth = panel
    res = sp.ackerberg_caves_frazer(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=3,
        productivity_degree=1,
    )
    assert _close(res.coef["l"], truth["beta_l"], tol=0.10)
    assert _close(res.coef["k"], truth["beta_k"], tol=0.10)
    assert res.method == "acf"


def test_wooldridge_runs():
    """Wooldridge (2009) stacked 2SLS instruments labor with its lag, so the
    panel carries a persistent wage shock (an i.i.d. labor shock leaves the
    instrument irrelevant once (k, m) absorb productivity)."""
    df, truth = _simulate_panel(persistent_wage=True)
    res = sp.wooldridge_prod(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
    )
    np.testing.assert_allclose(len(res.tfp), len(res.sample))
    assert _close(res.coef["l"], truth["beta_l"], tol=0.10)
    assert _close(res.coef["k"], truth["beta_k"], tol=0.15)
    assert res.method == "wrdg"
    assert res.diagnostics["vce"] == "cluster"
    assert np.all(np.isfinite(res.std_errors)) and np.all(res.std_errors > 0)
    assert res.diagnostics["n_obs_stacked"] == 2 * len(res.sample)


def test_wooldridge_conventions_and_vce():
    df, truth = _simulate_panel(n_firms=120, n_periods=10, persistent_wage=True)
    kw = dict(output="y", free="l", state="k", proxy="m", panel_id="id", time="year")
    gmm = {v: sp.wooldridge_prod(df, vce=v, **kw) for v in ("cluster", "robust")}
    np.testing.assert_allclose(
        gmm["robust"].params.to_numpy(),
        gmm["cluster"].params.to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(gmm["cluster"].std_errors, gmm["robust"].std_errors)
    # GMM estimates the Markov slope; prodest's convention fixes it at 1 and,
    # with mean-reverting productivity (rho = 0.7), biases capital.
    assert gmm["cluster"].productivity_process["rho"] == pytest.approx(
        truth["rho"], abs=0.1
    )
    pro = sp.wooldridge_prod(df, convention="prodest", vce="unadjusted", **kw)
    # the imposed unit slope is the Markov polynomial; ``rho`` is, as for
    # every estimator, the projection slope of TFP on its lag
    assert pro.diagnostics["markov_coef"][1] == 1.0
    assert abs(gmm["cluster"].coef["k"] - truth["beta_k"]) < abs(
        pro.coef["k"] - truth["beta_k"]
    )
    with pytest.raises(ValueError, match="vce"):
        sp.wooldridge_prod(df, vce="hc3", **kw)
    with pytest.raises(ValueError, match="unadjusted"):
        sp.wooldridge_prod(df, vce="unadjusted", **kw)
    with pytest.raises(ValueError, match="productivity_degree"):
        sp.wooldridge_prod(df, convention="prodest", productivity_degree=1, **kw)
    with pytest.raises(ValueError, match="convention"):
        sp.wooldridge_prod(df, convention="stata", **kw)


# ---------------------------------------------------------------------------
# Dispatcher + aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["op", "lp", "acf", "wrdg"])
def test_prod_fn_dispatcher(panel, method):
    df, _ = panel
    if method == "wrdg":
        df, _ = _simulate_panel(persistent_wage=True)
    proxy = "i" if method == "op" else "m"
    # wooldridge_prod has no productivity polynomial; the dispatcher only
    # forwards degrees that are given.
    degrees = {} if method == "wrdg" else {"productivity_degree": 1}
    res = sp.prod_fn(
        df,
        output="y",
        free="l",
        state="k",
        proxy=proxy,
        panel_id="id",
        time="year",
        method=method,
        polynomial_degree=2,
        **degrees,
    )
    np.testing.assert_allclose(len(res.tfp), len(res.sample))
    assert hasattr(res, "coef")
    assert hasattr(res, "tfp")
    assert len(res.tfp) > 0


def test_aliases_match_canonical(panel):
    df, _ = panel
    res1 = sp.acf(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
    res2 = sp.ackerberg_caves_frazer(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
    assert np.isclose(res1.coef["l"], res2.coef["l"])
    assert np.isclose(res1.coef["k"], res2.coef["k"])


# ---------------------------------------------------------------------------
# Bootstrap SE
# ---------------------------------------------------------------------------


def test_bootstrap_produces_se(panel):
    df, _ = panel
    res = sp.acf(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
        boot_reps=20,
        seed=0,
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
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )

    # markup() expects revenue/cost columns inside result.sample.
    res.sample["log_rev"] = (
        df.loc[res.sample.index, "log_rev"].to_numpy()
        if "log_rev" in df
        else df["log_rev"].to_numpy()[: len(res.sample)]
    )
    res.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res.sample)]

    # The flexible input here must be one of the production-function
    # coefficients. ACF coefs are {l, k}, so we re-fit with m as a free
    # input to expose theta_m.
    res2 = sp.acf(
        df,
        output="y",
        free=["l", "m"],
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
    res2.sample["log_rev"] = df["log_rev"].to_numpy()[: len(res2.sample)]
    res2.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res2.sample)]

    mu = sp.markup(
        res2, revenue="log_rev", input_cost="log_mat_cost", flexible_input="m"
    )
    theta_m = float(res2.coef["m"])
    cost_share = np.exp(
        res2.sample["log_mat_cost"].to_numpy(dtype=float)
        - (
            res2.sample["log_rev"].to_numpy(dtype=float)
            - res2.sample["eta"].to_numpy(dtype=float)
        )
    )
    np.testing.assert_allclose(mu.to_numpy(), theta_m / cost_share)
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
        sp.olley_pakes(
            df.drop(columns=["i"]),
            output="y",
            free="l",
            state="k",
            proxy="i",
            panel_id="id",
            time="year",
        )


def test_too_few_obs_raises():
    df = pd.DataFrame(
        {
            "y": [1.0, 1.1],
            "l": [0.5, 0.6],
            "k": [0.4, 0.4],
            "m": [0.3, 0.3],
            "id": [0, 0],
            "year": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="valid observations"):
        sp.acf(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )


def test_op_drops_zero_investment(panel):
    df, _ = panel
    df = df.copy()
    df.loc[df.index[:5], "i"] = 0.0  # five rows with zero investment
    n_before = len(df)
    res = sp.olley_pakes(
        df,
        output="y",
        free="l",
        state="k",
        proxy="i",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
    # Should drop the zero-i rows internally
    assert len(res.sample) < n_before


# ---------------------------------------------------------------------------
# Result object surface
# ---------------------------------------------------------------------------


def test_production_result_has_summary(panel):
    df, _ = panel
    res = sp.acf(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
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
    res = sp.acf(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
    )
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
    res = sp.acf(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=3,
        productivity_degree=1,
        functional_form="translog",
    )
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
    """Dispatcher should pass functional_form through to ACF."""
    df, _ = panel
    res = sp.prod_fn(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        method="acf",
        polynomial_degree=2,
        productivity_degree=1,
        functional_form="translog",
    )
    assert "ll" in res.coef and "kk" in res.coef and "lk" in res.coef


@pytest.mark.parametrize("method", ["op", "lp"])
def test_oplp_translog_raises(panel, method):
    """OP / LP read the free-input coefficients off the linear stage-1
    regression, so translog must raise rather than return wrong numbers."""
    df, _ = panel
    with pytest.raises(NotImplementedError, match="cobb-douglas"):
        sp.prod_fn(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i" if method == "op" else "m",
            panel_id="id",
            time="year",
            method=method,
            functional_form="translog",
        )


def test_translog_markup_uses_firm_time_elasticities(panel):
    """sp.markup with translog should pick up firm-time θ_v_it from the
    elasticity panel rather than a constant."""
    df, _ = panel
    rng = np.random.default_rng(0)
    df = df.copy()
    df["log_rev"] = df["y"] + np.log(1.20) + rng.normal(0, 0.05, len(df))
    df["log_mat_cost"] = df["m"] + np.log(0.80) + rng.normal(0, 0.05, len(df))

    res = sp.acf(
        df,
        output="y",
        free=["l", "m"],
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        polynomial_degree=2,
        productivity_degree=1,
        functional_form="translog",
    )
    res.sample["log_rev"] = df["log_rev"].to_numpy()[: len(res.sample)]
    res.sample["log_mat_cost"] = df["log_mat_cost"].to_numpy()[: len(res.sample)]

    mu = sp.markup(
        res, revenue="log_rev", input_cost="log_mat_cost", flexible_input="m"
    )
    assert isinstance(mu, pd.Series)
    assert (mu > 0).all()
    # Translog markup should vary across firm-times (not constant).
    assert mu.std() > 0


def test_wooldridge_translog_raises():
    """Wooldridge currently only supports Cobb-Douglas — translog must
    raise NotImplementedError, not silently return wrong numbers."""
    df, _ = _simulate_panel(n_firms=80, n_periods=8)
    with pytest.raises(NotImplementedError, match="cobb-douglas"):
        sp.wooldridge_prod(
            df,
            output="y",
            free="l",
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
            functional_form="translog",
        )


def test_unknown_functional_form_raises(panel):
    """Anything outside {cobb-douglas, translog} must raise immediately."""
    df, _ = panel
    with pytest.raises(ValueError, match="Unknown functional_form"):
        sp.acf(
            df,
            output="y",
            free="l",
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
            functional_form="ces",
        )


def test_panel_lag_matches_calendar_period():
    """The lag is the value at time - 1 of the same panel, not the previous
    row: a skipped year leaves NaN (as Stata's L. and prodest's lagPanel)."""
    from statspai.structural.production._core import panel_lag

    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "t": [2000, 2001, 2003, 2000, 2001],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[10, 11, 12, 13, 14],
    )
    lag = panel_lag(df, "x", "id", "t")
    assert list(lag.index) == [10, 11, 12, 13, 14]
    np.testing.assert_array_equal(lag.to_numpy(), [np.nan, 1.0, np.nan, np.nan, 4.0])
    # row order does not matter
    shuffled = df.iloc[[4, 2, 0, 3, 1]]
    np.testing.assert_array_equal(
        panel_lag(shuffled, "x", "id", "t").to_numpy(),
        [4.0, np.nan, np.nan, np.nan, 1.0],
    )
    with pytest.raises(ValueError, match="uniquely"):
        panel_lag(pd.concat([df, df.iloc[[0]]]), "x", "id", "t")
    with pytest.raises(TypeError, match="numeric"):
        panel_lag(df.assign(t=df["t"].astype(str)), "x", "id", "t")


def test_calendar_gaps_drop_the_period_after_the_gap():
    """Firms skipping a year lose the year after the gap from stage 2."""
    rng = np.random.default_rng(0)
    rows = []
    # 50 firms, but each skips year 5 (creating a 2-year gap)
    for fid in range(50):
        for t in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:  # gap at t=5
            rows.append(
                {
                    "id": fid,
                    "year": t,
                    "y": rng.normal(0, 1),
                    "l": rng.normal(0, 1),
                    "k": rng.normal(0, 1),
                    "m": rng.normal(0, 1),
                }
            )
    df = pd.DataFrame(rows)
    kw = dict(output="y", free="l", state="k", panel_id="id", time="year")
    res = sp.levinsohn_petrin(df, proxy="m", polynomial_degree=2, **kw)
    assert res.diagnostics["n_calendar_gaps"] == 50
    assert sorted(res.sample["year"].unique()) == [1, 2, 3, 4, 7, 8, 9, 10]
    assert len(res.sample) == 50 * 8


# ---------------------------------------------------------------------------
# Units of output, the persistence diagnostic, and what the bootstrap reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["op", "lp", "acf"])
def test_adding_a_constant_to_log_output_changes_nothing_but_the_level(panel, method):
    # y + c is the same output in other units. The input coefficients, the
    # convergence flag and the persistence of productivity cannot depend on c;
    # omega shifts by exactly c. Before the Markov basis was centred, c = 200
    # moved the LP capital coefficient by 1e-5, reported the stage-2 fit as
    # unconverged and gave rho = -32,396.
    df, _ = panel
    proxy = "i" if method == "op" else "m"
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in (0.0, 200.0):
            fits[c] = sp.prod_fn(
                df.assign(y=df["y"] + c),
                output="y",
                free="l",
                state="k",
                proxy=proxy,
                panel_id="id",
                time="year",
                method=method,
            )
    a, b = fits[0.0], fits[200.0]
    # 1e-6: the stage-2 criterion is flat to rounding over about 1e-8.
    np.testing.assert_allclose(b.params.to_numpy(), a.params.to_numpy(), atol=1e-6)
    assert a.diagnostics["stage2_converged"] and b.diagnostics["stage2_converged"]
    assert b.productivity_process["rho"] == pytest.approx(
        a.productivity_process["rho"], abs=1e-5
    )
    np.testing.assert_allclose(b.tfp - a.tfp, 200.0, atol=1e-4)


@pytest.mark.parametrize("method", ["op", "lp", "acf"])
def test_rho_is_the_markov_slope_when_g_is_linear(panel, method):
    df, _ = panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sp.prod_fn(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i" if method == "op" else "m",
            panel_id="id",
            time="year",
            method=method,
            productivity_degree=1,
        )
    assert res.productivity_process["rho"] == pytest.approx(
        res.diagnostics["markov_coef"][1], abs=1e-9
    )
    assert res.diagnostics["ar_rho"] == res.productivity_process["rho"]


def test_no_standard_errors_requested_is_not_a_singular_variance(panel):
    df, _ = panel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.levinsohn_petrin(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
    assert not [w for w in caught if issubclass(w.category, sp.ConvergenceWarning)]
    assert res.diagnostics["se_source"].startswith("none")
    assert np.all(np.isnan(res.std_errors))
    boot = sp.levinsohn_petrin(
        df,
        output="y",
        free="l",
        state="k",
        proxy="m",
        panel_id="id",
        time="year",
        boot_reps=20,
        seed=1,
    )
    assert boot.diagnostics["se_source"] == "firm bootstrap"
    lo, hi = np.array(boot.diagnostics["boot_ci_percentile"]).T
    assert np.all(lo < hi) and len(boot.diagnostics["boot_se_iqr"]) == 2


def test_bootstrap_flags_a_standard_error_made_by_a_few_replications():
    from statspai.structural.production import op_lp_acf as mod

    frame = pd.DataFrame({mod._PID: np.repeat(np.arange(40), 2), "x": 0.0})
    rng = np.random.default_rng(0)

    def run(outlier):
        calls = iter(range(10**6))

        def fit(_):
            i = next(calls)
            return np.array(
                [rng.normal(), 500.0 if outlier and i == 7 else rng.normal()]
            )

        info = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            se, _, n_ok = mod._firm_bootstrap(frame, fit, 100, 0, "T", info)
        return se, info, [str(w.message) for w in caught]

    se, info, msgs = run(outlier=False)
    assert not msgs
    # Normal draws: the two scales estimate the same standard deviation.
    np.testing.assert_allclose(se, info["boot_se_iqr"], rtol=0.35)
    se, info, msgs = run(outlier=True)
    assert len(msgs) == 1 and "1 of 2 parameters" in msgs[0]
    assert se[1] > 40 and info["boot_se_iqr"][1] < 1.5


def test_stage2_that_does_not_converge_says_so(panel, monkeypatch):
    from statspai.structural.production import op_lp_acf as mod

    df, _ = panel
    solve = mod._OPLPStage2.solve

    def unconverged(self):
        fit = solve(self)
        fit.status = 0
        return fit

    monkeypatch.setattr(mod._OPLPStage2, "solve", unconverged)
    with pytest.warns(sp.ConvergenceWarning, match="stage 2"):
        res = sp.levinsohn_petrin(
            df, output="y", free="l", state="k", proxy="m", panel_id="id", time="year"
        )
    assert res.diagnostics["stage2_converged"] is False


def test_op_reports_what_drop_zero_proxy_removed(panel):
    df, _ = panel
    kw = dict(output="y", free="l", state="k", proxy="i", panel_id="id", time="year")
    zeros = df.copy()
    zeros.loc[zeros.index[::7], "i"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.olley_pakes(zeros, **kw)
    assert res.diagnostics["n_dropped_nonpositive_proxy"] == len(zeros.index[::7])
    # A column in logs has negative values; dropping them is not what the
    # caller of an estimator for log inputs meant.
    logs = df.assign(i=np.log(df["i"]) - np.log(df["i"]).median())
    with pytest.warns(RuntimeWarning, match="drop_zero_proxy=False"):
        sp.olley_pakes(logs, **kw)
    kept = sp.olley_pakes(logs, drop_zero_proxy=False, **kw)
    assert kept.diagnostics["n_dropped_nonpositive_proxy"] == 0


def test_gross_output_with_the_proxy_as_an_input_is_refused(panel):
    df, _ = panel
    with pytest.raises(ValueError, match="gross-output"):
        sp.levinsohn_petrin(
            df,
            output="y",
            free=["l", "m"],
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
        )


# ---------------------------------------------------------------------------
# Option scan, round 6 (2026-10-27): the units of the log inputs
# ---------------------------------------------------------------------------
#
# Log capital measured in dollars instead of thousands of dollars is the same
# data plus a constant. The elasticities must not move. The stage-2 start of
# OP / LP (and the start and root choice of ACF) used to be the linear
# coefficient of the raw stage-1 polynomial, which depends on that constant.


def _fit_shifted(df, method, cols, shift, **kw):
    d = df.copy()
    for c in cols:
        d[c] = d[c] + shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return sp.prod_fn(
            d,
            output="y",
            free="l",
            state="k",
            proxy="i" if method == "op" else "m",
            panel_id="id",
            time="year",
            method=method,
            **kw,
        )


@pytest.mark.parametrize(
    "method, cols, shift",
    [
        ("op", ["k"], 8.0),
        ("lp", ["k", "m"], 3.0),
        ("acf", ["l", "k", "m"], 6.0),
    ],
)
def test_estimates_do_not_depend_on_the_origin_of_log_inputs(method, cols, shift):
    df, _ = _simulate_panel(seed=0, persistent_wage=True)
    base = _fit_shifted(df, method, cols, 0.0)
    moved = _fit_shifted(df, method, cols, shift)
    np.testing.assert_allclose(moved.params, base.params, rtol=1e-6, atol=1e-8)
    assert moved.productivity_process["rho"] == pytest.approx(
        base.productivity_process["rho"], rel=1e-6
    )


def test_wooldridge_does_not_depend_on_the_origin_of_log_inputs():
    """Elasticities, their standard errors and the persistence are the same
    when log capital and log materials are shifted; the Markov polynomial,
    written in powers of lagged TFP, moves with TFP: ``G'(w) = G(w - d) + d``
    where ``d`` is the shift of TFP."""
    df, _ = _simulate_panel(seed=0, persistent_wage=True)
    base = _fit_shifted(df, "wrdg", ["k", "m"], 0.0)
    moved = _fit_shifted(df, "wrdg", ["k", "m"], 7.0)
    np.testing.assert_allclose(moved.params, base.params, rtol=1e-7)
    np.testing.assert_allclose(moved.std_errors, base.std_errors, rtol=1e-6)
    assert moved.productivity_process["rho"] == pytest.approx(
        base.productivity_process["rho"], rel=1e-7
    )
    d = float(np.mean(moved.tfp - base.tfp))
    np.testing.assert_allclose(moved.tfp - base.tfp, d, atol=1e-7)
    G = np.polynomial.Polynomial(base.diagnostics["markov_coef"])
    G_moved = np.polynomial.Polynomial(moved.diagnostics["markov_coef"])
    w = np.linspace(base.tfp.min(), base.tfp.max(), 7) + d
    np.testing.assert_allclose(G_moved(w), G(w - d) + d, atol=1e-6)


@pytest.mark.parametrize("method", ["op", "lp", "acf", "wrdg"])
def test_sample_keeps_the_row_labels_of_the_data(method):
    """``res.sample`` is indexed like the data, so a column the estimator did
    not use (revenue for ``sp.markup``) is brought in with ``.loc``."""
    df, _ = _simulate_panel(n_firms=80, seed=1, persistent_wage=True)
    df = df.sample(frac=1.0, random_state=0)
    df.index = df.index + 1000
    res = _fit_shifted(df, method, [], 0.0)
    cols = ["y", "l", "k", "id", "year"]
    pd.testing.assert_frame_equal(
        res.sample[cols], df.loc[res.sample.index, cols], check_dtype=False
    )
    if method != "wrdg":
        el = res.model_info["elasticities"]
        assert el.index.equals(res.sample.index)
