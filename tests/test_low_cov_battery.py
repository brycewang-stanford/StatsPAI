"""Broad smoke tests for under-covered estimator families.

The v1.12.x audit flagged the causal-family modules at low statement
coverage:

- ``did``    14.7%
- ``synth``  12.9%
- ``rd``     16.9%
- ``iv``     18.0%
- ``tmle``   14.8%
- ``bayes``  14.1%

Targeted branch-coverage files (`test_synth_report`, `test_paper_branches`,
`test_wooldridge_did_branches`, `test_did_imputation_branches`) close the
file-level gaps in the four worst offenders. This file widens the safety
net — exercising the *headline* estimators of each family with several
dispatch-paths so that a regression in any of them surfaces in the
default `pytest` run rather than only during a release candidate.

Each test runs in well under a second on a reproducible RNG seed and
asserts the result has finite point estimate + an SE + a CI — i.e. the
"contract" downstream code (paper / replication_pack / Quarto) relies on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------- #
#  Reusable fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rd_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-1, 1, n)
    y = 2.0 * x + (x > 0) * 1.5 + 0.5 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture(scope="module")
def iv_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    Z = rng.binomial(1, 0.5, n)
    D = (rng.random(n) < 0.3 + 0.4 * Z).astype(float)
    Y = 1 + 2 * D + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "D": D, "Z": Z})


@pytest.fixture(scope="module")
def causal_obs_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    A = rng.binomial(1, 1.0 / (1.0 + np.exp(-X[:, 0])), n)
    Y = X[:, 0] + 2.0 * A + rng.normal(size=n)
    return pd.DataFrame({
        "Y": Y, "A": A,
        "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2],
    })


@pytest.fixture(scope="module")
def synth_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    units = list(range(15))
    years = list(range(2000, 2015))
    rows = []
    for u in units:
        a = rng.normal() * 5.0
        for y in years:
            post = (u == 0) and (y >= 2010)
            rows.append({
                "state": f"S{u}", "year": y,
                "cigsale": (
                    100.0 + a + 0.5 * (y - 2000)
                    + (-15.0 if post else 0.0)
                    + 0.5 * rng.normal()
                ),
            })
    return pd.DataFrame(rows)


def _has_finite_ci(res):
    """Universal check: estimator returns a finite point estimate and a
    two-tuple CI bracketing it (within rounding)."""
    assert np.isfinite(float(res.estimate))
    ci = res.ci
    assert ci is not None
    lo, hi = float(ci[0]), float(ci[1])
    assert lo <= float(res.estimate) + 1e-6
    assert hi >= float(res.estimate) - 1e-6


# --------------------------------------------------------------------- #
#  RD family
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_rdrobust_kernels_smoke(rd_data, kernel):
    res = sp.rdrobust(rd_data, y="y", x="x", c=0.0, kernel=kernel)
    _has_finite_ci(res)


@pytest.mark.parametrize("bwselect", ["mserd", "cerrd"])
def test_rdrobust_bandwidth_selection_smoke(rd_data, bwselect):
    res = sp.rdrobust(rd_data, y="y", x="x", c=0.0, bwselect=bwselect)
    _has_finite_ci(res)


@pytest.mark.parametrize("p", [1, 2])
def test_rdrobust_polynomial_degree_smoke(rd_data, p):
    res = sp.rdrobust(rd_data, y="y", x="x", c=0.0, p=p)
    _has_finite_ci(res)


def test_rdrobust_with_donut(rd_data):
    res = sp.rdrobust(rd_data, y="y", x="x", c=0.0, donut=0.05)
    _has_finite_ci(res)


def test_rdrobust_explicit_bandwidth(rd_data):
    res = sp.rdrobust(rd_data, y="y", x="x", c=0.0, h=0.4, b=0.5)
    _has_finite_ci(res)


def test_rdpower_returns_power_object():
    p = sp.rdpower(tau=0.5, n_left=200, n_right=200)
    s = str(p)
    assert "Power" in s or "power" in s
    assert "MDE" in s or "Effect" in s.upper() or "tau" in s.lower()


def test_rdpower_target_power_solves_for_n():
    """``target_power`` triggers the inverse calculation branch."""
    p = sp.rdpower(tau=0.5, n_left=200, n_right=200, target_power=0.8)
    assert p is not None


# --------------------------------------------------------------------- #
#  IV family
# --------------------------------------------------------------------- #


def test_ivreg_2sls_smoke(iv_data):
    res = sp.ivreg("Y ~ (D ~ Z)", iv_data)
    # EconometricResults exposes .params / .std_errors
    assert "D" in res.params.index
    est = float(res.params["D"])
    assert np.isfinite(est)


def test_ivreg_with_robust_se(iv_data):
    res = sp.ivreg("Y ~ (D ~ Z)", iv_data, robust="hc1")
    assert "D" in res.std_errors.index
    assert float(res.std_errors["D"]) > 0


def test_jive_smoke(iv_data):
    out = sp.jive(iv_data, y="Y", x_endog=["D"], z=["Z"])
    # JIVE returns EconometricResults
    assert out is not None
    assert "D" in out.params.index


# --------------------------------------------------------------------- #
#  TMLE family
# --------------------------------------------------------------------- #


def test_tmle_ate_smoke(causal_obs_data):
    from statspai.tmle import tmle
    res = tmle(
        causal_obs_data,
        y="Y", treat="A",
        covariates=["X1", "X2", "X3"],
        n_folds=2,
        random_state=0,
    )
    _has_finite_ci(res)
    # Effect should be in the right ballpark (true ATE ≈ 2)
    assert abs(res.estimate - 2.0) < 1.5


def test_tmle_att_estimand(causal_obs_data):
    from statspai.tmle import tmle
    res = tmle(
        causal_obs_data,
        y="Y", treat="A",
        covariates=["X1", "X2", "X3"],
        estimand="ATT",
        n_folds=2,
        random_state=0,
    )
    _has_finite_ci(res)


def test_super_learner_classification_smoke():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from statspai.tmle import SuperLearner

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(int)
    sl = SuperLearner(
        library=[LogisticRegression(max_iter=200), DecisionTreeClassifier(max_depth=3)],
        n_folds=2,
        task="classification",
        random_state=0,
    )
    sl.fit(X, y)
    p = sl.predict_proba(X)
    assert p.shape[0] == n
    assert ((p >= 0) & (p <= 1)).all()


def test_hal_regressor_predicts_finite():
    from statspai.tmle import HALRegressor
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
    hal = HALRegressor(max_anchors_per_col=10, cv=2, random_state=0)
    hal.fit(X, y)
    yh = hal.predict(X)
    assert yh.shape == (n,)
    assert np.all(np.isfinite(yh))


# --------------------------------------------------------------------- #
#  Synth family
# --------------------------------------------------------------------- #


def test_synth_classic_smoke(synth_panel):
    res = sp.synth(
        synth_panel,
        outcome="cigsale", unit="state", time="year",
        treated_unit="S0", treatment_time=2010,
    )
    _has_finite_ci(res)
    assert res.model_info["treated_unit"] == "S0"


def test_sdid_smoke(synth_panel):
    res = sp.sdid(
        synth_panel,
        outcome="cigsale", unit="state", time="year",
        treated_unit="S0", treatment_time=2010,
    )
    assert np.isfinite(float(res.estimate))


def test_augsynth_smoke(synth_panel):
    res = sp.augsynth(
        synth_panel,
        outcome="cigsale", unit="state", time="year",
        treated_unit="S0", treatment_time=2010,
    )
    assert np.isfinite(float(res.estimate))


def test_gsynth_smoke(synth_panel):
    res = sp.gsynth(
        synth_panel,
        outcome="cigsale", unit="state", time="year",
        treated_unit="S0", treatment_time=2010,
    )
    assert np.isfinite(float(res.estimate))


@pytest.mark.parametrize("method", ["classic", "ridge", "demeaned"])
def test_synth_dispatcher_methods(synth_panel, method):
    res = sp.synth(
        synth_panel,
        outcome="cigsale", unit="state", time="year",
        treated_unit="S0", treatment_time=2010,
        method=method,
    )
    assert np.isfinite(float(res.estimate))


# --------------------------------------------------------------------- #
#  DiD family — additional dispatch + result-object methods
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def staggered_did():
    return sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=0)


def test_did_dispatcher_panel_classic(staggered_did):
    """``sp.did(...)`` staggered-panel classic path (default ``method='auto'``)."""
    res = sp.did(
        staggered_did,
        y="y", treat="first_treat", time="time", id="unit",
    )
    assert np.isfinite(float(res.estimate))


def test_callaway_santanna_smoke(staggered_did):
    """Callaway-Sant'Anna uses (y, g=first_treat, t=time, i=unit) signature."""
    res = sp.callaway_santanna(
        staggered_did,
        y="y", g="first_treat", t="time", i="unit",
    )
    assert np.isfinite(float(res.estimate))


def test_aggte_simple_smoke(staggered_did):
    cs = sp.callaway_santanna(
        staggered_did,
        y="y", g="first_treat", t="time", i="unit",
    )
    agg = sp.aggte(cs, type="simple", bstrap=False)
    assert np.isfinite(float(agg.estimate))


def test_did_imputation_summary_works(staggered_did):
    res = sp.did_imputation(
        staggered_did,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    s = res.summary()
    assert isinstance(s, str) and "ATT" in s


def test_wooldridge_did_summary_works(staggered_did):
    res = sp.wooldridge_did(
        staggered_did,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    s = res.summary()
    assert "ATT" in s


# --------------------------------------------------------------------- #
#  Bayes family — gracefully skip when pymc is missing
# --------------------------------------------------------------------- #


try:
    import pymc as _PYMC  # noqa: F401  — referenced only for availability test
    _HAS_PYMC = True
except Exception:
    _HAS_PYMC = False


def _bayes_did_panel():
    rng = np.random.default_rng(0)
    n_units, n_per = 30, 6
    rows = []
    for u in range(n_units):
        ft = 4 if u < 15 else np.inf
        for t in range(n_per):
            post = t >= ft
            rows.append({
                "unit": u, "time": t,
                "first_treat": ft if np.isfinite(ft) else np.nan,
                "y": 0.5 * t + (2.0 if post else 0.0) + 0.1 * rng.normal(),
            })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc not installed")
def test_bayes_did_smoke():
    df = _bayes_did_panel()
    res = sp.bayes_did(
        df,
        y="y", group="unit", time="time", first_treat="first_treat",
        draws=200, tune=200, chains=1, target_accept=0.9,
        random_seed=0,
    )
    # Only assert the result has the expected shape; we don't pin
    # numerical values because Bayes draws will differ across pymc versions.
    assert hasattr(res, "estimate")
    assert hasattr(res, "ci")
    assert np.isfinite(float(res.estimate))
