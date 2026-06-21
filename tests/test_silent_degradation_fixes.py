"""Behavioral regression tests for silent-degradation correctness fixes.

CLAUDE.md §7 forbids silently swallowing a numerical failure and returning a
degraded estimate as if it were the requested one. These tests lock in the
fixes that turn previously-silent degradations into loud warnings + an audit
trail in the result's ``model_info`` / ``diagnostics``.

Each test forces the failure path deterministically (monkeypatching the
fragile sub-fit to fail) so it does not depend on a specific BLAS / statsmodels
separation behavior.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# WS-A1: covariate-adjusted logit silently reverting to the unadjusted
#        marginal mean (front_door, principal_strat).
# ---------------------------------------------------------------------------


def _front_door_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    D = (rng.uniform(size=n) < 0.5).astype(int)
    M = (rng.uniform(size=n) < 0.3 + 0.3 * D).astype(int)
    Y = 1.0 * M + 0.5 * X + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "D": D, "M": M, "X": X})


def test_front_door_warns_when_mediator_logit_degrades(monkeypatch):
    """When the covariate-adjusted mediator logit fails, front_door must warn
    that the reported ATE is no longer covariate-adjusted (not silently swap
    in the marginal P(M=1))."""
    # NB: ``statspai.inference.front_door`` resolves to the re-exported
    # function, so fetch the real module via sys.modules.
    fd = importlib.import_module("statspai.inference.front_door")

    df = _front_door_data()

    # Force the mediator logit to fail on every fit -> marginal-mean fallback.
    monkeypatch.setattr(fd, "_logit_fit", lambda y, X: None)

    with pytest.warns(RuntimeWarning, match="no longer covariate-adjusted"):
        res = sp.front_door(
            df,
            y="Y",
            treat="D",
            mediator="M",
            covariates=["X"],
            n_boot=10,
            seed=1,
        )
    assert res.model_info["mediator_model_degraded"] is True
    assert res.model_info["mediator_model_fallback_arms"] == 2


def test_front_door_clean_run_records_no_degradation():
    """A healthy fit must report mediator_model_degraded=False."""
    df = _front_door_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.front_door(
            df,
            y="Y",
            treat="D",
            mediator="M",
            covariates=["X"],
            n_boot=10,
            seed=1,
        )
    assert res.model_info["mediator_model_degraded"] is False
    assert res.model_info["mediator_model_fallback_arms"] == 0


def _principal_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    D = (rng.uniform(size=n) < 0.5).astype(int)
    # S = post-treatment intermediate (e.g. survival / employment)
    S = (rng.uniform(size=n) < 0.5 + 0.2 * D + 0.1 * X).astype(int)
    Y = 2.0 * S + 0.5 * X + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "D": D, "S": S, "X": X})


def test_principal_score_warns_when_logit_degrades(monkeypatch):
    """principal_strat(method='principal_score') must warn when the
    covariate-adjusted principal-score logit reverts to the marginal."""
    ps = importlib.import_module("statspai.principal_strat.principal_strat")

    df = _principal_data()
    monkeypatch.setattr(ps, "_logit_safe", lambda y, X: None)

    with pytest.warns(RuntimeWarning, match="no longer covariate-adjusted"):
        res = sp.principal_strat(
            df,
            y="Y",
            treat="D",
            strata="S",
            covariates=["X"],
            method="principal_score",
            n_boot=10,
            seed=1,
        )
    assert res.model_info["principal_score_degraded"] is True


# ---------------------------------------------------------------------------
# WS-A5: shared bootstrap-SE helper replacing `np.nanstd(boot) or 1e-6`.
# ---------------------------------------------------------------------------


def test_bootstrap_se_healthy_matches_nanstd():
    """On a fully-successful bootstrap, bootstrap_se == np.std(ddof=1)."""
    from statspai.core._bootstrap import bootstrap_se

    boot = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert bootstrap_se(boot, 5, "x") == pytest.approx(np.std(boot, ddof=1))


def test_bootstrap_se_warns_on_partial_failure():
    """Some failed replicates -> warn, still compute over survivors."""
    from statspai.core._bootstrap import bootstrap_se

    boot = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    with pytest.warns(RuntimeWarning, match="2/5 bootstrap replicates failed"):
        se = bootstrap_se(boot, 5, "myest")
    assert se == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))


def test_bootstrap_se_returns_nan_below_floor():
    """Below the success floor -> NaN, never a fabricated tiny SE."""
    from statspai.core._bootstrap import bootstrap_se

    boot = np.array([1.0, np.nan, np.nan, np.nan])
    with pytest.warns(RuntimeWarning, match="undefined"):
        se = bootstrap_se(boot, 4, "myest", min_success=2)
    assert np.isnan(se)


def test_cb_ipw_records_failure_diagnostics():
    """cb_ipw bridge must expose CB-path failure bookkeeping in detail."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    D = (rng.uniform(size=n) < 0.5).astype(int)
    Y = 2.0 * D + X @ np.array([1.0, -0.5]) + rng.normal(size=n)
    df = pd.DataFrame({"y": Y, "d": D, "x1": X[:, 0], "x2": X[:, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bridge(
            "cb_ipw",
            data=df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            n_boot=30,
            seed=1,
        )
    # Healthy run: CB path succeeds, bookkeeping present.
    assert res.detail["cb_failed"] is False
    assert "n_boot_cb_failed" in res.detail


# ---------------------------------------------------------------------------
# WS-A4: panel_unitroot silently dropping units from the test.
# ---------------------------------------------------------------------------


def test_panel_unitroot_warns_when_units_dropped():
    """A unit with too few periods is excluded; the test must warn rather
    than silently shrink the unit set."""
    rng = np.random.default_rng(0)
    rows = []
    # 6 healthy units with 30 periods each.
    for u in range(6):
        y = np.cumsum(rng.normal(size=30)) * 0.1 + rng.normal(size=30)
        for t in range(30):
            rows.append({"id": u, "time": t, "gdp": y[t]})
    # 1 too-short unit (3 periods) -> dropped.
    for t in range(3):
        rows.append({"id": 99, "time": t, "gdp": rng.normal()})
    df = pd.DataFrame(rows)
    with pytest.warns(RuntimeWarning, match="computed over 6/7 units"):
        res = sp.panel_unitroot(df, variable="gdp", id="id", time="time", test="ips")
    assert res.n_units == 6
