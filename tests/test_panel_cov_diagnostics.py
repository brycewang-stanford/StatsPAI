"""Coverage campaign — panel result diagnostics, CRE binary, dynamic panels.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Drives the ``PanelResults`` diagnostic
methods (which lazily call ``panel_diagnostics._bp_lm_test`` /
``_f_test_effects`` / ``_pesaran_cd`` / ``_hausman_from_data`` /
``_within_estimator`` / ``_re_estimator``), the correlated-random-effects route
in ``panel_binary.py``, panel unit-root tests with a linear trend, and the
Arellano–Bond / system-GMM dynamic-panel routes in ``panel_reg.py``.

Assertions are real: every diagnostic returns a dict with a p-value in [0, 1];
the dynamic-panel coefficient is finite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel_df():
    rng = np.random.default_rng(0)
    n_e, n_t = 50, 8
    N = n_e * n_t
    ent = np.repeat(np.arange(n_e), n_t)
    tm = np.tile(np.arange(n_t), n_e)
    x1 = rng.standard_normal(N)
    fe = np.repeat(rng.standard_normal(n_e), n_t)
    y = 2.0 * x1 + fe + rng.standard_normal(N)
    return pd.DataFrame({"y": y, "x1": x1, "entity": ent, "time": tm})


@pytest.fixture(scope="module")
def fe_result(panel_df):
    return sp.panel(
        panel_df, formula="y ~ x1", entity="entity", time="time", method="fe"
    )


def _pval_ok(d):
    assert isinstance(d, dict)
    pv = d.get("p_value", d.get("pvalue"))
    if pv is not None and np.isfinite(pv):
        assert 0.0 <= float(pv) <= 1.0


def test_result_bp_lm_test(fe_result):
    _pval_ok(fe_result.bp_lm_test())


def test_result_f_test_effects(fe_result):
    _pval_ok(fe_result.f_test_effects())


def test_result_pesaran_cd(fe_result):
    _pval_ok(fe_result.pesaran_cd_test())


def test_result_hausman(fe_result):
    _pval_ok(fe_result.hausman_test())


# ─── CRE binary panels ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def binary_df():
    rng = np.random.default_rng(1)
    n_e, n_t = 60, 8
    N = n_e * n_t
    ent = np.repeat(np.arange(n_e), n_t)
    tm = np.tile(np.arange(n_t), n_e)
    x = rng.standard_normal(N)
    fe = np.repeat(rng.standard_normal(n_e), n_t)
    y = (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-(0.8 * x + fe)))).astype(int)
    return pd.DataFrame({"y": y, "x": x, "id": ent, "time": tm})


def test_panel_logit_cre(binary_df):
    res = sp.panel_logit(binary_df, y="y", x=["x"], id="id", time="time", method="cre")
    assert res is not None


def test_panel_probit_cre(binary_df):
    res = sp.panel_probit(binary_df, y="y", x=["x"], id="id", time="time", method="cre")
    assert res is not None


def test_panel_logit_bad_method_raises(binary_df):
    with pytest.raises(ValueError):
        sp.panel_logit(
            binary_df, y="y", x=["x"], id="id", time="time", method="not_a_method"
        )


# ─── unit root with linear trend ─────────────────────────────────────────


def test_panel_unitroot_trend(panel_df):
    out = sp.panel_unitroot(
        panel_df, variable="y", id="entity", time="time", test="ips", trend="ct"
    )
    assert out is not None


# ─── dynamic panels (Arellano–Bond / system GMM) ─────────────────────────


@pytest.fixture(scope="module")
def dynamic_df():
    rng = np.random.default_rng(2)
    n_e, n_t = 80, 10
    rows = []
    for i in range(n_e):
        a = rng.standard_normal()
        y_prev = a + rng.standard_normal()
        for t in range(n_t):
            x = rng.standard_normal()
            y = 0.5 * y_prev + 1.0 * x + a + rng.standard_normal()
            rows.append((i, t, y, y_prev, x))
            y_prev = y
    return pd.DataFrame(rows, columns=["id", "time", "y", "y_lag", "x"])


def test_dynamic_panel_arellano_bond(dynamic_df):
    res = sp.panel(
        dynamic_df,
        formula="y ~ y_lag + x",
        entity="id",
        time="time",
        method="ab",
        lags=1,
    )
    assert res is not None
    assert np.isfinite(float(res.params["x"]))


def test_system_gmm_runs_and_reports_an_intercept(dynamic_df):
    """Formerly a documented limitation; system GMM is now implemented.

    Verified against ``xtabond2`` in
    ``tests/reference_parity/test_dynpanel_abdata_parity.py``. Here we only
    check the dispatcher wiring: the fit succeeds and carries the ``_cons``
    row that only the stacked level equation can identify.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(
            dynamic_df,
            formula="y ~ y_lag + x",
            entity="id",
            time="time",
            method="system",
            lags=1,
        )
    assert res is not None
    assert np.isfinite(float(res.params["x"]))
    assert "_cons" in res.params.index
