"""Coverage tests for panel unit-root, FGLS, binary-choice, interactive-FE.

Real synthetic panels; assert structure / sane statistical properties
(finite stats, valid p-values, coef shapes, positive SEs, warnings on
excluded units), not fabricated numbers.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.panel.unit_root import PanelUnitRootResult, panel_unitroot
from statspai.panel.panel_fgls import panel_fgls
from statspai.panel.panel_binary import panel_logit, panel_probit
from statspai.panel.interactive_fe import interactive_fe


# ── Unit-root data ──────────────────────────────────────────────────────

@pytest.fixture
def ur_df():
    rng = np.random.default_rng(5)
    n_id, T = 12, 30
    rows = []
    for i in range(n_id):
        # stationary AR(1) series
        e = rng.normal(0, 1, T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.4 * y[t - 1] + e[t]
        for t in range(T):
            rows.append({"id": i, "time": t, "g": y[t]})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("test", ["ips", "llc", "fisher", "hadri"])
def test_panel_unitroot_methods(ur_df, test):
    res = panel_unitroot(ur_df, variable="g", id="id", time="time", test=test)
    assert isinstance(res, PanelUnitRootResult)
    assert np.isfinite(res.statistic)
    assert 0.0 <= res.p_value <= 1.0
    s = res.summary()
    assert "Panel Unit Root Test" in s
    assert "Conclusion" in s


@pytest.mark.parametrize("trend", ["c", "ct", "n"])
def test_panel_unitroot_trends(ur_df, trend):
    res = panel_unitroot(ur_df, variable="g", id="id", time="time",
                         test="ips", trend=trend)
    assert np.isfinite(res.statistic)


def test_panel_unitroot_hadri_trend_ct(ur_df):
    res = panel_unitroot(ur_df, variable="g", id="id", time="time",
                         test="hadri", trend="ct")
    assert np.isfinite(res.statistic)


def test_panel_unitroot_unknown_test(ur_df):
    with pytest.raises(ValueError, match="Unknown test"):
        panel_unitroot(ur_df, variable="g", id="id", time="time", test="zzz")


def test_panel_unitroot_warns_on_short_units(ur_df):
    # Append an entity with only 3 periods -> excluded with a warning
    extra = pd.DataFrame({"id": [99, 99, 99], "time": [0, 1, 2],
                          "g": [0.1, 0.2, 0.3]})
    df = pd.concat([ur_df, extra], ignore_index=True)
    with pytest.warns(RuntimeWarning, match="units"):
        res = panel_unitroot(df, variable="g", id="id", time="time", test="ips")
    assert np.isfinite(res.statistic)


def test_panel_unitroot_all_short_raises():
    df = pd.DataFrame({"id": [0, 0, 1, 1], "time": [0, 1, 0, 1],
                       "g": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="no unit yielded a valid ADF"):
        panel_unitroot(df, variable="g", id="id", time="time", test="ips")


# ── Panel FGLS ──────────────────────────────────────────────────────────

@pytest.fixture
def fgls_df():
    rng = np.random.default_rng(9)
    n_id, T = 8, 20
    rows = []
    for i in range(n_id):
        scale = 0.5 + i * 0.2  # heteroskedastic across panels
        prev = 0.0
        for t in range(T):
            x1 = rng.normal()
            x2 = rng.normal()
            e = 0.5 * prev + rng.normal(0, scale)
            prev = e
            y = 1.0 + 2.0 * x1 - 0.5 * x2 + e
            rows.append({"id": i, "time": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("panels", ["homoskedastic", "heteroskedastic"])
@pytest.mark.parametrize("corr", ["independent", "ar1", "psar1"])
def test_panel_fgls_variants(fgls_df, panels, corr):
    res = panel_fgls(fgls_df, y="y", x=["x1", "x2"], id="id", time="time",
                     panels=panels, corr=corr)
    assert "_cons" in res.params.index
    assert len(res.params) == 3
    assert np.all(res.std_errors.values >= 0) or np.isnan(res.std_errors.values).any()
    assert "r_squared" in res.diagnostics


def test_panel_fgls_recovers_slope(fgls_df):
    res = panel_fgls(fgls_df, y="y", x=["x1", "x2"], id="id", time="time",
                     panels="heteroskedastic", corr="ar1")
    assert abs(res.params["x1"] - 2.0) < 0.5


# ── Panel binary choice ─────────────────────────────────────────────────

@pytest.fixture
def bin_df():
    rng = np.random.default_rng(13)
    n_id, T = 60, 6
    rows = []
    for i in range(n_id):
        alpha = rng.normal(0, 1)
        for t in range(T):
            x1 = rng.normal()
            eta = alpha + 1.0 * x1
            p = 1.0 / (1.0 + np.exp(-eta))
            y = int(rng.uniform() < p)
            rows.append({"id": i, "time": t, "y": y, "x1": x1})
    return pd.DataFrame(rows)


def test_panel_logit_fe(bin_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_logit(bin_df, y="y", x=["x1"], id="id", time="time",
                          method="fe")
    assert "x1" in res.params.index
    assert res.model_info["method"] == "fe"


def test_panel_logit_re(bin_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_logit(bin_df, y="y", x=["x1"], id="id", time="time",
                          method="re", n_quadrature=8)
    assert res.model_info["method"] == "re"
    assert "sigma_u" in res.model_info


def test_panel_logit_cre(bin_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_logit(bin_df, y="y", x=["x1"], id="id", time="time",
                          method="cre", n_quadrature=8)
    assert res.model_info["method"] == "cre"
    assert res.model_info["original_x"] == ["x1"]


def test_panel_logit_bad_method(bin_df):
    with pytest.raises(ValueError, match="method must be"):
        panel_logit(bin_df, y="y", x=["x1"], id="id", time="time",
                    method="bogus")


def test_panel_probit_re(bin_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_probit(bin_df, y="y", x=["x1"], id="id", time="time",
                           method="re", n_quadrature=8)
    assert res.model_info["link"] == "probit"


def test_panel_probit_cre(bin_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_probit(bin_df, y="y", x=["x1"], id="id", time="time",
                           method="cre", n_quadrature=8)
    assert res.model_info["method"] == "cre"


def test_panel_probit_fe_unsupported(bin_df):
    with pytest.raises(ValueError, match="incidental parameters"):
        panel_probit(bin_df, y="y", x=["x1"], id="id", time="time",
                     method="fe")


def test_panel_logit_cre_multiple_x(bin_df):
    df = bin_df.copy()
    rng = np.random.default_rng(2)
    df["x2"] = rng.normal(size=len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = panel_logit(df, y="y", x=["x1", "x2"], id="id", time="time",
                          method="cre", n_quadrature=6)
    assert res.model_info["method"] == "cre"


# ── Interactive fixed effects ───────────────────────────────────────────

@pytest.fixture
def ife_df():
    rng = np.random.default_rng(17)
    N, T, r = 15, 12, 2
    Lam = rng.normal(size=(N, r))
    F = rng.normal(size=(T, r))
    rows = []
    for i in range(N):
        for t in range(T):
            x1 = rng.normal()
            x2 = rng.normal()
            y = (0.8 * x1 - 0.4 * x2 + Lam[i] @ F[t]
                 + rng.normal(0, 0.3))
            rows.append({"id": i, "time": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


def test_interactive_fe_iterative(ife_df):
    res = interactive_fe(ife_df, y="y", x=["x1", "x2"], id="id", time="time",
                         n_factors=2)
    assert list(res.params.index) == ["x1", "x2"]
    assert np.all(res.std_errors.values >= 0)
    assert "eigenvalues" in res.diagnostics
    assert res.model_info["n_factors"] == 2


def test_interactive_fe_non_robust(ife_df):
    res = interactive_fe(ife_df, y="y", x=["x1", "x2"], id="id", time="time",
                         n_factors=1, robust=False)
    assert "r_squared" in res.diagnostics
