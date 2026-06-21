"""Coverage tests for scpi/scest/scdata, discos (distributional SC), and SDID."""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

scpi_mod = importlib.import_module("statspai.synth.scpi")
discos_mod = importlib.import_module("statspai.synth.discos")
sdid_mod = importlib.import_module("statspai.synth.sdid")


def _panel(
    n_units=9, n_periods=18, treatment_time=12, effect=4.0, seed=21, with_cov=False
):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            row = {"unit": f"u{i}", "time": t, "outcome": y}
            if with_cov:
                row["x1"] = alphas[i] + rng.normal(0, 0.1)
            records.append(row)
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=12
)


@pytest.fixture
def panel():
    return _panel()


# ---------------- scpi / scest / scdata ----------------
def test_scdata_structure(panel):
    d = scpi_mod.scdata(panel, **COMMON)
    assert isinstance(d, dict)
    assert "Y_pre" in d or "A" in d or len(d) > 0


@pytest.mark.parametrize("w_constr", ["simplex", "lasso", "ridge", "ols"])
def test_scest_constraints(panel, w_constr):
    d = scpi_mod.scest(
        panel, **COMMON, w_constr=w_constr, lasso_lambda=0.5, ridge_lambda=0.5
    )
    assert isinstance(d, dict)
    w = d.get("weights")
    assert w is not None
    w = np.asarray(list(w.values()) if isinstance(w, dict) else w, dtype=float)
    if w_constr == "simplex":
        assert w.min() >= -1e-6
        assert abs(w.sum() - 1.0) < 1e-2


@pytest.mark.parametrize("e_method", ["gaussian", "ls", "qreg"])
def test_scpi_e_methods(panel, e_method):
    res = scpi_mod.scpi(
        panel, **COMMON, w_constr="simplex", pi_type="both", e_method=e_method, seed=0
    )
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    # prediction intervals should be present
    assert any(
        "pi" in str(k).lower() or "interval" in str(k).lower() or "ci" in str(k).lower()
        for k in mi
    )


def test_scpi_pi_types(panel):
    for pi in ("in_sample", "out_of_sample", "both"):
        res = scpi_mod.scpi(panel, **COMMON, pi_type=pi, e_method="gaussian", seed=1)
        assert np.isfinite(res.estimate)


def test_scpi_lasso_ridge_constr(panel):
    res = scpi_mod.scpi(
        panel, **COMMON, w_constr="lasso", lasso_lambda=0.3, e_method="gaussian", seed=2
    )
    assert np.isfinite(res.estimate)
    res2 = scpi_mod.scpi(
        panel, **COMMON, w_constr="ridge", ridge_lambda=0.3, e_method="gaussian", seed=3
    )
    assert np.isfinite(res2.estimate)


# ---------------- discos ----------------
def _distributional_panel(
    n_units=8, n_periods=14, treatment_time=9, n_obs_per_cell=30, seed=31
):
    """Panel with multiple observations per (unit, time) so distributions
    are non-degenerate (required for distributional SC)."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_units):
        mu = rng.normal(5, 1)
        sd = rng.uniform(0.8, 1.5)
        for t in range(1, n_periods + 1):
            loc = mu + 0.1 * t
            for _ in range(n_obs_per_cell):
                y = rng.normal(loc, sd)
                if i == 0 and t >= treatment_time:
                    y += 2.0
                records.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(records)


@pytest.mark.parametrize("method", ["mixture", "quantile"])
def test_discos_methods(method):
    df = _distributional_panel()
    res = discos_mod.discos(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        method=method,
        n_quantiles=50,
        placebo=True,
        seed=0,
    )
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    assert len(mi) > 0


def test_qqsynth():
    df = _distributional_panel()
    res = discos_mod.qqsynth(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        n_quantiles=40,
        placebo=False,
        seed=1,
    )
    assert np.isfinite(res.estimate)


def test_discos_test_ks_cvm():
    df = _distributional_panel()
    res = discos_mod.discos(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        method="mixture",
        n_quantiles=40,
        placebo=False,
        seed=2,
    )
    for test in ("ks", "cvm"):
        out = discos_mod.discos_test(res, test=test)
        assert isinstance(out, dict)
        assert "statistic" in out or "pvalue" in out or len(out) > 0


def test_stochastic_dominance_orders():
    df = _distributional_panel()
    res = discos_mod.discos(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=9,
        method="quantile",
        n_quantiles=40,
        placebo=False,
        seed=3,
    )
    for order in (1, 2):
        out = discos_mod.stochastic_dominance(res, order=order)
        assert isinstance(out, dict)
        assert len(out) > 0


# ---------------- sdid ----------------
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_sdid_methods(panel, method):
    res = sdid_mod.sdid(
        panel,
        **COMMON,
        method=method,
        se_method="placebo",
        n_reps=20,
        backend="native",
        seed=0,
    )
    assert np.isfinite(res.estimate)


@pytest.mark.parametrize("se_method", ["placebo", "bootstrap", "jackknife"])
def test_sdid_se_methods(panel, se_method):
    res = sdid_mod.sdid(
        panel,
        **COMMON,
        method="sdid",
        se_method=se_method,
        n_reps=20,
        backend="native",
        seed=1,
    )
    assert np.isfinite(res.estimate)
    assert res.se is None or np.isfinite(res.se) or np.isnan(res.se)


def test_sdid_alias_estimators(panel):
    r1 = sdid_mod.synthdid_estimate(
        panel,
        y="outcome",
        unit="unit",
        time="time",
        treat_unit="u0",
        treat_time=12,
        n_reps=15,
        seed=0,
    )
    r2 = sdid_mod.sc_estimate(
        panel,
        y="outcome",
        unit="unit",
        time="time",
        treat_unit="u0",
        treat_time=12,
        n_reps=15,
        seed=0,
    )
    r3 = sdid_mod.did_estimate(
        panel,
        y="outcome",
        unit="unit",
        time="time",
        treat_unit="u0",
        treat_time=12,
        n_reps=15,
        seed=0,
    )
    for r in (r1, r2, r3):
        assert np.isfinite(r.estimate)


def test_sdid_with_covariates(panel):
    p = _panel(with_cov=True)
    res = sdid_mod.sdid(
        p,
        **COMMON,
        method="sdid",
        covariates=["x1"],
        n_reps=15,
        backend="native",
        seed=2,
    )
    assert np.isfinite(res.estimate)


def test_synthdid_placebo_table(panel):
    df = sdid_mod.synthdid_placebo(
        panel,
        y="outcome",
        unit="unit",
        time="time",
        treat_unit="u0",
        treat_time=12,
        method="sdid",
        n_reps=15,
        seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_california_prop99_loader():
    df = sdid_mod.california_prop99()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
