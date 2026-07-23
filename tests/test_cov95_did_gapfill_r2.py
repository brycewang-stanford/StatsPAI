"""Coverage gap-fill round 2 for ``statspai.did`` (93.3% → ≥95%).

Targets the remaining uncovered branches, primarily in:

- ``wooldridge_did.py`` — QR→pinv fallback, single-cluster raise, the
  per-cohort ``_etwfe_never_only`` estimator (R ``etwfe(cgroup='never')``
  semantics), the ``drdid(id=...)`` panel validation/error paths and the
  degenerate-data raises of the Sant'Anna–Zhao panel core, the calibrated
  propensity-score weight guard, the ``etwfe_emfx`` compatibility fallbacks
  for results missing vcov/weight columns, and ``twfe_decomposition`` with
  a single 2×2 comparison / missing cohort-period cells.
- ``honest_did.py`` — argument validation, the no-pre-period
  relative-magnitude branch, the R ``HonestDiD`` backend's validation and
  subprocess plumbing (driven through a stub ``Rscript`` on PATH; the
  numeric HonestDiD intervals themselves are NOT asserted here — parity
  is the job of tests/reference_parity/).
- ``callaway_santanna.py`` / ``pretrends.py`` — input-validation raises,
  user-supplied ``vcv_pre`` checking, degenerate pre-trend designs.
- ``bacon.py`` — always-treated comparisons, empty decompositions,
  unbalanced/non-monotone raises.
- ``gardner_2s.py`` — vce validation, never-reaching cohorts, single
  clusters, and the cluster-bootstrap event-study SEs.

Assertions check real behaviour: estimates near the DGP truth, SEs > 0,
specific exception types with message matches — never bare "it ran".
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    """Staggered DiD panel (dgp effect = 0.5) with helper columns."""
    df = sp.dgp_did(n_units=60, n_periods=8, staggered=True, seed=11).copy()
    rng = np.random.default_rng(7)
    df["xcov"] = rng.normal(size=len(df))
    df["const1"] = 1.0
    df["cl_one"] = 0
    return df


@pytest.fixture(scope="module")
def etwfe_fit(panel):
    return sp.etwfe(panel, y="y", group="unit", time="time", first_treat="first_treat")


@pytest.fixture(scope="module")
def cs_result():
    """Callaway–Sant'Anna fit on a +2 ATT staggered panel."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(90):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        for t in range(1, 11):
            y = fe + 0.3 * t + (2.0 if (g > 0 and t >= g) else 0.0)
            y += rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g})
    return sp.callaway_santanna(pd.DataFrame(rows), y="y", g="g", t="time", i="unit")


_KEEP = object()  # sentinel: keep the original detail frame


def _clone(result, *, detail=_KEEP, es_new=None, mi_pop=(), mi_set=None, **attrs):
    """Shallow-copy a CausalResult with independent detail/model_info.

    Used to emulate results produced by older StatsPAI versions (missing
    vcov matrices / weight columns) so the documented compatibility
    fallbacks can be exercised without mocking any numeric path.
    """
    r = copy.copy(result)
    if detail is _KEEP:
        r.detail = (
            result.detail.copy()
            if isinstance(result.detail, pd.DataFrame)
            else result.detail
        )
    else:
        r.detail = detail
    mi = dict(result.model_info)
    for k in mi_pop:
        mi.pop(k, None)
    if es_new is not None:
        mi["event_study"] = es_new
    if mi_set:
        mi.update(mi_set)
    r.model_info = mi
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


def _drdid_panel(n=240, seed=5):
    """Two-period panel; true panel ATT = 2.0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    ps = 1.0 / (1.0 + np.exp(-(0.2 + 0.4 * x)))
    d = rng.binomial(1, ps)
    rows = []
    for i in range(n):
        y0 = 1.0 + 0.5 * x[i] + 0.2 * d[i] + rng.normal()
        y1 = y0 + 1.0 + 2.0 * d[i] + rng.normal(scale=0.8)
        rows.append({"id": i, "post": 0, "treated": d[i], "x": x[i], "y": y0})
        rows.append({"id": i, "post": 1, "treated": d[i], "x": x[i], "y": y1})
    return pd.DataFrame(rows)


# ======================================================================
# wooldridge_did — OLS kernel fallbacks and cluster guard
# ======================================================================
def test_wooldridge_constant_control_uses_pinv_fallback(panel):
    # A constant control demeans to an exactly-zero column, so the QR
    # R-factor is singular and _ols_fit must fall back to the pinv path.
    # The estimate must agree with the no-control fit (the added column
    # carries no information).
    r_base = sp.wooldridge_did(
        panel, y="y", group="unit", time="time", first_treat="first_treat"
    )
    r_const = sp.wooldridge_did(
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        controls=["const1"],
    )
    assert np.isfinite(r_const.estimate)
    assert r_const.estimate == pytest.approx(r_base.estimate, abs=1e-8)


def test_wooldridge_single_cluster_raises(panel):
    with pytest.raises(DataInsufficient, match="at least\\s+two clusters"):
        sp.wooldridge_did(
            panel,
            y="y",
            group="unit",
            time="time",
            first_treat="first_treat",
            cluster="cl_one",
        )


def test_etwfe_xvar_all_never_treated_raises(panel):
    df = panel.copy()
    df["first_treat"] = np.nan
    with pytest.raises(DataInsufficient, match="No treated cohorts"):
        sp.etwfe(
            df,
            y="y",
            group="unit",
            time="time",
            first_treat="first_treat",
            xvar="xcov",
        )


# ======================================================================
# _etwfe_never_only — per-cohort never-treated-control ETWFE
# ======================================================================
def test_etwfe_never_only_recovers_effect(panel):
    from statspai.did.wooldridge_did import _etwfe_never_only

    r = _etwfe_never_only(
        panel, y="y", group="unit", time="time", first_treat="first_treat"
    )
    # dgp_did effect = 0.5; the per-cohort aggregation should land nearby.
    assert abs(r.estimate - 0.5) < 0.3
    assert r.se > 0
    assert 0.0 <= r.pvalue <= 1.0
    assert r.model_info["cgroup"] == "nevertreated"
    assert r.model_info["cohort_weighting"] == "cohort"
    n_cohorts = r.model_info["n_cohorts"]
    assert r.detail.shape[0] == n_cohorts
    assert {"cohort", "att", "se", "n_obs"} <= set(r.detail.columns)
    # aggregated n_obs is the sum over per-cohort regressions
    assert r.n_obs == int(r.detail["n_obs"].sum())
    # CI brackets the estimate
    assert r.ci[0] < r.estimate < r.ci[1]


def test_etwfe_never_only_xvar_branch(panel):
    from statspai.did.wooldridge_did import _etwfe_never_only

    r = _etwfe_never_only(
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        xvar=["xcov"],
    )
    assert np.isfinite(r.estimate)
    assert r.se > 0
    assert r.model_info["xvar"] == ["xcov"]


def test_etwfe_never_only_no_cohorts_raises(panel):
    from statspai.did.wooldridge_did import _etwfe_never_only

    df = panel.copy()
    df["first_treat"] = np.nan
    with pytest.raises(DataInsufficient, match="No treated cohorts"):
        _etwfe_never_only(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )


def test_etwfe_never_only_no_never_units_raises(panel):
    from statspai.did.wooldridge_did import _etwfe_never_only

    df = panel.loc[panel["first_treat"].notna()].reset_index(drop=True)
    with pytest.raises(DataInsufficient, match="never-treated"):
        _etwfe_never_only(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )


# ======================================================================
# drdid(id=...) — Sant'Anna–Zhao panel path validation + degenerate data
# ======================================================================
def test_drdid_id_not_a_column_raises():
    df = _drdid_panel(60)
    with pytest.raises(MethodIncompatibility, match="must be a column"):
        sp.drdid(df, y="y", group="treated", time="post", id="nope")


def test_drdid_id_requires_improved_method():
    df = _drdid_panel(60)
    with pytest.raises(MethodIncompatibility, match="method='imp'"):
        sp.drdid(df, y="y", group="treated", time="post", id="id", method="trad")


def test_drdid_id_missing_covariate_raises():
    df = _drdid_panel(60)
    with pytest.raises(MethodIncompatibility, match="Missing columns"):
        sp.drdid(df, y="y", group="treated", time="post", id="id", covariates=["ghost"])


def test_drdid_id_duplicate_unit_period_raises():
    df = _drdid_panel(60)
    dup = pd.concat([df, df.head(2)], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="at most one row"):
        sp.drdid(dup, y="y", group="treated", time="post", id="id")


def test_drdid_id_no_complete_pairs_raises():
    df = _drdid_panel(120)
    pre_only = df[(df["post"] == 0) & (df["id"] < 60)]
    post_only = df[(df["post"] == 1) & (df["id"] >= 60)]
    broken = pd.concat([pre_only, post_only], ignore_index=True)
    with pytest.raises(DataInsufficient, match="No complete pre/post"):
        sp.drdid(broken, y="y", group="treated", time="post", id="id")


def test_drdid_id_time_varying_group_raises():
    df = _drdid_panel(60)
    flip = (df["id"] == 0) & (df["post"] == 1)
    df.loc[flip, "treated"] = 1 - df.loc[flip, "treated"]
    with pytest.raises(MethodIncompatibility, match="time-invariant"):
        sp.drdid(df, y="y", group="treated", time="post", id="id")


def test_drdid_id_panel_without_covariates_recovers_att():
    # No covariates → intercept-only design (X_panel = ones); the panel
    # formula still identifies the ATT = 2.0 of the DGP.
    df = _drdid_panel(240, seed=5)
    r = sp.drdid(df, y="y", group="treated", time="post", id="id")
    assert abs(r.estimate - 2.0) < 0.35
    assert np.isfinite(r.se) and r.se > 0
    assert r.model_info["panel"] is True
    assert r.model_info["covariates"] == []


def test_drdid_id_panel_all_treated_raises():
    df = _drdid_panel(120)
    treated_rows = df[df["treated"] == 1]
    ctrl_pre_only = df[(df["treated"] == 0) & (df["post"] == 0)]
    # Controls exist only pre-period → dropped by the pre/post merge →
    # merged panel is all-treated.
    degenerate = pd.concat([treated_rows, ctrl_pre_only], ignore_index=True)
    with pytest.raises(DataInsufficient, match="treated and control"):
        sp.drdid(degenerate, y="y", group="treated", time="post", id="id")


def test_drdid_id_panel_too_few_controls_raises():
    df = _drdid_panel(120)
    rng = np.random.default_rng(0)
    df["x2"] = rng.normal(size=len(df))
    keep_ctrl = df.loc[df["treated"] == 0, "id"].unique()[:2]
    small = df[(df["treated"] == 1) | (df["id"].isin(keep_ctrl))]
    # 2 control units < 3 outcome-regression columns (const + 2 covariates)
    with pytest.raises(DataInsufficient, match="Not enough control units"):
        sp.drdid(
            small,
            y="y",
            group="treated",
            time="post",
            id="id",
            covariates=["x", "x2"],
        )


def test_drdid_id_panel_degenerate_covariate_pinv_fallback():
    # A zero-variance covariate makes the logistic Hessian exactly
    # singular; _logistic_coefficients must fall back to pinv and the
    # panel estimator must still recover the ATT.
    df = _drdid_panel(240, seed=5)
    df["zero"] = 0.0
    r = sp.drdid(
        df, y="y", group="treated", time="post", id="id", covariates=["x", "zero"]
    )
    assert abs(r.estimate - 2.0) < 0.35
    assert r.se > 0


def test_calibrated_pscore_i_weights():
    from statspai.did.wooldridge_did import _calibrated_pscore

    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(80), rng.normal(size=80)])
    D = (rng.random(80) < 0.4).astype(float)
    # Scaled (valid) weights are renormalised to mean one → same fit as
    # unit weights.
    ps_w, _ = _calibrated_pscore(X, D, i_weights=np.full(80, 2.0))
    ps_u, _ = _calibrated_pscore(X, D, i_weights=None)
    assert np.allclose(ps_w, ps_u, atol=1e-8)
    assert ps_w.min() > 0.0 and ps_w.max() < 1.0
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        _calibrated_pscore(X, D, i_weights=np.r_[-1.0, np.ones(79)])


# ======================================================================
# twfe_decomposition — degenerate designs
# ======================================================================
def test_twfe_decomposition_single_comparison_zero_se():
    # One treated cohort + never-treated ⇒ exactly one 2×2 comparison,
    # so the across-comparison SE is 0 and the p-value undefined.
    rng = np.random.default_rng(0)
    rows = []
    for u in range(20):
        g = 3 if u < 10 else np.nan
        for t in range(1, 5):
            y = 0.1 * u + 0.2 * t + (0.5 if (g == 3 and t >= 3) else 0.0)
            y += rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "y": y, "first_treat": g})
    r = sp.twfe_decomposition(
        pd.DataFrame(rows),
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
    )
    assert r.model_info["n_comparisons"] == 1
    assert r.se == 0.0
    assert np.isnan(r.pvalue)
    assert np.isfinite(r.estimate)


def test_twfe_decomposition_missing_cohort_period_cell():
    # Dropping every row of one cohort in one post period leaves an empty
    # (g, t) cell — the dCDH weight loop must skip it, not crash.
    df = sp.dgp_did(n_units=40, n_periods=6, staggered=True, seed=3)
    coh = sorted(df["first_treat"].dropna().unique())[0]
    t_max = df["time"].max()
    df = df[~((df["first_treat"] == coh) & (df["time"] == t_max))]
    r = sp.twfe_decomposition(
        df, y="y", group="unit", time="time", first_treat="first_treat"
    )
    assert np.isfinite(r.estimate)
    cells = r.model_info["dcdh_weights"]
    missing = (cells["cohort"] == int(coh)) & (cells["period"] == int(t_max))
    assert missing.sum() == 0


# ======================================================================
# etwfe_emfx — compatibility fallbacks for older / partial results
# ======================================================================
def test_emfx_non_dataframe_detail_raises(etwfe_fit):
    bad = _clone(etwfe_fit, detail=None)
    with pytest.raises(MethodIncompatibility, match="cohort-level detail"):
        sp.etwfe_emfx(bad, type="simple", weighting="cohort")


def test_emfx_simple_treated_without_event_vcov(etwfe_fit):
    # Without the stored event vcov the simple/treated headline must fall
    # back to the independent-coefficient SE, keeping the same estimate.
    r = _clone(etwfe_fit, mi_pop=("event_vcov",))
    e_fallback = sp.etwfe_emfx(r, type="simple", weighting="treated")
    e_full = sp.etwfe_emfx(etwfe_fit, type="simple", weighting="treated")
    assert e_fallback.estimate == pytest.approx(e_full.estimate, rel=1e-12)
    assert e_fallback.se > 0
    assert "independent-coefficient" in e_fallback.model_info["se_method"]
    assert "vcov" in e_full.model_info["se_method"]


def test_emfx_detail_without_att_column_raises(etwfe_fit):
    det = etwfe_fit.detail.rename(columns={"att": "zzz"})
    bad = _clone(etwfe_fit, detail=det)
    with pytest.raises(MethodIncompatibility, match="att"):
        sp.etwfe_emfx(bad, type="simple", weighting="cohort")


def test_emfx_missing_weight_column_raises(etwfe_fit):
    det = etwfe_fit.detail.drop(columns=["n_obs"])
    bad = _clone(etwfe_fit, detail=det)
    with pytest.raises(MethodIncompatibility, match="n_obs"):
        sp.etwfe_emfx(bad, type="simple", weighting="cohort")


def test_emfx_nonfinite_weights_fall_back_to_uniform(etwfe_fit):
    det = etwfe_fit.detail.copy()
    det["n_obs"] = np.nan
    r = _clone(etwfe_fit, detail=det)
    e = sp.etwfe_emfx(r, type="simple", weighting="cohort")
    # NaN weights → uniform weights → estimate is the plain cohort mean.
    assert e.estimate == pytest.approx(float(etwfe_fit.detail["att"].mean()), rel=1e-12)


def test_emfx_cohort_se_fallback_from_se_column(etwfe_fit):
    r = _clone(etwfe_fit, mi_pop=("cohort_vcov",))
    e = sp.etwfe_emfx(r, type="simple", weighting="cohort")
    assert e.se > 0
    assert "fallback" in e.model_info["se_method"]


def test_emfx_cohort_se_fallback_from_att_se_column(etwfe_fit):
    det = etwfe_fit.detail.rename(columns={"se": "att_se"})
    r = _clone(etwfe_fit, detail=det, mi_pop=("cohort_vcov",))
    e = sp.etwfe_emfx(r, type="simple", weighting="cohort")
    assert e.se > 0
    # matches the plain-'se' fallback numerically
    r2 = _clone(etwfe_fit, mi_pop=("cohort_vcov",))
    e2 = sp.etwfe_emfx(r2, type="simple", weighting="cohort")
    assert e.se == pytest.approx(e2.se, rel=1e-12)


def test_emfx_cohort_se_nan_when_no_se_columns(etwfe_fit):
    det = etwfe_fit.detail.drop(columns=["se"])
    r = _clone(etwfe_fit, detail=det, mi_pop=("cohort_vcov",))
    e = sp.etwfe_emfx(r, type="simple", weighting="cohort")
    assert np.isnan(e.se)
    assert np.isfinite(e.estimate)


def test_emfx_event_aggregation_without_event_vcov(etwfe_fit):
    r = _clone(etwfe_fit, mi_pop=("event_vcov",))
    e = sp.etwfe_emfx(r, type="event")
    assert "event_time" in e.detail.columns
    se = e.detail["se"].astype(float)
    assert np.isfinite(se).all() and (se > 0).all()


# ======================================================================
# honest_did — validation, no-pre branch, and the R backend plumbing
# ======================================================================
@pytest.mark.parametrize(
    "kwargs",
    [
        {"e": True},
        {"e": "x"},
        {"alpha": True},
        {"alpha": 1.5},
        {"method": 123},
    ],
)
def test_honest_did_argument_validation(cs_result, kwargs):
    with pytest.raises(MethodIncompatibility):
        sp.honest_did(cs_result, **kwargs)


def test_honest_did_bad_m_grid_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="m_grid"):
        sp.honest_did(cs_result, m_grid=["a"])
    with pytest.raises(DataInsufficient, match="at least one value"):
        sp.honest_did(cs_result, m_grid=[])


def test_honest_did_relative_magnitude_without_pre_periods(cs_result):
    # With no pre-treatment coefficients, max|δ_pre| = 0 and the bound
    # falls back to M̄ × SE. At M̄=0 the robust CI equals the plain CI.
    es = cs_result.model_info["event_study"]
    r = _clone(cs_result, es_new=es[es["relative_time"] >= 0].copy())
    tab = sp.honest_did(r, e=0, method="relative_magnitude", m_grid=[0.0, 1.0, 2.0])
    widths = (tab["ci_upper"] - tab["ci_lower"]).to_numpy()
    assert len(tab) == 3
    assert np.all(np.diff(widths) > 0)  # CI widens with M̄
    theta = float(es.loc[es["relative_time"] == 0, "att"].iloc[0])
    assert tab["ci_lower"].iloc[0] < theta < tab["ci_upper"].iloc[0]


def test_honest_did_event_study_missing_se_column(cs_result):
    es = cs_result.model_info["event_study"].drop(columns=["se"])
    r = _clone(cs_result, es_new=es)
    with pytest.raises(MethodIncompatibility, match="missing required columns"):
        sp.honest_did(r, e=0)


def test_breakdown_m_bad_method_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.breakdown_m(cs_result, method="bogus")


def test_honest_r_backend_bad_method_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.honest_did(cs_result, backend="honestdid", method="bogus")


def test_honest_r_backend_bad_solver_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="honestdid_method"):
        sp.honest_did(cs_result, backend="honestdid", honestdid_method="bogus")


def test_honest_r_backend_without_rscript_raises(cs_result, monkeypatch):
    # No resolvable Rscript → the backend must fail loudly with an
    # ImportError.  ``_find_rscript`` intentionally falls back to the
    # standard macOS install locations even when PATH is empty, so an
    # empty-PATH simulation is machine-dependent (it passed on CI but
    # failed on any runner with R installed at a standard location);
    # patch the resolver itself to model "no Rscript anywhere".
    import importlib

    # ``statspai.did.honest_did`` the *attribute* is the re-exported
    # function; fetch the module object explicitly.
    honest_mod = importlib.import_module("statspai.did.honest_did")
    monkeypatch.setattr(honest_mod, "_find_rscript", lambda: None)
    with pytest.raises(ImportError, match="Rscript"):
        sp.honest_did(cs_result, backend="honestdid")


@pytest.fixture()
def rscript_stub(tmp_path, monkeypatch):
    """Put a controllable ``Rscript`` stub first on PATH.

    Only the Python-side plumbing (validation, CSV/arg passing, JSON
    parsing, error propagation) is asserted against this stub; HonestDiD
    numeric parity is tested elsewhere with the real R package.
    """
    stub = tmp_path / "Rscript"

    def program(body):
        stub.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
        os.chmod(stub, 0o755)

    program(
        'echo \'[{"M":0.0,"ci_lower":1.0,"ci_upper":3.0},'
        '{"M":1.0,"ci_lower":-0.5,"ci_upper":4.0}]\''
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    return program


def test_honest_r_backend_validation_raises(cs_result, rscript_stub):
    es = cs_result.model_info["event_study"]
    # e not in the event-study table
    with pytest.raises(DataInsufficient, match="No event study estimate"):
        sp.honest_did(cs_result, e=99, backend="honestdid")
    # no pre-treatment periods
    r_nopre = _clone(cs_result, es_new=es[es["relative_time"] >= 0].copy())
    with pytest.raises(DataInsufficient, match="pre-treatment"):
        sp.honest_did(r_nopre, e=0, backend="honestdid")
    # e is a pre-treatment period
    e_pre = int(es["relative_time"].min())
    with pytest.raises(DataInsufficient, match="post-treatment"):
        sp.honest_did(cs_result, e=e_pre, backend="honestdid")


def test_honest_r_backend_parses_stub_output(cs_result, rscript_stub):
    tab = sp.honest_did(cs_result, e=0, backend="honestdid", m_grid=[0.0, 1.0])
    assert list(tab.columns) == ["M", "ci_lower", "ci_upper", "rejects_zero"]
    assert len(tab) == 2
    # rejects_zero is derived from the parsed CI, not echoed from R
    assert bool(tab["rejects_zero"].iloc[0]) is True  # [1.0, 3.0] excludes 0
    assert bool(tab["rejects_zero"].iloc[1]) is False  # [-0.5, 4.0] covers 0


def test_honest_r_backend_propagates_r_failure(cs_result, rscript_stub):
    rscript_stub('echo "boom from R" >&2\nexit 1')
    with pytest.raises(ConvergenceFailure, match="boom from R"):
        sp.honest_did(cs_result, e=0, backend="honestdid")


# ======================================================================
# callaway_santanna — argument validation
# ======================================================================
def test_cs_data_validation_raises():
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        sp.callaway_santanna([1, 2, 3], y="y", g="g", t="t", i="i")
    with pytest.raises(DataInsufficient, match="at least one row"):
        sp.callaway_santanna(pd.DataFrame(), y="y", g="g", t="t", i="i")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"y": 1},
        {"x": 5},
        {"estimator": 1},
        {"alpha": True},
        {"alpha": 1.5},
        {"anticipation": True},
        {"anticipation": "a"},
        {"panel": "yes"},
    ],
)
def test_cs_argument_validation(kwargs):
    df = sp.dgp_did(n_units=20, n_periods=4, staggered=True, seed=0)
    df["g"] = df["first_treat"].fillna(0)
    base = dict(y="y", g="g", t="time", i="unit")
    base.update(kwargs)
    with pytest.raises(MethodIncompatibility):
        sp.callaway_santanna(df, **base)


# ======================================================================
# pretrends — validation, vcv_pre handling, degenerate designs
# ======================================================================
@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": 1},
        {"type": "   "},
        {"alpha": True},
        {"alpha": "x"},
    ],
)
def test_pretrends_test_argument_validation(cs_result, kwargs):
    with pytest.raises(MethodIncompatibility):
        sp.pretrends_test(cs_result, **kwargs)


def test_pretrends_test_vcv_pre_validation(cs_result):
    es = cs_result.model_info["event_study"]
    K = int((es["relative_time"] < 0).sum())
    cases = [
        ("abc", "numeric"),
        (np.ones(3), "square"),
        (np.ones((K + 5, K + 5)), "incompatible shape"),
        (np.full((K, K), np.nan), "finite"),
    ]
    for vcv, msg in cases:
        r = _clone(cs_result, mi_set={"vcv_pre": vcv})
        with pytest.raises(MethodIncompatibility, match=msg):
            sp.pretrends_test(r)
    # A valid user-supplied K×K vcv is accepted and used.
    ok = sp.pretrends_test(_clone(cs_result, mi_set={"vcv_pre": np.eye(K) * 0.01}))
    assert 0.0 <= ok["pvalue"] <= 1.0
    assert ok["df"] == K


def test_pretrends_test_f_type_df_resid_fallback(cs_result):
    # No model_info df_resid and no n_obs attribute → conservative 1000.
    r = _clone(cs_result)
    if "n_obs" in r.__dict__:
        del r.__dict__["n_obs"]
    assert not hasattr(r, "n_obs")
    out = sp.pretrends_test(r, type="f")
    assert out["type"] == "f"
    assert ", 1000)" in out["stat_label"]
    assert 0.0 <= out["pvalue"] <= 1.0


def test_pretrends_nonnumeric_estimates_raise(cs_result):
    es = cs_result.model_info["event_study"].copy()
    es["att"] = es["att"].astype(object)
    es.iloc[0, es.columns.get_loc("att")] = "abc"
    with pytest.raises(MethodIncompatibility, match="numeric"):
        sp.pretrends_test(_clone(cs_result, es_new=es))


def test_pretrends_negative_se_raises(cs_result):
    es = cs_result.model_info["event_study"].copy()
    idx = es.index[es["relative_time"] < 0][0]
    es.loc[idx, "se"] = -0.1
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.pretrends_test(_clone(cs_result, es_new=es))


@pytest.mark.parametrize("bad_n_grid", [True, "x"])
def test_sensitivity_rr_bad_n_grid_raises(cs_result, bad_n_grid):
    with pytest.raises(MethodIncompatibility):
        sp.sensitivity_rr(cs_result, n_grid=bad_n_grid)


def test_sensitivity_rr_nonfinite_se_raises(cs_result):
    with pytest.raises(MethodIncompatibility, match="finite"):
        sp.sensitivity_rr(_clone(cs_result, se=np.nan))


def test_sensitivity_rr_collinear_pretrend_raises(cs_result):
    # Two pre-period rows at the same relative time → the WLS trend
    # design is rank-deficient and the slope is not identified.
    es = cs_result.model_info["event_study"]
    one_pre = es[es["relative_time"] == -2]
    es_dup = pd.concat(
        [one_pre, one_pre, es[es["relative_time"] >= 0]], ignore_index=True
    )
    with pytest.raises(NumericalInstability, match="not identified"):
        sp.sensitivity_rr(_clone(cs_result, es_new=es_dup))


# ======================================================================
# bacon — always-treated comparisons + degenerate/invalid panels
# ======================================================================
def _bacon_panel(cohorts, n_periods=7, seed=0, effect=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u, g in cohorts.items():
        for t in range(1, n_periods + 1):
            d = int(g != 99 and t >= g)
            rows.append(
                {
                    "unit": u,
                    "year": t,
                    "treated": d,
                    "y": u + 0.3 * t + effect * d + rng.normal(0, 0.4),
                }
            )
    return pd.DataFrame(rows)


def test_bacon_unbalanced_panel_raises():
    df = _bacon_panel({1: 3, 2: 3, 3: 99}).iloc[:-1]  # drop one obs
    with pytest.raises(MethodIncompatibility, match="Unbalanced"):
        sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="unit")


def test_bacon_nonmonotone_treatment_raises():
    df = _bacon_panel({1: 3, 2: 3, 3: 99})
    df.loc[(df["unit"] == 1) & (df["year"] == 5), "treated"] = 0  # switches off
    with pytest.raises(MethodIncompatibility, match="weakly increasing"):
        sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="unit")


def test_bacon_always_treated_comparisons():
    # Units treated from period 1 can only serve as (forbidden) controls.
    df = _bacon_panel({1: 1, 2: 1, 3: 4, 4: 4, 5: 99, 6: 99})
    out = sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="unit")
    types = set(out["decomposition"]["type"])
    assert "Later vs Always Treated" in types
    assert out["already_treated_control_weight_share"] > 0
    assert out["decomposition"]["weight"].sum() == pytest.approx(1.0)


def test_bacon_all_always_treated_empty_decomposition():
    df = _bacon_panel({1: 1, 2: 1, 3: 1, 4: 1})
    out = sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="unit")
    assert out["n_comparisons"] == 0
    assert out["decomposition"].empty
    # no treatment variation → TWFE beta degenerates to 0
    assert out["beta_twfe"] == 0.0
    assert out["weighted_sum"] == 0.0
    assert out["negative_weight_share"] == 0.0


# ======================================================================
# gardner_2s — vce validation, degenerate cohorts, clusters, bootstrap
# ======================================================================
def _gardner_panel(first_treat_at=5, n_units=30, n_periods=8, seed=42, effect=2.0):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_units)
    first = np.where(unit < n_units // 2, first_treat_at, 0)
    d = ((first > 0) & (time >= first)).astype(float)
    y = 0.5 * unit + 0.3 * time + effect * d + rng.normal(0, 0.5, unit.size)
    return pd.DataFrame({"y": y, "unit": unit, "time": time, "g": first})


def test_gardner_bad_vce_raises():
    df = _gardner_panel()
    with pytest.raises(ValueError, match="vce must be"):
        sp.gardner_did(
            df, y="y", group="unit", time="time", first_treat="g", vce="bogus"
        )


def test_gardner_event_study_cohort_never_reaches_treatment():
    # first_treat beyond the sample → relative time 0 has no support; the
    # horizon still includes 0 (with NaN estimate) and the aggregate ATT
    # is undefined rather than silently fabricated.
    df = _gardner_panel(first_treat_at=10, effect=0.0)  # sample ends at t=8
    r = sp.gardner_did(
        df,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        event_study=True,
        vce="none",
    )
    es = r.model_info["event_study"]
    assert "D_k+0" in es["coef"]
    assert np.isnan(es["coef"]["D_k+0"])
    assert np.isnan(r.estimate)


def test_gardner_single_cluster_paths():
    df = _gardner_panel()
    df["one"] = 1
    r = sp.gardner_did(
        df,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        cluster="one",
        vce="none",
    )
    assert abs(r.estimate - 2.0) < 0.4
    assert r.se > 0  # G=1 → dof correction falls back to 1.0, SE still finite
    r_es = sp.gardner_did(
        df,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        cluster="one",
        event_study=True,
        vce="none",
    )
    # single-cluster bins use the plain std-error-of-the-mean fallback
    assert np.isfinite(r_es.estimate)
    assert r_es.estimate == pytest.approx(r.estimate, abs=0.3)


def test_gardner_bootstrap_event_study():
    df = _gardner_panel()
    r = sp.gardner_did(
        df,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        event_study=True,
        vce="bootstrap",
        n_boot=25,
        boot_seed=1,
    )
    assert abs(r.estimate - 2.0) < 0.4
    assert np.isfinite(r.se) and r.se > 0
    es = r.model_info["event_study"]
    # bootstrap SEs must be present and positive for supported horizons
    post_ses = [
        es["se"][k]
        for k in es["se"]
        if int(k.split("k")[1]) >= 0 and np.isfinite(es["se"][k])
    ]
    assert len(post_ses) > 0 and all(s > 0 for s in post_ses)


def test_gardner_build_fe_design_default_levels():
    from statspai.did.gardner_2s import _build_fe_design

    unit = np.array([0, 0, 1, 1, 2, 2])
    time = np.array([0, 1, 0, 1, 0, 1])
    A_default, _, _ = _build_fe_design(unit, time, None)
    A_explicit, _, _ = _build_fe_design(
        unit, time, None, u_levels=np.unique(unit), t_levels=np.unique(time)
    )
    assert A_default.shape == A_explicit.shape
    assert np.allclose(A_default, A_explicit)
