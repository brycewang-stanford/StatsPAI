"""R parity for the remaining synthetic-control tools (campaign round 2, ``synth_rest``).

References are frozen in ``_fixtures/synth_rest_R.json``, written by
``_generate_synth_rest_R.R`` from the CSVs of ``_generate_synth_rest_data.py``.
Package versions are in the fixture's ``meta``.

Conventions each number depends on
----------------------------------
* SCM sensitivity tools (``synth_loo`` / ``synth_time_placebo`` /
  ``synth_donor_sensitivity`` / ``synth_rmspe_filter``). Each fit is the
  no-covariate SCM with ``standardize_predictors=True``: StatsPAI then
  range-scales each pre-period row and uses equal V (the default up to
  1.38.0; the default now fits the raw rows). ``Synth::synth`` sd-scales the rows, so the generator passes
  ``custom.v = var_k / range_k^2`` to make the two QPs identical. The weights
  come from the same strictly convex QP, solved exactly by
  ``quadprog::solve.QP``. ``Synth``'s own ipop solve is also recorded and
  agrees at ipop's precision (<= 1e-5). Time placebos whose pre-period is
  shorter than the donor count have no unique weights and are not compared. The placebo p-values are
  ``SCtools::mspe.test`` on a ``tdf`` object built as
  ``generate.placebos`` builds it. SCtools cuts on the pre-period MSPE and
  drops the treated unit from the placebo donor pools, so the test uses
  ``metric="mspe", placebo_pool="exclude_treated"``.
* ``conformal_synth``: ``scinference`` (authors' package) with
  ``estimation_method="sc"`` and moving-block permutations. ``limSolve::lsei``
  fails at two extreme grid values (``IsError``) and scinference uses the
  infeasible weights anyway, so there StatsPAI is held to the generator's
  replica of scinference with quadprog as the solver. Everywhere else it is
  held to scinference itself.
* ``multi_outcome_synth``: ``augsynth_multiout(progfunc="None", scm=TRUE,
  fixedeff=FALSE)``, with ``combine_method`` ``"concat"`` / ``"avg"`` and
  ``synth_qp`` re-run at OSQP eps 1e-12.
* ``sequential_sdid``: per-cohort ATT(g) against ``synthdid_estimate`` on
  the same sub-panel.
* ``shift_share_political``: ``AER::ivreg`` + HC1, ``ShiftShareSE::ivreg_ss``
  (AKM / AKM0 / EHW) and ``bartik.weight::bw``; share balance is
  ``anova(lm(c1 ~ 1), lm(c1 ~ S))``.
* ``shift_share_political_panel``: ``fixest::feols`` with
  ``ssc(adj=FALSE, cluster.adj=FALSE)`` (CR0; two-way is CGM); ivreg_ss
  with FE dummies as controls and W in industry x period blocks;
  bartik.weight with FE dummies; the FE partial F is computed from feols
  RSS.

Tolerance: 1e-9 relative (observed 1e-13 to 1e-16) on every deterministic
quantity. It is looser only where stated, e.g. the AKM p-value: ShiftShareSE
uses ``2 * (1 - pnorm)``, whose cancellation at p ~ 1e-9 costs ~3e-8 relative;
StatsPAI uses ``sf``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth._core import solve_simplex_weights
from statspai.synth.sensitivity import _fit_scm_core

FX = Path(__file__).parent / "_fixtures"
R = json.loads((FX / "synth_rest_R.json").read_text(encoding="utf-8"))
RTOL = 1e-9


@pytest.fixture(scope="module")
def scm():
    return pd.read_csv(FX / "synth_rest_scm.csv")


BASE = dict(outcome="y", unit="unit", time="time", treated_unit=1, treatment_time=15)
# The R references for the sensitivity tools were built for the range-scaled
# no-covariate fit (see the module docstring). Since 1.39.0 that is no longer
# the default, so the convention the fixture pins is asked for by name.
SCALED = {**BASE, "standardize_predictors": True}


# --------------------------------------------------------------------------
# SCM sensitivity tools vs Synth + SCtools
# --------------------------------------------------------------------------
def test_treated_fit_matches_synth(scm):
    ref = R["sensitivity"]["treated"]
    fit = _fit_scm_core(scm, **SCALED)
    np.testing.assert_allclose(fit["weights"], ref["weights"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit["att"], ref["att"], rtol=RTOL)
    np.testing.assert_allclose(fit["pre_rmse"], ref["pre_rmse"], rtol=RTOL)
    # Synth's own (ipop) solve lands on the same point at its precision
    assert ref["ipop_max_abs_diff"] < 1e-5


def test_synth_loo_matches_synth(scm):
    ref = R["sensitivity"]["loo"]
    out = sp.synth_loo(scm, **SCALED)
    assert out["dropped_unit"].tolist() == [r["dropped_unit"] for r in ref]
    np.testing.assert_allclose(out["att"], [r["att"] for r in ref], rtol=RTOL)
    np.testing.assert_allclose(out["pre_rmse"], [r["pre_rmse"] for r in ref], rtol=RTOL)
    assert max(r["ipop_max_abs_diff"] for r in ref) < 1e-4


def test_synth_time_placebo_matches_synth(scm):
    ref = R["sensitivity"]["time_placebo"]
    out = sp.synth_time_placebo(scm, **SCALED)
    assert out["placebo_time"].tolist() == [r["placebo_time"] for r in ref]
    ident = [r["identified"] for r in ref]
    assert sum(ident) == 5
    np.testing.assert_allclose(
        out["att"][ident], [r["att"] for r in ref if r["identified"]], rtol=RTOL
    )
    # (non-identified placebo times, T0 < J: no unique weights, not compared)
    # identity: a single placebo post-period has no SE -> NaN, not p = 0
    last = out.iloc[-1]
    assert np.isnan(last["se"]) and np.isnan(last["pvalue"])


def test_synth_donor_sensitivity_matches_synth(scm):
    ref = R["sensitivity"]["donor_subsets"]
    out = sp.synth_donor_sensitivity(scm, **SCALED, k=6, n_samples=5, seed=7)
    # the numpy replay in the data generator reproduces the function's draws
    assert out["donors_used"].tolist() == [r["donors"] for r in ref]
    np.testing.assert_allclose(out["att"], [r["att"] for r in ref], rtol=RTOL)
    np.testing.assert_allclose(out["pre_rmse"], [r["pre_rmse"] for r in ref], rtol=RTOL)


@pytest.mark.parametrize("pool", ["exclude_treated", "include_treated"])
def test_synth_rmspe_filter_matches_sctools(scm, pool):
    ref = R["sensitivity"][f"placebo_{pool}"]
    out = sp.synth_rmspe_filter(
        scm,
        **SCALED,
        thresholds=[2.0, 5.0, 20.0, np.inf],
        metric="mspe",
        placebo_pool=pool,
    )
    pl = out.attrs["placebos"]
    assert pl["unit"].tolist() == ref["placebo_units"]
    np.testing.assert_allclose(pl["pre_rmspe"], ref["placebo_pre_rmspe"], rtol=RTOL)
    np.testing.assert_allclose(pl["ratio"], ref["placebo_ratio"], rtol=RTOL)
    # SCtools' MSPE ratios are the squares of the RMSPE ratios
    np.testing.assert_allclose(
        np.r_[pl["ratio"] ** 2, out.attrs["treated_ratio"] ** 2],
        ref["mspe_ratios"],
        rtol=RTOL,
    )
    for row, lim in zip(out.itertuples(), ref["limited"]):
        assert row.threshold == lim["mspe_limit"]
        # SCtools discards placebos with pre-MSPE >= limit x treated; StatsPAI
        # keeps <= limit x treated. No placebo sits on a boundary here.
        assert row.n_placebos == lim["n_placebos"]
        assert row.pvalue == pytest.approx(lim["p_val"], abs=1e-15)
    assert out["pvalue"].iloc[-1] == pytest.approx(ref["p_val"], abs=1e-15)


def test_rmspe_filter_rmspe_metric_is_sqrt_of_mspe(scm):
    a = sp.synth_rmspe_filter(scm, **BASE, thresholds=[4.0, 25.0], metric="mspe")
    b = sp.synth_rmspe_filter(scm, **BASE, thresholds=[2.0, 5.0], metric="rmspe")
    np.testing.assert_array_equal(a["pvalue"], b["pvalue"])
    np.testing.assert_array_equal(a["n_placebos"], b["n_placebos"])


# --------------------------------------------------------------------------
# conformal_synth vs scinference
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def conf_res(scm):
    c = R["conformal"]
    g = c["grid"]
    return sp.conformal_synth(
        scm, **BASE, grid_size=len(g), grid_range=(g[0], g[-1]), alpha=0.1
    )


def test_conformal_joint_pvalues(conf_res):
    c = R["conformal"]
    grid = conf_res.model_info["joint_pvalue_grid"]
    np.testing.assert_allclose(grid["tau0"], c["grid"], rtol=0, atol=1e-12)
    p = grid["pvalue"].to_numpy()
    np.testing.assert_array_equal(p, c["joint_p_quadprog"])
    ok = ~np.array(c["joint_lsei_error"])
    assert ok.sum() >= len(ok) - 2
    np.testing.assert_array_equal(p[ok], np.array(c["joint_p"])[ok])
    assert conf_res.pvalue == c["joint_p0"]
    np.testing.assert_allclose(conf_res.estimate, c["att"], rtol=RTOL)


def test_conformal_pointwise(conf_res):
    c = R["conformal"]
    det = conf_res.detail
    np.testing.assert_array_equal(det["pvalue"], c["pointwise_p0"])
    np.testing.assert_allclose(
        det["ci_lower"], [x["lb"] for x in c["pointwise_quadprog"]]
    )
    np.testing.assert_allclose(
        det["ci_upper"], [x["ub"] for x in c["pointwise_quadprog"]]
    )
    # where no lsei call failed for a period, scinference itself agrees
    for t, x in enumerate(c["pointwise_quadprog"]):
        if not any(x["lsei_error"]):
            assert det["ci_lower"][t] == c["pointwise_lb_alpha10"][t]
            assert det["ci_upper"][t] == c["pointwise_ub_alpha10"][t]


def test_conformal_unbounded_when_alpha_below_min_p(scm):
    r = sp.conformal_synth(scm, **BASE, grid_size=11, alpha=0.05)
    # T0 = 14: smallest pointwise p is 1/15 > 0.05
    assert np.isinf(r.detail["ci_lower"]).all() and np.isinf(r.detail["ci_upper"]).all()


# --------------------------------------------------------------------------
# multi_outcome_synth vs augsynth_multiout
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method,cm", [("concatenated", "concat"), ("averaged", "avg")])
def test_multi_outcome_matches_augsynth(method, cm):
    df = pd.read_csv(FX / "synth_rest_multi.csv")
    ref = R["multi_outcome"]["tight"][cm]
    r = sp.multi_outcome_synth(
        df, ["gdp", "emp", "inv"], "unit", "time", 1, 12, method=method, placebo=False
    )
    w = r.model_info["weights"]
    assert list(w) == ref["donors"]
    np.testing.assert_allclose(list(w.values()), ref["weights"], rtol=0, atol=1e-10)
    assert ref["outcomes"] == ["gdp", "emp", "inv"]
    np.testing.assert_allclose(r.detail["att"], ref["att_post_mean"], rtol=RTOL)
    # augsynth at its stock OSQP eps 1e-8 agrees to that tolerance
    np.testing.assert_allclose(
        list(w.values()), R["multi_outcome"]["stock"][cm]["weights"], atol=1e-6
    )


# --------------------------------------------------------------------------
# sequential_sdid: per-cohort blocks vs synthdid
# --------------------------------------------------------------------------
def test_sequential_sdid_cohorts_match_synthdid():
    df = pd.read_csv(FX / "synth_rest_stag.csv")
    r = sp.sequential_sdid(
        df, outcome="y", unit="unit", time="time", cohort="cohort", seed=1, n_reps=20
    )
    ref = R["sequential_sdid"]
    assert r.detail["cohort"].tolist() == [x["cohort"] for x in ref]
    np.testing.assert_allclose(r.detail["att"], [x["att"] for x in ref], rtol=1e-8)
    w = np.array([x["n_treated"] for x in ref], dtype=float)
    np.testing.assert_allclose(
        r.estimate, np.sum(w / w.sum() * np.array([x["att"] for x in ref])), rtol=1e-8
    )


# --------------------------------------------------------------------------
# shift-share
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ss():
    return dict(
        pan=pd.read_csv(FX / "synth_rest_ss_panel.csv"),
        unb=pd.read_csv(FX / "synth_rest_ss_panel_unbal.csv"),
        sh=pd.read_csv(FX / "synth_rest_ss_shares.csv").set_index("unit"),
        sk=pd.read_csv(FX / "synth_rest_ss_shocks.csv").set_index("time"),
    )


def test_shift_share_political_matches_R(ss):
    c = R["shift_share_cs"]
    pan = ss["pan"]
    if True:
        r = sp.shift_share_political(
            pan[pan.time.isin([1, 5])],
            unit="unit",
            time="time",
            outcome="y",
            endog="x",
            shares=ss["sh"],
            shocks=ss["sk"].loc[5] - ss["sk"].loc[1],
            covariates=["c1"],
            leave_one_out=False,
        )
    np.testing.assert_allclose(r.estimate, c["beta"], rtol=RTOL)
    np.testing.assert_allclose(r.se, c["se_hc1"], rtol=RTOL)
    d = r.diagnostics
    np.testing.assert_allclose(d["akm_se"], c["ivreg_ss_se"]["AKM"], rtol=RTOL)
    np.testing.assert_allclose(
        d["akm_ci"], [c["ivreg_ss_ci_l"]["AKM"], c["ivreg_ss_ci_r"]["AKM"]], rtol=RTOL
    )
    np.testing.assert_allclose(d["akm_pvalue"], c["ivreg_ss_p"]["AKM"], rtol=1e-7)
    np.testing.assert_allclose(d["akm0_pvalue"], c["ivreg_ss_p"]["AKM0"], rtol=RTOL)
    assert c["ivreg_ss_ci_l"]["AKM0"] == "-Inf" and d["akm0_ci"] == (-np.inf, np.inf)
    np.testing.assert_allclose(d["ehw_se_no_ssc"], c["ivreg_ss_se"]["EHW"], rtol=RTOL)
    rt = r.rotemberg_top.set_index("industry").loc[c["rotemberg_industry"]]
    np.testing.assert_allclose(
        rt["rotemberg_weight"], c["rotemberg_alpha"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(rt["beta_k"], c["rotemberg_beta"], rtol=RTOL)
    b = r.share_balance.iloc[0]
    np.testing.assert_allclose(
        [b.F, b.pvalue, b.R2_on_shares],
        [c["balance_F"], c["balance_p"], c["balance_R2"]],
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "key,fe,cluster,data",
    [
        ("twoway_fe_unit", "two-way", "unit", "pan"),
        ("twoway_fe_time", "two-way", "time", "pan"),
        ("twoway_fe_twoway", "two-way", "twoway", "pan"),
        ("unit_fe_unit", "unit", "unit", "pan"),
        ("time_fe_unit", "time", "unit", "pan"),
        ("unbal_twoway_fe_unit", "two-way", "unit", "unb"),
    ],
)
def test_shift_share_panel_matches_fixest(ss, key, fe, cluster, data):
    f = R["shift_share_panel"]["fixest"][key]
    r = sp.shift_share_political_panel(
        ss[data],
        unit="unit",
        time="time",
        outcome="y",
        endog="x",
        shares=ss["sh"],
        shocks=ss["sk"],
        cluster=cluster,
        fe=fe,
    )
    np.testing.assert_allclose(r.estimate, f["beta"], rtol=RTOL)
    np.testing.assert_allclose(r.se, f["se"], rtol=RTOL)
    if cluster == "unit":
        np.testing.assert_allclose(r.diagnostics["first_stage_F"], f["fs_F"], rtol=RTOL)


def test_shift_share_panel_akm_rotemberg_per_period(ss):
    P = R["shift_share_panel"]
    r = sp.shift_share_political_panel(
        ss["pan"],
        unit="unit",
        time="time",
        outcome="y",
        endog="x",
        shares=ss["sh"],
        shocks=ss["sk"],
        cluster="shock",
    )
    np.testing.assert_allclose(r.estimate, P["ivreg_ss_beta"], rtol=RTOL)
    np.testing.assert_allclose(r.se, P["ivreg_ss_se"]["AKM"], rtol=RTOL)
    np.testing.assert_allclose(
        r.diagnostics["akm0_ci"],
        [P["ivreg_ss_ci_l"]["AKM0"], P["ivreg_ss_ci_r"]["AKM0"]],
        rtol=RTOL,
    )
    rp = r.rotemberg_panel.set_index("industry").loc[P["rotemberg_industry"]]
    np.testing.assert_allclose(
        rp["rotemberg_weight"], P["rotemberg_alpha"], rtol=0, atol=1e-12
    )
    pp = r.per_period
    np.testing.assert_allclose(
        pp["estimate"], [x["estimate"] for x in P["per_period"]], rtol=RTOL
    )
    np.testing.assert_allclose(
        pp["se"], [x["se_hc0"] for x in P["per_period"]], rtol=RTOL
    )


# --------------------------------------------------------------------------
# Reference-free identities for the class-6 tools
# --------------------------------------------------------------------------
def _copy_panel():
    """Treated unit is an exact copy of donor 'c2' before treatment."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(6):
        h = 0.03 + 0.012 * c
        for m in range(1, 21):
            rows.append((f"c{c}", m, float(np.exp(-h * m * (1 + 0.05 * rng.normal())))))
    df = pd.DataFrame(rows, columns=["u", "m", "s"])
    tr = df[df.u == "c2"].copy()
    tr["u"] = "T"
    tr.loc[tr.m >= 11, "s"] = tr.loc[tr.m >= 11, "s"] ** 0.8  # effect after 11
    df = pd.concat([df, tr], ignore_index=True)
    df["is_t"] = df.u == "T"
    return df


def test_synth_survival_exact_copy_identity():
    df = _copy_panel()
    r = sp.synth_survival(
        df, unit="u", time="m", survival="s", treated="is_t", treat_time=11
    )
    assert r.weights["c2"] == pytest.approx(1.0, abs=1e-9)
    assert np.max(np.abs(r.gap[r.time_grid < 11])) < 1e-9
    # band inverts gap - effect ~ placebo gaps: [gap - q_hi, gap - q_lo]
    q_lo = np.quantile(r.placebo_gaps, 0.025, axis=0)
    q_hi = np.quantile(r.placebo_gaps, 0.975, axis=0)
    np.testing.assert_allclose(r.ci_low, r.gap - q_hi)
    np.testing.assert_allclose(r.ci_high, r.gap - q_lo)


def test_synth_power_null_and_attainable_level(scm):
    # 9 placebos: p >= 1/10, so at alpha = 0.05 nothing can be rejected
    p = sp.synth_power(
        scm, **BASE, effect_sizes=[0.0, 50.0], n_simulations=5, alpha=0.05, seed=0
    )
    assert (p["power"] == 0).all()
    # at alpha = 0.1 a huge effect is always detected, a zero effect never
    # beats every placebo here
    p = sp.synth_power(
        scm, **BASE, effect_sizes=[0.0, 50.0], n_simulations=5, alpha=0.1, seed=0
    )
    assert p["power"].tolist()[1] == 1.0


def test_experimental_design_ranking_identity(scm):
    res = sp.synth_experimental_design(
        scm,
        unit="unit",
        time="time",
        outcome="y",
        k=3,
        pre_period=(1, 14),
        random_state=0,
    )
    wide = scm[scm.time <= 14].pivot(index="unit", columns="time", values="y")
    direct = {}
    for u in wide.index:
        X = wide.drop(index=u).to_numpy().T
        w = solve_simplex_weights(wide.loc[u].to_numpy(), X)
        direct[u] = float(np.mean((wide.loc[u].to_numpy() - X @ w) ** 2))
    got = res.ranking.set_index("unit")["pre_mspe"]
    np.testing.assert_allclose(
        [got[u] for u in direct], list(direct.values()), rtol=1e-10
    )
    assert res.selected == sorted(direct, key=direct.get)[:3]
    assert res.method == "loo_sc_fit_ranking"
