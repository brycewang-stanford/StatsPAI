"""Option-scan round 7: ``smart/`` and ``workflow/`` orchestration.

Each test pins the correct behaviour of an option that the orchestration
layer accepted but did not forward, or a number a report printed that
disagreed with the result object it summarised. Ground truth is always the
library's own direct call on the same data (an identity) or a value that can
be recomputed by hand from the result.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.workflow._degradation import WorkflowDegradedWarning


@pytest.fixture(scope="module")
def obs() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 3))
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    y = 1 + 2 * d + X @ np.array([1.0, 0.5, -0.5]) + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2]})
    # 30 clusters of 10 rows with a shared shock, so clustering matters
    df["g"] = np.repeat(np.arange(30), 10)
    df["y"] = df["y"] + np.repeat(rng.normal(size=30), 10)
    return df


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    """Twelve covariates: more than any of the old truncation limits."""
    rng = np.random.default_rng(7)
    n = 400
    k = 12
    X = rng.normal(size=(n, k))
    d = rng.normal(size=n) + X[:, 11]  # confounded through the 12th covariate
    beta = np.zeros(k)
    beta[11] = 3.0
    y = 1 + 0.5 * d + X @ beta + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(k)])
    df["y"] = y
    df["d"] = d
    df["b"] = (d > 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
#  Oster delta: the plug-in delta (1.0) was reported as the critical delta
# --------------------------------------------------------------------------- #


def test_assumption_audit_oster_reports_delta_for_zero(obs):
    reg = sp.regress("y ~ d + x1 + x2 + x3", data=obs, robust="hc1")
    audit = sp.assumption_audit(reg, data=obs, verbose=False)
    (check,) = [c for c in audit.checks if "Oster" in c.test_name]
    ref = sp.oster_bounds(data=obs, y="y", treat="d", controls=["x1", "x2", "x3"])
    # the check is "would unobservables have to be stronger than observables
    # to drive beta to zero", i.e. delta* = delta_for_zero, not the plug-in
    # delta the bounds were evaluated at (1.0 by default)
    assert check.statistic == pytest.approx(ref.delta_for_zero, rel=1e-12)
    assert check.passed is (abs(ref.delta_for_zero) > 1)


def test_sensitivity_dashboard_unobservables_dimension_runs(obs):
    reg = sp.regress("y ~ d + x1 + x2 + x3", data=obs, robust="hc1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkflowDegradedWarning)
        dash = sp.sensitivity_dashboard(
            reg, data=obs, dimensions=["unobservables"], verbose=False
        )
    ref = sp.oster_bounds(data=obs, y="y", treat="d", controls=["x1", "x2", "x3"])
    (dim,) = dash.dimensions
    assert dim["dimension"].startswith("Unobservable confounders (Oster)")
    assert dim["stable"] is bool(abs(ref.delta_for_zero) > 1)
    lo, hi = sorted([float(ref.beta_adjusted), float(ref.beta_long)])
    assert dim["min_est"] == pytest.approx(lo, rel=1e-12)
    assert dim["max_est"] == pytest.approx(hi, rel=1e-12)
    assert f"{ref.delta_for_zero:.2f}" in dim["remedy"]


# --------------------------------------------------------------------------- #
#  alpha= accepted but ignored
# --------------------------------------------------------------------------- #


def test_sensitivity_dashboard_alpha_sets_the_baseline_interval(obs):
    reg = sp.regress("y ~ d + x1 + x2 + x3", data=obs, robust="hc1")
    dash = sp.sensitivity_dashboard(
        reg, data=obs, dimensions=["sample"], alpha=0.5, verbose=False
    )
    b, se = float(reg.params["d"]), float(reg.std_errors["d"])
    z = stats.norm.ppf(0.75)
    lo, hi = dash.baseline["ci"]
    assert lo == pytest.approx(b - z * se, rel=1e-12)
    assert hi == pytest.approx(b + z * se, rel=1e-12)


def test_sensitivity_dashboard_warns_on_unimplemented_dimension(obs):
    reg = sp.regress("y ~ d + x1 + x2 + x3", data=obs, robust="hc1")
    with pytest.warns(WorkflowDegradedWarning, match="functional_form"):
        sp.sensitivity_dashboard(
            reg,
            data=obs,
            dimensions=["sample", "functional_form"],
            verbose=False,
        )


def test_compare_estimators_alpha_drives_significance_agreement(obs):
    methods = ["ols", "matching", "ipw", "g_computation"]
    base = sp.compare_estimators(
        obs, y="y", treatment="d", covariates=["x1"], methods=methods
    )
    p = base.estimates_table["p_value"].to_numpy()
    # pick an alpha that splits the p-values differently from 0.05
    srt = np.sort(p)
    alpha = float((srt[0] + srt[1]) / 2) if srt[1] < 0.05 else 0.999
    res = sp.compare_estimators(
        obs, y="y", treatment="d", covariates=["x1"], methods=methods, alpha=alpha
    )
    p = res.estimates_table["p_value"].to_numpy()
    expect = float(np.mean((p < alpha) == (p[0] < alpha)))
    assert res.agreement["sig_agree"] == pytest.approx(expect)
    z = stats.norm.ppf(1 - alpha / 2)
    t = res.estimates_table
    assert np.allclose(t["ci_upper"] - t["estimate"], z * t["se"], rtol=1e-12)
    assert f"{100 * (1 - alpha):g}% CI" in res.summary()


def test_compare_estimators_single_method_is_not_called_robust(obs):
    res = sp.compare_estimators(
        obs, y="y", treatment="d", covariates=["x1"], methods=["ols"]
    )
    assert np.isnan(res.agreement["cv"])
    assert "ROBUST" not in res.summary()


def test_compare_estimators_records_failed_and_skipped_methods(obs):
    with pytest.warns(WorkflowDegradedWarning):
        res = sp.compare_estimators(
            obs,
            y="y",
            treatment="d",
            covariates=["x1"],
            methods=["ols", "panel_fe", "not_a_method"],
        )
    sections = " ".join(d["section"] for d in res.degradations)
    assert "panel_fe" in sections and "not_a_method" in sections
    assert len(res.estimates_table) == 1


def test_compare_estimators_uses_every_covariate(wide):
    cov = [f"c{j}" for j in range(12)]
    res = sp.compare_estimators(
        wide, y="y", treatment="d", covariates=cov, methods=["ols"]
    )
    direct = sp.regress("y ~ d + " + " + ".join(cov), data=wide, robust="hc1")
    est = res.estimates_table.loc[0, "estimate"]
    assert est == pytest.approx(float(direct.params["d"]), rel=1e-12)


# --------------------------------------------------------------------------- #
#  sp.recommend: run() must fit what the card describes
# --------------------------------------------------------------------------- #


def test_recommend_regress_card_runs_the_adjusted_formula(wide):
    cov = [f"c{j}" for j in range(12)]
    rr = sp.recommend(
        data=wide, y="y", treatment="d", covariates=cov, design="observational"
    )
    (i,) = [k for k, r in enumerate(rr.recommendations) if r["function"] == "regress"]
    fit = rr.run(i)
    direct = sp.regress("y ~ d + " + " + ".join(cov), data=wide, robust="hc1")
    assert float(fit.params["d"]) == pytest.approx(float(direct.params["d"]), rel=1e-12)
    # the confounder c11 drives the unadjusted slope far from 0.5
    assert abs(float(fit.params["d"]) - 0.5) < 0.2


def test_recommend_adjusting_cards_keep_every_control(wide):
    cov = [f"c{j}" for j in range(12)]
    rr = sp.recommend(
        data=wide, y="y", treatment="b", covariates=cov, design="observational"
    )
    for rec in rr.recommendations:
        params = rec.get("params") or {}
        if "covariates" in params:
            assert list(params["covariates"]) == cov, rec["function"]


# --------------------------------------------------------------------------- #
#  sp.causal(cluster=): the cluster column reached only the diagnostics
# --------------------------------------------------------------------------- #


def test_causal_cluster_reaches_the_estimator(obs):
    w = sp.causal(
        obs,
        y="y",
        treatment="d",
        covariates=["x1", "x2", "x3"],
        design="rct",
        cluster="g",
        auto_run=False,
    )
    r = w.estimate()
    direct = sp.regress("y ~ d + x1 + x2 + x3", data=obs, robust="hc1", cluster="g")
    assert float(r.std_errors["d"]) == pytest.approx(
        float(direct.std_errors["d"]), rel=1e-12
    )


def test_causal_cluster_on_staggered_did_matches_direct_call():
    df = sp.dgp_did(n_units=60, n_periods=6, seed=0)
    df["state"] = df["unit"] // 5
    w = sp.causal(
        df,
        y="y",
        treatment="treated",
        id="unit",
        time="time",
        design="did",
        cluster="state",
        auto_run=False,
    )
    r = w.estimate()
    df2 = df.copy()
    df2["g"] = df2["first_treat"].fillna(0).astype(int)
    direct = sp.callaway_santanna(
        df2, y="y", g="g", t="time", i="unit", clustervars="state"
    )
    assert r.estimate == pytest.approx(direct.estimate, rel=1e-12)
    assert r.se == pytest.approx(direct.se, rel=1e-12)


def test_causal_cluster_unsupported_estimator_is_recorded(obs):
    w = sp.causal(
        obs,
        y="y",
        treatment="d",
        covariates=["x1", "x2", "x3"],
        design="observational",
        cluster="g",
        auto_run=False,
    )
    with pytest.warns(WorkflowDegradedWarning, match="cluster"):
        w.estimate()
    assert any("cluster" in d["section"] for d in w.degradations)


def test_causal_comparison_table_is_reproducible(obs):
    tables = []
    for _ in range(2):
        w = sp.causal(
            obs,
            y="y",
            treatment="d",
            covariates=["x1", "x2", "x3"],
            design="observational",
            auto_run=False,
        )
        t = w.compare_estimators()
        tables.append(t[t["estimator"] == "CBPS"][["estimate", "se"]].to_numpy())
    np.testing.assert_array_equal(tables[0], tables[1])


def test_causal_fallback_regression_keeps_every_covariate(wide):
    cov = [f"c{j}" for j in range(12)]
    w = sp.causal(
        wide,
        y="y",
        treatment="d",
        covariates=cov,
        design="observational",
        auto_run=False,
    )
    r = w._fallback_estimate(error="forced")
    assert set(cov) <= set(map(str, r.params.index))


# --------------------------------------------------------------------------- #
#  Level labels
# --------------------------------------------------------------------------- #


def test_brief_labels_a_97_5_percent_interval(obs):
    r = sp.ipw(
        obs, y="y", treat="d", covariates=["x1", "x2", "x3"], seed=0, alpha=0.025
    )
    line = sp.brief(r)
    assert "97.5% CI" in line


class _StubResult:
    """Minimal causal-style result: what sp.brief reads, nothing else."""

    method = "stub"
    estimand = "ATE"
    estimate = 1.0
    se = 0.5
    pvalue = 0.04
    ci = (0.0, 2.0)

    def __init__(self, alpha: float, n_obs: object) -> None:
        self.alpha = alpha
        self.n_obs = n_obs


@pytest.mark.parametrize("alpha, label", [(0.005, "99.5% CI"), (0.1, "90% CI")])
def test_brief_level_label_is_not_rounded(alpha, label):
    line = sp.brief(_StubResult(alpha, 300))
    assert label in line


@pytest.mark.parametrize("n_obs", [np.int64(300), np.int32(300), np.float64(300)])
def test_brief_reports_numpy_n_obs(n_obs):
    assert "N=300" in sp.brief(_StubResult(0.05, n_obs))


def test_robustness_battery_ci_label_follows_result_alpha(obs):
    from statspai.workflow._robustness import run_robustness_battery

    r = sp.ipw(obs, y="y", treat="d", covariates=["x1", "x2", "x3"], seed=0, alpha=0.1)
    rep = run_robustness_battery(r, design="rct")
    (f,) = [f for f in rep.findings if f.name == "ci_width"]
    assert f.label == "90% CI width"
    assert f.value == pytest.approx(r.ci[1] - r.ci[0], rel=1e-12)
