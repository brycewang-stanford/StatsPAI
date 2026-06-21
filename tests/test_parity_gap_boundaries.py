"""Executable bounds for documented parity/convention gaps.

These tests intentionally read committed parity artifacts only. They make the
"known gap" language auditable without requiring R or Stata at test time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import statspai as sp

ROOT = Path(__file__).resolve().parents[1]
R_RESULTS = ROOT / "tests" / "r_parity" / "results"
STATA_RESULTS = ROOT / "tests" / "stata_parity" / "results"


def _payload(module: str, side: str) -> dict[str, Any]:
    if side == "Stata":
        path = STATA_RESULTS / f"{module}_Stata.json"
    else:
        path = R_RESULTS / f"{module}_{side}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _row(module: str, side: str, statistic: str) -> dict[str, Any]:
    for row in _payload(module, side)["rows"]:
        if row["statistic"] == statistic:
            return row
    raise AssertionError(f"{module}_{side}.json has no statistic {statistic!r}")


def _estimate(module: str, side: str, statistic: str) -> float:
    value = _row(module, side, statistic)["estimate"]
    assert isinstance(value, (int, float)) and math.isfinite(value)
    return float(value)


def _rel_gap(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _extra(module: str, side: str = "py") -> dict[str, Any]:
    extra = _payload(module, side).get("extra", {})
    assert isinstance(extra, dict)
    return extra


def _gap_description(module: str, gap: str) -> str:
    for row in sp.parity_gap_report(repo_root=ROOT, fmt="records"):
        if row["module_id"] == module and row["gap"] == gap:
            return row["description"]
    raise AssertionError(f"parity_gap_report has no {module}/{gap} row")


def test_parity_gap_report_has_actionable_stable_rows():
    rows = sp.parity_gap_report(repo_root=ROOT, fmt="records")
    assert len(rows) >= 4

    expected = {
        ("07_scm", "native_note"),
    }
    observed = {(row["module_id"], row["gap"]) for row in rows}
    assert expected <= observed
    assert ("06_rd", "bandwidth_parity_note") not in observed
    assert ("06_rd", "bandwidth_selector_gap") not in observed
    assert ("08_dml", "fold_split_note") not in observed
    assert ("11_psm", "se_convention_note") not in observed
    assert ("12_sdid", "native_note") not in observed
    assert ("12_sdid", "native_parity_note") not in observed
    assert ("13_causal_forest", "methodological_disclosure") not in observed
    assert ("13_causal_forest", "note") not in observed
    assert ("16_bjs", "stata_gap_note") not in observed
    assert ("16_bjs", "stata_never_treated_coding") not in observed
    assert ("21_honest_relmags", "stata_gap_note") not in observed
    assert ("18_augsynth", "native_note") not in observed
    assert ("18_augsynth", "native_parity_note") not in observed
    assert ("19_gsynth", "native_note") not in observed
    assert ("19_gsynth", "native_parity_note") not in observed
    assert ("29_panel_sfa", "stata_gap_note") not in observed
    assert ("29_panel_sfa", "stata_scale_reference") not in observed
    assert ("30_oaxaca", "decomposition_note") not in observed
    assert ("30_oaxaca", "decomposition_reference") not in observed
    assert ("31_dfl", "decomposition_note") not in observed
    assert ("31_dfl", "ddecompose_reference_mapping") not in observed
    assert ("32_rif", "quantile_convention") not in observed
    assert ("32_rif", "reference_backend_note") not in observed
    assert ("34_lp", "identification_note") not in observed
    assert ("34_lp", "r_identification_reference") not in observed
    assert ("34_lp", "stata_gap_note") not in observed
    assert ("34_lp", "stata_direct_ols_reference") not in observed
    assert ("35_panel", "hausman_note") not in observed
    assert ("35_panel", "hausman_parity_note") not in observed
    assert ("38_drdid", "stata_status") not in observed
    assert ("39_arima", "loglik_note") not in observed
    assert ("52_scm_unique", "certification_note") not in observed
    assert ("53_cr2", "cr3_convention") not in observed
    assert ("53_cr2", "cr3_reference") not in observed
    assert ("54_twoway_cluster", "convention") not in observed
    assert ("54_twoway_cluster", "r_reference") not in observed
    assert ("54_twoway_cluster", "stata_gap_note") not in observed
    assert ("54_twoway_cluster", "stata_cluster_reference") not in observed
    assert ("56_multiway_cluster", "stata_gap_note") not in observed
    assert ("56_multiway_cluster", "stata_cluster_reference") not in observed
    assert ("21_honest_relmags", "stata_precision_note") not in observed

    for row in rows:
        assert row["module_id"]
        assert row["description"].strip()
        assert row["priority"] in {"high", "medium", "low"}
        assert row["next_action"].strip()


def test_stata_skip_reasons_are_not_reported_as_missing_harnesses():
    rows = sp.parity_gap_report(repo_root=ROOT, fmt="records")
    missing = {
        row["module_id"] for row in rows if row["kind"] == "stata_harness_missing"
    }
    skipped = {
        "13_causal_forest",
        "18_augsynth",
        "19_gsynth",
    }
    assert skipped.isdisjoint(missing)

    no_canonical = {
        row["module_id"]: row
        for row in rows
        if row["kind"] == "no_canonical_stata_reference"
    }
    not_materialized = {
        row["module_id"]: row
        for row in rows
        if row["kind"] == "stata_bridge_not_materialized"
    }
    assert set(no_canonical) == set()
    assert set(not_materialized) == {
        "13_causal_forest",
        "18_augsynth",
        "19_gsynth",
    }
    assert "allsynth" in not_materialized["18_augsynth"]["description"]
    assert "K + 2" in not_materialized["18_augsynth"]["description"]
    assert "distinct" in not_materialized["18_augsynth"]["description"]
    assert "Stata 19" in not_materialized["13_causal_forest"]["description"]
    assert "fect_stata" in not_materialized["19_gsynth"]["description"]
    assert "0.679854" in not_materialized["19_gsynth"]["description"]
    assert (
        "force(two-way/unit/time/none)" in not_materialized["19_gsynth"]["description"]
    )


def test_dml_dfl_rif_and_cr2_have_materialized_stata_mata_bridges():
    rows = sp.parity_gap_report(repo_root=ROOT, fmt="records")
    no_canonical = {
        row["module_id"]
        for row in rows
        if row["kind"] == "no_canonical_stata_reference"
    }

    assert {"08_dml", "31_dfl", "32_rif", "53_cr2"}.isdisjoint(no_canonical)

    dml_extra = _payload("08_dml", "Stata")["extra"]
    dfl_extra = _payload("31_dfl", "Stata")["extra"]
    rif_extra = _payload("32_rif", "Stata")["extra"]
    cr2_extra = _payload("53_cr2", "Stata")["extra"]
    assert dml_extra["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
    assert "foldwise OLS residualization" in dml_extra["stata_algorithm"]
    assert dfl_extra["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
    assert "Newton-Raphson logit" in dfl_extra["stata_algorithm"]
    assert rif_extra["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
    assert "R stats::density binned Gaussian density" in rif_extra["stata_algorithm"]
    assert cr2_extra["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
    assert "(I-H_gg)^-1/2" in cr2_extra["stata_algorithm"]

    py_dml = _row("08_dml", "py", "theta_DML_PLR")
    stata_dml = _row("08_dml", "Stata", "theta_DML_PLR")
    for key in ("estimate", "se", "ci_lo", "ci_hi"):
        assert _rel_gap(float(py_dml[key]), float(stata_dml[key])) < 1e-12

    for stat in (
        "total_diff",
        "explained",
        "unexplained",
        "explained_educ",
        "explained_exper",
    ):
        assert (
            _rel_gap(
                _estimate("32_rif", "py", stat),
                _estimate("32_rif", "Stata", stat),
            )
            < 1e-12
        )

    for stat in ("gap", "composition", "structure", "stat_a", "stat_b", "stat_cf"):
        assert (
            _rel_gap(
                _estimate("31_dfl", "py", stat),
                _estimate("31_dfl", "Stata", stat),
            )
            < 1e-12
        )

    for stat in (
        "cr2_(Intercept)",
        "cr3_(Intercept)",
        "cr2_treat",
        "cr3_treat",
        "cr2_year",
        "cr3_year",
    ):
        py = _row("53_cr2", "py", stat)
        stata = _row("53_cr2", "Stata", stat)
        assert _rel_gap(float(py["estimate"]), float(stata["estimate"])) < 1e-8
        assert _rel_gap(float(py["se"]), float(stata["se"])) < 1e-6


def test_rd_default_cct_bandwidth_matches_r_and_stata():
    extra = _extra("06_rd")
    assert extra["bwselect"] == "cct"
    assert "bandwidth_selector_gap" not in extra
    assert "matches R/Stata rdrobust" in extra["bandwidth_parity_note"]

    py_h = _estimate("06_rd", "py", "default_bandwidth_h")
    r_h = _estimate("06_rd", "R", "default_bandwidth_h")
    stata_h = _estimate("06_rd", "Stata", "default_bandwidth_h")
    assert _rel_gap(py_h, r_h) < 2e-6
    assert _rel_gap(py_h, stata_h) < 2e-6

    for stat in ("default_conventional_est", "default_robust_est"):
        assert (
            _rel_gap(_estimate("06_rd", "py", stat), _estimate("06_rd", "R", stat))
            < 1e-9
        )
        assert (
            _rel_gap(_estimate("06_rd", "py", stat), _estimate("06_rd", "Stata", stat))
            < 1e-8
        )

    legacy_h = _estimate("06_rd", "py", "legacy_internal_mserd_bandwidth_h")
    assert 0.04 <= legacy_h <= 0.05

    for stat in (
        "forced_h0.042287_conventional_est",
        "forced_h0.042287_robust_est",
    ):
        assert (
            _rel_gap(_estimate("06_rd", "py", stat), _estimate("06_rd", "R", stat))
            < 1e-10
        )

    se_rel = _rel_gap(
        _row("06_rd", "py", "forced_h0.042287_conventional_est")["se"],
        _row("06_rd", "R", "forced_h0.042287_conventional_est")["se"],
    )
    assert se_rel <= 0.07


def test_dml_shared_explicit_folds_close_split_noise():
    extra = _extra("08_dml")
    assert extra["fold_source"] == "user"
    assert extra["fold_column"] == "fold_id"
    assert "explicit sample-splitting" in extra["fold_parity_note"]

    py_theta = _estimate("08_dml", "py", "theta_DML_PLR")
    r_theta = _estimate("08_dml", "R", "theta_DML_PLR")
    py_se = _row("08_dml", "py", "theta_DML_PLR")["se"]
    r_se = _row("08_dml", "R", "theta_DML_PLR")["se"]
    stata_theta = _estimate("08_dml", "Stata", "theta_DML_PLR")
    stata_se = _row("08_dml", "Stata", "theta_DML_PLR")["se"]
    assert py_theta == pytest.approx(r_theta, abs=1e-14)
    assert py_se == pytest.approx(r_se, abs=1e-14)
    assert py_theta == pytest.approx(stata_theta, abs=1e-14)
    assert py_se == pytest.approx(stata_se, abs=1e-14)


def test_scm_nonunique_row_is_bounded_by_unique_solution_counterpart():
    nonunique = _extra("07_scm")
    stata_extra = _extra("07_scm", "Stata")
    assert nonunique["validation_tier"] == "identification_dependent_native"
    assert nonunique["tier"] == "T4"
    assert "not unique" in nonunique["native_note"]
    assert "multi-start diagnostics" in nonunique["native_note"]
    assert nonunique["weight_solution_nonunique"] is True
    assert nonunique["solver_best_start"] == "regression"
    assert nonunique["solver_near_best_start_count"] >= 2
    assert nonunique["solver_near_best_weight_class_count"] >= 2
    assert nonunique["solver_near_best_weight_l1_max"] > 0.004
    assert "donor weights" in stata_extra["weight_note"]

    py_gap = _estimate("07_scm", "py", "avg_post_gap")
    r_gap = _estimate("07_scm", "R", "avg_post_gap")
    stata_gap = _estimate("07_scm", "Stata", "avg_post_gap")
    assert _rel_gap(py_gap, r_gap) <= 0.025
    assert _rel_gap(py_gap, stata_gap) <= 0.001

    py_rmse = _estimate("07_scm", "py", "pre_treatment_rmse")
    stata_rmse = _estimate("07_scm", "Stata", "pre_treatment_rmse")
    assert _rel_gap(py_rmse, stata_rmse) <= 1e-6

    weight_rows = {
        side: {
            row["statistic"]: float(row["estimate"])
            for row in _payload("07_scm", side)["rows"]
            if row["statistic"].startswith("weight_")
        }
        for side in ("py", "R", "Stata")
    }
    assert set(weight_rows["py"]) == set(weight_rows["R"]) == set(weight_rows["Stata"])
    assert (
        {stat for stat, weight in weight_rows["py"].items() if weight > 0.004}
        == {stat for stat, weight in weight_rows["Stata"].items() if weight > 0.004}
        == {
            "weight_Asturias",
            "weight_Cataluna",
            "weight_Madrid",
        }
    )

    py_stata_l1 = sum(
        abs(weight_rows["py"][stat] - weight_rows["Stata"][stat])
        for stat in weight_rows["py"]
    )
    r_stata_l1 = sum(
        abs(weight_rows["R"][stat] - weight_rows["Stata"][stat])
        for stat in weight_rows["R"]
    )
    assert py_stata_l1 <= 4e-4
    assert r_stata_l1 >= 0.03

    unique = _extra("52_scm_unique")
    unique_r = _payload("52_scm_unique", "R")["extra"]
    assert unique["true_gap"] == 2.0
    assert "Unique-solution counterpart" in unique["certification_note"]
    assert unique_r["custom.v"] == ["uniform fixed predictor weights"]
    assert unique_r["Sigf.ipop"] == [20]
    assert unique_r["Margin.ipop"] == [1e-12]
    assert abs(_estimate("52_scm_unique", "py", "avg_post_gap") - 2.0) <= 1e-6
    assert _estimate("52_scm_unique", "py", "pre_treatment_rmse") == 0.0
    assert abs(_estimate("52_scm_unique", "Stata", "avg_post_gap") - 2.0) <= 1e-12
    assert _estimate("52_scm_unique", "Stata", "pre_treatment_rmse") <= 1e-12
    assert (
        _rel_gap(
            _estimate("52_scm_unique", "py", "avg_post_gap"),
            _estimate("52_scm_unique", "R", "avg_post_gap"),
        )
        <= 1e-6
    )
    assert (
        _rel_gap(
            _estimate("52_scm_unique", "py", "avg_post_gap"),
            _estimate("52_scm_unique", "Stata", "avg_post_gap"),
        )
        <= 1e-6
    )


def test_did_aggregation_convention_rows_keep_point_parity():
    bjs_note = _extra("16_bjs")
    assert "parity_note" in bjs_note
    assert "stata_never_treated_coding" in bjs_note
    assert (
        _rel_gap(
            _estimate("16_bjs", "py", "att_bjs"),
            _estimate("16_bjs", "R", "att_bjs"),
        )
        < 1e-7
    )
    assert (
        _rel_gap(
            _estimate("16_bjs", "py", "att_bjs"),
            _estimate("16_bjs", "Stata", "att_bjs"),
        )
        < 1e-6
    )
    assert _payload("16_bjs", "py")["rows"][0]["se"] is None
    assert _estimate("16_bjs", "py", "se_cluster_if") > 0
    assert _estimate("16_bjs", "R", "se_didimputation") > 0
    assert _estimate("16_bjs", "Stata", "se_stata_did_imputation") > 0

    etwfe_note = _extra("17_etwfe")["aggregation_note"]
    assert "weighting='treated'" in etwfe_note
    assert "Point estimates and clustered delta-method SEs match" in etwfe_note
    assert (
        _rel_gap(
            _estimate("17_etwfe", "py", "att_etwfe"),
            _estimate("17_etwfe", "R", "att_etwfe"),
        )
        < 1e-12
    )


def test_causal_forest_gap_is_within_combined_monte_carlo_error():
    extra = _extra("13_causal_forest")
    assert "clean overlap" in extra["dgp"]
    assert "AIPW doubly-robust" in extra["note"]

    for stat in ("ate_causal_forest", "att_causal_forest"):
        py = _row("13_causal_forest", "py", stat)
        r = _row("13_causal_forest", "R", stat)
        combined_se = math.sqrt(float(py["se"]) ** 2 + float(r["se"]) ** 2)
        assert abs(float(py["estimate"]) - float(r["estimate"])) / combined_se <= 0.5
        assert _rel_gap(float(py["estimate"]), float(r["estimate"])) <= 0.03
