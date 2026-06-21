"""Fast metadata tests for the cross-language parity harness.

These tests do not run R or Stata. They make the already-materialized
parity artifacts a pytest-enforced contract so the published tables,
registered tolerances, and validation APIs cannot silently drift apart.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
R_PARITY = ROOT / "tests" / "r_parity"
R_RESULTS = R_PARITY / "results"
STATA_RESULTS = ROOT / "tests" / "stata_parity" / "results"
STATA_PARITY = ROOT / "tests" / "stata_parity"
FIXTURE_LOCK_SCRIPT = ROOT / "scripts" / "tier_a_fixture_lock.py"


def _load_compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "statspai_r_parity_compare_for_tests",
        R_PARITY / "compare.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixture_lock() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "statspai_tier_a_fixture_lock_for_tests",
        FIXTURE_LOCK_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _module_stems(root: Path, suffix: str) -> set[str]:
    return {path.stem[: -len(suffix)] for path in root.glob(f"*{suffix}.json")}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _readme_module_numbers(path: Path) -> set[str]:
    numbers: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d{2})\s*\|", line)
        if match:
            numbers.add(match.group(1))
    return numbers


def _headline_rows(
    compare: ModuleType,
    module: str,
) -> tuple[dict[str, Any], list[Any]]:
    cfg = compare.HEADLINE[module]
    diffs = compare.collect(module)
    filtered = [diff for diff in diffs if cfg["headline_filter"](diff)]
    return cfg, filtered or diffs


def _snapshot_tol_value(text: str) -> float:
    match = re.fullmatch(r"\\\(10\^\{(-?\d+)\}\\\)", text)
    if match:
        return 10.0 ** int(match.group(1))
    match = re.fullmatch(r"\\\((\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}\\\)", text)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    return float(text)


def test_parity_artifact_inventory_has_explicit_contracts():
    compare = _load_compare()
    py_modules = _module_stems(R_RESULTS, "_py")
    r_modules = _module_stems(R_RESULTS, "_R")
    stata_modules = _module_stems(STATA_RESULTS, "_Stata")

    assert len(py_modules) >= 64
    assert len(py_modules & r_modules) >= 64
    assert len(stata_modules) >= 61
    assert py_modules == set(compare.TOLERANCES)
    assert py_modules == set(compare.HEADLINE)
    # Every Track A module now carries an R reference (50_xtabond gained its
    # plm::pgmm golden artifact, closing the last R-side gap).
    assert py_modules - r_modules == set()
    assert set(compare.STATA_SKIP_REASON) == py_modules - stata_modules
    assert set(compare.STATA_HEADLINE_GAP_EXCEPTIONS) <= stata_modules

    for module in sorted(py_modules & r_modules):
        assert compare.collect(module), f"{module} has no joined py/R rows"


def test_track_a_snapshot_tolerances_match_registered_budget():
    compare = _load_compare()
    special_rows = {"07_scm"}  # displays the T4 reference-disagreement threshold.
    for spec in compare.TRACK_A_SNAPSHOT_ROWS:
        module = spec["module"]
        if module in special_rows:
            continue
        tol = compare.TOLERANCES[module]
        registered = tol.get("rel_est", tol.get("abs_est"))
        assert registered is not None, module
        assert _snapshot_tol_value(spec["tol"]) == pytest.approx(registered), module


def test_strictness_tier_breakdown_matches_current_artifacts():
    compare = _load_compare()
    rendered_modules = [
        module
        for module in sorted(_module_stems(R_RESULTS, "_py"))
        if compare.collect(module)
    ]

    assert compare.tier_breakdown(rendered_modules) == {
        "machine": 57,
        "iterative": 5,
        "moderate": 1,
        "methodological": 1,
    }


def test_machine_level_promotions_have_headline_headroom():
    compare = _load_compare()
    promoted = {
        "04_csdid",
        "05_sunab",
        "06_rd",
        "14_ols_cluster",
        "20_bacon",
        "24_coxph",
        "25_lmm",
        "27_glmm_aghq",
        "28_frontier",
        "30_oaxaca",
        "31_dfl",
        "32_rif",
        "33_var",
        "35_panel",
        "37_ppmlhdfe",
        "38_drdid",
        "39_arima",
        "40_qreg",
        "41_tobit",
        "42_nbreg",
        "43_heckman",
        "44_mlogit",
        "45_ologit",
        "46_clogit",
        "47_ppmlhdfe_3fe",
        "48_probit",
        "49_oprobit",
        "51_newey",
        "52_scm_unique",
    }

    for module in sorted(promoted):
        assert compare.TOLERANCES[module]["rel_est"] == 1e-6
        assert compare.tolerance_tier(module) == "machine"
        cfg, rows = _headline_rows(compare, module)
        assert cfg["metric"] == "rel_est", module
        values = [
            value
            for row in rows
            for value in (row.rel_est, row.rel_est_st)
            if value is not None and math.isfinite(value)
        ]
        assert values, module
        assert max(values) <= 1e-6, module


def test_tier_a_parity_fixture_lock_is_current():
    lock = _load_fixture_lock()
    ok, diff = lock.verify()
    assert ok, (
        "Tier A parity fixture lock is stale. Review the fixture change, then "
        "refresh the committed lock with "
        "`python scripts/tier_a_fixture_lock.py --write`.\n" + diff
    )


def test_parity_json_rows_keep_the_joinable_schema():
    artifacts = [
        *R_RESULTS.glob("*_py.json"),
        *R_RESULTS.glob("*_R.json"),
        *STATA_RESULTS.glob("*_Stata.json"),
    ]
    assert artifacts

    for path in artifacts:
        payload = _read_json(path)
        if path.name.endswith("_py.json"):
            module, side = path.stem[:-3], "py"
        elif path.name.endswith("_R.json"):
            module, side = path.stem[:-2], "R"
        else:
            module, side = path.stem[:-6], "Stata"

        assert payload["module"] == module
        assert payload["side"] == side
        assert payload["rows"], f"{path.name} has no parity rows"

        seen: set[str] = set()
        for row in payload["rows"]:
            assert row["module"] == module
            assert row["side"] == side
            assert isinstance(row["statistic"], str) and row["statistic"]
            assert row["statistic"] not in seen
            seen.add(row["statistic"])

            for key in ("estimate", "se", "ci_lo", "ci_hi"):
                value = row.get(key)
                assert value is None or isinstance(value, (int, float))
                assert not isinstance(value, float) or math.isfinite(value)


def test_frontier_efficiency_rows_do_not_mix_jlms_and_bc_definitions():
    """R/sfaR and Stata/frontier name technical-efficiency predictors
    differently; the parity artifacts must not compare them under one
    ambiguous statistic name.
    """
    py_stats = {
        row["statistic"]
        for row in _read_json(R_RESULTS / "28_frontier_py.json")["rows"]
    }
    r_stats = {
        row["statistic"] for row in _read_json(R_RESULTS / "28_frontier_R.json")["rows"]
    }
    stata_stats = {
        row["statistic"]
        for row in _read_json(STATA_RESULTS / "28_frontier_Stata.json")["rows"]
    }

    assert "mean_efficiency" not in py_stats | r_stats | stata_stats
    assert "mean_efficiency_jlms" in py_stats & r_stats
    assert "mean_efficiency_bc" in py_stats & stata_stats


def test_psm_control_count_rows_use_explicit_support_definitions():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "11_psm_py.json")
    r_payload = _read_json(R_RESULTS / "11_psm_R.json")
    stata_payload = _read_json(STATA_RESULTS / "11_psm_Stata.json")
    rows = {row.statistic: row for row in compare.collect("11_psm")}
    py_rows = {row["statistic"]: row for row in py_payload["rows"]}
    r_rows = {row["statistic"]: row for row in r_payload["rows"]}
    stata_rows = {row["statistic"]: row for row in stata_payload["rows"]}
    py_stats = set(py_rows)
    r_stats = set(r_rows)
    stata_stats = set(stata_rows)

    assert "n_control" not in py_stats | r_stats | stata_stats
    assert "n_control_full" in py_stats & r_stats & stata_stats
    assert "n_control_matched" in r_stats
    assert py_rows["att_psm"]["se"] is None
    assert r_rows["att_psm"]["se"] is None
    assert stata_rows["att_psm"]["se"] is None
    assert "se_pair_effect" in py_stats
    assert "se_matchit_lm" in r_stats
    assert "se_teffects_ai" in stata_stats
    assert "se_reference" in py_payload["extra"]
    assert "se_convention_note" not in py_payload["extra"]
    assert compare.TOLERANCES["11_psm"]["rel_est"] == 1e-6
    assert compare.tolerance_tier("11_psm") == "machine"
    assert rows["att_psm"].rel_est < 1e-12
    assert rows["att_psm"].rel_est_st < 1e-12


def test_mediation_point_effects_are_exact_with_se_convention_guard():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "36_mediation_py.json")
    r_payload = _read_json(R_RESULTS / "36_mediation_R.json")
    stata_payload = _read_json(STATA_RESULTS / "36_mediation_Stata.json")
    rows = {row.statistic: row for row in compare.collect("36_mediation")}

    assert compare.TOLERANCES["36_mediation"] == {
        "rel_est": 1e-6,
        "rel_se": 0.10,
    }
    assert compare.tolerance_tier("36_mediation") == "machine"
    assert "bootstrap" in py_payload["extra"]["se_note"]
    assert r_payload["extra"]["method"] == ["mediation::mediate"]
    assert stata_payload["extra"]["se_method"] == "delta"
    assert "SEs differ" in stata_payload["extra"]["note"]

    for statistic in ("acme", "ade", "total_effect", "prop_mediated"):
        assert rows[statistic].rel_est < 1e-12
        assert (rows[statistic].rel_est_st or 0.0) < 1e-12

    assert rows["acme"].rel_se < 0.10
    assert rows["acme"].rel_se_st < 0.10
    assert rows["ade"].rel_se < 0.10
    assert rows["ade"].rel_se_st < 0.10


def test_glmm_parity_uses_tight_optimizer_solution_with_se_convention_guard():
    compare = _load_compare()

    expected = {
        "26_glmm_logit": 2e-4,
    }
    for module, rel_tol in expected.items():
        py_payload = _read_json(R_RESULTS / f"{module}_py.json")
        rows = {row.statistic: row for row in compare.collect(module)}

        assert compare.TOLERANCES[module]["rel_est"] == rel_tol
        assert compare.TOLERANCES[module]["rel_se"] == 5e-2
        assert compare.tolerance_tier(module) == "iterative"
        assert py_payload["extra"]["optimizer_tol"] == 1e-8
        assert (
            "tracks lme4/Stata fixed effects" in py_payload["extra"]["optimizer_note"]
        )

        for statistic in ("beta_intercept", "beta_x1"):
            assert rows[statistic].rel_est < rel_tol
            assert rows[statistic].rel_est_st < rel_tol
            assert rows[statistic].rel_se < 5e-2
            assert rows[statistic].rel_se_st < 5e-2

    py_payload = _read_json(R_RESULTS / "27_glmm_aghq_py.json")
    rows = {row.statistic: row for row in compare.collect("27_glmm_aghq")}

    assert compare.TOLERANCES["27_glmm_aghq"]["rel_est"] == 1e-6
    assert compare.TOLERANCES["27_glmm_aghq"]["rel_se"] == 5e-2
    assert compare.tolerance_tier("27_glmm_aghq") == "machine"
    assert py_payload["extra"]["optimizer_tol"] == 1e-12
    assert py_payload["extra"]["optimizer_maxiter"] == 5000
    assert "AGHQ reference optimiser budget" in py_payload["extra"]["optimizer_note"]

    for statistic in ("beta_intercept", "beta_x1"):
        assert rows[statistic].rel_est < 1e-6
        assert rows[statistic].rel_est_st < 1e-6
        assert rows[statistic].rel_se < 5e-2
        assert rows[statistic].rel_se_st < 5e-2


def test_oaxaca_rows_do_not_mix_twofold_and_threefold_definitions():
    py_payload = _read_json(R_RESULTS / "30_oaxaca_py.json")
    py_stats = {row["statistic"] for row in py_payload["rows"]}
    r_stats = {
        row["statistic"] for row in _read_json(R_RESULTS / "30_oaxaca_R.json")["rows"]
    }
    stata_stats = {
        row["statistic"]
        for row in _read_json(STATA_RESULTS / "30_oaxaca_Stata.json")["rows"]
    }

    assert "explained" not in py_stats | r_stats | stata_stats
    assert "explained_twofold" in py_stats & r_stats & stata_stats
    assert "explained_threefold_endowments" in r_stats & stata_stats
    assert "interaction_threefold" in r_stats & stata_stats
    assert "explained_educ" not in py_stats | r_stats
    assert "decomposition_reference" in py_payload["extra"]
    assert "decomposition_note" not in py_payload["extra"]


def test_dfl_uses_ddecompose_reference_mapping_for_component_parity():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "31_dfl_py.json")
    stata_payload = _read_json(STATA_RESULTS / "31_dfl_Stata.json")
    py_stats = {row["statistic"] for row in py_payload["rows"]}
    rows = {row.statistic: row for row in compare.collect("31_dfl")}

    assert "31_dfl" not in compare.STATA_SKIP_REASON
    assert (
        stata_payload["extra"]["stata_bridge_status"]
        == "audited Stata/Mata algorithm bridge"
    )
    assert "Newton-Raphson logit" in stata_payload["extra"]["stata_algorithm"]
    assert {"gap", "composition", "structure"} <= py_stats
    assert py_payload["extra"]["reference"] == 1
    assert "decomposition_note" not in py_payload["extra"]
    assert "ddecompose_reference_mapping" in py_payload["extra"]

    for statistic in ("gap", "composition", "structure"):
        assert rows[statistic].rel_est < 1e-8
        assert (rows[statistic].rel_est_st or 0.0) < 1e-8

    cfg, headline_rows = _headline_rows(compare, "31_dfl")
    assert cfg["metric"] == "rel_est"
    assert {row.statistic for row in headline_rows} == {
        "gap",
        "composition",
        "structure",
    }


def test_rif_uses_dineq_quantile_convention_with_tight_bridge():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "32_rif_py.json")
    stata_payload = _read_json(STATA_RESULTS / "32_rif_Stata.json")
    rows = {row.statistic: row for row in compare.collect("32_rif")}

    assert compare.TOLERANCES["32_rif"]["rel_est"] == 1e-6
    assert compare.tolerance_tier("32_rif") == "machine"
    assert py_payload["extra"]["quantile_convention"] == "dineq"
    assert stata_payload["extra"]["quantile_convention"] == "dineq"
    assert (
        "R stats::density binned Gaussian density"
        in stata_payload["extra"]["stata_algorithm"]
    )

    for statistic in (
        "total_diff",
        "explained",
        "unexplained",
        "explained_educ",
        "explained_exper",
    ):
        assert rows[statistic].rel_est < 1e-12
        assert (rows[statistic].rel_est_st or 0.0) < 1e-12

    cfg, headline_rows = _headline_rows(compare, "32_rif")
    assert cfg["metric"] == "rel_est"
    assert {row.statistic for row in headline_rows} == {"total_diff"}
    assert max(row.rel_est or 0.0 for row in headline_rows) < 1e-7


def test_drdid_has_materialized_stata_bridge():
    compare = _load_compare()
    script = STATA_PARITY / "38_drdid.do"

    assert script.exists()
    source = script.read_text(encoding="utf-8")
    py_payload = _read_json(R_RESULTS / "38_drdid_py.json")
    payload = _read_json(STATA_RESULTS / "38_drdid_Stata.json")
    extra = payload["extra"]
    assert py_payload["extra"]["id"] == "id"
    assert py_payload["extra"]["se_method"] == "influence_function"
    assert "calibrated propensity" in py_payload["extra"]["reference_method_note"]
    assert "38_drdid" not in compare.STATA_SKIP_REASON
    assert compare.TOLERANCES["38_drdid"]["rel_est"] == 1e-6
    assert compare.TOLERANCES["38_drdid"]["rel_se"] == 1e-6
    assert extra["stata_bridge_status"] == "materialized with licensed Stata/MP 18"
    assert (
        extra["stata_command"]
        == "drdid y x, ivar(id) time(post) treatment(treated) drimp"
    )
    assert "drdid y x, ivar(id) time(post) treatment(treated) drimp" in source
    assert "matrix B = e(b)" in source
    assert "matrix V = e(V)" in source
    rows = compare.collect("38_drdid")
    assert {row.statistic for row in rows} == {"att", "ci_lower", "ci_upper"}
    assert max(row.rel_est or 0.0 for row in rows) < 1e-6
    assert max(row.rel_est_st or 0.0 for row in rows) < 1e-6
    assert max(row.rel_se or 0.0 for row in rows) < 1e-6
    assert max(row.rel_se_st or 0.0 for row in rows) < 1e-6


def test_sdid_att_row_is_point_only_with_backend_se_diagnostics():
    py_payload = _read_json(R_RESULTS / "12_sdid_py.json")
    r_payload = _read_json(R_RESULTS / "12_sdid_R.json")
    stata_payload = _read_json(STATA_RESULTS / "12_sdid_Stata.json")
    py_rows = {row["statistic"]: row for row in py_payload["rows"]}
    r_rows = {row["statistic"]: row for row in r_payload["rows"]}
    stata_rows = {row["statistic"]: row for row in stata_payload["rows"]}

    assert py_rows["att_sdid"]["se"] is None
    assert r_rows["att_sdid"]["se"] is None
    assert stata_rows["att_sdid"]["se"] is None
    assert "se_native_placebo" in py_rows
    assert "se_synthdid_placebo" in r_rows
    assert "se_stata_sdid_placebo" in stata_rows
    assert py_rows["se_native_placebo"]["estimate"] > 0
    assert r_rows["se_synthdid_placebo"]["estimate"] > 0
    assert stata_rows["se_stata_sdid_placebo"]["estimate"] > 0
    assert "native_note" not in py_payload["extra"]
    assert "native_parity_note" in py_payload["extra"]


def test_rd_default_rows_use_cct_delegation_not_legacy_internal_selector():
    compare = _load_compare()
    py = _read_json(R_RESULTS / "06_rd_py.json")
    r_stats = {
        row["statistic"] for row in _read_json(R_RESULTS / "06_rd_R.json")["rows"]
    }
    stata_stats = {
        row["statistic"]
        for row in _read_json(STATA_RESULTS / "06_rd_Stata.json")["rows"]
    }

    assert py["extra"]["bwselect"] == "cct"
    assert "bandwidth_selector_gap" not in py["extra"]

    py_stats = {row["statistic"] for row in py["rows"]}
    assert "default_robust_est" in py_stats & r_stats & stata_stats
    assert "default_bandwidth_h" in py_stats & r_stats & stata_stats
    assert "legacy_internal_mserd_bandwidth_h" in py_stats
    assert "legacy_internal_mserd_bandwidth_h" not in r_stats | stata_stats

    cfg, rows = _headline_rows(compare, "06_rd")
    assert cfg["metric"] == "rel_est"
    assert {row.statistic for row in rows} == {
        "default_conventional_est",
        "default_robust_est",
    }
    assert max(row.rel_est or 0.0 for row in rows) < 1e-9
    assert max(row.rel_est_st or 0.0 for row in rows) < 1e-8


def test_sunab_weighted_att_uses_fixest_aggregation_not_event_time_average():
    compare = _load_compare()
    py = _read_json(R_RESULTS / "05_sunab_py.json")
    r_stats = {
        row["statistic"] for row in _read_json(R_RESULTS / "05_sunab_R.json")["rows"]
    }
    stata_stats = {
        row["statistic"]
        for row in _read_json(STATA_RESULTS / "05_sunab_Stata.json")["rows"]
    }

    assert py["extra"]["aggregation"] == "fixest_att"
    assert "aggregation_note" not in py["extra"]

    py_stats = {row["statistic"] for row in py["rows"]}
    assert "weighted_avg_ATT" in py_stats & r_stats & stata_stats
    assert "event_time_avg_ATT" in py_stats
    assert "event_time_avg_ATT" not in r_stats | stata_stats

    rows = {row.statistic: row for row in compare.collect("05_sunab")}
    assert rows["weighted_avg_ATT"].rel_est < 1e-10
    assert rows["weighted_avg_ATT"].rel_est_st < 1e-10

    cfg, headline_rows = _headline_rows(compare, "05_sunab")
    assert cfg["metric"] == "rel_est"
    assert any(row.statistic == "weighted_avg_ATT" for row in headline_rows)
    assert max(row.rel_est or 0.0 for row in headline_rows) < 1e-9


def test_bacon_decomposition_uses_r_stata_bacondecomp_conventions():
    compare = _load_compare()
    py = _read_json(R_RESULTS / "20_bacon_py.json")
    rows = {row.statistic: row for row in compare.collect("20_bacon")}

    assert py["extra"]["n_comparisons"] == 9
    assert "differ across implementations" not in py["extra"]["decomposition_note"]
    assert py["extra"]["already_treated_control_weight_share"] > 0.0

    for statistic in (
        "beta_twfe",
        "weighted_sum",
        "negative_weight_share",
        "pair_2006_vs_2004_est",
        "pair_2004_vs_2006_est",
        "pair_2004_vs_never_est",
    ):
        assert rows[statistic].rel_est < 1e-10

    assert rows["weighted_sum"].rel_est_st < 1e-8
    assert rows["negative_weight_share"].abs_est_st == 0.0


def test_native_synth_backend_notes_are_positive_parity_notes():
    for module in ("18_augsynth", "19_gsynth"):
        payload = _read_json(R_RESULTS / f"{module}_py.json")
        assert "native_parity_note" in payload["extra"]
        assert "native_note" not in payload["extra"]


def test_hdfe_parity_uses_fixest_small_sample_corrections():
    compare = _load_compare()
    hdfe = _read_json(R_RESULTS / "03_hdfe_py.json")
    hdfe_cluster = _read_json(R_RESULTS / "15_hdfe_cluster_py.json")

    assert hdfe["extra"]["ssc"] == "fixest"
    assert "df_convention" not in hdfe["extra"]
    assert hdfe_cluster["extra"]["ssc"] == "fixest"
    assert "cluster_ssc_note" not in hdfe_cluster["extra"]

    for module in ("03_hdfe", "15_hdfe_cluster"):
        rows = [
            row for row in compare.collect(module) if row.statistic.startswith("beta_")
        ]
        assert rows
        assert max(row.rel_est or 0.0 for row in rows) < 1e-12
        assert max(row.rel_se or 0.0 for row in rows) < 1e-10
        assert max(row.rel_est_st or 0.0 for row in rows) < 1e-8
        assert max(row.rel_se_st or 0.0 for row in rows) < 1e-8


def test_dml_plr_parity_uses_shared_explicit_folds():
    compare = _load_compare()
    py = _read_json(R_RESULTS / "08_dml_py.json")
    r = _read_json(R_RESULTS / "08_dml_R.json")
    stata = _read_json(STATA_RESULTS / "08_dml_Stata.json")
    row = {diff.statistic: diff for diff in compare.collect("08_dml")}["theta_DML_PLR"]

    assert "08_dml" not in compare.STATA_SKIP_REASON
    assert py["extra"]["fold_source"] == "user"
    assert py["extra"]["fold_column"] == "fold_id"
    assert "fold_split_note" not in py["extra"]
    assert r["extra"]["fold_source"] == ["user"]
    assert r["extra"]["fold_column"] == ["fold_id"]
    assert stata["extra"]["fold_source"] == "user"
    assert stata["extra"]["fold_column"] == "fold_id"
    assert (
        stata["extra"]["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
    )
    assert "foldwise OLS residualization" in stata["extra"]["stata_algorithm"]
    assert row.rel_est < 1e-12
    assert row.rel_se < 1e-12
    assert row.rel_est_st < 1e-12
    assert row.rel_se_st < 1e-12


def test_panel_hausman_uses_plm_style_covariance_and_splits_stata_sigmamore():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "35_panel_py.json")
    r_payload = _read_json(R_RESULTS / "35_panel_R.json")
    stata_payload = _read_json(STATA_RESULTS / "35_panel_Stata.json")
    py_rows = {row["statistic"]: row for row in py_payload["rows"]}
    r_rows = {row["statistic"]: row for row in r_payload["rows"]}
    stata_rows = {row["statistic"]: row for row in stata_payload["rows"]}

    assert py_payload["extra"]["hausman_parity_note"]
    assert "hausman_note" not in py_payload["extra"]
    assert "hausman_chi2" in set(py_rows) & set(r_rows)
    assert "hausman_pvalue" in set(py_rows) & set(r_rows)
    assert "hausman_chi2" not in stata_rows
    assert "hausman_pvalue" not in stata_rows
    assert "hausman_chi2_stata_sigmamore" in stata_rows
    assert "hausman_pvalue_stata_sigmamore" in stata_rows

    rows = {diff.statistic: diff for diff in compare.collect("35_panel")}
    assert rows["hausman_chi2"].rel_est < 1e-12
    assert rows["hausman_pvalue"].rel_est < 1e-12


def test_local_projection_splits_lpirfs_headline_and_stata_direct_ols():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "34_lp_py.json")
    stata_payload = _read_json(STATA_RESULTS / "34_lp_Stata.json")
    py_rows = {row["statistic"]: row for row in py_payload["rows"]}
    stata_rows = {row["statistic"]: row for row in stata_payload["rows"]}

    assert "r_identification_reference" in py_payload["extra"]
    assert "identification_note" not in py_payload["extra"]
    assert "34_lp" not in compare.STATA_HEADLINE_GAP_EXCEPTIONS

    headline = {
        diff.statistic: diff
        for diff in compare.collect("34_lp")
        if diff.statistic.startswith("irf_h")
    }
    assert set(headline) == {f"irf_h{h}" for h in range(6)}
    assert max(row.rel_est or 0.0 for row in headline.values()) < 1e-12

    for h in range(6):
        stat = f"irf_direct_ols_h{h}"
        assert stat in py_rows
        assert stat in stata_rows
        assert py_rows[stat]["estimate"] == pytest.approx(
            stata_rows[stat]["estimate"], abs=1e-12
        )


def test_multiway_cluster_matches_sandwich_and_bounds_reghdfe_convention():
    compare = _load_compare()
    expected_reghdfe_bounds = {
        "54_twoway_cluster": 0.007,
        "56_multiway_cluster": 0.012,
    }

    for module, reghdfe_bound in expected_reghdfe_bounds.items():
        py_payload = _read_json(R_RESULTS / f"{module}_py.json")
        stata_payload = _read_json(STATA_RESULTS / f"{module}_Stata.json")
        assert "convention" not in py_payload["extra"]
        extra = stata_payload["extra"]
        assert extra["stata_bridge_status"] == "audited Stata/Mata algorithm bridge"
        assert "Cameron-Gelbach-Miller" in extra["stata_algorithm"]
        assert "sandwich::vcovCL HC1/cadjust" in extra["stata_algorithm"]
        assert "reghdfe 6.12.3" in extra["stata_cluster_reference"]
        assert float(extra["stata_reghdfe_max_rel_py_se_diff"]) <= reghdfe_bound
        assert module not in compare.STATA_HEADLINE_GAP_EXCEPTIONS

        for row in compare.collect(module):
            assert row.rel_est < 1e-12
            assert row.rel_se < 1e-12
            assert (row.rel_est_st or 0.0) < 1e-12
            assert (row.rel_se_st or 0.0) < 1e-12


def test_cr2_cr3_both_match_clubsandwich_reference():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "53_cr2_py.json")
    stata_payload = _read_json(STATA_RESULTS / "53_cr2_Stata.json")

    assert "cr3_reference" in py_payload["extra"]
    assert "cr3_convention" not in py_payload["extra"]
    assert "53_cr2" not in compare.STATA_SKIP_REASON
    assert (
        stata_payload["extra"]["stata_bridge_status"]
        == "audited Stata/Mata algorithm bridge"
    )
    assert "cluster hat blocks" in stata_payload["extra"]["stata_reference_note"]

    rows = compare.collect("53_cr2")
    assert {row.statistic for row in rows} == {
        "cr2_(Intercept)",
        "cr3_(Intercept)",
        "cr2_treat",
        "cr3_treat",
        "cr2_year",
        "cr3_year",
    }
    assert max(row.rel_se or 0.0 for row in rows) < 1e-7
    assert max(row.rel_se_st or 0.0 for row in rows) < 1e-6


def test_arima_uses_innovations_mle_reference_mode_for_r_stata_coefficients():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "39_arima_py.json")
    r_payload = _read_json(R_RESULTS / "39_arima_R.json")
    stata_payload = _read_json(STATA_RESULTS / "39_arima_Stata.json")

    assert py_payload["extra"]["method"] == "innovations_mle"
    assert "innovations_mle" in py_payload["extra"]["reference_method_note"]
    assert "stats::arima(method='ML')" in py_payload["extra"]["reference_method_note"]
    assert "loglik_note" not in py_payload["extra"]
    assert r_payload["extra"]["package"] == ["stats::arima"]
    assert r_payload["extra"]["method"] == ["ML"]
    assert stata_payload["extra"]["method"] == "exact MLE"
    assert compare.TOLERANCES["39_arima"]["rel_est"] == 1e-6
    assert "tightly converged Stata arima" in compare.HEADLINE["39_arima"]["gap_note"]

    rows = [
        row
        for row in compare.collect("39_arima")
        if row.statistic.startswith("ar") or row.statistic == "logLik"
    ]
    assert {row.statistic for row in rows} == {"ar1", "ar2", "logLik"}
    assert max(row.rel_est or 0.0 for row in rows) < 1e-6
    assert max(row.rel_est_st or 0.0 for row in rows) < 1e-6


def test_R_golden_json_carry_engine_provenance():
    """Every committed _R.json must carry a `provenance` block recording the
    R version and the package set that produced it, so the golden value is
    self-describing for JSS reproducibility. This guards against a future
    regeneration (or a hand edit) silently dropping provenance.
    """
    r_files = sorted(R_RESULTS.glob("*_R.json"))
    assert r_files, "no _R.json golden files found"
    missing = []
    for path in r_files:
        prov = _read_json(path).get("provenance")
        if (
            not isinstance(prov, dict)
            or not prov.get("r_version")
            or not isinstance(prov.get("packages"), dict)
            or not prov["packages"]
        ):
            missing.append(path.name)
    assert not missing, (
        "These _R.json lack a complete provenance block (r_version + "
        "non-empty packages); regenerate via tests/r_parity and re-commit: "
        + ", ".join(missing)
    )


def test_Stata_golden_json_carry_engine_provenance():
    """Every committed _Stata.json must carry a `provenance` block recording
    the Stata engine (version + edition) that produced it. The community ado
    versions live in tests/stata_parity/STATA_ENVIRONMENT.md (Stata has no
    per-result packageVersion() primitive), so we assert the engine fields
    here, not a package map.
    """
    stata_files = sorted(STATA_RESULTS.glob("*_Stata.json"))
    assert stata_files, "no _Stata.json golden files found"
    missing = []
    for path in stata_files:
        prov = _read_json(path).get("provenance")
        if (
            not isinstance(prov, dict)
            or not prov.get("stata_version")
            or not prov.get("edition")
        ):
            missing.append(path.name)
    assert not missing, (
        "These _Stata.json lack a complete provenance block (stata_version + "
        "edition); regenerate via tests/stata_parity and re-commit: "
        + ", ".join(missing)
    )


def test_headline_passes_are_inside_registered_r_tolerance():
    compare = _load_compare()
    r_modules = _module_stems(R_RESULTS, "_R")

    for module in sorted(r_modules):
        cfg, rows = _headline_rows(compare, module)
        metric = cfg["metric"]
        values = [
            getattr(row, metric) for row in rows if getattr(row, metric) is not None
        ]
        assert values, f"{module} headline has no {metric} values"

        if "PASS" not in cfg["verdict"]:
            continue
        tolerance = compare.TOLERANCES[module].get(metric)
        assert tolerance is not None, f"{module} PASS lacks {metric} tolerance"
        assert (
            max(values) <= tolerance
        ), f"{module} R headline {metric}={max(values):.6g} exceeds {tolerance}"


def test_stata_headline_over_budget_modules_are_explicitly_registered():
    compare = _load_compare()
    stata_modules = _module_stems(STATA_RESULTS, "_Stata")
    over_budget: dict[str, tuple[float, float]] = {}

    for module in sorted(stata_modules):
        if not compare.collect(module):
            continue
        cfg, rows = _headline_rows(compare, module)
        metric = cfg["metric"]
        if metric == "rel_est":
            stata_metric = "rel_est_st"
        elif metric == "rel_se":
            stata_metric = "rel_se_st"
        else:
            stata_metric = "abs_est_st"
        values = [
            getattr(row, stata_metric)
            for row in rows
            if getattr(row, stata_metric) is not None
        ]
        if not values:
            continue
        tolerance = compare.TOLERANCES[module].get(metric)
        if tolerance is not None and max(values) > tolerance:
            over_budget[module] = (max(values), tolerance)

    assert set(over_budget) == set(compare.STATA_HEADLINE_GAP_EXCEPTIONS)


def test_honest_did_smoothness_uses_reference_backend_with_tight_stata_port():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "10_honest_did_py.json")

    assert py_payload["extra"]["backend"] == "HonestDiD"
    assert "backend='honestdid'" in py_payload["extra"]["reference_backend_note"]
    assert compare.TOLERANCES["10_honest_did"]["abs_est"] == 5e-4
    assert compare.HEADLINE["10_honest_did"]["metric"] == "abs_est"
    assert "Stata port within 0.0005" in compare.HEADLINE["10_honest_did"]["gap_note"]

    rows = [
        row
        for row in compare.collect("10_honest_did")
        if row.statistic.startswith("ci_")
    ]
    assert rows
    assert max(row.abs_est or 0.0 for row in rows) < 1e-12
    assert max(row.abs_est_st or 0.0 for row in rows) < 3.5e-4


def test_honest_relmags_conditional_bridge_is_exact_and_versioned():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "21_honest_relmags_py.json")
    r_payload = _read_json(R_RESULTS / "21_honest_relmags_R.json")
    stata_payload = _read_json(STATA_RESULTS / "21_honest_relmags_Stata.json")
    py_rows = {row["statistic"]: row["estimate"] for row in py_payload["rows"]}
    r_rows = {row["statistic"]: row["estimate"] for row in r_payload["rows"]}
    stata_rows = {row["statistic"]: row["estimate"] for row in stata_payload["rows"]}
    extra = stata_payload["extra"]

    assert py_payload["extra"]["honestdid_method"] == "Conditional"
    assert r_payload["extra"]["honestdid_method"] == ["Conditional"]
    assert extra["honestdid_method"] == "Conditional"
    assert extra["stata_reference_package"] == "honestdid 1.3.0"
    assert "method(Conditional) gridPoints(1000)" in extra["stata_command"]
    assert "grid_lb(-2) grid_ub(2)" in extra["stata_command"]
    assert "R HonestDiD 0.2.8" in extra["stata_precision_note"]

    max_abs_py_r = max(abs(py_rows[k] - r_rows[k]) for k in py_rows)
    max_abs_py_stata = max(abs(py_rows[k] - stata_rows[k]) for k in py_rows)
    assert max_abs_py_r <= 1e-12
    assert max_abs_py_stata == pytest.approx(
        float(extra["stata_reference_max_abs_py_diff"]),
        abs=1e-15,
    )
    assert max_abs_py_stata <= 1e-12
    assert "21_honest_relmags" not in compare.STATA_HEADLINE_GAP_EXCEPTIONS


def test_bjs_stata_never_treated_coding_matches_r_path():
    compare = _load_compare()
    py_payload = _read_json(R_RESULTS / "16_bjs_py.json")
    r_payload = _read_json(R_RESULTS / "16_bjs_R.json")
    stata_payload = _read_json(STATA_RESULTS / "16_bjs_Stata.json")
    py_rows = {row["statistic"]: row for row in py_payload["rows"]}
    r_rows = {row["statistic"]: row for row in r_payload["rows"]}
    stata_rows = {row["statistic"]: row for row in stata_payload["rows"]}
    row = {diff.statistic: diff for diff in compare.collect("16_bjs")}["att_bjs"]

    assert row.rel_est < 1e-6
    assert py_rows["att_bjs"]["se"] is None
    assert r_rows["att_bjs"]["se"] is None
    assert stata_rows["att_bjs"]["se"] is None
    assert py_rows["se_cluster_if"]["estimate"] > 0
    assert r_rows["se_didimputation"]["estimate"] > 0
    assert stata_rows["se_stata_did_imputation"]["estimate"] > 0
    assert row.rel_est_st < 1e-6
    assert "stata_never_treated_coding" in py_payload["extra"]
    assert "never_treated_coding" in stata_payload["extra"]
    assert "16_bjs" not in compare.STATA_HEADLINE_GAP_EXCEPTIONS


def test_panel_sfa_stata_gap_is_parameterisation_not_slope_drift():
    compare = _load_compare()
    cfg, headline_rows = _headline_rows(compare, "29_panel_sfa")
    rows = {diff.statistic: diff for diff in compare.collect("29_panel_sfa")}

    assert cfg["metric"] == "rel_est"
    assert {row.statistic for row in headline_rows} == {"beta_lnk", "beta_lnl"}
    assert rows["beta_lnk"].rel_est_st < 1e-3
    assert rows["beta_lnl"].rel_est_st < 1e-3
    assert 0.015 < rows["beta_intercept"].rel_est_st < 0.02
    assert rows["sigma_u"].rel_est_st > 0.25
    assert "29_panel_sfa" not in compare.STATA_HEADLINE_GAP_EXCEPTIONS


def test_generated_parity_tables_are_in_sync_with_comparator():
    compare = _load_compare()
    modules = sorted(
        path.stem.replace("_py", "") for path in R_RESULTS.glob("*_py.json")
    )

    assert (R_RESULTS / "parity_table.md").read_text(encoding="utf-8") == (
        compare.render_md(modules)
    )
    assert (R_RESULTS / "parity_table.tex").read_text(encoding="utf-8") == (
        compare.render_tex(modules)
    )
    assert (R_RESULTS / "parity_table_3way.md").read_text(encoding="utf-8") == (
        compare.render_md_3way(modules)
    )
    assert (R_RESULTS / "parity_table_3way.tex").read_text(encoding="utf-8") == (
        compare.render_tex_3way(modules)
    )


def test_parity_readmes_match_current_artifact_inventory():
    # The StatsPAI<->R table lists the current R-joined module inventory.
    # The historical 50_xtabond R-side gap is now closed through plm::pgmm,
    # so the readme inventory should cover every committed Python module row.
    py_numbers = {
        module.split("_", 1)[0]
        for module in _module_stems(R_RESULTS, "_py")
        if (R_RESULTS / f"{module}_R.json").exists()
    }
    stata_numbers = {
        module.split("_", 1)[0] for module in _module_stems(STATA_RESULTS, "_Stata")
    }

    r_readme = R_PARITY / "README.md"
    stata_readme = ROOT / "tests" / "stata_parity" / "README.md"

    assert py_numbers <= _readme_module_numbers(r_readme)
    assert stata_numbers == _readme_module_numbers(stata_readme)
    assert "Modules (36)" not in r_readme.read_text(encoding="utf-8")
    assert "21 of 36" not in stata_readme.read_text(encoding="utf-8")
    assert (R_PARITY / "50_xtabond.R").exists()
