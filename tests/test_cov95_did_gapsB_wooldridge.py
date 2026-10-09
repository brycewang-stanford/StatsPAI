"""Coverage gaps (batch B) for statspai.did.wooldridge_did.

Validation and degenerate-design branches of ``sp.wooldridge_did`` /
``sp.etwfe`` / ``sp.drdid`` / ``sp.twfe_decomposition`` / ``sp.etwfe_emfx``
that the main suites do not reach, plus the independent-cell fallbacks of
the two cell aggregators (checked against the formula they implement).
"""

import copy

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.wooldridge_did import (
    _aggregate_cells,
    _cohort_atts_from_cells,
    _drdid_imp_panel_core,
    _etwfe_repeated_cs,
)
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def panel():
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=3).copy()
    df["w"] = 1.0 + (df["unit"] % 3)
    df["x"] = np.random.default_rng(5).normal(size=len(df))
    return df


def _cells():
    return pd.DataFrame(
        {
            "cohort": [3, 3, 4],
            "rel_time": [0, 1, 0],
            "estimate": [1.0, 3.0, 2.0],
            "se": [0.3, 0.4, 0.5],
            "n_treated_obs": [10, 30, 20],
        }
    )


# ----------------------------------------------------------------------
# _cohort_atts_from_cells
# ----------------------------------------------------------------------
def test_cohort_atts_no_post_cells_raises():
    es = _cells().assign(rel_time=[-3, -2, -2])
    with pytest.raises(DataInsufficient, match="No post-treatment cohort x period"):
        _cohort_atts_from_cells(es, None, [3, 4], {3: 5, 4: 5}, 50, 0.05)


def test_cohort_atts_cohort_without_post_cell_raises():
    with pytest.raises(DataInsufficient, match="Cohort 5 has no post-treatment"):
        _cohort_atts_from_cells(_cells(), None, [3, 4, 5], {3: 5, 4: 5, 5: 5}, 50, 0.05)


def test_cohort_atts_independent_cell_fallback_matches_formula():
    detail, cohort_vcov, att, se, p, ci = _cohort_atts_from_cells(
        _cells(), None, [3, 4], {3: 6, 4: 2}, 50, 0.05
    )
    # ATT(3) is the treated-observation-weighted mean of its two cells.
    assert detail.loc[0, "att"] == pytest.approx(0.25 * 1.0 + 0.75 * 3.0)
    se3 = np.sqrt((0.25 * 0.3) ** 2 + (0.75 * 0.4) ** 2)
    assert detail.loc[0, "se"] == pytest.approx(se3)
    assert detail.loc[1, "se"] == pytest.approx(0.5)
    # Without a cell covariance the cohort covariance is diagonal.
    np.testing.assert_allclose(cohort_vcov, np.diag([se3**2, 0.25]))
    assert att == pytest.approx(0.75 * 2.5 + 0.25 * 2.0)
    assert se == pytest.approx(np.sqrt(0.75**2 * se3**2 + 0.25**2 * 0.25))
    assert ci[0] < att < ci[1] and 0.0 <= p <= 1.0


# ----------------------------------------------------------------------
# _aggregate_cells
# ----------------------------------------------------------------------
def test_aggregate_cells_empty_raises():
    with pytest.raises(DataInsufficient, match="No cohort × period cells"):
        _aggregate_cells(_cells().iloc[:0], None, "cohort", "n_treated_obs")


def test_aggregate_cells_missing_weight_column_is_unweighted():
    rows, vcov, vcov_based = _aggregate_cells(_cells(), None, "cohort", "absent")
    assert vcov is None and vcov_based is False
    assert rows["estimate"].tolist() == pytest.approx([2.0, 2.0])
    assert rows.loc[0, "se"] == pytest.approx(np.sqrt(0.15**2 + 0.2**2))
    assert rows["n_cells"].tolist() == [2, 1]


# ----------------------------------------------------------------------
# sp.wooldridge_did degenerate designs
# ----------------------------------------------------------------------
def test_wooldridge_only_reference_period_observed():
    df = pd.DataFrame(
        {
            "unit": [1, 2, 3, 4],
            "time": [2, 2, 2, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "ft": [3, 3, np.nan, np.nan],
        }
    )
    with pytest.raises(DataInsufficient, match="No cohort × period cell"):
        sp.wooldridge_did(df, y="y", group="unit", time="time", first_treat="ft")


def test_wooldridge_too_few_rows_for_saturated_design():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "unit": np.repeat([1, 2], 6),
            "time": np.tile(np.arange(1, 7), 2),
            "y": rng.normal(size=12),
            "ft": np.repeat([3.0, np.nan], 6),
        }
    )
    df = df.iloc[:5]
    with pytest.raises(DataInsufficient, match="saturated ETWFE design needs"):
        sp.wooldridge_did(df, y="y", group="unit", time="time", first_treat="ft")


def test_wooldridge_requires_a_never_treated_unit(panel):
    df = panel.loc[panel["first_treat"].notna()]
    with pytest.raises(DataInsufficient, match="at least one never-treated"):
        sp.wooldridge_did(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )


# ----------------------------------------------------------------------
# _etwfe_repeated_cs
# ----------------------------------------------------------------------
def test_repeated_cs_nevertreated_alias_equals_never(panel):
    kw = dict(y="y", time="time", first_treat="first_treat")
    a = _etwfe_repeated_cs(panel, cgroup="nevertreated", **kw)
    b = _etwfe_repeated_cs(panel, cgroup="never", **kw)
    assert a.estimate == b.estimate and a.se == b.se
    assert a.model_info["cgroup"] == b.model_info["cgroup"]


def test_repeated_cs_bad_cgroup(panel):
    with pytest.raises(MethodIncompatibility, match="cgroup must be"):
        _etwfe_repeated_cs(
            panel, y="y", time="time", first_treat="first_treat", cgroup="all"
        )


def test_repeated_cs_never_needs_never_treated(panel):
    df = panel.loc[panel["first_treat"].notna()]
    with pytest.raises(DataInsufficient, match="requires at least one never-treated"):
        _etwfe_repeated_cs(
            df, y="y", time="time", first_treat="first_treat", cgroup="never"
        )


def test_repeated_cs_xvar_incompatibilities(panel):
    kw = dict(y="y", time="time", first_treat="first_treat", xvar=["x"])
    with pytest.raises(MethodIncompatibility, match="xvar with cgroup"):
        _etwfe_repeated_cs(panel, cgroup="never", **kw)
    with pytest.raises(MethodIncompatibility, match="not yet supported together"):
        _etwfe_repeated_cs(panel, weights="w", **kw)


def test_repeated_cs_single_period_has_no_cells():
    df = pd.DataFrame(
        {
            "time": [1] * 6,
            "y": [1.0, 2.0, 0.5, 1.5, 3.0, 2.5],
            "ft": [1, 1, 1, np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(DataInsufficient, match="No cohort × period cells"):
        _etwfe_repeated_cs(df, y="y", time="time", first_treat="ft")


def test_repeated_cs_cohort_adopting_after_sample_end():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame(
        {
            "time": rng.integers(1, 5, n),
            "y": rng.normal(size=n),
            "ft": np.where(rng.random(n) < 0.5, 9.0, np.nan),
        }
    )
    with pytest.raises(DataInsufficient, match="No treated post-treatment cells"):
        _etwfe_repeated_cs(df, y="y", time="time", first_treat="ft", cgroup="never")


# ----------------------------------------------------------------------
# sp.drdid
# ----------------------------------------------------------------------
def _two_period_panel(seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
    y0 = x + rng.normal(size=n)
    y1 = y0 + 0.5 + 0.3 * x + 2.0 * d + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n)
    pre = pd.DataFrame({"id": np.arange(n), "t": 0, "y": y0, "d": d, "x": x, "w": w})
    post = pre.assign(t=1, y=y1)
    return pd.concat([pre, post], ignore_index=True)


@pytest.mark.parametrize("bad", [0.0, 1.5])
def test_drdid_trim_level_out_of_range(bad):
    df = _two_period_panel()
    with pytest.raises(MethodIncompatibility, match=r"trim_level must be in \(0, 1\]"):
        sp.drdid(df, y="y", group="d", time="t", covariates=["x"], trim_level=bad)


def test_drdid_panel_drops_units_with_missing_weight():
    # A unit whose weight is missing is not a complete case; dropping it must
    # give the same fit as removing the unit from the input.
    df = _two_period_panel()
    kw = dict(y="y", group="d", time="t", covariates=["x"], id="id", method="trad")
    holed = df.copy()
    holed.loc[holed["id"].isin([0, 1, 2]), "w"] = np.nan
    r_holed = sp.drdid(holed, weights="w", **kw)
    r_drop = sp.drdid(df.loc[~df["id"].isin([0, 1, 2])], weights="w", **kw)
    r_full = sp.drdid(df, weights="w", **kw)
    assert r_holed.estimate == pytest.approx(r_drop.estimate, rel=1e-12)
    assert r_holed.se == pytest.approx(r_drop.se, rel=1e-12)
    assert r_holed.estimate != r_full.estimate


def test_drdid_imp_panel_core_empty_input():
    with pytest.raises(DataInsufficient, match="at least one complete unit"):
        _drdid_imp_panel_core(np.array([]), np.array([]), np.ones((0, 1)))


# ----------------------------------------------------------------------
# sp.twfe_decomposition
# ----------------------------------------------------------------------
def test_twfe_decomposition_missing_columns(panel):
    with pytest.raises(MethodIncompatibility, match="columns not found") as ei:
        sp.twfe_decomposition(
            panel, y="nope", group="unit", time="time", first_treat="first_treat"
        )
    assert ei.value.diagnostics["missing"] == ["nope"]


def test_twfe_decomposition_duplicate_unit_period(panel):
    df = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="more than one row for a unit"):
        sp.twfe_decomposition(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )


def test_twfe_decomposition_unbalanced_and_too_large():
    n_units = 5000
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), 2),
            "time": np.tile([1, 2], n_units),
            "ft": np.repeat(np.where(np.arange(n_units) % 2 == 0, 2.0, np.nan), 2),
        }
    )
    df["y"] = np.random.default_rng(0).normal(size=len(df))
    df = df.iloc[1:]
    with pytest.raises(MethodIncompatibility, match="unbalanced and too large"):
        sp.twfe_decomposition(df, y="y", group="unit", time="time", first_treat="ft")


def test_twfe_decomposition_treatment_without_variation():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(10), 4),
            "time": np.tile(np.arange(1, 5), 10),
            "ft": 1.0,
        }
    )
    df["y"] = rng.normal(size=len(df))
    with pytest.raises(DataInsufficient, match="treatment does not vary"):
        sp.twfe_decomposition(df, y="y", group="unit", time="time", first_treat="ft")


# ----------------------------------------------------------------------
# sp.etwfe_emfx
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def weighted_fit(panel):
    return sp.etwfe(
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        weights="w",
    )


def test_emfx_scale_is_validated_and_inert_on_linear_fit(weighted_fit):
    with pytest.raises(MethodIncompatibility, match="scale='bogus' is not recognised"):
        sp.etwfe_emfx(weighted_fit, scale="bogus")
    a = sp.etwfe_emfx(weighted_fit, scale="response")
    b = sp.etwfe_emfx(weighted_fit)
    assert a.estimate == b.estimate and a.se == b.se


def test_emfx_cohort_weighting_uses_estimation_weight_totals(weighted_fit):
    out = sp.etwfe_emfx(weighted_fit, type="simple", weighting="cohort")
    assert out.model_info["weight_column"] == "w_obs"
    det = weighted_fit.detail
    w = det["w_obs"] / det["w_obs"].sum()
    assert out.estimate == pytest.approx(float(w @ det["att"]))
    # agg_weights='unit' goes back to the observation counts.
    unit = sp.etwfe_emfx(
        weighted_fit, type="simple", weighting="cohort", agg_weights="unit"
    )
    assert unit.model_info["weight_column"] == "n_obs"
    wn = det["n_obs"] / det["n_obs"].sum()
    assert unit.estimate == pytest.approx(float(wn @ det["att"]))


def test_emfx_group_without_cells_reads_cohort_detail(weighted_fit):
    legacy = copy.copy(weighted_fit)
    legacy.model_info = dict(weighted_fit.model_info, event_study=None)
    out = sp.etwfe_emfx(legacy, type="group")
    det = weighted_fit.detail
    assert out.detail["cohort"].tolist() == det["cohort"].astype(int).tolist()
    np.testing.assert_allclose(out.detail["estimate"], det["att"])
    np.testing.assert_allclose(out.detail["se"], det["se"])
    np.testing.assert_allclose(out.detail["pvalue"], det["pvalue"])
    assert (out.detail["ci_low"] < out.detail["estimate"]).all()
