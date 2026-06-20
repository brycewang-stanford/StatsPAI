"""Tests for the psmatch2-style matched-sample variables and sp.psmatch2.

Covers:
- the matched frame attached to sp.match / sp.psm (JOSS-safe additive path);
- Stata-faithful semantics of _weight / _support / _nn / _n1 / _pdif / _y;
- common-support trimming and caliper behaviour;
- sp.psmatch2 result surface (.matched_sample / .balance / .psm_did);
- the frequency-weighted PSM-DID recovery of a known effect.

Numerical parity with Stata 18 psmatch2 lives in
tests/reference_parity/test_psmatch2_parity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.matching._matched_frame import (
    build_matched_frame,
    common_support_mask,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def psm_data():
    """Selection-on-observables cross-section. True ATT = 2.0."""
    rng = np.random.default_rng(2024)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.7 * x1 + 0.4 * x2 - 0.2)))
    d = rng.binomial(1, p)
    y = 1 + 2.0 * d + 3 * x1 + x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "d": d, "y": y})


@pytest.fixture
def psm_panel():
    """Two-period panel with a true DiD effect of 2.5, matchable on x1, x2."""
    rng = np.random.default_rng(11)
    n = 700
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x1 + 0.5 * x2))))
    ufe = rng.normal(size=n)
    base = pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "d": d})
    rows = []
    ate = 2.5
    for i in range(n):
        for t in (0, 1):
            y = (
                1
                + 0.5 * t
                + 2.0 * x1[i]
                + x2[i]
                + ufe[i]
                + ate * d[i] * t
                + rng.normal(scale=0.4)
            )
            rows.append((i, t, d[i], y))
    panel = pd.DataFrame(rows, columns=["id", "time", "d", "y"])
    panel.attrs["true_did"] = ate
    return base, panel


def _manual_psmatch2(matched_data, *, model_info=None, covariates=None):
    result = CausalResult(
        method="Matching (test fixture)",
        estimand="ATT",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(-1.0, 1.0),
        alpha=0.05,
        n_obs=len(matched_data),
        detail=None,
        model_info=model_info or {},
        _citation_key="matching",
    )
    return sp.PSMatch2Result(
        result=result,
        matched_data=matched_data,
        treat="d",
        covariates=list(covariates or []),
        outcome=None,
        n_matches=1,
        common_support="none",
    )


# ======================================================================
# Matched frame attached to sp.match / sp.psm
# ======================================================================


class TestMatchedFrameAttached:

    def test_sp_match_attaches_matched_data(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        md = r.matched_data
        assert md is not None
        for col in [
            "_id",
            "_treated",
            "_pscore",
            "_support",
            "_weight",
            "_n1",
            "_nn",
            "_pdif",
            "_y",
        ]:
            assert col in md.columns
        # aligned to the original data (same number of rows)
        assert len(md) == len(psm_data)

    def test_sp_psm_alias_attaches_frame(self, psm_data):
        r = sp.psm(psm_data, y="y", d="d", X=["x1", "x2"])
        assert r.matched_data is not None
        assert "_weight" in r.matched_data.columns

    def test_default_att_unchanged_regression_guard(self, psm_data):
        """The additive frame must not perturb the historical ATT."""
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        # Recompute the matched-pair ATT straight from the frame and confirm
        # it equals result.estimate (the frame is consistent with the point
        # estimate, i.e. pure bookkeeping).
        md = r.matched_data
        treated = md["_treated"] == 1
        att_frame = float((md.loc[treated, "y"] - md.loc[treated, "_y"]).mean())
        assert att_frame == pytest.approx(r.estimate, abs=1e-9)

    def test_non_nearest_methods_have_no_frame(self, psm_data):
        r = sp.match(psm_data, y="y", treat="d", covariates=["x1", "x2"], method="cem")
        assert getattr(r, "matched_data", None) is None

    def test_ate_omits_att_style_matched_data(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            estimand="ATE",
        )
        assert getattr(r, "matched_data", None) is None
        assert "ATT assignment" in r.model_info["matched_data_note"]


# ======================================================================
# Stata-faithful semantics
# ======================================================================


class TestStataSemantics:

    def test_treated_weight_is_one(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        md = r.matched_data
        wt = md.loc[md["_treated"] == 1, "_weight"].dropna()
        assert np.allclose(wt, 1.0)

    def test_control_weights_sum_to_matched_treated(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        md = r.matched_data
        n_treated_matched = int((md["_weight"][md["_treated"] == 1].notna()).sum())
        ctrl_w = md.loc[md["_treated"] == 0, "_weight"].sum()  # skipna
        assert ctrl_w == pytest.approx(n_treated_matched)

    def test_control_nn_is_zero_not_nan(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        md = r.matched_data
        assert (md.loc[md["_treated"] == 0, "_nn"] == 0).all()

    def test_pdif_is_gap_to_nearest_only(self, psm_data):
        """_pdif under k=2 equals _pdif under k=1 (nearest neighbour gap)."""
        r1 = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
        )
        r2 = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=2,
        )
        treated = r1.matched_data["_treated"] == 1
        p1 = r1.matched_data.loc[treated, "_pdif"].to_numpy()
        p2 = r2.matched_data.loc[treated, "_pdif"].to_numpy()
        assert np.allclose(p1, p2, atol=1e-9, equal_nan=True)

    def test_k2_fractional_control_weights(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=2,
        )
        md = r.matched_data
        # each treated contributes 0.5 to each of 2 neighbours
        ctrl_w = md.loc[md["_treated"] == 0, "_weight"].dropna().to_numpy()
        assert np.any(np.isclose(ctrl_w % 0.5, 0.0))
        n_treated_matched = int(md.loc[md["_treated"] == 1, "_weight"].notna().sum())
        assert ctrl_w.sum() == pytest.approx(n_treated_matched)
        # two neighbour columns now exist
        assert "_n2" in md.columns

    def test_caliper_drops_treated_weight_missing(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=1,
            caliper=0.0005,
        )
        md = r.matched_data
        treated = md["_treated"] == 1
        # a tight caliper leaves some treated unmatched -> _weight NaN
        assert md.loc[treated, "_weight"].isna().any()


# ======================================================================
# Common support
# ======================================================================


class TestCommonSupport:

    def test_minmax_flags_and_trims(self, psm_data):
        r_none = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            common_support="none",
        )
        r_mm = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            common_support="minmax",
        )
        # 'none' marks everyone on support
        assert (r_none.matched_data["_support"] == 1).all()
        # 'minmax' can flag some treated off support
        supp = r_mm.matched_data["_support"]
        assert set(np.unique(supp.dropna())) <= {0.0, 1.0}
        # off-support treated are excluded from the matched sample
        md = r_mm.matched_data
        off = (md["_treated"] == 1) & (md["_support"] == 0)
        assert md.loc[off, "_weight"].isna().all()

    def test_common_support_mask_helper(self):
        ps = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
        treated = np.array([1, 1, 0, 0, 1])
        # controls span [0.5, 0.8]; treated outside are off support
        mask = common_support_mask(ps, treated, rule="minmax")
        assert mask.tolist() == [False, False, True, True, False]
        assert common_support_mask(ps, treated, rule="none").all()

    def test_invalid_common_support_raises(self, psm_data):
        with pytest.raises(ValueError, match="common_support"):
            sp.match(
                psm_data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="psm",
                common_support="bogus",
            )


# ======================================================================
# sp.psmatch2 surface
# ======================================================================


class TestPSMatch2Surface:

    def test_basic_run(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        assert isinstance(m, sp.PSMatch2Result)
        assert m.att == pytest.approx(m.result.estimate)
        assert abs(m.att - 2.0) < 0.5  # recovers true ATT = 2.0
        assert "psmatch2" in m.summary().lower()
        mi = m.result.model_info
        assert mi["propensity_model"] == "logit"
        assert mi["estimand_scope"] == "ATT"
        assert mi["outcome_status"] == "observed"
        assert mi["att_defined"] is True

    def test_outcome_optional(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", covariates=["x1", "x2"])
        assert np.isnan(m.att)
        assert "_weight" in m.matched_data.columns
        assert "_y" not in m.matched_data.columns  # no outcome -> no _y
        assert m.result.model_info["outcome_status"] == "omitted"
        assert m.result.model_info["att_defined"] is False
        assert "ATT matched-frame" in m.result.model_info["matched_frame_semantics"]

    def test_outcome_in_covariates_raises(self, psm_data):
        with pytest.raises(MethodIncompatibility, match="covariates"):
            sp.psmatch2(psm_data, treat="d", outcome="x1", covariates=["x1", "x2"])

    def test_requires_treat_and_covariates(self, psm_data):
        with pytest.raises(MethodIncompatibility, match="treat"):
            sp.psmatch2(psm_data, covariates=["x1", "x2"])

    def test_invalid_se_raises(self, psm_data):
        with pytest.raises(MethodIncompatibility, match="se"):
            sp.psmatch2(
                psm_data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                se="bogus",
            )

    def test_invalid_method_raises_taxonomy(self, psm_data):
        with pytest.raises(MethodIncompatibility, match="method"):
            sp.psmatch2(
                psm_data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="bad",
            )

    def test_scalar_covariate_name_is_one_column(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates="x1")
        assert m.covariates == ["x1"]
        assert "_weight" in m.matched_data.columns

    def test_neighbor_alias(self, psm_data):
        m1 = sp.psmatch2(
            psm_data, treat="d", outcome="y", covariates=["x1", "x2"], neighbor=2
        )
        m2 = sp.psmatch2(
            psm_data, treat="d", outcome="y", covariates=["x1", "x2"], n_matches=2
        )
        assert m1.att == pytest.approx(m2.att)

    def test_matched_sample_filters(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        samp = m.matched_sample()
        assert samp["_weight"].notna().all()
        assert len(samp) <= len(psm_data)

    def test_balance_before_after(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        bal = m.balance()
        assert {"smd_raw", "smd_weighted"} <= set(bal.table.columns)
        # well-specified PS on a CIA DGP should reduce imbalance here
        assert (
            bal.summary_stats["max_abs_smd_weighted"]
            <= bal.summary_stats["max_abs_smd_raw"] + 1e-9
        )

    def test_psplot_returns_axes(self, psm_data):
        pytest.importorskip("matplotlib")
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        fig, ax = m.psplot()
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_cite(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        assert "@" in m.cite()

    def test_summary_reports_treated_on_support_not_total_support(self):
        md = pd.DataFrame(
            {
                "d": [1, 1, 1, 0, 0],
                "_pscore": [0.2, 0.3, 0.9, 0.25, 0.35],
                "_support": [1, 1, 0, 1, 1],
                "_weight": [1.0, 1.0, np.nan, 1.0, 1.0],
            }
        )
        m = _manual_psmatch2(
            md,
            model_info={
                "n_treated": 3,
                "n_control": 2,
                "n_on_support": 4,
                "n_treated_on_support": 2,
                "n_matched_treated": 2,
                "se_method": "psmatch2",
            },
        )
        assert m.n_on_support == 2
        assert m.n_total_on_support == 4
        assert "on support: 2, matched: 2" in m.summary()

    def test_psplot_after_matching_excludes_unmatched_treated(self):
        pytest.importorskip("matplotlib")
        md = pd.DataFrame(
            {
                "d": [1, 1, 1, 0, 0, 0, 0],
                "_pscore": [0.2, 0.4, 0.8, 0.21, 0.41, 0.79, 0.9],
                "_support": [1, 1, 1, 1, 1, 1, 1],
                "_weight": [1.0, 1.0, np.nan, 1.0, 1.0, np.nan, np.nan],
            }
        )
        m = _manual_psmatch2(md)
        fig, ax = m.psplot(before=False)
        labels = [artist.get_label() for artist in ax.collections + ax.lines]
        assert "Matched treated (n=2)" in labels
        assert "Matched control (n=2)" in labels

        import matplotlib.pyplot as plt

        plt.close(fig)


# ======================================================================
# PSM-DID
# ======================================================================


class TestPSMDID:

    def test_pooled_recovers_effect(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        did = m.psm_did(panel, id="id", y="y", time="time", treat_time=1, treat="d")
        assert abs(did.estimate - panel.attrs["true_did"]) < 0.4
        assert did.model_info["did_term"] == "_did"

    def test_twfe_spec_drops_absorbed_mains(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        did = m.psm_did(
            panel,
            id="id",
            y="y",
            time="time",
            treat_time=1,
            treat="d",
            fixed_effects=["id", "time"],
            cluster="id",
        )
        # main effects absorbed by id/time FE -> only the interaction remains
        assert did.model_info["formula"] == "y ~ _did | id + time"
        assert abs(did.estimate - panel.attrs["true_did"]) < 0.4

    def test_post_column_accepted(self, psm_panel):
        base, panel = psm_panel
        panel = panel.assign(post=(panel["time"] >= 1).astype(int))
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        did = m.psm_did(panel, id="id", y="y", post="post", treat="d")
        assert abs(did.estimate - panel.attrs["true_did"]) < 0.4

    def test_unweighted_option(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        did = m.psm_did(
            panel, id="id", y="y", time="time", treat_time=1, treat="d", weight="none"
        )
        assert did.model_info["weight"] == "none"

    def test_fweight_affects_no_fe_did_and_ignores_panel_weight_columns(self):
        md = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "d": [1, 1, 1, 0, 0],
                "_weight": [1.0, 1.0, 1.0, 2.0, 1.0],
                "_support": [1, 1, 1, 1, 1],
            }
        )
        m = _manual_psmatch2(
            md,
            model_info={
                "n_treated": 3,
                "n_control": 2,
                "n_on_support": 5,
                "n_treated_on_support": 3,
                "n_matched_treated": 3,
            },
        )
        deltas = {0: 10.0, 1: 10.0, 2: 10.0, 3: 0.0, 4: 30.0}
        rows = []
        for unit, treated in zip(md["id"], md["d"]):
            for time in (0, 1):
                rows.append(
                    {
                        "id": int(unit),
                        "time": time,
                        "d": int(treated),
                        "y": deltas[int(unit)] * time,
                        "_weight": 99.0,
                        "_support": 0.0,
                    }
                )
        panel = pd.DataFrame(rows)

        weighted = m.psm_did(
            panel, id="id", y="y", time="time", treat_time=1, treat="d"
        )
        unweighted = m.psm_did(
            panel,
            id="id",
            y="y",
            time="time",
            treat_time=1,
            treat="d",
            weight="none",
        )

        assert weighted.estimate == pytest.approx(0.0, abs=1e-10)
        assert unweighted.estimate == pytest.approx(-5.0, abs=1e-10)
        assert weighted.model_info["weight_column"] not in {"_weight", "_support"}

    def test_invalid_psm_did_weight_raises(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        with pytest.raises(MethodIncompatibility, match="weight"):
            m.psm_did(
                panel,
                id="id",
                y="y",
                time="time",
                treat_time=1,
                treat="d",
                weight="fw",
            )

    def test_missing_post_and_time_raises(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        with pytest.raises(MethodIncompatibility, match="post"):
            m.psm_did(panel, id="id", y="y", treat="d")

    def test_missing_id_in_matching_data_raises(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base.drop(columns=["id"]), treat="d", covariates=["x1", "x2"])
        with pytest.raises(MethodIncompatibility, match="id"):
            m.psm_did(panel, id="id", y="y", time="time", treat_time=1, treat="d")

    def test_psm_did_no_matched_panel_rows_raises_taxonomy(self, psm_panel):
        base, panel = psm_panel
        m = sp.psmatch2(base, treat="d", covariates=["x1", "x2"])
        shifted = panel.copy()
        shifted["id"] += 10_000
        with pytest.raises(DataInsufficient, match="No matched panel rows"):
            m.psm_did(
                shifted,
                id="id",
                y="y",
                time="time",
                treat_time=1,
                treat="d",
            )


# ======================================================================
# Frame builder unit tests
# ======================================================================


class TestFrameBuilder:

    def test_builder_basic(self):
        # 2 treated (pos 0,1), 2 control (pos 2,3); 1:1 matching
        idx_t = np.array([0, 1])
        idx_c = np.array([2, 3])
        treated = np.array([1, 1, 0, 0])
        pscore = np.array([0.6, 0.4, 0.55, 0.45])
        # treated 0 -> control pos 0 (=row 2); treated 1 -> control pos 1 (row 3)
        matches = [np.array([0]), np.array([1])]
        weights = [np.array([1.0]), np.array([1.0])]
        frame = build_matched_frame(
            index=pd.RangeIndex(4),
            treated=treated,
            pscore=pscore,
            idx_t=idx_t,
            idx_c=idx_c,
            matches=matches,
            weights=weights,
            n_matches=1,
            outcome=np.array([10.0, 20.0, 1.0, 2.0]),
        )
        assert frame["_weight"].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert frame["_nn"].tolist() == [1.0, 1.0, 0.0, 0.0]
        # _n1 of treated points to control _id (row+1)
        assert frame["_n1"].iloc[0] == 3  # row 2 -> _id 3
        assert frame["_n1"].iloc[1] == 4  # row 3 -> _id 4
        # _pdif = |ps_t - ps_nearest|
        assert frame["_pdif"].iloc[0] == pytest.approx(abs(0.6 - 0.55))
        # _y = matched control outcome
        assert frame["_y"].iloc[0] == pytest.approx(1.0)

    def test_shared_control_accumulates_weight(self):
        # both treated match the SAME control -> its weight = 2
        idx_t = np.array([0, 1])
        idx_c = np.array([2])
        treated = np.array([1, 1, 0])
        pscore = np.array([0.6, 0.62, 0.61])
        matches = [np.array([0]), np.array([0])]
        weights = [np.array([1.0]), np.array([1.0])]
        frame = build_matched_frame(
            index=pd.RangeIndex(3),
            treated=treated,
            pscore=pscore,
            idx_t=idx_t,
            idx_c=idx_c,
            matches=matches,
            weights=weights,
            n_matches=1,
        )
        assert frame["_weight"].iloc[2] == pytest.approx(2.0)


# ======================================================================
# psmatch2 analytic SE
# ======================================================================


class TestPSMatch2SE:

    def test_se_method_option_changes_se(self, psm_data):
        ai = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="ai",
        )
        p = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="psmatch2",
        )
        assert ai.estimate == pytest.approx(p.estimate)  # same point estimate
        assert ai.se != pytest.approx(p.se)  # different SE estimator
        assert p.model_info["se_method"] == "psmatch2"

    def test_default_sp_match_se_is_ai_unchanged(self, psm_data):
        """sp.match default keeps the historical AI SE (JOSS-safe)."""
        default = sp.match(
            psm_data, y="y", treat="d", covariates=["x1", "x2"], method="psm"
        )
        ai = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="ai",
        )
        assert default.se == pytest.approx(ai.se)

    def test_psmatch2_se_formula_matches_helper(self, psm_data):
        from statspai.matching._matched_frame import psmatch2_se

        p = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="psmatch2",
        )
        md = p.matched_data
        se = psmatch2_se(
            md["y"].to_numpy(),
            md["_treated"].to_numpy(),
            md["_support"].to_numpy(),
            md["_weight"].to_numpy(),
        )
        assert p.se == pytest.approx(se, abs=1e-12)

    def test_psmatch2_default_se_is_psmatch2(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"])
        assert m.result.model_info["se_method"] == "psmatch2"

    def test_ai_robust_se_selected_by_ai_arg(self, psm_data):
        m = sp.psmatch2(psm_data, treat="d", outcome="y", covariates=["x1", "x2"], ai=2)
        assert m.result.model_info["se_method"] == "abadie_imbens"
        assert m.result.model_info["ai_matches"] == 2
        assert np.isfinite(m.se) and m.se > 0
        m1 = sp.psmatch2(
            psm_data, treat="d", outcome="y", covariates=["x1", "x2"], ai=1
        )
        assert m.att == pytest.approx(m1.att)  # SE-only option

    def test_abadie_imbens_se_string(self, psm_data):
        m = sp.psmatch2(
            psm_data,
            treat="d",
            outcome="y",
            covariates=["x1", "x2"],
            se="abadie_imbens",
        )
        assert m.result.model_info["se_method"] == "abadie_imbens"

    def test_ai_robust_via_sp_match(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="abadie_imbens",
            ai_matches=1,
        )
        assert r.model_info["se_method"] == "abadie_imbens"
        assert np.isfinite(r.se)

    def test_within_group_self_outcome_helper(self):
        from statspai.matching._matched_frame import _within_group_self_outcome

        y = np.array([10.0, 12.0, 20.0, 22.0])
        t = np.array([1, 1, 0, 0])
        ps = np.array([0.6, 0.65, 0.3, 0.35])
        sy = _within_group_self_outcome(y, t, ps, 1)
        assert sy[0] == 12.0 and sy[1] == 10.0  # treated pair
        assert sy[2] == 22.0 and sy[3] == 20.0  # control pair

    def test_bad_se_method_raises(self, psm_data):
        with pytest.raises(ValueError, match="se_method"):
            sp.match(
                psm_data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="psm",
                se_method="bogus",
            )


# ======================================================================
# Kernel / radius matching
# ======================================================================


class TestKernelRadius:

    def test_kernel_runs_and_recovers(self, psm_data):
        # a small bandwidth keeps the kernel local enough to recover ATT=2.0;
        # a wide bandwidth would average in far controls and bias upward.
        m = sp.psmatch2(
            psm_data,
            treat="d",
            outcome="y",
            covariates=["x1", "x2"],
            method="kernel",
            bwidth=0.06,
        )
        assert abs(m.att - 2.0) < 0.5  # true ATT = 2.0
        # kernel omits the discrete-neighbour columns (like Stata psmatch2)
        for c in ("_n1", "_nn", "_pdif"):
            assert c not in m.matched_data.columns
        assert "_weight" in m.matched_data.columns
        assert "_y" in m.matched_data.columns

    def test_kernel_control_weights_sum_to_matched_treated(self, psm_data):
        m = sp.psmatch2(
            psm_data,
            treat="d",
            outcome="y",
            covariates=["x1", "x2"],
            method="kernel",
            bwidth=0.5,
        )
        md = m.matched_data
        ctrl_w = md.loc[md["_treated"] == 0, "_weight"].sum()
        assert ctrl_w == pytest.approx(m.n_matched_treated, abs=1e-6)

    def test_radius_runs(self, psm_data):
        m = sp.psmatch2(
            psm_data,
            treat="d",
            outcome="y",
            covariates=["x1", "x2"],
            method="radius",
            caliper=0.05,
        )
        assert np.isfinite(m.att)
        assert abs(m.att - 2.0) < 0.5

    def test_radius_requires_caliper(self, psm_data):
        with pytest.raises(MethodIncompatibility, match="caliper"):
            sp.psmatch2(
                psm_data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="radius",
            )

    def test_kernel_tiny_bandwidth_drops_treated(self, psm_data):
        # a very small bandwidth orphans some treated -> off support
        m = sp.psmatch2(
            psm_data,
            treat="d",
            outcome="y",
            covariates=["x1", "x2"],
            method="kernel",
            bwidth=1e-4,
        )
        md = m.matched_data
        off = (md["_treated"] == 1) & (md["_support"] == 0)
        assert off.any()

    def test_bad_kernel_raises(self, psm_data):
        with pytest.raises(ValueError, match="kernel"):
            sp.match(
                psm_data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="kernel",
                kernel="bogus",
            )

    def test_kernel_radius_are_att_only(self, psm_data):
        with pytest.raises(ValueError, match="ATT"):
            sp.match(
                psm_data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="kernel",
                estimand="ATE",
            )
        with pytest.raises(ValueError, match="ATT"):
            sp.match(
                psm_data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="radius",
                caliper=0.2,
                estimand="ATE",
            )

    def test_kernel_via_sp_match(self, psm_data):
        r = sp.match(
            psm_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="radius",
            caliper=0.2,
        )
        assert r.matched_data is not None
        assert r.model_info["method"] == "radius"


# ======================================================================
# Registry
# ======================================================================


def test_psmatch2_registered():
    funcs = sp.list_functions()
    assert "psmatch2" in funcs
    spec = sp.describe_function("psmatch2")
    assert spec is not None
    params = {p["name"]: p for p in spec["params"]}
    assert params["outcome"]["required"] is False
    assert params["method"]["enum"] == ["neighbor", "kernel", "radius"]
    assert "kernel" in params
    assert params["bwidth"]["default"] == 0.06
    assert params["se"]["enum"] == ["psmatch2", "ai", "abadie_imbens"]
    assert "ai" in params
