"""Unit tests for the matched-frame builders in ``matching/_matched_frame``.

The builders are pure bookkeeping over an assignment the estimator has
already used for its point estimate, so the contract every test here checks
is the same one: *the frame's weights must rebuild the estimator's number.*

Integration coverage (the frames as attached by ``sp.match``) lives in
``tests/test_psmatch2.py``; Stata parity lives in
``tests/reference_parity/test_psmatch2_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.matching._matched_frame import (
    COL_STRATUM,
    build_ate_matched_frame,
    build_stratum_matched_frame,
    matched_columns,
)

# ======================================================================
# ATE frame
# ======================================================================


class TestBuildAteMatchedFrame:
    @pytest.fixture
    def tiny(self):
        """Two treated (positions 0, 1), two controls (positions 2, 3).

        Assignment: each treated takes one control 1:1 and vice versa, so
        every unit is used exactly once as somebody's match and every weight
        is 1 + 1 = 2.
        """
        return dict(
            index=pd.RangeIndex(4),
            treated=np.array([1, 1, 0, 0]),
            pscore=np.array([0.8, 0.7, 0.6, 0.5]),
            idx_t=np.array([0, 1]),
            idx_c=np.array([2, 3]),
            matches_tc=[np.array([0]), np.array([1])],
            weights_tc=[np.array([1.0]), np.array([1.0])],
            matches_ct=[np.array([1]), np.array([0])],
            weights_ct=[np.array([1.0]), np.array([1.0])],
            n_matches=1,
            outcome=np.array([10.0, 12.0, 4.0, 6.0]),
        )

    def test_weight_is_one_plus_match_count(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        np.testing.assert_allclose(frame["_weight"].to_numpy(), [2.0, 2.0, 2.0, 2.0])

    def test_reproduces_the_signed_ate_identity(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        w = frame["_weight"].to_numpy(dtype=float)
        t = frame["_treated"].to_numpy(dtype=float)
        y = tiny["outcome"]
        ate = np.sum((2 * t - 1) * w * y) / len(y)
        # Hand-computed: treated 0 vs control 2 -> 6; treated 1 vs control 3
        # -> 6; control 2 vs treated 1 -> 8; control 3 vs treated 0 -> 4.
        # ATE = (6 + 6 + 8 + 4) / 4 = 6.
        assert ate == pytest.approx(6.0)

    def test_both_arms_get_neighbour_columns(self, tiny):
        """Unlike the ATT frame, controls also name their matched partner."""
        frame = build_ate_matched_frame(**tiny)
        assert frame["_n1"].notna().all()
        assert (frame["_nn"] == 1).all()

    def test_control_neighbour_points_at_a_treated_unit(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        # _id is 1-based; control at position 2 matched treated position 1.
        assert frame.loc[2, "_n1"] == 2.0
        assert frame.loc[3, "_n1"] == 1.0

    def test_matched_outcome_is_the_counterfactual_arm(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        # Treated 0 <- control 2 (y=4); control 2 <- treated 1 (y=12).
        assert frame.loc[0, "_y"] == pytest.approx(4.0)
        assert frame.loc[2, "_y"] == pytest.approx(12.0)

    def test_pdif_is_the_propensity_gap_to_the_partner(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        assert frame.loc[0, "_pdif"] == pytest.approx(abs(0.8 - 0.6))
        assert frame.loc[2, "_pdif"] == pytest.approx(abs(0.6 - 0.7))

    def test_unused_unit_gets_missing_weight(self, tiny):
        """A unit nobody matched and that matched nobody leaves the sample."""
        tiny = dict(tiny)
        tiny["index"] = pd.RangeIndex(5)
        tiny["treated"] = np.array([1, 1, 0, 0, 0])
        tiny["pscore"] = np.array([0.8, 0.7, 0.6, 0.5, 0.01])
        tiny["idx_c"] = np.array([2, 3, 4])
        # Control 4 matches nobody and is matched by nobody.
        tiny["matches_ct"] = [np.array([1]), np.array([0]), np.array([], dtype=int)]
        tiny["weights_ct"] = [
            np.array([1.0]),
            np.array([1.0]),
            np.array([], dtype=float),
        ]
        tiny["outcome"] = np.array([10.0, 12.0, 4.0, 6.0, 99.0])
        frame = build_ate_matched_frame(**tiny)
        assert np.isnan(frame.loc[4, "_weight"])
        assert frame.loc[4, "_nn"] == 0

    def test_fractional_shares_accumulate(self, tiny):
        """With k=2 each partner receives half, so K_M sums to the same total."""
        tiny = dict(tiny)
        tiny["n_matches"] = 2
        tiny["matches_tc"] = [np.array([0, 1]), np.array([0, 1])]
        tiny["weights_tc"] = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        tiny["matches_ct"] = [np.array([0, 1]), np.array([0, 1])]
        tiny["weights_ct"] = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        frame = build_ate_matched_frame(**tiny)
        # Every unit is matched by both opposite-arm units at 1/2 each.
        np.testing.assert_allclose(frame["_weight"].to_numpy(), [2.0, 2.0, 2.0, 2.0])
        assert "_n2" in frame.columns

    def test_support_defaults_to_all_on_support(self, tiny):
        frame = build_ate_matched_frame(**tiny)
        assert (frame["_support"] == 1.0).all()

    def test_outcome_is_optional(self, tiny):
        tiny = dict(tiny)
        tiny["outcome"] = None
        frame = build_ate_matched_frame(**tiny)
        assert "_y" not in frame.columns


# ======================================================================
# Stratum frame
# ======================================================================


class TestBuildStratumMatchedFrame:
    @pytest.fixture
    def cells(self):
        """Cell A: 2 treated, 1 control.  Cell B: 1 treated, 2 controls.

        Cell C holds a lone control and is therefore dropped.
        """
        return dict(
            index=pd.RangeIndex(7),
            treated=np.array([1, 1, 0, 1, 0, 0, 0]),
            pscore=np.array([0.9, 0.85, 0.8, 0.4, 0.35, 0.3, 0.01]),
            stratum=np.array(["A", "A", "A", "B", "B", "B", "C"]),
            keep=np.array([True, True, True, True, True, True, False]),
            outcome=np.array([10.0, 12.0, 5.0, 20.0, 8.0, 10.0, 99.0]),
        )

    def test_treated_weight_is_one(self, cells):
        frame = build_stratum_matched_frame(**cells)
        treated = frame["_treated"] == 1
        np.testing.assert_allclose(frame.loc[treated, "_weight"].to_numpy(), 1.0)

    def test_control_weight_is_the_cell_arm_ratio(self, cells):
        frame = build_stratum_matched_frame(**cells)
        # Cell A: 2 treated / 1 control -> 2.0
        assert frame.loc[2, "_weight"] == pytest.approx(2.0)
        # Cell B: 1 treated / 2 controls -> 0.5
        assert frame.loc[4, "_weight"] == pytest.approx(0.5)
        assert frame.loc[5, "_weight"] == pytest.approx(0.5)

    def test_control_weights_sum_to_kept_treated(self, cells):
        frame = build_stratum_matched_frame(**cells)
        n_treated = int(((frame["_treated"] == 1) & frame["_weight"].notna()).sum())
        ctrl_w = frame.loc[frame["_treated"] == 0, "_weight"].sum()
        assert ctrl_w == pytest.approx(n_treated)

    def test_reproduces_the_stratified_att(self, cells):
        frame = build_stratum_matched_frame(**cells)
        w = frame["_weight"].to_numpy(dtype=float)
        t = frame["_treated"].to_numpy(dtype=float)
        y = cells["outcome"]
        ok = np.isfinite(w)
        att = np.average(y[ok & (t == 1)], weights=w[ok & (t == 1)]) - np.average(
            y[ok & (t == 0)], weights=w[ok & (t == 0)]
        )
        # Cell A tau = 11 - 5 = 6 (2 treated); cell B tau = 20 - 9 = 11 (1).
        # ATT = (2*6 + 1*11) / 3 = 23/3.
        assert att == pytest.approx(23.0 / 3.0)

    def test_dropped_cell_has_missing_weight_and_zero_support(self, cells):
        frame = build_stratum_matched_frame(**cells)
        assert np.isnan(frame.loc[6, "_weight"])
        assert frame.loc[6, "_support"] == 0.0

    def test_stratum_label_is_carried_through(self, cells):
        frame = build_stratum_matched_frame(**cells)
        assert frame[COL_STRATUM].tolist() == ["A", "A", "A", "B", "B", "B", "C"]

    def test_nn_counts_the_opposite_arm_in_the_cell(self, cells):
        frame = build_stratum_matched_frame(**cells)
        assert frame.loc[0, "_nn"] == 1.0  # treated in A sees 1 control
        assert frame.loc[2, "_nn"] == 2.0  # control in A sees 2 treated
        assert frame.loc[6, "_nn"] == 0.0  # dropped

    def test_matched_outcome_is_the_opposite_arm_cell_mean(self, cells):
        frame = build_stratum_matched_frame(**cells)
        assert frame.loc[0, "_y"] == pytest.approx(5.0)
        assert frame.loc[4, "_y"] == pytest.approx(20.0)

    def test_no_ordered_neighbour_columns(self, cells):
        frame = build_stratum_matched_frame(**cells)
        assert "_n1" not in frame.columns
        assert "_pdif" not in frame.columns

    def test_all_cells_dropped_yields_an_empty_matched_sample(self, cells):
        cells = dict(cells)
        cells["keep"] = np.zeros(7, dtype=bool)
        frame = build_stratum_matched_frame(**cells)
        assert frame["_weight"].isna().all()
        assert (frame["_support"] == 0).all()

    def test_outcome_is_optional(self, cells):
        cells = dict(cells)
        cells["outcome"] = None
        frame = build_stratum_matched_frame(**cells)
        assert "_y" not in frame.columns


class TestMatchedColumns:
    def test_stratum_layout_omits_neighbour_columns(self):
        cols = matched_columns(3, with_outcome=True, stratum=True)
        assert cols == [
            "_id",
            "_treated",
            "_pscore",
            "_support",
            "_weight",
            "_stratum",
            "_nn",
            "_y",
        ]

    def test_neighbour_layout_is_unchanged(self):
        cols = matched_columns(2, with_outcome=True)
        assert cols == [
            "_id",
            "_treated",
            "_pscore",
            "_support",
            "_weight",
            "_n1",
            "_n2",
            "_nn",
            "_pdif",
            "_y",
        ]

    def test_kernel_layout_drops_neighbours_but_keeps_outcome(self):
        cols = matched_columns(1, with_outcome=True, neighbors=False)
        assert cols == ["_id", "_treated", "_pscore", "_support", "_weight", "_y"]
