"""Unit tests for ``statspai.matching._weight_modes``.

These cover the primitives in isolation; the Stata-facing behaviour they
implement is pinned in ``tests/reference_parity/test_psmdid_weight_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import MethodIncompatibility
from statspai.matching._weight_modes import (
    MAX_EXPANDED_ROWS,
    WEIGHT_MODES,
    expand_frequency_weights,
    integerize_weights,
    resolve_weight_mode,
    weight_regime_info,
)


class TestResolveWeightMode:
    @pytest.mark.parametrize("mode", WEIGHT_MODES)
    def test_accepts_supported_modes(self, mode):
        assert resolve_weight_mode(mode) == mode

    @pytest.mark.parametrize(
        "raw,expected",
        [("AWEIGHT", "aweight"), ("  fweight ", "fweight"), ("None", "none")],
    )
    def test_normalises_case_and_whitespace(self, raw, expected):
        assert resolve_weight_mode(raw) == expected

    def test_rejects_unknown_string(self):
        with pytest.raises(MethodIncompatibility, match="weight must be one of"):
            resolve_weight_mode("pweight")

    @pytest.mark.parametrize("bad", [None, 1, True, 1.0, ["aweight"]])
    def test_rejects_non_string(self, bad):
        with pytest.raises(MethodIncompatibility, match="weight must be a string"):
            resolve_weight_mode(bad)

    def test_error_carries_recovery_hint(self):
        with pytest.raises(MethodIncompatibility) as exc:
            resolve_weight_mode("pweight")
        assert "aweight" in exc.value.recovery_hint
        assert exc.value.diagnostics["supported"] == list(WEIGHT_MODES)


class TestIntegerizeWeights:
    def test_exact_integers(self):
        out = integerize_weights(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    def test_float_accumulation_noise_is_tolerated(self):
        """1/3 + 1/3 + 1/3 does not land exactly on 1.0."""
        noisy = np.array([1 / 3 + 1 / 3 + 1 / 3, 2.9999999999999996])
        out = integerize_weights(noisy)
        np.testing.assert_array_equal(out, np.array([1, 3]))

    def test_genuinely_fractional_returns_none(self):
        assert integerize_weights(np.array([1.0, 0.5])) is None

    def test_half_shares_from_two_neighbour_matching(self):
        assert integerize_weights(np.array([1.0, 1.5, 0.5])) is None


class TestExpandFrequencyWeights:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"y": [1.0, 2.0, 3.0], "w": [1.0, 2.0, 3.0]})

    def test_row_count_is_sum_of_weights(self, frame):
        out = expand_frequency_weights(frame, "w")
        assert len(out) == 6

    def test_rows_are_replicated_in_order(self, frame):
        out = expand_frequency_weights(frame, "w")
        assert out["y"].tolist() == [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]

    def test_index_is_reset(self, frame):
        out = expand_frequency_weights(frame, "w")
        assert out.index.tolist() == list(range(6))

    def test_weight_column_is_retained(self, frame):
        assert "w" in expand_frequency_weights(frame, "w").columns

    def test_expansion_reproduces_weighted_least_squares_point_estimate(self):
        """Expansion must not move the coefficient — only the df."""
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "w": rng.integers(1, 4, size=n).astype(float),
            }
        )
        df["y"] = 2.0 + 3.0 * df["x"] + rng.normal(size=n)

        X = np.column_stack([np.ones(n), df["x"].to_numpy()])
        w = df["w"].to_numpy()
        beta_wls = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (df["y"] * w))

        big = expand_frequency_weights(df, "w")
        Xb = np.column_stack([np.ones(len(big)), big["x"].to_numpy()])
        beta_exp = np.linalg.solve(Xb.T @ Xb, Xb.T @ big["y"].to_numpy())

        np.testing.assert_allclose(beta_exp, beta_wls, rtol=1e-12)

    def test_fractional_weight_rejected(self, frame):
        frame.loc[0, "w"] = 1.5
        with pytest.raises(MethodIncompatibility, match="must be integers"):
            expand_frequency_weights(frame, "w")

    def test_fractional_error_names_the_offending_value(self, frame):
        frame.loc[0, "w"] = 1.5
        with pytest.raises(MethodIncompatibility) as exc:
            expand_frequency_weights(frame, "w")
        assert exc.value.diagnostics["example"] == pytest.approx(1.5)
        assert "aweight" in exc.value.recovery_hint

    def test_missing_weight_rejected(self, frame):
        frame.loc[1, "w"] = np.nan
        with pytest.raises(MethodIncompatibility, match="finite"):
            expand_frequency_weights(frame, "w")

    def test_zero_weight_rejected(self, frame):
        frame.loc[1, "w"] = 0.0
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            expand_frequency_weights(frame, "w")

    def test_negative_weight_rejected(self, frame):
        frame.loc[1, "w"] = -2.0
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            expand_frequency_weights(frame, "w")

    def test_unknown_column_rejected(self, frame):
        with pytest.raises(MethodIncompatibility, match="not in the frame"):
            expand_frequency_weights(frame, "nope")

    def test_absurd_expansion_is_refused_not_attempted(self):
        """A probability weight passed as a frequency must not OOM the box."""
        df = pd.DataFrame({"y": [1.0], "w": [float(MAX_EXPANDED_ROWS + 1)]})
        with pytest.raises(MethodIncompatibility, match="limit"):
            expand_frequency_weights(df, "w")

    def test_context_appears_in_message(self, frame):
        frame.loc[0, "w"] = 1.5
        with pytest.raises(MethodIncompatibility, match="my-caller"):
            expand_frequency_weights(frame, "w", context="my-caller")


class TestWeightRegimeInfo:
    def test_none_mode(self):
        info = weight_regime_info("none", np.array([1.0, 2.0]))
        assert info["weight"] == "none"
        assert info["weight_semantics"] == "unweighted"
        assert "weight_sum" not in info

    def test_aweight_mode_reports_row_df(self):
        info = weight_regime_info("aweight", np.array([1.0, 2.0, 3.0]))
        assert "n_rows - k" in info["weight_semantics"]
        assert info["weight_sum"] == pytest.approx(6.0)
        assert info["weight_is_integer"] is True

    def test_fweight_mode_reports_summed_df(self):
        info = weight_regime_info("fweight", np.array([1.0, 2.0]))
        assert "sum(w) - k" in info["weight_semantics"]

    def test_nan_weights_excluded_from_the_sum(self):
        info = weight_regime_info("aweight", np.array([1.0, np.nan, 2.0]))
        assert info["weight_sum"] == pytest.approx(3.0)

    def test_fractional_weights_flagged(self):
        info = weight_regime_info("aweight", np.array([0.5, 1.5]))
        assert info["weight_is_integer"] is False
