"""Tests for ``sp.brief(result)`` / ``result.brief()``.

Pins: bounded length (~120 chars), method-name surfacing for both
CausalResult and EconometricResults, significance stars, NaN-safe
formatting, alias equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _causal(method: str = "did_2x2", *,
             estimate=1.5, se=0.5, pvalue=0.003, ci=(0.5, 2.5),
             n_obs=1000, model_info=None):
    return CausalResult(
        method=method, estimand="ATT",
        estimate=estimate, se=se, pvalue=pvalue, ci=ci,
        alpha=0.05, n_obs=n_obs,
        model_info=model_info or {},
        _citation_key=method,
    )


@pytest.fixture
def regress_result():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=300),
        "x1": rng.normal(size=300),
        "treat": rng.integers(0, 2, size=300),
    })
    df["y"] = df["y"] + 2.0 * df["treat"] + 0.5 * df["x1"]
    return sp.regress("y ~ x1 + treat", data=df, robust="hc1")


# ---------------------------------------------------------------------------
#  Top-level export
# ---------------------------------------------------------------------------


class TestExport:

    def test_callable(self):
        assert callable(sp.brief)

    def test_in_all(self):
        assert "brief" in sp.__all__


# ---------------------------------------------------------------------------
#  Length budget
# ---------------------------------------------------------------------------


class TestLengthBudget:

    def test_causal_under_120_chars(self):
        s = sp.brief(_causal())
        assert len(s) <= 130, (
            f"causal brief is {len(s)} chars; cap is ~120 (with a "
            "small slack for ATTRIBUTION + violations)")

    def test_regress_under_140_chars(self, regress_result):
        s = regress_result.brief()
        assert len(s) <= 140

    def test_long_method_name_truncated(self):
        s = sp.brief(_causal(
            method="some_extremely_long_method_name_we_want_truncated"))
        # Must still be a single line and reasonably short.
        assert "\n" not in s
        assert len(s) <= 140


# ---------------------------------------------------------------------------
#  Content
# ---------------------------------------------------------------------------


class TestCausalContent:

    def test_method_surfaced(self):
        s = sp.brief(_causal(method="did_2x2"))
        assert "did_2x2" in s

    def test_estimate_and_se_present(self):
        s = sp.brief(_causal(estimate=0.412, se=0.087))
        assert "0.412" in s
        assert "0.087" in s

    def test_significance_stars_at_three_levels(self):
        # *** (p<0.01)
        assert "***" in sp.brief(_causal(pvalue=0.001))
        # ** (p<0.05)
        assert "**" in sp.brief(_causal(pvalue=0.03))
        # * (p<0.10)
        assert "*" in sp.brief(_causal(pvalue=0.08))
        # No stars at p≥0.10
        s = sp.brief(_causal(pvalue=0.5))
        assert "***" not in s and "**" not in s
        # Ignore the lone "*" check because it can match other tokens

    def test_n_obs_formatted_with_thousands_separator(self):
        s = sp.brief(_causal(n_obs=12345))
        assert "12,345" in s

    def test_violation_flag_appears_when_present(self):
        # Build a result with a real pretrend failure → violations()
        # fires, brief() should surface a warning glyph.
        r = _causal(model_info={"pretrend_test": {"pvalue": 0.005}})
        s = sp.brief(r)
        assert "pretrend" in s

    def test_nan_estimate_does_not_crash(self):
        s = sp.brief(_causal(estimate=float("nan"), se=float("nan")))
        # Must produce a string and not raise. Em-dash placeholder.
        assert isinstance(s, str)
        assert "—" in s or "nan" not in s.lower()


class TestEconometricContent:

    def test_method_surfaced_from_model_info(self, regress_result):
        s = regress_result.brief()
        # OLS reports 'Least Squares' or similar via model_info.
        assert "?" not in s.split("]")[0], (
            "method label should come from model_info, not show '?'")

    def test_n_terms_reported(self, regress_result):
        s = regress_result.brief()
        assert "k=" in s

    def test_n_obs_reported(self, regress_result):
        s = regress_result.brief()
        assert "N=300" in s

    def test_best_term_surfaced(self, regress_result):
        # treat or x1 — the most-significant non-intercept coefficient.
        s = regress_result.brief()
        assert "best:" in s
        # Surfaced term must be one of the model's covariates.
        assert ("treat" in s or "x1" in s)


# ---------------------------------------------------------------------------
#  Alias parity
# ---------------------------------------------------------------------------


class TestAliasParity:

    def test_method_and_top_level_agree_for_causal(self):
        r = _causal()
        assert r.brief() == sp.brief(r)

    def test_method_and_top_level_agree_for_regress(self, regress_result):
        assert regress_result.brief() == sp.brief(regress_result)

    def test_brief_round_trips_through_str(self, regress_result):
        # No newlines, no NULs, no unprintable bytes.
        s = sp.brief(regress_result)
        assert "\n" not in s
        assert "\x00" not in s


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_unknown_object_does_not_crash(self):
        # A bare object with no useful attributes should produce a
        # graceful fallback string, not an exception.
        class Stub:
            method = "stub"
        s = sp.brief(Stub())
        assert isinstance(s, str)
        assert "stub" in s
