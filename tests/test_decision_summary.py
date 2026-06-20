"""Tests for ``CausalResult.decision_summary`` (practical-significance /
ROPE decision layer).

The verdict taxonomy is exercised by constructing ``CausalResult``
objects directly so each CI-vs-ROPE branch is hit deterministically,
without depending on any estimator's numerics.
"""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _mk(estimate, se, ci, *, pvalue=0.05, alpha=0.05, method="DID", estimand="ATT"):
    return sp.CausalResult(
        method=method,
        estimand=estimand,
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=1000,
    )


# --------------------------------------------------------------------- #
#  Verdict taxonomy
# --------------------------------------------------------------------- #


def test_meaningful_effect():
    # CI [1.6, 2.4] lies entirely beyond ROPE (-0.5, 0.5).
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=0.5)
    assert s.verdict == "meaningful_effect"
    assert s.statistically_significant is True
    assert s.practically_significant is True
    assert s.ci_vs_rope == "outside"
    assert "matter" in s.text


def test_practically_equivalent_to_zero():
    # CI [-0.096, 0.296] inside ROPE and includes 0 -> equivalence.
    s = _mk(0.1, 0.1, (-0.096, 0.296), pvalue=0.32).decision_summary(rope=0.5)
    assert s.verdict == "equivalent"
    assert s.statistically_significant is False
    assert s.practically_significant is False
    assert s.ci_vs_rope == "inside"


def test_significant_but_negligible():
    # Precise small effect: CI [0.06, 0.14] excludes 0 but inside ROPE.
    s = _mk(0.1, 0.02, (0.06, 0.14), pvalue=1e-6).decision_summary(rope=0.5)
    assert s.verdict == "negligible_significant"
    assert s.statistically_significant is True
    assert s.practically_significant is False
    assert "negligible" in s.text


def test_significant_uncertain_magnitude():
    # CI [0.01, 1.19] excludes 0 but straddles the ROPE upper bound.
    s = _mk(0.6, 0.3, (0.01, 1.19), pvalue=0.045).decision_summary(rope=0.5)
    assert s.verdict == "significant_uncertain_magnitude"
    assert s.statistically_significant is True
    assert s.practically_significant is None
    assert s.ci_vs_rope == "overlap"


def test_inconclusive_underpowered():
    # CI [-0.68, 1.28] includes 0 and the ROPE bound -> inconclusive.
    s = _mk(0.3, 0.5, (-0.68, 1.28), pvalue=0.55).decision_summary(rope=0.5)
    assert s.verdict == "inconclusive"
    assert s.statistically_significant is False
    assert s.practically_significant is None
    assert "underpowered" in s.text


# --------------------------------------------------------------------- #
#  Honest abstention when no ROPE is supplied
# --------------------------------------------------------------------- #


def test_no_rope_significant():
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary()
    assert s.verdict == "statistically_significant"
    assert s.practically_significant is None
    assert "smallest effect size of interest" in s.text


def test_no_rope_not_significant_is_not_no_effect():
    s = _mk(0.1, 0.2, (-0.29, 0.49), pvalue=0.62).decision_summary()
    assert s.verdict == "not_significant"
    # Must NOT claim "no effect" — only that significance wasn't reached.
    assert "not evidence of no effect" in s.text


# --------------------------------------------------------------------- #
#  Input handling
# --------------------------------------------------------------------- #


def test_sesoi_is_alias_for_symmetric_rope():
    base = _mk(0.1, 0.02, (0.06, 0.14), pvalue=1e-6)
    a = base.decision_summary(rope=0.5)
    b = base.decision_summary(sesoi=0.5)
    assert a.verdict == b.verdict
    assert b.rope == (-0.5, 0.5)


def test_asymmetric_rope_tuple():
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=(-1.0, 1.0))
    assert s.rope == (-1.0, 1.0)
    assert s.verdict == "meaningful_effect"


def test_rope_and_sesoi_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=0.5, sesoi=0.5)


def test_negative_scalar_rope_rejected():
    with pytest.raises(ValueError, match="positive"):
        _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=-0.5)


def test_bad_tuple_rope_rejected():
    with pytest.raises(ValueError, match="lo < hi"):
        _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=(0.5, -0.5))


# --------------------------------------------------------------------- #
#  Alpha override recomputes a normal-approx interval
# --------------------------------------------------------------------- #


def test_alpha_override_recomputes_ci():
    s = _mk(1.0, 0.5, (0.02, 1.98), alpha=0.05).decision_summary(rope=0.3, alpha=0.10)
    # 90% normal-approx CI is narrower than the stored 95% one.
    lo, hi = s.ci
    assert lo > 0.02 and hi < 1.98
    assert "normal-approx" in s.text
    assert abs(lo - (1.0 - 1.6448536 * 0.5)) < 1e-3


# --------------------------------------------------------------------- #
#  Degenerate estimate
# --------------------------------------------------------------------- #


def test_nonfinite_estimate_is_undefined():
    s = _mk(np.nan, 0.2, None).decision_summary(rope=0.5)
    assert s.verdict == "undefined"
    assert s.practically_significant is None
    assert "no decision" in s.text


# --------------------------------------------------------------------- #
#  Output shapes
# --------------------------------------------------------------------- #


def test_table_is_one_row_dataframe():
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=0.5)
    assert isinstance(s.table, pd.DataFrame)
    assert len(s.table) == 1
    assert s.table.loc[0, "verdict"] == "meaningful_effect"
    assert {"estimate", "ci_low", "ci_high", "rope_low", "label"} <= set(
        s.table.columns
    )


def test_to_dict_is_json_serializable():
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=0.5)
    payload = s.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["verdict"] == "meaningful_effect"
    assert payload["rope"] == [-0.5, 0.5]


def test_str_returns_prose():
    s = _mk(2.0, 0.2, (1.6, 2.4)).decision_summary(rope=0.5)
    assert str(s) == s.text
    assert s.text.startswith("DID estimates the ATT")


# --------------------------------------------------------------------- #
#  Diagnostics tie-in: a meaningful effect on a violated assumption
#  is flagged.
# --------------------------------------------------------------------- #


def test_violation_flag_appended_to_text():
    res = sp.CausalResult(
        method="DID",
        estimand="ATT",
        estimate=2.0,
        se=0.2,
        pvalue=0.001,
        ci=(1.6, 2.4),
        alpha=0.05,
        n_obs=1000,
        model_info={"pretrend_test": {"pvalue": 0.001}},  # parallel trends rejected
    )
    s = res.decision_summary(rope=0.5)
    assert s.verdict == "meaningful_effect"
    assert "Caution" in s.text
    assert "pretrend" in s.text
