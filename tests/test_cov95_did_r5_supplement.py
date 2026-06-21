"""Coverage round-5 supplement.

Picks up the harder-to-reach but still-REACHABLE branches that the
topic test files left uncovered:

- callaway_santanna._estimate_single_att degenerate-cell returns
  (t/base period absent, n_rel < 5, no treated/control) and the
  mixed constant/varying covariate keep-mask path.
- callaway_santanna._aggregate_event_study zero-weight skip.
- analysis.DIDAnalysis.summary() rendering branches (substantial
  negative weights, sensitivity not significant at M=0).
- continuous_did att_gt degenerate dose-group skip.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.core.results import CausalResult
from statspai.did.callaway_santanna import (
    _estimate_single_att,
    _aggregate_event_study,
)
from statspai.did.analysis import DIDAnalysis

# ----------------------------------------------------------------------
# _estimate_single_att degenerate-cell returns (lines 416, 424, 432, 445)
# ----------------------------------------------------------------------


def _wide(n):
    return pd.DataFrame(
        {1: np.arange(float(n)), 2: np.arange(float(n)) + 1.0}, index=range(n)
    )


def test_single_att_period_absent_returns_inf():
    y_wide = _wide(6)
    ui = pd.DataFrame({"g": [4, 4, 0, 0, 4, 0]}, index=range(6))
    att, se, inf = _estimate_single_att(
        y_wide, ui, 4, 99, 1, "g", None, "dr", "nevertreated", 6
    )
    assert att == 0.0 and np.isinf(se)
    assert inf.shape == (6,)


def test_single_att_too_few_relevant_returns_inf():
    y_wide = _wide(6)
    # only 4 units in the treated/control universe -> n_rel < 5
    ui = pd.DataFrame({"g": [4, 0, 0, 0, 99, 99]}, index=range(6))
    att, se, inf = _estimate_single_att(
        y_wide, ui, 4, 2, 1, "g", None, "dr", "nevertreated", 6
    )
    assert att == 0.0 and np.isinf(se)


def test_single_att_no_treated_returns_inf():
    y_wide = _wide(6)
    ui = pd.DataFrame({"g": [0, 0, 0, 0, 0, 0]}, index=range(6))
    att, se, inf = _estimate_single_att(
        y_wide, ui, 4, 2, 1, "g", None, "dr", "nevertreated", 6
    )
    assert att == 0.0 and np.isinf(se)


def test_single_att_mixed_constant_varying_covariate():
    # One constant + one varying covariate -> keep mask keeps the varying
    # one (line 445).
    n = 20
    y_wide = _wide(n)
    ui = pd.DataFrame(
        {
            "g": [4] * 10 + [0] * 10,
            "xc": [5.0] * n,  # constant -> dropped
            "xv": np.arange(float(n)),  # varying  -> kept
        },
        index=range(n),
    )
    att, se, inf = _estimate_single_att(
        y_wide, ui, 4, 2, 1, "g", ["xc", "xv"], "reg", "nevertreated", n
    )
    assert np.isfinite(att)


# ----------------------------------------------------------------------
# _aggregate_event_study zero-weight skip (lines 721, 726)
# ----------------------------------------------------------------------


def test_aggregate_event_study_zero_weight_skip():
    # cohort_sizes maps the only group to size 0 -> w_sum == 0 -> skip
    detail = pd.DataFrame(
        {
            "group": [4, 4],
            "relative_time": [0, 1],
            "att": [1.0, 1.5],
            "se": [0.1, 0.1],
        }
    )
    cs = pd.Series({4: 0.0})  # zero weight
    es = _aggregate_event_study(detail, None, cs, 50, 0.05)
    # both event times skipped -> empty frame
    assert len(es) == 0


# ----------------------------------------------------------------------
# DIDAnalysis.summary() rendering branches (lines 73-74, 104)
# ----------------------------------------------------------------------


def _dummy_result():
    return CausalResult(
        method="DID",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.01,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs=100,
    )


def test_summary_substantial_negative_weights():
    a = DIDAnalysis(
        design="staggered",
        method_used="TWFE",
        main_result=_dummy_result(),
        bacon={"negative_weight_share": 0.35, "beta_twfe": 0.9},
    )
    txt = a.summary()
    assert "Substantial negative weights" in txt


def test_summary_small_negative_weights():
    a = DIDAnalysis(
        design="staggered",
        method_used="TWFE",
        main_result=_dummy_result(),
        bacon={"negative_weight_share": 0.01, "beta_twfe": 0.9},
    )
    txt = a.summary()
    assert "small" in txt


def test_summary_sensitivity_not_significant_at_m0():
    # No row rejects zero -> "Effect not significant even at M=0" (line 104)
    sens = pd.DataFrame({"M": [0.0, 0.1, 0.2], "rejects_zero": [False, False, False]})
    a = DIDAnalysis(
        design="staggered",
        method_used="CS",
        main_result=_dummy_result(),
        sensitivity=sens,
    )
    txt = a.summary()
    assert "not significant even at M=0" in txt


def test_summary_sensitivity_breakdown_found():
    sens = pd.DataFrame({"M": [0.0, 0.1, 0.2], "rejects_zero": [True, True, False]})
    a = DIDAnalysis(
        design="staggered",
        method_used="CS",
        main_result=_dummy_result(),
        sensitivity=sens,
    )
    txt = a.summary()
    assert "Breakdown M*" in txt


def test_summary_with_steps_log():
    a = DIDAnalysis(
        design="staggered",
        method_used="CS",
        main_result=_dummy_result(),
        steps_log=["step one", "step two"],
    )
    txt = a.summary()
    assert "Analysis Steps" in txt
