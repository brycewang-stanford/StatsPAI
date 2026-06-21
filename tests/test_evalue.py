"""Comprehensive tests for the E-value module (sp.evalue / sp.evalue_rd /
sp.bias_factor).

The reference numbers are the R ``EValue`` package outputs (VanderWeele,
Ding, Mathur, Smith), hard-coded here so the suite runs without R. The
live three-language parity harness is ``tests/r_parity/23_evalue.{py,R}``.

Covers: every effect measure (RR/OR/HR rare+common, MD/SMD, OLS, RD),
non-null ``true`` E-values, the confidence-interval null-crossing guard,
mathematical invariants, input validation, and backwards compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.diagnostics.evalue import _threshold
from statspai.exceptions import MethodIncompatibility

# Tolerance vs the R EValue package (closed-form / deterministic grid).
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Parity with R EValue across every measure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,r_point,r_ci",
    [
        # (call kwargs, R evalues.* point, R CI-limit E-value)
        (dict(estimate=2.5, ci=(1.8, 3.2), measure="RR"), 4.436492, 3.000000),
        (dict(estimate=0.6, ci=(0.4, 0.9), measure="RR"), 2.720759, 1.462475),
        (dict(estimate=1.1, ci=(0.9, 1.3), measure="RR"), 1.431662, 1.000000),
        (dict(estimate=2.5, ci=(1.8, 3.2), measure="RR", true=1.5), 2.720759, 1.689898),
        (
            dict(estimate=2.0, ci=(1.5, 2.7), measure="OR", rare=False),
            2.179580,
            1.749392,
        ),
        (
            dict(estimate=2.0, ci=(1.5, 2.7), measure="OR", rare=True),
            3.414214,
            2.366025,
        ),
        (
            dict(estimate=1.5, ci=(1.1, 2.0), measure="HR", rare=False),
            1.978541,
            1.338382,
        ),
        (
            dict(estimate=1.5, ci=(1.1, 2.0), measure="HR", rare=True),
            2.366025,
            1.431662,
        ),
        (dict(estimate=0.3, se=0.1, measure="MD"), 1.956110, 1.430704),
        (dict(estimate=0.3, se=0.1, measure="SMD"), 1.956110, 1.430704),
        (dict(estimate=0.5, se=0.1, sd=2.0, measure="OLS"), 1.821775, 1.561607),
    ],
)
def test_parity_with_r_evalue(kwargs, r_point, r_ci):
    out = sp.evalue(**kwargs)
    assert out["evalue_estimate"] == pytest.approx(r_point, rel=RTOL)
    assert out["evalue_ci"] == pytest.approx(r_ci, rel=RTOL)


def test_rd_table_parity_with_r():
    # R: evalues.RD(200,150,100,250) -> point 3.414214, lower 2.726306
    out = sp.evalue_rd(200, 150, 100, 250)
    assert out["evalue_estimate"] == pytest.approx(3.414214, rel=RTOL)
    assert out["evalue_ci"] == pytest.approx(2.726306, rel=RTOL)
    assert out["rd"] == pytest.approx(0.285714, rel=RTOL)


def test_point_estimate_closed_form():
    rr = 1.3251
    out = sp.evalue(estimate=rr, measure="RR")
    assert out["evalue_estimate"] == pytest.approx(
        rr + np.sqrt(rr * (rr - 1)), abs=1e-9
    )
    assert out["evalue_ci"] is None  # no CI/SE given


# ---------------------------------------------------------------------------
# Confidence-interval null-crossing guard (regression for the bug the
# NHEFS / What If reproduction surfaced)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimate,ci",
    [
        (0.90, (0.79, 1.22)),  # RR < 1, CI crosses the null
        (1.10, (0.95, 1.27)),  # RR > 1, CI crosses the null
        (1.00, (0.80, 1.25)),  # exactly at the null
    ],
)
def test_ci_crossing_null_returns_one(estimate, ci):
    assert sp.evalue(estimate=estimate, ci=ci, measure="RR")[
        "evalue_ci"
    ] == pytest.approx(1.0, abs=1e-12)


def test_borderline_limit_at_null_is_one():
    assert sp.evalue(estimate=1.30, ci=(1.00, 1.60), measure="RR")[
        "evalue_ci"
    ] == pytest.approx(1.0, abs=1e-12)


def test_protective_ci_clearing_null():
    out = sp.evalue(estimate=0.70, ci=(0.55, 0.88), measure="RR")
    assert out["evalue_ci"] > 1.0


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rr", [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 12.0])
def test_evalue_never_below_one(rr):
    assert sp.evalue(estimate=rr, measure="RR")["evalue_estimate"] >= 1.0 - 1e-12


@pytest.mark.parametrize("rr", [1.2, 1.5, 2.0, 3.0, 7.0])
def test_symmetry_rr_and_reciprocal(rr):
    # E-value is symmetric under RR <-> 1/RR.
    a = sp.evalue(estimate=rr, measure="RR")["evalue_estimate"]
    b = sp.evalue(estimate=1.0 / rr, measure="RR")["evalue_estimate"]
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("rr", [1.3, 1.8, 2.5, 4.0])
def test_point_evalue_at_least_ci_evalue(rr):
    out = sp.evalue(estimate=rr, ci=(rr * 0.7, rr * 1.3), measure="RR")
    assert out["evalue_estimate"] >= out["evalue_ci"] - 1e-9


@pytest.mark.parametrize("rr", [1.5, 2.0, 3.41421356, 5.0])
def test_bias_factor_inverts_threshold(rr):
    # threshold(rr) is the E s.t. bias_factor(E, E) == rr.
    e = _threshold(rr)
    assert sp.bias_factor(e, e) == pytest.approx(rr, rel=1e-9)


def test_monotonic_in_rr():
    rrs = [1.1, 1.5, 2.0, 3.0, 5.0]
    evs = [sp.evalue(estimate=r, measure="RR")["evalue_estimate"] for r in rrs]
    assert evs == sorted(evs)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_measure():
    with pytest.raises(ValueError):
        sp.evalue(estimate=2.0, measure="bogus")


def test_negative_rr_rejected():
    with pytest.raises(ValueError):
        sp.evalue(estimate=-1.0, measure="RR")


def test_negative_se_rejected():
    with pytest.raises(ValueError):
        sp.evalue(estimate=2.0, se=-0.1, measure="RR")


def test_nonfinite_inputs_rejected():
    with pytest.raises(MethodIncompatibility, match="estimate"):
        sp.evalue(estimate=np.nan, measure="RR")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.evalue(estimate=2.0, measure="RR", alpha=0.0)
    with pytest.raises(MethodIncompatibility, match="rare"):
        sp.evalue(estimate=2.0, measure="OR", rare="yes")


def test_ci_misordered_rejected():
    with pytest.raises(ValueError):
        sp.evalue(estimate=2.0, ci=(3.0, 1.0), measure="RR")


def test_ci_shape_and_finiteness_rejected():
    with pytest.raises(MethodIncompatibility, match="two-element"):
        sp.evalue(estimate=2.0, ci=(1.0, 2.0, 3.0), measure="RR")
    with pytest.raises(MethodIncompatibility, match="finite"):
        sp.evalue(estimate=2.0, ci=(1.0, np.inf), measure="RR")


def test_estimate_outside_ci_rejected():
    with pytest.raises(ValueError):
        sp.evalue(estimate=5.0, ci=(1.1, 2.0), measure="RR")


def test_ols_requires_sd():
    with pytest.raises(ValueError):
        sp.evalue(estimate=0.5, se=0.1, measure="OLS")


def test_rd_negative_cells_rejected():
    with pytest.raises(ValueError):
        sp.evalue_rd(-1, 150, 100, 250)


def test_rd_rejects_bad_alpha_grid_and_nonfinite_cells():
    with pytest.raises(MethodIncompatibility, match="grid"):
        sp.evalue_rd(200, 150, 100, 250, grid=0.0)
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.evalue_rd(200, 150, 100, 250, alpha=1.0)
    with pytest.raises(MethodIncompatibility, match="n11"):
        sp.evalue_rd(np.nan, 150, 100, 250)


def test_rd_requires_positive_rd():
    # p1 < p0 -> risk difference negative -> must relabel.
    with pytest.raises(ValueError):
        sp.evalue_rd(100, 250, 200, 150)


def test_bias_factor_requires_associations_above_one():
    with pytest.raises(ValueError):
        sp.bias_factor(0.5, 2.0)


def test_bias_factor_rejects_nonfinite_inputs():
    with pytest.raises(MethodIncompatibility, match="rr_eu"):
        sp.bias_factor(np.inf, 2.0)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_legacy_keys_present():
    out = sp.evalue(estimate=1.5, ci=(1.1, 2.0), measure="RR")
    for k in (
        "evalue_estimate",
        "evalue_ci",
        "rr_estimate",
        "rr_ci",
        "measure",
        "original_estimate",
        "interpretation",
        "ci",
    ):
        assert k in out


def test_rare_outcome_alias_matches_rare():
    a = sp.evalue(estimate=2.0, measure="OR", rare_outcome=True)["evalue_estimate"]
    b = sp.evalue(estimate=2.0, measure="OR", rare=True)["evalue_estimate"]
    assert a == pytest.approx(b)


def test_evalue_from_result_roundtrip():
    class _R:
        estimate = 2.0
        se = None
        ci = (1.5, 2.7)
        alpha = 0.05

    out = sp.evalue_from_result(_R(), measure="RR")
    assert out["evalue_estimate"] == pytest.approx(3.414214, rel=RTOL)


def test_evalue_from_result_bad_type():
    with pytest.raises(MethodIncompatibility):
        sp.evalue_from_result(object())


def test_interpretation_mentions_evalue():
    out = sp.evalue(estimate=2.5, ci=(1.8, 3.2), measure="RR")
    assert "E-value" in out["interpretation"]


# ---------------------------------------------------------------------------
# Branch coverage: alternative input paths
# ---------------------------------------------------------------------------


def test_ratio_with_se_builds_ci():
    out = sp.evalue(estimate=2.0, se=0.15, measure="RR")
    assert "ci" in out and out["evalue_ci"] is not None
    assert out["evalue_ci"] < out["evalue_estimate"]


def test_ratio_true_negative_rejected():
    with pytest.raises(ValueError):
        sp.evalue(estimate=2.0, measure="RR", true=-1.0)


def test_md_with_ci_instead_of_se():
    out = sp.evalue(estimate=0.3, ci=(0.1, 0.5), measure="SMD")
    assert out["evalue_estimate"] == pytest.approx(1.956110, rel=RTOL)
    assert out["evalue_ci"] is not None


def test_ols_with_ci_instead_of_se():
    out = sp.evalue(estimate=0.5, ci=(0.3, 0.7), sd=2.0, measure="OLS")
    assert out["evalue_estimate"] == pytest.approx(1.821775, rel=RTOL)
    assert out["evalue_ci"] is not None


@pytest.mark.parametrize(
    "call",
    [
        dict(estimate=0.2, measure="DIFF"),
        dict(estimate=0.2, ci=(0.05, 0.35), measure="RD"),
        dict(estimate=0.2, se=0.07, measure="DIFF"),
    ],
)
def test_diff_scalar_path(call):
    out = sp.evalue(**call)
    assert out["evalue_estimate"] >= 1.0
    assert out["measure"] in ("DIFF", "RD")


def test_threshold_true_greater_than_estimate():
    # estimate below the non-null reference -> threshold's true>x branch.
    out = sp.evalue(estimate=1.2, measure="RR", true=2.0)
    assert out["evalue_estimate"] == pytest.approx(2.720759, rel=RTOL)


def test_evalue_rd_ci_crosses_true_returns_one():
    # Small, uncertain effect -> lower confidence limit <= true=0 -> CI E=1.
    out = sp.evalue_rd(22, 78, 18, 82)
    assert out["evalue_ci"] == pytest.approx(1.0, abs=1e-9)


def test_evalue_rd_nonnull_true():
    out = sp.evalue_rd(200, 150, 100, 250, true=0.1)
    assert out["evalue_estimate"] > 1.0
    assert out["true"] == 0.1


@pytest.mark.parametrize(
    "rr,word",
    [
        (2.5, "very robust"),  # E ~ 4.4
        (1.6, "moderately robust"),  # E ~ 2.6
        (1.3, "somewhat robust"),  # E ~ 1.7
        (1.05, "potentially sensitive"),  # E ~ 1.28
    ],
)
def test_interpretation_strength_levels(rr, word):
    out = sp.evalue(estimate=rr, measure="RR")
    assert word in out["interpretation"]


def test_evalue_from_result_with_se_path():
    class _R:
        estimate = 0.3
        se = 0.1
        ci = None
        alpha = 0.05

    out = sp.evalue_from_result(_R(), measure="SMD")
    assert out["evalue_estimate"] == pytest.approx(1.956110, rel=RTOL)


def test_threshold_none_and_nan():
    assert _threshold(None) is None
    assert _threshold(float("nan")) is None


def test_md_point_only_no_ci():
    out = sp.evalue(estimate=0.3, measure="SMD")
    assert out["evalue_estimate"] == pytest.approx(1.956110, rel=RTOL)
    assert out["evalue_ci"] is None


def test_rd_empty_exposure_group_rejected():
    with pytest.raises(ValueError):
        sp.evalue_rd(0, 0, 100, 250)


def test_rd_true_above_observed_rejected():
    with pytest.raises(ValueError):
        sp.evalue_rd(200, 150, 100, 250, true=0.5)
