"""``unified_sensitivity`` must answer about the right coefficient, on the
right scale.

Two defects lived here, and the second hid behind the first.

1. ``_extract_estimate`` took ``params.iloc[0]`` and ``std_errors.iloc[0]``.
   For any formula regression that is the **intercept**, so
   ``sp.unified_sensitivity(ols_fit)`` silently answered "how sensitive is
   the intercept to unmeasured confounding?" while the caller read it as a
   statement about their treatment effect.

2. The E-value is defined on the risk-ratio scale, but a raw regression
   coefficient was passed through as though it already were one. A $1,548
   treatment effect became "RR = 1548" and an E-value of 3096 — arithmetic
   without meaning. Standardising first (VanderWeele & Ding 2017,
   ``RR ~ exp(0.91 * d)``) gives 1.71, a number one can actually argue
   about.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.robustness.unified_sensitivity import _extract_estimate

COVARIATES = [
    "age",
    "educ",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


@pytest.fixture(scope="module")
def lalonde():
    return sp.datasets.nsw_lalonde(simulated=False)


@pytest.fixture(scope="module")
def ols_fit(lalonde):
    formula = "re78 ~ treat + " + " + ".join(COVARIATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.regress(formula, lalonde, robust="HC1")


def _expected_evalue(fit, term, lalonde):
    """E-value recomputed here from first principles, not from the code
    under test: standardise by the outcome SD, map to a risk ratio with
    RR = exp(0.91 * d), then E = RR + sqrt(RR * (RR - 1))."""
    d = float(fit.params[term]) / float(lalonde["re78"].std(ddof=1))
    rr = math.exp(0.91 * d)
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def _dash(fit, lalonde, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.unified_sensitivity(
            fit, data=lalonde, y="re78", controls=COVARIATES, **kwargs
        )


# --------------------------------------------------------------------- #
# Which coefficient
# --------------------------------------------------------------------- #


def test_extractor_never_returns_the_intercept(ols_fit):
    estimate, se, _ = _extract_estimate(ols_fit, term="treat")
    assert estimate == pytest.approx(float(ols_fit.params["treat"]))
    assert estimate != pytest.approx(float(ols_fit.params["Intercept"]))
    assert se == pytest.approx(float(ols_fit.std_errors["treat"]))


def test_estimate_se_and_ci_all_describe_the_same_term(ols_fit):
    """Mixing terms across the triple is what made the CI check fire."""
    estimate, _, ci = _extract_estimate(ols_fit, term="treat")
    expected = ols_fit.conf_int().loc["treat"]
    assert ci[0] == pytest.approx(float(expected.iloc[0]))
    assert ci[1] == pytest.approx(float(expected.iloc[1]))
    assert ci[0] <= estimate <= ci[1]


def test_ambiguous_result_raises_instead_of_guessing(ols_fit):
    with pytest.raises(MethodIncompatibility) as excinfo:
        sp.unified_sensitivity(ols_fit)
    assert "which coefficient" in str(excinfo.value)


def test_unknown_term_lists_the_available_ones(ols_fit):
    with pytest.raises(MethodIncompatibility) as excinfo:
        sp.unified_sensitivity(ols_fit, term="not_a_column")
    assert "not a coefficient" in str(excinfo.value)


def test_single_coefficient_result_needs_no_term():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"d": rng.normal(size=400)})
    df["y"] = 2.0 * df["d"] + rng.normal(size=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.regress("y ~ d", df, robust="HC1")
    estimate, _, _ = _extract_estimate(fit)
    assert estimate == pytest.approx(float(fit.params["d"]))


def test_treat_kwarg_doubles_as_the_term(ols_fit, lalonde):
    """`treat=` already names the treatment; don't demand it twice."""
    dash = _dash(ols_fit, lalonde, treat="treat")
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "treat", lalonde), rel=1e-9
    )


def test_explicit_term_wins_over_treat(ols_fit, lalonde):
    dash = _dash(ols_fit, lalonde, term="educ", treat="treat")
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "educ", lalonde), rel=1e-9
    )


def test_result_sensitivity_method_no_longer_views_the_intercept(ols_fit, lalonde):
    """EconometricResults.sensitivity() built a fake scalar view of iloc[0]."""
    with pytest.raises(MethodIncompatibility):
        ols_fit.sensitivity()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = ols_fit.sensitivity(term="treat", data=lalonde, y="re78")
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "treat", lalonde), rel=1e-9
    )


# --------------------------------------------------------------------- #
# Which scale
# --------------------------------------------------------------------- #


def test_raw_coefficient_is_standardised_not_read_as_a_risk_ratio(ols_fit, lalonde):
    """The headline of the second defect: 3096 -> 1.71."""
    dash = _dash(ols_fit, lalonde, term="treat")
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "treat", lalonde), rel=1e-9
    )
    # The old behaviour read $1,548 as RR=1548; nothing near that survives.
    assert dash.e_value_point < 10.0


def test_regression_result_supplies_its_own_scale(ols_fit, lalonde):
    """A fitted regression already knows its outcome SD.

    EconometricResults keeps the outcome vector it was fitted on, so the
    E-value is available without the caller re-passing the frame. This is
    what lets the MCP `sensitivity` tool — which only receives a cached
    result handle — return a real number instead of nothing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(ols_fit, term="treat")
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "treat", lalonde), rel=1e-9
    )


def test_missing_scale_skips_the_evalue_with_a_reason():
    """A result carrying neither data nor an outcome vector has no scale.

    No outcome SD means no defensible E-value — say so, don't invent one.
    """
    bare = SimpleNamespace(estimate=0.35, se=0.10, ci=(0.15, 0.55))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(bare)
    assert math.isnan(dash.e_value_point)
    notes = " ".join(str(n) for n in (dash.notes or []))
    assert "E-value skipped" in notes
    assert "outcome SD" in notes


def test_outcome_sd_can_be_supplied_directly(ols_fit, lalonde):
    sd = float(lalonde["re78"].std(ddof=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(ols_fit, term="treat", outcome_sd=sd)
    assert dash.e_value_point == pytest.approx(
        _expected_evalue(ols_fit, "treat", lalonde), rel=1e-9
    )


def test_explicit_measure_rr_is_honoured(ols_fit):
    """If the caller says it really is a ratio, believe them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(ols_fit, term="treat", measure="RR")
    rr = float(ols_fit.params["treat"])
    assert dash.e_value_point == pytest.approx(
        rr + math.sqrt(rr * (rr - 1.0)), rel=1e-9
    )


def test_scalar_result_needs_its_scale_declared():
    """A bare estimate/se/ci carries no scale, so neither guess is safe."""
    result = SimpleNamespace(estimate=0.35, se=0.10, ci=(0.15, 0.55))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(sp.unified_sensitivity(result).e_value_point)
        assert sp.unified_sensitivity(result, outcome_sd=2.0).e_value_point >= 1.0
        ratio = SimpleNamespace(estimate=1.35, se=0.10, ci=(1.15, 1.55))
        assert sp.unified_sensitivity(ratio, measure="RR").e_value_point >= 1.0


# --------------------------------------------------------------------- #
# sp.sensitivity_dashboard: an empty dashboard must not read as a pass
# --------------------------------------------------------------------- #


def test_empty_dashboard_warns_instead_of_grading_silently(ols_fit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dash = sp.sensitivity_dashboard(ols_fit, verbose=False)
    assert not dash.dimensions
    assert dash.overall_stability == "?"
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime, "an empty dashboard must say so"
    assert "no dimensions" in str(runtime[0].message)


def test_dashboard_with_data_runs_dimensions_and_stays_quiet(ols_fit, lalonde):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dash = sp.sensitivity_dashboard(ols_fit, data=lalonde, verbose=False)
    assert dash.dimensions
    assert dash.overall_stability in set("ABCDF")
    empties = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "no dimensions" in str(w.message)
    ]
    assert not empties


def test_dashboard_baseline_is_the_treatment_not_the_intercept(ols_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.sensitivity_dashboard(ols_fit, verbose=False)
    assert float(dash.baseline["estimate"]) == pytest.approx(
        float(ols_fit.params["treat"])
    )


# --------------------------------------------------------------------- #
# Oster: one specification must not yield two different delta*
# --------------------------------------------------------------------- #


def test_oster_delta_matches_sp_oster_delta_on_the_same_spec(ols_fit, lalonde):
    """A report must not disagree with itself.

    unified_sensitivity consumed `r2_treated`/`r2_controlled` as the
    short/long regression R^2, but those names read like sensemakr's
    partial R^2. Feeding sensemakr-style values gave delta* = -12.765
    while sp.oster_delta reported -2.339 for the same specification, in
    the same pipeline. The R^2 are now derived from the data instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = sp.oster_delta(
            lalonde,
            y="re78",
            x_base=["treat"],
            x_controls=COVARIATES,
            r_max=0,
            n_boot=0,
        )
        dash = _dash(ols_fit, lalonde, treat="treat")
    assert dash.oster is not None, "Oster should be derived from the data"
    assert dash.oster["delta"] == pytest.approx(
        float(reference.model_info["delta_star"]), rel=1e-9
    )


def test_legacy_r2_aliases_still_work_but_warn(ols_fit, lalonde):
    sd = float(lalonde["re78"].std(ddof=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.unified_sensitivity(
            ols_fit,
            term="treat",
            outcome_sd=sd,
            r2_treated=0.00152,
            r2_controlled=0.14776,
            beta_uncontrolled=-635.026212,
        )
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "the misleading aliases must announce themselves"
    assert "r2_short" in str(deprecations[0].message)


def test_oster_skipped_message_names_what_is_missing(ols_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(ols_fit, term="treat")
    assert dash.oster is None
    notes = " ".join(str(n) for n in (dash.notes or []))
    assert "Oster delta skipped" in notes
    assert "r2_short" in notes
