"""``unified_sensitivity`` must analyse the coefficient the user meant.

Until 1.21.0 ``_extract_estimate`` took ``params.iloc[0]`` and
``std_errors.iloc[0]``. For any formula regression that is the
**intercept**, so ``sp.unified_sensitivity(ols_fit)`` silently answered
"how sensitive is the intercept to unmeasured confounding?" — paired with
the intercept's standard error — while the caller read it as a statement
about their treatment effect.

The bug was only visible at all because the intercept's CI happened to
span zero, which sent the risk-ratio conversion down a branch that
produced an interval excluding its own point estimate and tripped an
assertion inside ``evalue``. On any design where the intercept CI stayed
positive it would have returned a confident, wrong number.
"""

from __future__ import annotations

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
def ols_fit():
    df = sp.datasets.nsw_lalonde(simulated=False)
    formula = "re78 ~ treat + " + " + ".join(COVARIATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.regress(formula, df, robust="HC1")


def test_extractor_never_returns_the_intercept(ols_fit):
    """The regression test for the actual defect."""
    estimate, se, ci = _extract_estimate(ols_fit, term="treat")
    assert estimate == pytest.approx(float(ols_fit.params["treat"]))
    assert estimate != pytest.approx(float(ols_fit.params["Intercept"]))
    assert se == pytest.approx(float(ols_fit.std_errors["treat"]))


def test_estimate_se_and_ci_all_describe_the_same_term(ols_fit):
    """Mixing terms across the triple is what made the CI check fire."""
    estimate, se, ci = _extract_estimate(ols_fit, term="treat")
    expected = ols_fit.conf_int().loc["treat"]
    assert ci[0] == pytest.approx(float(expected.iloc[0]))
    assert ci[1] == pytest.approx(float(expected.iloc[1]))
    assert ci[0] <= estimate <= ci[1]


def test_ambiguous_result_raises_instead_of_guessing(ols_fit):
    """Nine candidate coefficients is not a guess worth making."""
    with pytest.raises(MethodIncompatibility) as excinfo:
        sp.unified_sensitivity(ols_fit)
    message = str(excinfo.value)
    assert "which coefficient" in message
    assert "term=" in message or "term" in message


def test_unknown_term_lists_the_available_ones(ols_fit):
    with pytest.raises(MethodIncompatibility) as excinfo:
        sp.unified_sensitivity(ols_fit, term="not_a_column")
    assert "not a coefficient" in str(excinfo.value)


def test_dashboard_runs_clean_once_the_term_is_named(ols_fit):
    """The E-value degradation was a symptom; naming the term clears it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dash = sp.unified_sensitivity(
            ols_fit, term="treat", r2_treated=0.05, r2_controlled=0.10
        )
    # VanderWeele-Ding: E = RR + sqrt(RR * (RR - 1)) on the RR scale.
    rr = float(ols_fit.params["treat"])
    expected = rr + (rr * (rr - 1.0)) ** 0.5
    assert dash.e_value_point == pytest.approx(expected, rel=1e-9)
    failures = [
        note
        for note in (getattr(dash, "degradations", None) or [])
        if "E-value" in str(note)
    ]
    assert not failures, f"E-value still degraded: {failures}"


def test_single_coefficient_result_needs_no_term():
    """One non-intercept coefficient is unambiguous."""
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


def test_scalar_estimate_results_are_untouched():
    """CausalResult-shaped objects never went through the params path."""
    result = SimpleNamespace(estimate=0.35, se=0.10, ci=(0.15, 0.55))
    dash = sp.unified_sensitivity(result)
    assert dash.e_value_point >= 1.0


# --------------------------------------------------------------------- #
# sp.sensitivity_dashboard: an empty dashboard must not read as a pass
# --------------------------------------------------------------------- #


def test_empty_dashboard_warns_instead_of_grading_silently(ols_fit):
    """Most dimensions need the estimation data; without it none run.

    Returning ``overall_stability='?'`` and zero dimensions looked like a
    result, so a pipeline calling it without ``data=`` printed a
    robustness section that had tested nothing.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dash = sp.sensitivity_dashboard(ols_fit, verbose=False)
    assert not dash.dimensions
    assert dash.overall_stability == "?"
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime, "an empty dashboard must say so"
    assert "no dimensions" in str(runtime[0].message)


def test_dashboard_with_data_runs_dimensions_and_stays_quiet(ols_fit):
    df = sp.datasets.nsw_lalonde(simulated=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dash = sp.sensitivity_dashboard(ols_fit, data=df, verbose=False)
    assert dash.dimensions
    assert dash.overall_stability in set("ABCDF")
    empties = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "no dimensions" in str(w.message)
    ]
    assert not empties


def test_dashboard_baseline_is_the_treatment_not_the_intercept(ols_fit):
    """Sibling of the unified_sensitivity defect — verify it is absent here."""
    dash = sp.sensitivity_dashboard(ols_fit, verbose=False)
    assert float(dash.baseline["estimate"]) == pytest.approx(
        float(ols_fit.params["treat"])
    )
