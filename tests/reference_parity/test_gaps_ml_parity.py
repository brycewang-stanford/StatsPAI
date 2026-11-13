"""Reference parity for the experiment / ML items of the 2026-11 gap list.

* ``sp.srm_test`` (new): Pearson chi-square goodness-of-fit test of arm
  counts against the designed allocation, against R ``chisq.test(x, p=)``.
  Frozen reference ``_fixtures/gaps_srm_R.json`` from
  ``_generate_gaps_srm_R.R`` (R 4.5.2). The statistic is a finite sum of
  ratios, so the tolerance is rounding only (1e-12 relative).
* ``sp.dml_sensitivity(rho=, level=)`` (new options) against ``doubleml``
  0.11.4 ``sensitivity_analysis(cf_y, cf_d, rho, level)``. Frozen reference
  ``_fixtures/gaps_dml_sensitivity_doubleml.json`` from
  ``_generate_gaps_dml_sensitivity_doubleml.py`` on the committed
  ``dml_benchmark_data.csv`` with deterministic learners and the folds of
  the ``fold`` column. Bounds: 1e-10. Standard errors of the bounds: 1e-10,
  crosswise (``doubleml`` reports the upper bound's standard error for the
  lower bound and vice versa; see
  ``tests/external_parity/test_dml_sensitivity_parity.py`` and the
  finite-difference check in ``test_dml_sensitivity_bound_scores.py``).
  Interval: ``doubleml``'s number is rebuilt from our bounds and the
  swapped standard errors at 1e-10, which exercises ``level``. Robustness
  value: ``doubleml`` stops a scalar root finder early (it is up to 1.3e-6
  away in absolute terms on these fits), so the reference is compared at
  5e-6 absolute and the defining identity (the bound at ``rv`` equals the
  null) is asserted at 1e-9.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
)

FIX = pathlib.Path(__file__).parent / "_fixtures"
X = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


# ---------------------------------------------------------------------------
# sp.srm_test against R chisq.test(x, p = p)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def srm_ref() -> dict:
    ref = json.loads((FIX / "gaps_srm_R.json").read_text(encoding="utf-8"))
    ref.pop("_meta")
    return ref


SRM_CASES = [
    "two_equal",
    "two_balanced",
    "two_unequal",
    "three_equal",
    "three_ratio",
    "four_design",
    "empty_arm",
    "five_small",
]


@pytest.mark.parametrize("case", SRM_CASES)
def test_srm_test_matches_chisq_test(srm_ref, case):
    want = srm_ref[case]
    arms = [f"arm{i}" for i in range(len(want["x"]))]
    counts = dict(zip(arms, want["x"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        got = sp.srm_test(counts, expected=dict(zip(arms, want["p"])))
    assert got.df == want["df"]
    assert got.statistic == pytest.approx(want["statistic"], rel=1e-12)
    assert got.p_value == pytest.approx(want["p_value"], rel=1e-10)
    np.testing.assert_allclose(got.table["expected"], want["expected"], rtol=1e-12)
    np.testing.assert_allclose(
        got.table["residual"], want["residuals"], rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        got.table["stdres"], want["stdres"], rtol=1e-12, atol=1e-13
    )


@pytest.mark.parametrize("case", ["two_unequal", "three_ratio", "four_design"])
def test_srm_test_row_level_data_equals_counts(srm_ref, case):
    """Rows, a share sequence in sorted arm order and unnormalised ratios."""
    want = srm_ref[case]
    x = [int(v) for v in want["x"]]
    arm = np.repeat(np.arange(len(x)), x)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"arm": rng.permutation(arm)})
    got = sp.srm_test(df, "arm", expected=[100 * p for p in want["p"]])
    assert got.statistic == pytest.approx(want["statistic"], rel=1e-12)
    assert got.p_value == pytest.approx(want["p_value"], rel=1e-10)
    assert got.n == sum(x)
    assert list(got.table["observed"]) == x


def test_srm_test_default_is_equal_shares_and_two_arms_is_a_z_test(srm_ref):
    want = srm_ref["two_equal"]
    got = sp.srm_test({"c": 5000, "t": 5300})
    assert got.statistic == pytest.approx(want["statistic"], rel=1e-12)
    # With two arms the statistic is the square of the binomial z statistic.
    n = 10300
    z = (5300 - n / 2) / np.sqrt(n * 0.25)
    assert got.statistic == pytest.approx(z**2, rel=1e-12)
    assert got.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    # p = 0.0031 is above the default alarm level and below 0.01
    assert got.mismatch is False
    assert sp.srm_test({"c": 5000, "t": 5300}, alpha=0.01).mismatch is True


def test_srm_test_missing_arms_are_left_out_and_counted():
    df = pd.DataFrame({"arm": ["a"] * 40 + ["b"] * 60 + [None] * 7})
    got = sp.srm_test(df, "arm")
    assert got.n == 100 and got.n_missing == 7
    assert got.statistic == pytest.approx(4.0, rel=1e-12)  # (10^2 + 10^2) / 50


def test_srm_test_small_expected_count_warns(srm_ref):
    with pytest.warns(AssumptionWarning, match="below 5"):
        got = sp.srm_test({"a": 3, "b": 4})
    assert got.diagnostics["small_expected"] == ["a", "b"]


def test_srm_test_refuses_impossible_input():
    with pytest.raises(DataInsufficient, match="at least two"):
        sp.srm_test({"a": 10})
    with pytest.raises(DataInsufficient, match="no units"):
        sp.srm_test({"a": 0, "b": 0})
    with pytest.raises(DataInsufficient, match="whole numbers"):
        sp.srm_test({"a": 10.5, "b": 9})
    with pytest.raises(DataInsufficient, match="whole numbers"):
        sp.srm_test({"a": -1, "b": 9})
    with pytest.raises(MethodIncompatibility, match="expected share of zero"):
        sp.srm_test({"a": 10, "b": 9}, expected={"a": 1.0, "b": 0.0})
    with pytest.raises(MethodIncompatibility, match="2 arms"):
        sp.srm_test({"a": 10, "b": 9}, expected=[0.5, 0.25, 0.25])
    with pytest.raises(MethodIncompatibility, match="no entry"):
        sp.srm_test({"a": 10, "b": 9}, expected={"a": 0.5, "c": 0.5})
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.srm_test({"a": 10, "b": 9}, expected=[-0.5, 1.5])
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.srm_test({"a": 10, "b": 9}, alpha=0.0)
    with pytest.raises(ColumnNotFound):
        sp.srm_test(pd.DataFrame({"x": [0, 1]}), "arm")
    with pytest.raises(MethodIncompatibility, match="treat="):
        sp.srm_test(pd.DataFrame({"x": [0, 1]}))


# ---------------------------------------------------------------------------
# sp.dml_sensitivity(rho=, level=) against doubleml
# ---------------------------------------------------------------------------


def _logit() -> LogisticRegression:
    return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=200)


@pytest.fixture(scope="module")
def dml_data() -> pd.DataFrame:
    return pd.read_csv(FIX / "dml_benchmark_data.csv")


@pytest.fixture(scope="module")
def dml_ref() -> dict:
    return json.loads(
        (FIX / "gaps_dml_sensitivity_doubleml.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def dml_fits(dml_data) -> dict:
    fits = {}
    for model in ("plr", "irm"):
        ml_m = LinearRegression() if model == "plr" else _logit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[model] = sp.dml(
                dml_data,
                y="re78",
                treat="treat",
                covariates=X,
                model=model,
                fold_indices="fold",
                ml_g=LinearRegression(),
                ml_m=ml_m,
            )
    return fits


@pytest.mark.parametrize("model", ["plr", "irm"])
@pytest.mark.parametrize("i", range(5))
def test_dml_sensitivity_rho_level_match_doubleml(dml_fits, dml_ref, model, i):
    fit = dml_fits[model]
    want = dml_ref[model]["scenarios"][i]
    assert fit.estimate == pytest.approx(dml_ref[model]["theta"], rel=1e-10)
    sens = sp.dml_sensitivity(
        fit,
        cf_y=want["cf_y"],
        cf_d=want["cf_d"],
        rho=want["rho"],
        level=want["level"],
    )
    assert sens.rho == want["rho"]
    assert sens.alpha == pytest.approx(1 - want["level"], rel=1e-12)
    # the two bounds
    assert sens.adjusted_estimate_low == pytest.approx(want["theta_lower"], rel=1e-10)
    assert sens.adjusted_estimate_high == pytest.approx(want["theta_upper"], rel=1e-10)
    # their standard errors, crosswise (see the module docstring)
    assert sens.se_low == pytest.approx(want["se_upper"], rel=1e-10)
    assert sens.se_high == pytest.approx(want["se_lower"], rel=1e-10)
    # doubleml's interval is its bounds -/+ z(level) times its own labels
    z = stats.norm.ppf(want["level"])
    assert sens.adjusted_estimate_low - z * sens.se_high == pytest.approx(
        want["ci_lower"], rel=1e-10
    )
    assert sens.adjusted_estimate_high + z * sens.se_low == pytest.approx(
        want["ci_upper"], rel=1e-10
    )
    # ours uses each bound's own standard error at the same level
    assert sens.ci_low == pytest.approx(
        sens.adjusted_estimate_low - z * sens.se_low, rel=1e-12
    )
    assert sens.ci_high == pytest.approx(
        sens.adjusted_estimate_high + z * sens.se_high, rel=1e-12
    )
    # robustness value at this rho: doubleml's root finder stops early
    assert sens.rv_q == pytest.approx(want["rv"], abs=5e-6)
    # ... and ours satisfies the definition: at cf_y = cf_d = rv the lower
    # bound (both estimates are positive) is the null value 0
    at_rv = sp.dml_sensitivity(fit, cf_y=sens.rv_q, cf_d=sens.rv_q, rho=want["rho"])
    assert abs(at_rv.adjusted_estimate_low) <= 1e-9 * abs(fit.estimate)


@pytest.mark.parametrize("model", ["plr", "irm"])
def test_dml_sensitivity_defaults_are_the_worst_case_at_the_fit_level(dml_fits, model):
    """rho=1.0 and level=1-alpha reproduce the call without the options."""
    fit = dml_fits[model]
    base = sp.dml_sensitivity(fit, cf_y=0.03, cf_d=0.03)
    same = sp.dml_sensitivity(fit, cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95)
    for name in (
        "rv_q",
        "rv_qa",
        "bias_bound",
        "adjusted_estimate_low",
        "adjusted_estimate_high",
        "se_low",
        "se_high",
        "ci_low",
        "ci_high",
    ):
        assert getattr(base, name) == getattr(same, name), name
    assert base.rho == 1.0 and base.alpha == pytest.approx(0.05)
    # the bias bound scales with |rho|, whatever its sign
    half = sp.dml_sensitivity(fit, cf_y=0.03, cf_d=0.03, rho=-0.5)
    assert half.bias_bound == pytest.approx(0.5 * base.bias_bound, rel=1e-12)
    # a smaller |rho| needs a stronger confounder to reach the null
    assert half.rv_q > base.rv_q
    assert "Degree of adversity" in half.summary()
    assert "Degree of adversity" not in base.summary()


def test_dml_sensitivity_rho_zero_and_bad_options(dml_fits):
    fit = dml_fits["plr"]
    zero = sp.dml_sensitivity(fit, cf_y=0.3, cf_d=0.3, rho=0.0)
    assert zero.bias_bound == 0.0
    assert zero.adjusted_estimate_low == zero.adjusted_estimate_high == fit.estimate
    assert zero.rv_q == 1.0  # no strength moves the estimate
    with pytest.raises(MethodIncompatibility, match=r"rho must lie in \[-1, 1\]"):
        sp.dml_sensitivity(fit, rho=1.5)
    with pytest.raises(MethodIncompatibility, match="level must lie"):
        sp.dml_sensitivity(fit, level=95)
