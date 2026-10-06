"""``sp.path_analysis`` against ``lavaan::sem`` on the same data.

Four observed-variable models on one simulated data set (400 rows, heavy
tails and heteroskedasticity in two equations so that the robust results
differ from the normal-theory ones):

* ``mediation``: two parallel mediators, labelled paths, a residual
  covariance, a covariate and four defined effects;
* ``chain``: an over-identified chain the data reject;
* ``constrained``: an equality constraint, a fixed coefficient and a
  product term;
* ``saturated``: a just-identified system.

For each, lavaan's defaults (ML, expected information, ``fixed.x = TRUE``)
and ``estimator = "MLM"`` (Satorra-Bentler). Fixtures:
``_fixtures/path_analysis.csv`` and ``path_analysis_lavaan_R.json``, written by
``_generate_path_analysis_lavaan.R`` (lavaan 0.6-21, R 4.5.2).

Tolerances. Both sides maximise the same likelihood; lavaan stops at its
optimiser's tolerance, StatsPAI finishes with Newton steps. Where the
solution is closed-form the two agree to 1e-13; elsewhere to lavaan's
1e-5.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
FIT_KEYS = {
    "chisq": "chisq", "df": "df", "pvalue": "pvalue",
    "baseline.chisq": "baseline_chisq", "baseline.df": "baseline_df",
    "cfi": "cfi", "tli": "tli", "rmsea": "rmsea",
    "rmsea.ci.lower": "rmsea_ci_lower", "rmsea.ci.upper": "rmsea_ci_upper",
    "srmr": "srmr", "logl": "logl", "aic": "aic", "bic": "bic", "npar": "npar",
}  # fmt: skip


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "path_analysis_lavaan_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "path_analysis.csv")


def _num(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


CASES = ["mediation", "chain", "constrained", "saturated"]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("se, key", [("standard", "ml"), ("robust", "mlm")])
def test_parameter_table_equals_lavaan(ref, data, name, se, key):
    case = ref["cases"][name]
    fit = sp.path_analysis(case["model"], data, se=se)
    expected = pd.DataFrame(case[key])
    got = fit.params
    # same rows in the same order
    assert list(got["lhs"]) == list(expected["lhs"])
    assert list(got["op"]) == list(expected["op"])
    assert list(got["rhs"]) == list(expected["rhs"])
    np.testing.assert_allclose(got["est"], _num(expected["est"]), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(got["se"], _num(expected["se"]), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        got["std_all"], _num(expected["std.all"]), rtol=2e-5, atol=1e-6
    )
    free = _num(expected["se"]) > 0
    np.testing.assert_allclose(
        got["pvalue"][free], _num(expected["pvalue"])[free], rtol=1e-3, atol=1e-8
    )


@pytest.mark.parametrize("name", CASES)
def test_fit_measures_equal_lavaan(ref, data, name):
    case = ref["cases"][name]
    fit = sp.path_analysis(case["model"], data)
    for theirs, ours in FIT_KEYS.items():
        want = case["ml_fit"][theirs]
        if want is None:  # lavaan reports NA for the p-value at df = 0
            continue
        assert fit.fit[ours] == pytest.approx(want, rel=1e-6, abs=1e-6), theirs
    np.testing.assert_allclose(
        fit.implied_cov.loc[case["implied_names"], case["implied_names"]].to_numpy(),
        np.array(case["implied"]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("name", ["mediation", "chain", "constrained"])
def test_scaled_test_equals_lavaan_mlm(ref, data, name):
    case = ref["cases"][name]
    fit = sp.path_analysis(case["model"], data, se="robust")
    want = case["mlm_fit"]
    assert fit.fit["chisq_scaled"] == pytest.approx(want["chisq.scaled"], rel=1e-6)
    assert fit.fit["scaling_factor"] == pytest.approx(
        want["chisq.scaling.factor"], rel=1e-6
    )
    assert fit.fit["pvalue_scaled"] == pytest.approx(
        want["pvalue.scaled"], rel=1e-5, abs=1e-12
    )


def test_closed_form_cases_agree_to_rounding(ref, data):
    """Without a residual covariance or a constraint the solution is OLS
    equation by equation, and nothing depends on an optimiser."""
    for name in ("chain", "saturated"):
        case = ref["cases"][name]
        fit = sp.path_analysis(case["model"], data)
        expected = pd.DataFrame(case["ml"])
        np.testing.assert_allclose(fit.params["est"], _num(expected["est"]), rtol=1e-12)
        np.testing.assert_allclose(fit.params["se"], _num(expected["se"]), rtol=1e-12)


def test_recursive_model_is_equation_by_equation_ols(data):
    fit = sp.path_analysis("m1 ~ x + w\ny ~ x + w + m1", data)
    ols = sp.regress("y ~ x + w + m1", data=data)
    est = fit.params.set_index(["lhs", "rhs"])["est"]
    for v in ("x", "w", "m1"):
        assert est[("y", v)] == pytest.approx(ols.params[v], rel=1e-10)
    assert fit.fit["df"] == 0 and fit.fit["chisq"] == pytest.approx(0, abs=1e-8)
    assert fit.r2["y"] == pytest.approx(ols.diagnostics["R-squared"], rel=1e-8)


def test_product_term_spelling(data):
    a = sp.path_analysis("y ~ x + w + x:w", data)
    b = sp.path_analysis("y ~ x + w + xw", data)
    np.testing.assert_allclose(a.params["est"], b.params["est"], rtol=1e-9)
    assert list(a.params["rhs"][:3]) == ["x", "w", "x:w"]


def test_indirect_effect_recovers_the_truth_and_its_interval_covers():
    rng = np.random.default_rng(20261006)
    reps, covered, estimates = 200, 0, []
    for _ in range(reps):
        n = 300
        x = rng.standard_normal(n)
        m = 0.5 * x + rng.standard_normal(n)
        y = 0.4 * m + 0.2 * x + rng.standard_normal(n)
        fit = sp.path_analysis(
            "m ~ a*x\ny ~ b*m + c*x\nind := a*b", pd.DataFrame({"x": x, "m": m, "y": y})
        )
        row = fit.effect("ind")
        estimates.append(row["est"])
        covered += row["ci_lower"] <= 0.2 <= row["ci_upper"]
    assert np.mean(estimates) == pytest.approx(0.2, abs=0.01)
    # 200 draws of a 95% interval: above 90% with probability > 0.99
    assert covered / reps >= 0.90


def test_what_the_model_cannot_be(data):
    with pytest.raises(MethodIncompatibility, match="latent"):
        sp.path_analysis("f =~ m1 + m2\ny ~ f", data)
    with pytest.raises(MethodIncompatibility, match="intercepts"):
        sp.path_analysis("y ~ 1 + x", data)
    with pytest.raises(MethodIncompatibility, match="not a column"):
        sp.path_analysis("y ~ nope", data)
    with pytest.raises(MethodIncompatibility, match="not a label"):
        sp.path_analysis("y ~ b*x\neff := b*zz", data)
    with pytest.raises(MethodIncompatibility, match="appears twice"):
        sp.path_analysis("y ~ x + x", data)
    with pytest.raises(MethodIncompatibility, match="not identified"):
        # a feedback loop plus a residual covariance: more parameters than
        # moments
        sp.path_analysis("m1 ~ y + x\ny ~ m1 + x\nm1 ~~ y", data)
    with pytest.raises(MethodIncompatibility, match="se must be"):
        sp.path_analysis("y ~ x", data, se="bootstrap")
    with pytest.raises(DataInsufficient, match="linearly dependent"):
        sp.path_analysis("y ~ x + dup", data.assign(dup=2 * data["x"]))


def test_definitions_evaluate_arithmetic_only(data):
    with pytest.raises(MethodIncompatibility, match="cannot evaluate"):
        sp.path_analysis("y ~ b*x\neff := __import__('os').getcwd()", data)
    fit = sp.path_analysis("y ~ b*x\nsq := b^2 + 1", data)
    b = fit.effect("b")["est"]
    assert fit.effect("sq")["est"] == pytest.approx(b**2 + 1, rel=1e-12)


def test_missing_rows_are_dropped_and_counted(data):
    holes = data.copy()
    holes.loc[:9, "m1"] = np.nan
    fit = sp.path_analysis("m1 ~ x\ny ~ m1", holes)
    assert fit.n_obs == len(data) - 10 and fit.n_dropped == 10
    assert "dropped" in fit.summary()
