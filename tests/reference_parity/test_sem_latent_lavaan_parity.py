"""Latent variables, mean structures and growth curves against lavaan.

``sp.path_analysis`` with ``=~`` lines, ``meanstructure=``, ``std_lv=`` and
``growth=`` against ``lavaan::sem`` / ``growth`` (0.6-21) on the same CSV
bytes: nine models, each with normal-theory and Satorra-Bentler standard
errors. Regenerate the reference with
``Rscript tests/reference_parity/_fixtures/_generate_sem_latent_lavaan.R``.

Tolerance. lavaan stops its optimiser (``nlminb``) at a relative change of
about 1e-6 in the estimates; the engine here iterates Fisher scoring until
the gradient is below 1e-9. The comparison is therefore bounded by lavaan's
own stopping rule, not by a difference of method, and 5e-6 (relative, with a
floor of one) covers every number in every case. The growth and the
observed-variable cases, which lavaan solves in a few exact steps, agree to
1e-8.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "sem_latent_lavaan_R.json").read_text(encoding="utf-8"))
TOL = 5e-6

OPTIONS = {
    "cfa_stdlv": {"std_lv": True},
    "cfa_means": {"meanstructure": True},
    "growth": {"growth": True},
    # lavaan::sem lets terminal outcomes covary; here that is on request
    "two_outcomes": {"auto_cov_y": True},
    "latent_and_outcome": {"auto_cov_y": True},
}
FIT_NAMES = {
    "chisq": "chisq", "df": "df", "pvalue": "pvalue",
    "baseline.chisq": "baseline_chisq", "baseline.df": "baseline_df",
    "cfi": "cfi", "tli": "tli", "rmsea": "rmsea",
    "rmsea.ci.lower": "rmsea_ci_lower", "rmsea.ci.upper": "rmsea_ci_upper",
    "srmr": "srmr", "logl": "logl", "aic": "aic", "bic": "bic", "npar": "npar",
}  # fmt: skip
CASES = sorted(REF["cases"])


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "sem_latent.csv")


def _close(ours, theirs):
    return abs(ours - theirs) <= TOL * max(1.0, abs(theirs))


def _row(table, ref):
    rhs = ref["rhs"].replace(" ", "") if ref["op"] == ":=" else ref["rhs"]
    for key in ((ref["lhs"], ref["op"], rhs), (rhs, ref["op"], ref["lhs"])):
        if key in table.index:
            return table.loc[key]
    raise AssertionError(f"no row for {ref['lhs']} {ref['op']} {ref['rhs']}")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("which", ["ml", "mlm"])
def test_parameter_table_matches_lavaan(data, case, which):
    ref = REF["cases"][case]
    fit = sp.path_analysis(
        ref["model"],
        data,
        se="standard" if which == "ml" else "robust",
        **OPTIONS.get(case, {}),
    )
    table = fit.params.set_index(["lhs", "op", "rhs"])
    assert len(table) == len(ref[which]), "same parameters, no more and no fewer"
    for r in ref[which]:
        row = _row(table, r)
        assert _close(row["est"], r["est"]), (case, r, row["est"])
        assert _close(row["se"], r["se"]), (case, r, row["se"])
        if r.get("std.all") is not None:
            assert _close(row["std_all"], r["std.all"]), (case, r)


@pytest.mark.parametrize("case", CASES)
def test_fit_measures_match_lavaan(data, case):
    ref = REF["cases"][case]
    fit = sp.path_analysis(ref["model"], data, **OPTIONS.get(case, {}))
    for theirs, ours in FIT_NAMES.items():
        value = ref["ml_fit"][theirs]
        if value is None:  # the p-value of a saturated model
            continue
        assert _close(fit.fit[ours], value), (case, theirs, fit.fit[ours], value)


@pytest.mark.parametrize("case", CASES)
def test_scaled_test_matches_lavaan(data, case):
    ref = REF["cases"][case]
    if ref["mlm_fit"]["chisq.scaling.factor"] is None:
        pytest.skip("saturated model: nothing to scale")
    fit = sp.path_analysis(ref["model"], data, se="robust", **OPTIONS.get(case, {}))
    assert _close(fit.fit["scaling_factor"], ref["mlm_fit"]["chisq.scaling.factor"])
    assert _close(fit.fit["chisq_scaled"], ref["mlm_fit"]["chisq.scaled"])
    assert _close(fit.fit["pvalue_scaled"], ref["mlm_fit"]["pvalue.scaled"])


@pytest.mark.parametrize("case", ["cfa", "cfa_means", "growth"])
def test_factor_scores_match_lavpredict(data, case):
    ref = REF["cases"][case]
    fit = sp.path_analysis(ref["model"], data, **OPTIONS.get(case, {}))
    for name, values in ref["factor_scores"].items():
        np.testing.assert_allclose(
            fit.factor_scores[name].to_numpy()[: len(values)], values, atol=TOL
        )


def test_choice_of_scale_does_not_change_the_fit(data):
    """Marker loading or unit variance: the same model, the same test."""
    model = REF["cases"]["cfa"]["model"]
    marker = sp.path_analysis(model, data)
    unit = sp.path_analysis(model, data, std_lv=True)
    assert unit.fit["chisq"] == pytest.approx(marker.fit["chisq"], abs=1e-8)
    np.testing.assert_allclose(unit.implied_cov, marker.implied_cov, atol=1e-8)
    # the standardised loadings do not depend on the scale either
    a = marker.params.query("op == '=~'")["std_all"].to_numpy()
    b = unit.params.query("op == '=~'")["std_all"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-8)
    # and the latent correlation is the unit-variance covariance
    corr = marker.latent_cov.loc["f1", "f2"] / np.sqrt(
        marker.latent_cov.loc["f1", "f1"] * marker.latent_cov.loc["f2", "f2"]
    )
    assert unit.latent_cov.loc["f1", "f2"] == pytest.approx(corr, abs=1e-8)


def test_a_mean_structure_alone_changes_nothing_else(data):
    model = REF["cases"]["cfa"]["model"]
    plain = sp.path_analysis(model, data)
    means = sp.path_analysis(model, data, meanstructure=True)
    keep = plain.params["op"] != "~1"
    np.testing.assert_allclose(
        means.params.loc[means.params["op"] != "~1", "est"],
        plain.params.loc[keep, "est"],
        atol=1e-9,
    )
    cols = list(means.implied_mean.index)
    np.testing.assert_allclose(means.implied_mean, data[cols].mean(), atol=1e-9)


def test_terminal_outcomes_are_uncorrelated_unless_asked(data):
    model = REF["cases"]["two_outcomes"]["model"]
    as_written = sp.path_analysis(model, data)
    assert as_written.fit["df"] == 1
    assert not (
        (as_written.params["lhs"] == "a1") & (as_written.params["rhs"] == "b1")
    ).any()
    spelled = sp.path_analysis(model + "\na1 ~~ b1", data)
    asked = sp.path_analysis(model, data, auto_cov_y=True)
    assert spelled.fit["df"] == asked.fit["df"] == 0
    np.testing.assert_allclose(spelled.implied_cov, asked.implied_cov, atol=1e-10)


def test_latent_slope_is_not_attenuated():
    """Known truth: a slope of 0.8 on a predictor measured with error."""
    rng = np.random.default_rng(7)
    n = 20000
    skill = rng.normal(size=n)
    df = pd.DataFrame({f"t{j}": skill + rng.normal(size=n) for j in (1, 2, 3)})
    df["y"] = 0.8 * skill + rng.normal(size=n)
    fit = sp.path_analysis("skill =~ t1 + t2 + t3\ny ~ b*skill", df)
    b = fit.effect("b")
    assert abs(b["est"] - 0.8) < 3 * b["se"]
    naive = sp.path_analysis("y ~ t1", df).params.iloc[0]["est"]
    assert naive == pytest.approx(0.4, abs=0.03)  # reliability one half
    assert fit.r2[["t1", "t2", "t3"]].to_numpy() == pytest.approx(0.5, abs=0.03)


def test_what_is_not_identified_is_refused(data):
    with pytest.raises((DataInsufficient, MethodIncompatibility)):
        # one indicator, nothing else: variance and error cannot be separated
        sp.path_analysis("f =~ a1\ny ~ f", data)
    with pytest.raises((DataInsufficient, MethodIncompatibility)):
        # no scale for the latent variable
        sp.path_analysis("f1 =~ NA*a1 + a2 + a3", data)
    with pytest.raises(MethodIncompatibility, match="name of a column"):
        sp.path_analysis("x =~ a1 + a2 + a3", data)
    with pytest.raises(MethodIncompatibility, match="exogenous"):
        sp.path_analysis("y ~ x\nx ~ 1", data)
    with pytest.raises(MethodIncompatibility, match="modifier"):
        sp.path_analysis("f1 =~ a1 + start(0.5)*a2 + a3", data)


def test_heywood_case_warns():
    """Two weak indicators and a strong one: a negative error variance."""
    rng = np.random.default_rng(3)
    n = 60
    f = rng.normal(size=n)
    df = pd.DataFrame(
        {"a": f + 0.05 * rng.normal(size=n), "b": 0.2 * f + rng.normal(size=n),
         "c": 0.2 * f + rng.normal(size=n)}
    )  # fmt: skip
    found = False
    for seed in range(40):
        sample = df.sample(n, replace=True, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = sp.path_analysis("f =~ b + a + c", sample)
            except DataInsufficient:
                continue
        variances = fit.params.query("op == '~~' and lhs == rhs")["est"]
        if (variances < 0).any():
            assert any("Heywood" in str(w.message) for w in caught)
            found = True
            break
    assert found, "no bootstrap sample produced a negative variance"
