"""``sp.didregress`` against Stata 18 ``didregress`` and ``xtdidregress``.

Data: ``sp.datasets.mpdta()`` with ``d = 1`` from a county's first treated
year on and a deterministic covariate ``x = sin(0.37 county + 1.3 year)``,
written to a .dta. The two-group sample keeps the never-treated counties
and the 2006 cohort (250 counties, 2003-2007)::

    didregress (lemp [x]) (d), group(countyreal) time(year)
    xtset countyreal year
    xtdidregress (lemp [x]) (d), group(countyreal) time(year)
    estat ptrends
    estat granger

Evidence tier. The ATET, its standard error and the Granger test agree to
1e-11. The linear-trend test agrees with Stata's ``estat ptrends`` to 3e-6
only, and that is Stata's side: the same model fitted in Stata by ``areg``
(or ``xtreg, fe``) with time centred at the last pre-treatment year gives
the value here to 1e-12, with the trend on raw years it differs from both
by 1e-6. Raw years near 2005 make the trend nearly collinear with the
treatment dummy. The centred values are the references below; the
``estat ptrends`` values are kept with their looser tolerance.
"""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

RTOL = 1e-10


@pytest.fixture(scope="module")
def staggered():
    df = sp.datasets.mpdta().copy()
    df["d"] = ((df.first_treat > 0) & (df.year >= df.first_treat)).astype(int)
    df["x"] = np.sin(df.countyreal * 0.37 + df.year * 1.3)
    return df


@pytest.fixture(scope="module")
def two_group(staggered):
    return staggered[staggered.first_treat.isin([0, 2006])].copy()


def _fit(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.didregress(df, "lemp", "d", group="countyreal", time="year", **kwargs)


# label: (kwargs, ATET, se, ptrends F [centred areg / xtreg], estat ptrends F,
#         granger F, granger p)
TWO_GROUP = {
    "didregress": (
        {},
        -0.030029608985,
        0.010253787165,
        1.610043309851,
        1.610038825225,
        0.861879525827,
        0.42362360391240,
    ),
    "didregress_x": (
        {"covariates": ["x"]},
        -0.030021113388,
        0.010249206006,
        1.612052942013,
        1.612049827927,
        0.862639705549,
        0.42330390949980,
    ),
    "xtdidregress": (
        {"id": "countyreal"},
        -0.030029608985,
        0.009170344455,
        2.013770182110,
        2.013764574092,
        1.078000373693,
        0.34185789750183,
    ),
}


@pytest.mark.parametrize("label", sorted(TWO_GROUP))
def test_atet_and_tests_match_stata(two_group, label):
    kwargs, atet, se, ptr, ptr_estat, gra, gra_p = TWO_GROUP[label]
    res = _fit(two_group, **kwargs)
    assert res.estimate == pytest.approx(atet, rel=RTOL)
    assert res.se == pytest.approx(se, rel=RTOL)
    assert res.model_info["df"] == 249.0
    ptrends = sp.estat(res, "ptrends", print_results=False)
    assert ptrends["statistic"] == pytest.approx(ptr, rel=RTOL)
    # Stata's estat ptrends carries ~3e-6 of conditioning noise (docstring).
    assert ptrends["statistic"] == pytest.approx(ptr_estat, rel=1e-5)
    assert ptrends["statistic_label"] == "F(1, 249)"
    granger = sp.estat(res, "granger", print_results=False)
    assert granger["statistic"] == pytest.approx(gra, rel=RTOL)
    assert granger["pvalue"] == pytest.approx(gra_p, rel=1e-9)
    assert granger["statistic_label"] == "F(2, 249)"


def test_xtdidregress_with_a_covariate(two_group):
    res = _fit(two_group, id="countyreal", covariates=["x"])
    assert res.estimate == pytest.approx(-0.030021113388, rel=RTOL)
    assert res.se == pytest.approx(0.009165324594, rel=RTOL)
    granger = sp.estat(res, "granger", print_results=False)
    assert granger["statistic"] == pytest.approx(1.079169228414, rel=RTOL)
    assert granger["pvalue"] == pytest.approx(0.34146197670912, rel=1e-9)


def test_staggered_adoption_matches_stata(staggered):
    did = _fit(staggered, covariates=["x"])
    assert did.estimate == pytest.approx(-0.037506713802, rel=RTOL)
    assert did.se == pytest.approx(0.006468680499, rel=RTOL)
    assert did.model_info["df"] == 499.0
    # 2 * ttail(499, |t|)
    assert did.pvalue == pytest.approx(1.190300e-08, rel=1e-6)
    xt = _fit(staggered, covariates=["x"], id="countyreal")
    assert xt.estimate == pytest.approx(did.estimate, rel=1e-12)
    assert xt.se == pytest.approx(0.005785183498, rel=RTOL)


def test_tests_are_refused_with_several_treatment_dates(staggered):
    # Stata: "treatment assignment times vary; not allowed with estat ptrends"
    res = _fit(staggered)
    assert res.model_info["treatment_times"] == [2004, 2006, 2007]
    for which in ("ptrends", "granger"):
        with pytest.raises(MethodIncompatibility, match="assignment times vary"):
            sp.estat(res, which, print_results=False)


def test_estat_refuses_other_results(two_group):
    fit = sp.regress("lemp ~ d", two_group)
    with pytest.raises(MethodIncompatibility, match="sp.didregress"):
        sp.estat(fit, "ptrends", print_results=False)


def test_equals_the_dummy_variable_regression(two_group):
    res = _fit(two_group, covariates=["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sp.regress(
            "lemp ~ d + x + C(countyreal) + C(year)", two_group, cluster="countyreal"
        )
    assert res.estimate == pytest.approx(ols.params["d"], rel=1e-10)
    assert res.se == pytest.approx(ols.std_errors["d"], rel=1e-9)
    table = res.detail.set_index("term")
    assert table.loc["x", "estimate"] == pytest.approx(ols.params["x"], rel=1e-9)
    assert table.loc["x", "se"] == pytest.approx(ols.std_errors["x"], rel=1e-9)


def test_unbalanced_panel(two_group):
    holes = two_group.drop(two_group.index[::7])
    res = _fit(holes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sp.regress(
            "lemp ~ d + C(countyreal) + C(year)", holes, cluster="countyreal"
        )
    assert res.estimate == pytest.approx(ols.params["d"], rel=1e-9)
    assert res.se == pytest.approx(ols.std_errors["d"], rel=1e-8)


def test_bad_input_raises(two_group):
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.didregress(two_group, "lemp", "year", group="countyreal", time="year")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.didregress(two_group, "lemp", "nope", group="countyreal", time="year")
    never = two_group.assign(d=0)
    with pytest.raises(DataInsufficient, match="does not vary"):
        sp.didregress(never, "lemp", "d", group="countyreal", time="year")
    # treatment that varies inside a group-period cell is not assigned by group
    coarse = two_group.assign(state=two_group.countyreal // 1000)
    with pytest.raises(MethodIncompatibility, match="varies within"):
        sp.didregress(coarse, "lemp", "d", group="state", time="year")
    doubled = two_group.loc[two_group.index.repeat(2)]
    with pytest.raises(MethodIncompatibility, match="one row per unit"):
        sp.didregress(
            doubled, "lemp", "d", group="countyreal", time="year", id="countyreal"
        )
    with pytest.raises(DataInsufficient, match="collinear"):
        sp.didregress(
            two_group.assign(z=two_group.year * 2.0),
            "lemp",
            "d",
            group="countyreal",
            time="year",
            covariates=["z"],
        )


def test_one_pre_period_has_no_trend_test(two_group):
    late = two_group[two_group.year >= 2005]
    res = _fit(late)
    assert res.model_info["ptrends"] == {
        "unavailable": "fewer than two pre-treatment periods"
    }


def test_stata_lines_run(two_group):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata(
            "didregress (lemp x) (d), group(countyreal) time(year)", data=two_group
        )
        out = sp.stata(
            """
            xtset countyreal year
            xtdidregress (lemp) (d), group(countyreal) time(year)
            estat granger
            """,
            data=two_group,
        )
    assert res.se == pytest.approx(0.010249206006, rel=RTOL)
    assert out["statistic"] == pytest.approx(1.078000373693, rel=RTOL)
    refused = sp.from_stata(
        "didregress (lemp) (d), group(countyreal) time(year) wildbootstrap(rseed(1))"
    )
    assert refused["untranslated_options"] == ["wildbootstrap"]
