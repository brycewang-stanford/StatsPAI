"""The four applications of de Chaisemartin and D'Haultfoeuille's
difference-in-differences textbook, against Stata 18.

The authors distribute the data and solution do-files of the applications as
the SSC package ``cc_xd_didtextbook`` (``ssc describe cc_xd_didtextbook``,
then ``net get``): Wolfers (2006) on unilateral divorce, Moser and Voena
(2012) on compulsory licensing, Pierce and Schott (2016) on trade
liberalisation, Gentzkow, Shapiro and Sinkinson (2011) on newspapers. The
do-files come without output, so the numbers below are what the commands
print when run in Stata 18 with the packages current on 2026-10-05
(``twowayfeweights``, ``did_multiplegt_dyn`` of 17 January 2026,
``did_multiplegt_old``, ``did_had``, ``did_imputation``,
``eventstudyinteract`` 0.1).

Neither the data nor the do-files are redistributed here. Point
``STATSPAI_DCDH_TEXTBOOK_DIR`` at the folder that holds ``Data sets`` to run
this; it is skipped otherwise.

    STATSPAI_DCDH_TEXTBOOK_DIR=/path/to/cc_xd_didtextbook \\
        pytest tests/external_parity/test_dcdh_did_textbook.py

Stata prints a fixed number of digits, so a number is compared to within
0.6 of a unit of the last digit printed (``_printed``). What the pass found is in
``docs/dev/2026-10-05-dcdh-did-textbook-review.md``; the same estimators are
pinned on a synthetic panel, to more digits and in every CI run, in
``tests/reference_parity/test_dcdh_textbook_stata_parity.py``.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_DCDH_TEXTBOOK_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "Data sets").is_dir(),
    reason="set STATSPAI_DCDH_TEXTBOOK_DIR to the cc_xd_didtextbook folder",
)



def _printed(value):
    """``value`` as Stata printed it.

    Equal up to the last printed digit, or to 2e-5 relative when that is
    larger: ``did_imputation``, ``did_multiplegt_dyn`` and
    ``twowayfeweights`` hold their working variables in single precision,
    which moves the sixth or seventh significant digit of what they print.
    """
    text = repr(float(value))
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text else 0
    tolerance = max(0.6 * 10.0 ** (-decimals), 2e-5 * abs(float(value)))
    return pytest.approx(float(value), abs=tolerance)


def _load(folder, name):
    return pd.read_stata(
        Path(ROOT) / "Data sets" / folder / name, convert_categoricals=False
    )


@pytest.fixture(scope="module")
def wolfers():
    return _load("Wolfers 2006", "wolfers2006_didtextbook.dta")


@pytest.fixture(scope="module")
def pierce():
    return _load("Pierce and Schott 2016", "pierce_schott_didtextbook.dta")


@pytest.fixture(scope="module")
def gentzkow():
    return _load("Gentzkow et al 2011", "gentzkowetal_didtextbook.dta")


@pytest.fixture(scope="module")
def moser():
    return _load("Moser and Voena 2012", "moser_voena_didtextbook.dta")


def _weights(result, n_plus, sum_plus, n_minus, sum_minus):
    mi = result.model_info
    assert (mi["n_positive"], mi["n_negative"]) == (n_plus, n_minus)
    assert mi["sum_positive"] == pytest.approx(sum_plus, abs=5e-5)
    assert mi["sum_negative"] == pytest.approx(sum_minus, abs=5e-5)


def _event_study(result):
    return result.model_info["event_study"].set_index("relative_time")


# --------------------------------------------------------------------------
# Wolfers (2006): binary staggered treatment, population weights
# --------------------------------------------------------------------------


def test_wolfers_static_twfe_and_its_weights(wolfers):
    fit = sp.regress(
        "div_rate ~ udl + C(state) + C(year)",
        data=wolfers,
        weights="stpop",
        cluster="state",
    )
    assert fit.params["udl"] == _printed(-0.0548378)
    assert fit.std_errors["udl"] == _printed(0.1507695)
    w = sp.twowayfeweights(
        wolfers,
        "div_rate",
        "state",
        "year",
        "udl",
        weights="stpop",
        test_random_weights=["exposurelength"],
    )
    assert w.estimate == pytest.approx(fit.params["udl"], rel=1e-9)
    _weights(w, 490, 1.0259, 32, -0.0259)
    row = w.model_info["random_weights"].loc["exposurelength"]
    assert row["coef"] == _printed(-8.2883613)
    assert row["se"] == _printed(0.21360588)
    assert row["correlation"] == _printed(-0.73253245)


def test_wolfers_event_study_coefficient_is_contaminated(wolfers):
    """The first effect of the TWFE event study, other effects as treatments."""
    others = [f"rel_time{k}" for k in range(2, 17)]
    leads = [f"rel_timeminus{k}" for k in range(1, 10)]
    w = sp.twowayfeweights(
        wolfers,
        "div_rate",
        "state",
        "year",
        "rel_time1",
        weights="stpop",
        other_treatments=others,
        controls=leads,
        test_random_weights=["year"],
    )
    assert w.estimate == _printed(0.2891561)
    _weights(w, 27, 1.0, 0, 0.0)
    other = w.model_info["other_treatments"]
    assert (other["rel_time2"]["n_positive"], other["rel_time2"]["n_negative"]) == (
        16,
        13,
    )
    assert other["rel_time2"]["sum_positive"] == pytest.approx(0.0119, abs=5e-5)
    assert (other["rel_time16"]["n_positive"], other["rel_time16"]["n_negative"]) == (
        65,
        35,
    )
    row = w.model_info["random_weights"].loc["year"]
    # the command works in single precision here
    assert row["coef"] == pytest.approx(-15.333901, rel=1e-5)
    assert row["correlation"] == pytest.approx(-0.23224592, rel=1e-5)


def test_wolfers_did_multiplegt_dyn(wolfers):
    r = sp.did_multiplegt_dyn(
        wolfers,
        "div_rate",
        group="state",
        time="year",
        treatment="udl",
        dynamic=12,
        placebo=13,
        weights="stpop",
        se_method="analytic",
        aggregation="switchers",
    )
    es = _event_study(r)
    for h, est, se in (
        (0, 0.3009669, 0.0875908),
        (3, 0.1816435, 0.0848985),
        (12, -0.5012722, 0.1654563),
        (-1, 0.0468044, 0.0493928),
        (-13, 0.0256646, 0.1832985),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)
    assert r.estimate == _printed(-0.0151066)
    assert r.se == _printed(0.0987446)
    assert r.model_info["joint_placebo_test"]["pvalue"] == pytest.approx(
        1.226e-06, rel=1e-3
    )
    # the textbook's condition number of the placebos' covariance
    V = r.model_info["event_study_vcov"]
    pre = [h for h in V.index if h < 0]
    eig = np.linalg.eigvalsh(V.loc[pre, pre].to_numpy())
    assert eig.max() / eig.min() == _printed(4286.808)


def test_wolfers_imputation(wolfers):
    d = wolfers.copy()
    d.loc[d["cohort"] == 0, "cohort"] = np.nan
    r = sp.did_imputation(
        d,
        "div_rate",
        "state",
        "year",
        "cohort",
        horizon=list(range(13)),
        pretrends=13,
        autosample=True,
        weights="stpop",
    )
    es = _event_study(r)
    for h, est, se in (
        (0, 0.2649129, 0.1022424),
        (12, -0.4521017, 0.1956893),
        (-1, -0.0063006, 0.1702816),
        (-13, -0.0784114, 0.0749313),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)


def test_wolfers_sun_abraham_with_binned_ends(wolfers):
    """eventstudyinteract with ``<= -14`` and ``>= 12`` bins, time-varying weights.

    52 state-years have no divorce rate, so the cohort shares at a relative
    time are shares of observations, not of states.
    """
    d = wolfers[wolfers["cohort"] != 1956].copy()
    d.loc[d["cohort"] == 0, "cohort"] = np.nan
    r = sp.sun_abraham(
        d,
        "div_rate",
        g="cohort",
        t="year",
        i="state",
        weights="stpop",
        cluster="state",
        event_window=(-14, 12),
        window_rule="bin",
    )
    es = _event_study(r)
    for e, est, se in (
        (0, 0.2927196, 0.2009387),
        (1, 0.2981485, 0.0886750),
        (12, -0.5268966, 0.1957938),
        (-2, 0.0585340, 0.0547403),
        (-14, -0.0118369, 0.1649912),
    ):
        assert es.loc[e, "att"] == _printed(est)
        assert es.loc[e, "se"] == _printed(se)


# --------------------------------------------------------------------------
# Pierce and Schott (2016): one period, continuous treatment
# --------------------------------------------------------------------------


def test_pierce_hc2_with_adjusted_degrees_of_freedom(pierce):
    r = sp.regress("delta2001 ~ ntrgap", data=pierce, robust="hc2", dfadjust=True)
    ci = r.conf_int()
    assert r.params["ntrgap"] == _printed(-0.0612112)
    assert r.std_errors["ntrgap"] == _printed(0.0401833)
    assert r.pvalues["ntrgap"] == pytest.approx(0.136, abs=5e-4)
    assert ci.loc["ntrgap"].iloc[0] == _printed(-0.1426751)
    assert ci.loc["ntrgap"].iloc[1] == _printed(0.0202527)
    pre = sp.regress(
        "ntrgap ~ lemp1997 + lemp1998 + lemp1999 + lemp2000",
        data=pierce,
        robust="hc2",
        dfadjust=True,
    )
    joint = sp.test(pre, "lemp1997 lemp1998 lemp1999 lemp2000")
    assert joint["statistic"] == pytest.approx(2.39, abs=5e-3)
    assert joint["pvalue"] == pytest.approx(0.0560, abs=5e-5)
    assert joint["df"] == (4, 98)


def test_pierce_first_difference_weights(pierce):
    w = sp.twowayfeweights(
        pierce,
        "delta2001",
        "indusid",
        "cons",
        "ntrgap",
        type="fdTR",
        treat_level="ntrgap",
    )
    assert w.estimate == _printed(-0.0612112)
    _weights(w, 62, 1.3190, 41, -0.3190)


def test_pierce_did_had(pierce):
    long = pd.wide_to_long(pierce, "lemp", i="indusid", j="year").reset_index()
    long.loc[long["year"] <= 2000, "ntrgap"] = 0
    r = sp.did_had(long, "lemp", "indusid", "year", "ntrgap", effects=4, placebo=3)
    d = r.detail.set_index(["type", "relative_time"])
    for key, est, se in (
        (("effect", 1), 0.0149623, 0.0866229),
        (("effect", 4), 0.0348117, 0.6413456),
        (("placebo", -1), -0.0198445, 0.1299684),
        (("placebo", -3), -0.1213829, 0.1814087),
    ):
        assert d.loc[key, "estimate"] == _printed(est)
        assert d.loc[key, "se"] == _printed(se)
    first = d.loc[("effect", 1)]
    assert first["bandwidth"] == _printed(0.1465417)
    assert first["n_in_bw"] == 21
    assert first["qug_pvalue"] == _printed(0.1398623)


# --------------------------------------------------------------------------
# Gentzkow, Shapiro and Sinkinson (2011): a count treatment, both directions
# --------------------------------------------------------------------------

#: The two counties the reference command discards as controls because of
#: what their treatment does after they switch (module docstring of
#: statspai.did.did_multiplegt_dyn). Without them the two implementations
#: estimate on the same sample.
DISCARDED_BY_THE_COMMAND = [30093, 53033]


def test_gentzkow_weights(gentzkow):
    g = gentzkow
    w = sp.twowayfeweights(g, "prestout", "cnty90", "year", "numdailies")
    assert w.estimate == _printed(0.0029393)
    _weights(w, 6180, 1.4740, 4198, -0.4740)
    assert w.model_info["n_treated_cells"] == 10378

    trends = pd.get_dummies(g["styr"], prefix="styr", dtype=float)
    gt = pd.concat([g, trends], axis=1)
    cols = list(trends.columns)
    fe = sp.twowayfeweights(gt, "prestout", "cnty90", "year", "numdailies", controls=cols)
    assert fe.estimate == _printed(-0.00121217)
    _weights(fe, 6195, 1.5331, 4147, -0.5331)
    fd = sp.twowayfeweights(
        gt,
        "changeprestout",
        "cnty90",
        "year",
        "changedailies",
        type="fdTR",
        treat_level="numdailies",
        controls=cols,
        test_random_weights=["year"],
    )
    assert fd.estimate == _printed(0.0025907)
    assert fd.se == _printed(0.0009351)
    _weights(fd, 5371, 2.4271, 4505, -1.4271)
    row = fd.model_info["random_weights"].loc["year"]
    assert row["coef"] == _printed(-0.1674271)
    assert row["se"] == _printed(0.05101171)
    assert row["correlation"] == _printed(-0.0631614)

    lag = sp.twowayfeweights(
        g, "prestout", "cnty90", "year", "numdailies", other_treatments=["lag_numdailies"]
    )
    assert lag.estimate == _printed(-0.0007962)
    _weights(lag, 5754, 1.8541, 4302, -0.8541)
    other = lag.model_info["other_treatments"]["lag_numdailies"]
    assert (other["n_positive"], other["n_negative"]) == (4721, 4618)
    assert other["sum_positive"] == pytest.approx(1.2137, abs=5e-5)


def test_gentzkow_did_m_per_unit_of_treatment(gentzkow):
    """``did_multiplegt_old``: 0.0057790681, placebos -0.0000125426 and -0.0005833373.

    The estimator used to sign every switcher from a positive baseline as a
    switch off and returned -0.00082.
    """
    r = sp.did_multiplegt(
        gentzkow,
        "prestout",
        "cnty90",
        "year",
        "numdailies",
        placebo=2,
        n_boot=2,
        seed=0,
        placebo_sign="r",
    )
    assert r.estimate == pytest.approx(0.00577906811025418, rel=1e-7)
    assert r.model_info["n_switchers"] == 4423
    placebo = {p["lag"]: p["estimate"] for p in r.model_info["placebo"]}
    assert placebo[-1] == pytest.approx(-0.000012542572448937, rel=1e-5)
    assert placebo[-2] == _printed(-0.000583337279135521)


def _newspapers(data, **kwargs):
    return sp.did_multiplegt_dyn(
        data,
        "prestout",
        group="cnty90",
        time="year",
        treatment="numdailies",
        se_method="analytic",
        aggregation="switchers",
        **kwargs,
    )


def test_gentzkow_event_study_on_the_command_s_sample(gentzkow):
    g = gentzkow[~gentzkow["cnty90"].isin(DISCARDED_BY_THE_COMMAND[:1])]
    r = _newspapers(g, dynamic=3, placebo=4, effects_equal=True)
    es = _event_study(r)
    for h, est, se, n in (
        (0, 0.0144244, 0.0042477, 1119),
        (1, 0.0190899, 0.0058429, 1054),
        (2, 0.0207147, 0.0079164, 984),
        (3, 0.0272653, 0.0097924, 917),
        (-1, -0.0005025, 0.0051322, 902),
        (-2, 0.0020594, 0.0085031, 746),
        (-4, 0.0006573, 0.0175032, 441),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)
        assert es.loc[h, "n_switchers"] == n
    assert r.estimate == _printed(0.0160565)
    assert r.se == _printed(0.0047761)
    mi = r.model_info
    assert mi["joint_effects_test"]["pvalue"] == _printed(0.00681389)
    assert mi["joint_placebo_test"]["pvalue"] == pytest.approx(0.99219793, rel=1e-7)
    assert mi["effects_equal_test"]["pvalue"] == _printed(0.41515197)

    n = _newspapers(g, dynamic=3, placebo=4, normalized=True, normalized_weights=True)
    es = _event_study(n)
    for h, est, se in (
        (0, 0.0120186, 0.0035392),
        (3, 0.0053873, 0.0019348),
        (-1, -0.0004158, 0.0042470),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)
    lags = n.model_info["normalized_weights"]
    np.testing.assert_allclose(
        lags[4].to_numpy(), [0.2777, 0.2568, 0.2271, 0.2383], atol=5e-5
    )
    np.testing.assert_allclose(lags[2].dropna().to_numpy(), [0.4849, 0.5151], atol=5e-5)


def test_gentzkow_keeps_the_control_the_command_discards(gentzkow):
    """One county, three switchers: the documented departure."""
    r = _newspapers(gentzkow, dynamic=3, placebo=1)
    es = _event_study(r)
    assert list(es.loc[[0, 1, 2, 3], "n_switchers"]) == [1122, 1057, 988, 917]
    assert es.loc[0, "att"] == pytest.approx(0.014548, abs=5e-7)
    # the last effect does not involve the three switchers: same as Stata
    assert es.loc[3, "att"] == _printed(0.0272653)
    assert es.loc[3, "se"] == _printed(0.0097924)


def test_gentzkow_paths(gentzkow):
    g = gentzkow[~gentzkow["cnty90"].isin(DISCARDED_BY_THE_COMMAND[:1])]
    r = _newspapers(g, dynamic=1, by_path=3, design=0.8)
    expected = [
        ((0.0, 1.0, 1.0), 343, 0.0131318, 0.0071358, 0.009357, 0.0090553, 0.0112444),
        ((0.0, 1.0, 0.0), 187, 0.0144984, 0.0103889, 0.0082605, 0.0115639, 0.0227589),
        ((0.0, 1.0, 2.0), 131, 0.0212045, 0.0103034, 0.0314473, 0.0132732, 0.0175506),
    ]
    for got, (path, n, e1, s1, e2, s2, avg) in zip(r.model_info["by_path"], expected):
        assert got["path"] == path and got["n_switchers"] == n
        es = got["event_study"]
        assert es["att"].iloc[0] == _printed(e1)
        assert es["se"].iloc[0] == _printed(s1)
        assert es["att"].iloc[1] == _printed(e2)
        assert es["se"].iloc[1] == _printed(s2)
        assert got["estimate"] == _printed(avg)
    design = r.model_info["design"]
    assert list(design["n_groups"].iloc[:5]) == [343, 187, 131, 57, 47]


def test_gentzkow_same_switchers_on_a_subsample(gentzkow):
    g = gentzkow[~gentzkow["cnty90"].isin(DISCARDED_BY_THE_COMMAND)]
    sub = g[(g["year"] < g["first_change"]) | (g["same_treat_after_first_change"] == 1)]
    r = _newspapers(sub, dynamic=1, same_switchers=True, effects_equal=True)
    es = _event_study(r)
    assert list(es["n_switchers"]) == [509, 509]
    assert es.loc[0, "att"] == _printed(0.015118)
    assert es.loc[0, "se"] == _printed(0.005914)
    assert es.loc[1, "att"] == _printed(0.0158182)
    assert es.loc[1, "se"] == _printed(0.0073511)
    assert r.estimate == _printed(0.0135048)
    assert r.model_info["effects_equal_test"]["pvalue"] == pytest.approx(
        0.8971383, rel=1e-6
    )


# --------------------------------------------------------------------------
# Moser and Voena (2012): one treatment date, 7,248 subclasses
# --------------------------------------------------------------------------


def test_moser_imputation_equals_the_event_study_without_leads(moser):
    m = moser.copy()
    m["cohort"] = np.where(m["treatmentgroup"] == 1, 1919, np.nan)
    r = sp.did_imputation(
        m, "patents", "subclass", "year", "cohort", horizon=list(range(21)),
        autosample=True,
    )  # fmt: skip
    es = _event_study(r)
    for h, est, se in (
        (0, 0.0108005, 0.0309719),
        (13, 0.6294968, 0.1040416),
        (20, 0.7847135, 0.1189967),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)


@pytest.mark.slow
def test_moser_did_multiplegt_dyn_within_baseline_patent_cells(moser):
    r = sp.did_multiplegt_dyn(
        moser,
        "patents",
        group="subclass",
        time="year",
        treatment="twea",
        dynamic=20,
        placebo=18,
        trends_nonparam=["patents1900"],
        se_method="analytic",
        aggregation="switchers",
    )
    es = _event_study(r)
    for h, est, se in (
        (0, 0.0123774, 0.0407900),
        (20, 0.7738304, 0.1203650),
        (-1, -0.0210605, 0.0457692),
        (-18, 0.0500484, 0.0489266),
    ):
        assert es.loc[h, "att"] == _printed(est)
        assert es.loc[h, "se"] == _printed(se)
    assert r.estimate == _printed(0.2893714)
    assert r.se == _printed(0.0489889)
    assert r.model_info["joint_placebo_test"]["pvalue"] == pytest.approx(
        1.685e-12, rel=1e-3
    )
