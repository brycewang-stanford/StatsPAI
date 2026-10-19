"""Reference parity: weighted and counting-process ``sp.cox`` /
``sp.kaplan_meier`` vs R ``survival`` 3.8.3 and Stata 18 ``stcox``.

Frozen references, generated from the committed CSVs:

* ``_fixtures/survival_weighted_R.json`` by
  ``_generate_survival_weighted_R.R`` (``coxph`` with ``weights=``,
  ``robust=``, ``cluster=``, ``Surv(start, stop, event)``, ``strata()``;
  ``survfit`` with weights, entry times and the infinitesimal-jackknife
  variance);
* ``_fixtures/survival_weighted_Stata.csv`` by
  ``_generate_survival_weighted_stata.do`` (``stcox [pw=]``,
  ``vce(robust)``, ``vce(cluster)``, ``stset ..., enter()`` and
  multi-record ``time0() id()``).

Tolerances (observed worst gaps in parentheses):

* R: coefficients and SEs 1e-10 relative (~7e-15), log partial
  likelihood 1e-8 absolute (~4e-12), concordance 1e-12 (~4e-16); KM
  survival / SE / interval 1e-12 relative (~4e-15).
* Stata: coefficients and SEs 1e-7 relative (~1e-9). Stata stops its
  Newton iterations at a looser tolerance than ours, which is the whole
  gap; the variances are the same formula (``ssc='stata'``: ``N/(N-1)``
  robust, ``G/(G-1)`` clustered).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.survival._cox_core import harrell_c
from statspai.survival._cox_inputs import risk_set_concordance

_FIX = pathlib.Path(__file__).parent / "_fixtures"
X3 = ["x1", "x2", "a"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_FIX / "survival_weighted_data.csv")


@pytest.fixture(scope="module")
def cp():
    return pd.read_csv(_FIX / "survival_counting_data.csv")


@pytest.fixture(scope="module")
def ref_r():
    with open(_FIX / "survival_weighted_R.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ref_stata():
    return pd.read_csv(_FIX / "survival_weighted_Stata.csv")


def _r_fit(tag, ties, df, cp):
    base = dict(duration="time", event="event", x=X3, ties=ties)
    if tag == "w_model":
        return sp.cox(data=df, weights="w", weights_as="frequency", **base)
    if tag == "w_robust":
        return sp.cox(data=df, weights="w", weights_as="sampling", **base)
    if tag == "w_cluster":
        return sp.cox(data=df, weights="w", cluster="cl", ssc="r", **base)
    if tag == "entry":
        return sp.cox(data=df, start="entry", **base)
    if tag == "entry_w_robust":
        return sp.cox(
            data=df, start="entry", weights="w", weights_as="sampling", **base
        )
    if tag == "unw_robust":
        return sp.cox(data=df, robust="hc0", **base)
    cpb = dict(duration="tstop", start="tstart", event="event", ties=ties)
    if tag == "cp_id":
        return sp.cox(data=cp, x=["z", "x2", "a"], cluster="id", ssc="r", **cpb)
    if tag == "cp_strata":
        return sp.cox(data=cp, x=["z", "a"], strata="x2", **cpb)
    raise AssertionError(tag)


R_CASES = [
    f"{t}_{ties}"
    for ties in ("efron", "breslow")
    for t in (
        "w_model",
        "w_robust",
        "w_cluster",
        "entry",
        "entry_w_robust",
        "cp_id",
        "cp_strata",
        "unw_robust",
    )
]


@pytest.mark.parametrize("case", R_CASES)
def test_cox_matches_coxph(case, df, cp, ref_r):
    tag, ties = case.rsplit("_", 1)
    r = ref_r["cox"][case]
    fit = _r_fit(tag, ties, df, cp)
    np.testing.assert_allclose(fit.params.values, r["coef"], rtol=1e-10)
    np.testing.assert_allclose(fit.std_errors.values, r["se"], rtol=1e-10)
    assert abs(fit.diagnostics["Log-likelihood"] - r["loglik"]) < 1e-8
    assert abs(fit.concordance - r["concordance"]) < 1e-12


def _stata_fit(tag, df, cp):
    base = dict(data=df, duration="time", event="event", x=X3, ssc="stata")
    if tag == "pw_breslow":
        return sp.cox(**base, weights="w", weights_as="sampling", ties="breslow")
    if tag == "pw_cluster_breslow":
        return sp.cox(
            **base, weights="w", weights_as="sampling", cluster="cl", ties="breslow"
        )
    if tag == "unw_robust_efron":
        return sp.cox(**base, robust="hc0")
    if tag == "unw_robust_breslow":
        return sp.cox(**base, robust="hc0", ties="breslow")
    if tag == "zw_breslow":
        d = df.assign(zw=df["w"])
        d.loc[:2, "zw"] = 0.0
        return sp.cox(
            **{**base, "data": d}, weights="zw", weights_as="sampling", ties="breslow"
        )
    if tag == "entry_efron":
        return sp.cox(data=df, duration="time", event="event", x=X3, start="entry")
    if tag == "entry_robust_efron":
        return sp.cox(**base, start="entry", robust="hc0")
    if tag == "cp_id_efron":
        return sp.cox(
            data=cp,
            duration="tstop",
            start="tstart",
            event="event",
            x=["z", "x2", "a"],
            cluster="id",
            ssc="stata",
        )
    raise AssertionError(tag)


@pytest.mark.parametrize(
    "tag",
    [
        "pw_breslow",
        "pw_cluster_breslow",
        "unw_robust_efron",
        "unw_robust_breslow",
        "zw_breslow",
        "entry_efron",
        "entry_robust_efron",
        "cp_id_efron",
    ],
)
def test_cox_matches_stcox(tag, df, cp, ref_stata):
    ref = ref_stata[ref_stata["tag"] == tag].sort_values("j")
    fit = _stata_fit(tag, df, cp)
    np.testing.assert_allclose(fit.params.values, ref["coef"].values, rtol=1e-7)
    np.testing.assert_allclose(fit.std_errors.values, ref["se"].values, rtol=1e-7)
    # Stata drops zero-weight observations from N, as sp.cox does
    assert fit.data_info["nobs"] == int(ref["N"].iloc[0])


KM_CASES = {
    "w_ij": lambda d, c: sp.kaplan_meier(
        d[d.a == 1], "time", "event", weights="w", conf_type="log-log"
    ),
    "w_greenwood": lambda d, c: sp.kaplan_meier(
        d[d.a == 1],
        "time",
        "event",
        weights="w",
        se_method="greenwood",
        conf_type="log",
    ),
    "unw_ij": lambda d, c: sp.kaplan_meier(
        d, "time", "event", se_method="ij", conf_type="plain"
    ),
    "entry_greenwood": lambda d, c: sp.kaplan_meier(
        d, "time", "event", start="entry", conf_type="log"
    ),
    "cp_cluster": lambda d, c: sp.kaplan_meier(
        c,
        "tstop",
        "event",
        start="tstart",
        weights="w",
        cluster="id",
        conf_type="log-log",
    ),
    "w_cluster": lambda d, c: sp.kaplan_meier(
        d, "time", "event", weights="w", cluster="cl", conf_type="log-log"
    ),
}


@pytest.mark.parametrize("case", list(KM_CASES))
def test_kaplan_meier_matches_survfit(case, df, cp, ref_r):
    r = ref_r["km"][case]
    tab = KM_CASES[case](df, cp).survival_table.iloc[1:]
    np.testing.assert_allclose(tab["time"].values, r["time"], rtol=0, atol=0)
    for col, key in [
        ("survival", "surv"),
        ("std_err", "se"),
        ("n_risk", "n_risk"),
        ("ci_lower", "lower"),
        ("ci_upper", "upper"),
    ]:
        a = tab[col].values.astype(float)
        b = np.array([np.nan if v is None else v for v in r[key]], dtype=float)
        ok = np.isfinite(b)
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, atol=1e-15)


# -- identities and boundaries ------------------------------------------------


def test_unit_weights_reproduce_the_unweighted_fit(df):
    a = sp.cox(data=df, duration="time", event="event", x=X3, robust="hc0")
    b = sp.cox(
        data=df.assign(one=1.0),
        duration="time",
        event="event",
        x=X3,
        weights="one",
        robust="hc0",
    )
    np.testing.assert_allclose(b.params.values, a.params.values, rtol=1e-12)
    np.testing.assert_allclose(b.std_errors.values, a.std_errors.values, rtol=1e-12)
    # risk-set concordance with unit weights is Harrell's C
    assert abs(b.concordance - a.concordance) < 1e-14


def test_breslow_integer_weights_equal_duplicated_rows(df):
    """Under Breslow ties a weight of 2 is the row counted twice."""
    w = (df["id"] % 3 == 0).astype(int) + 1
    dup = df.loc[df.index.repeat(w)].reset_index(drop=True)
    a = sp.cox(
        data=df.assign(fw=w),
        duration="time",
        event="event",
        x=X3,
        weights="fw",
        ties="breslow",
    )
    b = sp.cox(data=dup, duration="time", event="event", x=X3, ties="breslow")
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-10)
    np.testing.assert_allclose(a.std_errors.values, b.std_errors.values, rtol=1e-10)
    np.testing.assert_allclose(
        a.diagnostics["Log-likelihood"], b.diagnostics["Log-likelihood"], rtol=1e-12
    )


def test_split_intervals_without_time_varying_covariate_change_nothing(df, cp):
    """Splitting each subject at t=3 with the covariate unchanged leaves the
    partial likelihood unchanged."""
    d = df[df["entry"] == 0]
    full = sp.cox(data=d, duration="time", event="event", x=X3)
    rows = []
    for r in d.itertuples():
        if r.time > 3:
            rows.append((r.id, 0.0, 3.0, 0, r.x1, r.x2, r.a))
            rows.append((r.id, 3.0, r.time, r.event, r.x1, r.x2, r.a))
        else:
            rows.append((r.id, 0.0, r.time, r.event, r.x1, r.x2, r.a))
    sp_df = pd.DataFrame(rows, columns=["id", "t0", "t1", "event", *X3])
    split = sp.cox(data=sp_df, duration="t1", start="t0", event="event", x=X3)
    np.testing.assert_allclose(split.params.values, full.params.values, rtol=1e-10)
    np.testing.assert_allclose(
        split.std_errors.values, full.std_errors.values, rtol=1e-10
    )
    assert abs(split.concordance - full.concordance) < 1e-12


def test_risk_set_concordance_is_harrell_without_weights_or_entry():
    rng = np.random.default_rng(3)
    T = np.round(rng.exponential(size=400), 1)
    E = rng.integers(0, 2, 400)
    score = np.round(rng.normal(size=400), 1)  # ties in score and time
    assert abs(risk_set_concordance(score, T, E) - harrell_c(score, T, E)) < 1e-14


def test_zero_weight_rows_are_dropped(df):
    d = df.assign(zw=df["w"])
    d.loc[:4, "zw"] = 0.0
    a = sp.cox(
        data=d,
        duration="time",
        event="event",
        x=X3,
        weights="zw",
        weights_as="sampling",
    )
    b = sp.cox(
        data=d.iloc[5:],
        duration="time",
        event="event",
        x=X3,
        weights="zw",
        weights_as="sampling",
    )
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors.values, b.std_errors.values, rtol=1e-12)
    assert a.data_info["nobs"] == len(df) - 5


def test_ssc_stata_scales_robust_variance_by_n_over_n_minus_1(df):
    a = sp.cox(data=df, duration="time", event="event", x=X3, robust="hc0")
    b = sp.cox(data=df, duration="time", event="event", x=X3, robust="hc0", ssc="stata")
    n = len(df)
    np.testing.assert_allclose(
        b.std_errors.values, a.std_errors.values * np.sqrt(n / (n - 1)), rtol=1e-13
    )


def test_bad_inputs_fail_loudly(df, cp):
    with pytest.raises(MethodIncompatibility):
        sp.cox(
            data=df.assign(nw=-df["w"]),
            duration="time",
            event="event",
            x=X3,
            weights="nw",
        )
    with pytest.raises(DataInsufficient):
        sp.cox(data=df, duration="entry", start="time", event="event", x=X3)
    with pytest.raises(MethodIncompatibility):
        sp.cox(data=df, duration="time", event="event", x=X3, ssc="fixest")
    with pytest.raises(MethodIncompatibility):
        sp.kaplan_meier(
            df, "time", "event", cluster="cl", se_method="greenwood", conf_type="log"
        )
    fit = sp.cox(
        data=df,
        duration="time",
        event="event",
        x=X3,
        weights="w",
        weights_as="sampling",
    )
    with pytest.raises(MethodIncompatibility):
        fit.ph_test()


def test_non_integer_weights_with_model_based_variance_warn(df):
    with pytest.warns(UserWarning, match="weights_as='sampling'"):
        sp.cox(data=df, duration="time", event="event", x=X3, weights="w")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sp.cox(
            data=df,
            duration="time",
            event="event",
            x=X3,
            weights="w",
            weights_as="sampling",
        )


@pytest.mark.parametrize(
    "tag, script, which",
    [
        (
            "pw_breslow",
            "stset time [pw=w], failure(event)\nstcox x1 x2 a, breslow",
            "df",
        ),
        (
            "pw_cluster_breslow",
            "stset time [pw=w], failure(event)\nstcox x1 x2 a, vce(cluster cl)",
            "df",
        ),
        (
            "unw_robust_efron",
            "stset time, failure(event)\nstcox x1 x2 a, efron vce(robust)",
            "df",
        ),
        (
            "entry_robust_efron",
            "stset time, failure(event) enter(time entry)\n"
            "stcox x1 x2 a, efron vce(robust)",
            "df",
        ),
        (
            "cp_id_efron",
            "stset tstop, failure(event) time0(tstart) id(id)\n"
            "stcox z x2 a, efron vce(robust)",
            "cp",
        ),
    ],
)
def test_stata_scripts_reproduce_stcox(tag, script, which, df, cp, ref_stata):
    """The same do-file lines run through sp.stata give Stata's numbers."""
    ref = ref_stata[ref_stata["tag"] == tag].sort_values("j")
    fit = sp.stata(script, df if which == "df" else cp)
    np.testing.assert_allclose(np.asarray(fit.params), ref["coef"].values, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(fit.std_errors), ref["se"].values, rtol=1e-7)


def test_stata_refuses_pweights_with_efron(df):
    with pytest.raises(MethodIncompatibility, match="efron"):
        sp.stata("stset time [pw=w], failure(event)\nstcox x1 x2 a, efron", df)
