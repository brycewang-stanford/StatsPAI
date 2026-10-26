"""``crit_dist=`` on the regression event studies and its 1.41 default.

Owner decision 2026-10-21, announced in 1.40.1 with a ``FutureWarning`` and
in effect since 1.41: ``sp.event_study``, ``sp.sun_abraham`` and
``sp.did_2x2`` use the Stata / fixest critical values (t(G - 1) with
cluster-robust SEs, t(N - K) otherwise) by default. Pinned here:

* the default reproduces Stata 18 ``reghdfe`` / ``lincom`` /
  ``eventstudyinteract`` / ``regress`` on the castle-doctrine panel
  (``_fixtures/es_crit_Stata.csv``) to 1e-10, and the MIGRATION.md table;
* the default is the same fit as ``crit_dist='t'``, which gives exactly
  the numbers of ``parity='stata'`` (held to Stata and fixest by
  ``test_es_crit_parity.py``);
* the legacy values (``'normal'``, ``'residual'``) keep the pre-1.41
  numbers, and no call warns any more;
* contradicting ``parity=`` raises a ``MethodIncompatibility`` with a
  recovery hint;
* library entry points that call these functions on the user's behalf
  follow the default and do not warn.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"
TOL = 1e-10

ES_TERMS = {
    k: (f"es_m{-k}" if k < 0 else f"es_p{k}") for k in [-5, -4, -3, -2, 0, 1, 2, 3, 4]
}
SA_STATA = {k: (f"L{-k}" if k < 0 else f"F{k}") for k in [*range(-9, -1), *range(0, 6)]}


def _future(record) -> list:
    return [
        w
        for w in record
        if issubclass(w.category, FutureWarning) and "crit_dist" in str(w.message)
    ]


def _run(fn, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fn(**kw)
    return res, _future(rec)


@pytest.fixture(scope="module")
def castle() -> pd.DataFrame:
    d = sp.datasets.castle_doctrine()
    return d.assign(gvar=d.effyear.fillna(0).astype(int))


@pytest.fixture(scope="module")
def castle_2x2(castle) -> pd.DataFrame:
    d2 = castle[(castle.effyear == 2006) | castle.effyear.isna()].copy()
    d2["tr"] = d2.effyear.notna().astype(int)
    d2["po"] = (d2.year >= 2006).astype(int)
    return d2


def _es(castle, **kw):
    return _run(
        sp.event_study,
        data=castle,
        y="l_homicide",
        treat_time="effyear",
        time="year",
        unit="sid",
        window=(-5, 4),
        weights="popwt",
        **kw,
    )


def _sa(castle, panel=True, **kw):
    return _run(
        sp.sun_abraham,
        data=castle,
        y="l_homicide",
        g="gvar",
        t="year",
        i="sid" if panel else None,
        weights="popwt",
        control_group="nevertreated",
        aggregation="event_time",
        **kw,
    )


def _dd(castle_2x2, **kw):
    return _run(
        sp.did_2x2, data=castle_2x2, y="l_homicide", treat="tr", time="po", **kw
    )


def _same(a, b, table="event_study") -> None:
    if table == "detail":
        pd.testing.assert_frame_equal(a.detail, b.detail, check_exact=True)
    else:
        pd.testing.assert_frame_equal(
            a.model_info[table], b.model_info[table], check_exact=True
        )
    assert (a.estimate, a.se, a.pvalue, tuple(a.ci)) == (
        b.estimate,
        b.se,
        b.pvalue,
        tuple(b.ci),
    )
    ca, cb = dict(a.model_info["conventions"]), dict(b.model_info["conventions"])
    ca.pop("parity"), cb.pop("parity")
    assert ca == cb


def _stata(tag: str) -> pd.DataFrame:
    ref = pd.read_csv(
        FIX / "es_crit_Stata.csv", skipinitialspace=True, dtype={"term": str}
    )
    return ref[ref.tag == tag].set_index("term")


def _check(table: pd.DataFrame, ref: pd.DataFrame, terms: dict) -> None:
    assert len(terms) > 0
    for k, term in terms.items():
        row, r = table.loc[k], ref.loc[term]
        assert row["se"] == pytest.approx(r["se"], rel=TOL), (k, "se")
        assert abs(row["pvalue"] - r["pvalue"]) < TOL, (k, "pvalue")
        assert abs(row["ci_lower"] - r["ll"]) < TOL, (k, "ll")
        assert abs(row["ci_upper"] - r["ul"]) < TOL, (k, "ul")


# --------------------------------------------------------------------------
# The 1.41 default reproduces Stata (and so the MIGRATION.md table)
# --------------------------------------------------------------------------


def test_event_study_default_reproduces_stata(castle):
    res, w = _es(castle)
    assert w == []
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        _stata("es_castle_cl"),
        ES_TERMS,
    )
    att = _stata("es_castle_cl_att").loc["att"]
    assert res.estimate == pytest.approx(att["b"], rel=1e-12)
    assert abs(res.pvalue - att["pvalue"]) < TOL
    assert abs(res.ci[0] - att["ll"]) < TOL and abs(res.ci[1] - att["ul"]) < TOL
    assert res.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "t",
        "crit_df": 49,
    }
    # MIGRATION.md, "1.40.2 -> 1.41": lead -3 and the headline ATT
    lead = res.model_info["event_study"].set_index("relative_time").loc[-3]
    assert round(lead["ci_lower"], 5) == -0.00053
    assert round(lead["ci_upper"], 5) == 0.10196
    assert round(lead["pvalue"], 4) == 0.0523
    assert (round(res.ci[0], 5), round(res.ci[1], 5)) == (0.04948, 0.14593)
    assert f"{res.pvalue:.1e}" == "1.7e-04"


def test_did_2x2_clustered_default_reproduces_stata(castle_2x2):
    res, w = _dd(castle_2x2, cluster="sid")
    assert w == []
    r = _stata("2x2_castle_cl").loc["trpo"]
    assert abs(res.pvalue - r["pvalue"]) < TOL
    assert abs(res.ci[0] - r["ll"]) < TOL and abs(res.ci[1] - r["ul"]) < TOL
    assert res.model_info["conventions"]["crit_df"] == int(r["df"]) == 41
    # MIGRATION.md: [-0.10174, 0.23821], p = 0.422
    assert (round(res.ci[0], 5), round(res.ci[1], 5)) == (-0.10174, 0.23821)
    assert round(res.pvalue, 3) == 0.422


def test_sun_abraham_default_reproduces_eventstudyinteract(castle):
    res, w = _sa(castle)
    assert w == []
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        _stata("sa_castle_cl"),
        SA_STATA,
    )
    assert res.model_info["conventions"]["crit_df"] == res.model_info["n_clusters"] - 1


# --------------------------------------------------------------------------
# The default is crit_dist='t' (= parity='stata'); legacy values keep 1.40
# --------------------------------------------------------------------------


def test_event_study_t_equals_parity_stata(castle):
    t, w = _es(castle, crit_dist="t")
    st, _ = _es(castle, parity="stata")
    assert w == []
    _same(t, st)
    conv = t.model_info["conventions"]
    assert conv == {"parity": None, "crit_dist": "t", "crit_df": 49}
    # the Stata reghdfe row for the lead -3 (es_castle_cl / es_m3), directly
    ref = pd.read_csv(FIX / "es_crit_Stata.csv", skipinitialspace=True)
    r = ref[(ref.tag == "es_castle_cl") & (ref.term == "es_m3")].iloc[0]
    row = t.model_info["event_study"].set_index("relative_time").loc[-3]
    assert abs(row["pvalue"] - r["pvalue"]) < 1e-10
    assert abs(row["ci_lower"] - r["ll"]) < 1e-10
    assert abs(row["ci_upper"] - r["ul"]) < 1e-10


def test_event_study_default_is_t_and_normal_is_the_old_rule(castle):
    base, w0 = _es(castle)
    t, _ = _es(castle, crit_dist="t")
    leg, w1 = _es(castle, crit_dist="normal")
    assert w0 == [] and w1 == []
    _same(base, t)
    assert leg.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "normal",
        "crit_df": None,
    }
    # same coefficients and SEs; the interval is the normal quantile
    es = leg.model_info["event_study"]
    es = es[~es.is_reference]
    pd.testing.assert_series_equal(
        es.se, base.model_info["event_study"].loc[es.index, "se"]
    )
    z = stats.norm.ppf(0.975)
    assert ((es.ci_upper - es.att) / es.se - z).abs().max() < 1e-12
    # MIGRATION.md: the pre-1.41 lead -3 interval [0.00074, 0.10070]
    lead = es.set_index("relative_time").loc[-3]
    assert (round(lead["ci_lower"], 5), round(lead["ci_upper"], 5)) == (
        0.00074,
        0.10070,
    )


@pytest.mark.parametrize("panel", [True, False])
def test_sun_abraham_t_equals_parity_stata(castle, panel):
    t, w = _sa(castle, panel, crit_dist="t")
    st, _ = _sa(castle, panel, parity="stata")
    assert w == []
    _same(t, st)
    # crit_dist touches the critical values only, never the share term
    assert t.model_info["share_variance"] is True


@pytest.mark.parametrize("panel", [True, False])
def test_sun_abraham_default_is_t_and_normal_is_the_old_rule(castle, panel):
    base, w0 = _sa(castle, panel)
    t, _ = _sa(castle, panel, crit_dist="t")
    leg, w1 = _sa(castle, panel, crit_dist="normal")
    assert w0 == [] and w1 == []
    _same(base, t)
    assert leg.model_info["conventions"]["crit_dist"] == "normal"
    assert leg.model_info["conventions"]["crit_df"] is None


def test_sun_abraham_t_with_fixed_shares(castle):
    """crit_dist='t' + share_variance=False is parity='r'."""
    a, _ = _sa(castle, crit_dist="t", share_variance=False)
    b, _ = _sa(castle, parity="r")
    _same(a, b)


@pytest.mark.parametrize(
    "kw",
    [{}, {"cluster": "sid"}, {"cluster": "sid", "weights": "popwt"}],
)
def test_did_2x2_t_equals_parity_stata(castle_2x2, kw):
    t, w = _dd(castle_2x2, crit_dist="t", **kw)
    st, _ = _dd(castle_2x2, parity="stata", **kw)
    assert w == []
    _same(t, st, table="detail")


@pytest.mark.parametrize(
    "kw",
    [{}, {"cluster": "sid"}, {"cluster": "sid", "weights": "popwt"}],
)
def test_did_2x2_default_is_t_and_residual_is_the_old_rule(castle_2x2, kw):
    base, w0 = _dd(castle_2x2, **kw)
    t, _ = _dd(castle_2x2, crit_dist="t", **kw)
    leg, w1 = _dd(castle_2x2, crit_dist="residual", **kw)
    assert w0 == [] and w1 == []
    _same(base, t, table="detail")
    assert leg.model_info["conventions"]["crit_df"] == len(castle_2x2) - 4
    if "cluster" not in kw:
        # without clusters the two rules are the same t(N - K)
        _same(base, leg, table="detail")


def test_did_2x2_t_clustered_matches_stata(castle_2x2):
    res, _ = _dd(castle_2x2, crit_dist="t", cluster="sid")
    ref = pd.read_csv(FIX / "es_crit_Stata.csv", skipinitialspace=True)
    r = ref[(ref.tag == "2x2_castle_cl") & (ref.term == "trpo")].iloc[0]
    assert abs(res.pvalue - r["pvalue"]) < 1e-10
    assert abs(res.ci[0] - r["ll"]) < 1e-10 and abs(res.ci[1] - r["ul"]) < 1e-10
    assert res.model_info["conventions"]["crit_df"] == int(r["df"]) == 41


def test_no_call_warns_about_the_default(castle, castle_2x2):
    """The 1.40.1 FutureWarning is gone with the switch (any FutureWarning)."""
    calls = [
        lambda: sp.event_study(
            castle, y="l_homicide", treat_time="effyear", time="year", unit="sid"
        ),
        lambda: sp.sun_abraham(castle, y="l_homicide", g="gvar", t="year", i="sid"),
        lambda: sp.did_2x2(
            castle_2x2, y="l_homicide", treat="tr", time="po", cluster="sid"
        ),
    ]
    for call in calls:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            call()
        assert [w for w in rec if issubclass(w.category, FutureWarning)] == []


@pytest.mark.parametrize("parity", ["stata", "r"])
def test_no_warning_under_parity(castle, castle_2x2, parity):
    assert _es(castle, parity=parity)[1] == []
    assert _sa(castle, parity=parity)[1] == []
    assert _dd(castle_2x2, parity=parity, cluster="sid")[1] == []


# --------------------------------------------------------------------------
# Contradictions and bad values raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parity", ["stata", "r"])
def test_crit_dist_contradicting_parity_raises(castle, castle_2x2, parity):
    for call in (
        lambda: _es(castle, parity=parity, crit_dist="normal"),
        lambda: _sa(castle, parity=parity, crit_dist="normal"),
        lambda: _dd(castle_2x2, parity=parity, crit_dist="residual"),
    ):
        with pytest.raises(MethodIncompatibility, match="contradicts") as ei:
            call()
        assert "crit_dist" in ei.value.recovery_hint
    # agreeing values are accepted and change nothing
    a, _ = _es(castle, parity=parity, crit_dist="t")
    b, _ = _es(castle, parity=parity)
    _same(a, b)


def test_unknown_crit_dist_raises(castle, castle_2x2):
    with pytest.raises(MethodIncompatibility, match="crit_dist"):
        _es(castle, crit_dist="residual")
    with pytest.raises(MethodIncompatibility, match="crit_dist"):
        _sa(castle, crit_dist="z")
    with pytest.raises(MethodIncompatibility, match="crit_dist"):
        _dd(castle_2x2, crit_dist="normal")


def test_did_2x2_crit_dist_refused_under_bayes(castle_2x2):
    with pytest.raises(MethodIncompatibility, match="crit_dist"):
        _dd(castle_2x2, crit_dist="t", engine="bayes")


@pytest.mark.parametrize("crit_dist", [None, "normal"])
def test_single_cluster_raises_with_a_hint(castle, crit_dist):
    """One cluster: G/(G-1) and t(G-1) are undefined. 1.40.2 crashed with a
    ZeroDivisionError in the SE correction whatever crit_dist was."""
    one = castle.assign(one=1)
    kw = {} if crit_dist is None else {"crit_dist": crit_dist}
    with pytest.raises(MethodIncompatibility, match="two clusters") as ei:
        sp.event_study(
            one,
            y="l_homicide",
            treat_time="effyear",
            time="year",
            unit="sid",
            cluster="one",
            **kw,
        )
    assert "cluster" in ei.value.recovery_hint
    assert ei.value.diagnostics["n_clusters"] == 1


# --------------------------------------------------------------------------
# Library entry points follow the default and do not warn
# --------------------------------------------------------------------------


def test_did_analysis_follows_the_default(castle, castle_2x2):
    out, w = _run(
        sp.did_analysis,
        data=castle_2x2,
        y="l_homicide",
        treat="tr",
        time="po",
        method="2x2",
        cluster="sid",
    )
    assert w == []
    direct, _ = _dd(castle_2x2, cluster="sid")
    assert (out.main_result.pvalue, out.main_result.ci) == (direct.pvalue, direct.ci)
    assert out.main_result.model_info["conventions"]["crit_df"] == 41
    out, w = _run(
        sp.did_analysis,
        data=castle,
        y="l_homicide",
        treat="gvar",
        time="year",
        id="sid",
        method="sa",
    )
    assert w == []
    assert out.main_result.model_info["conventions"]["crit_dist"] == "t"
    assert out.main_result.model_info["summary_aggregation"] == "fixest_att"


def test_compare_event_study_conventions_does_not_warn(castle):
    one = castle[(castle.effyear == 2006) | castle.effyear.isna()]
    _, w = _run(
        sp.compare_event_study_conventions,
        data=one,
        y="l_homicide",
        first_treat="gvar",
        time="year",
        unit="sid",
        window=(-3, 3),
        estimators=["event_study", "sun_abraham"],
    )
    assert w == []
