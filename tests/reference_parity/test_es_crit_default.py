"""``crit_dist=`` on the regression event studies and the announced 1.41 default.

Owner decision 2026-10-21: ``sp.event_study``, ``sp.sun_abraham`` and
``sp.did_2x2`` move to the Stata / fixest critical values (t(G - 1) with
cluster-robust SEs, t(N - K) otherwise) in 1.41. This release adds
``crit_dist=`` and warns where the default will change. Pinned here:

* ``crit_dist='t'`` gives exactly the numbers of ``parity='stata'`` (which
  ``test_es_crit_parity.py`` holds to the Stata and fixest references), so
  it inherits that evidence; one row is re-checked against the Stata
  fixture directly.
* the legacy values (``'normal'``, ``'residual'``) give exactly today's
  default numbers, with no warning;
* the ``FutureWarning`` fires once per call exactly where the intervals
  would change, and is attributed to the caller's line;
* contradicting ``parity=`` raises a ``MethodIncompatibility`` with a
  recovery hint;
* library entry points that call these functions on the user's behalf
  pin the legacy value and do not warn.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"


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


# --------------------------------------------------------------------------
# crit_dist='t' is parity='stata'; the legacy value is today's default
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


def test_event_study_normal_is_todays_default(castle):
    base, w0 = _es(castle)
    leg, w1 = _es(castle, crit_dist="normal")
    assert len(w0) == 1 and w1 == []
    _same(base, leg)
    assert leg.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "normal",
        "crit_df": None,
    }


@pytest.mark.parametrize("panel", [True, False])
def test_sun_abraham_t_equals_parity_stata(castle, panel):
    t, w = _sa(castle, panel, crit_dist="t")
    st, _ = _sa(castle, panel, parity="stata")
    assert w == []
    _same(t, st)
    # crit_dist touches the critical values only, never the share term
    assert t.model_info["share_variance"] is True


@pytest.mark.parametrize("panel", [True, False])
def test_sun_abraham_normal_is_todays_default(castle, panel):
    base, w0 = _sa(castle, panel)
    leg, w1 = _sa(castle, panel, crit_dist="normal")
    assert len(w0) == 1 and w1 == []
    _same(base, leg)


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
def test_did_2x2_residual_is_todays_default(castle_2x2, kw):
    base, _ = _dd(castle_2x2, **kw)
    leg, w = _dd(castle_2x2, crit_dist="residual", **kw)
    assert w == []
    _same(base, leg, table="detail")
    assert leg.model_info["conventions"]["crit_df"] == len(castle_2x2) - 4


def test_did_2x2_t_clustered_matches_stata(castle_2x2):
    res, _ = _dd(castle_2x2, crit_dist="t", cluster="sid")
    ref = pd.read_csv(FIX / "es_crit_Stata.csv", skipinitialspace=True)
    r = ref[(ref.tag == "2x2_castle_cl") & (ref.term == "trpo")].iloc[0]
    assert abs(res.pvalue - r["pvalue"]) < 1e-10
    assert abs(res.ci[0] - r["ll"]) < 1e-10 and abs(res.ci[1] - r["ul"]) < 1e-10
    assert res.model_info["conventions"]["crit_df"] == int(r["df"]) == 41


# --------------------------------------------------------------------------
# The FutureWarning: exactly where the numbers would change
# --------------------------------------------------------------------------


def test_warning_text_and_location(castle):
    _, w = _es(castle)
    assert len(w) == 1
    msg = str(w[0].message)
    assert "1.41" in msg and "t(49)" in msg and "normal" in msg
    assert "crit_dist='t'" in msg and "crit_dist='normal'" in msg
    assert "fixest" in msg and "reghdfe" in msg
    # attributed to the user's call, not to a statspai frame
    assert "statspai" not in Path(w[0].filename).parts[-3:-1]


def test_did_2x2_warns_only_when_clustered(castle_2x2):
    _, w = _dd(castle_2x2)
    assert w == []  # t(N-K) is already the Stata / fixest value
    _, w = _dd(castle_2x2, cluster="sid")
    assert len(w) == 1
    assert "t(458)" in str(w[0].message) and "t(41)" in str(w[0].message)
    assert "crit_dist='residual'" in str(w[0].message)


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


# --------------------------------------------------------------------------
# Library entry points pin the legacy value: no warning the user cannot act on
# --------------------------------------------------------------------------


def test_did_analysis_does_not_warn(castle, castle_2x2):
    _, w = _run(
        sp.did_analysis,
        data=castle_2x2,
        y="l_homicide",
        treat="tr",
        time="po",
        method="2x2",
        cluster="sid",
    )
    assert w == []
    _, w = _run(
        sp.did_analysis,
        data=castle,
        y="l_homicide",
        treat="gvar",
        time="year",
        id="sid",
        method="sa",
    )
    assert w == []


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
