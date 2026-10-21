"""``parity=`` on the regression event studies: Stata and fixest critical values.

``sp.event_study``, ``sp.sun_abraham`` and ``sp.did_2x2`` reproduce the
standard errors of Stata (``reghdfe`` / ``eventstudyinteract`` /
``regress``) and R ``fixest``, but turn them into p-values and intervals
with their own distribution: normal for the two event studies, t(N - K)
for ``did_2x2``. Both reference programs use t(G - 1) with
cluster-robust SEs and t(N - K) otherwise. ``parity='stata'`` /
``parity='r'`` switch to that convention.

References, run on the same bytes (the bundled castle-doctrine CSV and
``tests/fixtures/divorce_no_fault_panel.csv``):

* Stata 18 MP: ``_fixtures/_generate_es_crit_Stata.do`` ->
  ``_fixtures/es_crit_Stata.csv`` (``r(table)`` of each fit, and
  ``lincom`` for the headline average).
* R fixest 0.14.0 (GPL-3, black box): ``_fixtures/_generate_es_crit_R.R``
  -> ``_fixtures/es_crit_R.csv`` (``coeftable`` / ``confint``).

Tolerance: 1e-10 absolute on p-values and confidence limits and 1e-10
relative on SEs. The point estimates and SEs agree to about 1e-13 (same
algebra, different summation order); a p-value or limit is a smooth
function of them, so 1e-10 leaves three orders of margin while still
failing on any change of the distribution or its degrees of freedom (the
smallest such change here, t(49) against the normal, moves the limits by
2e-3).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).resolve().parent / "_fixtures"
ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-10

ES_TERMS = {
    k: (f"es_m{-k}" if k < 0 else f"es_p{k}") for k in [-5, -4, -3, -2, 0, 1, 2, 3, 4]
}
SA_STATA = {k: (f"L{-k}" if k < 0 else f"F{k}") for k in [*range(-9, -1), *range(0, 6)]}
SA_R = {k: str(k) for k in SA_STATA}
DIV_TERMS = {
    k: (f"lead{-k}" if k < 0 else f"lag{k}") for k in [*range(-10, -1), *range(0, 16)]
}


def _ref(program: str) -> pd.DataFrame:
    path = FIX / f"es_crit_{program}.csv"
    return pd.read_csv(path, skipinitialspace=True, dtype={"term": str})


@pytest.fixture(scope="module")
def castle() -> pd.DataFrame:
    d = sp.datasets.castle_doctrine()
    d = d.assign(
        region=(1 * d.northeast + 2 * d.midwest + 3 * d.south + 4 * d.west).astype(int),
        gvar=d.effyear.fillna(0).astype(int),
    )
    return d


@pytest.fixture(scope="module")
def divorce() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "tests" / "fixtures" / "divorce_no_fault_panel.csv")
    first = b[b.post == 1].groupby("stfips").year.min()
    b["cohort"] = b.stfips.map(first).fillna(0).astype(int)
    b = b[b.cohort != 1964].copy()
    b["tt"] = np.where(b.cohort > 0, b.cohort, np.nan)
    return b


def _check(table: pd.DataFrame, ref: pd.DataFrame, terms: dict) -> None:
    ref = ref.set_index("term")
    assert len(terms) > 0
    for k, term in terms.items():
        row, r = table.loc[k], ref.loc[term]
        assert row["se"] == pytest.approx(r["se"], rel=TOL), (k, "se")
        assert abs(row["pvalue"] - r["pvalue"]) < TOL, (k, "pvalue")
        assert abs(row["ci_lower"] - r["ll"]) < TOL, (k, "ll")
        assert abs(row["ci_upper"] - r["ul"]) < TOL, (k, "ul")


def _es(data, parity, cluster="sid", alpha=0.05):
    return sp.event_study(
        data,
        y="l_homicide",
        treat_time="effyear",
        time="year",
        unit="sid",
        window=(-5, 4),
        ref_period=-1,
        weights="popwt",
        cluster=cluster,
        alpha=alpha,
        parity=parity,
    )


@pytest.mark.parametrize("program,parity", [("Stata", "stata"), ("R", "r")])
@pytest.mark.parametrize(
    "tag,cluster,alpha",
    [
        ("es_castle_cl", "sid", 0.05),
        ("es_castle_cl90", "sid", 0.10),
        ("es_castle_region", "region", 0.05),
    ],
)
def test_event_study_matches_reference(castle, program, parity, tag, cluster, alpha):
    res = _es(castle, parity, cluster, alpha)
    ref = _ref(program)
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        ref[ref.tag == tag],
        ES_TERMS,
    )
    g = res.model_info["n_clusters"]
    assert res.model_info["conventions"] == {
        "parity": parity,
        "crit_dist": "t",
        "crit_df": g - 1,
    }
    assert int(ref[ref.tag == tag].df.iloc[0]) == g - 1


def test_event_study_headline_equals_lincom(castle):
    res = _es(castle, "stata")
    ref = _ref("Stata").set_index("tag").loc["es_castle_cl_att"]
    assert res.estimate == pytest.approx(ref["b"], rel=1e-12)
    assert res.se == pytest.approx(ref["se"], rel=TOL)
    assert abs(res.pvalue - ref["pvalue"]) < TOL
    assert abs(res.ci[0] - ref["ll"]) < TOL and abs(res.ci[1] - ref["ul"]) < TOL


@pytest.mark.parametrize("program,parity", [("Stata", "stata"), ("R", "r")])
def test_event_study_divorce(divorce, program, parity):
    res = sp.event_study(
        divorce,
        y="asmrs",
        treat_time="tt",
        time="year",
        unit="stfips",
        window=(-10, 15),
        ref_period=-1,
        cluster="stfips",
        parity=parity,
    )
    ref = _ref(program)
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        ref[ref.tag == "es_divorce_cl"],
        DIV_TERMS,
    )
    assert res.model_info["conventions"]["crit_df"] == 40


def _sa(data, parity, panel=True, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sun_abraham(
            data,
            y="l_homicide",
            g="gvar",
            t="year",
            i="sid" if panel else None,
            weights="popwt",
            control_group="nevertreated",
            aggregation="event_time",
            parity=parity,
            **kw,
        )


@pytest.mark.parametrize(
    "panel,tag", [(True, "sa_castle_cl"), (False, "sa_castle_rob")]
)
def test_sun_abraham_stata(castle, panel, tag):
    """eventstudyinteract: share term carried, t(G-1) clustered, t(N-K) robust."""
    res = _sa(castle, "stata", panel)
    ref = _ref("Stata")
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        ref[ref.tag == tag],
        SA_STATA,
    )
    conv = res.model_info["conventions"]
    assert conv["share_variance"] is True
    df_ref = int(ref[ref.tag == tag].df.iloc[0])
    assert conv["crit_df"] == df_ref
    expected = (
        res.model_info["n_clusters"] - 1
        if panel
        else res.n_obs - res.model_info["dof_K"]
    )
    assert df_ref == expected


@pytest.mark.parametrize(
    "panel,tag", [(True, "sa_castle_cl"), (False, "sa_castle_rob")]
)
def test_sun_abraham_r(castle, panel, tag):
    """fixest::sunab: fixed cohort shares and the same t critical values."""
    res = _sa(castle, "r", panel)
    ref = _ref("R")
    _check(
        res.model_info["event_study"].set_index("relative_time"),
        ref[ref.tag == tag],
        SA_R,
    )
    assert res.model_info["conventions"]["share_variance"] is False
    assert res.model_info["share_variance"] is False


def test_sun_abraham_explicit_share_variance_conflict(castle):
    with pytest.raises(MethodIncompatibility, match="contradicts"):
        _sa(castle, "r", share_variance=True)
    with pytest.raises(MethodIncompatibility, match="contradicts"):
        _sa(castle, "stata", share_variance=False)
    # agreeing explicit values are accepted and change nothing
    a = _sa(castle, "r", share_variance=False).model_info["event_study"]
    b = _sa(castle, "r").model_info["event_study"]
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture(scope="module")
def castle_2x2(castle) -> pd.DataFrame:
    d2 = castle[(castle.effyear == 2006) | castle.effyear.isna()].copy()
    d2["tr"] = d2.effyear.notna().astype(int)
    d2["po"] = (d2.year >= 2006).astype(int)
    return d2


@pytest.mark.parametrize(
    "program,parity,term", [("Stata", "stata", "trpo"), ("R", "r", "tr:po")]
)
@pytest.mark.parametrize(
    "tag,kw",
    [
        ("2x2_castle_cl", {"cluster": "sid"}),
        ("2x2_castle_rob", {}),
        ("2x2_castle_clw", {"cluster": "sid", "weights": "popwt"}),
    ],
)
def test_did_2x2_matches_reference(castle_2x2, program, parity, term, tag, kw):
    res = sp.did_2x2(
        castle_2x2, y="l_homicide", treat="tr", time="po", parity=parity, **kw
    )
    ref = _ref(program)
    r = ref[(ref.tag == tag) & (ref.term == term)].iloc[0]
    assert res.se == pytest.approx(r["se"], rel=TOL)
    assert abs(res.pvalue - r["pvalue"]) < TOL
    assert abs(res.ci[0] - r["ll"]) < TOL and abs(res.ci[1] - r["ul"]) < TOL
    assert res.model_info["conventions"]["crit_df"] == int(r["df"])


def test_did_2x2_without_cluster_parity_is_the_default(castle_2x2):
    """Robust SEs: Stata and fixest use t(N - K), which is the default."""
    a = sp.did_2x2(castle_2x2, y="l_homicide", treat="tr", time="po")
    b = sp.did_2x2(castle_2x2, y="l_homicide", treat="tr", time="po", parity="stata")
    assert (a.pvalue, a.ci) == (b.pvalue, b.ci)
    assert a.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "t",
        "crit_df": len(castle_2x2) - 4,
    }


# --------------------------------------------------------------------------
# Identities: the critical value is the stated quantile, nothing else moves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parity", [None, "stata", "r"])
def test_interval_half_width_is_the_stated_quantile(castle, parity):
    alpha = 0.10
    res = _es(castle, parity, alpha=alpha)
    conv = res.model_info["conventions"]
    q = (
        stats.norm.ppf(1 - alpha / 2)
        if conv["crit_dist"] == "normal"
        else stats.t.ppf(1 - alpha / 2, conv["crit_df"])
    )
    es = res.model_info["event_study"]
    es = es[~es.is_reference]
    np.testing.assert_allclose((es.ci_upper - es.att) / es.se, q, rtol=1e-12)
    z = (es.att / es.se).abs()
    p = 2 * (
        stats.norm.sf(z)
        if conv["crit_dist"] == "normal"
        else stats.t.sf(z, conv["crit_df"])
    )
    np.testing.assert_allclose(es.pvalue, p, rtol=1e-12)
    assert (res.ci[1] - res.estimate) / res.se == pytest.approx(q, rel=1e-12)


def test_parity_moves_only_the_inference(castle):
    base, st = _es(castle, None), _es(castle, "stata")
    a, b = base.model_info["event_study"], st.model_info["event_study"]
    pd.testing.assert_series_equal(a.att, b.att)
    pd.testing.assert_series_equal(a.se, b.se)
    assert (base.estimate, base.se) == (st.estimate, st.se)
    assert base.model_info["pretrend_test"] == st.model_info["pretrend_test"]
    assert base.model_info["conventions"] == {
        "parity": None,
        "crit_dist": "normal",
        "crit_df": None,
    }
    sa0, sa1 = _sa(castle, None), _sa(castle, "stata")
    pd.testing.assert_series_equal(
        sa0.model_info["event_study"].se, sa1.model_info["event_study"].se
    )


@pytest.mark.parametrize("fn", ["event_study", "sun_abraham", "did_2x2"])
def test_unknown_parity_raises(castle, castle_2x2, fn):
    with pytest.raises(MethodIncompatibility, match="parity"):
        if fn == "event_study":
            _es(castle, "sas")
        elif fn == "sun_abraham":
            _sa(castle, "Python")
        else:
            sp.did_2x2(castle_2x2, y="l_homicide", treat="tr", time="po", parity="spss")
