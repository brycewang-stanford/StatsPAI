"""Post-estimation surface of ``sp.etwfe`` found missing by the sjjj-2026
replication: event-study plots, regression-table FE / cluster rows,
``result.pretrend_test``, a uniform result API (``nobs`` / ``n_obs`` /
``vcov()`` / ``conf_int()``), the git revision in provenance, and Stata
factor terms in ``controls`` / ``xvar``."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.did._factor_terms import expand_factor_terms  # noqa: E402
from statspai.exceptions import MethodIncompatibility  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    n, T = 300, 10
    d = pd.DataFrame(
        {"id": np.repeat(np.arange(n), T), "t": np.tile(np.arange(2011, 2011 + T), n)}
    )
    d["g"] = np.repeat(rng.choice([0, 2015, 2018], n), T)
    d["region"] = np.repeat(rng.integers(1, 4, n), T)
    d["node"] = np.repeat(rng.integers(0, 2, n), T)
    d["x"] = rng.normal(size=len(d))
    treated = (d["g"] > 0) & (d["t"] >= d["g"])
    d["y"] = rng.poisson(np.exp(0.3 * treated + 0.05 * d["region"] + 0.1 * d["x"]))
    d["D"] = treated.astype(float)
    return d


def _etwfe(d, **kw):
    kw.setdefault("family", "poisson")
    kw.setdefault("fe", "unit")
    kw.setdefault("scale", "link")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.etwfe(d, y="y", group="id", time="t", first_treat="g", **kw)


@pytest.fixture(scope="module")
def never(panel):
    return _etwfe(panel, cgroup="nevertreated")


# --- 1. event-study plots ---------------------------------------------------


def test_etwfe_event_study_plots_draw(never):
    es = never.model_info["event_study"]
    assert "ci_lower" not in es.columns  # the historical table, no interval
    fig, ax = never.plot()
    band = ax.collections[0].get_paths()[0].vertices
    z = stats.norm.ppf(0.975)
    lo = es["att"] - z * es["se"]
    assert np.isclose(band[:, 1].min(), lo.min())
    for f in (never.event_study_plot, lambda: sp.enhanced_event_study_plot(never)):
        fig, _ = f()
        assert type(fig).__name__ == "Figure"
    ev = sp.etwfe_emfx(never, type="event", scale="link", include_leads=True)
    fig, _ = sp.enhanced_event_study_plot(ev)
    plt.close("all")


def test_existing_interval_is_kept():
    from statspai.core.results import _event_study_with_ci

    es = pd.DataFrame(
        {
            "relative_time": [0],
            "att": [1.0],
            "se": [1.0],
            "ci_lower": [-9.0],
            "ci_upper": [9.0],
        }
    )
    assert _event_study_with_ci(es, 0.05) is es
    with pytest.raises(MethodIncompatibility):
        _event_study_with_ci(pd.DataFrame({"relative_time": [0], "att": [1.0]}), 0.05)


# --- 2. regression-table rows ----------------------------------------------


def _rows(results):
    from statspai.output._diagnostics import extract_fe_cluster_indicators

    return dict(extract_fe_cluster_indicators(results))


def test_regtable_reports_etwfe_fe_and_cluster(panel):
    unit = _etwfe(panel)
    cohort = _etwfe(panel, fe="cohort")
    ppml = sp.ppmlhdfe(data=panel, y="y", x=["D"], absorb="id + t", cluster="id")
    rows = _rows([unit, cohort, ppml])
    assert rows["Id FE"] == ["Yes", "No", "Yes"]
    assert rows["G FE"] == ["No", "Yes", "No"]
    assert rows["T FE"] == ["Yes", "Yes", "Yes"]
    assert rows["Cluster SE"] == ["id", "id", "id"]
    assert unit.estimand == "ATT (link scale)"
    assert "log-point" in unit.model_info["estimand_description"]
    text = str(sp.regtable([unit, ppml]))
    assert "ATT (link scale)" in text


def test_causal_result_without_fe_metadata_is_blank(panel):
    ppml = sp.ppmlhdfe(data=panel, y="y", x=["D"], absorb="id + t", cluster="id")
    cr = sp.CausalResult(
        method="some did",
        estimand="ATT",
        estimate=0.1,
        se=0.05,
        pvalue=0.04,
        ci=(0.0, 0.2),
        alpha=0.05,
        n_obs=10,
    )
    rows = _rows([ppml, cr])
    assert rows["Id FE"] == ["Yes", ""]
    assert rows["Cluster SE"] == ["id", ""]


# --- 3. pretrend_test --------------------------------------------------------


def test_pretrend_test_method_equals_function(never):
    a = never.pretrend_test()
    b = sp.pretrends_test(never)
    assert a["statistic"] == pytest.approx(b["statistic"], rel=1e-12)
    w = never.pretrend_test(window=(-3, -2))
    assert list(w["pre_periods"]) == [-3, -2]
    assert w["statistic"] == pytest.approx(
        sp.pretrends_test(never, window=(-3, -2))["statistic"], rel=1e-12
    )


def test_pretrend_test_stored_result_unchanged():
    df = sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=1)
    es = sp.event_study(df, y="y", treat_time="first_treat", time="time", unit="unit")
    assert es.pretrend_test() is es.model_info["pretrend_test"]
    cr = sp.CausalResult(
        method="x",
        estimand="ATE",
        estimate=1.0,
        se=1.0,
        pvalue=0.3,
        ci=(-1.0, 3.0),
        alpha=0.05,
        n_obs=5,
    )
    with pytest.raises(ValueError, match="no event-study"):
        cr.pretrend_test()


# --- 4. uniform result API ---------------------------------------------------


def test_nobs_vcov_conf_int_on_both_result_classes(panel):
    r = _etwfe(panel)
    ppml = sp.ppmlhdfe(data=panel, y="y", x=["D", "x"], absorb="id + t", cluster="id")
    assert r.nobs == r.n_obs == len(panel)
    assert ppml.n_obs == ppml.nobs == len(panel)
    V = r.vcov()
    assert list(V.index) == [r.estimand]
    assert V.iloc[0, 0] == pytest.approx(r.se**2, rel=1e-12)
    Vp = ppml.vcov()
    assert list(Vp.index) == ["D", "x"]
    np.testing.assert_allclose(np.sqrt(np.diag(Vp)), ppml.std_errors, rtol=1e-10)
    ci = r.conf_int()
    assert tuple(ci.iloc[0]) == pytest.approx(r.ci, rel=1e-12)
    ci90 = r.conf_int(0.10)
    z = stats.norm.ppf(0.95)
    assert ci90.iloc[0, 0] == pytest.approx(r.estimate - z * r.se, rel=1e-12)
    assert list(ci90.columns) == ["0.050", "0.950"]


def test_n_obs_absent_when_nobs_unrecorded():
    res = sp.EconometricResults(
        params=pd.Series({"a": 1.0, "b": 2.0}),
        std_errors=pd.Series({"a": 0.1, "b": 0.2}),
        model_info={},
        data_info={},
    )
    assert not hasattr(res, "n_obs")
    res.n_obs = 12
    assert res.nobs == 12
    with pytest.raises(MethodIncompatibility, match="covariance"):
        res.vcov()


# --- 5. version / provenance -------------------------------------------------


def test_version_info_and_provenance_revision(panel, monkeypatch):
    from statspai import _build_info

    info = sp.version_info()
    assert info["version"] == sp.__version__
    assert info["consistent"] == ("warning" not in info)
    prov = sp.get_provenance(_etwfe(panel))
    assert prov.statspai_revision == _build_info.source_revision()
    if prov.statspai_revision:
        assert f"(git {prov.statspai_revision})" in sp.format_provenance(prov)

    _build_info.source_revision.cache_clear()
    monkeypatch.setattr(_build_info, "_checkout_root", lambda start, max_up=4: None)
    try:
        assert _build_info.source_revision() is None  # installed wheel
    finally:
        _build_info.source_revision.cache_clear()
    monkeypatch.setattr(_build_info, "_metadata_version", lambda: "0.0.1")
    bad = _build_info.version_info()
    assert bad["consistent"] is False and "stale" in bad["warning"]


# --- 6. Stata factor terms ---------------------------------------------------


def test_factor_expansion_rules():
    df = pd.DataFrame(
        {
            "f": [1, 2, 3, 1, np.nan],
            "h": [0, 1, 1, 0, 1],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    _, names = expand_factor_terms(df, ["i.f"])
    assert names == ["2.f", "3.f"]
    out, names = expand_factor_terms(df, ["c.c#i.h"])
    assert names == ["c.c#0.h", "c.c#1.h"]  # every level: no main effect
    np.testing.assert_allclose(out["c.c#1.h"], df["c"] * (df["h"] == 1))
    out, names = expand_factor_terms(df, ["i.f##i.h"])
    assert names == ["2.f", "3.f", "1.h", "2.f#1.h", "3.f#1.h"]
    assert np.isnan(out["2.f#1.h"].iloc[4])  # missing factor -> missing
    same, names = expand_factor_terms(df, ["c", "h"])
    assert same is df and names == ["c", "h"]
    with pytest.raises(MethodIncompatibility, match="not a column"):
        expand_factor_terms(df, ["i.nope"])
    clash = df.assign(**{"2.f": 0.0})
    with pytest.raises(MethodIncompatibility, match="clashes"):
        expand_factor_terms(clash, ["i.f"])


def test_etwfe_factor_terms_equal_hand_built_columns(panel):
    d = panel.copy()
    for y in range(2011, 2021):
        for v in (0, 1):
            d[f"c_{y}_{v}"] = ((d["t"] == y) & (d["node"] == v)).astype(float)
    hand = _etwfe(d, controls=[c for c in d if c.startswith("c_")])
    auto = _etwfe(panel, controls=["i.t#i.node"])
    assert auto.estimate == pytest.approx(hand.estimate, rel=1e-10)
    assert auto.se == pytest.approx(hand.se, rel=1e-8)

    cat = _etwfe(panel.assign(region=panel["region"].astype("category")), xvar="region")
    fac = _etwfe(panel, xvar="i.region")
    assert fac.estimate == pytest.approx(cat.estimate, rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="interactions"):
        _etwfe(panel, xvar="i.region#i.node")


def test_jwdid_linear_equals_etwfe_unit(panel):
    lin = sp.jwdid(panel, "y", ivar="id", tvar="t", gvar="g")
    ref = sp.etwfe(panel, y="y", group="id", time="t", first_treat="g", fe="unit")
    assert lin.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert lin.model_info["stata_equivalent"] == "jwdid y, ivar(id) tvar(t) gvar(g)"
