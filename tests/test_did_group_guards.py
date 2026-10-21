"""``group`` is the unit id across the DiD family; ``sp.drdid`` takes ``treat``.

Owner decision 2026-10-21 (follow-up to the replication report, item 5):

* the swap guard of ``sp.did_imputation`` (``did/_arg_roles.py``) is
  extended to ``gardner_did``, ``stacked_did``, ``etwfe``,
  ``wooldridge_did`` and ``did_multiplegt``. A treatment dummy passed as
  the unit id fails loudly and the hint names ``id=``; a correct call
  returns exactly what it returned without the guard (checked by running
  it again with the guard switched off);
* ``sp.drdid``'s parameter is ``treat``; the keyword ``group=`` still works
  with a ``DeprecationWarning`` and the same result, positional calls are
  silent;
* ``sp.did(parity=)`` and ``sp.cs_report(parity=)`` forward to the
  estimators; the forwarded call equals the direct one exactly.
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._arg_roles import ArgumentRoleWarning

# module that holds each estimator's reference to the guard
GUARD_MODULES = {
    "gardner_did": "statspai.did.gardner_2s",
    "stacked_did": "statspai.did.stacked_did",
    "etwfe": "statspai.did.wooldridge_did",
    "wooldridge_did": "statspai.did.wooldridge_did",
    "did_multiplegt": "statspai.did.did_multiplegt",
}
COHORT_ESTIMATORS = ["gardner_did", "stacked_did", "etwfe", "wooldridge_did"]


@pytest.fixture(scope="module")
def staggered() -> pd.DataFrame:
    df = sp.dgp_did(n_units=80, n_periods=7, staggered=True, seed=5)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    df["ever"] = (df["g"] > 0).astype(int)
    df["d"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)
    return df


@pytest.fixture(scope="module")
def single_cohort(staggered: pd.DataFrame) -> pd.DataFrame:
    df = staggered.copy()
    df.loc[df["g"] > 0, "g"] = 4
    df["ever"] = (df["g"] > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def two_by_two(staggered: pd.DataFrame) -> pd.DataFrame:
    d = staggered[staggered["time"].isin([2, 3]) & staggered["g"].isin([0, 3])].copy()
    d["treated"] = (d["g"] > 0).astype(int)
    d["post"] = (d["time"] == 3).astype(int)
    return d


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _same(a, b) -> None:
    assert a.estimate == b.estimate
    assert a.se == b.se
    if a.detail is None:
        assert b.detail is None
    else:
        pd.testing.assert_frame_equal(a.detail, b.detail, check_exact=True)


def _without_guard(monkeypatch, name):
    mod = importlib.import_module(GUARD_MODULES[name])
    monkeypatch.setattr(mod, "_check_unit_is_not_treatment", lambda *a, **k: None)


# ------------------------------------------------------------ swap guards
@pytest.mark.parametrize("name", COHORT_ESTIMATORS)
def test_treatment_dummy_as_unit_id_is_refused(staggered, name):
    with pytest.raises(sp.MethodIncompatibility, match="treatment indicator") as e:
        _quiet(
            getattr(sp, name),
            staggered,
            y="y",
            group="ever",
            time="time",
            first_treat="g",
        )
    assert "id=" in e.value.recovery_hint
    assert e.value.diagnostics["unit"] == "ever"


@pytest.mark.parametrize("name", COHORT_ESTIMATORS)
def test_correct_call_is_unchanged_by_the_guard(staggered, monkeypatch, name):
    fn = getattr(sp, name)
    kw = dict(y="y", group="unit", time="time", first_treat="g")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        guarded = fn(staggered, **kw)
    assert not [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]
    _without_guard(monkeypatch, name)
    _same(guarded, _quiet(fn, staggered, **kw))


@pytest.mark.parametrize("name", COHORT_ESTIMATORS)
def test_two_group_layout_is_flagged_not_refused(single_cohort, monkeypatch, name):
    # One cohort, so a 0/1 "unit" column is also a two-group repeated
    # cross-section with group fixed effects: it runs, warns, and returns
    # what it returned before.
    fn = getattr(sp, name)
    kw = dict(y="y", group="ever", time="time", first_treat="g")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        flagged = fn(single_cohort, **kw)
    roles = [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]
    assert roles and "id=" in roles[0].message.recovery_hint
    _without_guard(monkeypatch, name)
    _same(flagged, _quiet(fn, single_cohort, **kw))


def test_etwfe_repeated_cross_section_is_not_checked(single_cohort):
    # panel=False: group is not a unit fixed effect there; no role warning.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sp.etwfe(
            single_cohort,
            y="y",
            group="ever",
            time="time",
            first_treat="g",
            panel=False,
        )
    assert not [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]


def test_did_multiplegt_refuses_a_treatment_dummy_as_group(staggered):
    # 0/1 group, many rows per period and a staggered treatment that varies
    # within each (group, period) cell: the period-to-period merge on the
    # group would compare rows of different units.
    with pytest.raises(sp.MethodIncompatibility, match="varies within") as e:
        _quiet(
            sp.did_multiplegt,
            staggered,
            y="y",
            group="ever",
            time="time",
            treatment="d",
            n_boot=5,
        )
    assert "id=" in e.value.recovery_hint
    assert "treatment='d'" in e.value.recovery_hint


def test_did_multiplegt_two_group_layout_is_flagged_not_refused(
    single_cohort, monkeypatch
):
    # One adoption date: the treatment is constant within each (group,
    # period) cell, so this is also a two-group design. It runs and warns.
    df = single_cohort.assign(
        d=((single_cohort["g"] > 0) & (single_cohort["time"] >= 4)).astype(int)
    )
    kw = dict(y="y", group="ever", time="time", treatment="d", n_boot=5, seed=0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        flagged = sp.did_multiplegt(df, **kw)
    assert [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]
    _without_guard(monkeypatch, "did_multiplegt")
    _same(flagged, _quiet(sp.did_multiplegt, df, **kw))


def test_did_multiplegt_correct_call_is_unchanged(staggered, monkeypatch):
    kw = dict(y="y", group="unit", time="time", treatment="d", n_boot=20, seed=3)
    guarded = _quiet(sp.did_multiplegt, staggered, **kw)
    _without_guard(monkeypatch, "did_multiplegt")
    _same(guarded, _quiet(sp.did_multiplegt, staggered, **kw))


def test_did_multiplegt_two_groups_one_row_each_runs(staggered):
    # A genuine two-group panel coded 0/1, one row per (group, period).
    cells = (
        staggered.groupby(["ever", "time"], as_index=False)
        .agg(y=("y", "mean"))
        .assign(d=lambda f: ((f["ever"] == 1) & (f["time"] >= 3)).astype(int))
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = sp.did_multiplegt(
            cells, y="y", group="ever", time="time", treatment="d", n_boot=5, seed=0
        )
    assert not [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]
    # Identity: one switch-on period with one stayer group is the 2x2 DiD of
    # the cell means between periods 2 and 3.
    w = cells.pivot(index="time", columns="ever", values="y")
    assert r.estimate == pytest.approx(
        (w.loc[3, 1] - w.loc[2, 1]) - (w.loc[3, 0] - w.loc[2, 0]), abs=1e-12
    )


# ------------------------------------------------- drdid group= -> treat=
def test_drdid_group_keyword_is_deprecated_same_result(two_by_two):
    with pytest.warns(DeprecationWarning, match=r"group=.*1\.42.*treat=") as rec:
        old = sp.drdid(two_by_two, y="y", group="treated", time="post")
    # The warning points at this file, not into the library.
    assert any(w.filename == __file__ for w in rec)
    new = sp.drdid(two_by_two, y="y", treat="treated", time="post")
    _same(old, new)


def test_drdid_positional_and_treat_calls_are_silent(two_by_two):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pos = sp.drdid(two_by_two, "y", "treated", "post")
        kw = sp.drdid(two_by_two, y="y", treat="treated", time="post")
        r_name = sp.drdid(two_by_two, yname="y", dname="treated", tname="post")
    _same(pos, kw)
    _same(pos, r_name)


def test_drdid_group_and_treat_together_is_refused(two_by_two):
    with pytest.raises(TypeError, match="only 'treat'"):
        sp.drdid(two_by_two, y="y", group="treated", treat="treated", time="post")


def test_drdid_schema_names_treat_not_group():
    assert list(inspect.signature(sp.drdid).parameters)[:4] == [
        "data",
        "y",
        "treat",
        "time",
    ]
    card = sp.describe_function("drdid")
    names = [p["name"] for p in card["params"]]
    assert "treat" in names and "group" not in names
    schema = sp.function_schema("drdid")
    props = schema.get("parameters", schema).get("properties", {})
    assert "treat" in props and "group" not in props
    # still accepted (tool layers check accepted names), not advertised
    assert "group" not in (card.get("aliases") or {})
    assert sp.drdid.__statspai_aliases__["group"] == "treat"


# ------------------------------------------------------- parity forwarding
def test_did_parity_cs_equals_direct_call(staggered):
    for parity in ("r", "stata"):
        a = _quiet(
            sp.did, staggered, y="y", treat="g", time="time", id="unit", parity=parity
        )
        b = _quiet(
            sp.callaway_santanna,
            staggered,
            y="y",
            g="g",
            t="time",
            i="unit",
            parity=parity,
        )
        _same(a, b)
        assert a.model_info["conventions"] == b.model_info["conventions"]


@pytest.mark.parametrize("parity", ["r", "stata"])
def test_did_parity_cs_aggregation_equals_direct_call(staggered, parity):
    a = _quiet(
        sp.did,
        staggered,
        y="y",
        treat="g",
        time="time",
        id="unit",
        parity=parity,
        aggregation="dynamic",
        random_state=11,
    )
    fit = _quiet(
        sp.callaway_santanna, staggered, y="y", g="g", t="time", i="unit", parity=parity
    )
    if parity == "r":
        # sp.did's own bootstrap request reaches aggte (R did bootstraps).
        b = _quiet(sp.aggte, fit, type="dynamic", n_boot=1000, random_state=11)
    else:
        # csdid's estat is analytic; random_state was passed and is kept.
        b = _quiet(sp.aggte, fit, type="dynamic", random_state=11)
    _same(a, b)


def test_did_parity_stata_refuses_an_explicit_bootstrap(staggered):
    with pytest.raises(sp.MethodIncompatibility, match="n_boot"):
        _quiet(
            sp.did,
            staggered,
            y="y",
            treat="g",
            time="time",
            id="unit",
            parity="stata",
            aggregation="simple",
            n_boot=499,
        )


def test_did_parity_sun_abraham_equals_direct_call(staggered):
    a = _quiet(
        sp.did,
        staggered,
        y="y",
        treat="g",
        time="time",
        id="unit",
        method="sun_abraham",
        parity="stata",
    )
    b = _quiet(
        sp.sun_abraham, staggered, y="y", g="g", t="time", i="unit", parity="stata"
    )
    _same(a, b)


def test_did_parity_2x2_equals_direct_call(two_by_two):
    a = _quiet(
        sp.did,
        two_by_two,
        y="y",
        treat="treated",
        time="post",
        method="2x2",
        cluster="unit",
        parity="stata",
    )
    b = _quiet(
        sp.did_2x2,
        two_by_two,
        y="y",
        treat="treated",
        time="post",
        cluster="unit",
        parity="stata",
    )
    _same(a, b)
    off = _quiet(
        sp.did,
        two_by_two,
        y="y",
        treat="treated",
        time="post",
        method="2x2",
        cluster="unit",
    )
    # t(G-1) under parity versus the normal default: same point, wider CI.
    assert a.estimate == off.estimate
    assert a.ci[1] - a.ci[0] > off.ci[1] - off.ci[0]


@pytest.mark.parametrize("method", ["bjs", "ddd", "sdid"])
def test_did_parity_refused_where_not_implemented(staggered, method):
    with pytest.raises(sp.MethodIncompatibility, match="parity") as e:
        sp.did(
            staggered,
            y="y",
            treat="g",
            time="time",
            id="unit",
            method=method,
            parity="r",
            subgroup="ever",
        )
    assert "method='cs'" in e.value.recovery_hint


def test_did_without_parity_is_unchanged(staggered):
    a = _quiet(
        sp.did,
        staggered,
        y="y",
        treat="g",
        time="time",
        id="unit",
        aggregation="simple",
        random_state=4,
    )
    fit = _quiet(
        sp.callaway_santanna,
        staggered,
        y="y",
        g="g",
        t="time",
        i="unit",
        base_period="universal",
    )
    _same(a, _quiet(sp.aggte, fit, type="simple", n_boot=1000, random_state=4))


def test_did_crit_dist_2x2_equals_direct_call(two_by_two):
    kw = dict(y="y", treat="treated", time="post", cluster="unit")
    for crit in ("t", "residual"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            a = sp.did(two_by_two, method="2x2", crit_dist=crit, **kw)
            b = sp.did_2x2(two_by_two, crit_dist=crit, **kw)
        _same(a, b)
        assert (a.ci, a.pvalue) == (b.ci, b.pvalue)
    # 't' is the critical value parity='stata' uses.
    t = _quiet(sp.did, two_by_two, method="2x2", crit_dist="t", **kw)
    st = _quiet(sp.did, two_by_two, method="2x2", parity="stata", **kw)
    assert t.ci == st.ci


def test_did_crit_dist_sun_abraham_equals_direct_call(staggered):
    for crit in ("t", "normal"):
        a = _quiet(
            sp.did,
            staggered,
            y="y",
            treat="g",
            time="time",
            id="unit",
            method="sun_abraham",
            crit_dist=crit,
            aggregation="event_time",
        )
        b = _quiet(
            sp.sun_abraham,
            staggered,
            y="y",
            g="g",
            t="time",
            i="unit",
            crit_dist=crit,
            aggregation="event_time",
        )
        _same(a, b)
        assert (a.ci, a.pvalue) == (b.ci, b.pvalue)


def test_did_without_crit_dist_warns_at_the_caller(two_by_two):
    with pytest.warns(FutureWarning, match="crit_dist") as rec:
        sp.did(
            two_by_two,
            y="y",
            treat="treated",
            time="post",
            method="2x2",
            cluster="unit",
        )
    assert any(
        w.filename == __file__ for w in rec if issubclass(w.category, FutureWarning)
    )


def test_did_crit_dist_legacy_is_the_pre_141_rule_for_any_method(two_by_two, staggered):
    kw = dict(y="y", treat="treated", time="post", cluster="unit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        a = sp.did(two_by_two, method="2x2", crit_dist="legacy", **kw)
        b = sp.did_2x2(two_by_two, crit_dist="residual", **kw)
        sa = sp.did(
            staggered,
            y="y",
            treat="g",
            time="time",
            id="unit",
            method="sun_abraham",
            crit_dist="legacy",
            aggregation="event_time",
        )
        sa_ref = sp.sun_abraham(
            staggered,
            y="y",
            g="g",
            t="time",
            i="unit",
            crit_dist="normal",
            aggregation="event_time",
        )
    _same(a, b)
    assert a.ci == b.ci
    _same(sa, sa_ref)
    # No option on Callaway-Sant'Anna: 'legacy' is a no-op there...
    cs = _quiet(
        sp.did, staggered, y="y", treat="g", time="time", id="unit", crit_dist="legacy"
    )
    _same(cs, _quiet(sp.did, staggered, y="y", treat="g", time="time", id="unit"))
    # ... and under parity= (which sets the t rule itself).
    par = _quiet(
        sp.did, two_by_two, method="2x2", parity="stata", crit_dist="legacy", **kw
    )
    assert par.ci == _quiet(sp.did, two_by_two, method="2x2", parity="stata", **kw).ci


@pytest.mark.parametrize("method", ["cs", "bjs"])
def test_did_refuses_crit_dist_where_not_implemented(staggered, method):
    with pytest.raises(sp.MethodIncompatibility, match="crit_dist") as e:
        sp.did(
            staggered,
            y="y",
            treat="g",
            time="time",
            id="unit",
            method=method,
            crit_dist="t",
        )
    assert "legacy" in e.value.recovery_hint


def test_library_callers_through_sp_did_do_not_warn(two_by_two):
    # sp.compare / causal_question reach sp.did's auto 2x2 branch (0/1
    # flag, 0/1 time, repeated rows per unit and period -> clustered 2x2).
    rep = pd.concat([two_by_two, two_by_two], ignore_index=True)
    q = sp.causal_question(
        treatment="treated", outcome="y", design="did", time="post", id="unit", data=rep
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        r = q.estimate()
        direct = sp.did(
            rep, y="y", treat="treated", time="post", id="unit", crit_dist="residual"
        )
    assert r.estimate == direct.estimate


@pytest.mark.parametrize("parity", ["r", "stata"])
def test_cs_report_parity_equals_direct_calls(staggered, parity):
    rpt = (
        _quiet(
            sp.cs_report,
            staggered,
            y="y",
            g="g",
            t="time",
            i="unit",
            parity=parity,
            n_boot=99,
            random_state=1,
            verbose=False,
            balance=False,
            triangulate=False,
            functional_form=False,
        )
        if parity == "r"
        else _quiet(
            sp.cs_report,
            staggered,
            y="y",
            g="g",
            t="time",
            i="unit",
            parity=parity,
            verbose=False,
            balance=False,
            triangulate=False,
            functional_form=False,
        )
    )
    fit = _quiet(
        sp.callaway_santanna, staggered, y="y", g="g", t="time", i="unit", parity=parity
    )
    boot = dict(n_boot=99, random_state=1) if parity == "r" else {}
    simple = _quiet(sp.aggte, fit, type="simple", **boot)
    dynamic = _quiet(sp.aggte, fit, type="dynamic", **boot)
    assert rpt.overall["estimate"] == simple.estimate
    assert rpt.overall["se"] == simple.se
    pd.testing.assert_frame_equal(
        rpt.dynamic.reset_index(drop=True),
        dynamic.detail.reset_index(drop=True),
        check_exact=True,
    )


def test_cs_report_prefitted_result_with_other_parity_is_refused(staggered):
    fit = _quiet(
        sp.callaway_santanna, staggered, y="y", g="g", t="time", i="unit", parity="r"
    )
    with pytest.raises(sp.MethodIncompatibility, match="parity"):
        _quiet(sp.cs_report, fit, parity="stata", verbose=False)


def test_parity_is_in_the_agent_schema():
    for name in ("did", "cs_report"):
        schema = sp.function_schema(name)
        props = schema.get("parameters", schema).get("properties", {})
        assert "parity" in props, name
