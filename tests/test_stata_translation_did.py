"""Translation of ``drdid``, ``jwdid``, ``csdid_estat`` and the ``estat``
aggregations: what the payload says, what is refused, and that the line and
the call it names are the same computation. The numbers against Stata are in
``tests/reference_parity/test_stata_did_commands_parity.py``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    df = sp.dgp_did(n_units=150, n_periods=6, staggered=True, seed=7)
    df["first_treat"] = df["first_treat"].fillna(0).astype(int)
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=df["unit"].nunique()), index=df["unit"].unique())
    df["x"] = df["unit"].map(x)
    return df


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ----------------------------------------------------------------------
# drdid
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("", {"est_method": "dr", "method": "imp"}),
        ("drimp", {"est_method": "dr", "method": "imp"}),
        ("dripw", {"est_method": "dr", "method": "trad"}),
        ("reg", {"est_method": "reg"}),
        ("stdipw", {"est_method": "ipw", "normalized": True}),
        ("ipw", {"est_method": "ipw", "normalized": False}),
    ],
)
def test_drdid_estimator_flags(flag, expected):
    out = sp.from_stata(f"drdid y x1 x2, ivar(id) time(t) treatment(d) {flag}")
    assert out["ok"] and out["tool"] == "drdid"
    args = out["arguments"]
    assert args["y"] == "y" and args["group"] == "d" and args["time"] == "t"
    assert args["id"] == "id" and args["covariates"] == ["x1", "x2"]
    for key, value in expected.items():
        assert args[key] == value
    assert out["untranslated_options"] == []


def test_drdid_abbreviated_options_and_repeated_cross_sections():
    out = sp.from_stata("drdid y x, t(year) tr(d) dripw rc1 pscoretrim(0.99)")
    args = out["arguments"]
    assert "id" not in args
    assert args["locally_efficient"] is False
    assert args["trim_level"] == 0.99


@pytest.mark.parametrize(
    "line, words",
    [
        ("drdid y x, ivar(id) time(t) treatment(d) all", "once per estimator"),
        ("drdid y x, ivar(id) time(t) treatment(d) ipwra", "ipwra"),
        ("drdid y x, ivar(id) treatment(d)", "time("),
        ("drdid y i.x, ivar(id) time(t) treatment(d)", "factor-variable"),
        ("drdid y x, ivar(id) time(t) treatment(d) reg ipw", "only one estimator"),
    ],
)
def test_drdid_refusals(line, words):
    out = sp.from_stata(line)
    assert not out["ok"]
    assert words in out["error"]


@pytest.mark.parametrize("option", ["wboot", "gmm", "cluster(st)", "rc1"])
def test_drdid_options_that_are_not_carried_over_are_reported(option):
    out = sp.from_stata(f"drdid y x, ivar(id) time(t) treatment(d) reg {option}")
    assert out["ok"]
    assert option.split("(")[0] in out["untranslated_options"]


def test_drdid_line_is_the_direct_call(panel):
    two = panel[panel["time"].isin([2, 5]) & panel["first_treat"].isin([0, 4])].copy()
    two["d"] = (two["first_treat"] == 4).astype(int)
    via = _quiet(sp.stata, "drdid y x, ivar(unit) time(time) treatment(d) dripw", two)
    direct = _quiet(
        sp.drdid,
        two,
        y="y",
        group="d",
        time="time",
        covariates=["x"],
        id="unit",
        est_method="dr",
        method="trad",
    )
    assert via.estimate == direct.estimate and via.se == direct.se


# ----------------------------------------------------------------------
# csdid
# ----------------------------------------------------------------------


def test_csdid_method_drimp():
    out = sp.from_stata("csdid y x, ivar(id) time(t) gvar(g) method(drimp)")
    assert out["ok"] and out["arguments"]["estimator"] == "drimp"


def test_unknown_estimator_is_refused_by_callaway_santanna(panel):
    with pytest.raises(MethodIncompatibility):
        sp.callaway_santanna(
            panel, y="y", i="unit", t="time", g="first_treat", estimator="dr_imp"
        )


def test_drimp_on_repeated_cross_sections_runs(panel):
    res = _quiet(
        sp.callaway_santanna,
        panel,
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        x=["x"],
        estimator="drimp",
        panel=False,
    )
    assert np.isfinite(res.estimate) and res.se > 0


def test_drimp_with_weights_reduces_to_unweighted_at_unit_weights(panel):
    kw = dict(y="y", i="unit", t="time", g="first_treat", x=["x"], estimator="drimp")
    plain = _quiet(sp.callaway_santanna, panel, **kw)
    ones = _quiet(sp.callaway_santanna, panel.assign(w=2.0), weights="w", **kw)
    np.testing.assert_allclose(plain.detail["att"], ones.detail["att"], rtol=1e-10)
    np.testing.assert_allclose(plain.detail["se"], ones.detail["se"], rtol=1e-10)


# ----------------------------------------------------------------------
# estat / csdid_estat aggregations
# ----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["estat", "csdid_estat"])
@pytest.mark.parametrize("kind", ["simple", "group", "calendar", "event"])
def test_aggregation_translation(command, kind):
    out = sp.from_stata(f"{command} {kind}, post estore(m)")
    assert out["ok"] and out["tool"] == "estat"
    assert out["arguments"] == {"test": kind, "print_results": False}
    assert out["python_code"].startswith("sp.estat(result, ")
    assert out["untranslated_options"] == []


def test_event_window_translation():
    out = sp.from_stata("csdid_estat event, window(-4 5)")
    assert out["arguments"]["window"] == (-4, 5)
    assert not sp.from_stata("estat group, window(-1 1)")["ok"]
    assert not sp.from_stata("csdid_estat pretrend")["ok"]
    assert not sp.from_stata("estat event, window(a b)")["ok"]


def test_estat_aggregations_are_the_direct_calls(panel):
    cs = _quiet(sp.callaway_santanna, panel, y="y", i="unit", t="time", g="first_treat")
    for kind, typ in [
        ("simple", "simple"),
        ("calendar", "calendar"),
        ("event", "dynamic"),
    ]:
        via = sp.estat(cs, kind, print_results=False)
        direct = sp.aggte(cs, type=typ)
        assert via.estimate == direct.estimate and via.se == direct.se
    grouped = sp.estat(cs, "group", print_results=False)
    direct = sp.aggte(cs, type="group", share_variance=False)
    assert grouped.se == direct.se
    windowed = sp.estat(cs, "event", window=(0, 1), print_results=False)
    assert sorted(windowed.detail["relative_time"]) == [0, 1]
    direct = sp.aggte(cs, type="dynamic", min_e=0, max_e=1)
    assert windowed.estimate == direct.estimate and windowed.se == direct.se

    jw = _quiet(sp.jwdid, panel, "y", ivar="unit", tvar="time", gvar="first_treat")
    for kind in ("group", "calendar", "event"):
        via = sp.estat(jw, kind, print_results=False)
        direct = sp.etwfe_emfx(jw, type=kind)
        assert via.estimate == direct.estimate and via.se == direct.se


def test_estat_aggregation_refuses_what_it_cannot_aggregate(panel):
    ols = sp.regress("y ~ x", data=panel)
    with pytest.raises(MethodIncompatibility, match="staggered"):
        sp.estat(ols, "event", print_results=False)
    cs = _quiet(sp.callaway_santanna, panel, y="y", i="unit", t="time", g="first_treat")
    with pytest.raises(MethodIncompatibility, match="window"):
        sp.estat(cs, "group", window=(-1, 1), print_results=False)
    with pytest.raises(MethodIncompatibility, match="two event times"):
        sp.estat(cs, "event", window=(1,), print_results=False)
    jw = _quiet(sp.jwdid, panel, "y", ivar="unit", tvar="time", gvar="first_treat")
    with pytest.raises(MethodIncompatibility, match="window"):
        sp.estat(jw, "event", window=(-1, 1), print_results=False)


def test_estat_needs_a_fit_before_it(panel):
    with pytest.raises(TypeError, match="post-estimation"):
        sp.stata("csdid_estat event", panel)


# ----------------------------------------------------------------------
# jwdid
# ----------------------------------------------------------------------


def test_jwdid_translation():
    out = sp.from_stata(
        "jwdid y x, ivar(id) tvar(year) gvar(g) never method(ppmlhdfe) cluster(st)"
    )
    assert out["ok"] and out["tool"] == "jwdid"
    assert out["arguments"] == {
        "y": "y",
        "ivar": "id",
        "tvar": "year",
        "gvar": "g",
        "x": ["x"],
        "method": "ppmlhdfe",
        "never": True,
        "cluster": "st",
    }


@pytest.mark.parametrize(
    "line",
    [
        "jwdid y, tvar(year) gvar(g)",
        "jwdid y, ivar(id) tvar(year) trtvar(d)",
    ],
)
def test_jwdid_refusals(line):
    assert not sp.from_stata(line)["ok"]


def test_jwdid_options_that_are_not_carried_over_are_reported():
    out = sp.from_stata("jwdid y, ivar(id) tvar(year) gvar(g) group")
    assert "group" in out["untranslated_options"]
    out = sp.from_stata("jwdid y, ivar(id) tvar(year) gvar(g) method(probit, iter(5))")
    assert "method" in out["untranslated_options"]


def test_jwdid_line_is_the_direct_call(panel):
    via = _quiet(
        sp.stata, "jwdid y, ivar(unit) tvar(time) gvar(first_treat) never", panel
    )
    direct = _quiet(
        sp.jwdid, panel, "y", ivar="unit", tvar="time", gvar="first_treat", never=True
    )
    assert via.estimate == direct.estimate and via.se == direct.se
