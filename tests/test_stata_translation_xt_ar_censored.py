"""``sp.from_stata`` / ``sp.stata`` for ``xtregar``, ``xttobit`` and
``xtintreg``.

Executed equivalence: each Stata line runs through ``sp.stata`` on the
fixtures of ``tests/test_xtregar_stata.py`` and ``tests/test_xttobit_stata.py``
and is compared with the Stata 18 MP numbers frozen there
(``tests/fixtures/xtregar/stata_xtregar.json``,
``tests/fixtures/xttobit/stata_xttobit.json``), at the tolerances those
tests use for the direct calls. The commands are written with Stata's own
abbreviations (``rhot(tsc)``, ``two``, ``intp(40)``) so the option grammar
is exercised too.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures"


def _load(sub, name):
    spec = importlib.util.spec_from_file_location(
        f"_{sub}_data", _FIX / sub / "_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = json.loads((_FIX / sub / name).read_text(encoding="utf-8"))
    return mod.make_panel(), ref


AR_DF, AR = _load("xtregar", "stata_xtregar.json")
TOB_DF, TOB = _load("xttobit", "stata_xttobit.json")


# ---------------------------------------------------------------- xtregar

_XTREGAR = [
    ("fe", "xtregar y x1 x2, fe"),
    ("re", "xtregar y x1 x2"),
    ("fe_tscorr_2s", "xtregar y x1 x2, fe rhot(tsc) two"),
    ("re_theil", "xtregar y x1 x2, re rhotype(th)"),
    ("fe_onestep", "xtregar y x1 x2, fe rhotype(onestep)"),
    ("re_dw_2s", "xtregar y x1 x2, re twostep"),
    ("fe_rhof", "xtregar y x1 x2, fe rhof(0.5)"),
    ("fe_lbi", "xtregar y x1 x2, fe lbi"),
    ("re_bal", "xtregar y x1 x2 if bal == 1, re lbi"),
]


@pytest.mark.parametrize("tag,line", _XTREGAR)
def test_xtregar_runs_as_stata(tag, line):
    fit = sp.stata(f"xtset id t\n{line}", data=AR_DF)
    ref = AR[tag]
    for v in ["x1", "x2", "_cons"]:
        np.testing.assert_allclose(fit.params[v], ref[v][0], rtol=1e-10)
        se = fit.std_errors[v]
        if tag.startswith("fe") and v == "_cons":
            # the documented departure of sp.xtregar (module docs): Stata
            # leaves the constant's SE undivided by 1 - rho
            se = se * (1 - fit.model_info["rho_ar"])
        np.testing.assert_allclose(se, ref[v][1], rtol=1e-10)
    np.testing.assert_allclose(
        fit.model_info["rho_ar"], ref["e(rho_ar)"][0], rtol=1e-10
    )
    assert fit.data_info["nobs"] == ref["e(N)"][0]
    if ref.get("e(LBI)", [None])[0] is not None:
        np.testing.assert_allclose(fit.model_info["lbi"], ref["e(LBI)"][0], rtol=1e-10)


def test_xtregar_translation_payload():
    out = sp.from_stata("xtregar y x1 x2, fe rhot(reg) two lbi level(90)")
    assert out["ok"] and out["complete"]
    assert out["arguments"] == {
        "y": "y",
        "x": ["x1", "x2"],
        "id": "<panel_id>",
        "time": "<panel_time>",
        "method": "fe",
        "rhotype": "regress",
        "twostep": True,
        "lbi": True,
        "alpha": 0.1,
    }
    filled = sp.from_stata("xtregar y x1 x2, i(firm) t(year)")
    assert filled["arguments"]["id"] == "firm"
    assert filled["arguments"]["time"] == "year"
    assert filled["arguments"]["method"] == "re"


@pytest.mark.parametrize(
    "line,match",
    [
        ("xtregar y x1, rhotype(prais)", "rhotype"),
        ("xtregar y x1, rhof(a)", "rhof"),
        ("xtregar y x1, fe re", "fe and re"),
        ("xtregar y", "regressor"),
        ("xtregar y i.g x1", "plain variable"),
    ],
)
def test_xtregar_refusals(line, match):
    out = sp.from_stata(line)
    assert not out["ok"]
    assert match in out["error"]


def test_xtregar_weights_refused():
    # Stata allows fweight / aweight only for fe with regress / freg / a
    # fixed rho; sp.xtregar takes no weights
    assert not sp.from_stata("xtregar y x1 [aw=w], fe rhof(0.5)")["ok"]


# ------------------------------------------------------- xttobit / xtintreg


def _check_tobit(fit, tag, eq, rtol_b, rtol_se):
    r = TOB[tag]
    for v in ["x1", "x2", "_cons"]:
        np.testing.assert_allclose(fit.params[v], r[f"{eq}:{v}"][0], rtol=rtol_b)
        np.testing.assert_allclose(fit.std_errors[v], r[f"{eq}:{v}"][1], rtol=rtol_se)
    np.testing.assert_allclose(fit.params["sigma_u"], r["e(sigma_u)"][0], rtol=rtol_b)
    np.testing.assert_allclose(fit.params["sigma_e"], r["e(sigma_e)"][0], rtol=rtol_b)
    np.testing.assert_allclose(
        fit.model_info["log_likelihood"], r["e(ll)"][0], rtol=rtol_b
    )
    assert fit.model_info["n_groups"] == r["e(N_g)"][0]


@pytest.mark.parametrize(
    "tag,line",
    [
        ("tobit_q40", "xttobit y x1 x2, ll(0) ul(4) intp(40) tobit"),
        ("tobit_ll_q40", "xttobit y x1 x2, ll(0) intpoints(40) vce(oim)"),
        ("tobit_q12", "xttobit y x1 x2, ll(0) ul(4) intpoints(12)"),
    ],
)
def test_xttobit_runs_as_stata(tag, line):
    fit = sp.stata(f"xtset id t\n{line}", data=TOB_DF)
    _check_tobit(fit, tag, "y", rtol_b=1e-6, rtol_se=1e-6)
    np.testing.assert_allclose(
        fit.model_info["lr_sigma_u0"], TOB[tag]["e(chi2_c)"][0], rtol=1e-6
    )


def test_xtintreg_runs_as_stata():
    # point estimates and log likelihood at the tolerance of the direct
    # call; Stata's adaptive-quadrature standard errors are not the
    # curvature of its own likelihood (module docs), so the SEs are checked
    # against its non-adaptive 100-point fit
    fit = sp.stata("xtset id t\nxtintreg lo hi x1 x2, intp(40) intreg", data=TOB_DF)
    _check_tobit(fit, "intreg_q40", "lo", rtol_b=1e-6, rtol_se=1e-2)
    r = TOB["intreg_gh100"]
    for v in ["x1", "x2", "_cons"]:
        np.testing.assert_allclose(fit.std_errors[v], r[f"lo:{v}"][1], rtol=1e-6)


def test_censored_translation_payloads():
    out = sp.from_stata("xttobit y x1 x2, ll(0) ul(.) intp(20) i(firm)")
    assert out["ok"] and out["complete"]
    assert out["arguments"] == {
        "y": "y",
        "x": ["x1", "x2"],
        "id": "firm",
        "ll": 0.0,
        "intpoints": 20,
    }
    out = sp.from_stata("xtintreg lo hi x1, intmethod(mvaghermite)")
    assert out["ok"] and out["complete"]
    assert out["arguments"]["y_lower"] == "lo"
    # non-adaptive quadrature approximates the likelihood differently: kept
    # out of the call, so sp.stata refuses to run the line
    out = sp.from_stata("xttobit y x1, ll(0) intmethod(ghermite)")
    assert out["ok"] and not out["complete"]
    assert "intmethod" in out["untranslated_options"]
    with pytest.raises(MethodIncompatibility):
        sp.stata("xtset id t\nxttobit y x1 x2, ll(0) intm(gh)", data=TOB_DF)
    # noconstant has no counterpart in sp.xttobit
    out = sp.from_stata("xttobit y x1, ll(0) nocons")
    assert "noconstant" in out["untranslated_options"]


@pytest.mark.parametrize(
    "line,match",
    [
        ("xttobit y x1, ll", "without a value"),
        ("xttobit y x1, ll(lim)", "not a number"),
        ("xttobit y x1", "censors nothing"),
        ("xttobit y x1 [iw=w], ll(0)", "weights"),
        ("xtintreg lo hi", "at least one"),
        ("xtintreg lo hi x1, intpoints(x)", "integer"),
    ],
)
def test_censored_refusals(line, match):
    out = sp.from_stata(line)
    assert not out["ok"]
    assert match in out["error"]
