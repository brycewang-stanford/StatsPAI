"""``sp.from_stata`` / ``sp.stata`` for ``mmqreg`` (and the ``xtqreg``
refusal).

Executed equivalence: each ``mmqreg`` line runs through ``sp.stata`` on the
panel of ``tests/test_mmqreg.py`` and is compared with the Stata 18 MP
``mmqreg`` 2.6 numbers frozen there (``tests/fixtures/mmqreg/
stata_mmqreg.json``): the whole covariance of the location, scale and
quantile equations at 1e-10, as for the direct call. The lines use
``mmqreg``'s own abbreviations (``abs()``, ``q()``, ``r``, ``cl()``).
``mmqreg`` reads ``q(.5)`` as the median, as ``q(50)`` (checked on Stata
18: identical ``e(qval)``), so values below 1 are fractions, as in qreg.

``xtqreg`` is refused because its quantile-coefficient standard errors use
a GLS form that is not reproduced; the call it suggests is executed and
matches ``xtqreg``'s coefficients, ``e(q)`` and location / scale
coefficients at 1e-10.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "fixtures" / "mmqreg"
_spec = importlib.util.spec_from_file_location("_mmqreg_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_mmqreg.json").read_text(encoding="utf-8"))
XS = ["x1", "x2", "ax1"]
# panel 61 is a single observation; the Stata runs drop it
DF = _mod.make_panel()
DF = DF[DF["id"] != 61]


def _mat(tag, name):
    m = STATA[tag][name]
    return np.array(m["data"]), m["rows"], m["cols"]


_VCE = {"gls": "", "robust": " r", "cluster": " cl(id)"}
_CASES = [
    (f"fe1_{v}_{q}", f"mmqreg y x1 x2 ax1, abs(id) q({q}){_VCE[v]}")
    for q in (25, 50, 90)
    for v in _VCE
]
_CASES += [
    (f"fe2_{v}_50", f"mmqreg y x1 x2 ax1, absorb(id t) q(.5){_VCE[v]} nols")
    for v in _VCE
]
_CASES += [
    (f"nofe_{v}_50", f"mmqreg y x1 x2 ax1{','+_VCE[v] if v != 'gls' else ''}")
    for v in _VCE
]


@pytest.mark.parametrize("tag,line", _CASES)
def test_mmqreg_runs_as_stata(tag, line):
    fit = sp.stata(line, data=DF)
    V, rows, cols = _mat(tag, "e(V)")
    ours = fit.model_info["vcov_all"].loc[rows, cols].to_numpy()
    np.testing.assert_allclose(ours, V, rtol=1e-10, atol=1e-14 * np.abs(V).max())
    gls = tag.rsplit("_", 2)[0] + "_gls_" + tag.rsplit("_", 1)[1]
    b = _mat(gls, "e(b)")[0].ravel()
    coef = np.r_[
        fit.model_info["location"]["coef"],
        fit.model_info["scale"]["coef"],
        np.asarray(fit.params),
    ]
    np.testing.assert_allclose(coef, b, rtol=1e-10, atol=1e-13)
    assert fit.nobs == STATA[gls]["e(N)"]["data"][0][0]


def test_mmqreg_translation_payload():
    out = sp.from_stata("mmqreg y x1 x2, absorb(id year) q(90) cluster(id) level(90)")
    assert out["ok"] and out["complete"]
    assert out["arguments"] == {
        "y": "y",
        "x": ["x1", "x2"],
        "absorb": ["id", "year"],
        "quantile": 0.9,
        "vce": "cluster",
        "cluster": "id",
        "alpha": 0.1,
    }
    assert sp.from_stata("mmqreg y x1, vce(robust)")["arguments"]["vce"] == "robust"
    assert sp.from_stata("mmqreg y x1")["arguments"]["quantile"] == 0.5


@pytest.mark.parametrize(
    "line,match",
    [
        ("mmqreg y x1, q(25 50)", "one quantile per call"),
        ("mmqreg y x1, q(10(20)90)", "one quantile per call"),
        ("mmqreg y x1, q(100)", "outside"),
        ("mmqreg y x1, jk", "jackknife"),
        ("mmqreg y x1, dfadj", "dfadj"),
        ("mmqreg y x1, den(kernel)", "density"),
        ("mmqreg y x1, vce(hc3)", "not translated"),
        ("mmqreg y x1, cluster(a b)", "clusters on one"),
        ("mmqreg y i.g x1", "plain variable"),
        ("mmqreg y x1 [pw=w]", "weights"),
        ("mmqreg y", "regressor"),
    ],
)
def test_mmqreg_refusals(line, match):
    out = sp.from_stata(line)
    assert not out["ok"]
    assert match in out["error"]


@pytest.mark.parametrize("q", [25, 50, 90])
def test_xtqreg_refused_with_an_executable_pointer(q):
    out = sp.from_stata(f"xtqreg y x1 x2 ax1, i(id) quantile({q / 100}) ls")
    assert not out["ok"]
    assert "sp.mmqreg" in out["error"] and "GLS" in out["error"]
    (call,) = out["suggestions"]
    fit = eval(call, {"sp": sp, "df": DF})  # the suggested line, as written
    tag = f"xtqreg_{q}"
    np.testing.assert_allclose(fit.params[XS], _mat(tag, "e(b)")[0].ravel(), rtol=1e-10)
    assert fit.model_info["q"] == pytest.approx(_mat(tag, "e(q)")[0][0, 0], rel=1e-12)
    loc = fit.model_info["location"]["coef"][XS].to_numpy()
    np.testing.assert_allclose(loc, _mat(tag, "e(b_location)")[0].ravel(), rtol=1e-10)
    with pytest.raises(Exception):
        sp.stata(f"xtqreg y x1 x2 ax1, i(id) quantile({q})", data=DF)


def test_coverage_lists_mmqreg_but_not_xtqreg_as_translated():
    rows = {r["command"]: r for r in sp.translation_coverage()["stata"]}
    assert rows["mmqreg"]["targets"] == ["sp.mmqreg"]
    assert rows["xtqreg"]["targets"] == []
    assert any("xtqreg" in line for line in sp.translation_coverage()["limitations"])
