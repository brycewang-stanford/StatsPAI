"""``areg`` and ``rdbwselect`` translations, checked against Stata 18.

The expected numbers were printed by Stata 18 MP (``%12.8f``) on
``tests/fixtures/areg_stata18.csv`` and on
``tests/reference_parity/_fixtures/rd_cluster_cer.csv``, so the tolerance
is 1e-7. The fixtures hold 600 and 1,200 simulated rows; the first has
five singleton groups, which is where ``areg`` and ``reghdfe`` part ways
on robust standard errors.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

HERE = Path(__file__).parent

#: line -> (b[x1], se[x1]) from Stata 18.
AREG_STATA = {
    "areg y x1 x2, absorb(id)": (0.49436342, 0.07589178),
    "areg y x1 x2, absorb(id) vce(robust)": (0.49436342, 0.08956845),
    "areg y x1 x2, absorb(id) vce(cluster c)": (0.49436342, 0.09002401),
    "areg y x1 x2, absorb(id) vce(cluster id)": (0.49436342, 0.10499470),
    "areg y x1 x2 [aw=w], absorb(id)": (0.54608217, 0.07550365),
    "areg y x1 x2 [aw=w], absorb(id) vce(robust)": (0.54608217, 0.08869172),
    "areg y x1 x2 [aw=w], absorb(id) vce(cluster c)": (0.54608217, 0.08516396),
    "areg y x1 x2 [pw=w], absorb(id) vce(cluster id)": (0.54608217, 0.09600298),
}


@pytest.fixture(scope="module")
def areg_frame() -> pd.DataFrame:
    return pd.read_csv(HERE / "fixtures" / "areg_stata18.csv")


def _run(line: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(line, data=data)


@pytest.mark.parametrize("line", sorted(AREG_STATA))
def test_areg_reproduces_stata(areg_frame, line):
    res = _run(line, areg_frame)
    b, se = AREG_STATA[line]
    assert float(res.params["x1"]) == pytest.approx(b, abs=1e-7)
    assert float(res.std_errors["x1"]) == pytest.approx(se, abs=1e-7)


def test_areg_is_not_reghdfe_when_clustering_on_the_absorbed_variable(areg_frame):
    """The reason the translation does not go to ``sp.hdfe_ols``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdfe = sp.hdfe_ols("y ~ x1 + x2 | id", data=areg_frame, cluster="id")
    areg_se = AREG_STATA["areg y x1 x2, absorb(id) vce(cluster id)"][1]
    assert float(hdfe.params["x1"]) == pytest.approx(0.49436342, abs=1e-7)
    assert abs(float(hdfe.std_errors["x1"]) - areg_se) > 1e-3


def test_areg_translation_shape():
    out = sp.from_stata("qui areg y x1 i.year [aw=w], a(id) cl(c) noheader")
    assert out["ok"] is True
    assert out["tool"] == "hdfe_ols"
    assert out["arguments"] == {
        "formula": "y ~ x1 + i.year | id",
        "drop_singletons": False,
        "absorb_dof": "areg",
        "cluster": "c",
        "weights": "w",
    }
    assert out["untranslated_options"] == []
    assert out["ignored_display_options"] == ["noheader"]
    # variances the absorbing path does not offer keep the dummy variables
    for line in (
        "areg y x1 [aw=w], absorb(id) vce(robust)",
        "areg y x1 [pw=w], absorb(id)",
        "areg y x1, absorb(id) vce(hc3)",
    ):
        dummies = sp.from_stata(line)
        assert dummies["tool"] == "regress", line
        assert dummies["arguments"]["formula"] == "y ~ x1 + C(id)", line


def test_areg_absorbing_path_is_the_dummy_variable_regression():
    """absorb_dof='areg' with drop_singletons=False equals sp.regress with
    C(g) for every variance the translation sends there, including a fixed
    effect nested in the cluster variable and singleton groups."""
    rng = np.random.default_rng(0)
    n = 600
    d = pd.DataFrame(
        {
            "g": rng.integers(0, 60, n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "w": rng.uniform(0.5, 2, n),
            "year": rng.integers(2000, 2005, n),
        }
    )
    d["c"] = d["g"] // 4  # g is nested in c
    d.loc[len(d)] = [999, 0.3, 0.1, 1.0, 2001, 3]  # a singleton group
    d["y"] = d.x1 + 0.5 * d.x2 + 0.01 * d.g + rng.normal(size=len(d))
    cases = [
        {},
        {"vce": "robust"},
        {"cluster": "g"},
        {"cluster": "c"},
        {"weights": "w"},
        {"weights": "w", "cluster": "g"},
    ]
    for kw in cases:
        dummies = sp.regress("y ~ x1 + x2 + C(year) + C(g)", data=d, **kw)
        absorbed = sp.hdfe_ols(
            "y ~ x1 + x2 + i.year | g",
            data=d,
            drop_singletons=False,
            absorb_dof="areg",
            **kw,
        )
        for name in ("x1", "x2"):
            assert absorbed.params[name] == pytest.approx(
                dummies.params[name], rel=1e-10
            )
            assert absorbed.std_errors[name] == pytest.approx(
                dummies.std_errors[name], rel=1e-10
            ), kw
    # reghdfe's rule differs exactly where a fixed effect is nested
    reghdfe = sp.hdfe_ols("y ~ x1 + x2 | g", data=d, drop_singletons=False, cluster="g")
    areg = sp.hdfe_ols(
        "y ~ x1 + x2 | g", data=d, drop_singletons=False, cluster="g", absorb_dof="areg"
    )
    assert areg.std_errors["x1"] > reghdfe.std_errors["x1"] * 1.01
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="absorb_dof"):
        sp.hdfe_ols("y ~ x1 | g", data=d, absorb_dof="xtreg")


@pytest.mark.parametrize(
    "line",
    ["areg y x1", "areg y x1, absorb(id year)", "areg y x1, absorb(id#year)"],
)
def test_areg_needs_exactly_one_absorbed_variable(line):
    out = sp.from_stata(line)
    assert out["ok"] is False
    assert "exactly one" in out["error"]


def test_rdbwselect_options_map_onto_sp_rdbwselect():
    out = sp.from_stata(
        "rdbwselect y x, c(0.1) p(2) kernel(uni) bwselect(certwo) covs(z w) "
        "vce(cluster g) masspoints(off) all"
    )
    assert out["ok"] is True
    assert out["tool"] == "rdbwselect"
    assert out["arguments"] == {
        "y": "y",
        "x": "x",
        "c": 0.1,
        "p": 2,
        "kernel": "uniform",
        "bwselect": "certwo",
        "covs": ["z", "w"],
        "cluster": "g",
        "all": True,
        "masspoints": "off",
    }
    assert out["untranslated_options"] == []


def test_rdbwselect_default_vce_is_accepted_and_others_are_reported():
    same = sp.from_stata("rdbwselect y x, vce(nn 3)")
    assert same["untranslated_options"] == []
    assert same["arguments"] == {"y": "y", "x": "x", "c": 0.0}
    lost = sp.from_stata("rdbwselect y x, vce(hc2) scaleregul(0) weights(w)")
    assert lost["untranslated_options"] == ["vce", "weights", "scaleregul"]
    with pytest.raises(sp.MethodIncompatibility, match="vce"):
        sp.stata(
            "rdbwselect y x, vce(hc2)", data=pd.DataFrame({"y": [0.0], "x": [0.0]})
        )


def test_rdbwselect_reproduces_stata():
    frame = pd.read_csv(HERE / "reference_parity" / "_fixtures" / "rd_cluster_cer.csv")
    # Stata 18: e(h_mserd), e(b_mserd)
    a = _run("rdbwselect y x, c(0.1)", frame).iloc[0]
    assert a["h_left"] == pytest.approx(0.34991159, abs=1e-7)
    assert a["b_left"] == pytest.approx(0.52448847, abs=1e-7)
    # e(h_msetwo_l), e(h_msetwo_r), e(b_msetwo_l), e(b_msetwo_r)
    b = _run("rdbwselect y x, c(0.1) kernel(uniform) p(2) bwselect(msetwo)", frame)
    assert tuple(
        b.iloc[0][["h_left", "h_right", "b_left", "b_right"]]
    ) == pytest.approx((0.49609777, 0.37035171, 0.70749196, 0.54943516), abs=1e-7)
    # e(h_cerrd), e(b_cerrd) with covariates and clustering
    c = _run("rdbwselect y x, c(0.1) covs(z) vce(cluster g) bwselect(cerrd)", frame)
    assert c.iloc[0]["h_left"] == pytest.approx(0.23049687, abs=1e-7)
    assert c.iloc[0]["b_left"] == pytest.approx(0.44934339, abs=1e-7)
    # all: e(h_mserd), e(h_msesum), e(h_cercomb2_l), e(h_cercomb2_r)
    d = _run("rdbwselect y x, c(0.1) all", frame).set_index("method")
    assert d.loc["mserd", "h_left"] == pytest.approx(0.34991159, abs=1e-7)
    assert d.loc["msesum", "h_left"] == pytest.approx(0.41165761, abs=1e-7)
    assert d.loc["cercomb2", "h_left"] == pytest.approx(0.25280700, abs=1e-7)
    assert d.loc["cercomb2", "h_right"] == pytest.approx(0.25682212, abs=1e-7)
