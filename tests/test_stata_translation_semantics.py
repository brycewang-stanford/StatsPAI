"""``sp.from_stata`` / ``sp.stata``: factor notation, weights, exposure (review §6.6).

Before 1.32 ``reg y x i.g [aw=w]`` translated with ``ok=True`` to the
formula ``y ~ x + i.g + [aw=w]``: factor-variable notation and the weight
clause were pasted into the formula, and ``poisson ..., exposure(e)`` lost
its exposure. They are now translated -- or refused with the reason -- and
each translation lists the conventions it relies on under ``semantics``.

Stata 18 MP references on ``reference_parity/_fixtures/margins_ext_data.csv``:

    reg yl x i.g##c.z [pw=w]         _b[x] .579828415480673  _se[x] .0623508905787352
                                     _b[2.g#c.z] -.0484612023965034 _se .110563502618333
    reg yl c.x##c.x i.g [aw=w]       _b[x] .58034306985793   _se[x] .0387402521543367
                                     _b[c.x#c.x] .345450527248457 _se .0265684763696519
                                     _b[3.g] -.51188738125711 _se .102388010073851
    poisson yc x z i.g, exposure(expo)   _b[x] .325134788361568 _se .0205324711160059
                                     (Stata default ML tolerance: ~1e-8)
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

_DATA = (
    Path(__file__).parent / "reference_parity" / "_fixtures" / "margins_ext_data.csv"
)


@pytest.fixture(scope="module")
def d():
    return pd.read_csv(_DATA)


def _run(cmd, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(cmd, data=d)


def test_pweights_and_factor_interaction_match_stata(d):
    r = _run("reg yl x i.g##c.z [pw=w]", d)
    assert r.params["x"] == pytest.approx(0.579828415480673, rel=1e-10)
    assert r.std_errors["x"] == pytest.approx(0.0623508905787352, rel=1e-10)
    assert r.params["C(g)[T.2]:z"] == pytest.approx(-0.0484612023965034, rel=1e-10)
    assert r.std_errors["C(g)[T.2]:z"] == pytest.approx(0.110563502618333, rel=1e-10)


def test_aweights_and_quadratic_match_stata(d):
    r = _run("reg yl c.x##c.x i.g [aw=w]", d)
    assert r.params["x"] == pytest.approx(0.58034306985793, rel=1e-10)
    assert r.std_errors["x"] == pytest.approx(0.0387402521543367, rel=1e-10)
    assert r.params["I(x ** 2)"] == pytest.approx(0.345450527248457, rel=1e-10)
    assert r.std_errors["I(x ** 2)"] == pytest.approx(0.0265684763696519, rel=1e-10)
    assert r.params["C(g)[T.3]"] == pytest.approx(-0.51188738125711, rel=1e-10)


def test_poisson_exposure_is_carried_over(d):
    r = _run("poisson yc x z i.g, exposure(expo)", d)
    assert r.params["x"] == pytest.approx(0.325134788361568, rel=1e-7)
    assert r.std_errors["x"] == pytest.approx(0.0205324711160059, rel=1e-7)


@pytest.mark.parametrize(
    "line,needle",
    [
        ("reg y x L.x", "time-series operator"),
        ("reg y x [iw=w]", "importance weights"),
        ("reg y x [fw=w]", "repeat"),
        ("rddensity x [aw=w]", "weights"),
    ],
)
def test_untranslatable_pieces_are_refused(line, needle):
    out = sp.from_stata(line)
    assert out["ok"] is False
    assert needle in out["error"] + " ".join(out.get("suggestions", []))


def test_semantics_are_reported():
    out = sp.from_stata("reg y x i.g [pw=w]")
    assert out["ok"]
    assert any("base" in s for s in out["semantics"])
    assert any("robust" in s for s in out["semantics"])
    assert out["arguments"]["weights"] == "w" and out["arguments"]["robust"] == "hc1"
