"""Reference parity: ``sp.rddensity`` vs R rddensity 2.6.

Unlike the ``sp.rdrobust`` cascade, which was substantively wrong, this
module was already correct -- it was simply **unanchored**: no test tied
any of its numbers to Cattaneo, Jansson & Ma's own implementation, so a
future refactor could have shifted them without anything noticing. That is
the gap this file closes.

Measured agreement with ``rddensity`` 2.6, over six cells (three designs x
p in {2, 3}), on the manipulation test statistic, its p-value, both
bandwidths and both one-sided density estimates::

    T statistic   max rel 1.2e-09
    p-value       max rel 2.2e-08
    h_left/right  max rel 9.8e-11
    f_left/right  max rel 1.3e-10

Three designs, chosen so the suite can distinguish a working test from one
that merely always agrees:

* ``senate``   -- ``rdrobust``'s own margin-of-victory data, heavily tied.
* ``null``     -- N(0,1), a smooth density with nothing at the cutoff. A
  test that always rejects fails here.
* ``manip``    -- N(0,1) with 45% of the mass in ``(-0.35, 0)`` deleted. A
  test that never rejects fails here.

``test_design_is_discriminating`` asserts R itself separates the last two,
so the fixture cannot silently decay into one that any implementation
passes.

References
----------
cattaneo2020simple
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

# Both sides implement the same closed forms; the residual is float
# accumulation through the bandwidth selector, not a modelling difference.
RTOL = 1e-6

_DESIGNS = ["senate", "null", "manip"]
_ORDERS = [2, 3]


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rddensity_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rddensity_R.R to build rddensity_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def designs():
    senate = pd.read_csv(_FIX / "rdsenate.csv").rename(columns={"margin": "x"})
    return {
        "senate": senate,
        "null": pd.read_csv(_FIX / "rddensity_null.csv"),
        "manip": pd.read_csv(_FIX / "rddensity_manip.csv"),
    }


# ── the fixture must be able to tell right from wrong ──────────────────── #


def test_design_is_discriminating(rjson):
    """R itself must separate the null design from the manipulated one.

    Without this, every assertion below would also pass for a test that
    always rejects, or one that never does.
    """
    assert rjson["null_p2"]["p_jk"] > 0.05, (
        f"the 'null' design already rejects in R "
        f"(p={rjson['null_p2']['p_jk']}); it cannot show a false positive"
    )
    assert rjson["manip_p2"]["p_jk"] < 0.05, (
        f"the 'manip' design does not reject in R "
        f"(p={rjson['manip_p2']['p_jk']}); it cannot show a false negative"
    )
    assert (
        rjson["manip_p2"]["T_jk"] > 0 > rjson["null_p2"]["T_jk"]
    ), "the two designs should push the statistic in opposite directions"


def test_datasets_match_the_r_side(rjson, designs):
    meta = rjson["_meta"]
    assert len(designs["null"]) == int(meta["n_null"])
    assert len(designs["manip"]) == int(meta["n_manip"])
    assert len(designs["senate"]) == int(meta["n_senate"])


# ── parity ──────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("p", _ORDERS)
@pytest.mark.parametrize("design", _DESIGNS)
def test_statistic_and_pvalue_match_r(rjson, designs, design, p):
    ref = rjson[f"{design}_p{p}"]
    res = sp.rddensity(designs[design], x="x", c=0, p=p)
    assert float(res.estimate) == pytest.approx(ref["T_jk"], rel=RTOL)
    assert float(res.pvalue) == pytest.approx(ref["p_jk"], rel=RTOL)


@pytest.mark.parametrize("p", _ORDERS)
@pytest.mark.parametrize("design", _DESIGNS)
def test_bandwidths_match_r(rjson, designs, design, p):
    ref = rjson[f"{design}_p{p}"]
    mi = sp.rddensity(designs[design], x="x", c=0, p=p).model_info
    assert mi["bandwidth_left"] == pytest.approx(ref["hl"], rel=RTOL)
    assert mi["bandwidth_right"] == pytest.approx(ref["hr"], rel=RTOL)


@pytest.mark.parametrize("p", _ORDERS)
@pytest.mark.parametrize("design", _DESIGNS)
def test_density_estimates_match_r(rjson, designs, design, p):
    """The one-sided densities, not just the test built on them.

    A sign error or a swapped side can leave the two-sided statistic intact
    while both densities are wrong, so these are pinned separately.
    """
    ref = rjson[f"{design}_p{p}"]
    mi = sp.rddensity(designs[design], x="x", c=0, p=p).model_info
    assert mi["density_left"] == pytest.approx(ref["fl"], rel=RTOL)
    assert mi["density_right"] == pytest.approx(ref["fr"], rel=RTOL)
    assert mi["density_diff"] == pytest.approx(ref["fr"] - ref["fl"], rel=RTOL)


@pytest.mark.parametrize("design", _DESIGNS)
def test_effective_sample_sizes_match_r(rjson, designs, design):
    """Guards against a right answer reached on the wrong window.

    ``n_left``/``n_right`` are the full-side counts; ``n_eff_*`` are the
    observations inside the bandwidth, which is what R reports as
    ``N$eff_*`` and what a paper's table should carry. On the senate data
    the two differ by a factor of five, so conflating them misstates the
    sample by a lot. ``n_eff_*`` was added when this test first ran.
    """
    ref = rjson[f"{design}_p2"]
    mi = sp.rddensity(designs[design], x="x", c=0, p=2).model_info
    assert int(mi["n_eff_left"]) == int(ref["Nl"])
    assert int(mi["n_eff_right"]) == int(ref["Nr"])
    assert mi["n_eff_left"] <= mi["n_left"]
    assert mi["n_eff_right"] <= mi["n_right"]


# ── behaviour, no reference needed ─────────────────────────────────────── #


def test_verdict_direction_is_right(designs):
    """The whole point of the test: reject under manipulation, not otherwise."""
    p_null = float(sp.rddensity(designs["null"], x="x", c=0).pvalue)
    p_manip = float(sp.rddensity(designs["manip"], x="x", c=0).pvalue)
    assert p_null > 0.05, f"false positive on a smooth density (p={p_null})"
    assert p_manip < 0.05, f"missed a 45% hole at the cutoff (p={p_manip})"
