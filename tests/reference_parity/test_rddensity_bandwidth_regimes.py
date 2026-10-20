"""Reference parity: ``sp.rddensity``'s default bandwidths where the
sample-size rules bind, vs R ``rddensity`` 2.6 / ``rdbwdensity``.

``test_rddensity_parity.py`` pins the default path on three well-behaved
designs. The selector also has corner rules (a pilot or final bandwidth
is widened to hold ``20 + p + 1`` observations and distinct values per
side, capped at the farthest observation; a bias constant that cannot be
estimated sends the bandwidth to that bound). The clean-room rewrite of
2026-10-20 found those rules by black-box probing, so this file pins them
against ``rdbwdensity_R.json`` (``_fixtures/_generate_rdbwdensity_R.R``):
five designs (a hole around the cutoff, 15 observations on one side, two
mass points, nine support points, a non-zero cutoff) x ``p`` in 1-3.

Measured 2026-10-20: bandwidths and variance constants agree to 6e-10 or
better, squared bias constants to 4e-9, densities and statistics to 2e-8
(``gap_p3``, an order-4 fit on a sparse window). ``RTOL`` is set from
those numbers.

References
----------
cattaneo2018manipulation
"""

from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL_BW = 1e-8
RTOL_FIT = 1e-7
_TARGETS = ("l", "r", "diff", "sum")


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "rdbwdensity_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def designs():
    return pd.read_csv(_FIX / "rdbwdensity_designs.csv")


def _module():
    return sys.modules["statspai.diagnostics.rddensity"]


def _cells(ref, with_error=False):
    keys = [k for k in ref if not k.startswith("_")]
    return [k for k in keys if ("r_error" in ref[k]) == with_error]


def _x(designs, cell):
    return designs.loc[designs["design"] == cell.rsplit("_p", 1)[0], "x"].to_numpy()


def _f(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


CELLS = [
    f"{d}_p{p}"
    for d in ("gap", "tinyleft", "heap", "integer", "shifted")
    for p in (1, 2, 3)
]


@pytest.mark.parametrize("cell", CELLS)
def test_bandwidth_table_matches_rdbwdensity(ref, designs, cell):
    want = ref[cell]
    if "r_error" in want:
        pytest.skip(f"R stops on this cell: {want['r_error']}")
    x = _x(designs, cell)
    table = _module()._bandwidth_table(np.sort(x - want["c"]), want["p"])
    got = np.array([table[t] for t in _TARGETS], dtype=float)
    np.testing.assert_allclose(got[:, 0], want["bw"], rtol=RTOL_BW)
    np.testing.assert_allclose(got[:, 1], want["variance"], rtol=RTOL_BW)
    np.testing.assert_allclose(
        got[:, 2], _f(want["biassq"]), rtol=RTOL_FIT, equal_nan=True
    )


@pytest.mark.parametrize("cell", CELLS)
def test_default_test_matches_rddensity(ref, designs, cell):
    want = ref[cell]
    if "r_error" in want:
        pytest.skip(f"R stops on this cell: {want['r_error']}")
    res = sp.rddensity(
        pd.DataFrame({"x": _x(designs, cell)}), x="x", c=want["c"], p=want["p"]
    )
    mi = res.model_info
    assert mi["bandwidth_left"] == pytest.approx(want["h_left"], rel=RTOL_BW)
    assert mi["bandwidth_right"] == pytest.approx(want["h_right"], rel=RTOL_BW)
    assert (mi["n_eff_left"], mi["n_eff_right"]) == (
        want["N_eff_left"],
        want["N_eff_right"],
    )
    if want["T_jk"] is None:
        # R returns NA: one side's fit is not identified on its window.
        # sp.rddensity marks the densities NaN and reports T = 0, p = 1.
        assert np.isnan(mi["density_left"]) or np.isnan(mi["density_right"])
        return
    assert mi["density_left"] == pytest.approx(want["f_left"], rel=RTOL_FIT)
    assert mi["density_right"] == pytest.approx(want["f_right"], rel=RTOL_FIT)
    assert float(res.estimate) == pytest.approx(want["T_jk"], rel=RTOL_FIT)


def test_fixture_covers_the_corner_rules(ref, designs):
    """Each corner rule must actually bind somewhere in the fixture."""
    from statspai.rd._lpdensity import kth_distance, normal_reference_bandwidth

    def bounds(cell, k):
        d = _x(designs, cell) - ref[cell]["c"]
        sides = (np.sort(-d[d < 0]), np.sort(d[d >= 0]))
        lower = max(kth_distance(s, k) for s in sides)
        return lower, max(s[-1] for s in sides)

    # fewer observations below the cutoff than the minimum local sample
    assert int((_x(designs, "tinyleft_p1") < 0).sum()) < 20 + 1 + 1
    # the variance pilot is widened (tinyleft) and capped (gap)
    x = _x(designs, "tinyleft_p1")
    assert normal_reference_bandwidth(x, 0.0, 1, 1) < bounds("tinyleft_p1", 22)[0]
    x = _x(designs, "gap_p1")
    assert normal_reference_bandwidth(x, 0.0, 1, 1) > bounds("gap_p1", 22)[1]
    # the bias constant is not estimable on the integer design
    assert all(v is None for v in ref["integer_p2"]["biassq"])


@pytest.mark.parametrize(
    "P, v, want",
    [
        # normal-reference pilot for the variance constant (P = p, v = 1)
        (1, 1, Fraction(12, 5)),
        (2, 1, Fraction(960, 7)),
        (3, 1, Fraction(16800)),
        (4, 1, Fraction(203212800, 55)),
        # pilot for the derivative F^(p+1) (P = p + 2, v = p + 1)
        (3, 2, Fraction(58240, 3)),
        (4, 3, Fraction(47174400, 11)),
    ],
)
def test_pilot_constants_reproduce_the_reference(P, v, want):
    """``h^(2P+1) = k f(c) / (F^(P+1)(c)^2 n)`` under a normal reference.

    The constants ``k`` were backed out of rdbwdensity's variance and bias
    columns on normal samples (black box, 2026-10-20) and identified as
    the uniform-kernel boundary MSE constants. This pins the closed form
    to those measured values.
    """
    from statspai.rd._lpdensity import _uniform_boundary_constants

    V, B = _uniform_boundary_constants(P, v)
    k = (2 * v - 1) * V / (2 * (P + 1 - v) * B * B)
    assert k == pytest.approx(float(want), rel=1e-12)
