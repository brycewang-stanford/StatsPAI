"""Combination identities for the CCT ``comb`` bandwidth selectors.

Track A module 88 pins ``sp.rdbwselect`` numerically against
``rdrobust::rdbwselect`` and the Stata ado. This file pins the *rule*
instead, and does it without R in the loop:

    comb1 = min(rd, sum)                elementwise, per side
    comb2 = median(two, rd, sum)        elementwise, per side

The distinction matters because of how the defect these tests guard
against actually behaved. ``msecomb1`` / ``msecomb2`` / ``cercomb1`` /
``cercomb2`` were not implemented at all -- they fell through to the
plain ``rd`` cascade -- and ``comb1`` still agreed with R on the Lee
2008 senate replica, because ``rd`` happens to be the smaller of the
pair there. A numerical fixture on one dataset can therefore certify a
``comb1`` that is not computing ``comb1``. An identity holds on every
dataset, so these tests fail wherever the fixture would not.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.rd._cct_bandwidth import cct_bandwidth

QUANTITIES = ("h_left", "h_right", "b_left", "b_right")


def _panel(seed: int, n: int = 1200):
    """A deterministic RD design with a jump at zero."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.6 * x + 1.5 * x**2 + 2.0 * (x >= 0) + rng.normal(0.0, 0.4, n)
    return y, x


def _bw(y, x, method: str, **kw):
    return cct_bandwidth(y, x, c=0.0, bwselect=method, **kw)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 11])
@pytest.mark.parametrize("prefix", ["mse", "cer"])
def test_comb1_is_the_elementwise_minimum(seed: int, prefix: str) -> None:
    """``comb1`` is min(rd, sum) on every quantity and every side."""
    y, x = _panel(seed)
    rd = _bw(y, x, f"{prefix}rd")
    sm = _bw(y, x, f"{prefix}sum")
    comb1 = _bw(y, x, f"{prefix}comb1")
    for q in QUANTITIES:
        assert comb1[q] == pytest.approx(min(rd[q], sm[q]), rel=0, abs=0), q


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 11])
@pytest.mark.parametrize("prefix", ["mse", "cer"])
def test_comb2_is_the_elementwise_median(seed: int, prefix: str) -> None:
    """``comb2`` is median(two, rd, sum) on every quantity and every side."""
    y, x = _panel(seed)
    two = _bw(y, x, f"{prefix}two")
    rd = _bw(y, x, f"{prefix}rd")
    sm = _bw(y, x, f"{prefix}sum")
    comb2 = _bw(y, x, f"{prefix}comb2")
    for q in QUANTITIES:
        expected = float(np.median([two[q], rd[q], sm[q]]))
        assert comb2[q] == pytest.approx(expected, rel=0, abs=0), q


@pytest.mark.parametrize("prefix", ["mse", "cer"])
def test_comb2_is_reachable_and_distinct_from_rd(prefix: str) -> None:
    """At least one seed must separate ``comb2`` from the ``rd`` cascade.

    The regression being guarded resolved every comb variant to ``rd``.
    A suite in which ``comb2 == rd`` on all fixtures would pass against
    the broken implementation too, so this asserts the discriminating
    case exists rather than assuming it.
    """
    separated = False
    for seed in range(12):
        y, x = _panel(seed)
        if (
            _bw(y, x, f"{prefix}comb2")["h_right"]
            != _bw(y, x, f"{prefix}rd")["h_right"]
        ):
            separated = True
            break
    assert separated, (
        f"no fixture separates {prefix}comb2 from {prefix}rd; these tests "
        "would not detect the fall-through they exist to catch"
    )


def test_public_selector_exposes_all_ten_methods() -> None:
    """``sp.rdbwselect(all=True)`` returns every selector rdrobust offers.

    ``msesum`` and ``cersum`` were missing from ``_VALID_METHODS``, which
    made the two sum-form cascades that ``comb1`` / ``comb2`` are built
    from unreachable from the public API.
    """
    df = sp.datasets.lee_2008_senate()
    out = sp.rdbwselect(df, y="y", x="x", c=0.0, all=True)
    assert list(out["method"]) == [
        "mserd",
        "msetwo",
        "msesum",
        "msecomb1",
        "msecomb2",
        "cerrd",
        "certwo",
        "cersum",
        "cercomb1",
        "cercomb2",
    ]
    for method in ("msesum", "cersum"):
        single = sp.rdbwselect(df, y="y", x="x", c=0.0, bwselect=method)
        assert len(single) == 1 and single.iloc[0]["method"] == method


def test_selector_output_is_not_rounded() -> None:
    """Bandwidths come back at full precision, not rounded to 6 decimals.

    A bandwidth is an input to the next estimator: rounding it capped
    downstream agreement at ~1e-6 relative and perturbed
    ``sp.rdrobust(h=...)`` when a user fed one back in.
    """
    df = sp.datasets.lee_2008_senate()
    h = float(sp.rdbwselect(df, y="y", x="x", c=0.0).iloc[0]["h_left"])
    assert h != round(h, 6), "selector output still looks rounded"
    # And it is the same number the estimator picks for itself.
    direct = cct_bandwidth(df["y"].to_numpy(float), df["x"].to_numpy(float), c=0.0)[
        "h_left"
    ]
    assert h == pytest.approx(direct, rel=1e-15)


def test_selector_agrees_with_the_estimators_own_default_h() -> None:
    """The published selector and ``sp.rdrobust`` cannot disagree.

    They ran different algorithms until Track A module 88: the selector
    returned h = 4.63 where the estimator used 17.75 on this fixture, so
    a user who read a bandwidth from one and passed it to the other got a
    materially different estimate with nothing to signal the mismatch.
    """
    df = sp.datasets.lee_2008_senate()
    h = float(sp.rdbwselect(df, y="y", x="x", c=0.0).iloc[0]["h_left"])
    fit_default = sp.rdrobust(df, y="y", x="x", c=0.0)
    fit_forced = sp.rdrobust(df, y="y", x="x", c=0.0, h=h, b=h)
    forced_at_default_b = sp.rdrobust(
        df,
        y="y",
        x="x",
        c=0.0,
        h=h,
        b=float(sp.rdbwselect(df, y="y", x="x", c=0.0).iloc[0]["b_left"]),
    )
    assert forced_at_default_b.model_info["conventional"]["estimate"] == pytest.approx(
        fit_default.model_info["conventional"]["estimate"], rel=1e-10
    )
    # Sanity: the forced fit is a real fit, not a degenerate one.
    assert np.isfinite(fit_forced.model_info["conventional"]["estimate"])
