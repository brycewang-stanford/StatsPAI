"""Reference parity: ``sp.rd_honest`` vs R RDHonest 1.0.1.9000 (WP-6).

``sp.rd_honest`` produced the right point estimate but built its interval
from two approximations that are not the Armstrong-Kolesar construction:

1. **The worst-case bias was a closed form**, ``M h^2 C_kernel``. The honest
   bias depends on the *realised* kernel weights, so no function of ``h``
   alone can be it. It came out **1.58-1.60x** too large -- and the ratio
   moved with ``h``, the signature of a quantity that is design-dependent on
   one side of the comparison and not the other.

2. **The interval double-counted the bias**::

       honest_ci = tau +/- (cv * se + bias)      # was
       honest_ci = tau +/- cv * se               # correct

   ``cv = cv_{1-alpha}(bias/se)`` already accounts for the bias; adding it
   again on top is not the procedure. At ``bias/se = 1.3`` the correct
   half-length is ``2.95*se`` where the old form gave ``6.24*se``.

Together the reported intervals were **1.67-1.75x wider** than RDHonest's.
Being too wide is not a safe failure here: it is a different, less
informative procedure reported under Armstrong-Kolesar's name, and it
understates what the data support.

The p-value was also the naive ``2*(1 - Phi(|tau|/se))``, which ignores the
very bias the rest of the function exists to bound -- so a result could
show an honest CI containing zero next to a p-value below 0.05. It is now
obtained by inverting the honest interval.

Structure: cells with **both** ``M`` and ``h`` fixed isolate the CI formula
from the selectors; cells with ``M`` fixed and ``h`` chosen isolate the
bandwidth rule; the free cells exercise the whole chain including the rule
of thumb for ``M``. All 18 agree with R to <= 7.6e-08, the residual being
optimiser tolerance on a deliberately flat objective.

References
----------
armstrong2018optimal, armstrong2020simple
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

# Fixed M and h: closed forms on both sides, so machine precision is the bar.
RTOL_FIXED = 1e-9
# Selected h: both sides run a numerical optimiser over a flat objective.
RTOL_SELECTED = 1e-6

_DATASETS = ["curved", "lee08"]
# The curved design lives on [-1, 1] with a genuine |f''| = 3; lee08 is a
# vote share against a margin, so both M and h need rescaling per dataset.
_SCALE = {"curved": 20.0, "lee08": 1.0}
_M_SCALE = {"curved": 1.0, "lee08": 0.01}
_M_SEL = {"curved": 3.0, "lee08": 0.02}


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdhonest_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdhonest_R.R to build rdhonest_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def designs():
    return {
        "curved": pd.read_csv(_FIX / "rdhonest_curved.csv"),
        "lee08": pd.read_csv(_FIX / "rdhonest_lee08.csv"),
    }


def _check(res, ref, rtol):
    __tracebackhide__ = True
    mi = res.model_info
    lo, hi = mi["honest_ci"]
    assert float(res.estimate) == pytest.approx(ref["estimate"], rel=rtol), "estimate"
    assert float(res.se) == pytest.approx(ref["se"], rel=rtol), "std.error"
    assert mi["bias_bound"] == pytest.approx(ref["bias"], rel=rtol), "maximum.bias"
    assert lo == pytest.approx(ref["ci_lower"], rel=rtol), "conf.low"
    assert hi == pytest.approx(ref["ci_upper"], rel=rtol), "conf.high"


# ── the primitives, isolated ────────────────────────────────────────────── #


def test_critical_value_function_matches_r(rjson):
    """``CVb``: the 1-alpha quantile of ``|N(t, 1)|``.

    Pinned separately because every interval in this file is built on it,
    and because it is the piece the old implementation had right while
    using it wrongly.
    """
    from statspai.rd._rdhonest import cv_bias

    ref = rjson["cvb"]
    for t, cv95, cv90 in zip(ref["t"], ref["cv95"], ref["cv90"]):
        assert cv_bias(t, 0.05) == pytest.approx(cv95, rel=1e-10)
        assert cv_bias(t, 0.10) == pytest.approx(cv90, rel=1e-10)


def test_critical_value_is_below_the_naive_sum():
    """``cv(t) < t + z`` for every finite ``t``: this is the whole point.

    The old interval used ``cv * se + bias``, i.e. it paid for the bias
    twice. This test states the inequality that makes that wasteful.
    """
    from statspai.rd._rdhonest import cv_bias

    z = 1.959963984540054
    for t in (0.1, 0.5, 1.0, 2.0, 5.0):
        assert cv_bias(t, 0.05) < t + z
    # and it is bounded below by the no-bias critical value
    assert cv_bias(0.0, 0.05) == pytest.approx(z, rel=1e-12)


def test_holder_bound_is_tighter_than_taylor(designs):
    """Holder allows the two sides' curvature to cancel; Taylor does not."""
    from statspai.rd._rdhonest import honest_bias, honest_weights

    x = designs["curved"]["x"].to_numpy()
    w = honest_weights(np.sort(x), 0.0, 0.25)
    xs = np.sort(x)
    h_bound = honest_bias(w, xs, 0.0, 2.0, "H")
    t_bound = honest_bias(w, xs, 0.0, 2.0, "T")
    assert 0 < h_bound < t_bound


# ── formula, with M and h both fixed ───────────────────────────────────── #


@pytest.mark.parametrize("h", [5, 10])
@pytest.mark.parametrize("M", [0.5, 2, 6])
@pytest.mark.parametrize("ds", _DATASETS)
def test_fixed_bandwidth_and_smoothness_match_r(rjson, designs, ds, M, h):
    ref = rjson[f"{ds}_fixed_M{M:g}_h{h:g}"]
    res = sp.rd_honest(
        designs[ds],
        y="y",
        x="x",
        c=0,
        M=M * _M_SCALE[ds],
        h=h / _SCALE[ds],
    )
    _check(res, ref, RTOL_FIXED)


@pytest.mark.parametrize("kern", ["uniform", "epanechnikov"])
@pytest.mark.parametrize("ds", _DATASETS)
def test_kernels_match_r(rjson, designs, ds, kern):
    ref = rjson[f"{ds}_kern_{kern}"]
    h = 0.4 if ds == "curved" else 8
    res = sp.rd_honest(designs[ds], y="y", x="x", c=0, M=_M_SEL[ds], h=h, kernel=kern)
    _check(res, ref, RTOL_FIXED)


# ── the selectors ───────────────────────────────────────────────────────── #


@pytest.mark.parametrize("crit", ["mse", "flci"])
@pytest.mark.parametrize("ds", _DATASETS)
def test_bandwidth_selection_matches_r(rjson, designs, ds, crit):
    """``OptBW``, including its three-deep preliminary-variance chain.

    ``OptBW`` -> ``PrelimVar(EHW)`` -> ``IKBW`` -> ``PrelimVar(Silverman)``.
    The objective is flat near its minimum, so a small error anywhere in
    that chain moves ``h`` by far more than it moves the objective: an
    early version was 0.9% off in ``h`` while the objective differed by
    0.018%. Matching ``h`` is therefore the sharper test.
    """
    ref = rjson[f"{ds}_bwsel_{crit.upper()}"]
    res = sp.rd_honest(designs[ds], y="y", x="x", c=0, M=_M_SEL[ds], opt_criterion=crit)
    assert res.model_info["bandwidth"] == pytest.approx(ref["h"], rel=RTOL_SELECTED)
    _check(res, ref, RTOL_SELECTED)


@pytest.mark.parametrize("ds", _DATASETS)
def test_fully_data_driven_matches_r(rjson, designs, ds):
    """Everything chosen from the data, including ``M`` via the rule of thumb."""
    ref = rjson[f"{ds}_free"]
    res = sp.rd_honest(designs[ds], y="y", x="x", c=0)
    assert res.model_info["M"] == pytest.approx(ref["M"], rel=RTOL_SELECTED)
    assert res.model_info["bandwidth"] == pytest.approx(ref["h"], rel=RTOL_SELECTED)
    _check(res, ref, RTOL_SELECTED)
    assert res.model_info["M_estimated"] is True


def test_pilot_bandwidth_is_not_clamped_to_the_data_range(designs):
    """The IK pilot may exceed the support, and RDHonest lets it.

    On the curved design (x in [-1, 1]) it comes out at 2.62. Clamping it
    to the range flattens nothing and truncates the triangular weighting,
    which shifted the preliminary variance ~5% and the selected bandwidth
    0.9%. Pinned because clamping looks like an obvious safety check.
    """
    from statspai.rd._rdhonest import _ik_bandwidth

    x = designs["curved"]["x"].to_numpy()
    y = designs["curved"]["y"].to_numpy()
    order = np.argsort(x, kind="mergesort")
    h_pilot = _ik_bandwidth(x[order], y[order], 0.0)
    assert h_pilot > np.max(np.abs(x)), (
        f"pilot bandwidth {h_pilot} no longer exceeds the support; this "
        "test no longer exercises the un-clamped path"
    )


# ── the defects this file exists to prevent ────────────────────────────── #


def test_interval_does_not_double_count_the_bias(designs):
    """``tau +/- cv*se``, not ``tau +/- (cv*se + bias)``."""
    res = sp.rd_honest(designs["curved"], y="y", x="x", c=0)
    mi = res.model_info
    lo, hi = mi["honest_ci"]
    half = (hi - lo) / 2
    assert half == pytest.approx(mi["ak_critical_value"] * float(res.se), rel=1e-10)
    # the old form, for contrast
    old = mi["ak_critical_value"] * float(res.se) + mi["bias_bound"]
    assert half < old


def test_bias_bound_is_not_a_closed_form_in_h(designs):
    """The honest bias must depend on the design, not just on ``M`` and ``h``.

    Two datasets at the same ``M`` and ``h`` must give different bias
    bounds. A closed form ``M h^2 C`` gives identical ones, which is how
    the old implementation could be 1.6x off without anything noticing.
    """
    b = [
        sp.rd_honest(designs[ds], y="y", x="x", c=0, M=1.0, h=0.5).model_info[
            "bias_bound"
        ]
        for ds in _DATASETS
    ]
    assert b[0] != pytest.approx(b[1], rel=1e-6), (
        f"identical bias bounds on two different designs ({b}); the bound "
        "has gone back to being a closed form in M and h"
    )


def test_pvalue_is_consistent_with_the_honest_interval(designs):
    """A p-value below alpha must mean the honest CI excludes zero.

    The naive ``2*(1 - Phi(|tau|/se))`` ignores the bias, so it could report
    p < 0.05 alongside an honest CI containing zero.
    """
    for ds in _DATASETS:
        for alpha in (0.01, 0.05, 0.10):
            res = sp.rd_honest(designs[ds], y="y", x="x", c=0, alpha=alpha)
            lo, hi = res.model_info["honest_ci"]
            excludes_zero = not (lo <= 0 <= hi)
            assert (
                float(res.pvalue) < alpha
            ) == excludes_zero, (
                f"{ds} alpha={alpha}: p={res.pvalue} but CI=({lo}, {hi})"
            )


def test_honest_interval_is_wider_than_the_naive_one(designs):
    """Sanity: accounting for bias cannot shrink the interval."""
    for ds in _DATASETS:
        mi = sp.rd_honest(designs[ds], y="y", x="x", c=0).model_info
        hlo, hhi = mi["honest_ci"]
        nlo, nhi = mi["naive_ci"]
        assert (hhi - hlo) > (nhi - nlo)


def test_sclass_is_validated(designs):
    with pytest.raises(ValueError, match="sclass"):
        sp.rd_honest(designs["curved"], y="y", x="x", c=0, sclass="Q")


def test_datasets_match_the_r_side(rjson, designs):
    meta = rjson["_meta"]
    assert len(designs["curved"]) == int(meta["n_curved"])
    assert len(designs["lee08"]) == int(meta["n_lee08"])
