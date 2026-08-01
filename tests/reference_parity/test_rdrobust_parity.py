"""Reference parity: ``sp.rdrobust`` / ``sp.rdbwselect`` vs R rdrobust 3.0.0.

Status: **these tests encode the CORRECT answers and currently FAIL.**

They are the test-first half of WP-1 in ``docs/rfc/rd_three_month_plan.md``.
The bandwidth selector is being rebuilt; until it lands, the parity
assertions are marked ``xfail(strict=True)`` so that

* the defect is permanent and visible in CI rather than living in a
  scratch script, and
* the fix cannot be "achieved" by loosening a tolerance -- ``strict=True``
  means these turn into failures the moment they start passing, forcing the
  xfail markers to be removed deliberately.

What is wrong (measured, ``rdrobust``'s own ``rdrobust_RDsenate``, n=1297)
--------------------------------------------------------------------------
``sp.rdrobust`` reports **12.39** on the default specification where R
reports **7.41** -- a 67% overstatement of the headline RD effect.

It decomposes into three independent defects:

A. **The MSE bandwidth is 2.8-4.8x too narrow and ignores ``p``.**
   R gives ``h = 17.75`` (p=1) and ``22.26`` (p=2) for triangular/mserd;
   StatsPAI gives ``4.633`` for *both*. Across all 36 grid cells StatsPAI
   produces only 12 distinct bandwidths, varying with the kernel alone,
   while R's vary with kernel, ``p`` and ``bwselect``. The rate exponent is
   hard-coded to ``1/5``, which is CCT's formula only when ``p == 1``; the
   general case is ``1/(2p+3)``.

B. **The treatment effect is wrong as a consequence.** 24/24 conventional
   and 23/24 robust coefficients deviate by more than 1%.

   The defect is *isolated* to bandwidth selection: forcing R's bandwidth
   (``h=17.7544``) makes StatsPAI's conventional estimate **7.4141**,
   matching R to the printed digits. The local-polynomial engine is fine.
   ``test_engine_is_correct_given_the_bandwidth`` pins that, and is NOT
   xfailed -- it passes today and must keep passing through the rebuild.

C. **The bias bandwidth ``b`` is never computed**: ``b == h`` in 36/36
   cells, where R has ``b`` substantially wider (``h=17.75`` -> ``b=28.03``).
   ``sp.rdbwselect`` does return a distinct ``b``, but ``sp.rdrobust``
   discards it. So even a correct ``h`` would leave the robust
   bias-correction wrong.

D. **``bwselect='msesum'`` / ``'cersum'`` raise ``ValueError``** -- StatsPAI
   uses ``msecomb1``/``msecomb2`` instead, so R scripts do not port.
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

# Closed-form quantities on both sides: parity should be near machine
# precision once the same formula is implemented, not merely "close".
RTOL = 1e-6

_WP1 = "WP-1: RD bandwidth selector rebuild (docs/rfc/rd_three_month_plan.md)"


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdrobust_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdrobust_R.R to build rdrobust_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate.csv")


def _specs(rjson):
    return [k for k in rjson if not k.startswith("_")]


def _scalar(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        return float(np.ravel(v)[0])
    return float(v)


# ── The part that already works: pin it so the rebuild cannot break it ── #


def test_engine_is_correct_given_the_bandwidth(rjson, senate):
    """With R's bandwidth supplied, the local-polynomial fit matches R.

    This is the evidence that the defect is confined to bandwidth
    selection. It passes today; if the WP-1 rebuild breaks it, the rebuild
    has damaged something that was already right.
    """
    ref = rjson["mserd_p1_triangular"]
    res = sp.rdrobust(
        senate,
        y="vote",
        x="margin",
        c=0,
        p=1,
        kernel="triangular",
        h=ref["h_left"],
    )
    conventional = float(res.detail["estimate"][0])
    assert conventional == pytest.approx(
        ref["coef_conventional"], rel=1e-4
    ), f"conventional {conventional} vs R {ref['coef_conventional']}"


def test_dataset_matches_the_r_side(rjson, senate):
    """Guard against blaming the estimator for a data mismatch."""
    assert len(senate) == int(rjson["_meta"]["n"])
    assert {"vote", "margin"} <= set(senate.columns)


# ── A. bandwidth selection ─────────────────────────────────────────── #


@pytest.mark.xfail(strict=True, reason=_WP1)
@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_bandwidth_matches_r(rjson, senate, kernel):
    key = f"mserd_p1_{kernel}"
    ref = rjson[key]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=1, kernel=kernel)
    h = _scalar(res.model_info["bandwidth_h"])
    assert h == pytest.approx(ref["h_left"], rel=RTOL)


@pytest.mark.xfail(strict=True, reason=_WP1)
def test_bandwidth_responds_to_polynomial_order(senate):
    """``h`` must change with ``p``: the rate exponent is ``1/(2p+3)``.

    StatsPAI currently returns 4.633 for both p=1 and p=2 (triangular),
    which is only possible if ``p`` never enters the formula. This test
    needs no R reference -- it is a property of the estimator.
    """
    h1 = _scalar(
        sp.rdrobust(
            senate, y="vote", x="margin", c=0, p=1, kernel="triangular"
        ).model_info["bandwidth_h"]
    )
    h2 = _scalar(
        sp.rdrobust(
            senate, y="vote", x="margin", c=0, p=2, kernel="triangular"
        ).model_info["bandwidth_h"]
    )
    assert h1 != pytest.approx(
        h2, rel=1e-9
    ), f"bandwidth is identical for p=1 and p=2 ({h1}); p is being ignored"


@pytest.mark.xfail(strict=True, reason=_WP1)
def test_bias_bandwidth_is_computed_and_wider(rjson, senate):
    """``b`` must be a separate, wider bandwidth -- not a copy of ``h``."""
    ref = rjson["mserd_p1_triangular"]
    mi = sp.rdrobust(
        senate, y="vote", x="margin", c=0, p=1, kernel="triangular"
    ).model_info
    h, b = _scalar(mi["bandwidth_h"]), _scalar(mi["bandwidth_b"])
    assert b != pytest.approx(h, rel=1e-12), f"b == h == {h}: b never computed"
    assert b > h, f"b ({b}) should exceed h ({h})"
    assert b == pytest.approx(ref["b_left"], rel=RTOL)


# ── B. the estimates themselves ────────────────────────────────────── #


@pytest.mark.xfail(strict=True, reason=_WP1)
@pytest.mark.parametrize("p", [1, 2])
def test_conventional_coefficient_matches_r(rjson, senate, p):
    ref = rjson[f"mserd_p{p}_triangular"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, kernel="triangular")
    got = float(res.detail["estimate"][0])
    assert got == pytest.approx(ref["coef_conventional"], rel=RTOL)


@pytest.mark.xfail(strict=True, reason=_WP1)
def test_robust_coefficient_and_se_match_r(rjson, senate):
    ref = rjson["mserd_p1_triangular"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=1, kernel="triangular")
    got = float(res.detail["estimate"][1])
    got_se = float(res.detail["se"][1])
    assert got == pytest.approx(ref["coef_robust"], rel=RTOL)
    assert got_se == pytest.approx(ref["se_robust"], rel=RTOL)


@pytest.mark.xfail(strict=True, reason=_WP1)
def test_headline_estimate_is_not_inflated(rjson, senate):
    """The single number a user sees on the canonical dataset.

    R: 7.41. StatsPAI today: 12.39.
    """
    ref = rjson["mserd_p1_triangular"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0)
    assert float(res.estimate) == pytest.approx(ref["coef_robust"], rel=1e-3)


@pytest.mark.xfail(strict=True, reason=_WP1)
def test_full_grid_agrees_with_r(rjson, senate):
    """All 36 cells at once, so a partial fix cannot look complete."""
    failures = []
    for key in _specs(rjson):
        ref = rjson[key]
        try:
            res = sp.rdrobust(
                senate,
                y="vote",
                x="margin",
                c=0,
                p=ref["p"],
                kernel=ref["kernel"],
                bwselect=ref["bwselect"],
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{key}: raised {type(exc).__name__}: {exc}")
            continue
        h = _scalar(res.model_info["bandwidth_h"])
        conv = float(res.detail["estimate"][0])
        if h != pytest.approx(ref["h_left"], rel=RTOL):
            failures.append(f"{key}: h {h:.4f} vs R {ref['h_left']:.4f}")
        if conv != pytest.approx(ref["coef_conventional"], rel=RTOL):
            failures.append(
                f"{key}: conv {conv:.4f} vs R {ref['coef_conventional']:.4f}"
            )
    assert not failures, (
        f"{len(failures)} mismatches across {len(_specs(rjson))} specs:\n  "
        + "\n  ".join(failures[:12])
    )


# ── D. API compatibility with R ────────────────────────────────────── #


@pytest.mark.xfail(strict=True, reason=_WP1)
@pytest.mark.parametrize("bwselect", ["msesum", "cersum"])
def test_r_bwselect_names_are_accepted(senate, bwselect):
    """R's six MSE/CER variants must all be callable by their R names.

    StatsPAI currently exposes ``msecomb1``/``msecomb2`` and rejects
    ``msesum``/``cersum``, so an R script does not port across.
    """
    sp.rdrobust(
        senate,
        y="vote",
        x="margin",
        c=0,
        kernel="triangular",
        bwselect=bwselect,
    )
