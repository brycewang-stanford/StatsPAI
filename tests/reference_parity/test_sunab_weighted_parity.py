"""Weighted Sun-Abraham against ``fixest::sunab`` (fixest 0.14.0).

What this module does and does not claim
----------------------------------------
It claims **point-estimate** parity for the interaction-weighted
event-study coefficients, weighted and unweighted, at the same
precision. That is the validation that matters for the weighting work:
under omega the fixed-effect projection has to be carried out in the
weighted inner product, and a wrong weighted demeaning shows up
immediately in the coefficients.

It also claims **standard-error** parity under the fixest convention.
``fixest::sunab`` treats the cohort shares as fixed when it aggregates
cohort-by-relative-time coefficients; Sun & Abraham (2021, Prop. 3) and
Stata ``eventstudyinteract`` add the share-estimation term
``β' Var(ŵ) β``. StatsPAI's default ``share_variance=True`` follows the
authors' implementation (pinned to Stata at 8e-12 by parity module 05);
``share_variance=False`` reproduces ``fixest`` at machine level, weighted
and unweighted, which is what is asserted here. Before 1.24.0 the
degrees-of-freedom factor counted unobserved cohort-by-relative-time
cells and omitted the non-nested time effects, which is why the two
implementations appeared to differ by 0.7% (unweighted) to 2.2%
(weighted) even at single-cohort relative times; both defects are fixed
and the only remaining difference is the documented share term.

Reference command::

    feols(y ~ sunab(gg, t) | i + t, data = d, weights = ~w, cluster = ~i)

with ``gg = ifelse(g == 0, 10000, g)`` (fixest marks never-treated with a
large cohort label rather than 0). Reference values are transcribed from
that call; the panel is the shared weighted-CS fixture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

_PANEL = Path(__file__).parent / "_fixtures" / "cs_weighted_panel.csv"

# fixest 0.14.0, per-relative-time coefficients: {weighted: {e: (att, se)}}
_FIXEST = {
    False: {
        -4: (0.0123845803, 0.0733885504),
        -3: (-0.0171282768, 0.0528206700),
        -2: (-0.0015857186, 0.0332713060),
        0: (0.3456846242, 0.1477075771),
        1: (0.3760631347, 0.1479370124),
        2: (0.3503349955, 0.1514708582),
        3: (0.2735878522, 0.1821237947),
        4: (0.0185695467, 0.2545341329),
    },
    True: {
        -4: (-0.0955866937, 0.1276934169),
        -3: (0.0132461133, 0.0793115597),
        -2: (-0.0163856468, 0.0581330223),
        0: (3.3688885271, 0.0939691591),
        1: (3.4008760967, 0.0969477714),
        2: (3.4094340769, 0.1048072291),
        3: (3.3749429817, 0.1372471035),
        4: (3.1211036631, 0.2679689897),
    },
}

# Point estimates: both sides solve the same weighted least-squares
# problem after the same projection, so agreement is bounded only by the
# alternating-projection tolerance. Observed worst case 3e-8.
_ATT_RTOL = 1e-6

# Standard errors under share_variance=False: same sandwich, same
# nested-K small-sample factor, so agreement is again bounded by the
# projection tolerance. Observed worst case 1.2e-9.
_SE_RTOL = 1e-7
# The default share_variance=True adds a positive semi-definite term at
# multi-cohort relative times; on this fixture it is at most 0.76%
# (unweighted) / 2.2% (weighted) of the SE.
_SHARE_TERM_CEIL = {False: 1e-2, True: 2.5e-2}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL, encoding="utf-8")


def _event_study(result) -> dict[int, tuple[float, float]]:
    es = result.model_info["event_study"]
    col = "relative_time" if "relative_time" in es.columns else "rel_time"
    ac = "att" if "att" in es.columns else "estimate"
    return {int(r[col]): (float(r[ac]), float(r["se"])) for _, r in es.iterrows()}


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "omega"])
def test_event_study_point_estimates_match_fixest(panel, weighted):
    r = sp.sun_abraham(
        panel, y="y", g="g", t="t", i="i", weights="w" if weighted else None
    )
    got = _event_study(r)
    for e, (ref_att, _) in _FIXEST[weighted].items():
        assert e in got, f"event time {e} missing"
        assert got[e][0] == pytest.approx(
            ref_att, rel=_ATT_RTOL
        ), f"e={e}: {got[e][0]} vs fixest {ref_att}"


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "omega"])
def test_event_study_standard_errors_match_fixest_under_fixed_shares(panel, weighted):
    """share_variance=False reproduces fixest's SEs at machine level."""
    r = sp.sun_abraham(
        panel,
        y="y",
        g="g",
        t="t",
        i="i",
        weights="w" if weighted else None,
        share_variance=False,
    )
    got = _event_study(r)
    for e, (_, ref_se) in _FIXEST[weighted].items():
        assert got[e][1] == pytest.approx(
            ref_se, rel=_SE_RTOL
        ), f"e={e}: fixed-share SE {got[e][1]:.10f} vs fixest {ref_se:.10f}"


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "omega"])
def test_default_share_term_is_positive_and_bounded(panel, weighted):
    """The Prop. 3 share term only ever adds variance, and stays small."""
    r = sp.sun_abraham(
        panel, y="y", g="g", t="t", i="i", weights="w" if weighted else None
    )
    es = r.detail
    for _, row in es.iterrows():
        e = int(row["relative_time"])
        ref_se = _FIXEST[weighted][e][1]
        if int(row["n_cohorts"]) == 1:
            assert row["se"] == pytest.approx(ref_se, rel=_SE_RTOL), f"e={e}"
        else:
            assert (
                ref_se < row["se"] <= ref_se * (1 + _SHARE_TERM_CEIL[weighted])
            ), f"e={e}: default SE {row['se']:.10f} vs fixest {ref_se:.10f}"


def test_weighting_moves_the_estimates(panel):
    """Guard against omega being dropped again on this path."""
    unw = sp.sun_abraham(panel, y="y", g="g", t="t", i="i")
    wtd = sp.sun_abraham(panel, y="y", g="g", t="t", i="i", weights="w")
    assert abs(wtd.estimate - unw.estimate) > 1.0


def test_constant_weights_reduce_to_unweighted(panel):
    """omega == c must be a no-op through projection, solve and aggregation."""
    unw = sp.sun_abraham(panel, y="y", g="g", t="t", i="i")
    const = sp.sun_abraham(panel.assign(w=3.0), y="y", g="g", t="t", i="i", weights="w")
    assert const.estimate == pytest.approx(unw.estimate, rel=1e-9)
    assert const.se == pytest.approx(unw.se, rel=1e-9)


def test_scale_invariance(panel):
    a = sp.sun_abraham(panel, y="y", g="g", t="t", i="i", weights="w")
    b = sp.sun_abraham(
        panel.assign(w=panel["w"] * 1e6), y="y", g="g", t="t", i="i", weights="w"
    )
    assert b.estimate == pytest.approx(a.estimate, rel=1e-9)


def test_dispatcher_forwards_weights(panel):
    unw = sp.did(panel, y="y", treat="g", time="t", id="i", method="sa")
    wtd = sp.did(panel, y="y", treat="g", time="t", id="i", method="sa", weights="w")
    assert abs(wtd.estimate - unw.estimate) > 1.0
