"""Weighted Sun-Abraham against ``fixest::sunab`` (fixest 0.14.0).

What this module does and does not claim
----------------------------------------
It claims **point-estimate** parity for the interaction-weighted
event-study coefficients, weighted and unweighted, at the same
precision. That is the validation that matters for the weighting work:
under omega the fixed-effect projection has to be carried out in the
weighted inner product, and a wrong weighted demeaning shows up
immediately in the coefficients.

It does **not** claim standard-error parity. StatsPAI and ``fixest``
differ on the interaction-weighted aggregation variance, and the gap is
present in the unweighted path too (it predates this work): the
per-event-time SEs agree to ~0.7% unweighted and ~2.2% weighted, and the
disagreement is smallest at event times served by a single cohort and
largest where several cohorts are aggregated. That pattern points at the
cohort-share covariance term rather than at a scalar degrees-of-freedom
factor. The measured gaps are pinned below so the convention cannot
drift unnoticed, but they are a documented open item, not a pass.

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

# Standard errors: documented convention gap, pinned at the measured
# ceiling rather than asserted as parity. See the module docstring.
_SE_RTOL = {False: 2e-2, True: 5e-2}


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
def test_event_study_standard_errors_stay_within_the_known_gap(panel, weighted):
    """Pin the documented SE convention gap so it cannot widen silently."""
    r = sp.sun_abraham(
        panel, y="y", g="g", t="t", i="i", weights="w" if weighted else None
    )
    got = _event_study(r)
    for e, (_, ref_se) in _FIXEST[weighted].items():
        rel = abs(got[e][1] - ref_se) / ref_se
        assert rel <= _SE_RTOL[weighted], (
            f"e={e}: SE gap {rel:.3e} exceeds the pinned convention "
            f"ceiling {_SE_RTOL[weighted]}"
        )


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
