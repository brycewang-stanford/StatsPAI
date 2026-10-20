"""``sp.breakdown_m`` under HonestDiD's sign, shape and SDRM restrictions.

R ``HonestDiD`` 0.2.8 has no breakdown function; its sensitivity table is
read off instead. With ``method = "Conditional"`` (no simulation) the native
sets accept the same grid points as HonestDiD
(``test_honest_restrictions_R_parity.py``), so HonestDiD's table on a grid of
``M`` around the native breakdown value ``M*`` must exclude zero at every
``M`` below it and contain zero just above it:

* at ``0, 0.5 M*, 0.9 M*, 0.997 M*`` HonestDiD's set excludes zero;
* at ``1.003 M*`` it contains zero (``M*`` is bisected to 1e-3 relative).

Where ``M* = inf`` (a sign or shape restriction bounds the violation on the
side that matters), HonestDiD's set excludes zero on the whole grid. The sets
are not nested in ``M`` under these restrictions -- HonestDiD's own
``Delta^RMNB`` set flips between containing and excluding zero just above
the crossing -- so ``M*`` is the *first* ``M`` at which the set contains
zero; above ``1.003 M*`` nothing is asserted.

The native value is also checked against ``sp.honest_did`` on the same
input: ``rejects_zero`` is True just below ``M*`` and False just above.

Fixture: ``_fixtures/honest_breakdown_R.json`` from
``_fixtures/_generate_honest_breakdown_R.sh``.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.did.honest_did import event_study_from_moments

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "honest_breakdown_R.json").read_text(encoding="utf-8"))
INP = REF["_inputs"]
CASES = sorted(k for k in REF if not k.startswith("_"))

pytestmark = pytest.mark.slow

_KW = {
    "SD": dict(method="smoothness"),
    "SDPB": dict(method="smoothness", bias_direction="positive"),
    "SDNB": dict(method="smoothness", bias_direction="negative"),
    "SDD": dict(method="smoothness", monotonicity_direction="decreasing"),
    "RM": dict(method="relative_magnitude"),
    "RMPB": dict(method="relative_magnitude", bias_direction="positive"),
    "RMNB": dict(method="relative_magnitude", bias_direction="negative"),
    "RMI": dict(method="relative_magnitude", monotonicity_direction="increasing"),
    "SDRM": dict(method="smoothness_relative_magnitude"),
    "SDRMPB": dict(method="sdrm", bias_direction="positive"),
}


def _result():
    return event_study_from_moments(
        np.asarray(INP["betahat"], float),
        np.asarray(INP["sigma"], float),
        event_times=INP["event_times"],
    )


@pytest.fixture(scope="module")
def native():
    res = _result()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key in CASES:
            out[key] = sp.breakdown_m(
                res, e=0, honestdid_method="Conditional", **_KW[key]
            )
    return out


def _excludes(lb, ub):
    return lb is not None and not (lb <= 0 <= ub)


@pytest.mark.parametrize("key", CASES)
def test_honestdid_table_brackets_the_breakdown(key, native):
    c = REF[key]
    m_star = native[key]
    rows = list(zip(c["M"], c["lb"], c["ub"]))
    if math.isinf(m_star):
        assert all(_excludes(lb, ub) for _, lb, ub in rows), (key, rows)
        return
    below = [r for r in rows if r[0] <= 0.9971 * m_star]
    above = [r for r in rows if m_star < r[0] <= 1.0031 * m_star]
    assert len(below) == 4 and len(above) == 1, (key, m_star, rows)
    assert all(_excludes(lb, ub) for _, lb, ub in below), (key, m_star, below)
    assert not _excludes(above[0][1], above[0][2]), (key, m_star, above)


@pytest.mark.parametrize("key", ["SDPB", "RMNB", "SDRM"])
def test_breakdown_is_where_honest_did_stops_rejecting(key, native):
    """The breakdown value is a property of the set ``sp.honest_did`` reports."""
    m_star = native[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = sp.honest_did(
            _result(),
            e=0,
            honestdid_method="Conditional",
            m_grid=[0.999 * m_star, 1.003 * m_star],
            **_KW[key],
        )
    assert tab["rejects_zero"].tolist() == [True, False]


def test_sign_restriction_on_the_far_side_never_breaks_down(native):
    """``delta_post <= 0`` with a positive estimate: ``theta = beta - delta``
    is at least ``beta``, so no M brings zero into the set."""
    assert math.isinf(native["SDNB"]) and math.isinf(native["SDD"])
