"""``sp.breakdown_m(method='relative_magnitude')`` (C-LF, the default) vs
R ``HonestDiD`` 0.2.8.

The breakdown value is the first ``Mbar`` at which the C-LF
``Delta^RM(Mbar)`` set contains zero, found by the log-grid scan and
bisection used for every other restriction (the test-inverted set need not
be nested in ``Mbar``; through 1.39 a doubling bracket was bisected).
HonestDiD has no breakdown function, so its C-LF sensitivity table at
multiples of the native ``Mbar*`` is read instead: it excludes zero at
0.5, 0.9 and 0.97 ``Mbar*`` and contains it at 1.03, 1.1 and 1.5 ``Mbar*``
on six cases. C-LF calibrates a least-favourable critical value by
simulation (R's RNG against NumPy's), so this is a stochastic screen with
a 3% bracket, not exact parity.

Fixtures: ``_fixtures/_generate_honest_breakdown_rm_clf.py`` (native
``Mbar*`` and inputs) then ``_generate_honest_breakdown_rm_clf.R``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.did.honest_did import event_study_from_moments

_FIX = pathlib.Path(__file__).parent / "_fixtures"
INP = json.loads(
    (_FIX / "honest_breakdown_rm_clf_inputs.json").read_text(encoding="utf-8")
)
REF = json.loads((_FIX / "honest_breakdown_rm_clf_R.json").read_text(encoding="utf-8"))
CASES = sorted(INP)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("key", CASES)
def test_native_value_and_honestdid_bracket(key):
    c = INP[key]
    res = event_study_from_moments(
        np.asarray(c["b"], float), np.asarray(c["S"], float), event_times=c["times"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_star = sp.breakdown_m(res, e=c["e"], method="relative_magnitude")
    assert m_star == pytest.approx(c["mstar"], rel=1e-9)
    r = REF[key]
    for mult, lb, ub in zip(r["mult"], r["lb"], r["ub"]):
        contains = lb <= 0 <= ub
        assert contains == (mult > 1), (key, mult, lb, ub)
