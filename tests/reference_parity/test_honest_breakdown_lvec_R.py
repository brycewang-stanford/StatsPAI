"""``sp.breakdown_m(l_vec=, window=)`` vs R ``HonestDiD`` 0.2.8.

The breakdown value for a target ``l' tau_post`` (e.g. the average
post-treatment effect) is the first ``M`` at which the set
``sp.honest_did(..., l_vec=)`` reports contains zero. HonestDiD has no
breakdown function, so its sensitivity table at multiples of the native
value is read instead: the Conditional set (deterministic) excludes zero at
0.997 ``M*`` and contains it at 1.003 ``M*``; the C-LF and FLCI sets, whose
critical values are simulated (R's draws against NumPy's), at 0.97 and
1.03 ``M*`` (a stochastic screen, not exact parity).

Cases: the castle-doctrine event study of the 2026-10 replication
(Callaway-Sant'Anna with population weights, average of e = 0..4) under
``Delta^RM`` (C-LF, Conditional) and ``Delta^SD`` (FLCI), the same average
on ``window=(-3, 2)``, and ``corr42`` of ``honest_rm_R.json``. The castle
C-LF value, 0.406, is the "about 0.4" read off a fine ``Mbar`` grid by hand
in that replication (Stata ``honestdid``: 0.40 excludes zero, 0.45 does not).

Fixtures: ``_fixtures/_generate_honest_breakdown_lvec.py`` then
``_generate_honest_breakdown_lvec.R``.
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
    (_FIX / "honest_breakdown_lvec_inputs.json").read_text(encoding="utf-8")
)
REF = json.loads((_FIX / "honest_breakdown_lvec_R.json").read_text(encoding="utf-8"))
CASES = sorted(INP)


def _res(c):
    return event_study_from_moments(
        np.asarray(c["b"], float), np.asarray(c["S"], float), event_times=c["times"]
    )


def _kw(c):
    return dict(
        method=c["method"],
        honestdid_method=c["hm"],
        l_vec=c["l_vec"],
        window=None if c["window"] is None else tuple(c["window"]),
    )


@pytest.mark.parametrize("key", CASES)
def test_honestdid_brackets_the_native_value(key):
    r = REF[key]
    for mult, lb, ub in zip(r["mult"], r["lb"], r["ub"]):
        assert (lb <= 0 <= ub) == (mult > 1), (key, mult, lb, ub)


@pytest.mark.parametrize(
    "key",
    [
        # the Conditional scans take minutes; the castle C-LF case is the
        # headline value of the replication and stays in the default run
        k if k == "castle_avg_rm_clf" else pytest.param(k, marks=pytest.mark.slow)
        for k in CASES
    ],
)
def test_native_value_and_its_defining_property(key):
    c = INP[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_star = sp.breakdown_m(_res(c), **_kw(c))
        table = sp.honest_did(
            _res(c), m_grid=[0.997 * m_star, 1.003 * m_star], **_kw(c)
        )
    assert m_star == pytest.approx(c["mstar"], rel=1e-9)
    assert list(table["rejects_zero"]) == [True, False]
