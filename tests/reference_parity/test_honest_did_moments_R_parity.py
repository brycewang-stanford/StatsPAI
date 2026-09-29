"""``sp.honest_did_from_moments`` against R ``HonestDiD`` on raw arrays.

``sp.honest_did`` only accepted StatsPAI result objects, so an event study
from ``reghdfe`` / ``sp.hdfe_ols`` / a stacked regression / a published table
could not be stress-tested with its joint covariance (Minimum Wages, QJE
2019; Princelings, QJE 2019). ``honest_did_from_moments(betahat, sigma, ...)``
is R's ``createSensitivityResults(betahat, sigma, numPrePeriods, ...)``
interface. ``sp.honest_did_from_result`` gives the MCP tool's name a Python
counterpart (``design_audit`` already pointed users to it).

Reference: R 4.5.2, ``HonestDiD`` 0.2.8 on the package's own
``BCdata_EventStudy`` (4 pre, 4 post), from
``_fixtures/_generate_honest_did_moments_R.R``. Relative magnitudes
(Conditional) are deterministic and agree exactly; R's FLCI uses a simulated
folded-normal quantile and a coarse search, held to 1e-3.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "honest_did_moments_R.json").read_text(encoding="utf-8"))
B = np.asarray(R["betahat"], dtype=float)
S = np.asarray(R["sigma"], dtype=float)
K = R["num_pre"]
TARGETS = {"e0": {"e": 0}, "avg": {"l_vec": "average"}}


@pytest.mark.parametrize("tgt", sorted(TARGETS))
def test_relative_magnitudes_match_r(tgt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.honest_did_from_moments(
            B,
            S,
            num_pre_periods=K,
            method="relative_magnitude",
            honestdid_method="Conditional",
            m_grid=R["Mbar"],
            **TARGETS[tgt],
        )
    np.testing.assert_allclose(out["ci_lower"], R[f"rm_{tgt}"]["lb"], atol=1e-10)
    np.testing.assert_allclose(out["ci_upper"], R[f"rm_{tgt}"]["ub"], atol=1e-10)


@pytest.mark.parametrize("tgt", sorted(TARGETS))
def test_flci_matches_r(tgt):
    out = sp.honest_did_from_moments(
        B, S, num_pre_periods=K, method="smoothness", m_grid=R["M"], **TARGETS[tgt]
    )
    np.testing.assert_allclose(out["ci_lower"], R[f"sd_{tgt}"]["lb"], atol=1e-3)
    np.testing.assert_allclose(out["ci_upper"], R[f"sd_{tgt}"]["ub"], atol=1e-3)


def test_event_times_equal_the_num_pre_layout():
    a = sp.honest_did_from_moments(B, S, num_pre_periods=K, m_grid=[0.0, 0.01])
    times = list(range(-K - 1, -1)) + list(range(len(B) - K))
    b = sp.honest_did_from_moments(B, S, event_times=times, m_grid=[0.0, 0.01])
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float))


def test_same_as_honest_did_on_a_fit():
    """Moments taken from a CS fit give the same answer as the fit itself."""
    rng = np.random.default_rng(3)
    import pandas as pd

    rows = []
    for u in range(300):
        g = [4, 6, 0][u % 3]
        a = rng.normal()
        for t in range(1, 10):
            rows.append(
                dict(i=u, t=t, g=g, y=a + 0.2 * t + (t >= g > 0) + rng.normal())
            )
    cs = sp.callaway_santanna(
        pd.DataFrame(rows),
        y="y",
        g="g",
        t="t",
        i="i",
        control_group="nevertreated",
        base_period="universal",
    )
    ev = sp.event_study_vcov(cs, allow_diagonal=False)
    kw = dict(
        e=1,
        m_grid=[0.0, 0.5],
        method="relative_magnitude",
        honestdid_method="Conditional",
    )
    direct = sp.honest_did(cs, **kw)
    moments = sp.honest_did_from_moments(ev.beta, ev.vcov, event_times=ev.times, **kw)
    np.testing.assert_allclose(direct.to_numpy(float), moments.to_numpy(float))
    alias = sp.honest_did_from_result(cs, **kw)
    np.testing.assert_allclose(direct.to_numpy(float), alias.to_numpy(float))


def test_bad_inputs_raise():
    with pytest.raises(sp.MethodIncompatibility, match="sigma"):
        sp.honest_did_from_moments(B, S[:-1, :-1], num_pre_periods=K)
    with pytest.raises(sp.MethodIncompatibility, match="event_times"):
        sp.honest_did_from_moments(B, S)
    with pytest.raises(sp.MethodIncompatibility, match="pre-treatment"):
        sp.honest_did_from_moments(B, S, event_times=list(range(len(B))))
