"""The smoothness FLCI of ``sp.honest_did`` on an ill-conditioned case.

The half-length of the Rambachan-Roth fixed-length interval is minimised
over a one-parameter family of affine estimators, and near the minimum it
is flat in that parameter. Through 1.45.0 the family was indexed by the
standard deviation ``h`` and the worst-case bias ``b(h)`` came from an
SLSQP solve accurate to about 1e-7; on the event study below (the
unilateral-divorce panel, ``M = 1``) that error moved the minimiser, and
the centre of the interval, by 6e-4 of a bound between SciPy 1.15 and
SciPy 1.17. The family is now indexed by the bias bound ``b`` and the
smallest standard deviation ``h(b)`` is a strictly convex quadratic
program, which pins the minimiser.

Reference: Stata ``honestdid`` 1.3.0, ``delta(sd)``, on the ``csdid``
moments of the same panel (fixture ``honest_flci_divorce_stata.json``).
The two programs minimise the same objective with different optimisers,
and the objective is flat, so the bounds agree to about 3e-6 absolute, not
to rounding; the tolerance below is 1e-5. The earlier solver was 3.5e-5
from Stata at ``M = 1`` under SciPy 1.15 and 4e-3 under SciPy 1.17.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

_flci = importlib.import_module("statspai.did._flci")

FIXTURE = Path(__file__).parent / "_fixtures" / "honest_flci_divorce_stata.json"


@pytest.fixture(scope="module")
def case():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d["betahat"] = np.asarray(d["betahat"], dtype=float)
    d["sigma"] = np.asarray(d["sigma"], dtype=float)
    d["l_post"] = np.asarray(d["l_post"], dtype=float)
    return d


def _fit(case, m):
    return _flci.flci_delta_sd(
        case["betahat"],
        case["sigma"],
        case["n_pre"],
        case["n_post"],
        m,
        case["l_post"],
        case["alpha"],
    )


@pytest.mark.parametrize("m", ["0.5", "1.0"])
def test_bounds_match_stata_honestdid(case, m):
    res = _fit(case, float(m))
    lower, upper = case["stata"][m]
    assert res.ci_lower == pytest.approx(lower, abs=1e-5)
    assert res.ci_upper == pytest.approx(upper, abs=1e-5)


@pytest.mark.parametrize("m", [0.5, 1.0])
def test_reported_pieces_are_those_of_the_reported_estimator(case, m):
    """``h`` is the SD of the affine estimator the weights define, the
    centre is that estimator, and the half-length is the folded-normal
    quantile at ``bias / h`` times ``h``."""
    res = _fit(case, m)
    weights = np.concatenate([res.pre_period_weights, case["l_post"]])
    sd = float(np.sqrt(weights @ case["sigma"] @ weights))
    assert res.h == pytest.approx(sd, rel=1e-9)
    assert res.estimate == pytest.approx(float(weights @ case["betahat"]), rel=1e-12)
    half = _flci.folded_normal_quantile(1 - case["alpha"], res.worst_case_bias / sd)
    assert res.half_length == pytest.approx(half * sd, rel=1e-9)
    assert res.ci_upper - res.ci_lower == pytest.approx(2 * res.half_length, rel=1e-12)


@pytest.mark.parametrize("m", [0.5, 1.0])
def test_no_estimator_on_the_frontier_gives_a_shorter_interval(case, m):
    prog = _flci._FLCIProgram(
        case["betahat"],
        case["sigma"],
        case["n_pre"],
        case["n_post"],
        case["l_post"],
    )
    best = prog.ci(m, case["alpha"]).half_length
    others = [
        prog.half_length_at_bias(b, m, case["alpha"])
        for b in np.linspace(prog.b_min, prog.b_max, 257)
    ]
    assert best <= min(others) * (1 + 1e-12)


def test_frontier_ends_are_the_two_extreme_estimators(case):
    """At the largest bias bound the program returns the minimum-variance
    estimator, and its SD falls as the bound is relaxed."""
    prog = _flci._FLCIProgram(
        case["betahat"],
        case["sigma"],
        case["n_pre"],
        case["n_post"],
        case["l_post"],
    )
    assert prog.sd(prog.b_max)[0] == pytest.approx(prog.h_min, rel=1e-8)
    sds = [prog.sd(b)[0] for b in np.linspace(prog.b_min, prog.b_max, 9)]
    assert all(a >= b * (1 - 1e-9) for a, b in zip(sds, sds[1:]))
    assert sds[0] > sds[-1]
