"""``sp.xtevent_smpath``: the closed form is the constrained optimum.

Suggestion 6 of Freyaldenhoven, Hansen, Perez Perez and Shapiro: among
polynomials through the normalised coefficient whose path lies in the Wald
region, take the lowest degree and, at that degree, the smallest leading
coefficient. These tests check that definition directly, against a generic
constrained optimiser, and its limiting cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import MethodIncompatibility


def _result(att, V, times, normalized=(-1,)):
    times = list(times)
    est = [t for t in times if t not in normalized]
    es = pd.DataFrame(
        {
            "relative_time": times,
            "att": [0.0 if t in normalized else att[est.index(t)] for t in times],
        }
    )
    return CausalResult(
        method="Panel event study (xtevent)",
        estimand="x",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(0.0, 0.0),
        alpha=0.05,
        n_obs=100,
        model_info={
            "event_study": es,
            "vcov_event": pd.DataFrame(V, index=est, columns=est),
            "normalized": list(normalized),
        },
    )


TIMES = [-3, -2, -1, 0, 1, 2, 3]
EST = [-3, -2, 0, 1, 2, 3]


def _wald(coef, b, W):
    path = np.polyval(coef[::-1], np.asarray(EST, float))
    r = path - b
    return float(r @ W @ r)


def test_zero_path_when_no_effect_and_no_pretrend_is_consistent():
    b = np.array([0.01, -0.02, 0.015, 0.0, -0.01, 0.02])
    out = sp.xtevent_smpath(_result(b, np.eye(6) * 0.01, TIMES))
    assert out.attrs["order"] == 0
    np.testing.assert_allclose(out["smpath"], 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_form_equals_numerical_constrained_optimum(seed):
    rng = np.random.default_rng(seed)
    k = np.asarray(EST, float)
    b = 0.05 * k + 0.02 * k**2 + rng.normal(0, 0.05, 6)
    A = rng.normal(size=(6, 6))
    V = 0.002 * (A @ A.T / 6 + np.eye(6))
    W = np.linalg.inv(V)
    out = sp.xtevent_smpath(_result(b, V, TIMES))
    deg = out.attrs["order"]
    crit = stats.chi2.ppf(0.95, 6)
    assert out.attrs["critical"] == pytest.approx(crit)
    # Degree deg - 1 cannot reach the region: its least Wald statistic
    # (least squares over polynomials vanishing at -1) exceeds the critical value.
    assert out.attrs["wald_by_order"][-2] > crit
    assert out.attrs["wald_by_order"][-1] <= crit

    # Same problem, generic solver: minimise v_p^2 over polynomials of this
    # degree that vanish at -1 and lie in the Wald region.
    p = deg + 1
    cons = [
        {"type": "eq", "fun": lambda v: np.polyval(v[::-1], -1.0)},
        {"type": "ineq", "fun": lambda v: crit - _wald(v, b, W)},
    ]
    x0 = np.asarray(out.attrs["coefficients"]) * 1.05
    best = None
    for start in (x0, np.zeros(p)):
        res = optimize.minimize(
            lambda v: v[-1] ** 2,
            start,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    assert best is not None
    lead = out.attrs["coefficients"][-1]
    assert abs(lead) <= abs(best.x[-1]) * (1 + 1e-6) + 1e-10
    assert abs(lead) == pytest.approx(abs(best.x[-1]), rel=1e-4)
    assert out.attrs["wald"] == pytest.approx(crit, rel=1e-9)
    assert abs(out.set_index("relative_time").loc[-1, "smpath"]) < 1e-12


def test_postwindow_drops_later_post_coefficients_from_the_region():
    b = np.array([0.0, 0.0, 0.3, 0.5, 5.0, -5.0])
    V = np.eye(6) * 0.01
    out = sp.xtevent_smpath(_result(b, V, TIMES), postwindow=2)
    assert out["in_region"].tolist() == [True, True, False, True, True, False, False]
    assert out.attrs["df"] == 4


def test_level_controls_the_region():
    b = np.array([-0.1, -0.05, 0.2, 0.3, 0.4, 0.5])
    r = _result(b, np.eye(6) * 0.004, TIMES)
    wide = sp.xtevent_smpath(r, level=0.99)
    narrow = sp.xtevent_smpath(r, level=0.5)
    assert wide.attrs["critical"] > narrow.attrs["critical"]
    assert wide.attrs["order"] <= narrow.attrs["order"]


def test_refuses_static_and_bad_options():
    r = _result(np.zeros(6), np.eye(6), TIMES)
    with pytest.raises(MethodIncompatibility, match="maxorder"):
        sp.xtevent_smpath(r, maxorder=11)
    with pytest.raises(MethodIncompatibility, match="level"):
        sp.xtevent_smpath(r, level=1.5)
    static = CausalResult(
        method="x",
        estimand="x",
        estimate=0.0,
        se=1.0,
        pvalue=1.0,
        ci=(0.0, 0.0),
        alpha=0.05,
        n_obs=1,
        model_info={"static": True},
    )
    with pytest.raises(MethodIncompatibility, match="event study") as err:
        sp.xtevent_smpath(static)
    assert err.value.recovery_hint
