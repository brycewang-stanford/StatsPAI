"""sp.fect placebo / carryover / leave-one-period-out tests: known truth.

Numbers against R ``fect::fect`` are in
``tests/reference_parity/test_synth_family_R_parity.py``. Here the outcome
is an exact two-way fixed-effects surface, so every out-of-sample effect is
known: zero before treatment, the carryover after a switch-off.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel(carry=0.0, reversals=True, n=14, T=12):
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 1, n)
    xi = rng.normal(0, 1, T)
    D = np.zeros((n, T), dtype=int)
    for i in range(4, 9):
        D[i, 4 + (i % 3) :] = 1
    if reversals:
        for i in range(9, n):
            D[i, 4 + (i % 2) : 7 + (i % 2)] = 1
    rows = []
    for i in range(n):
        for t in range(T):
            off = t > 0 and D[i, t - 1] == 1 and D[i, t] == 0
            y = alpha[i] + xi[t] + 1.5 * D[i, t] + (carry if off else 0.0)
            rows.append((i, t, y, int(D[i, t])))
    return pd.DataFrame(rows, columns=["id", "t", "y", "d"])


KW = dict(y="y", treat="d", unit="id", time="t", tol=1e-12, max_iter=20000)


def test_placebo_effect_is_zero_without_pretrends():
    r = sp.fect(_panel(), placebo_period=(-2, 0), vce="jackknife", **KW)
    pt = r.model_info["placebo_test"]
    assert abs(pt["estimate"]) < 1e-8
    assert pt["n_cells"] > 0 and pt["period"] == (-2, 0)


def test_carryover_effect_is_recovered():
    r = sp.fect(_panel(carry=0.7), carryover_period=1, vce="jackknife", **KW)
    assert abs(r.model_info["carryover_test"]["estimate"] - 0.7) < 1e-8
    # the carryover cells left the fit, so the ATT is exact as well
    assert abs(r.estimate - 1.5) < 1e-8


def test_loo_effects_are_zero_and_tests_reported():
    r = sp.fect(_panel(), loo=True, vce="jackknife", **KW)
    tab = r.model_info["loo"]["table"]
    assert (tab["period"] <= 0).all()
    np.testing.assert_allclose(tab["att"], 0.0, atol=1e-8)
    assert "pretrend_test" in r.model_info and "test" in r.model_info["loo"]


def test_tests_need_resampling():
    with pytest.raises(MethodIncompatibility, match="resampling"):
        sp.fect(_panel(), placebo_period=(-2, 0), **KW)


def test_one_test_at_a_time():
    with pytest.raises(MethodIncompatibility, match="one at a time"):
        sp.fect(_panel(), placebo_period=0, loo=True, vce="jackknife", **KW)


def test_carryover_needs_reversals():
    with pytest.raises(MethodIncompatibility, match="never switches off"):
        sp.fect(_panel(reversals=False), carryover_period=1, vce="jackknife", **KW)


@pytest.mark.parametrize(
    "arg", [{"placebo_period": (-1, 2)}, {"carryover_period": (0, 1)}]
)
def test_windows_on_the_wrong_side_are_refused(arg):
    with pytest.raises(MethodIncompatibility):
        sp.fect(_panel(), vce="jackknife", **arg, **KW)
