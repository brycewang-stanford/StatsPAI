"""Tests for ``sp.selection_bounds`` — Lee (2009) trimming bounds for the ATE
under sample selection (previously untested public function).

The defining property of Lee bounds: when selection is *independent* of
treatment there is no differential attrition, so the trimming proportion is
~0 and the bounds collapse around the true effect. When treatment shifts the
selection probability, the identified set must widen. Both directions are
checked here.

References
----------
Lee, D.S. (2009). "Training, Wages, and Sample Selection: Estimating Sharp
Bounds on Treatment Effects." Review of Economic Studies 76(3), 1071-1102.
"""

import numpy as np
import pandas as pd

import statspai as sp

TRUE_ATE = 2.0


def _make(diff_selection, seed=0, n=3000):
    rng = np.random.default_rng(seed)
    D = rng.integers(0, 2, n)
    Y = 1.0 + TRUE_ATE * D + rng.normal(0, 1, n)
    if diff_selection:
        p = np.where(D == 1, 0.9, 0.6)  # treatment raises selection prob
    else:
        p = np.full(n, 0.75)  # selection independent of treatment
    S = (rng.random(n) < p).astype(int)
    Yobs = np.where(S == 1, Y, np.nan)
    return pd.DataFrame({"y": Yobs, "treatment": D, "selection": S})


def test_bounds_are_ordered_and_bracket_truth_under_independence():
    res = sp.selection_bounds(
        _make(False),
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=0,
    )
    assert res.lower < res.upper
    # No differential attrition -> tight bounds containing the true ATE.
    assert res.lower <= TRUE_ATE <= res.upper
    assert res.width < 0.3


def test_differential_selection_widens_identified_set():
    indep = sp.selection_bounds(
        _make(False),
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=0,
    )
    diff = sp.selection_bounds(
        _make(True),
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=0,
    )
    # Differential attrition must enlarge the Lee identified set.
    assert diff.width > 3 * indep.width


def test_selection_bounds_result_contract():
    res = sp.selection_bounds(
        _make(False),
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=0,
    )
    # ci_lower / ci_upper are the bootstrap CIs *for each bound* (2-tuples).
    assert res.ci_lower[0] <= res.ci_lower[1]
    assert res.ci_upper[0] <= res.ci_upper[1]
    assert res.se_lower > 0 and res.se_upper > 0
    # Derived quantities are internally consistent.
    np.testing.assert_allclose(res.width, res.upper - res.lower, rtol=1e-8)
    np.testing.assert_allclose(res.midpoint, 0.5 * (res.lower + res.upper), rtol=1e-8)
    assert res.n_obs > 0


def test_selection_bounds_is_seed_reproducible():
    data = _make(False)
    a = sp.selection_bounds(
        data,
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=123,
    )
    b = sp.selection_bounds(
        data,
        y="y",
        treatment="treatment",
        selection="selection",
        n_boot=50,
        random_state=123,
    )
    np.testing.assert_allclose(a.lower, b.lower)
    np.testing.assert_allclose(a.upper, b.upper)
    np.testing.assert_allclose(a.ci_lower, b.ci_lower)
