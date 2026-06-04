"""Regression tests for the single-period stabilized-IPTW correctness fix.

Before the fix, ``stabilized_weights`` on a single-period panel produced an
all-zero lagged-treatment column that made the logistic design singular.
``_logit_proba`` swallowed the resulting ``LinAlgError`` and silently returned
the marginal mean for *both* numerator and denominator, so every stabilized
weight collapsed to exactly ``1.0`` — turning the MSM into an unweighted,
confounded regression with no warning (a CLAUDE.md §7 silent-degradation
violation).

The fix drops zero-variance columns before fitting (which is the numerically
correct thing to do — such columns only duplicate the intercept) and warns
loudly if the treatment model genuinely fails to fit.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _single_period_panel(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 1, n)
    # treatment depends on the confounder L -> needs IPTW adjustment
    A = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.5 * L))).astype(float)
    Y = 2.0 * A + 1.0 * L + rng.normal(0, 1, n)
    return pd.DataFrame({"id": np.arange(n), "time": 0, "A": A, "L": L, "Y": Y})


def test_single_period_weights_are_not_collapsed_to_one():
    """Weights must reflect the confounder, not silently degrade to 1.0."""
    df = _single_period_panel()
    sw = np.asarray(
        sp.stabilized_weights(df, treat="A", id="id", time="time",
                              time_varying=["L"])
    )
    # The pre-fix bug produced var == 0 and a single unique value of 1.0.
    assert np.var(sw) > 0.1, "stabilized weights collapsed to a constant"
    assert len(np.unique(np.round(sw, 6))) > 100


def test_single_period_weights_match_hand_computed_iptw():
    """The fixed path must equal a textbook stabilized-IPTW computation."""
    sm = pytest.importorskip("statsmodels.api")
    df = _single_period_panel()
    A = df["A"].values
    sw = np.asarray(
        sp.stabilized_weights(df, treat="A", id="id", time="time",
                              time_varying=["L"])
    )
    p_den = sm.Logit(A, sm.add_constant(df["L"].values)).fit(disp=0).predict()
    p_num = np.full(len(A), A.mean())
    sw_hand = np.where(A == 1, p_num / p_den, (1 - p_num) / (1 - p_den))
    np.testing.assert_allclose(sw, sw_hand, rtol=1e-8, atol=1e-8)


def test_multi_period_weights_still_vary():
    """Guard against regressing the (already-correct) multi-period path."""
    rng = np.random.default_rng(2)
    ids = np.repeat(np.arange(300), 4)
    t = np.tile(np.arange(4), 300)
    n = len(ids)
    L = rng.normal(0, 1, n)
    A = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-L))).astype(float)
    Y = A + L + rng.normal(0, 1, n)
    df = pd.DataFrame({"id": ids, "time": t, "A": A, "L": L, "Y": Y})
    sw = np.asarray(
        sp.stabilized_weights(df, treat="A", id="id", time="time",
                              time_varying=["L"])
    )
    assert np.var(sw) > 0.0
    assert len(np.unique(np.round(sw, 4))) > 10
