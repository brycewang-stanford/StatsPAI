"""``sp.ltmle(q_learner=, g_learner=)`` and ``sp.lmtp`` beyond the R rows.

R agreement is in ``tests/reference_parity/test_teffects_R_parity.py``,
``test_ltmle_weights_parity.py`` and ``test_lmtp_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import statspai as sp


def _dgp(n, rng, a0=None, a1=None):
    l0 = rng.normal(size=n)
    p0 = 1 / (1 + np.exp(-(0.5 * l0**2 - 0.5)))
    A0 = rng.binomial(1, p0) if a0 is None else np.full(n, a0)
    l1 = 0.5 * l0 + 0.5 * A0 + rng.normal(size=n)
    p1 = 1 / (1 + np.exp(-(0.8 * np.sin(2 * l1))))
    A1 = rng.binomial(1, p1) if a1 is None else np.full(n, a1)
    y = l0**2 + A0 + A1 + 2 * np.sin(l1) + 0.5 * rng.normal(size=n)
    return pd.DataFrame(dict(L0=l0, A0=A0, L1=l1, A1=A1, Y=y))


def test_logistic_g_learner_reproduces_the_glm_fit():
    df = _dgp(1500, np.random.default_rng(1))
    a = sp.ltmle(df, "Y", ["A0", "A1"], [["L0"], ["L1"]])
    b = sp.ltmle(
        df,
        "Y",
        ["A0", "A1"],
        [["L0"], ["L1"]],
        g_learner=LogisticRegression(penalty=None, tol=1e-12, max_iter=5000),
    )
    assert abs(a.ate - b.ate) < 1e-8 and abs(a.se - b.se) < 1e-8


def test_flexible_learners_recover_a_nonlinear_truth():
    rng = np.random.default_rng(0)
    truth = _dgp(200_000, rng, 1, 1).Y.mean() - _dgp(200_000, rng, 0, 0).Y.mean()
    df = _dgp(3000, np.random.default_rng(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ltmle(
            df,
            "Y",
            ["A0", "A1"],
            [["L0"], ["L1"]],
            q_learner=GradientBoostingRegressor(
                max_depth=2, n_estimators=150, random_state=0
            ),
            g_learner=GradientBoostingClassifier(
                max_depth=2, n_estimators=150, random_state=0
            ),
        )
    assert abs(r.ate - truth) < 3 * r.se


def test_lmtp_recovers_a_shift_effect():
    # Y = A + W + e and the policy adds 1: E[Y(d)] - E[Y] = 1 exactly
    rng = np.random.default_rng(4)
    n = 2000
    w = rng.normal(size=n)
    a = 0.5 * w + rng.normal(size=n)
    df = pd.DataFrame({"W": w, "A": a, "Y": a + w + rng.normal(size=n)})
    r = sp.lmtp(df, "Y", ["A"], shift=1.0, baseline=["W"])
    eff, se = r.model_info["effect"], r.model_info["effect_se"]
    assert abs(eff - 1.0) < 3 * se
