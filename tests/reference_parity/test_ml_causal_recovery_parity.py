"""Analytical parity: DML / ML-causal estimators recover known truths.

These estimators were **absent from the parity index entirely**. That is
not the same as untested -- each has unit tests elsewhere in the suite --
but the index is built by scanning ``tests/reference_parity/``, so an
estimator whose only evidence lives in ``tests/test_*.py`` shows up to
``sp.parity_status`` and to a reader of ``docs/parity.md`` exactly like
one with no evidence at all. An honest ``analytical-only`` grade is far
better than a blank.

Each test below states a population truth the estimator must recover on a
deterministic DGP, and asserts recovery within a stated number of
standard errors (or, where the quantity is an identity rather than an
estimate, exactly). No cross-package reference is involved -- that is
precisely what ``analytical-only`` means, and why these are filed here
rather than presented as cross-language parity.

Where an estimator *does* have a cross-package anchor it is pinned
elsewhere and not duplicated here: ``sp.metalearner`` against ``econml``
(``tests/external_parity/test_metalearner_econml_parity.py``),
``sp.tmle`` against ``tmle::tmle`` (Track A module 72), ``sp.dml``
against ``DoubleML`` (modules 08 and 71).

References
----------
[@chernozhukov2018double], [@kunzel2019metalearners],
[@vanderlaan2007super]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# Panel DML: within-unit confounding removed by unit fixed effects
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel_dgp():
    """Unit effects confound D and Y; the within-unit effect is 1.5."""
    rng = np.random.default_rng(20260731)
    n_units, n_periods, theta = 150, 6, 1.5
    rows = []
    for i in range(n_units):
        alpha = rng.normal(scale=2.0)  # unit effect entering both D and Y
        for t in range(n_periods):
            x = rng.normal()
            d = 0.4 * x + alpha + rng.normal(scale=0.5)
            y = theta * d + 0.6 * x + 3.0 * alpha + rng.normal(scale=0.5)
            rows.append({"unit": i, "time": t, "x": x, "d": d, "y": y})
    return pd.DataFrame(rows), theta


def test_dml_panel_recovers_the_within_unit_effect(panel_dgp):
    """Pooled DML would be badly biased here; the panel estimator is not."""
    df, theta = panel_dgp
    res = sp.dml_panel(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        unit="unit",
        time="time",
        ml_g="linear",
        ml_m="linear",
        n_folds=5,
    )
    est, se = float(res.estimate), float(res.se)
    assert se > 0
    assert abs(est - theta) <= 4.0 * se, (
        f"panel DML estimate {est:.4f} is {abs(est - theta) / se:.1f} SE "
        f"from the truth {theta}"
    )


def test_model_averaging_dml_recovers_the_truth(panel_dgp):
    """Short-stacking over candidate learners must stay consistent."""
    rng = np.random.default_rng(7)
    n, theta = 1500, 0.8
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
    y = theta * d + x1 + 0.4 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    res = sp.model_averaging_dml(df, y="y", treat="d", covariates=["x1", "x2"])
    est, se = float(res.estimate), float(res.se)
    assert se > 0
    assert abs(est - theta) <= 4.0 * se


# ---------------------------------------------------------------------------
# CATE learners and their evaluation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cate_dgp():
    """tau(x) = 1 + 0.5*x2, randomised-ish treatment, e in (0.25, 0.75)."""
    rng = np.random.default_rng(3)
    n = 1500
    X = rng.normal(size=(n, 3))
    e = 0.5 + 0.25 * np.tanh(X[:, 0])
    d = (rng.uniform(size=n) < e).astype(int)
    tau = 1.0 + 0.5 * X[:, 1]
    y = X[:, 0] + tau * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["d"], df["y"] = d, y
    return df, X, tau


def test_xlearner_recovers_the_cate_surface(cate_dgp):
    """Not just the average: the fitted CATE must track tau(x) itself."""
    df, _, tau = cate_dgp
    res = sp.xlearner(df, y="y", d="d", X=["x1", "x2", "x3"])
    cate = np.asarray(res.model_info["cate"], dtype=float)
    assert cate.shape == tau.shape
    # Correlation with the true surface is the discriminating check; a
    # constant-effect fit would pass a mean-only test and fail this.
    assert float(np.corrcoef(cate, tau)[0, 1]) > 0.9
    assert float(np.mean(cate)) == pytest.approx(float(np.mean(tau)), abs=0.2)


def test_auto_cate_selects_a_learner_and_recovers_the_average(cate_dgp):
    """The selector must return one of the learners it was offered."""
    df, _, tau = cate_dgp
    res = sp.auto_cate(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2", "x3"],
        learners=("s", "t", "x"),
    )
    assert str(res.best_learner).lower()[0] in {"s", "t", "x"}
    # The leaderboard must actually rank every learner it was offered,
    # otherwise "auto" selection is picking from a shorter list than
    # the caller asked for.
    assert len(res.leaderboard) == 3
    best = res.best_result
    assert float(best.estimate) == pytest.approx(float(np.mean(tau)), abs=0.25)


def test_cate_eval_detects_true_heterogeneity_and_rejects_none(cate_dgp):
    """RATE must separate a perfect CATE from a constant one.

    ``sp.cate_eval`` reports RATE-family statistics (AUTOC / Qini,
    Yadlowsky et al.). The discriminating property is directional: fed
    the *true* CATE it must find significant heterogeneity, and fed a
    constant it must not. A test that only checked "returns a number"
    would pass for an estimator that ignored its input entirely.
    """
    df, X, tau = cate_dgp
    Y = df["y"].to_numpy(dtype=float)
    T = df["d"].to_numpy(dtype=float)

    truth = sp.cate_eval(cate=tau.copy(), Y=Y, T=T, X=X)
    assert truth.autoc_se > 0
    assert truth.autoc / truth.autoc_se > 3.0, (
        f"AUTOC = {truth.autoc:.4f} (SE {truth.autoc_se:.4f}) on the true "
        "CATE: heterogeneity that is present by construction was not "
        "detected"
    )
    lo, hi = truth.autoc_ci
    assert lo <= truth.autoc <= hi

    constant = sp.cate_eval(cate=np.full_like(tau, float(np.mean(tau))), Y=Y, T=T, X=X)
    assert abs(constant.autoc) < abs(truth.autoc), (
        "a constant CATE carries no ranking information, so its AUTOC "
        "must not exceed that of the true CATE"
    )


# ---------------------------------------------------------------------------
# Super Learner: the ensemble underneath TMLE
# ---------------------------------------------------------------------------


def test_super_learner_weights_form_a_convex_combination():
    """Weights must be a simplex, and the ensemble must fit a known signal."""
    rng = np.random.default_rng(5)
    n = 800
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.3, size=n)
    sl = sp.super_learner(X, y, n_folds=5, task="regression", random_state=0)
    w = np.asarray(sl.weights_, dtype=float)
    assert np.all(w >= -1e-9), "ensemble weights must be non-negative"
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    pred = np.asarray(sl.predict(X), dtype=float)
    # R^2 against a signal this clean should be high for any sane ensemble.
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.9
