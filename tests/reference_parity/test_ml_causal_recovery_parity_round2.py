"""Analytical parity, round 2: the DML / ML-causal estimators still absent
from the parity index.

Companion to ``test_ml_causal_recovery_parity.py``, which covered six of
them. These are the remainder: ``hal_tmle``, ``ltmle``,
``ltmle_survival``, ``auto_cate_tuned``, ``cluster_cate`` and
``focal_cate``. Each was registered and reachable from ``sp.*`` but
carried no entry in ``sp.parity_status`` at all — indistinguishable, to a
user or to ``docs/parity.md``, from an estimator with no evidence behind
it.

The LTMLE standard error
------------------------
``test_ltmle_standard_error_tracks_the_sampling_distribution`` is not a
formality. Before v1.21 ``sp.ltmle`` computed only the last term of the
efficient influence curve, ``Q*_1 - psi``, dropping the martingale sum
``sum_k H_k (Q*_{k+1} - Q*_k)``. What remained was the dispersion of a
*fitted conditional mean*, not the sampling variability of the estimator:
the reported SE came out 250-400x too small and did not even converge at
the sqrt(n) rate, so every confidence interval and p-value the function
produced was meaningless. The test below compares the reported SE against
the Monte-Carlo standard deviation of the estimator itself, which is the
only check that would have caught it.

References
----------
[@vanderlaan2006targeted], [@kunzel2019metalearners]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# Longitudinal TMLE
# ---------------------------------------------------------------------------


def _ltmle_panel(n: int, seed: int) -> pd.DataFrame:
    """Two-period DGP with time-varying confounding.

    ``L2`` is affected by ``A1`` and confounds ``A2``, so a naive
    regression on both treatments is biased and the sequential-regression
    machinery is actually needed.
    """
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=n)
    A1 = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * L1))).astype(int)
    L2 = 0.5 * L1 + 0.8 * A1 + rng.normal(size=n)
    A2 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.4 * L2)))).astype(int)
    Y = 1.0 * A1 + 1.0 * A2 + 0.5 * L1 + 0.3 * L2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"L1": L1, "A1": A1, "L2": L2, "A2": A2, "Y": Y})


# Substituting L2 into Y gives Y = 1.24*A1 + 1.0*A2 + 0.65*L1, so the
# always-treat vs never-treat contrast is 1.24 + 1.00.
_LTMLE_TRUTH = 2.24
_LTMLE_KW = dict(y="Y", treatments=["A1", "A2"], covariates_time=[["L1"], ["L2"]])


def test_ltmle_recovers_the_sequential_regression_truth():
    """The point estimate must find 2.24, which OLS on (A1, A2) cannot."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ltmle(_ltmle_panel(4000, 0), **_LTMLE_KW)
    assert abs(float(res.ate) - _LTMLE_TRUTH) <= 4.0 * float(res.se)
    assert res.K == 2
    assert tuple(res.regime_treated) == (1, 1)
    assert tuple(res.regime_control) == (0, 0)


@pytest.mark.slow
def test_ltmle_standard_error_tracks_the_sampling_distribution():
    """The reported SE must match the estimator's actual dispersion.

    Comparing the SE against the Monte-Carlo standard deviation of the
    point estimate is the check that catches an influence curve missing a
    term: a wrong SE is invisible to any single-fit test, however precise.
    """
    n, reps = 1500, 25
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reported = float(sp.ltmle(_ltmle_panel(n, 0), **_LTMLE_KW).se)
        draws = [
            float(sp.ltmle(_ltmle_panel(n, s), **_LTMLE_KW).ate)
            for s in range(1, reps + 1)
        ]
    mc_sd = float(np.std(draws, ddof=1))
    ratio = reported / mc_sd
    assert 0.6 < ratio < 1.6, (
        f"reported SE {reported:.5f} vs Monte-Carlo sd {mc_sd:.5f} "
        f"(ratio {ratio:.3f}). The efficient influence curve is "
        f"mis-specified — most likely the martingale sum "
        f"sum_k H_k (Q*_{{k+1}} - Q*_k) is missing, which leaves only the "
        f"dispersion of a fitted conditional mean."
    )


@pytest.mark.slow
def test_ltmle_standard_error_converges_at_the_root_n_rate():
    """se * sqrt(n) must be stable; a wrong IC gets the *rate* wrong.

    The pre-v1.21 SE shrank faster than 1/sqrt(n), so its ratio to the
    truth degraded as the sample grew — a defect a fixed-n check cannot
    see.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = [
            float(sp.ltmle(_ltmle_panel(n, 0), **_LTMLE_KW).se) * np.sqrt(n)
            for n in (1000, 4000)
        ]
    assert scaled[0] > 0
    assert 0.7 < scaled[1] / scaled[0] < 1.4, (
        f"se*sqrt(n) moved from {scaled[0]:.4f} to {scaled[1]:.4f} between "
        "n=1000 and n=4000; the standard error is not converging at the "
        "sqrt(n) rate."
    )


def test_ltmle_survival_curves_are_monotone_and_consistent_with_rmst():
    """Survival LTMLE: the curves must be valid, and RMST must follow them.

    A survival probability cannot rise over time, and the restricted mean
    survival time is an integral of the curve — so the sign of the RMST
    difference has to agree with which curve lies above the other. Checking
    the two together catches an arm swap that either alone would miss.
    """
    rng = np.random.default_rng(5)
    n = 800
    L1 = rng.normal(size=n)
    A1 = (rng.uniform(size=n) < 0.5).astype(int)
    d1 = rng.uniform(size=n) < 1 / (1 + np.exp(-(-1.5 + 0.4 * L1 - 0.5 * A1)))
    L2 = 0.5 * L1 + rng.normal(size=n)
    A2 = (rng.uniform(size=n) < 0.5).astype(int)
    d2 = d1 | (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1.2 + 0.3 * L2 - 0.5 * A2))))
    df = pd.DataFrame(
        {
            "L1": L1,
            "A1": A1,
            "D1": d1.astype(int),
            "L2": L2,
            "A2": A2,
            "D2": d2.astype(int),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ltmle_survival(
            df,
            event_indicators=["D1", "D2"],
            treatments=["A1", "A2"],
            covariates_time=[["L1"], ["L2"]],
        )

    s_t = np.asarray(res.survival_treated, dtype=float)
    s_c = np.asarray(res.survival_control, dtype=float)
    assert s_t.shape == s_c.shape == np.asarray(res.times).shape

    for arm, curve in (("treated", s_t), ("control", s_c)):
        assert np.all(
            (curve >= 0) & (curve <= 1)
        ), f"{arm} survival probability outside [0, 1]"
        assert np.all(np.diff(curve) <= 1e-9), (
            f"{arm} survival curve increased over time, which a survival "
            "function cannot do"
        )

    # The DGP gives treatment a protective effect at both periods, so the
    # treated curve should sit above the control one; RMST must agree.
    assert np.all(s_t >= s_c - 1e-9)
    assert float(res.rmst_difference) > 0
    assert float(res.rmst_treated) > float(res.rmst_control)
    assert float(res.rmst_se) > 0
    lo, hi = res.rmst_ci
    assert lo < float(res.rmst_difference) < hi


# ---------------------------------------------------------------------------
# HAL-TMLE and the CATE reporting helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cate_panel():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 3))
    e = 0.5 + 0.25 * np.tanh(X[:, 0])
    d = (rng.uniform(size=n) < e).astype(int)
    tau = 1.0 + 0.5 * X[:, 1]
    y = X[:, 0] + tau * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["d"], df["y"] = d, y
    return df, tau


def test_hal_tmle_recovers_the_known_ate(cate_panel):
    df, tau = cate_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.hal_tmle(df, y="y", treat="d", covariates=["x1", "x2", "x3"])
    truth = float(np.mean(tau))
    assert float(res.se) > 0
    assert abs(float(res.estimate) - truth) <= 4.0 * float(res.se)
    lo, hi = res.ci
    assert lo < float(res.estimate) < hi


def test_auto_cate_tuned_ranks_every_learner_it_was_offered(cate_panel):
    df, tau = cate_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.auto_cate_tuned(
            df,
            y="y",
            treat="d",
            covariates=["x1", "x2", "x3"],
            learners=("s", "t"),
            n_trials=2,
        )
    assert str(res.best_learner).lower()[0] in {"s", "t"}
    assert len(res.leaderboard) == 2
    assert abs(float(res.best_result.estimate) - float(np.mean(tau))) < 0.3


def test_cluster_cate_partitions_every_observation_once(cate_panel):
    """The cluster table must account for the whole sample, exactly once."""
    df, _ = cate_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.cluster_cate(
            df, y="y", treat="d", covariates=["x1", "x2", "x3"], n_clusters=3
        )
    table = res.cluster_table
    assert res.n_clusters == 3
    assert len(table) == 3
    size_col = next(c for c in table.columns if c.lower() in {"n", "size", "n_obs"})
    assert int(table[size_col].sum()) == int(res.n_obs)


def test_focal_cate_grid_is_aligned_and_finite(cate_panel):
    """The CATE grid and its SEs must line up and be usable."""
    df, _ = cate_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.focal_cate(
            df, y_columns=["y"], treat="d", covariates=["x1", "x2", "x3"]
        )
    cate = np.asarray(res.cate_grid, dtype=float)
    ses = np.asarray(res.se_grid, dtype=float)
    assert cate.shape == ses.shape
    assert cate.size > 0
    assert np.isfinite(cate).all(), "CATE grid contains non-finite values"
    assert np.all(ses >= 0), "negative standard errors on the CATE grid"
