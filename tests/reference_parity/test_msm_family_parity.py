"""Analytical parity: sp.msm marginal structural model via stabilized IPTW.

Two analytical identities are checked (no external reference software):

1. **Longitudinal recovery.** In the canonical Robins (2000) DGP — a
   time-varying confounder L_t that (a) responds to past treatment
   (L_t = L_{t-1} + 0.4*A_{t-1} + noise) and (b) drives the current
   treatment propensity (logit P(A_t=1) = 0.6*L_t + 0.2*V) — the outcome
   is generated as Y_t = 0.5 * cum(A)_t + 1.0*L_0 + 0.3*V + noise, so
   the true marginal cumulative-exposure slope is exactly 0.5 by
   construction. Naive pooled OLS of Y on cum(A) + V is upward-biased
   (L_0 raises both treatment uptake and Y), while the IPTW-weighted
   MSM with a correctly specified denominator model (logit on A_lag, V,
   L) is consistent for 0.5. Both the propensity and outcome models are
   correctly specified, so the only error is sampling noise.

2. **Single-period reduction.** With one period the stabilized-IPTW MSM
   collapses algebraically to the Hajek stabilized-IPW ATE: a WLS fit of
   Y on [1, A] with weights sw = P(A)/P(A|L) reproduces the weighted
   difference in group means exactly. We recompute that quantity by hand
   (statsmodels logistic propensity + closed-form weighted means) and
   require machine-precision agreement, plus recovery of the known
   truth tau = 2.0.

Analytical evidence tier: parameters are known by construction; no
R/Stata reference run is involved.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TRUE_SLOPE = 0.5  # longitudinal cumulative-exposure slope
TRUE_TAU = 2.0  # single-period ATE


def _make_longitudinal_panel(n=1200, T=4, alpha=TRUE_SLOPE, seed=0):
    """Time-varying confounding where naive regression is biased.

    L_t is affected by past treatment and predicts current treatment;
    Y_t depends on cum(A)_t (the causal channel, slope ``alpha``) and on
    the baseline confounder value L_0 (the biasing channel).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        L = rng.normal(0.0, 1.0)
        L0 = L
        V = rng.normal(0.0, 1.0)
        cum_a = 0.0
        a_prev = 0.0
        for t in range(T):
            # L measured before A_t; responds to lagged treatment.
            L = L + 0.4 * a_prev + rng.normal(0.0, 0.3)
            pr = 1.0 / (1.0 + np.exp(-(0.6 * L + 0.2 * V)))
            a = float(rng.binomial(1, pr))
            cum_a += a
            y = alpha * cum_a + 1.0 * L0 + 0.3 * V + rng.normal(0.0, 0.5)
            rows.append(
                {"pid": i, "visit": t, "a": a, "l": L, "v": V, "y": y}
            )
            a_prev = a
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def longitudinal_panel():
    return _make_longitudinal_panel()


@pytest.fixture(scope="module")
def longitudinal_fit(longitudinal_panel):
    # trim=0: weight truncation trades variance for bias, which would
    # blur the analytical identity this test freezes. The DGP keeps
    # positivity healthy (|logit| mostly < 2), so untrimmed weights are
    # well-behaved here.
    return sp.msm(
        longitudinal_panel,
        y="y",
        treat="a",
        id="pid",
        time="visit",
        time_varying=["l"],
        baseline=["v"],
        exposure="cumulative",
        trim=0.0,
    )


@pytest.fixture(scope="module")
def single_period_panel():
    rng = np.random.default_rng(0)
    n = 4000
    L = rng.normal(0.0, 1.0, n)
    A = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.2 * L))).astype(float)
    Y = 2.0 + TRUE_TAU * A + 1.5 * L + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {"pid": np.arange(n), "t": 0, "a": A, "l": L, "y": Y}
    )


@pytest.fixture(scope="module")
def single_period_fit(single_period_panel):
    # trim=0 so the estimate equals the untruncated Hajek IPW identity
    # exactly (default quantile trimming would clip the extreme weights
    # and break machine-precision parity).
    return sp.msm(
        single_period_panel,
        y="y",
        treat="a",
        id="pid",
        time="t",
        time_varying=["l"],
        exposure="current",
        trim=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Longitudinal cumulative-exposure recovery
# ---------------------------------------------------------------------------


def test_cumulative_slope_recovers_truth(longitudinal_fit):
    # Both nuisance models are correctly specified, so the estimator is
    # consistent and the only error is sampling noise. Across seeds 0-7
    # of this exact DGP the error never exceeded 0.043; the fit's
    # cluster-robust SE is ~0.023, so 0.08 is ~3.5 SEs of headroom
    # while still rejecting the naive bias (~+0.30) by a wide margin.
    assert abs(longitudinal_fit.estimate - TRUE_SLOPE) < 0.08


def test_naive_pooled_ols_is_biased_msm_is_not(longitudinal_panel, longitudinal_fit):
    # The bias channel is analytical: L_0 raises every A_t (through L_t)
    # and enters Y with coefficient 1.0, so pooled OLS of Y on cum(A)+V
    # without weighting inherits a positive omitted-variable bias
    # (observed ~+0.30 across seeds; we require > 0.2). The MSM error
    # must also be strictly smaller than the naive error — this is what
    # shows the weights are doing real causal work on the same data.
    df = longitudinal_panel.copy()
    df["cum_a"] = df.groupby("pid")["a"].cumsum()
    X = np.column_stack([np.ones(len(df)), df["cum_a"], df["v"]])
    naive = float(np.linalg.lstsq(X, df["y"].values, rcond=None)[0][1])
    assert naive - TRUE_SLOPE > 0.2, "DGP no longer confounded enough"
    assert abs(longitudinal_fit.estimate - TRUE_SLOPE) < abs(naive - TRUE_SLOPE)


def test_stabilized_weight_mean_is_one(longitudinal_fit):
    # Stabilized weights have E[sw] = 1 under a correctly specified
    # numerator/denominator pair; finite-sample deviation is O(1/sqrt(n))
    # (observed |mean-1| < 0.016 across seeds; 0.05 is the same margin
    # the IPCW parity test uses for n of this order).
    assert abs(longitudinal_fit.model_info["sw_mean"] - 1.0) < 0.05


def test_ci_covers_truth_and_metadata(longitudinal_fit):
    # With a consistent estimator and correct cluster-robust SEs, the
    # 95% CI should cover the truth under this seed (error ~0 SEs at
    # seed 0). Also freeze the estimand bookkeeping.
    lo, hi = longitudinal_fit.ci
    assert lo < TRUE_SLOPE < hi
    assert lo < longitudinal_fit.estimate < hi
    assert longitudinal_fit.model_info["n_units"] == 1200
    assert longitudinal_fit.model_info["n_periods"] == 4
    assert longitudinal_fit.n_obs == 1200 * 4


# ---------------------------------------------------------------------------
# 2. Single-period reduction to stabilized IPW
# ---------------------------------------------------------------------------


def test_single_period_recovers_known_ate(single_period_fit):
    # tau = 2.0 by construction; the Hajek stabilized-IPW estimator has
    # a simulation SD of ~0.06 here (observed max |error| 0.12 across
    # seeds 0-5), so 0.2 is > 3 simulation SDs while naive bias is ~1.4.
    assert abs(single_period_fit.estimate - TRUE_TAU) < 0.2


def test_single_period_equals_hand_computed_hajek_ipw(
    single_period_panel, single_period_fit
):
    # Frozen-formula identity: with one period, sw = P(A)/P(A|L) and the
    # weighted regression of Y on [1, A] equals the Hajek weighted
    # difference in means. Machine-precision agreement required
    # (observed < 1e-15; atol 1e-10 leaves headroom for BLAS variation).
    sm = pytest.importorskip("statsmodels.api")
    A = single_period_panel["a"].values
    Y = single_period_panel["y"].values
    L = single_period_panel["l"].values
    p_den = sm.Logit(A, sm.add_constant(L)).fit(disp=0).predict()
    p_num = np.full(len(A), A.mean())
    w = np.where(A == 1, p_num / p_den, (1 - p_num) / (1 - p_den))
    hajek = float(
        np.sum(w * Y * A) / np.sum(w * A)
        - np.sum(w * Y * (1 - A)) / np.sum(w * (1 - A))
    )
    assert single_period_fit.estimate == pytest.approx(hajek, abs=1e-10)


def test_single_period_naive_difference_is_biased(single_period_panel, single_period_fit):
    # Naive treated-minus-control difference absorbs 1.5*E[L|A=1]-1.5*E[L|A=0]
    # (~+1.4 under this DGP); require > 0.8 bias and that the MSM error
    # is strictly smaller on the same data.
    A = single_period_panel["a"].values
    Y = single_period_panel["y"].values
    naive = float(Y[A == 1].mean() - Y[A == 0].mean())
    assert naive - TRUE_TAU > 0.8, "DGP no longer confounded enough"
    assert abs(single_period_fit.estimate - TRUE_TAU) < abs(naive - TRUE_TAU)
