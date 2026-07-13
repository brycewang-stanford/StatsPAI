"""Analytical parity: fairness-family metrics on hand-constructed data.

All fairness metrics covered here have closed forms, so we verify them
against hand-computed values on fully constructed data instead of an
external reference implementation. Analytical evidence tier.

DGP 1 (group metrics) — a fixed confusion table, 16 rows per group:

    group 0:  y=1 (8 rows): yhat = 1 on 6  ->  TPR_0 = 6/8 = 0.75
              y=0 (8 rows): yhat = 1 on 2  ->  FPR_0 = 2/8 = 0.25
    group 1:  y=1 (8 rows): yhat = 1 on 2  ->  TPR_1 = 2/8 = 0.25
              y=0 (8 rows): yhat = 1 on 4  ->  FPR_1 = 4/8 = 0.50

    demographic parity: P(yhat=1|A=0) = 8/16 = 0.5,
                        P(yhat=1|A=1) = 6/16 = 0.375, gap = 0.125
    equalized odds:     TPR gap = 0.5, FPR gap = 0.25,
                        gap = max(0.5, 0.25) = 0.5

Every rate is a dyadic rational (k / 2^m), hence exactly representable in
binary floating point — the identities are asserted with `==`, no tolerance.

DGP 2 (counterfactual metrics) — a tiny linear SCM

    A ~ {0, 1},  X = k/16 (dyadic, independent of A),
    do(A = a') simply sets A = a' (X is a non-descendant of A).

    unfair predictor f(X, A) = 0.5*X + 1.5*A:
        f under do(A=a') - f factual = 1.5*(a' - A), so the per-unit
        counterfactual change is exactly 1.5 whenever a' != A.
    fair predictor   f(X)    = 0.5*X:
        counterfactual change is exactly 0.

Because 0.5, 1.5 and the X values are dyadic and small, all predictor
arithmetic is exact in float64, so these gaps are also asserted with `==`.
The same SCM drives evidence_without_injustice with X declared admissible:
freezing X leaves diff = 1.5*|a' - A| (unfair) or 0 (fair), and the paired
bootstrap over identical per-unit statistics gives a degenerate CI.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def group_frame() -> pd.DataFrame:
    """DGP 1: fixed confusion table (see module docstring)."""
    a = np.repeat([0, 1], 16)
    y = np.tile(np.repeat([1, 0], 8), 2)
    yhat = np.concatenate(
        [
            [1] * 6 + [0] * 2,  # group 0, y=1: TPR_0 = 0.75
            [1] * 2 + [0] * 6,  # group 0, y=0: FPR_0 = 0.25
            [1] * 2 + [0] * 6,  # group 1, y=1: TPR_1 = 0.25
            [1] * 4 + [0] * 4,  # group 1, y=0: FPR_1 = 0.50
        ]
    )
    # Dyadic feature (multiples of 1/16) for the audit's counterfactual leg.
    rng = np.random.default_rng(0)
    x = rng.integers(0, 16, size=32) / 16.0
    return pd.DataFrame({"A": a, "Y": y, "Y_hat": yhat, "X": x})


@pytest.fixture(scope="module")
def scm_frame() -> pd.DataFrame:
    """DGP 2: X dyadic and independent of A (non-descendant in the SCM)."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "A": np.tile([0, 1], 16),
            "X": rng.integers(0, 16, size=32) / 16.0,
        }
    )


def _unfair_predictor(d: pd.DataFrame) -> np.ndarray:
    # Direct dependence on A with dyadic coefficients -> exact 1.5 gap.
    return 0.5 * d["X"].to_numpy() + 1.5 * d["A"].to_numpy()


def _fair_predictor(d: pd.DataFrame) -> np.ndarray:
    return 0.5 * d["X"].to_numpy()


def _scm_flip(d: pd.DataFrame, value) -> pd.DataFrame:
    # do(A = value); X is a non-descendant of A, so nothing else updates.
    out = d.copy()
    out["A"] = value
    return out


# -------------------------------------------------------------------------
# demographic_parity — exact group-rate gap
# -------------------------------------------------------------------------


def test_demographic_parity_gap_exact(group_frame):
    res = sp.demographic_parity(
        group_frame, predictions="Y_hat", protected="A", threshold=0.1
    )
    # Hand computation: 8/16 = 0.5 vs 6/16 = 0.375, gap = 0.125 (dyadic -> exact).
    assert res.per_group[0] == 0.5
    assert res.per_group[1] == 0.375
    assert res.value == 0.125
    assert res.passes is False  # 0.125 > 0.1


# -------------------------------------------------------------------------
# equalized_odds — exact TPR/FPR gaps
# -------------------------------------------------------------------------


def test_equalized_odds_tpr_fpr_gaps_exact(group_frame):
    res = sp.equalized_odds(
        group_frame, predictions="Y_hat", labels="Y", protected="A", threshold=0.1
    )
    # Hand computation (all dyadic -> exact):
    #   TPR_0 = 6/8 = 0.75, TPR_1 = 2/8 = 0.25 -> TPR gap = 0.5
    #   FPR_0 = 2/8 = 0.25, FPR_1 = 4/8 = 0.50 -> FPR gap = 0.25
    #   gap = max(0.5, 0.25) = 0.5
    assert res.per_group["TPR[0]"] == 0.75
    assert res.per_group["TPR[1]"] == 0.25
    assert res.per_group["FPR[0]"] == 0.25
    assert res.per_group["FPR[1]"] == 0.5
    assert res.value == 0.5
    assert res.passes is False


# -------------------------------------------------------------------------
# fairness_audit — internal-consistency identity vs. standalone metrics
# -------------------------------------------------------------------------


def test_fairness_audit_components_match_standalone_exactly(group_frame):
    audit = sp.fairness_audit(
        group_frame,
        predictions="Y_hat",
        protected="A",
        labels="Y",
        predictor=_unfair_predictor,
        scm_intervention=_scm_flip,
        threshold=0.1,
    )
    dp = sp.demographic_parity(
        group_frame, predictions="Y_hat", protected="A", threshold=0.1
    )
    eo = sp.equalized_odds(
        group_frame, predictions="Y_hat", labels="Y", protected="A", threshold=0.1
    )
    cf = sp.counterfactual_fairness(
        group_frame,
        predictor=_unfair_predictor,
        protected="A",
        scm_intervention=_scm_flip,
        threshold=0.05,  # audit uses threshold / 2 for the CF leg
    )
    # The audit must reproduce the standalone metrics exactly (same code path,
    # same deterministic inputs — an identity, not an approximation).
    assert audit.n == len(group_frame) == 32
    assert audit.protected_attribute == "A"
    assert audit.demographic_parity.value == dp.value == 0.125
    assert audit.demographic_parity.per_group == dp.per_group
    assert audit.equalized_odds is not None
    assert audit.equalized_odds.value == eo.value == 0.5
    assert audit.equalized_odds.per_group == eo.per_group
    assert audit.counterfactual_fairness is not None
    assert audit.counterfactual_fairness.value == cf.value == 1.5
    assert audit.counterfactual_fairness.passes is cf.passes is False


# -------------------------------------------------------------------------
# orthogonal_to_bias — residuals numerically orthogonal to protected
# -------------------------------------------------------------------------


def test_orthogonal_to_bias_residuals_orthogonal_to_protected():
    rng = np.random.default_rng(2)
    n = 400
    a = rng.binomial(1, 0.5, size=n).astype(float)
    df = pd.DataFrame(
        {
            "A": a,
            "X1": 2.0 * a + rng.normal(0.0, 0.5, size=n),  # strongly A-loaded
            "X2": rng.normal(0.0, 1.0, size=n),
        }
    )
    out = sp.orthogonal_to_bias(df, features=["X1", "X2"], protected="A")
    # OLS residuals are orthogonal to the design (intercept + A), so the
    # sample correlation with A must vanish up to lstsq round-off.
    for col in ("X1", "X2"):
        corr = np.corrcoef(out[col].to_numpy(), df["A"].to_numpy())[0, 1]
        assert abs(corr) < 1e-8, (col, corr)
    # Non-feature columns pass through untouched.
    np.testing.assert_array_equal(out["A"].to_numpy(), df["A"].to_numpy())
    assert out.shape == df.shape


# -------------------------------------------------------------------------
# counterfactual_fairness — exact gap in both directions
# -------------------------------------------------------------------------


def test_counterfactual_fairness_unfair_predictor_exact_gap(scm_frame):
    res = sp.counterfactual_fairness(
        scm_frame,
        predictor=_unfair_predictor,
        protected="A",
        scm_intervention=_scm_flip,
        threshold=0.05,
    )
    # f = 0.5*X + 1.5*A and do(A=a') leaves X fixed, so every unit's
    # counterfactual change is exactly 1.5 (dyadic arithmetic -> exact).
    assert res.value == 1.5
    assert res.per_group[0] == 1.5
    assert res.per_group[1] == 1.5
    assert res.passes is False


def test_counterfactual_fairness_fair_predictor_zero_gap(scm_frame):
    res = sp.counterfactual_fairness(
        scm_frame,
        predictor=_fair_predictor,
        protected="A",
        scm_intervention=_scm_flip,
        threshold=0.05,
    )
    # f depends only on X, a non-descendant of A: the counterfactual
    # prediction is bit-identical to the factual one -> gap exactly 0.
    assert res.value == 0.0
    assert res.per_group[0] == 0.0
    assert res.per_group[1] == 0.0
    assert res.passes is True


# -------------------------------------------------------------------------
# evidence_without_injustice — degenerate bootstrap on exact statistics
# -------------------------------------------------------------------------


def test_evidence_without_injustice_flags_unfair_predictor(scm_frame):
    res = sp.evidence_without_injustice(
        scm_frame,
        _unfair_predictor,
        protected="A",
        admissible_features=["X"],
        scm_intervention=_scm_flip,
        n_boot=120,
        random_state=0,
    )
    # With X frozen at factual values, the counterfactual change reduces to
    # the direct A-term: 1.5*|a' - A| = 1.5 for every unit, exactly. Every
    # bootstrap replicate therefore also equals 1.5 -> degenerate CI.
    assert res.value == 1.5
    assert res.ci == (1.5, 1.5)
    assert res.per_group[0] == 1.5
    assert res.per_group[1] == 1.5
    assert res.passes is False  # CI upper 1.5 >= threshold 0.05
    assert res.n_boot == 120  # no bootstrap replicate may fail
    assert res.admissible_features == ["X"]
    # se(boots) == 0 exactly, so the normal-approx p-value is undefined.
    assert math.isnan(res.pvalue)


def test_evidence_without_injustice_passes_fair_predictor(scm_frame):
    res = sp.evidence_without_injustice(
        scm_frame,
        _fair_predictor,
        protected="A",
        admissible_features=["X"],
        scm_intervention=_scm_flip,
        n_boot=120,
        random_state=0,
    )
    # f uses only the admissible feature X, which is frozen: counterfactual
    # predictions are bit-identical -> T = 0 and CI = (0, 0), exactly.
    assert res.value == 0.0
    assert res.ci == (0.0, 0.0)
    assert res.passes is True  # CI upper 0 < threshold 0.05
    assert res.n_boot == 120
