"""
Tests for sp.fairness — counterfactual fairness and bias audits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_audit_data(*, n: int = 800, bias: float = 0.2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = rng.binomial(1, 0.5, size=n)
    X = rng.normal(0, 1, size=n)
    # Ground-truth positive rates identical across groups → equalized odds
    # violated if Y_hat depends on A directly.
    Y = rng.binomial(1, 0.3, size=n)
    # Predictions: base classifier that depends on X, with A-dependent bias.
    logit = 0.7 * X + bias * A
    p = 1.0 / (1.0 + np.exp(-logit))
    Y_hat = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"A": A, "X": X, "Y": Y, "Y_hat": Y_hat})


# -------------------------------------------------------------------------
# demographic_parity
# -------------------------------------------------------------------------


def test_demographic_parity_flags_biased_classifier():
    df = _make_audit_data(bias=1.5, seed=3)
    res = sp.demographic_parity(df, predictions="Y_hat", protected="A")
    assert res.metric == "demographic_parity"
    assert res.value > 0.1
    assert res.passes is False
    assert set(res.per_group.keys()) == {0, 1}


def test_demographic_parity_unbiased_classifier_passes():
    df = _make_audit_data(bias=0.0, seed=5)
    res = sp.demographic_parity(df, predictions="Y_hat", protected="A",
                                 threshold=0.1)
    # Stochastic — but with n=800 and bias=0 we expect gap < 0.1 most of the time.
    assert res.value < 0.15


def test_demographic_parity_single_group_errors():
    df = pd.DataFrame({"Y_hat": [0, 1, 1], "A": [1, 1, 1]})
    with pytest.raises(ValueError, match="only one level"):
        sp.demographic_parity(df, predictions="Y_hat", protected="A")


def test_demographic_parity_non_binary_pred_errors():
    df = pd.DataFrame({"Y_hat": [0.3, 0.5, 0.7], "A": [0, 1, 1]})
    with pytest.raises(ValueError, match="must be binary"):
        sp.demographic_parity(df, predictions="Y_hat", protected="A")


# -------------------------------------------------------------------------
# equalized_odds
# -------------------------------------------------------------------------


def test_equalized_odds_decomposes_into_tpr_fpr():
    df = _make_audit_data(bias=1.5, seed=7)
    res = sp.equalized_odds(df, predictions="Y_hat", labels="Y", protected="A")
    assert res.metric == "equalized_odds"
    # TPR/FPR per group should be recorded.
    keys = set(res.per_group.keys())
    assert any(k.startswith("TPR[") for k in keys)
    assert any(k.startswith("FPR[") for k in keys)


def test_equalized_odds_requires_both_labels_in_each_group():
    df = pd.DataFrame({
        "Y_hat": [1, 1, 0], "Y": [1, 1, 1], "A": [0, 0, 1],
    })  # Group 1 has no negatives, so FPR is undefined for group 1.
    with pytest.raises(ValueError, match="positive and one negative"):
        sp.equalized_odds(df, predictions="Y_hat", labels="Y", protected="A")


# -------------------------------------------------------------------------
# counterfactual_fairness
# -------------------------------------------------------------------------


def test_counterfactual_fairness_detects_direct_dependence():
    rng = np.random.default_rng(11)
    n = 500
    A = rng.binomial(1, 0.5, size=n)
    X = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"A": A, "X": X})

    def biased_predictor(d: pd.DataFrame) -> np.ndarray:
        """Classifier that depends directly on A — violates CF."""
        return 0.5 * d["X"].to_numpy() + 2.0 * d["A"].to_numpy()

    def scm_flip(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        # No descendants updated — X is independent of A by construction.
        return out

    res = sp.counterfactual_fairness(
        df, predictor=biased_predictor, protected="A",
        scm_intervention=scm_flip, threshold=0.05,
    )
    assert res.value > 0.5  # |2 * 1| = 2 on flipped units, averaged with 0.
    assert res.passes is False


def test_counterfactual_fairness_unbiased_predictor_passes():
    rng = np.random.default_rng(13)
    n = 500
    A = rng.binomial(1, 0.5, size=n)
    X = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"A": A, "X": X})

    def fair_predictor(d: pd.DataFrame) -> np.ndarray:
        return 0.5 * d["X"].to_numpy()

    def scm_flip(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        return out

    res = sp.counterfactual_fairness(
        df, predictor=fair_predictor, protected="A",
        scm_intervention=scm_flip, threshold=0.01,
    )
    assert res.value < 1e-10  # Predictor doesn't use A at all.
    assert res.passes is True


def test_counterfactual_fairness_predictor_length_mismatch_errors():
    df = pd.DataFrame({"A": [0, 1, 1], "X": [0.0, 1.0, 2.0]})
    def bad_predictor(d): return np.array([0.0])
    def scm(d, v):
        out = d.copy(); out["A"] = v; return out
    with pytest.raises(ValueError, match="one value per row"):
        sp.counterfactual_fairness(
            df, predictor=bad_predictor, protected="A",
            scm_intervention=scm,
        )


# -------------------------------------------------------------------------
# orthogonal_to_bias
# -------------------------------------------------------------------------


def test_orthogonal_to_bias_removes_correlation():
    rng = np.random.default_rng(17)
    n = 1000
    A = rng.binomial(1, 0.5, size=n).astype(float)
    # X1 heavily correlated with A
    X1 = 2.0 * A + rng.normal(0, 0.5, size=n)
    X2 = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"A": A, "X1": X1, "X2": X2})
    out = sp.orthogonal_to_bias(df, features=["X1", "X2"], protected="A")
    # Residualized X1 should be (numerically) uncorrelated with A.
    corr = np.corrcoef(out["X1"].to_numpy(), out["A"].to_numpy())[0, 1]
    assert abs(corr) < 1e-8, corr
    # A column unchanged.
    np.testing.assert_allclose(out["A"].to_numpy(), df["A"].to_numpy())


def test_orthogonal_to_bias_categorical_protected():
    rng = np.random.default_rng(19)
    n = 600
    A = rng.choice(["a", "b", "c"], size=n)
    X = np.where(A == "a", 0.0, np.where(A == "b", 1.0, 2.0)) + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"A": A, "X": X})
    out = sp.orthogonal_to_bias(df, features=["X"], protected="A")
    # Residual mean per group should be ~0.
    for g in ["a", "b", "c"]:
        assert abs(out.loc[out["A"] == g, "X"].mean()) < 1e-8


# -------------------------------------------------------------------------
# fairness_audit
# -------------------------------------------------------------------------


def test_fairness_audit_combines_metrics():
    df = _make_audit_data(bias=1.5, seed=23)
    audit = sp.fairness_audit(
        df, predictions="Y_hat", protected="A", labels="Y",
    )
    assert audit.demographic_parity.metric == "demographic_parity"
    assert audit.equalized_odds is not None
    assert audit.counterfactual_fairness is None  # no predictor supplied
    assert audit.n == len(df)
    assert "Fairness Audit" in audit.summary()


def test_fairness_in_registry():
    fns = set(sp.list_functions())
    assert "counterfactual_fairness" in fns
    assert "orthogonal_to_bias" in fns
    assert "demographic_parity" in fns
    assert "equalized_odds" in fns
    assert "fairness_audit" in fns
