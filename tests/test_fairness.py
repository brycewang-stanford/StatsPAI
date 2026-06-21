"""
Tests for sp.fairness — counterfactual fairness and bias audits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)


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
    res = sp.demographic_parity(df, predictions="Y_hat", protected="A", threshold=0.1)
    # Stochastic — but with n=800 and bias=0 we expect gap < 0.1 most of the time.
    assert res.value < 0.15


def test_demographic_parity_single_group_errors():
    df = pd.DataFrame({"Y_hat": [0, 1, 1], "A": [1, 1, 1]})
    with pytest.raises(DataInsufficient, match="only one level"):
        sp.demographic_parity(df, predictions="Y_hat", protected="A")


def test_demographic_parity_non_binary_pred_errors():
    df = pd.DataFrame({"Y_hat": [0.3, 0.5, 0.7], "A": [0, 1, 1]})
    with pytest.raises(MethodIncompatibility, match="must be binary"):
        sp.demographic_parity(df, predictions="Y_hat", protected="A")


def test_demographic_parity_rejects_bad_threshold_and_missing_column():
    df = pd.DataFrame({"Y_hat": [0, 1, 1], "A": [0, 1, 1]})
    with pytest.raises(MethodIncompatibility, match="threshold"):
        sp.demographic_parity(df, predictions="Y_hat", protected="A", threshold=np.nan)
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.demographic_parity(df, predictions="missing", protected="A")


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
    df = pd.DataFrame(
        {
            "Y_hat": [1, 1, 0],
            "Y": [1, 1, 1],
            "A": [0, 0, 1],
        }
    )  # Group 1 has no negatives, so FPR is undefined for group 1.
    with pytest.raises(DataInsufficient, match="positive and one negative"):
        sp.equalized_odds(df, predictions="Y_hat", labels="Y", protected="A")


# -------------------------------------------------------------------------
# counterfactual_fairness
# -------------------------------------------------------------------------


def test_counterfactual_fairness_detects_direct_dependence():
    rng = np.random.default_rng(11)
    n = 500
    A = np.tile([0, 1], n // 2)
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
        df,
        predictor=biased_predictor,
        protected="A",
        scm_intervention=scm_flip,
        threshold=0.05,
    )
    np.testing.assert_allclose(res.value, 2.0)
    np.testing.assert_allclose([res.per_group[0], res.per_group[1]], [2.0, 2.0])
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
        df,
        predictor=fair_predictor,
        protected="A",
        scm_intervention=scm_flip,
        threshold=0.01,
    )
    np.testing.assert_allclose(res.value, 0.0)
    assert res.passes is True


def test_counterfactual_fairness_predictor_length_mismatch_errors():
    df = pd.DataFrame({"A": [0, 1, 1], "X": [0.0, 1.0, 2.0]})

    def bad_predictor(d):
        return np.array([0.0])

    def scm(d, v):
        out = d.copy()
        out["A"] = v
        return out

    with pytest.raises(MethodIncompatibility, match="one value per row"):
        sp.counterfactual_fairness(
            df,
            predictor=bad_predictor,
            protected="A",
            scm_intervention=scm,
        )


def test_counterfactual_fairness_validates_predictor_and_scm_outputs():
    df = pd.DataFrame({"A": [0, 1, 1], "X": [0.0, 1.0, 2.0]})

    def nonfinite_predictor(d):
        return np.array([0.0, np.inf, 1.0])

    def vector_predictor(d):
        return d["X"].to_numpy()

    def scm(d, value):
        out = d.copy()
        out["A"] = value
        return out

    def bad_scm(d, value):
        return {"A": value}

    with pytest.raises(NumericalInstability, match="non-finite"):
        sp.counterfactual_fairness(
            df,
            predictor=nonfinite_predictor,
            protected="A",
            scm_intervention=scm,
        )
    with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
        sp.counterfactual_fairness(
            df,
            predictor=vector_predictor,
            protected="A",
            scm_intervention=bad_scm,
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
    X = np.where(A == "a", 0.0, np.where(A == "b", 1.0, 2.0))
    X = X + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"A": A, "X": X})
    out = sp.orthogonal_to_bias(df, features=["X"], protected="A")
    # Residual mean per group should be ~0.
    for g in ["a", "b", "c"]:
        assert abs(out.loc[out["A"] == g, "X"].mean()) < 1e-8


def test_orthogonal_to_bias_accepts_scalar_feature_and_rejects_bad_inputs():
    df = pd.DataFrame({"A": [0, 0, 1, 1], "X": [0.0, 1.0, 2.0, 3.0]})
    out = sp.orthogonal_to_bias(df, features="X", protected="A")
    assert list(out.columns) == ["A", "X"]
    with pytest.raises(MethodIncompatibility, match="Unknown method"):
        sp.orthogonal_to_bias(df, features=["X"], protected="A", method="drop")
    with pytest.raises(NumericalInstability, match="non-finite"):
        sp.orthogonal_to_bias(
            pd.DataFrame({"A": [0, 1], "X": [0.0, np.inf]}),
            features=["X"],
            protected="A",
        )


# -------------------------------------------------------------------------
# fairness_audit
# -------------------------------------------------------------------------


def test_fairness_audit_combines_metrics():
    df = _make_audit_data(bias=1.5, seed=23)
    audit = sp.fairness_audit(
        df,
        predictions="Y_hat",
        protected="A",
        labels="Y",
    )
    assert audit.demographic_parity.metric == "demographic_parity"
    assert audit.equalized_odds is not None
    assert audit.counterfactual_fairness is None  # no predictor supplied
    assert audit.n == len(df)
    assert "Fairness Audit" in audit.summary()


# -------------------------------------------------------------------------
# evidence_without_injustice
# -------------------------------------------------------------------------


def _make_ewi_data(n: int = 120, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = np.tile([0, 1], n // 2)
    credit = 600 + 80 * A + rng.normal(0, 10, size=n)
    x = rng.normal(size=n)
    return pd.DataFrame({"A": A, "credit": credit, "x": x})


def test_evidence_without_injustice_passes_when_admissible_path_is_frozen():
    df = _make_ewi_data()

    def predictor(d: pd.DataFrame) -> np.ndarray:
        return d["credit"].to_numpy() / 1000.0

    def scm(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        out["credit"] = 600 + 80 * value + (d["credit"] - (600 + 80 * d["A"]))
        return out

    res = sp.fairness.evidence_without_injustice(
        df,
        predictor,
        protected="A",
        admissible_features="credit",
        scm_intervention=scm,
        n_boot=120,
        random_state=0,
    )

    assert res.value == pytest.approx(0.0)
    assert res.ci == pytest.approx((0.0, 0.0))
    assert res.passes is True
    assert res.admissible_features == ["credit"]


def test_evidence_without_injustice_validation_uses_taxonomy():
    df = _make_ewi_data()

    def predictor(d: pd.DataFrame) -> np.ndarray:
        return d["x"].to_numpy()

    def scm(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        return out

    with pytest.raises(MethodIncompatibility, match="protected") as excinfo:
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="missing",
            admissible_features=[],
            scm_intervention=scm,
        )
    assert isinstance(excinfo.value, ValueError)
    assert "available_columns" in excinfo.value.diagnostics

    with pytest.raises(MethodIncompatibility, match="admissible_features"):
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="A",
            admissible_features=["missing"],
            scm_intervention=scm,
        )
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=scm,
            alpha=0,
        )
    with pytest.raises(MethodIncompatibility, match="threshold"):
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=scm,
            threshold=np.nan,
        )
    with pytest.raises(MethodIncompatibility, match="n_boot"):
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=scm,
            n_boot=10,
        )


def test_evidence_without_injustice_predictor_and_scm_contracts():
    df = _make_ewi_data()

    def bad_len_predictor(d: pd.DataFrame) -> np.ndarray:
        return np.ones(max(len(d) - 1, 0))

    def nonfinite_predictor(d: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(d))
        out[0] = np.inf
        return out

    def good_predictor(d: pd.DataFrame) -> np.ndarray:
        return d["x"].to_numpy()

    def good_scm(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        return out

    with pytest.raises(MethodIncompatibility, match="wrong length"):
        sp.fairness.evidence_without_injustice(
            df,
            bad_len_predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=good_scm,
        )
    with pytest.raises(NumericalInstability, match="non-finite"):
        sp.fairness.evidence_without_injustice(
            df,
            nonfinite_predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=good_scm,
        )

    def bad_scm_type(d: pd.DataFrame, value):
        return {"A": value}

    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        sp.fairness.evidence_without_injustice(
            df,
            good_predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=bad_scm_type,
        )

    def bad_scm_length(d: pd.DataFrame, value) -> pd.DataFrame:
        return d.iloc[:-1].copy()

    with pytest.raises(MethodIncompatibility, match="length mismatch"):
        sp.fairness.evidence_without_injustice(
            df,
            good_predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=bad_scm_length,
        )


def test_evidence_without_injustice_single_level_and_bootstrap_failures():
    df = _make_ewi_data()

    def predictor(d: pd.DataFrame) -> np.ndarray:
        return d["x"].to_numpy()

    def scm(d: pd.DataFrame, value) -> pd.DataFrame:
        out = d.copy()
        out["A"] = value
        return out

    with pytest.raises(DataInsufficient, match="only one level"):
        sp.fairness.evidence_without_injustice(
            df.assign(A=1),
            predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=scm,
        )

    calls = {"n": 0}

    def scm_fails_after_observed_stat(d: pd.DataFrame, value):
        calls["n"] += 1
        if calls["n"] > 2:
            return {"bad": value}
        out = d.copy()
        out["A"] = value
        return out

    with pytest.raises(ConvergenceFailure, match="bootstrap only produced") as excinfo:
        sp.fairness.evidence_without_injustice(
            df,
            predictor,
            protected="A",
            admissible_features=[],
            scm_intervention=scm_fails_after_observed_stat,
            n_boot=99,
        )
    assert isinstance(excinfo.value, RuntimeError)
    assert excinfo.value.diagnostics == {"n_ok": 0, "n_boot": 99}


def test_fairness_in_registry():
    fns = set(sp.list_functions())
    assert "counterfactual_fairness" in fns
    assert "orthogonal_to_bias" in fns
    assert "demographic_parity" in fns
    assert "equalized_odds" in fns
    assert "fairness_audit" in fns
