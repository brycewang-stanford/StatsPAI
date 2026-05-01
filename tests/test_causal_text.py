"""Tests for P1-B (causal_text MVP, experimental)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_text._common import (
    hash_embed_texts, embed_texts,
)
from statspai.exceptions import DataInsufficient, IdentificationFailure


# --------------------------------------------------------------------- #
#  Hash embedder
# --------------------------------------------------------------------- #


def test_hash_embed_returns_correct_shape():
    out = hash_embed_texts(["foo bar", "baz", ""], n_components=8)
    assert out.shape == (3, 8)


def test_hash_embed_deterministic():
    a = hash_embed_texts(["the quick brown fox"], n_components=16, seed=0)
    b = hash_embed_texts(["the quick brown fox"], n_components=16, seed=0)
    assert np.allclose(a, b)


def test_hash_embed_seed_changes_output():
    a = hash_embed_texts(["the quick brown fox"], n_components=16, seed=0)
    b = hash_embed_texts(["the quick brown fox"], n_components=16, seed=1)
    # Different seed -> different bucketing.
    assert not np.allclose(a, b)


def test_embed_texts_dispatches_to_callable():
    def custom(texts):
        return np.array([[len(t)] for t in texts], dtype=np.float64)

    out = embed_texts(["a", "abc"], embedder=custom)
    assert out.shape == (2, 1)
    assert out[0, 0] == 1.0 and out[1, 0] == 3.0


def test_embed_texts_unknown_embedder_raises():
    with pytest.raises(ValueError, match="Unknown embedder"):
        embed_texts(["x"], embedder="bogus")


# --------------------------------------------------------------------- #
#  text_treatment_effect (Veitch et al. MVP)
# --------------------------------------------------------------------- #


def _text_treatment_dgp(seed: int = 0, n: int = 500, true_ate: float = 1.5):
    """Synthetic: text contains keywords associated with treatment AND
    outcome; embedding should adjust for text-based confounding."""
    rng = np.random.default_rng(seed)
    keywords = ["great", "awesome", "love", "amazing"]
    texts, treats, outs = [], [], []
    for _ in range(n):
        positive_topic = rng.random() < 0.5
        if positive_topic:
            text = " ".join(rng.choice(keywords,
                                       size=rng.integers(1, 5)))
        else:
            text = "boring meh okay"
        treat = int(positive_topic) ^ int(rng.random() < 0.1)
        outcome = (true_ate * treat
                   + 0.5 * positive_topic
                   + 0.3 * rng.standard_normal())
        texts.append(text)
        treats.append(treat)
        outs.append(outcome)
    return pd.DataFrame({"text": texts, "treatment": treats,
                         "outcome": outs})


def test_text_treatment_recovers_synthetic_ate():
    df = _text_treatment_dgp(seed=0, n=600, true_ate=1.5)
    r = sp.text_treatment_effect(
        df, text_col="text", outcome="outcome", treatment="treatment",
        n_components=12,
    )
    # Estimate within 0.5 of truth (this is an MVP — not a tight bound)
    assert abs(r.estimate - 1.5) < 0.5, f"Estimate {r.estimate}"
    # Marked experimental
    assert r.diagnostics["status"] == "experimental"


def test_text_treatment_hash_embedder_deterministic():
    df = _text_treatment_dgp(seed=1, n=400)
    r1 = sp.text_treatment_effect(
        df, text_col="text", outcome="outcome", treatment="treatment",
        embedder="hash", n_components=8, seed=42,
    )
    r2 = sp.text_treatment_effect(
        df, text_col="text", outcome="outcome", treatment="treatment",
        embedder="hash", n_components=8, seed=42,
    )
    assert r1.estimate == r2.estimate
    assert r1.se == r2.se


def test_text_treatment_custom_embedder_honored():
    df = _text_treatment_dgp(seed=2, n=400)

    def custom(texts):
        # word-count single-feature embedder
        return np.array([[len(t.split())] for t in texts], dtype=np.float64)

    r = sp.text_treatment_effect(
        df, text_col="text", outcome="outcome", treatment="treatment",
        embedder=custom, n_components=1,
    )
    assert r.embedding_dim == 1
    assert r.embedder_name == "callable"


def test_text_treatment_too_few_rows_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "text": ["a", "b", "c"],
        "treatment": [0, 1, 0],
        "outcome": [1.0, 2.0, 3.0],
    })
    with pytest.raises(DataInsufficient):
        sp.text_treatment_effect(
            df, text_col="text", outcome="outcome",
            treatment="treatment", n_components=10,
        )


def test_text_treatment_missing_column_raises():
    df = _text_treatment_dgp()
    with pytest.raises(ValueError, match="not in data"):
        sp.text_treatment_effect(
            df, text_col="missing", outcome="outcome",
            treatment="treatment",
        )


def test_text_treatment_marks_experimental():
    df = _text_treatment_dgp()
    r = sp.text_treatment_effect(
        df, text_col="text", outcome="outcome", treatment="treatment",
        n_components=8,
    )
    assert r.diagnostics["status"] == "experimental"
    assert "text_diagnostics" in r.model_info


# --------------------------------------------------------------------- #
#  llm_annotator_correct (Egami et al. MVP)
# --------------------------------------------------------------------- #


def _annotator_dgp(seed: int = 0, n: int = 1500, n_val: int = 200,
                   misclass: float = 0.18, true_ate: float = 1.0):
    rng = np.random.default_rng(seed)
    T_true = (rng.random(n) > 0.5).astype(int)
    noise = (rng.random(n) < misclass).astype(int)
    T_llm = (T_true ^ noise).astype(int)
    y = true_ate * T_true + rng.standard_normal(n)
    human = pd.Series(
        [T_true[i] if i < n_val else np.nan for i in range(n)]
    )
    return pd.Series(T_llm), human, pd.Series(y)


def test_llm_annotator_corrects_known_bias():
    T_llm, T_human, y = _annotator_dgp(seed=0, true_ate=1.0,
                                       misclass=0.18)
    r = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human,
        outcome=y,
    )
    # Naive estimate is biased downward (~0.64 in the smoke test).
    assert r.naive_estimate < r.estimate
    # Corrected estimate is within 0.3 of truth (1.0).
    assert abs(r.estimate - 1.0) < 0.3, (
        f"Corrected {r.estimate}, naive {r.naive_estimate}"
    )
    # Diagnostics populated.
    diag = r.annotator_diagnostics
    assert 0.0 < diag["p_01"] < 0.5
    assert 0.0 < diag["p_10"] < 0.5
    assert diag["status"] == "experimental"


def test_llm_annotator_requires_validation_subset():
    T_llm, _, y = _annotator_dgp()
    with pytest.raises(DataInsufficient,
                       match="annotations_human"):
        sp.llm_annotator_correct(
            annotations_llm=T_llm, annotations_human=None,
            outcome=y,
        )


def test_llm_annotator_validation_too_small_raises():
    T_llm, T_human_full, y = _annotator_dgp(n_val=200)
    # Reduce validation to 5 rows.
    T_human_small = pd.Series(
        [T_human_full.iloc[i] if i < 5 else np.nan
         for i in range(len(T_human_full))]
    )
    with pytest.raises(DataInsufficient,
                       match="validation rows"):
        sp.llm_annotator_correct(
            annotations_llm=T_llm,
            annotations_human=T_human_small, outcome=y,
        )


def test_llm_annotator_severe_misclassification_raises():
    """If misclassification is so severe that 1-p01-p10 <= 0, the
    correction is not identified; we expect IdentificationFailure."""
    T_llm, T_human, y = _annotator_dgp(seed=1, n=1500,
                                       misclass=0.55)
    # With misclassification rate ~ 0.55, p_01+p_10 may exceed 1.
    # We accept either an IdentificationFailure or the result with
    # |correction_factor| being very small (close to zero).
    try:
        r = sp.llm_annotator_correct(
            annotations_llm=T_llm, annotations_human=T_human,
            outcome=y,
        )
        # If it didn't raise, the correction factor should be close
        # to zero — the test passes either way as long as the API
        # signals the danger.
        assert abs(r.correction_factor) < 0.2
    except IdentificationFailure:
        pass


def test_llm_annotator_unknown_method_raises():
    T_llm, T_human, y = _annotator_dgp()
    with pytest.raises(ValueError, match="Unknown method"):
        sp.llm_annotator_correct(
            annotations_llm=T_llm, annotations_human=T_human,
            outcome=y, method="bogus",
        )


def test_llm_annotator_with_covariates():
    rng = np.random.default_rng(2)
    n, n_val = 1200, 150
    T_true = (rng.random(n) > 0.5).astype(int)
    noise = (rng.random(n) < 0.15).astype(int)
    T_llm = (T_true ^ noise).astype(int)
    x = rng.standard_normal(n)
    y = 1.0 * T_true + 0.3 * x + rng.standard_normal(n)
    human = pd.Series(
        [T_true[i] if i < n_val else np.nan for i in range(n)]
    )
    r = sp.llm_annotator_correct(
        annotations_llm=pd.Series(T_llm),
        annotations_human=human, outcome=pd.Series(y),
        covariates=pd.DataFrame({"x": x}),
    )
    assert abs(r.estimate - 1.0) < 0.3


# --------------------------------------------------------------------- #
#  llm_annotator_correct — v1.7 deferred work
# --------------------------------------------------------------------- #


def _multiclass_dgp(seed: int = 0, n: int = 1500, n_val: int = 200,
                    K: int = 3):
    """K-class DGP with non-trivial confusion matrix and class-specific
    treatment effects (β_1 = 1.5, β_2 = -0.7 relative to ref class 0).
    """
    rng = np.random.default_rng(seed)
    M_true = np.array([[0.85, 0.10, 0.05],
                       [0.08, 0.80, 0.12],
                       [0.06, 0.09, 0.85]])
    T_true = rng.integers(0, K, size=n)
    T_llm = np.array([rng.choice(K, p=M_true[t]) for t in T_true])
    mu = np.where(T_true == 0, 0.0,
                  np.where(T_true == 1, 1.5, -0.7))
    y = mu + rng.standard_normal(n)
    human = pd.Series(
        [float(T_true[i]) if i < n_val else np.nan for i in range(n)]
    )
    return pd.Series(T_llm.astype(float)), human, pd.Series(y)


def test_llm_annotator_multiclass_recovers_effects():
    """K=3: head-line .estimate (class 1 vs ref 0) and the full per-class
    .detail vector should debias the naive (attenuated) coefficients."""
    T_llm, T_human, y = _multiclass_dgp(seed=0, n=2500, n_val=400)
    r = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
    )
    assert r.annotator_diagnostics["n_classes"] == 3
    assert abs(r.estimate - 1.5) < 0.4, (
        f"Headline estimate {r.estimate}; expected ~1.5"
    )
    # Per-class detail frame
    assert r.detail is not None
    assert len(r.detail) == 2
    classes = r.detail["class"].tolist()
    assert classes == [1.0, 2.0] or classes == [1, 2]
    # Naive estimates attenuate toward zero relative to corrected ones.
    naive_abs = np.abs(r.detail["naive_estimate"].values)
    corr_abs = np.abs(r.detail["corrected_estimate"].values)
    assert (corr_abs >= naive_abs - 1e-6).all(), (
        "Corrected magnitudes should not be smaller than naive ones."
    )


def test_llm_annotator_multiclass_diagnostics_populated():
    T_llm, T_human, y = _multiclass_dgp(seed=1)
    r = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
    )
    d = r.annotator_diagnostics
    # Confusion matrix shape and probability rows
    M = np.asarray(d["confusion_matrix"])
    assert M.shape == (3, 3)
    assert np.allclose(M.sum(axis=1), 1.0, atol=1e-9)
    # Bayes posterior columns sum to 1 wherever T_obs=j is non-empty.
    Q = np.asarray(d["q_posterior"])
    assert Q.shape == (3, 3)
    col_sums = Q.sum(axis=0)
    assert np.allclose(col_sums[col_sums > 0], 1.0, atol=1e-9)
    # Inflation factor is a finite >= 1 multiplier.
    infl = d["se_inflation_factor"]
    assert np.isfinite(infl) and infl >= 1.0 - 1e-9
    # Headline contrast labels the smallest non-reference class.
    assert "vs ref=" in d["headline_contrast"]


def test_llm_annotator_multiclass_singular_raises():
    """If the LLM is degenerate (e.g. always predicts class 0 in
    validation), the transform matrix is singular and the correction
    must signal IdentificationFailure rather than silently dividing."""
    rng = np.random.default_rng(7)
    n, n_val, K = 800, 150, 3
    T_true = rng.integers(0, K, size=n)
    T_llm = np.zeros(n, dtype=float)        # LLM says 0 for everyone
    y = (T_true == 1) * 1.5 + (T_true == 2) * -0.7 + rng.standard_normal(n)
    human = pd.Series(
        [float(T_true[i]) if i < n_val else np.nan for i in range(n)]
    )
    with pytest.raises(IdentificationFailure):
        sp.llm_annotator_correct(
            annotations_llm=pd.Series(T_llm),
            annotations_human=human, outcome=pd.Series(y),
        )


def test_llm_annotator_inflation_factor_binary():
    """Binary path exposes a finite SE inflation factor >= 1, monotone
    in validation-set noise (smaller n_val => larger inflation)."""
    rng = np.random.default_rng(0)
    n = 2000
    T_true = (rng.random(n) > 0.5).astype(int)
    noise = (rng.random(n) < 0.15).astype(int)
    T_llm = (T_true ^ noise).astype(int)
    y = 1.0 * T_true + rng.standard_normal(n)

    def _infl(n_val):
        human = pd.Series(
            [T_true[i] if i < n_val else np.nan for i in range(n)]
        )
        r = sp.llm_annotator_correct(
            annotations_llm=pd.Series(T_llm),
            annotations_human=human, outcome=pd.Series(y),
        )
        return r.annotator_diagnostics["se_inflation_factor"]

    infl_small = _infl(50)
    infl_large = _infl(500)
    assert infl_small >= 1.0
    assert infl_large >= 1.0
    # Smaller validation set => more validation-set noise => larger
    # inflation. Allow a small tolerance for sampling noise.
    assert infl_small > infl_large - 1e-3


def test_llm_annotator_bootstrap_widens_ci():
    """Bias-corrected bootstrap must produce a CI at least as wide as
    the first-order CI in the binary case where validation noise
    matters."""
    T_llm, T_human, y = _annotator_dgp(seed=0, n=1500, n_val=150,
                                       misclass=0.18)
    r_fo = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
    )
    r_b = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
        bootstrap=True, n_bootstrap=300, bootstrap_seed=42,
    )
    fo_width = r_fo.ci[1] - r_fo.ci[0]
    b_width = r_b.ci[1] - r_b.ci[0]
    assert b_width >= fo_width * 0.9, (
        f"Bootstrap CI width {b_width:.4f} should be >= 0.9 * first-"
        f"order width {fo_width:.4f}"
    )
    # Bootstrap-aware metadata in diagnostics.
    d = r_b.annotator_diagnostics
    assert d["se_correction"] == "bias_corrected_bootstrap"
    assert d["bootstrap"]["n_bootstrap"] == 300
    assert d["bootstrap"]["seed"] == 42
    assert d["bootstrap"]["method"] == "bias_corrected_percentile"
    # First-order SE/CI still available for inspection.
    assert d["first_order_se"] == r_fo.se
    assert d["first_order_ci"] == r_fo.ci


def test_llm_annotator_bootstrap_reproducible():
    T_llm, T_human, y = _annotator_dgp(seed=0, n=1200, n_val=150)
    r1 = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
        bootstrap=True, n_bootstrap=200, bootstrap_seed=0,
    )
    r2 = sp.llm_annotator_correct(
        annotations_llm=T_llm, annotations_human=T_human, outcome=y,
        bootstrap=True, n_bootstrap=200, bootstrap_seed=0,
    )
    assert r1.ci == r2.ci
    assert r1.se == r2.se


def test_llm_annotator_bootstrap_too_few_replicates_raises():
    T_llm, T_human, y = _annotator_dgp(seed=0)
    with pytest.raises(ValueError, match="too small"):
        sp.llm_annotator_correct(
            annotations_llm=T_llm, annotations_human=T_human, outcome=y,
            bootstrap=True, n_bootstrap=10,
        )


# --------------------------------------------------------------------- #
#  Registry / agent surface
# --------------------------------------------------------------------- #


def test_text_treatment_registered():
    assert "text_treatment_effect" in sp.list_functions()
    spec = sp.describe_function("text_treatment_effect")
    assert spec["category"] == "causal_text"


def test_llm_annotator_registered():
    assert "llm_annotator_correct" in sp.list_functions()
    spec = sp.describe_function("llm_annotator_correct")
    assert spec["category"] == "causal_text"
    assert any("Egami" in r for r in [spec.get("reference", "")])


def test_top_level_imports_present():
    assert hasattr(sp, "text_treatment_effect")
    assert hasattr(sp, "llm_annotator_correct")
    assert hasattr(sp, "TextTreatmentResult")
    assert hasattr(sp, "LLMAnnotatorResult")
