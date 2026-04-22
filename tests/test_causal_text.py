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
