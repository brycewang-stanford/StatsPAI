"""Tests for the continuous (regression-calibration) path of
``sp.llm_annotator_correct`` — the AI-labeled-regressor attenuation fix.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import IdentificationFailure, NumericalInstability


def _make_continuous(n=400, n_audit=50, beta=30.0, noise=0.45, seed=42):
    rng = np.random.default_rng(seed)
    s_true = rng.uniform(-1, 1, n)
    sales = 100 + beta * s_true + rng.normal(0, 6, n)
    s_ai = np.clip(s_true + rng.normal(0, noise, n), -1, 1)
    audit = rng.choice(n, n_audit, replace=False)
    human = pd.Series([np.nan] * n, dtype=float)
    human.iloc[audit] = s_true[audit]
    return pd.Series(s_ai), human, pd.Series(sales), audit, s_true


class TestContinuousCorrectness:
    def test_recovers_attenuated_slope(self):
        s_ai, human, sales, _, _ = _make_continuous()
        r = sp.llm_annotator_correct(
            annotations_llm=s_ai, annotations_human=human, outcome=sales
        )
        # Naive slope is attenuated well below 30; the correction must
        # move it back toward the truth.
        assert r.naive_estimate < 27.0
        assert abs(r.estimate - 30.0) < abs(r.naive_estimate - 30.0)
        assert r.estimand == "slope"
        assert r.model_info["annotation_type"] == "continuous"

    def test_no_covariates_equals_reliability_ratio(self):
        """With no covariates, regression calibration reduces exactly
        to dividing the naive slope by the calibration slope."""
        s_ai, human, sales, _, _ = _make_continuous()
        r = sp.llm_annotator_correct(
            annotations_llm=s_ai, annotations_human=human, outcome=sales
        )
        gamma1 = r.model_info["calibration_slope"]
        assert r.estimate == pytest.approx(r.naive_estimate / gamma1, rel=1e-10)
        # And the reported unconditional reliability equals gamma1 here.
        assert r.model_info["reliability"] == pytest.approx(gamma1, rel=1e-10)

    def test_covariates_path_runs(self):
        s_ai, human, sales, _, _ = _make_continuous()
        rng = np.random.default_rng(0)
        cov = pd.DataFrame({"size": rng.normal(size=len(s_ai))})
        r = sp.llm_annotator_correct(
            annotations_llm=s_ai,
            annotations_human=human,
            outcome=sales,
            covariates=cov,
        )
        assert np.isfinite(r.estimate)
        assert np.isfinite(r.se)

    def test_bootstrap_ci_covers_truth(self):
        # Larger audit sample so the reliability estimate itself is not
        # the binding source of noise for this fixed seed.
        s_ai, human, sales, _, _ = _make_continuous(n=800, n_audit=200)
        r = sp.llm_annotator_correct(
            annotations_llm=s_ai,
            annotations_human=human,
            outcome=sales,
            bootstrap=True,
            n_bootstrap=200,
            bootstrap_seed=7,
        )
        lo, hi = r.ci
        assert lo < 30.0 < hi
        assert lo < r.estimate < hi
        assert r.model_info["se_correction"] == "bias_corrected_bootstrap"

    def test_explicit_reliability_method(self):
        s_ai, human, sales, _, _ = _make_continuous()
        r_auto = sp.llm_annotator_correct(
            annotations_llm=s_ai, annotations_human=human, outcome=sales
        )
        r_rel = sp.llm_annotator_correct(
            annotations_llm=s_ai,
            annotations_human=human,
            outcome=sales,
            method="reliability",
        )
        assert r_rel.estimate == pytest.approx(r_auto.estimate, rel=1e-12)

    def test_summary_renders_continuous_block(self):
        s_ai, human, sales, _, _ = _make_continuous()
        r = sp.llm_annotator_correct(
            annotations_llm=s_ai, annotations_human=human, outcome=sales
        )
        text = r.summary()
        assert "continuous score" in text
        assert "Reliability" in text
        assert "N classes" not in text


class TestContinuousBoundaries:
    def test_hausman_on_continuous_raises(self):
        s_ai, human, sales, _, _ = _make_continuous()
        with pytest.raises(ValueError, match="continuous"):
            sp.llm_annotator_correct(
                annotations_llm=s_ai,
                annotations_human=human,
                outcome=sales,
                method="hausman",
            )

    def test_unknown_method_raises(self):
        s_ai, human, sales, _, _ = _make_continuous()
        with pytest.raises(ValueError, match="Unknown method"):
            sp.llm_annotator_correct(
                annotations_llm=s_ai,
                annotations_human=human,
                outcome=sales,
                method="magic",
            )

    def test_constant_llm_scores_raise(self):
        n = 100
        rng = np.random.default_rng(0)
        human = pd.Series([np.nan] * n, dtype=float)
        human.iloc[:40] = rng.uniform(-1, 1, 40)
        with pytest.raises(NumericalInstability):
            sp.llm_annotator_correct(
                annotations_llm=pd.Series(np.full(n, 0.5)),
                annotations_human=human,
                outcome=pd.Series(rng.normal(size=n)),
                method="reliability",
            )

    def test_uninformative_scores_raise_identification(self):
        """LLM scores uncorrelated with the truth: calibration slope
        <= 0 must fail loudly, not return a wild number."""
        n = 400
        rng = np.random.default_rng(3)
        s_true = rng.uniform(-1, 1, n)
        s_ai = -s_true + rng.normal(0, 0.1, n)  # anti-correlated
        sales = 100 + 30 * s_true + rng.normal(0, 6, n)
        human = pd.Series([np.nan] * n, dtype=float)
        human.iloc[:50] = s_true[:50]
        with pytest.raises(IdentificationFailure):
            sp.llm_annotator_correct(
                annotations_llm=pd.Series(s_ai),
                annotations_human=human,
                outcome=pd.Series(sales),
            )

    def test_binary_path_unchanged(self):
        """Discrete labels must still route to the Hausman path."""
        n, n_val = 1000, 100
        rng = np.random.default_rng(0)
        t_true = (rng.random(n) > 0.5).astype(int)
        noise = (rng.random(n) < 0.15).astype(int)
        t_llm = (t_true ^ noise).astype(int)
        y = 1.0 * t_true + rng.standard_normal(n)
        human = pd.Series([float(t_true[i]) if i < n_val else np.nan for i in range(n)])
        r = sp.llm_annotator_correct(
            annotations_llm=pd.Series(t_llm.astype(float)),
            annotations_human=human,
            outcome=pd.Series(y),
        )
        assert r.estimand == "ATE"
        assert "p_01" in r.model_info
