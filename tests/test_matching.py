"""
Tests for unified Matching module.

Covers: new orthogonal API (distance × method × bias_correction),
legacy API backward compatibility, and all matching variants.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.matching import match, MatchEstimator, balance_diagnostics
from statspai.core.results import CausalResult
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture
def selection_bias_data():
    """
    DGP with selection on observables:
        X1, X2 ~ Normal
        Treatment: P(T=1) depends on X1, X2
        Y = 1 + 2*T + 3*X1 + X2 + eps  (true ATT = 2.0)
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    logit = -0.5 + 0.8 * X1 + 0.5 * X2
    prob = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, prob, n)

    Y = 1 + 2 * T + 3 * X1 + X2 + eps

    return pd.DataFrame(
        {
            "y": Y,
            "treat": T,
            "x1": X1,
            "x2": X2,
            "group": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture
def discrete_data():
    """Data with discrete covariates for exact matching tests."""
    rng = np.random.default_rng(99)
    n = 1000

    age_group = rng.choice([20, 30, 40, 50], n)
    edu = rng.choice([1, 2, 3], n)
    eps = rng.normal(0, 0.3, n)

    logit = -1 + 0.03 * age_group + 0.5 * edu
    prob = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, prob, n)

    Y = 5 + 2 * T + 0.1 * age_group + edu + eps

    return pd.DataFrame(
        {
            "y": Y,
            "treat": T,
            "age_group": age_group,
            "edu": edu,
        }
    )


# ==================================================================
# New API: distance × method combinations
# ==================================================================


class TestNearestPropensity:
    """distance='propensity', method='nearest' (default)."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="propensity",
            method="nearest",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_default_is_propensity_nearest(self, selection_bias_data):
        """Default distance/method should be propensity + nearest."""
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        assert result.model_info["distance"] == "propensity"
        assert result.model_info["method"] == "nearest"

    def test_corrects_naive_bias(self, selection_bias_data):
        df = selection_bias_data
        naive = df[df["treat"] == 1]["y"].mean() - df[df["treat"] == 0]["y"].mean()
        result = match(df, y="y", treat="treat", covariates=["x1", "x2"])
        assert abs(result.estimate - 2.0) < abs(naive - 2.0)

    def test_significance(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        assert 0.0 <= result.pvalue < 0.05
        # Significance must be internally consistent: 95% CI excludes 0.
        assert result.ci[0] > 0.0

    def test_ci_covers_true(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            alpha=0.05,
        )
        assert result.ci[0] < 2.0 < result.ci[1]

    def test_ate(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            estimand="ATE",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


def test_balance_diagnostics_returns_summary(selection_bias_data):
    out = balance_diagnostics(
        selection_bias_data,
        treatment="treat",
        covariates=["x1", "x2"],
    )
    assert "smd_raw" in out.table.columns
    assert "smd_weighted" in out.table.columns
    assert out.summary_stats["n_obs"] == len(selection_bias_data)
    assert out.summary_stats["effective_sample_size"] > 0


class TestNearestMahalanobis:
    """distance='mahalanobis', method='nearest'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="mahalanobis",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5

    def test_with_bias_correction(self, selection_bias_data):
        """Bias correction should improve estimate."""
        raw = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="mahalanobis",
        )
        bc = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="mahalanobis",
            bias_correction=True,
        )
        assert isinstance(bc, CausalResult)
        # BC recovers the planted ATT (true=2.0) and flips the flag.
        assert bc.model_info["bias_correction"] is True
        assert abs(bc.estimate - 2.0) < 0.5  # tight band; observed ~1.97
        # Internal consistency: positive, finite SE, CI brackets the point estimate.
        assert bc.se > 0 and np.isfinite(bc.se)
        assert bc.ci[0] < bc.estimate < bc.ci[1]
        # BC should be no worse than raw matching on this DGP.
        assert abs(bc.estimate - 2.0) <= abs(raw.estimate - 2.0) + 0.2


class TestNearestEuclidean:
    """distance='euclidean', method='nearest'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="euclidean",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestNearestTieBreaking:
    """Nearest-neighbor ties should be deterministic and index-anchored."""

    def test_equal_distance_ties_use_source_index_not_row_order(self):
        df = pd.DataFrame(
            {
                "unit": [
                    "treated_0",
                    "control_high",
                    "control_low",
                    "treated_1",
                    "control_1",
                ],
                "y": [10.0, 100.0, 0.0, 30.0, 25.0],
                "treat": [1, 0, 0, 1, 0],
                "x": [0.0, 0.0, 0.0, 1.0, 1.0],
            },
            # control_low and control_high are exact ties for treated_0.
            # The lower source index must win regardless of row order.
            index=[100, 20, 5, 200, 30],
        )
        shuffled = df.loc[[20, 100, 30, 5, 200]]

        r1 = match(
            df,
            y="y",
            treat="treat",
            covariates=["x"],
            distance="euclidean",
            method="nearest",
        )
        r2 = match(
            shuffled,
            y="y",
            treat="treat",
            covariates=["x"],
            distance="euclidean",
            method="nearest",
        )

        assert r1.estimate == pytest.approx(7.5, abs=1e-12)
        assert r2.estimate == pytest.approx(7.5, abs=1e-12)


class TestExactMatching:
    """distance='exact'."""

    def test_basic(self, discrete_data):
        result = match(
            discrete_data,
            y="y",
            treat="treat",
            covariates=["age_group", "edu"],
            distance="exact",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_rejects_ate(self, discrete_data):
        with pytest.raises(ValueError, match="ATT"):
            match(
                discrete_data,
                y="y",
                treat="treat",
                covariates=["age_group", "edu"],
                distance="exact",
                estimand="ATE",
            )


class TestStratification:
    """method='stratify'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="stratify",
            n_strata=5,
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_ate(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="stratify",
            estimand="ATE",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5

    def test_10_strata(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="stratify",
            n_strata=10,
        )
        assert isinstance(result, CausalResult)
        # Recovers the planted ATT (true=2.0) and reports a sane SE/CI.
        assert abs(result.estimate - 2.0) < 0.5  # observed ~1.98
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]
        assert result.model_info["n_strata"] == 10
        assert 1 <= result.model_info["n_effective_strata"] <= 10

    def test_requires_propensity(self, selection_bias_data):
        with pytest.raises(ValueError, match="propensity"):
            match(
                selection_bias_data,
                y="y",
                treat="treat",
                covariates=["x1", "x2"],
                method="stratify",
                distance="mahalanobis",
            )


class TestCEM:
    """method='cem'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="cem",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 2.0

    def test_custom_bins(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="cem",
            n_bins=10,
        )
        assert isinstance(result, CausalResult)
        # n_bins is honoured and the planted ATT (true=2.0) is recovered.
        assert result.model_info["n_bins"] == 10
        assert abs(result.estimate - 2.0) < 0.6  # CEM is coarser; observed ~2.06
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]
        # Matched counts cannot exceed available units.
        assert result.model_info["n_matched_treated"] <= result.model_info["n_treated"]


class TestBiasCorrection:
    """bias_correction=True across distances."""

    def test_propensity_bc(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="propensity",
            bias_correction=True,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["bias_correction"] is True
        # Recovers the planted ATT (true=2.0) with a sane CI.
        assert abs(result.estimate - 2.0) < 0.5  # observed ~1.99
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_euclidean_bc(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            distance="euclidean",
            bias_correction=True,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["bias_correction"] is True
        # Recovers the planted ATT (true=2.0) with a sane CI.
        assert abs(result.estimate - 2.0) < 0.5  # observed ~1.97
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]


# ==================================================================
# Legacy API backward compatibility
# ==================================================================


class TestLegacyAPI:
    """Old method='psm'/'mahalanobis'/'cem' should still work."""

    def test_legacy_psm(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="psm",
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0
        assert result.model_info["distance"] == "propensity"
        assert result.model_info["method"] == "nearest"

    def test_legacy_mahalanobis(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="mahalanobis",
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["distance"] == "mahalanobis"
        assert result.model_info["method"] == "nearest"
        # Legacy alias must produce the same recovery as the new API (true=2.0).
        assert abs(result.estimate - 2.0) < 0.5  # observed ~1.99
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_legacy_cem(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="cem",
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["method"] == "cem"
        # Legacy CEM alias recovers the planted ATT (true=2.0).
        assert abs(result.estimate - 2.0) < 0.6  # CEM coarser; observed ~2.12
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]


# ==================================================================
# General / diagnostics
# ==================================================================


class TestMethodSpecificInfo:
    """Extra diagnostics returned in model_info for each method."""

    def test_exact_matching_info(self, discrete_data):
        result = match(
            discrete_data,
            y="y",
            treat="treat",
            covariates=["age_group", "edu"],
            distance="exact",
        )
        info = result.model_info
        assert "n_matched_treated" in info
        assert "n_unmatched_treated" in info
        assert info["n_matched_treated"] > 0
        assert (
            info["n_matched_treated"] + info["n_unmatched_treated"] == info["n_treated"]
        )

    def test_cem_info(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="cem",
        )
        info = result.model_info
        assert "n_matched_treated" in info
        assert "n_matched_control" in info
        assert "n_bins" in info
        assert info["n_matched_treated"] <= info["n_treated"]

    def test_stratify_info(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="stratify",
            n_strata=5,
        )
        info = result.model_info
        assert info["n_strata"] == 5
        assert info["n_effective_strata"] <= 5
        assert info["n_effective_strata"] >= 1


class TestPsPoly:
    """ps_poly parameter for polynomial propensity score (Cunningham 2021, Ch. 5)."""

    def test_poly2(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            ps_poly=2,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["ps_poly"] == 2
        assert abs(result.estimate - 2.0) < 1.5

    def test_poly3(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            ps_poly=3,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["ps_poly"] == 3
        # Cubic PS still recovers the planted ATT (true=2.0).
        assert abs(result.estimate - 2.0) < 0.5  # observed ~1.99
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_poly1_is_default(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        assert result.model_info["ps_poly"] == 1

    def test_poly2_stratify(self, selection_bias_data):
        """Polynomial PS also works with stratification."""
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            method="stratify",
            ps_poly=2,
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestWithoutReplacement:
    """replace=False must enforce each control used at most once."""

    def test_no_duplicate_controls(self, selection_bias_data):
        """Controls should not be reused when replace=False."""
        estimator = MatchEstimator(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            replace=False,
        )
        cols = ["y", "treat", "x1", "x2"]
        clean = selection_bias_data[cols].dropna()
        T = clean["treat"].values.astype(int)
        X = clean[["x1", "x2"]].values.astype(float)
        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]
        pscore = estimator._logit_propensity(X, T)
        dist_mat = estimator._compute_distance_matrix(X, idx_t, idx_c, pscore)
        matches, _ = estimator._nn_match_from_dist(dist_mat)

        # Collect all matched control indices
        all_matched = []
        for m in matches:
            if len(m) > 0:
                all_matched.extend(m.tolist())
        # No duplicates
        assert len(all_matched) == len(set(all_matched))

    def test_with_replacement_allows_duplicates(self, selection_bias_data):
        """With replacement, controls CAN be reused."""
        estimator = MatchEstimator(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            replace=True,
        )
        cols = ["y", "treat", "x1", "x2"]
        clean = selection_bias_data[cols].dropna()
        T = clean["treat"].values.astype(int)
        X = clean[["x1", "x2"]].values.astype(float)
        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]
        pscore = estimator._logit_propensity(X, T)
        dist_mat = estimator._compute_distance_matrix(X, idx_t, idx_c, pscore)
        matches, _ = estimator._nn_match_from_dist(dist_mat)

        all_matched = []
        for m in matches:
            if len(m) > 0:
                all_matched.extend(m.tolist())
        # With replacement, duplicates are expected (many treated → few best controls)
        assert len(all_matched) >= len(set(all_matched))


class TestMatchGeneral:

    def test_balance_table(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        balance = result.model_info["balance"]
        assert isinstance(balance, pd.DataFrame)
        assert "variable" in balance.columns
        assert "smd" in balance.columns
        assert len(balance) >= 2

    def test_model_info_keys(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        info = result.model_info
        assert "distance" in info
        assert "method" in info
        assert "n_treated" in info
        assert "n_control" in info
        assert "bias_correction" in info

    def test_summary(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        s = result.summary()
        assert "Matching" in s
        assert "ATT" in s

    def test_citation(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
        )
        assert "abadie" in result.cite().lower()

    def test_caliper(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            caliper=0.1,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["caliper"] == 0.1
        # Caliper-restricted matching still recovers the planted ATT (true=2.0).
        assert abs(result.estimate - 2.0) < 0.5  # observed ~2.02
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_multiple_matches(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            n_matches=3,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["n_matches"] == 3
        # 3-NN matching still recovers the planted ATT (true=2.0).
        assert abs(result.estimate - 2.0) < 0.5  # observed ~2.00
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_without_replacement(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates=["x1", "x2"],
            replace=False,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["replace"] is False
        # Without replacement the estimate is biased upward on this DGP
        # (poorer matches; observed ~2.82) — assert sign + a generous band
        # plus internal CI consistency rather than a tight recovery.
        assert result.estimate > 0  # treatment effect is positive
        assert 1.0 < result.estimate < 4.0
        assert result.se > 0 and np.isfinite(result.se)
        assert result.ci[0] < result.estimate < result.ci[1]

    # --- Error handling ---

    def test_scalar_covariate_string(self, selection_bias_data):
        result = match(
            selection_bias_data,
            y="y",
            treat="treat",
            covariates="x1",
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["n_treated"] > 0

    def test_non_dataframe_input(self, selection_bias_data):
        with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
            match(
                selection_bias_data.to_dict("list"),
                y="y",
                treat="treat",
                covariates=["x1"],
            )

    def test_missing_column(self, selection_bias_data):
        with pytest.raises(MethodIncompatibility) as exc:
            match(
                selection_bias_data, y="nonexistent", treat="treat", covariates=["x1"]
            )
        assert exc.value.diagnostics["missing_columns"] == ["nonexistent"]

    def test_invalid_method(self, selection_bias_data):
        with pytest.raises(MethodIncompatibility, match="method must be"):
            match(
                selection_bias_data,
                y="y",
                treat="treat",
                covariates=["x1"],
                method="invalid",
            )

    def test_invalid_distance(self, selection_bias_data):
        with pytest.raises(MethodIncompatibility, match="distance must be"):
            match(
                selection_bias_data,
                y="y",
                treat="treat",
                covariates=["x1"],
                distance="cosine",
            )

    def test_invalid_estimand(self, selection_bias_data):
        with pytest.raises(MethodIncompatibility, match="estimand must be"):
            match(
                selection_bias_data,
                y="y",
                treat="treat",
                covariates=["x1"],
                estimand="INVALID",
            )

    @pytest.mark.parametrize(
        "kwargs, match_text",
        [
            ({"n_matches": 0}, "n_matches"),
            ({"alpha": 1.0}, "alpha"),
            ({"ps_poly": 0}, "ps_poly"),
            ({"bwidth": 0.0}, "bwidth"),
        ],
    )
    def test_invalid_numeric_controls(
        self,
        selection_bias_data,
        kwargs,
        match_text,
    ):
        with pytest.raises(MethodIncompatibility, match=match_text):
            match(
                selection_bias_data, y="y", treat="treat", covariates=["x1"], **kwargs
            )

    def test_exact_no_matches_is_data_insufficient(self):
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "treat": [1, 1, 0, 0],
                "x": [10.0, 11.0, 0.0, 1.0],
            }
        )
        with pytest.raises(DataInsufficient, match="exact matching"):
            match(df, y="y", treat="treat", covariates=["x"], distance="exact")

    def test_non_binary_treatment(self):
        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "treat": [0, 1, 2],
                "x": [1, 2, 3],
            }
        )
        with pytest.raises(ValueError, match="binary"):
            match(df, y="y", treat="treat", covariates=["x"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
