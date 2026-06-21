"""Tests for the survey design module."""

import numpy as np
import pandas as pd
import pytest


def _make_survey_data(n=500, seed=42):
    """Generate synthetic survey data with strata and clusters."""
    rng = np.random.RandomState(seed)
    n_strata = 5
    n_psu_per_stratum = 4

    strata = np.repeat(np.arange(n_strata), n // n_strata)
    psu = np.tile(
        np.repeat(np.arange(n_psu_per_stratum), n // (n_strata * n_psu_per_stratum)),
        n_strata,
    )
    # Make PSU ids unique across strata
    psu_unique = strata * 100 + psu

    weights = rng.uniform(0.5, 3.0, size=n)
    income = 30000 + 5000 * strata + rng.randn(n) * 10000
    education = 12 + rng.randint(0, 9, size=n)
    age = 25 + rng.randint(0, 40, size=n)

    df = pd.DataFrame(
        {
            "stratum": strata,
            "psu": psu_unique,
            "weight": weights,
            "income": income,
            "education": education,
            "age": age,
        }
    )
    return df


class TestSurveyDesign:
    def test_create_design(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        assert design.n == 500
        assert "SurveyDesign" in repr(design)

    def test_svydesign_weighted_mean_matches_manual_formula(self):
        from statspai.survey import svydesign

        df = pd.DataFrame(
            {
                "weight": [1.0, 2.0, 3.0, 4.0],
                "stratum": [0, 0, 1, 1],
                "psu": [10, 11, 20, 21],
                "income": [10.0, 20.0, 40.0, 80.0],
            }
        )

        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.mean("income")

        expected = np.average(df["income"], weights=df["weight"])
        np.testing.assert_allclose(result.estimate["income"], expected)

    def test_design_no_strata(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", cluster="psu")
        assert design.n == 500

    def test_design_no_cluster(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum")
        assert design.n == 500

    def test_bad_weights_raises(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        df.loc[0, "weight"] = -1.0
        with pytest.raises(ValueError, match="positive"):
            svydesign(df, weights="weight")

    def test_nonfinite_weights_raise(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        df.loc[0, "weight"] = np.nan
        with pytest.raises(ValueError, match="finite"):
            svydesign(df, weights="weight")

    def test_array_weights_length_mismatch_raises(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        with pytest.raises(ValueError, match="weights length"):
            svydesign(df, weights=np.ones(len(df) - 1))


class TestSvyMean:
    def test_mean_single_var(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.mean("income")
        assert len(result.estimate) == 1
        assert result.std_error["income"] > 0
        assert result.ci_lower["income"] < result.estimate["income"]
        assert result.ci_upper["income"] > result.estimate["income"]

    def test_mean_multiple_vars(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.mean(["income", "education"])
        assert len(result.estimate) == 2
        assert "income" in result.estimate.index
        assert "education" in result.estimate.index

    def test_mean_summary(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.mean("income")
        tbl = result.summary()
        assert "Estimate" in tbl.columns
        assert "Std.Err" in tbl.columns


class TestSvyTotal:
    def test_total_basic(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.total("income")
        assert result.estimate["income"] > 0
        assert result.std_error["income"] > 0


class TestSvyGLM:
    def test_gaussian_glm(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.glm("income ~ education + age")
        assert len(result.estimate) == 3  # Intercept + education + age
        assert np.all(result.std_error > 0)

    def test_glm_summary(self):
        from statspai.survey import svydesign

        df = _make_survey_data()
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.glm("income ~ education + age")
        tbl = result.summary()
        assert len(tbl) == 3

    def test_binomial_glm(self):
        from statspai.survey import svydesign

        rng = np.random.RandomState(123)
        n = 400
        df = pd.DataFrame(
            {
                "weight": rng.uniform(0.5, 3.0, n),
                "stratum": np.repeat([0, 1], n // 2),
                "psu": np.tile(np.repeat(np.arange(4), n // 8), 2),
                "x": rng.randn(n),
            }
        )
        prob = 1 / (1 + np.exp(-(0.5 + 0.8 * df["x"])))
        df["y"] = rng.binomial(1, prob)
        design = svydesign(df, weights="weight", strata="stratum", cluster="psu")
        result = design.glm("y ~ x", family="binomial")
        assert len(result.estimate) == 2  # Intercept + x
