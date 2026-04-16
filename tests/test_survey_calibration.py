"""Survey calibration (raking / linear) tests."""
import numpy as np, pandas as pd, pytest
from statspai.survey.calibration import rake, linear_calibration


@pytest.fixture(scope="module")
def survey_data():
    rng = np.random.default_rng(0)
    n = 500
    sex = rng.choice(["M", "F"], n, p=[0.6, 0.4])  # biased sample
    age = rng.choice(["18-34", "35-64", "65+"], n, p=[0.5, 0.3, 0.2])
    income = rng.lognormal(10, 1, n)
    return pd.DataFrame({"sex": sex, "age": age, "income": income})


def test_rake_hits_margins(survey_data):
    margins = {
        "sex": {"M": 0.49, "F": 0.51},
        "age": {"18-34": 0.30, "35-64": 0.45, "65+": 0.25},
    }
    res = rake(survey_data, margins)
    w = res.calibrated_weights
    # Weighted sex proportions should match targets
    for cat, target in margins["sex"].items():
        actual = w[survey_data["sex"] == cat].sum()
        assert abs(actual - target) < 0.01


def test_rake_converges(survey_data):
    margins = {"sex": {"M": 0.49, "F": 0.51}}
    res = rake(survey_data, margins)
    assert res.converged


def test_linear_calibration_hits_total(survey_data):
    target_income_total = 500 * 25000  # target population total
    res = linear_calibration(survey_data, totals={"income": target_income_total})
    actual = (res.calibrated_weights * survey_data["income"]).sum()
    np.testing.assert_allclose(actual / 500, target_income_total / 500, rtol=0.01)


def test_exported():
    import statspai as sp
    assert callable(sp.rake)
    assert callable(sp.linear_calibration)
