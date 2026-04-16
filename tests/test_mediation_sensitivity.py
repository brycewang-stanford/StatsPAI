"""Mediation sensitivity analysis tests."""
import numpy as np, pandas as pd, pytest
from statspai.mediation.sensitivity import mediate_sensitivity


@pytest.fixture(scope="module")
def med_dgp():
    rng = np.random.default_rng(0)
    n = 500
    T = rng.binomial(1, 0.5, n)
    X = rng.standard_normal(n)
    M = 0.5 * T + 0.3 * X + rng.standard_normal(n)
    Y = 1.0 * T + 0.8 * M + 0.5 * X + rng.standard_normal(n)
    return pd.DataFrame({"Y": Y, "T": T, "M": M, "X": X})


def test_baseline_acme_near_truth(med_dgp):
    r = mediate_sensitivity(med_dgp, "Y", "T", "M", covariates=["X"])
    assert abs(r.acme_at_zero - 0.40) < 0.1


def test_rho_at_zero_positive(med_dgp):
    r = mediate_sensitivity(med_dgp, "Y", "T", "M", covariates=["X"])
    assert r.rho_at_zero is not None
    assert r.rho_at_zero > 0.5


def test_acme_decreases_with_positive_rho(med_dgp):
    r = mediate_sensitivity(med_dgp, "Y", "T", "M", covariates=["X"])
    assert r.acme_at_rho[0] > r.acme_at_rho[-1]


def test_summary_prints(med_dgp):
    r = mediate_sensitivity(med_dgp, "Y", "T", "M", covariates=["X"])
    s = r.summary()
    assert "Sensitivity" in s
    assert "ρ" in s


def test_exported():
    import statspai as sp
    assert callable(sp.mediate_sensitivity)
