"""Columbus cross-validation against PySAL's ``spreg`` (Anselin's benchmark dataset).

Reference numbers were generated once with libpysal 4.14.1 + spreg 1.9.0 and
checked into ``fixtures/columbus_reference.json`` so the test does not require
either package at runtime. The acceptance target for SP-01 S1.1 is
``rtol < 1e-4`` on all parameter estimates for both SAR and SEM.

This is **the acceptance gate** for the spatial core sprint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.spatial import sar, sem
from statspai.spatial.weights.core import W


FIXTURE = Path(__file__).parent / "fixtures" / "columbus_reference.json"


@pytest.fixture(scope="module")
def reference():
    with FIXTURE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def columbus(reference):
    neighbors = {int(k): v for k, v in reference["neighbors"].items()}
    w = W(neighbors); w.transform = "R"
    df = pd.DataFrame({
        "CRIME": reference["y"],
        "INC":   reference["INC"],
        "HOVAL": reference["HOVAL"],
    })
    return w, df


# spreg ML_Lag betas are ordered [Intercept, INC, HOVAL, rho].
# Our sar returns params indexed  [const,     INC, HOVAL, rho].
# Same numerical positions.

def test_sar_matches_spreg(columbus, reference):
    w, df = columbus
    res = sar(w, df, "CRIME ~ INC + HOVAL")
    expected = reference["ml_lag"]["betas_with_rho"]   # [b0, b_INC, b_HOVAL, rho]
    got = res.params.values
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    # tighten the spatial coefficient specifically
    np.testing.assert_allclose(
        res.model_info["spatial_param_value"],
        reference["ml_lag"]["rho"],
        rtol=1e-5,
    )


def test_sem_matches_spreg(columbus, reference):
    w, df = columbus
    res = sem(w, df, "CRIME ~ INC + HOVAL")
    expected = reference["ml_err"]["betas"]            # [b0, b_INC, b_HOVAL, lam]
    got = res.params.values
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    np.testing.assert_allclose(
        res.model_info["spatial_param_value"],
        reference["ml_err"]["lam"],
        rtol=1e-5,
    )


def test_sar_sigma2_matches_spreg(columbus, reference):
    w, df = columbus
    res = sar(w, df, "CRIME ~ INC + HOVAL")
    np.testing.assert_allclose(
        res.diagnostics["sigma2"],
        reference["ml_lag"]["sigma2"],
        rtol=1e-4,
    )


def test_sem_sigma2_matches_spreg(columbus, reference):
    w, df = columbus
    res = sem(w, df, "CRIME ~ INC + HOVAL")
    np.testing.assert_allclose(
        res.diagnostics["sigma2"],
        reference["ml_err"]["sigma2"],
        rtol=1e-4,
    )
