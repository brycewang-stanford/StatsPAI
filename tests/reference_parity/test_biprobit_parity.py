"""Frozen-R parity: sp.biprobit vs R VGAM::vglm(binom2.rho) bivariate probit.

Both maximize the same joint bivariate-normal likelihood, so the two equations'
slope/intercept coefficients and the error correlation rho match VGAM to the
shared optimizer tolerance (observed <= 2e-7). VGAM encodes rho through the
rhobit link; the reference stores rho = tanh(rhobit / 2).

Regenerate the reference with:
    Rscript tests/reference_parity/_generate_biprobit_R.R
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv(_FIX / "biprobit_data.csv")
    ref = json.loads((_FIX / "biprobit_R.json").read_text(encoding="utf-8"))
    r = sp.biprobit(df, y1="y1", y2="y2", x1=["x"])
    return dict(r.params), r.diagnostics, ref


def test_coefficients_match_vgam(fitted):
    p, _, ref = fitted
    pairs = [
        ("eq1._cons", "eq1_intercept"),
        ("eq1.x", "eq1_x"),
        ("eq2._cons", "eq2_intercept"),
        ("eq2.x", "eq2_x"),
    ]
    for pk, rk in pairs:
        assert float(p[pk]) == pytest.approx(ref[rk], abs=1e-6)


def test_rho_and_loglik_match_vgam(fitted):
    p, diag, ref = fitted
    assert float(p["rho"]) == pytest.approx(ref["rho"], abs=1e-6)
    assert float(diag["log_likelihood"]) == pytest.approx(ref["loglik"], rel=1e-6)
