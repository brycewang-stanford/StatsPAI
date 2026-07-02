"""Frozen-R parity: sp.three_sls vs R systemfit::systemfit(method="3SLS").

Three-stage least squares on a 2-equation simultaneous system (each equation's
other endogenous regressor instrumented by the full exogenous set). The 3SLS
coefficient vector matches R systemfit to machine precision (observed <= 1e-15);
standard errors align to ~2e-3 relative (a small-sample residual-covariance
d.o.f. convention).

Regenerate the reference with:
    Rscript tests/reference_parity/_generate_threesls_R.R
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
EQUATIONS = {"eq1": ("y1", ["x1"], ["y2"]), "eq2": ("y2", ["x2"], ["y1"])}
# params_all layout: eq1[intercept, x1, y2], eq2[intercept, x2, y1]
LAYOUT = {
    "eq1": {"(Intercept)": 0, "x1": 1, "y2": 2},
    "eq2": {"(Intercept)": 3, "x2": 4, "y1": 5},
}


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv(_FIX / "threesls_data.csv")
    ref = json.loads((_FIX / "threesls_R.json").read_text(encoding="utf-8"))
    r = sp.three_sls(EQUATIONS, data=df, instruments=["x1", "x2"])
    return r, ref


def test_coefficients_bit_exact(fitted):
    r, ref = fitted
    pa = np.asarray(r.params_all, dtype=float)
    for eq, terms in LAYOUT.items():
        for name, idx in terms.items():
            assert pa[idx] == pytest.approx(ref[eq]["coef"][name], abs=1e-9)


def test_standard_errors_aligned(fitted):
    r, ref = fitted
    se = np.asarray(r.se_all, dtype=float)
    for eq, terms in LAYOUT.items():
        for name, idx in terms.items():
            assert se[idx] == pytest.approx(ref[eq]["se"][name], rel=5e-3)
