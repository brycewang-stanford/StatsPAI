"""sp.ridge against R MASS::lm.ridge outputs (black box, MASS 7.3-65).

Fixture: ``_fixtures/_generate_ridge_blackbox_R.R``. Covers longley (with
intercept) and a synthetic design with and without an intercept; the
clean-room specification is ``docs/dev/cleanroom/ridge.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "ridge_blackbox_R.json").read_text(encoding="utf-8"))
# Closed-form linear algebra on the same doubles; R prints 15+ digits.
RTOL = 1e-10


def _fit(case: str):
    ref = REF[case]
    if case == "longley":
        data = pd.read_csv(FIX / "ridge_blackbox_longley.csv")
        x = [c for c in data.columns if c != "y"]
        fit = sp.ridge(y="y", x=x, data=data, lambda_=ref["lambda"])
    else:
        data = pd.read_csv(FIX / "ridge_blackbox_data.csv")
        rhs = "x1 + x2 + x3 + x4"
        formula = f"y ~ {rhs}" if case == "syn" else f"y ~ 0 + {rhs}"
        fit = sp.ridge(formula, data, lambda_=ref["lambda"])
    return fit, ref


@pytest.mark.parametrize("case", ["longley", "syn", "syn_noint"])
def test_path_gcv_and_plugins_match_lm_ridge(case):
    fit, ref = _fit(case)
    coef_cols = [c for c in fit.path.columns if c not in ("lambda", "gcv", "df")]
    np.testing.assert_allclose(
        fit.path[coef_cols].to_numpy(), np.asarray(ref["coef"]), rtol=RTOL
    )
    np.testing.assert_allclose(fit.path["gcv"], ref["gcv"], rtol=RTOL)
    np.testing.assert_allclose(fit.lambda_hkb, ref["kHKB"], rtol=RTOL)
    np.testing.assert_allclose(fit.lambda_lw, ref["kLW"], rtol=RTOL)
    assert fit.lambda_ == ref["lambda"][int(np.argmin(ref["gcv"]))]


def test_reported_df_is_the_hat_trace_with_the_intercept():
    # Identity: at lambda = 0 the hat matrix projects onto the column space
    # of [1, X], so its trace is p + 1 with an intercept and p without.
    with_int, _ = _fit("syn")
    without, _ = _fit("syn_noint")
    assert with_int.path["df"].iloc[0] == pytest.approx(5.0, rel=1e-12)
    assert without.path["df"].iloc[0] == pytest.approx(4.0, rel=1e-12)
