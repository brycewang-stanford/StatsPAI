"""sp.residual_balance against balanceHD, the method authors' R package.

Fixture: ``_fixtures/residual_balance_balancehd.json``, produced by
``_fixtures/_generate_residual_balance_R.R`` from the committed CSV
(balanceHD 1.0 with ``optimizer="quadprog"``, used as a black box).

What is compared
----------------
* The balancing weights themselves, for three values of ``zeta``, for
  the ATT target and without the non-negativity constraint.
* The pure weighting estimate (``fit.method="none"``), which is a
  deterministic function of the weights, for ATE / ATT / ATC and with
  and without covariate scaling.

The outcome-model step is a cross-validated elastic net on random
folds in both packages, so the augmented estimate is not comparable
number for number and is not part of this file.

Tolerance
---------
Both sides solve the same strictly convex quadratic programme, so the
solution is unique. balanceHD's quadprog path stops short of the
optimum: at every configuration here its objective value is above ours
by 1e-10 to 1e-8 (checked below, and against scipy's trust-constr in
development). The weights therefore agree to about 1e-6 rather than
machine precision, and the tolerance of 2e-5 on weights and estimates
is set by the reference's solver, not by a difference in method.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching.residual_balance import approx_balance_weights

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "residual_balance_balancehd.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FIX / "residual_balance_data.csv")
COVS = [c for c in DATA.columns if c.startswith("x")]
X = DATA[COVS].to_numpy()
W = DATA["W"].to_numpy()
WEIGHT_TOL = 2e-5
EST_TOL = 2e-5


def _objective(M, target, g, zeta):
    return (1 - zeta) * g @ g + zeta * np.max(np.abs(M.T @ g - target)) ** 2


@pytest.mark.parametrize(
    "key, arm, target, zeta, negative",
    [
        ("gamma_treated_zeta05", 1, "all", 0.5, False),
        ("gamma_treated_zeta01", 1, "all", 0.1, False),
        ("gamma_treated_zeta09", 1, "all", 0.9, False),
        ("gamma_control_att", 0, "treated", 0.5, False),
        ("gamma_treated_negative", 1, "all", 0.5, True),
    ],
)
def test_weights_match_balancehd(key, arm, target, zeta, negative):
    M = X[W == arm]
    tgt = X.mean(axis=0) if target == "all" else X[W == 1].mean(axis=0)
    g, info = approx_balance_weights(M, tgt, zeta=zeta, allow_negative_weights=negative)
    ref = np.asarray(REF[key])
    assert info["converged"]
    assert abs(g.sum() - 1) < 1e-12
    np.testing.assert_allclose(g, ref, atol=WEIGHT_TOL, rtol=0)
    # The programme is strictly convex: the better objective is the more
    # accurate solution, and ours must not be the worse one.
    assert _objective(M, tgt, g, zeta) <= _objective(M, tgt, ref, zeta) + 1e-13
    if not negative:
        assert g.min() >= 0


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("ate_none", {}),
        ("ate_none_unscaled", {"standardize": False}),
        ("att_none", {"estimand": "ATT"}),
        ("atc_none", {"estimand": "ATC"}),
        ("ate_none_zeta02", {"zeta": 0.2}),
    ],
)
def test_weighting_estimate_matches_balancehd(key, kwargs):
    r = sp.residual_balance(DATA, "Y", "W", COVS, outcome_model="none", **kwargs)
    assert r.estimate == pytest.approx(REF[key], abs=EST_TOL)
    # No outcome model, so no residual-based standard error.
    assert np.isnan(r.se)
