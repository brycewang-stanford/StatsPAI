"""``sp.positive_correlation_test(method='biprobit')`` vs Stata 18 ``biprobit``.

Fixture: ``_fixtures/_generate_positive_correlation_test_data.py`` writes
600 rows (floats as %.17g); ``_fixtures/_generate_positive_correlation_test_Stata.do``
runs ``biprobit`` three ways and stores rho, athrho and its standard
error, the chi-squared test of rho = 0 (likelihood ratio without
clusters, Wald with ``vce(cluster g)``) and the log likelihood.

Tolerances: the log likelihood and the LR statistic to 1e-12 relative
(the maximised value is flat in the parameters, so both optimisers agree
to rounding). rho and athrho to 1e-8 relative: they are argmax values and
the two optimisers stop at different points of a flat optimum (observed
1.6e-9). The athrho standard errors and the Wald statistic to 1e-8
(observed 2e-10).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_HERE = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads(
    (_HERE / "positive_correlation_test_Stata.json").read_text(encoding="utf-8")
)
DATA = pd.read_csv(
    _HERE / "positive_correlation_test_data.csv", float_precision="round_trip"
)

CASES = {
    "same": {"covariates": ["x1", "x2"]},
    "seemingly": {"covariates": ["x1", "x2"], "claim_covariates": ["x1", "x3"]},
    "cluster": {"covariates": ["x1", "x2"], "cluster": "g"},
}


@pytest.mark.parametrize("key", list(CASES))
def test_biprobit_matches_stata(key):
    ref = REF[key]
    res = sp.positive_correlation_test(DATA, "cover", "claim", **CASES[key])
    assert res.n_obs == ref["N"]
    assert res.detail["ll_biprobit"] == pytest.approx(ref["ll"], rel=1e-12)
    assert res.estimate == pytest.approx(ref["rho"], rel=1e-8)
    assert res.detail["athrho"] == pytest.approx(ref["athrho"], rel=1e-8)
    assert res.detail["athrho_se"] == pytest.approx(ref["athrho_se"], rel=1e-8)
    tol = 1e-12 if key != "cluster" else 1e-8
    assert res.statistic == pytest.approx(ref["chi2_c"], rel=tol)
