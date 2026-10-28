"""``sp.positive_correlation_test(method='residuals')`` with ordered outcomes
vs Stata 18 ``oprobit`` / ``probit``.

Fixture: ``_fixtures/_generate_positive_correlation_ordered_data.py`` writes
800 rows (floats as %.17g);
``_fixtures/_generate_positive_correlation_ordered_Stata.do`` fits
``oprobit`` (three coverage tiers, claim counts top-coded at 2) and
``probit``, builds the generalized residuals from ``e(b)`` and stores the
score ``sum e1 e2``, its outer-product variance (plain and summed by the
cluster ``g``), ``W`` and the residual correlation. Stata runs at tight
convergence tolerances (at its defaults the probit stops 1e-7 short of
the maximum).

Observed: the coverage ``oprobit`` agrees to 2e-16 and the ordered-coverage
/ binary-claim statistics to 1.3e-15; with the claim count, the statistics
agree to 4e-10, the stopping point of ``sp.oprobit``'s default tolerance on
that fit. Tolerances: 1e-12 for the coverage fit, 1e-9 for the statistics.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_HERE = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads(
    (_HERE / "positive_correlation_ordered_Stata.json").read_text(encoding="utf-8")
)
DATA = pd.read_csv(
    _HERE / "positive_correlation_ordered_data.csv", float_precision="round_trip"
)

CASES = {
    "ord_bin": {"claim": "claim"},
    "ord_ord": {"claim": "nclaim"},
    "ord_ord_cl": {"claim": "nclaim", "cluster": "g"},
}


def test_oprobit_step_matches_stata():
    fit = sp.oprobit(data=DATA, y="cover", x=["x1", "x2"])
    ref = REF["oprobit_cover"]
    assert fit.model_info["log_likelihood"] == pytest.approx(ref["ll"], rel=1e-12)
    assert fit.params["x1"] == pytest.approx(ref["b_x1"], rel=1e-12)
    assert fit.params["/cut2"] == pytest.approx(ref["cut2"], rel=1e-12)


@pytest.mark.parametrize("key", list(CASES))
def test_ordered_residual_test_matches_stata(key):
    ref = REF[key]
    kw = dict(CASES[key])
    claim = kw.pop("claim")
    res = sp.positive_correlation_test(
        DATA,
        "cover",
        claim,
        ["x1", "x2"],
        method="residuals",
        claim_covariates=["x1", "x3"],
        **kw,
    )
    assert res.n_obs == ref["N"]
    assert res.detail["score"] == pytest.approx(ref["score"], rel=1e-9)
    assert res.detail["score_variance"] == pytest.approx(
        ref["score_variance"], rel=1e-9
    )
    assert res.statistic == pytest.approx(ref["W"], rel=1e-9)
    assert res.estimate == pytest.approx(ref["corr"], rel=1e-9)
    assert set(res.detail["ordered"]) == (
        {"cover"} if claim == "claim" else {"cover", "nclaim"}
    )
