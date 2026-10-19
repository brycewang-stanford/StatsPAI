"""``sp.finegray(weights=)`` vs Stata 18 ``stcrreg`` after ``stset [pw=]``.

Reference: ``_fixtures/finegray_weighted_stata.json`` written by
``_fixtures/_generate_finegray_weighted_stata.do`` on
``survival_epi_cr.csv`` with two deterministic weight vectors and both
causes. Stata is used as a black box (outputs only).

Stata's pweighted Fine-Gray weights every sum: the censoring Kaplan-Meier,
the risk sets and the event counts, and the robust variance uses the
weighted score residuals including the censoring term, with ``n/(n-1)``
(``small_sample=True`` here). Its log pseudo-likelihood is that of the
weights rescaled to mean one. Stata stops Newton-Raphson after three
iterations (``ll`` change < 1e-7), so its coefficients are converged to
~1e-8; tolerance 1e-6 relative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "finegray_weighted_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    d = pd.read_csv(_FIX / "survival_epi_cr.csv")
    d["w"] = 1 + 0.25 * (np.arange(len(d)) % 7)
    d["w2"] = np.exp(0.3 * d["x1"])
    return d


@pytest.mark.parametrize("cause", [1, 2])
@pytest.mark.parametrize("wcol", ["w", "w2"])
def test_weighted_finegray_matches_stcrreg(data, wcol, cause):
    ref = S[f"{wcol}_cause{cause}"]
    r = sp.finegray(
        data,
        "time",
        "status",
        ["x1", "x2"],
        cause=cause,
        weights=wcol,
        small_sample=True,
        tol=1e-12,
    )
    np.testing.assert_allclose(r.params, ref["b"], rtol=1e-6)
    np.testing.assert_allclose(r.vcov.ravel(), ref["V"], rtol=1e-6)
    assert r.loglik == pytest.approx(ref["ll"], rel=1e-9)


def test_integer_weights_equal_replicated_rows(data):
    """Frequency-weight identity: weights 2 == every row twice (point estimate)."""
    d = data.assign(two=2.0)
    a = sp.finegray(d, "time", "status", ["x1", "x2"], weights="two", tol=1e-12)
    b = sp.finegray(
        pd.concat([data, data], ignore_index=True),
        "time",
        "status",
        ["x1", "x2"],
        tol=1e-12,
    )
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10)


def test_weights_refuse_model_variance(data):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        sp.finegray(data, "time", "status", ["x1"], weights="w", vce="model")
