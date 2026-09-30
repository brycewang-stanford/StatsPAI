"""Ordered probit / logit with many dummies: speed and Stata parity.

Busting the Princelings (QJE 2019) fits ``oprobit`` with ~300 prefecture
dummies. Through 1.32 ``sp.oprobit`` optimised with BFGS on finite-difference
gradients and built the Hessian by complex-step differentiation -- 260 s at
100 dummies, stopping early on an unscaled regressor (0.7397 vs Stata's
0.742). The analytic Newton engine (``regression/_ml_newton.py``) fits
the fixture below (119 dummies, ``x ~ 1000 +- 50``) in well under a second.

Reference: Stata 18 ``oprobit`` / ``ologit`` on
``_fixtures/oprobit_dummies.csv`` (``_generate_oprobit_dummies_Stata.do``).
Coefficients, cutpoints and their SEs -- including the robust and clustered
cutpoint SEs, which through 1.32 were model-based whatever ``vce`` said --
agree to 1e-9 relative (measured: coefficients 7e-12, SEs 3e-10 with
clustering). Both sides are driven to the optimum: Stata with
``nrtolerance(1e-13)``, ``sp.oprobit`` with Newton polishing steps after its
``tol`` is met (stopping at ``g'(-H)^{-1}g < 1e-8`` alone left the estimates
~3e-5 relative from the optimum).
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "oprobit_dummies_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def design():
    df = pd.read_csv(_FIX / "oprobit_dummies.csv")
    dummies = pd.get_dummies(df.g, prefix="g", drop_first=True, dtype=float)
    return pd.concat([df, dummies], axis=1), ["x", "z"] + list(dummies.columns)


def _stata_order(ref, n_groups=120):
    """Drop Stata's base level (0b.g) so the vector lines up with ours."""
    b = np.asarray(ref["b"])
    se = np.asarray(ref["se"])
    keep = np.r_[0, 1, np.arange(3, 2 + n_groups), np.arange(2 + n_groups, len(b))]
    return b[keep], se[keep]


CASES = [
    ("oprobit", sp.oprobit, {}),
    ("oprobit_robust", sp.oprobit, {"robust": "robust"}),
    ("oprobit_cluster", sp.oprobit, {"cluster": "cl"}),
    ("ologit_cluster", sp.ologit, {"cluster": "cl"}),
]


@pytest.mark.parametrize("key, fn, kw", CASES, ids=[c[0] for c in CASES])
def test_matches_stata(design, key, fn, kw):
    df, xs = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        r = fn(data=df, y="y", x=xs, **kw)
        elapsed = time.perf_counter() - t0
    b_ref, se_ref = _stata_order(STATA[key])
    assert r.model_info["converged"]
    assert r.model_info["log_likelihood"] == pytest.approx(STATA[key]["ll"], rel=1e-12)
    np.testing.assert_allclose(r.params.to_numpy(), b_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(r.std_errors.to_numpy(), se_ref, rtol=1e-9)
    # 122 parameters: minutes before the analytic engine.
    assert elapsed < 20


def test_unscaled_regressor_converges_to_the_optimum(design):
    """The early-stopping bug: a coefficient on x ~ 1000 is still exact."""
    df, xs = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.oprobit(data=df, y="y", x=xs)
    b_ref, _ = _stata_order(STATA["oprobit"])
    assert r.params["x"] == pytest.approx(b_ref[0], rel=1e-8)
    assert r.model_info["iterations"] <= 20
