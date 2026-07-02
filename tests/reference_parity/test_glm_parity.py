"""Frozen-R parity: sp.glm vs base R stats::glm (binomial logit + Poisson log).

IRLS converges to the identical unpenalized MLE, so coefficients, the maximized
log-likelihood, and AIC match base R to machine precision (observed <= 5e-13).
Model-based standard errors match to a looser tolerance (~1e-4 relative), a
convergence-precision artifact; the coefficients are the bit-exact headline.

Regenerate the reference with:
    Rscript tests/reference_parity/_generate_glm_R.R
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
_MAP = {"(Intercept)": "Intercept", "x1": "x1", "x2": "x2"}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "glm_data.csv")


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "glm_R.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "family,yvar,key",
    [("binomial", "yb", "binomial"), ("poisson", "yc", "poisson")],
)
def test_coefficients_bit_exact(data, ref, family, yvar, key):
    r = sp.glm(f"{yvar} ~ x1 + x2", data=data, family=family)
    rc = ref[key]["coef"]
    for rname, pyname in _MAP.items():
        assert float(r.params[pyname]) == pytest.approx(rc[rname], abs=1e-9)


@pytest.mark.parametrize(
    "family,yvar,key",
    [("binomial", "yb", "binomial"), ("poisson", "yc", "poisson")],
)
def test_loglik_and_aic_bit_exact(data, ref, family, yvar, key):
    r = sp.glm(f"{yvar} ~ x1 + x2", data=data, family=family)
    g = r.glance()

    def _scalar(v):
        return float(v.iloc[0] if hasattr(v, "iloc") else v)

    assert _scalar(g["log_likelihood"]) == pytest.approx(ref[key]["loglik"], abs=1e-8)
    assert _scalar(g["aic"]) == pytest.approx(ref[key]["aic"], abs=1e-8)


@pytest.mark.parametrize(
    "family,yvar,key",
    [("binomial", "yb", "binomial"), ("poisson", "yc", "poisson")],
)
def test_standard_errors_aligned(data, ref, family, yvar, key):
    r = sp.glm(f"{yvar} ~ x1 + x2", data=data, family=family)
    rse = ref[key]["se"]
    for rname, pyname in _MAP.items():
        assert float(r.std_errors[pyname]) == pytest.approx(rse[rname], rel=1e-3)
