"""``sp.aipw(outcome_model=)`` against Stata 18 ``teffects aipw``.

Reference: ``_fixtures/aipw_outcome_models_stata.txt`` written by
``_fixtures/_generate_aipw_outcome_models_Stata.do`` on
``_fixtures/aipw_outcome_models_data.csv`` (the TMLE design data plus a
count outcome). ``teffects aipw (y x, logit | probit | poisson) (d x)``: logit
treatment model, per-arm outcome model by maximum likelihood, robust
standard errors from the stacked estimating equations.

StatsPAI side: ``cross_fit=False, se_method='sandwich'``, which is the
same estimator. Three designs: plain, clustered, and sampling weights.
``teffects aipw`` refuses ``[pw=]``; ``[iw=]`` with one cluster per row
gives the sampling-weight sandwich (the convention pinned in
``test_teffects_design_stata_parity.py``).

Tolerance 1e-6 (CLAUDE.md 5.1 default); observed 4e-9 at worst, which is
the convergence tolerance of the two maximum-likelihood fits.
"""

from __future__ import annotations

import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6
COV = ["x1", "x2", "x3"]
OUTCOME = {"logit": "yb", "probit": "yb", "poisson": "yc"}
CASES: Dict[str, Dict[str, Any]] = {
    "plain": {},
    "weights": {"weights": "w"},
    "cluster": {"cluster": "g"},
}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, float]:
    path = _FIX / "aipw_outcome_models_stata.txt"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_aipw_outcome_models_Stata.do first")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        out[key] = float(value)
    return out


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "aipw_outcome_models_data.csv")


def _fit(model: str, **extra: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.aipw(
            _data(),
            y=OUTCOME[model],
            treat="d",
            covariates=COV,
            cross_fit=False,
            se_method="sandwich",
            outcome_model=model,
            **extra,
        )


@pytest.mark.parametrize("model", list(OUTCOME))
@pytest.mark.parametrize("case", list(CASES))
def test_matches_teffects_aipw(model: str, case: str) -> None:
    ref = _ref()
    res = _fit(model, **CASES[case])
    key = f"{model}_{case}_"
    np.testing.assert_allclose(res.estimate, ref[key + "ate"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref[key + "ate_se"], rtol=RTOL)
    means = res.model_info["potential_outcome_means"]
    ses = res.model_info["potential_outcome_means_se"]
    for arm in (0, 1):
        np.testing.assert_allclose(means[arm], ref[f"{key}po{arm}"], rtol=RTOL)
        np.testing.assert_allclose(ses[arm], ref[f"{key}po{arm}_se"], rtol=RTOL)


def test_logit_predictions_stay_in_range_where_linear_ones_do_not() -> None:
    # Rare outcome with a strong covariate: a per-arm linear probability
    # model predicts negative risks; the logit does not.
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 0.5 * d + 1.5 * x)))).astype(float)
    df = pd.DataFrame({"x": x, "d": d, "y": y})
    kw = dict(y="y", treat="d", covariates=["x"], cross_fit=False)
    lin = sp.aipw(df, **kw)
    logit = sp.aipw(df, outcome_model="logit", **kw)
    assert logit.model_info["outcome_model"] == "logit"
    assert abs(lin.estimate - logit.estimate) < 3 * lin.se
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X[d == 0], y[d == 0], rcond=None)[0]
    assert (X @ beta).min() < 0


def test_outcome_must_suit_the_model() -> None:
    d = _data()
    kw = dict(treat="d", covariates=COV)
    with pytest.raises(MethodIncompatibility, match=r"\[0, 1\]"):
        sp.aipw(d, y="yc", outcome_model="logit", **kw)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.aipw(d, y="y", outcome_model="poisson", **kw)
    with pytest.raises(MethodIncompatibility, match=r"\[0, 1\]"):
        sp.aipw(d, y="yc", outcome_model="probit", **kw)
    with pytest.raises(MethodIncompatibility, match="outcome_model"):
        sp.aipw(d, y="yb", outcome_model="cloglog", **kw)
