"""Reference parity: ``sp.rlassologit`` vs R ``hdm::rlassologit`` / ``glmnet``.

``hdm::rlassologit`` is the logistic rigorous (post-)Lasso; its penalized
fit is ``glmnet(family="binomial", alpha=1, lambda=λ, standardize=TRUE)``
at a single data-driven ``λ``.  StatsPAI reproduces glmnet's binomial
lasso at that ``λ`` directly (IRLS + weighted coordinate descent).

Tolerance discipline
--------------------
- the glmnet engine's **selected support matches exactly**; its
  coefficients match R ``glmnet`` 4.1 to ~1e-6 (glmnet's own convergence
  tolerance — there is no tighter ground truth).
- ``post=True`` (default): coefficients/intercept/residuals come from an
  *unpenalized* logistic refit on the selected set, so they match
  ``hdm`` to ~1e-6 (observed ~1e-9).
- ``post=False``: coefficients are the glmnet-penalized fit, matched to
  ~1e-5.

Fixture lifecycle: ``_generate_rlassologit.R`` writes the data and the
hdm/glmnet reference; re-run only on contract change.

References
----------
- Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
  High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
  [@chernozhukov2016hdm]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.rlasso._logit import _glmnet_logit_lasso

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def R():
    with open(_FIXTURE_DIR / "rlassologit_R.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(_FIXTURE_DIR / "rlassologit_data.csv")
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    cols = list(df.drop(columns=["y"]).columns)
    return X, y, cols


def test_glmnet_engine_matches_R(data, R):
    """The internal binomial-lasso engine reproduces R glmnet at the same λ:
    exact support, coefficients to ~1e-6."""
    X, y, _ = data
    exp = R["glmnet_post"]
    beta, a0 = _glmnet_logit_lasso(X, y, exp["lambda"], intercept=True)
    sel = (np.where(np.abs(beta) > 1e-8)[0] + 1).tolist()
    assert sel == exp["sel"]
    np.testing.assert_allclose(beta, exp["beta"], atol=1e-5)
    np.testing.assert_allclose(a0, exp["a0"], atol=1e-5)


@pytest.mark.parametrize(
    "key,kwargs,btol",
    [
        ("rlassologit_post_int", dict(post=True, intercept=True), 1e-5),
        ("rlassologit_lasso_int", dict(post=False, intercept=True), 1e-4),
        ("rlassologit_post_noint", dict(post=True, intercept=False), 1e-5),
    ],
)
def test_rlassologit_matches_hdm(data, R, key, kwargs, btol):
    X, y, cols = data
    exp = R[key]
    fit = sp.rlassologit(X, y, colnames=cols, **kwargs)
    # exact support — the crux for post-Lasso correctness
    assert np.array_equal(fit.index, np.array(exp["index"], dtype=bool))
    assert fit.n_selected == exp["n_selected"]
    np.testing.assert_allclose(fit.beta, exp["beta"], atol=btol)
    np.testing.assert_allclose(fit.lambda0, exp["lambda0"], rtol=1e-8)
    np.testing.assert_allclose(fit.residuals, exp["residuals"], atol=btol)
    if exp["intercept"] is not None:
        np.testing.assert_allclose(fit.intercept, exp["intercept"], atol=btol)


def test_rlassologit_predict_matches_hdm(data, R):
    X, y, cols = data
    fit = sp.rlassologit(X, y, post=True, colnames=cols)
    np.testing.assert_allclose(
        fit.predict(X, type="response")[:10], R["predict_first10_response"], atol=1e-5
    )
    np.testing.assert_allclose(
        fit.predict(X, type="link")[:10], R["predict_first10_link"], atol=1e-5
    )
