"""Clean, actionable errors when a result-consumer gets the wrong result type.

Both functions previously raised a raw, opaque error from deep inside the
implementation when handed an incompatible result:
- ``sp.evalue_from_result`` -> ``AttributeError: 'EconometricResults' object
  has no attribute 'estimate'`` on a plain regression result.
- ``sp.predict_cate`` -> ``AttributeError: 'CausalForest' object has no
  attribute 'model_info'`` on a causal-forest result.

They now raise a typed error naming the expected input, while still working on
the correct result type.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n = 600
    df = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    df["t"] = (df["x1"] + rng.randn(n) > 0).astype(int)
    df["y"] = 2.0 * df["t"] + df["x1"] + rng.randn(n)
    return df


def test_evalue_from_result_rejects_non_causal_result(data):
    reg = sp.regress("y ~ x1", data)  # EconometricResults, no .estimate
    with pytest.raises(MethodIncompatibility, match="CausalResult"):
        sp.evalue_from_result(reg)


def test_evalue_from_result_works_on_causal_result(data):
    ml = sp.metalearner(data, y="y", treat="t", covariates=["x1", "x2"], learner="t")
    ev = sp.evalue_from_result(ml)
    assert ev["evalue_estimate"] >= 1.0  # E-values are always >= 1


def test_predict_cate_rejects_result_without_estimator(data):
    cf = sp.causal_forest(
        data=data,
        Y=data["y"].values,
        T=data["t"].values,
        X=data[["x1", "x2"]].values,
    )
    with pytest.raises(ValueError, match="fitted estimator"):
        sp.predict_cate(cf, data)


def test_predict_cate_works_on_metalearner(data):
    ml = sp.metalearner(data, y="y", treat="t", covariates=["x1", "x2"], learner="t")
    out = sp.predict_cate(ml, data.head(10))
    assert np.asarray(out).shape == (10,)
