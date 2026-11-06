"""Storage types that used to change what a model read without a word.

* a ``Decimal`` object column became a 200-level factor (probe P7);
* an int64 column near 2**62 collapsed in float64 and was reported
  "constant" with no mention of precision (P8);
* a constant effect modifier in ``sp.causal_forest`` changed the ATE with no
  note (P9: a string with one level and a constant number move it alike).
"""

from __future__ import annotations

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._formula_dtypes import coerce_formula_value_columns
from statspai.exceptions import StatsPAIWarning


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 1 + df.x + rng.normal(size=n)
    df["xd"] = [Decimal(repr(v)) for v in df.x]
    df["xo"] = pd.Series(np.round(df.x * 1000).astype(int).tolist(), dtype=object)
    df["big"] = (2**62 + np.arange(n)).astype("int64")
    return df


@pytest.mark.parametrize("fit", [sp.regress, sp.feols], ids=["regress", "feols"])
def test_decimal_column_is_read_as_numbers(frame, fit):
    with pytest.warns(StatsPAIWarning, match="numbers stored as Python objects"):
        got = fit("y ~ xd", data=frame)
    ref = fit("y ~ x", data=frame)
    np.testing.assert_allclose(got.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)
    assert len(got.params) == 2


def test_python_int_object_column_is_read_as_numbers(frame):
    with pytest.warns(StatsPAIWarning):
        got = sp.regress("y ~ xo", data=frame)
    ref = sp.regress("y ~ xo", data=frame.assign(xo=frame.xo.astype(float)))
    np.testing.assert_allclose(got.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)


def test_factor_coding_is_kept_when_asked(frame):
    small = frame.iloc[:40].assign(k=[Decimal(i % 3) for i in range(40)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatsPAIWarning)
        res = sp.regress("y ~ C(k)", data=small)
    assert len(res.params) == 3


def test_text_object_column_is_left_alone():
    df = pd.DataFrame({"s": ["a", 1, 2.0], "v": [1.0, 2.0, 3.0]})
    out = coerce_formula_value_columns(df, ["s", "v"])
    assert out is df


def test_large_int64_column_is_flagged(frame):
    with pytest.warns(StatsPAIWarning, match=r"beyond 2\*\*53") as rec:
        sp.regress("y ~ x + big", data=frame)
    w = next(r.message for r in rec if "2**53" in str(r.message))
    assert w.diagnostics["n_distinct_int"] == 200
    assert w.diagnostics["n_distinct_float64"] < 200


def test_int64_within_exact_range_is_silent(frame):
    d = frame.assign(year=np.arange(len(frame), dtype="int64") + 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatsPAIWarning)
        sp.regress("y ~ x + year", data=d)


def test_causal_forest_flags_constant_modifiers():
    rng = np.random.default_rng(4)
    n = 400
    X = rng.normal(size=(n, 2))
    W = (rng.uniform(size=n) < 0.5).astype(int)
    df = pd.DataFrame(
        {"y": X[:, 0] + W * (1 + X[:, 1]) + rng.normal(size=n), "w": W}
    ).assign(x1=X[:, 0], x2=X[:, 1], s="a", c=1.0, s2=np.where(X[:, 1] > 0, "h", "l"))
    for extra, flagged in [("s", "s[a]"), ("c", "c")]:
        with pytest.warns(StatsPAIWarning, match="single value") as rec:
            sp.causal_forest(
                data=df,
                y="y",
                treat="w",
                covariates=["x1", "x2", extra],
                n_estimators=50,
                random_state=0,
            )
        msg = next(r.message for r in rec if "single value" in str(r.message))
        assert msg.diagnostics["constant_covariates"] == [flagged]
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatsPAIWarning)
        sp.causal_forest(
            data=df,
            y="y",
            treat="w",
            covariates=["x1", "x2", "s2"],
            n_estimators=50,
            random_state=0,
        )
