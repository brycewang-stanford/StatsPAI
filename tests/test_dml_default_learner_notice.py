"""Announced change of the ``sp.dml(model='plr')`` default learners.

The gradient-boosting default under-covers in the package's Monte Carlo
checks (``tests/coverage_monte_carlo/mechanisms/results_dml_plr_learners.json``
and ``tests/reliability/dml_learners_results.json``), so it is announced to
be replaced no earlier than 1.45. Today's numbers stay; a PLR call that
relies on a default learner gets a ``FutureWarning`` naming it, and the pin
the warning names reproduces the default bit for bit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

X = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(20261117)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=X)
    df["d"] = df["x1"] + rng.normal(size=n)
    df["y"] = 1.0 * df["d"] + np.sin(df["x2"]) + rng.normal(size=n)
    df["b"] = (df["x1"] + rng.normal(size=n) > 0).astype(int)
    df["yb"] = 1.0 * df["b"] + df["x2"] + rng.normal(size=n)
    return df


def _future(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [w for w in caught if issubclass(w.category, FutureWarning)]


def test_default_plr_warns_at_the_callers_line(data):
    _, fw = _future(lambda: sp.dml(data, y="y", treat="d", covariates=X))
    assert len(fw) == 1
    msg = str(fw[0].message)
    assert "ml_g and ml_m not given" in msg
    assert "no earlier than 1.45" in msg
    assert "ml_g='gbm', ml_m='gbm'" in msg
    assert fw[0].filename == __file__


def test_the_named_pin_reproduces_the_default_bit_for_bit(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        default = sp.dml(data, y="y", treat="d", covariates=X)
    kept, fw = _future(
        lambda: sp.dml(data, y="y", treat="d", covariates=X, ml_g="gbm", ml_m="gbm")
    )
    assert fw == []
    assert float(kept.estimate) == float(default.estimate)
    assert float(kept.se) == float(default.se)


def test_one_learner_given_warns_for_the_other(data):
    _, fw = _future(lambda: sp.dml(data, y="y", treat="d", covariates=X, ml_g="linear"))
    assert len(fw) == 1
    assert str(fw[0].message).startswith("dml(model='plr'): ml_m not given")


def test_class_entry_point_warns_on_fit(data):
    est = sp.DoubleML(data, "y", "d", X, "plr", ml_m="linear")
    _, fw = _future(est.fit)
    assert len(fw) == 1
    assert "ml_g not given" in str(fw[0].message)
    assert fw[0].filename == __file__


@pytest.mark.parametrize("model", ["irm", "iivm"])
def test_other_models_keep_their_defaults_silently(data, model):
    kw = {}
    if model == "iivm":
        data = data.assign(z=(data["x3"] > 0).astype(int))
        kw = {"instrument": "z"}
    _, fw = _future(
        lambda: sp.dml(data, y="yb", treat="b", covariates=X[:2], model=model, **kw)
    )
    assert fw == []
