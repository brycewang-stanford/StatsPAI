"""The ``sp.dml(model='plr')`` default learners after the 1.45 switch.

Gradient boosting, the default through 1.44, under-covers in the
package's Monte Carlo checks
(``tests/coverage_monte_carlo/mechanisms/results_dml_plr_learners.json``
and ``tests/reliability/dml_learners_results.json``). 1.44 announced its
replacement by short stacking with a ``FutureWarning``
(``tests/reliability/dml_default_candidates.py`` records the choice) and
1.45 made the switch. The tests pin the switch: an unset PLR nuisance is
the default roster, the warning is gone, and ``'gbm'`` builds the
earlier default exactly.
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


def test_default_plr_is_short_stacking_and_silent(data):
    default, fw = _future(lambda: sp.dml(data, y="y", treat="d", covariates=X))
    assert fw == []
    stack = sp.dml(data, y="y", treat="d", covariates=X, ml_g="stack", ml_m="stack")
    assert float(default.estimate) == float(stack.estimate)
    assert float(default.se) == float(stack.se)
    assert default.model_info["ml_g"] == default.model_info["ml_m"] == "ShortStack"


def test_gbm_alias_builds_the_default_before_1_45(data):
    from sklearn.ensemble import GradientBoostingRegressor

    def old():
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )

    kept, fw = _future(
        lambda: sp.dml(data, y="y", treat="d", covariates=X, ml_g="gbm", ml_m="gbm")
    )
    assert fw == []
    explicit = sp.dml(data, y="y", treat="d", covariates=X, ml_g=old(), ml_m=old())
    assert float(kept.estimate) == float(explicit.estimate)
    assert float(kept.se) == float(explicit.se)


def test_one_learner_given_stacks_the_other(data):
    half, fw = _future(
        lambda: sp.dml(data, y="y", treat="d", covariates=X, ml_g="linear")
    )
    assert fw == []
    full = sp.dml(data, y="y", treat="d", covariates=X, ml_g="linear", ml_m="stack")
    assert float(half.estimate) == float(full.estimate)
    assert float(half.se) == float(full.se)
    assert half.model_info["ml_m"] == "ShortStack"
    assert "ml_g" not in half.model_info["stacking"]


def test_class_entry_point_follows(data):
    est = sp.DoubleML(data, "y", "d", X, "plr", ml_m="linear")
    fit, fw = _future(est.fit)
    assert fw == []
    direct = sp.dml(data, y="y", treat="d", covariates=X, ml_m="linear")
    assert float(fit.estimate) == float(direct.estimate)
    assert float(fit.se) == float(direct.se)


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
