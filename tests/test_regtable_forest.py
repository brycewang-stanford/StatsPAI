"""sp.regtable tabulates causal forests and their scalar effects.

Through 1.42.1 a fitted ``sp.causal_forest`` gave a blank column (its
``params`` vector is empty) and ``cf.ate()`` / ``cf.att()`` were refused.
The column is now the forest's own doubly robust effect and inference.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    w = rng.binomial(1, 0.5, n)
    y = x + (1 + x) * w + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "w": w, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = sp.causal_forest("y ~ w | x", data=df, random_state=1, n_estimators=200)
    return sp.regress("y ~ w + x", data=df), cf


def _coef(table, i, term):
    return table.to_dict()["models"][i]["coefficients"][term]


def test_forest_column_is_its_ate_with_the_forests_inference(fits):
    ols, cf = fits
    table = sp.regtable(ols, cf, cf.att())
    ate, att = cf.ate(), cf.att()
    for i, effect in ((1, ate), (2, att)):
        c = _coef(table, i, effect.estimand)
        assert c["estimate"] == float(effect)
        assert c["std_error"] == effect.se
        assert c["p_value"] == effect.pvalue
        assert (c["conf_low"], c["conf_high"]) == effect.ci
        assert c["reference_df"] == "z"
    assert table.to_dict()["models"][1]["stats"]["N"] == 800
    # identity with the forest's dict-valued accessor
    detail = cf.average_treatment_effect()
    assert _coef(table, 1, "ATE")["estimate"] == pytest.approx(
        detail["estimate"], rel=1e-15
    )
    text = table.to_text()
    assert f"{float(ate):.3f}" in text and f"({ate.se:.3f})" in text


def test_interval_at_another_level_uses_the_normal(fits):
    _, cf = fits
    ate = cf.ate()
    from statspai.output.estimates import _ci_bounds, _extract_model_data

    lo, hi = _ci_bounds(_extract_model_data(cf), "ATE", 0.10)
    z = stats.norm.ppf(0.95)
    assert lo == pytest.approx(float(ate) - z * ate.se, rel=1e-12)
    assert hi == pytest.approx(float(ate) + z * ate.se, rel=1e-12)


def test_effect_without_inference_shows_no_standard_error():
    effect = sp.ScalarEffect(1.5, estimand="ATE", inference_error="no score")
    c = _coef(sp.regtable(effect), 0, "ATE")
    assert c["estimate"] == 1.5
    assert c["std_error"] is None or np.isnan(c["std_error"])
