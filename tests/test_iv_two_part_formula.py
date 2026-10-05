"""R's two-part IV formula, ``y ~ regressors | instruments``, in sp.iv."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.regression.iv import _two_part_to_block_formula as rewrite


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 500
    z1, z2, x, u = rng.standard_normal((4, n))
    g = rng.integers(0, 3, n)
    d = 0.6 * z1 + 0.4 * z2 + 0.5 * u + rng.standard_normal(n)
    y = 1 + 2 * d + 0.5 * x + 0.3 * g + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "z1": z1, "z2": z2, "g": g})


def test_rewrite():
    assert rewrite("y ~ x + d | x + z1 + z2") == "y ~ x + (d ~ z1 + z2)"
    assert rewrite("Y ~ X | Z1 + Z2") == "Y ~ (X ~ Z1 + Z2)"
    assert rewrite("y ~ x + d - 1 | x + z - 1") == "y ~ x + (d ~ z) - 1"
    # already in block form, with or without absorbed effects: untouched
    assert rewrite("y ~ (d ~ z) + x") == "y ~ (d ~ z) + x"
    assert rewrite("y ~ x + (d ~ z) | fe") == "y ~ x + (d ~ z) | fe"
    assert rewrite("y ~ [d ~ z] + x") == "y ~ [d ~ z] + x"


@pytest.mark.parametrize("fit", [sp.iv, sp.ivreg])
def test_same_fit_as_block_formula(data, fit):
    two_part = fit("y ~ x + d | x + z1 + z2", data=data)
    block = fit("y ~ x + (d ~ z1 + z2)", data=data)
    pd.testing.assert_series_equal(
        two_part.params[block.params.index], block.params, rtol=0, atol=0
    )
    pd.testing.assert_series_equal(
        two_part.std_errors[block.params.index], block.std_errors, rtol=0, atol=0
    )
    assert abs(two_part.params["d"] - 2) < 0.3


def test_categorical_terms_on_both_sides(data):
    two_part = sp.iv("y ~ x + C(g) + d | x + C(g) + z1 + z2", data=data)
    block = sp.iv("y ~ x + C(g) + (d ~ z1 + z2)", data=data)
    assert two_part.params["d"] == block.params["d"]


def test_prediction_uses_the_rewritten_formula(data):
    fit = sp.iv("y ~ x + d | x + z1 + z2", data=data)
    block = sp.iv("y ~ x + (d ~ z1 + z2)", data=data)
    np.testing.assert_allclose(fit.predict(data.head()), block.predict(data.head()))


def test_degenerate_two_part_formulas_are_refused(data):
    with pytest.raises(MethodIncompatibility, match="no endogenous regressor"):
        sp.iv("y ~ x + d | x + d", data=data)
    with pytest.raises(MethodIncompatibility, match="no excluded instrument"):
        sp.iv("y ~ x + d | x", data=data)
    with pytest.raises(MethodIncompatibility, match="shorthand"):
        sp.iv("y ~ x + d | . + z1", data=data)
