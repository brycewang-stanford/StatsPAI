"""Ordinal outcomes whose category order is not declared are refused.

``sp.ologit("y ~ x")`` on string levels low / mid / high sorted them
alphabetically (high < low < mid) and reversed the slope with no warning
(rOpenSci audit probe P3). Numbers, booleans and ordered Categoricals carry
their order; text labels and unordered Categoricals of text do not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.regression.multinomial import outcome_categories, require_ordered_outcome

LEVELS = ["low", "mid", "high"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    code = np.digitize(1.1 * x + rng.logistic(size=n), [-1, 1])
    lab = np.array(LEVELS)[code]
    return pd.DataFrame(
        {
            "x": x,
            "y_int": code,
            "y_str": lab,
            "y_ord": pd.Categorical(lab, categories=LEVELS, ordered=True),
            "y_unord": pd.Categorical(lab),
            "y_objint": pd.Series(code.tolist(), dtype=object),
            "y_numcat": pd.Categorical(code),
            "g": np.repeat(np.arange(30), 20),
        }
    )


@pytest.mark.parametrize("fn", [sp.ologit, sp.oprobit])
@pytest.mark.parametrize("y", ["y_str", "y_unord"])
def test_ordered_models_refuse_unordered_labels(frame, fn, y):
    with pytest.raises(MethodIncompatibility) as err:
        fn(f"{y} ~ x", data=frame)
    assert "ordered=True" in err.value.recovery_hint
    assert err.value.diagnostics["levels_alphabetical"] == ["high", "low", "mid"]


@pytest.mark.parametrize("fn", [sp.ologit, sp.oprobit])
@pytest.mark.parametrize("y", ["y_ord", "y_objint", "y_numcat"])
def test_ordered_labels_equal_integer_codes(frame, fn, y):
    ref = fn("y_int ~ x", data=frame).params.to_numpy()
    got = fn(f"{y} ~ x", data=frame).params.to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
    assert ref[0] > 0  # the slope of the latent index is positive by design


def test_gologit_refuses_text_and_accepts_ordered(frame):
    with pytest.raises(MethodIncompatibility):
        sp.gologit(data=frame, y="y_str", x=["x"])
    ref = sp.gologit(data=frame, y="y_int", x=["x"]).params.to_numpy()
    got = sp.gologit(data=frame, y="y_ord", x=["x"]).params.to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_meologit_refuses_unordered_categorical(frame):
    with pytest.raises(MethodIncompatibility, match="meologit"):
        sp.meologit(data=frame, y="y_unord", x_fixed=["x"], group="g")


def test_bayes_ordered_probit_refuses_text(frame):
    with pytest.raises(MethodIncompatibility, match="bayes_regress"):
        sp.bayes_regress("y_str ~ x", frame, model="oprobit", draws=50, burnin=10)


def test_mlogit_base_follows_declared_category_order(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_str = sp.mlogit("y_str ~ x", data=frame)
        by_ord = sp.mlogit("y_ord ~ x", data=frame)
    # strings: sorted, so "high" is the base; Categorical: its first level
    assert by_str.model_info["base_category"] == "high"
    assert by_ord.model_info["base_category"] == "low"
    # a different base reparameterises the same model
    assert by_ord.model_info["categories"] == LEVELS
    assert by_ord.model_info["log_likelihood"] == pytest.approx(
        by_str.model_info["log_likelihood"], rel=1e-9
    )
    # [mid] vs [low] contrasts: beta_mid(base low) = beta_mid(base high) - beta_low(base high)
    b_s = by_str.params
    b_o = by_ord.params
    assert b_o["[mid]x"] == pytest.approx(b_s["[mid]x"] - b_s["[low]x"], abs=1e-5)


def test_require_ordered_outcome_accepts_numbers_and_bools():
    require_ordered_outcome(pd.Series([1, 2, 3]), function="t")
    require_ordered_outcome(pd.Series([True, False]), function="t")
    require_ordered_outcome(pd.Series([1.0, None, 3], dtype=object), function="t")
    with pytest.raises(MethodIncompatibility):
        require_ordered_outcome(pd.Series(["a", 1], dtype=object), function="t")


def test_outcome_categories_unchanged_without_ordinal():
    s = pd.Series(["b", "a", "c"])
    assert list(outcome_categories(s)) == ["a", "b", "c"]
