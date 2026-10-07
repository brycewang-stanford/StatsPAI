"""``sp.ipw(propensity=)``: inverse weighting by a design probability."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility


def _trial(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = np.where(x > 0, 0.7, 0.4)
    a = rng.binomial(1, p)
    y = 1.0 * a + x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "a": a, "y": y, "p": p})


def test_constant_probability_is_the_difference_in_means() -> None:
    # Hajek weights with one probability for everyone cancel: the estimate
    # is the difference in means and the sandwich its HC0 standard error.
    df = _trial()
    res = sp.ipw(
        df, y="y", treat="a", covariates=["x"], propensity=0.5, se_method="sandwich"
    )
    y1, y0 = df.loc[df.a == 1, "y"], df.loc[df.a == 0, "y"]
    np.testing.assert_allclose(res.estimate, y1.mean() - y0.mean(), rtol=1e-12)
    se = np.sqrt(
        ((y1 - y1.mean()) ** 2).sum() / len(y1) ** 2
        + ((y0 - y0.mean()) ** 2).sum() / len(y0) ** 2
    )
    np.testing.assert_allclose(res.se, se, rtol=1e-12)
    assert res.model_info["propensity"] == "known"


def test_design_probabilities_enter_the_weights_as_given() -> None:
    df = _trial()
    res = sp.ipw(
        df, y="y", treat="a", covariates=["x"], propensity="p", se_method="sandwich"
    )
    a, y, p = (df[c].to_numpy(float) for c in ("a", "y", "p"))
    w1, w0 = a / p, (1 - a) / (1 - p)
    np.testing.assert_allclose(
        res.estimate, np.sum(w1 * y) / w1.sum() - np.sum(w0 * y) / w0.sum(), rtol=1e-12
    )
    boot = sp.ipw(
        df, y="y", treat="a", covariates=["x"], propensity="p", n_bootstrap=400, seed=1
    )
    assert boot.estimate == res.estimate
    # Bootstrap and sandwich estimate the same variance.
    assert abs(boot.se / res.se - 1) < 0.15


def test_estimating_a_known_propensity_is_the_more_precise_choice() -> None:
    # The textbook surprise: the fitted propensity absorbs chance imbalance.
    df = _trial(n=2000, seed=3)
    kw = dict(y="y", treat="a", covariates=["x"], se_method="sandwich")
    known = sp.ipw(df, propensity="p", **kw)
    fitted = sp.ipw(df, **kw)
    assert fitted.se < known.se


def test_out_of_range_probability_is_refused() -> None:
    with pytest.raises(MethodIncompatibility, match="strictly between"):
        sp.ipw(_trial(), y="y", treat="a", covariates=["x"], propensity=0.0)
