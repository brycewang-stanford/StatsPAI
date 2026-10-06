"""``sp.causal_gap``: the critical causal gap of Schuler and van der Laan."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility


def test_book_example() -> None:
    # "our estimated effect was 1 +/- 0.5 ... the exact value of the causal
    # gap for which statistical significance ceases to hold is 0.5", and
    # 0.2 when effects below 0.3 are of no practical interest.
    tbl = sp.causal_gap(1.0, ci=(0.5, 1.5), null=[0.0, 0.3])
    np.testing.assert_allclose(tbl["critical_gap"], [0.5, 0.2])
    np.testing.assert_allclose(tbl["gap_to_point"], [1.0, 0.7])
    np.testing.assert_allclose(tbl["share_of_estimate"], [0.5, 0.2 / 0.7])
    assert tbl["conclusion_holds"].all()


def test_gap_is_the_distance_from_the_nearer_limit() -> None:
    z = stats.norm.ppf(0.975)
    up = sp.causal_gap(2.0, se=0.5)
    np.testing.assert_allclose(up["critical_gap"].iloc[0], 2.0 - z * 0.5)
    down = sp.causal_gap(-2.0, se=0.5)
    np.testing.assert_allclose(down["critical_gap"].iloc[0], -(2.0 - z * 0.5))
    inside = sp.causal_gap(0.5, se=0.5)
    assert inside["critical_gap"].iloc[0] == 0.0
    assert not inside["conclusion_holds"].iloc[0]


def test_alpha_widens_the_interval() -> None:
    a = sp.causal_gap(2.0, se=0.5, alpha=0.05)["critical_gap"].iloc[0]
    b = sp.causal_gap(2.0, se=0.5, alpha=0.01)["critical_gap"].iloc[0]
    assert b < a


def test_ratio_scale_is_the_difference_scale_of_the_logs() -> None:
    ratio = sp.causal_gap(1.4, ci=(1.26, 1.56), scale="ratio")
    logs = sp.causal_gap(np.log(1.4), ci=(np.log(1.26), np.log(1.56)))
    np.testing.assert_allclose(
        np.log(ratio["critical_gap"]), logs["critical_gap"], rtol=1e-12
    )
    np.testing.assert_allclose(
        ratio["share_of_estimate"], logs["share_of_estimate"], rtol=1e-12
    )
    protective = sp.causal_gap(0.5, ci=(0.4, 0.625), scale="ratio")
    np.testing.assert_allclose(protective["critical_gap"].iloc[0], 0.625)
    from_se = sp.causal_gap(1.4, se=0.05, scale="ratio")
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(from_se["ci_lower"].iloc[0], 1.4 * np.exp(-z * 0.05))


def test_curve_crosses_at_the_critical_gap() -> None:
    tbl = sp.causal_gap(1.0, ci=(0.5, 1.5), n_grid=201)
    curve = tbl.attrs["curve"]
    assert list(curve.columns) == [
        "gap",
        "estimate",
        "ci_lower",
        "ci_upper",
        "conclusion_holds",
    ]
    assert curve["gap"].iloc[0] == 0.0 and curve["gap"].iloc[-1] == 2.0
    held = curve.loc[curve["conclusion_holds"], "gap"]
    # Significance is lost at 0.5 and regained, with the opposite sign,
    # once the whole interval is below zero (gap > 1.5).
    assert (
        held[held < 1.0].max()
        < 0.5
        <= curve.loc[~curve["conclusion_holds"], "gap"].min()
    )
    np.testing.assert_allclose(curve["estimate"], 1.0 - curve["gap"])


def test_accepts_a_fitted_result() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-x)))
    y = 2.0 * a + x + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "a": a, "y": y})
    res = sp.aipw(df, y="y", treat="a", covariates=["x"])
    tbl = sp.causal_gap(res)
    np.testing.assert_allclose(tbl["critical_gap"].iloc[0], res.ci[0])
    assert tbl["estimate"].iloc[0] == res.estimate


def test_bad_input_is_refused() -> None:
    with pytest.raises(MethodIncompatibility, match="se= or ci="):
        sp.causal_gap(1.0)
    with pytest.raises(MethodIncompatibility, match="scale"):
        sp.causal_gap(1.0, se=1.0, scale="odds")
    with pytest.raises(MethodIncompatibility, match="positive"):
        sp.causal_gap(-1.0, se=1.0, scale="ratio")
    with pytest.raises(MethodIncompatibility, match="contain the estimate"):
        sp.causal_gap(1.0, ci=(1.2, 1.5))
    with pytest.raises(MethodIncompatibility, match="result"):
        sp.causal_gap("not a result")
