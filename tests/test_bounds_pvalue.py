"""The p-value of a partially identified effect agrees with its interval.

``sp.lee_bounds`` and ``sp.manski_bounds`` report the midpoint of the
bounds as ``estimate``. Until 1.43.1 their ``pvalue`` was a two-sided test
of that midpoint against zero, so a result could print ``p < 0.001`` (and
three stars) next to a 95% interval that contained zero. The p-value is
now the one the reported interval implies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.bounds.lee_manski import _bounds_pvalue, _imbens_manski_cn


def _selection_data(shift: float, seed: int = 0, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    u = rng.normal(size=n)
    s = ((0.3 + 0.4 * d + u) > 0).astype(int)
    y = np.where(s == 1, 1 + shift * d + 0.8 * u + rng.normal(size=n), np.nan)
    return pd.DataFrame({"y": y, "d": d, "s": s})


def test_zero_inside_the_estimated_bounds_has_p_value_one():
    # true effect 0.5; Lee bounds about [-0.05, 0.71]
    df = _selection_data(0.5, seed=4)
    r = sp.lee_bounds(df, "y", "d", "s", se_method="analytic")
    lb, ub = r.model_info["lower_bound"], r.model_info["upper_bound"]
    assert lb < 0 < ub
    assert r.pvalue == 1.0
    # the midpoint is more than 3 of its standard errors from zero, which
    # is what the old p-value (below 0.002 here) tested
    assert abs(r.estimate / r.se) > 3


@pytest.mark.parametrize("shift", [0.5, 0.62, 0.7, 0.8, 1.5])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_lee_p_value_is_below_alpha_exactly_when_the_interval_excludes_zero(
    shift, alpha
):
    df = _selection_data(shift, seed=1)
    r = sp.lee_bounds(df, "y", "d", "s", se_method="analytic", alpha=alpha)
    excludes_zero = r.ci[0] > 0 or r.ci[1] < 0
    assert (r.pvalue < alpha) == excludes_zero
    # the p-value does not depend on the level the interval was asked for
    r05 = sp.lee_bounds(df, "y", "d", "s", se_method="analytic")
    assert r.pvalue == pytest.approx(r05.pvalue, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize(
    "width,se_lb,se_ub", [(0.0, 0.1, 0.1), (0.05, 0.1, 0.2), (3.0, 0.2, 0.1)]
)
def test_p_value_equals_alpha_when_the_interval_ends_at_zero(
    alpha, width, se_lb, se_ub
):
    """Inverting the Imbens-Manski interval: lb - c_n se_lb = 0 gives p = alpha."""
    c_n = _imbens_manski_cn(width, max(se_lb, se_ub), alpha)
    lb = c_n * se_lb
    assert _bounds_pvalue(lb, lb + width, se_lb, se_ub) == pytest.approx(
        alpha, abs=1e-9
    )
    # and on the other side of zero
    ub = -c_n * se_ub
    assert _bounds_pvalue(ub - width, ub, se_lb, se_ub) == pytest.approx(
        alpha, abs=1e-9
    )


def test_point_identified_limit_is_the_two_sided_normal_test():
    assert _bounds_pvalue(0.3, 0.3, 0.1, 0.1) == pytest.approx(
        2 * stats.norm.sf(3.0), rel=1e-12
    )


def test_wide_set_limit_is_the_one_sided_normal_test():
    assert _bounds_pvalue(0.3, 50.0, 0.1, 0.1) == pytest.approx(
        stats.norm.sf(3.0), rel=1e-12
    )


def test_manski_p_value_matches_its_interval():
    rng = np.random.default_rng(3)
    n = 2000
    d = rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.3 + 0.2 * d).astype(float)
    df = pd.DataFrame({"y": y, "d": d})
    # worst-case bounds always contain zero when the outcome range is [0, 1]
    r = sp.manski_bounds(df, "y", "d", y_lower=0.0, y_upper=1.0)
    assert r.model_info["lower_bound"] < 0 < r.model_info["upper_bound"]
    assert r.pvalue == 1.0
    # monotone treatment selection and response: [0, E[Y|D=1] - E[Y|D=0]]
    r = sp.manski_bounds(df, "y", "d", y_lower=0.0, y_upper=1.0, assumption="mts_mtr")
    assert r.model_info["lower_bound"] == 0.0
    assert r.pvalue == 1.0
    # the set-covering interval inverts to 2 * Phi(-distance / se)
    assert _bounds_pvalue(0.2, 0.5, 0.1, 0.3, "set") == pytest.approx(
        2 * stats.norm.sf(2.0), rel=1e-12
    )
    assert _bounds_pvalue(-0.5, -0.3, 0.1, 0.1, "set") == pytest.approx(
        2 * stats.norm.sf(3.0), rel=1e-12
    )


# ---------------------------------------------------------------------------
# sp.ml_bounds: the bootstrap band
# ---------------------------------------------------------------------------


def _ml_bounds_data(seed: int = 0, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.3 + 0.2 * d).astype(float)
    return pd.DataFrame({"y": y, "d": d, "x": rng.normal(size=n)})


def test_ml_bounds_band_has_the_width_of_the_bound_s_own_sampling_error():
    """The bound is a mean of observed unit terms; the band's half-width is
    the 97.5% normal quantile times their standard error.

    Tolerance: 2,000 bootstrap draws put the 2.5% quantile of a normal
    within about 4% of its value (one Monte Carlo standard error), so 15%
    is more than three of them.
    """
    df = _ml_bounds_data()
    res = sp.ml_bounds(
        df, "y", "d", ["x"], y_min=0.0, y_max=1.0, n_bootstrap=2000, random_state=1
    )
    y, d = df["y"].to_numpy(), df["d"].to_numpy()
    psi_lower = d * y - d * 1.0 - (1 - d) * y
    psi_upper = d * y + (1 - d) * 1.0 - (1 - d) * y
    # the unit terms average to the plain-means Manski bounds exactly
    assert psi_lower.mean() == pytest.approx(res.manski_lower, abs=1e-12)
    assert psi_upper.mean() == pytest.approx(res.manski_upper, abs=1e-12)
    z = stats.norm.ppf(0.975)
    half_lower = z * psi_lower.std(ddof=1) / np.sqrt(len(df))
    half_upper = z * psi_upper.std(ddof=1) / np.sqrt(len(df))
    assert res.manski_lower - res.lower == pytest.approx(half_lower, rel=0.15)
    assert res.upper - res.manski_upper == pytest.approx(half_upper, rel=0.15)
    # the per-bound intervals are centred on the plain-means bounds
    assert res.lower_ci[0] < res.manski_lower < res.lower_ci[1]
    assert res.upper_ci[0] < res.manski_upper < res.upper_ci[1]


def test_ml_bounds_band_does_not_shrink_when_the_covariate_is_noise():
    """Before, the band came from the spread of fitted values across
    covariate values, which is nil when the covariate is irrelevant: on
    this sample it reached 0.76 standard errors below the plain-means
    bound (0.47 below its own centre) where 1.96 are needed."""
    df = _ml_bounds_data(seed=2)
    res = sp.ml_bounds(
        df, "y", "d", ["x"], y_min=0.0, y_max=1.0, n_bootstrap=400, random_state=0
    )
    y, d = df["y"].to_numpy(), df["d"].to_numpy()
    se_lower = (d * y - d - (1 - d) * y).std(ddof=1) / np.sqrt(len(df))
    assert res.manski_lower - res.lower > 1.5 * se_lower
    assert res.upper - res.manski_upper > 1.5 * se_lower
