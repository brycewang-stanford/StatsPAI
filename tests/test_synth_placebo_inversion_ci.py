"""Constant-effect confidence interval for classic SCM by test inversion."""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.synth._core import placebo_inversion_ci, placebo_rank_pvalue


@pytest.fixture(scope="module")
def prop99():
    df = sp.datasets.california_prop99()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.synth(
            df,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )
    info = res.model_info
    post = np.asarray(info["gap_table"]["post_treatment"], dtype=bool)
    return {
        "res": res,
        "gap": np.asarray(info["gap_table"]["gap"], dtype=float),
        "G": np.asarray(info["placebo_gaps"], dtype=float),
        "w": np.asarray(info["placebo_weights"])[:, 0].astype(float),
        "post": post,
        "pre": ~post,
    }


def _p_value(c, s):
    """The RMSPE-ratio rank p-value of H0: effect = c, computed directly."""
    r1 = np.sqrt(
        np.mean((s["gap"][s["post"]] - c) ** 2) / np.mean(s["gap"][s["pre"]] ** 2)
    )
    rj = np.sqrt(
        np.mean((s["G"][s["post"]] + s["w"] * c) ** 2, axis=0)
        / np.mean(s["G"][s["pre"]] ** 2, axis=0)
    )
    return placebo_rank_pvalue(r1, rj)


def test_null_of_zero_is_the_reported_p_value(prop99):
    assert _p_value(0.0, prop99) == pytest.approx(prop99["res"].pvalue, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_interval_equals_brute_force(prop99, alpha):
    grid = np.linspace(-200, 150, 7001)  # step 0.05
    accepted = grid[np.array([_p_value(c, prop99) for c in grid]) >= alpha - 1e-12]
    lo, hi = placebo_inversion_ci(
        prop99["gap"], prop99["G"], prop99["w"], prop99["pre"], prop99["post"], alpha
    )
    assert lo == pytest.approx(accepted.min(), abs=0.05)
    assert hi == pytest.approx(accepted.max(), abs=0.05)
    assert lo <= accepted.min() and hi >= accepted.max()


def test_interval_agrees_with_the_test(prop99):
    """p = 3/39 = 0.077: zero is inside the 95% interval, outside the 90%."""
    res = prop99["res"]
    assert res.pvalue == pytest.approx(3 / 39)
    lo, hi = res.model_info["ci_permutation"]
    assert lo < 0.0 < hi
    lo, hi = placebo_inversion_ci(
        prop99["gap"], prop99["G"], prop99["w"], prop99["pre"], prop99["post"], 0.1
    )
    assert lo < res.estimate < hi < 0.0


def test_unattainable_level_gives_the_whole_line(prop99):
    """39 units: the smallest p-value is 1/39, so nothing is rejected at 2%."""
    assert placebo_inversion_ci(
        prop99["gap"], prop99["G"], prop99["w"], prop99["pre"], prop99["post"], 0.02
    ) == (-np.inf, np.inf)


def test_known_shift_is_recovered():
    """Placebos with no fit error except noise; the treated gap is noise + 5."""
    rng = np.random.default_rng(5)
    n_pre, n_post, n_placebo = 20, 10, 59
    pre = np.r_[np.ones(n_pre, bool), np.zeros(n_post, bool)]
    G = rng.standard_normal((n_pre + n_post, n_placebo))
    gap = rng.standard_normal(n_pre + n_post) + 5.0 * ~pre
    lo, hi = placebo_inversion_ci(gap, G, np.zeros(n_placebo), pre, ~pre, alpha=0.1)
    assert lo < 5.0 < hi
    assert hi - lo < 4.0 and lo > 0.0


def test_empty_acceptance_region_is_nan():
    """When every constant effect is rejected (the post-period gap is not
    a constant shift of the pre-period one) the set is empty."""
    pre = np.r_[np.ones(20, bool), np.zeros(10, bool)]
    rng = np.random.default_rng(0)
    G = rng.standard_normal((30, 59))
    gap = rng.standard_normal(30) * 0.2 + np.r_[np.zeros(20), np.linspace(-20, 20, 10)]
    lo, hi = placebo_inversion_ci(gap, G, np.zeros(59), pre, ~pre, alpha=0.1)
    assert np.isnan(lo) and np.isnan(hi)


def test_perfect_pre_fit_returns_nan():
    pre = np.r_[np.ones(5, bool), np.zeros(3, bool)]
    gap = np.r_[np.zeros(5), 1.0, 2.0, 3.0]
    G = np.random.default_rng(1).standard_normal((8, 30))
    lo, hi = placebo_inversion_ci(gap, G, np.zeros(30), pre, ~pre)
    assert np.isnan(lo) and np.isnan(hi)


def test_reported_interval_is_the_permutation_interval(prop99):
    """``ci`` is dual to ``pvalue``; the old normal interval is kept."""
    res = prop99["res"]
    info = res.model_info
    assert info["ci_method"] == "placebo_inversion"
    assert tuple(res.ci) == tuple(info["ci_permutation"])
    lo, hi = info["ci_normal"]
    assert lo == pytest.approx(res.estimate - 1.959964 * res.se, rel=1e-6)
    assert hi == pytest.approx(res.estimate + 1.959964 * res.se, rel=1e-6)


def test_few_donors_give_an_unbounded_interval_and_say_so():
    """Nine units: the smallest p-value is 1/9, so nothing is rejected at
    5% and the summary prints the infinite ends instead of blanks."""
    df = sp.datasets.california_prop99()
    keep = ["California"] + sorted(set(df["state"]) - {"California"})[:8]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.synth(
            df[df["state"].isin(keep)],
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )
    assert tuple(res.ci) == (-np.inf, np.inf)
    assert res.pvalue >= 1 / 9
    assert "[-inf,  inf]" in res.summary()
    assert np.isfinite(res.model_info["ci_normal"]).all()
