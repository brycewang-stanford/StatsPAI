"""Option scan, round 5: ``power/``.

Each test states the correct behaviour; the truth is a closed form or the
documented contract of the function.
"""

import pytest
from scipy.stats import norm

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# -- alternative= spellings in the study-design calculators ----------------


@pytest.mark.parametrize("spelling", ["two.sided", "two_sided", "Two-Sided"])
def test_two_proportions_two_sided_spellings_are_two_sided(spelling):
    ref = sp.power_two_proportions(n=400, p1=0.5, p2=0.65).power
    got = sp.power_two_proportions(n=400, p1=0.5, p2=0.65, alternative=spelling).power
    assert got == ref


@pytest.mark.parametrize(
    "fn,kw",
    [
        (sp.power_two_proportions, dict(n=400, p1=0.5, p2=0.65)),
        (sp.power_logrank, dict(n=300, hazard_ratio=0.6, prob_event=0.7)),
        (sp.power_case_control, dict(n_cases=200)),
    ],
)
def test_unknown_alternative_is_refused(fn, kw):
    with pytest.raises(MethodIncompatibility):
        fn(alternative="2-sided", **kw)


def test_logrank_two_sided_spelling():
    kw = dict(n=300, hazard_ratio=0.6, prob_event=0.7)
    assert (
        sp.power_logrank(alternative="two_sided", **kw).power
        == sp.power_logrank(**kw).power
    )


# -- sample-size solver below n = 11 -----------------------------------------


def test_solve_for_n_returns_the_minimum_below_eleven():
    # cluster RCT: 30 per cluster, icc 0.05, effect 1 sd. The power formula
    # gives 0.858 at 3 clusters and 0.697 at 2, so the answer is 3.
    res = sp.power("cluster_rct", effect_size=1.0, power=0.8, cluster_size=30, icc=0.05)
    assert res.n == 3
    p = lambda k: sp.power_cluster_rct(  # noqa: E731
        n_clusters=k, cluster_size=30, effect_size=1.0, icc=0.05
    ).power
    assert p(3) >= 0.8 > p(2)


def test_solve_for_n_rct_small():
    res = sp.power("rct", effect_size=3.0, power=0.8)
    p = lambda k: sp.power_rct(n=k, effect_size=3.0).power  # noqa: E731
    assert p(res.n) >= 0.8 > p(res.n - 1)


def test_solve_for_n_unchanged_above_ten():
    res = sp.power("rct", effect_size=0.2, power=0.8)
    p = lambda k: sp.power_rct(n=k, effect_size=0.2).power  # noqa: E731
    assert p(res.n) >= 0.8 > p(res.n - 1)


# -- MDE solver on the outcome scale -----------------------------------------


@pytest.mark.parametrize("sigma", [1e-4, 1000.0])
def test_mde_scales_with_sigma(sigma):
    # power_rct depends on effect / sigma only, so MDE(sigma) = sigma MDE(1).
    base = sp.mde("rct", n=100).effect_size
    got = sp.mde("rct", n=100, sigma=sigma).effect_size
    assert got == pytest.approx(sigma * base, rel=1e-5)
    # and the closed form: (z_{1-a/2} + z_{0.8}) * 2 sigma / sqrt(n)
    closed = (norm.ppf(0.975) + norm.ppf(0.8)) * 2 * sigma / 10
    assert got == pytest.approx(closed, rel=1e-5)
