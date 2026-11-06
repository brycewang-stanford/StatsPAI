"""``sp.optimal_design`` against Stata 18 ``power twomeans, knownsds``.

Reference numbers: ``_fixtures/optimal_design_Stata.csv``, written by
``_fixtures/_generate_optimal_design_Stata.do`` (Stata version in
``optimal_design_Stata_version.txt``). Eight calls: sample sizes with equal
and unequal allocation, a covariate R-squared (passed to Stata as
``sd(sigma * sqrt(1 - R2))``), two cluster designs (``cluster m1() m2()
rho()``), and the minimum detectable difference for a given n.

Sample sizes are integers and are compared exactly: per arm, cluster
counts and totals.

One documented difference. ``sp.optimal_design`` uses the textbook closed
form ``N = (z_{1-alpha/2} + z_{power})^2 ...``, which ignores the chance of
rejecting in the wrong direction. Stata's power for a two-sided test adds
that second tail. Below power 0.95 the tail is about 1e-6 or smaller, so
the unrounded sample size and the detectable difference differ by up to
2.5e-6 relative and the rounded sample sizes agree. The tests rebuild
Stata's unrounded figures from our standard error with the second tail
added, to 1e-9.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
REF = pd.read_csv(
    _FIX / "optimal_design_Stata.csv", na_values=".", skipinitialspace=True
).set_index("id")
REF.columns = [c.strip() for c in REF.columns]

# id -> (StatsPAI call, sigma, alpha, power, treated share)
CASES = {
    "a": (dict(mde=0.2), 1.0, 0.05, 0.8, 0.5),
    "b": (dict(mde=0.35, sigma=2, power=0.9, alpha=0.01), 2.0, 0.01, 0.9, 0.5),
    # Stata's group 1 is control, group 2 treated: nratio = N2 / N1 = 0.3 / 0.7
    "c": (dict(mde=0.2, prop_treat=0.3), 1.0, 0.05, 0.8, 0.3),
    "d": (dict(mde=0.2, r2=0.2), 1.0, 0.05, 0.8, 0.5),
    "e": (
        dict(design="cluster", mde=0.2, icc=0.05, cluster_size=20),
        1.0,
        0.05,
        0.8,
        0.5,
    ),
    "f": (
        dict(
            design="cluster", mde=0.15, sigma=1.5, power=0.85, icc=0.1, cluster_size=12
        ),
        1.5,
        0.05,
        0.85,
        0.5,
    ),
}


def _two_tailed_power(effect: float, se: float, alpha: float) -> float:
    z = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(effect / se - z) + stats.norm.cdf(-effect / se - z)


@pytest.mark.parametrize("cid", sorted(CASES))
def test_sample_sizes_equal_stata(cid):
    kw = CASES[cid][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.optimal_design(**kw)
    row = REF.loc[cid]
    assert res.n_total == int(row["N"])
    assert res.n_per_arm == int(max(row["N1"], row["N2"]))
    if kw.get("design") == "cluster":
        assert res.n_clusters == int(row["K1"] + row["K2"])


@pytest.mark.parametrize("cid", ["a", "b", "c", "e", "f"])
def test_unrounded_sample_size_is_stata_with_the_second_tail(cid):
    """Ours ignores the wrong-direction tail; adding it to our variance
    formula gives Stata's ``nfractional`` total to 1e-9."""
    kw, sigma, alpha, power, p = CASES[cid]
    row = REF.loc[cid]
    deff = 1.0
    if kw.get("design") == "cluster":
        deff = 1 + (kw["cluster_size"] - 1) * kw["icc"]
    mde = kw["mde"]

    def se(n_total):
        return sigma * np.sqrt(deff / (n_total * p * (1 - p)))

    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    ours_frac = (z * sigma / mde) ** 2 * deff / (p * (1 - p))
    stata_frac = float(row["N_frac"])
    # the closed form, before rounding up, is within the size of the tail
    assert ours_frac == pytest.approx(stata_frac, rel=3e-6)
    rebuilt = optimize.brentq(
        lambda n: _two_tailed_power(mde, se(n), alpha) - power,
        ours_frac * 0.9,
        ours_frac * 1.1,
        xtol=1e-12,
    )
    assert rebuilt == pytest.approx(stata_frac, rel=1e-9)


@pytest.mark.parametrize(
    "cid,kw,sigma,alpha,power,p",
    [
        ("g", dict(n=800), 1.0, 0.05, 0.8, 0.5),
        ("h", dict(n=500, sigma=2, power=0.9, prop_treat=0.6), 2.0, 0.05, 0.9, 0.6),
    ],
)
def test_detectable_difference_is_stata_with_the_second_tail(
    cid, kw, sigma, alpha, power, p
):
    res = sp.optimal_design(**kw)
    stata = float(REF.loc[cid, "delta"])
    assert res.mde == pytest.approx(stata, rel=2.5e-6)
    se = sigma / np.sqrt(kw["n"] * p * (1 - p))
    # our MDE is z * se exactly
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    assert res.mde == pytest.approx(z * se, rel=1e-12)
    rebuilt = optimize.brentq(
        lambda d: _two_tailed_power(d, se, alpha) - power,
        0.5 * res.mde,
        res.mde,
        xtol=1e-15,
    )
    assert rebuilt == pytest.approx(stata, rel=1e-9)
