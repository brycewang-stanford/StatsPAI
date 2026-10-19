"""``sp.power('simulation', ...)`` and the power / power_ttest aliases.

Simulation power is checked against the exact noncentral-t power of the same
design (known truth), with tolerances in Monte Carlo standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _dgp(n, rng, effect_size=0.0):
    d = rng.permutation(np.repeat([0, 1], n // 2))
    return pd.DataFrame({"d": d, "y": effect_size * d + rng.normal(size=n)})


def _welch(df):
    return sp.difference_in_means(df, "y", "d")


def test_simulated_power_matches_exact_t_power():
    n_sims = 600
    r = sp.power(
        "simulation",
        n=[60, 120],
        effect_size=0.5,
        dgp=_dgp,
        estimator=_welch,
        n_sims=n_sims,
        seed=2026,
    )
    exact = [sp.power_ttest(n=n, delta=0.5).power for n in (60, 120)]
    for sim, ex in zip(r.power, exact):
        assert abs(sim - ex) < 4 * np.sqrt(ex * (1 - ex) / n_sims)
    tab = r.params["table"]
    assert np.all(np.abs(tab.bias) < 4 * tab.mc_se_bias)
    assert np.all(np.abs(tab.coverage - 0.95) < 4 * np.sqrt(0.95 * 0.05 / n_sims))


def test_size_under_the_null_and_regression_term():
    n_sims = 600
    r = sp.power(
        "simulation",
        n=100,
        effect_size=0.0,
        dgp=_dgp,
        estimator=lambda df: sp.regress("y ~ d", data=df),
        term="d",
        n_sims=n_sims,
        seed=5,
    )
    assert abs(r.power - 0.05) < 4 * np.sqrt(0.05 * 0.95 / n_sims)
    assert r.params["n_failed"] == 0


def test_tuple_estimator_and_reproducibility():
    def est(df):
        m = df.groupby("d")["y"]
        diff = m.mean()[1] - m.mean()[0]
        se = np.sqrt((m.var() / m.count()).sum())
        return diff, se

    a = sp.power(
        "simulation", n=80, effect_size=0.4, dgp=_dgp, estimator=est, n_sims=50, seed=9
    )
    b = sp.power(
        "simulation", n=80, effect_size=0.4, dgp=_dgp, estimator=est, n_sims=50, seed=9
    )
    assert a.power == b.power and a.params["rmse"] == b.params["rmse"]


def test_failed_draws_are_counted_and_warned():
    calls = {"k": 0}

    def flaky(df):
        calls["k"] += 1
        if calls["k"] % 10 == 0:
            raise np.linalg.LinAlgError("singular")
        return _welch(df)

    with pytest.warns(RuntimeWarning, match="draws failed"):
        r = sp.power(
            "simulation",
            n=40,
            effect_size=0.3,
            dgp=_dgp,
            estimator=flaky,
            n_sims=50,
            seed=1,
        )
    assert r.params["n_failed"] == 5 and r.params["n_sims"] == 45


def test_simulation_refuses_to_solve_for_n():
    with pytest.raises(MethodIncompatibility):
        sp.power(
            "simulation", n=None, effect_size=0.3, power=0.8, dgp=_dgp, estimator=_welch
        )


def test_power_ttest_designs_and_aliases():
    a = sp.power("twomeans", n=100, effect_size=0.5)
    b = sp.power_ttest(n=100, delta=0.5)
    assert a.power == b.power
    assert sp.power("ttest", n=100, delta=0.5).power == b.power
    c = sp.power_ttest(effect_size=0.5, power_target=0.8)
    d = sp.power_ttest(delta=0.5, power=0.8)
    assert c.n == d.n
    assert (
        sp.power("onemean", effect_size=0.4, power=0.9).n
        == sp.power_ttest(delta=0.4, power=0.9, type="one-sample").n
    )
    assert (
        sp.power("pairedmeans", n=30, effect_size=2, sigma=4).power
        == sp.power_ttest(n=30, delta=2, sd=4, type="paired").power
    )
    assert sp.power("rct", n=500, delta=0.3).power == sp.power_rct(500, 0.3).power
    with pytest.raises(MethodIncompatibility):
        sp.power("rct", n=500, delta=0.3, effect_size=0.3)
