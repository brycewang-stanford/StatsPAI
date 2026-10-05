"""The studentized (bootstrap-t) interval of ``sp.bootstrap``.

Chapter 2 and 11 of *Causal Inference in R* report ``rsample::int_t``
intervals. The interval is defined from the resampled estimates and their
standard errors as ``[est - q(1 - a/2) se, est - q(a/2) se]`` with ``q``
the quantiles of ``t* = (est* - est) / se*``. The resamples themselves
cannot be matched across languages, so the tests check the definition on
the draws ``sp.bootstrap`` made, and the behaviour the method is used for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


def _mean_and_se(frame: pd.DataFrame):
    x = frame["x"].to_numpy()
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def test_interval_is_the_definition_on_the_draws_made():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.exponential(size=60)})
    seen = []

    def statistic(frame):
        pair = _mean_and_se(frame)
        seen.append(pair)
        return pair

    res = sp.bootstrap(df, statistic, n_boot=400, ci_method="studentized", seed=1)
    (est, se), draws = seen[0], np.array(seen[1:])
    t_star = (draws[:, 0] - est) / draws[:, 1]
    lo, hi = np.percentile(t_star, [2.5, 97.5])
    assert res.estimate == pytest.approx(est)
    assert res.ci_lower == pytest.approx(est - hi * se, rel=1e-12)
    assert res.ci_upper == pytest.approx(est - lo * se, rel=1e-12)
    assert res.ci_method == "studentized"
    # the reported standard error is still the bootstrap one
    assert res.se == pytest.approx(draws[:, 0].std(ddof=1), rel=1e-12)


def test_skewed_data_give_an_interval_that_reaches_further_right():
    """For the mean of a right-skewed sample the bootstrap-t interval is
    shifted right of the symmetric one; that asymmetry is its point."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.lognormal(sigma=1.2, size=40)})
    stud = sp.bootstrap(df, _mean_and_se, n_boot=2000, ci_method="studentized", seed=0)
    est, se = _mean_and_se(df)
    assert stud.ci_upper - est > est - stud.ci_lower
    crit = stats.t.ppf(0.975, len(df) - 1)
    assert stud.ci_upper > est + crit * se


def test_coverage_for_a_skewed_mean_beats_the_percentile_interval():
    truth = np.exp(0.5)  # mean of a standard lognormal
    rng = np.random.default_rng(11)
    hits = {"studentized": 0, "percentile": 0}
    reps = 200
    for r in range(reps):
        df = pd.DataFrame({"x": rng.lognormal(size=25)})
        for method in hits:
            stat = (
                _mean_and_se if method == "studentized" else (lambda d: d["x"].mean())
            )
            res = sp.bootstrap(df, stat, n_boot=399, ci_method=method, seed=r)
            hits[method] += res.ci_lower <= truth <= res.ci_upper
    cover = {k: v / reps for k, v in hits.items()}
    # nominal 0.95; with n = 25 lognormal draws the percentile interval
    # undercovers badly and the studentized one much less (Monte Carlo
    # standard error about 0.02 at 200 replications)
    assert cover["studentized"] > cover["percentile"] + 0.03
    assert cover["studentized"] > 0.88


def test_statistic_must_return_estimate_and_se():
    df = pd.DataFrame({"x": np.arange(20.0)})
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="pair"):
        sp.bootstrap(df, lambda d: d["x"].mean(), ci_method="studentized", n_boot=20)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="standard error"):
        sp.bootstrap(
            df, lambda d: (d["x"].mean(), 0.0), ci_method="studentized", n_boot=20
        )
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="Unknown ci_method"):
        sp.bootstrap(df, lambda d: d["x"].mean(), ci_method="t", n_boot=20)


def test_resamples_without_a_standard_error_are_counted_as_failed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=30)})
    calls = {"n": 0}

    def statistic(frame):
        calls["n"] += 1
        est, se = _mean_and_se(frame)
        return est, (np.nan if calls["n"] % 10 == 0 else se)

    with pytest.warns(RuntimeWarning, match="replications failed"):
        res = sp.bootstrap(df, statistic, n_boot=200, ci_method="studentized", seed=0)
    assert res.n_boot == 180
