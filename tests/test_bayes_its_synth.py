"""Tests for the Bayesian time-series designs ``sp.bayes_its`` / ``sp.bayes_synth``.

Both require the optional ``bayes`` extra (PyMC); the whole module skips when it
is absent, matching the rest of the Bayesian suite. Happy paths recover a known
effect on simulated / canonical data with small chains; validation tests lock in
the fail-loud contract. Both designs also feed the shared counterfactual contract
(``sp.counterfactual_data``) with credible bands.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz")

import statspai as sp  # noqa: E402

_SAMPLE = dict(draws=400, tune=400, chains=2, random_state=0, progressbar=False)


# --------------------------------------------------------------------- #
#  bayes_its
# --------------------------------------------------------------------- #
def _its_df(seed=0, n=60, level=3.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 1 + 0.1 * t + (t >= 30) * level + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "t": t})


def test_bayes_its_recovers_level_change():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bayes_its(_its_df(), y="y", time="t", intervention=30, **_SAMPLE)
    assert isinstance(res, sp.BayesianCausalResult)
    assert res.estimand == "level change"
    assert 2.4 < res.posterior_mean < 3.8  # recovers +3 (plus slope drift)
    assert res.hdi_lower < res.posterior_mean < res.hdi_upper
    assert "slope_change" in res.model_info


def test_bayes_its_feeds_counterfactual_contract():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bayes_its(_its_df(), y="y", time="t", intervention=30, **_SAMPLE)
    data = sp.counterfactual_data(res)
    assert {"observed", "counterfactual", "point_effect", "post"}.issubset(data.columns)
    assert {"cf_lower", "cf_upper"}.issubset(data.columns)  # credible bands
    np.testing.assert_allclose(
        data["point_effect"], data["observed"] - data["counterfactual"]
    )


def test_bayes_its_missing_intervention_raises():
    with pytest.raises(ValueError, match="intervention"):
        sp.bayes_its(_its_df(), y="y", time="t")


def test_bayes_its_intervention_out_of_range_raises():
    with pytest.raises(ValueError, match="must satisfy"):
        sp.bayes_its(_its_df(n=40), y="y", time="t", intervention=40)


def test_bayes_its_missing_column_raises():
    with pytest.raises(ValueError, match="not found"):
        sp.bayes_its(_its_df(), y="nope", time="t", intervention=30)


# --------------------------------------------------------------------- #
#  bayes_synth
# --------------------------------------------------------------------- #
def _synth_call(**over):
    kwargs = dict(
        outcome="packspercapita",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    kwargs.update(over)
    return sp.bayes_synth(sp.california_prop99(), **kwargs, **_SAMPLE)


def test_bayes_synth_recovers_negative_att():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _synth_call()
    assert isinstance(res, sp.BayesianCausalResult)
    assert res.estimand == "ATT"
    assert res.posterior_mean < 0  # Prop 99 cut cigarette consumption
    weights = res.model_info["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6  # simplex


def test_bayes_synth_feeds_counterfactual_contract():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _synth_call()
    data = sp.counterfactual_data(res)
    assert {"cf_lower", "cf_upper", "post"}.issubset(data.columns)
    # post-period mean gap equals the reported ATT (definitional)
    post_gap = data.loc[data["post"], "point_effect"].mean()
    assert abs(post_gap - res.posterior_mean) < 0.5


def test_bayes_synth_unknown_treated_unit_raises():
    with pytest.raises(ValueError, match="treated_unit"):
        _synth_call(treated_unit="Atlantis")


def test_bayes_synth_too_few_donors_raises():
    with pytest.raises(ValueError, match="donor"):
        _synth_call(donors=["Nevada"])


def test_bayes_synth_no_post_period_raises():
    with pytest.raises(ValueError, match="post"):
        _synth_call(treatment_time=3000)


# --------------------------------------------------------------------- #
#  Discoverability
# --------------------------------------------------------------------- #
def test_exports():
    assert hasattr(sp, "bayes_its") and hasattr(sp, "bayes_synth")
    names = [str(getattr(f, "name", f)) for f in sp.list_functions()]
    assert "bayes_its" in names
    assert "bayes_synth" in names


def test_bayes_synth_interval_includes_the_treated_units_own_noise():
    """Known asymptotics. With a long pre-period the weights are pinned
    down, and what remains uncertain about the average effect over ``T1``
    post periods is the treated unit's own noise, ``sigma / sqrt(T1)``.
    Before 1.39 the posterior had the weights' uncertainty only, here a
    fifth of the right standard deviation."""
    rng = np.random.default_rng(5)
    n_donors, t0, t1, sigma = 5, 80, 5, 0.5
    T = t0 + t1
    factor = np.cumsum(rng.normal(size=T)) * 0.5
    load = rng.uniform(0.5, 1.5, size=n_donors)
    donors = np.outer(factor, load) + 0.05 * rng.normal(size=(T, n_donors))
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    treated = donors @ w + sigma * rng.normal(size=T)
    treated[t0:] += 2.0
    Y = np.column_stack([treated, donors])
    rows = [(u, t, Y[t, u]) for u in range(n_donors + 1) for t in range(T)]
    df = pd.DataFrame(rows, columns=["unit", "time", "y"])
    r = sp.bayes_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=t0,
        draws=800,
        tune=800,
        chains=2,
        random_state=3,
    )
    floor = sigma / np.sqrt(t1)  # 0.224
    assert 0.8 * floor < r.posterior_sd < 1.4 * floor
    assert r.hdi_lower < 2.0 < r.hdi_upper
    # the band is predictive after treatment, a band for the mean before
    d = r.model_info["detail"]
    width = d["cf_upper"] - d["cf_lower"]
    assert width[d["post"]].mean() > 3 * width[~d["post"]].mean()
