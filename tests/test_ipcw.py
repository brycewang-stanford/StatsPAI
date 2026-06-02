"""Tests for inverse-probability-of-censoring weights (``sp.ipcw``).

IPCW corrects for informative censoring / loss-to-follow-up — a core tool
for cohort epidemiology and per-protocol target-trial analyses. The
``censoring`` module previously had no dedicated test file; these checks pin
the weight construction's defining properties.
"""
import numpy as np
import pandas as pd

import statspai as sp


def _censored_cohort(n=1000, seed=0):
    """Right-censored times where censoring depends on a covariate x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    event_time = rng.exponential(1.0, n)
    censor_time = rng.exponential(np.exp(0.4 * x))   # informative censoring
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"time": time, "event": event, "x": x})


def test_ipcw_weights_shape_and_positivity():
    df = _censored_cohort(seed=0)
    res = sp.ipcw(df, time="time", event="event", censor_covariates=["x"])
    w = np.asarray(res.weights, dtype=float)
    assert w.shape == (len(df),)
    assert np.all(w > 0)
    assert np.all(np.isfinite(w))


def test_unstabilized_weights_are_at_least_one():
    df = _censored_cohort(seed=1)
    res = sp.ipcw(df, time="time", event="event", censor_covariates=["x"],
                  stabilize=False)
    w = np.asarray(res.weights, dtype=float)
    # Unstabilized IPC weights are 1 / P(uncensored) >= 1 by construction.
    assert w.min() >= 1.0 - 1e-6
    assert res.stabilized is False


def test_stabilized_weights_centre_near_one():
    df = _censored_cohort(seed=2)
    res = sp.ipcw(df, time="time", event="event", censor_covariates=["x"],
                  stabilize=True)
    w = np.asarray(res.weights, dtype=float)
    # Stabilized weights are designed to have mean ~ 1.
    assert res.stabilized is True
    assert abs(float(w.mean()) - 1.0) < 0.2
    assert res.summary_stats["effective_sample_size"] <= len(df)
    assert res.summary_stats["effective_sample_size"] > 0


def test_truncation_caps_extreme_weights():
    df = _censored_cohort(seed=3)
    untrunc = sp.ipcw(df, time="time", event="event", censor_covariates=["x"],
                      stabilize=False, truncate=None)
    trunc = sp.ipcw(df, time="time", event="event", censor_covariates=["x"],
                    stabilize=False, truncate=(0.05, 0.95))
    assert np.max(trunc.weights) <= np.max(untrunc.weights) + 1e-9


def test_no_censoring_gives_unit_weights():
    df = _censored_cohort(seed=4)
    df = df.assign(event=1)   # everyone has the event observed; nobody censored
    res = sp.ipcw(df, time="time", event="event", censor_covariates=["x"],
                  stabilize=False)
    w = np.asarray(res.weights, dtype=float)
    assert np.allclose(w, 1.0, atol=1e-6)
