"""Tests for competing-risks survival analysis (``sp.cuminc``, ``sp.finegray``).

Validation strategy (no external R/Stata dependency):

* CIF self-consistency: the cumulative incidence of every cause plus the
  overall Kaplan-Meier survival sum to 1 at the last event time.
* Aalen-Johansen asymptote: with light censoring, CIF_k(inf) approaches the
  cause-k hazard fraction h_k / sum(h).
* Analytic delta-method SE matches a nonparametric bootstrap SE.
* Fine-Gray recovers a positive subdistribution hazard ratio for a covariate
  that raises the cause-of-interest hazard.
* Gray's K-sample test rejects under a strong group difference and does not
  reject under the null.
"""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _two_cause_dgp(n=3000, h1=0.6, h2=0.4, cens=2.5, seed=0):
    """Two competing causes via cause-specific exponential hazards."""
    rng = np.random.default_rng(seed)
    t1 = rng.exponential(1.0 / h1, n)
    t2 = rng.exponential(1.0 / h2, n)
    t_event = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    c = rng.exponential(cens, n)
    time = np.minimum(t_event, c)
    event = np.where(t_event <= c, cause, 0)
    return pd.DataFrame({"time": time, "event": event})


def _overall_km_last(time, event):
    """Overall (all-cause) KM survival at the last event time."""
    times = np.sort(np.unique(time[event != 0]))
    surv = 1.0
    for t in times:
        n_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event != 0))
        surv *= 1.0 - d / n_risk
    return surv


def test_cif_plus_survival_sums_to_one():
    df = _two_cause_dgp(seed=0)
    ci = sp.cuminc(df, "time", "event")
    cif1 = ci.cif_table[ci.cif_table["cause"] == 1]["cif"].iloc[-1]
    cif2 = ci.cif_table[ci.cif_table["cause"] == 2]["cif"].iloc[-1]
    s = _overall_km_last(df["time"].to_numpy(), df["event"].to_numpy())
    assert cif1 + cif2 + s == pytest.approx(1.0, abs=1e-9)
    assert ci.causes == [1, 2]


def test_cif_asymptote_matches_hazard_fraction():
    # Light censoring so almost everyone fails -> CIF_k(inf) ~ h_k / (h1+h2).
    df = _two_cause_dgp(n=8000, h1=0.6, h2=0.4, cens=50.0, seed=1)
    ci = sp.cuminc(df, "time", "event")
    cif1_inf = ci.cif_table[ci.cif_table["cause"] == 1]["cif"].iloc[-1]
    assert cif1_inf == pytest.approx(0.6, abs=0.03)


def test_analytic_se_matches_bootstrap():
    df = _two_cause_dgp(n=600, seed=2)
    t0 = 1.0

    def cif1_se_at(d):
        c = sp.cuminc(d, "time", "event")
        sub = c.cif_table[(c.cif_table["cause"] == 1)
                          & (c.cif_table["time"] <= t0)]
        if not len(sub):
            return 0.0, 0.0
        return float(sub["cif"].iloc[-1]), float(sub["se"].iloc[-1])

    est, se_analytic = cif1_se_at(df)
    rng = np.random.default_rng(7)
    n = len(df)
    boots = []
    for _ in range(120):
        idx = rng.integers(0, n, n)
        e, _ = cif1_se_at(df.iloc[idx].reset_index(drop=True))
        boots.append(e)
    se_boot = float(np.std(boots))
    assert se_analytic == pytest.approx(se_boot, rel=0.30)


def test_finegray_recovers_positive_shr():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.binomial(1, 0.5, n).astype(float)
    h1 = 0.4 * np.exp(0.8 * x)        # x raises the cause-1 hazard
    t1 = rng.exponential(1.0 / h1)
    t2 = rng.exponential(1.0 / 0.4, n)
    te = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    c = rng.exponential(3.0, n)
    time = np.minimum(te, c)
    event = np.where(te <= c, cause, 0)
    df = pd.DataFrame({"time": time, "event": event, "x": x})

    fg = sp.finegray(df, "time", "event", x=["x"], cause=1)
    assert fg.shr[0] > 1.0                       # positive subdistribution effect
    assert fg.pvalues[0] < 0.01
    assert fg.conf_int[0, 0] < fg.params[0] < fg.conf_int[0, 1]
    assert fg.n_events > 0
    assert "Fine-Gray" in fg.summary()
    td = fg.tidy()
    assert list(td["term"]) == ["x"]


def _grouped_dgp(n, effect, seed):
    rng = np.random.default_rng(seed)
    g = rng.binomial(1, 0.5, n)
    h1 = 0.5 * np.exp(effect * g)
    t1 = rng.exponential(1.0 / h1)
    t2 = rng.exponential(1.0 / 0.5, n)
    te = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    c = rng.exponential(3.0, n)
    time = np.minimum(te, c)
    event = np.where(te <= c, cause, 0)
    return pd.DataFrame({"time": time, "event": event, "g": g})


def test_gray_test_rejects_under_strong_difference():
    df = _grouped_dgp(800, effect=0.9, seed=4)
    ci = sp.cuminc(df, "time", "event", group="g")
    assert ci.gray_test is not None
    assert ci.gray_test[1]["p_value"] < 0.01
    assert ci.gray_test[1]["df"] == 1


def test_gray_test_does_not_reject_under_null():
    df = _grouped_dgp(1500, effect=0.0, seed=5)
    ci = sp.cuminc(df, "time", "event", group="g")
    assert ci.gray_test[1]["p_value"] > 0.05


def test_cuminc_rejects_all_censored():
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [0, 0, 0]})
    with pytest.raises(ValueError, match="censored"):
        sp.cuminc(df, "time", "event")


def test_finegray_rejects_absent_cause():
    df = _two_cause_dgp(n=200, seed=6)
    df["x"] = 1.0
    with pytest.raises(ValueError, match="cause"):
        sp.finegray(df, "time", "event", x=["x"], cause=3)
