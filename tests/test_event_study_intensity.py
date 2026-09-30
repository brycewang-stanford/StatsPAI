"""``sp.event_study(intensity=, absorb=)``: exposure x event-time designs.

A common shock at ``T0`` hits units with different exposure; the regressors
are ``exposure x 1[t - T0 = k]``. On the 1.5-million-row replication of
Zheng, Huang and Zhu (2026, Figure 3: four absorbed FE groups, industry
clusters) the result equals the hand-built ``t2pre* / t2post*`` regression to
7e-9 in the coefficients and 6e-8 in the SEs, and ``sp.honest_did`` runs on
it directly. Here: the same identity on simulated data (exact up to the
absorber's tolerance) and recovery of a known dose-response path.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(3)
    n_u, T, T0 = 400, 10, 6
    unit = np.repeat(np.arange(n_u), T)
    t = np.tile(np.arange(1, T + 1), n_u)
    expo = rng.uniform(0, 1, n_u)
    expo[:60] = 0.0  # unexposed comparison units
    region = rng.integers(0, 8, n_u)[unit]
    effect = {0: -0.2, 1: -0.3, 2: -0.4, 3: -0.4, 4: -0.4}
    path = np.array([effect.get(k, 0.0) for k in t - T0])
    y = (
        path * expo[unit]
        + rng.normal(size=n_u)[unit]
        + rng.normal(size=(8, T + 1))[region, t]
        + rng.normal(scale=0.5, size=unit.size)
    )
    return (
        pd.DataFrame(
            {"unit": unit, "t": t, "expo": expo[unit], "region": region, "y": y}
        ),
        T0,
        effect,
    )


def test_equals_the_hand_built_hdfe_regression(panel):
    df, T0, _ = panel
    es = sp.event_study(
        df,
        y="y",
        treat_time=T0,
        time="t",
        unit="unit",
        intensity="expo",
        absorb="region^t",
        window=(-5, 4),
    )
    assert es.method == "Intensity event study (HDFE)"
    names = {}
    d = df.copy()
    for k in range(-5, 5):
        if k == -1:
            continue
        nm = f"e{k + 10}"
        d[nm] = d.expo * (d.t - T0 == k)
        names[k] = nm
    ref = sp.hdfe_ols(
        "y ~ " + " + ".join(names.values()) + " | unit + t + region^t",
        d,
        cluster="unit",
    )
    tab = es.model_info["event_study"].set_index("relative_time")
    for k, nm in names.items():
        assert tab.loc[k, "att"] == pytest.approx(
            float(ref.coef[nm]), rel=1e-6, abs=1e-9
        )
        assert tab.loc[k, "se"] == pytest.approx(float(ref.se[nm]), rel=1e-6)
    assert tab.loc[-1, "att"] == 0.0 and bool(tab.loc[-1, "is_reference"])
    assert es.n_obs == ref.n_obs


def test_recovers_the_dose_response_path_and_feeds_honest_did(panel):
    df, T0, effect = panel
    es = sp.event_study(
        df,
        y="y",
        treat_time=T0,
        time="t",
        unit="unit",
        intensity="expo",
        absorb=["region^t"],
        window=(-5, 4),
    )
    tab = es.model_info["event_study"].set_index("relative_time")
    for k, v in effect.items():
        assert abs(tab.loc[k, "att"] - v) < 4 * tab.loc[k, "se"]
    for k in (-5, -4, -3, -2):
        assert abs(tab.loc[k, "att"]) < 4 * tab.loc[k, "se"]
    hd = sp.honest_did(es, m_grid=[0.0], method="relative_magnitude", l_vec="average")
    assert bool(hd.loc[0, "rejects_zero"])


def test_scalar_treat_time_equals_a_constant_column(panel):
    df, T0, _ = panel
    kw = dict(y="y", time="t", unit="unit", intensity="expo", window=(-3, 3))
    a = sp.event_study(df, treat_time=T0, **kw)
    b = sp.event_study(df.assign(T=T0), treat_time="T", **kw)
    pd.testing.assert_frame_equal(
        a.model_info["event_study"], b.model_info["event_study"]
    )
    plain = sp.event_study(
        df.assign(first=np.where(df.expo > 0, T0, np.nan)),
        y="y",
        treat_time="first",
        time="t",
        unit="unit",
        window=(-3, 3),
    )
    assert plain.method == "OLS Event Study (TWFE)"  # historical path untouched
