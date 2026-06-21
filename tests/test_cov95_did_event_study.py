"""Coverage tests for statspai.did.event_study (OLS TWFE event study)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(seed=0, n_units=80, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": fe + 0.3 * t + te + rng.normal(0, 0.4),
                    "g": g,
                    "x1": rng.normal(),
                    "cl": u % 12,
                    "w": rng.uniform(0.5, 2.0),
                }
            )
    df = pd.DataFrame(rows)
    df["ft"] = df["g"].replace(0, np.nan)
    return df


@pytest.fixture(scope="module")
def panel():
    return _panel()


def test_event_study_basic(panel):
    r = sp.event_study(
        panel, y="y", treat_time="ft", time="time", unit="unit", window=(-3, 3)
    )
    es = r.model_info["event_study"]
    # reference period present with zero estimate
    ref = es[es["relative_time"] == -1]
    assert float(ref["estimate"].iloc[0]) == 0.0
    # post-treatment effects positive and increasing roughly
    post = es[es["relative_time"] >= 1].sort_values("relative_time")
    assert post["estimate"].iloc[-1] > 0
    assert r.estimate > 0
    assert "pretrend_test" in r.model_info
    pt = r.model_info["pretrend_test"]
    assert 0.0 <= pt["pvalue"] <= 1.0


def test_event_study_covariates(panel):
    r = sp.event_study(
        panel,
        y="y",
        treat_time="ft",
        time="time",
        unit="unit",
        window=(-2, 2),
        covariates=["x1"],
    )
    assert "event_study" in r.model_info


def test_event_study_explicit_cluster(panel):
    r = sp.event_study(
        panel,
        y="y",
        treat_time="ft",
        time="time",
        unit="unit",
        window=(-2, 2),
        cluster="cl",
    )
    assert r.model_info["cluster_var"] == "cl"
    assert r.model_info["n_clusters"] >= 1


def test_event_study_weights(panel):
    r = sp.event_study(
        panel,
        y="y",
        treat_time="ft",
        time="time",
        unit="unit",
        window=(-2, 2),
        weights="w",
    )
    assert r.model_info["weights"] == "w"
    assert r.estimate is not None


def test_event_study_negative_weights_raise(panel):
    bad = panel.copy()
    bad["w"] = -1.0
    with pytest.raises(ValueError):
        sp.event_study(
            bad,
            y="y",
            treat_time="ft",
            time="time",
            unit="unit",
            window=(-2, 2),
            weights="w",
        )


def test_event_study_string_time():
    # non-numeric time column → time_map branch (lines 119-124)
    rng = np.random.default_rng(1)
    rows = []
    years = ["2010", "2011", "2012", "2013", "2014", "2015"]
    for u in range(60):
        treated = u % 2 == 0
        fe = rng.normal()
        for i, yr in enumerate(years):
            t = i + 1
            g = 4 if treated else 0
            te = 1.0 * (t - g + 1) if (treated and t >= g) else 0.0
            rows.append(
                {
                    "unit": u,
                    "yr": yr,
                    "y": fe + 0.3 * t + te + rng.normal(0, 0.4),
                    "ft": ("2013" if treated else None),
                }
            )
    df = pd.DataFrame(rows)
    r = sp.event_study(
        df, y="y", treat_time="ft", time="yr", unit="unit", window=(-2, 2)
    )
    assert "event_study" in r.model_info


def test_event_study_window_binning(panel):
    # narrow window forces endpoint binning of far lags/leads
    r = sp.event_study(
        panel, y="y", treat_time="ft", time="time", unit="unit", window=(-1, 1)
    )
    es = r.model_info["event_study"]
    assert set(es["relative_time"]) <= {-1, 0, 1}
