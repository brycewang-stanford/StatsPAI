"""Round-4 coverage margin: synth.cluster (cluster_synth).

Drives the cluster-SCM estimator across its clustering backends
(kmeans / spectral / hierarchical), with augmentation on/off and
placebo inference, plus the input-validation guard branches.

Real multi-cluster donor panels (sklearn clustering). No mocking.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.synth import cluster as _cluster

cluster_synth = _cluster.cluster_synth


def _clustered_panel(n_per_cluster=6, n_pre=15, n_post=8, seed=0):
    """Panel with two well-separated donor clusters + a treated unit."""
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    years = np.arange(2000, 2000 + T)
    rows = []
    # Cluster A: level ~10, Cluster B: level ~50
    levels = [10.0] * n_per_cluster + [50.0] * n_per_cluster
    donor_series = []
    for j, lvl in enumerate(levels):
        s = lvl + np.cumsum(rng.normal(0, 0.5, T))
        donor_series.append(s)
        for i, yr in enumerate(years):
            rows.append((f"d{j}", yr, s[i]))
    # Treated tracks cluster A donors in the pre-period, then jumps.
    treated = 0.5 * donor_series[0] + 0.5 * donor_series[1] + rng.normal(0, 0.2, T)
    treated[n_pre:] += 7.0
    for i, yr in enumerate(years):
        rows.append(("T", yr, treated[i]))
    df = pd.DataFrame(rows, columns=["unit", "year", "y"])
    return df, years[n_pre]


@pytest.mark.parametrize("method", ["kmeans", "spectral", "hierarchical"])
def test_cluster_synth_methods(method):
    df, tt = _clustered_panel(seed=1)
    res = cluster_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        cluster_method=method,
        n_clusters=2,
        placebo=True,
        seed=2,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0  # genuine positive effect


def test_cluster_synth_auto_n_clusters():
    # n_clusters=None -> silhouette-based selection path.
    df, tt = _clustered_panel(seed=3)
    res = cluster_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        cluster_method="kmeans",
        n_clusters=None,
        placebo=False,
        seed=4,
    )
    assert np.isfinite(res.estimate)
    assert res.model_info["n_clusters"] >= 2


def test_cluster_synth_augment():
    # augment=True pulls in nearest out-of-cluster donors.
    df, tt = _clustered_panel(seed=5)
    res = cluster_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        cluster_method="kmeans",
        n_clusters=3,
        augment=True,
        max_augment=2,
        placebo=True,
        seed=6,
    )
    assert np.isfinite(res.estimate)


def test_cluster_synth_with_covariates():
    # Covariate columns -> _build_features covariate-appending path.
    df, tt = _clustered_panel(seed=12)
    rng = np.random.default_rng(12)
    df = df.copy()
    df["pop"] = rng.normal(100, 5, len(df))
    df["gdp"] = rng.normal(50, 3, len(df))
    res = cluster_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        cluster_method="kmeans",
        n_clusters=2,
        covariates=["pop", "gdp"],
        placebo=False,
        seed=13,
    )
    assert np.isfinite(res.estimate)


def test_cluster_synth_too_few_pre_periods():
    df, tt = _clustered_panel(n_pre=1, n_post=5, seed=7)
    with pytest.raises(ValueError, match="2 pre-treatment"):
        cluster_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=tt,
        )


def test_cluster_synth_too_few_donors():
    # Only 2 donors -> below the 3-donor clustering minimum.
    rng = np.random.default_rng(8)
    years = np.arange(2000, 2020)
    rows = []
    for u in ["d0", "d1", "T"]:
        s = np.cumsum(rng.normal(0, 1, len(years))) + 20
        for i, yr in enumerate(years):
            rows.append((u, yr, s[i]))
    df = pd.DataFrame(rows, columns=["unit", "year", "y"])
    with pytest.raises(ValueError, match="3 donor"):
        cluster_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=2010,
        )


def test_cluster_synth_treatment_time_not_in_data():
    df, tt = _clustered_panel(seed=9)
    last = df["year"].max()
    with pytest.raises(ValueError):
        cluster_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=last + 100,
        )


def test_cluster_synth_missing_column():
    df, tt = _clustered_panel(seed=10)
    with pytest.raises(ValueError, match="not found"):
        cluster_synth(
            data=df,
            outcome="nope",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=tt,
        )


def test_cluster_synth_treated_unit_missing():
    df, tt = _clustered_panel(seed=11)
    with pytest.raises(ValueError, match="not in data"):
        cluster_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="ZZZ",
            treatment_time=tt,
        )


def test_augment_donors_empty_out_of_cluster():
    # All donors in the treated cluster -> nothing to augment with.
    X_donors = np.zeros((4, 3))
    X_treated = np.zeros(3)
    labels = np.zeros(4, dtype=int)  # all in cluster 0
    out = _cluster._augment_donors(X_donors, X_treated, labels, 0, max_augment=2)
    assert out == []
