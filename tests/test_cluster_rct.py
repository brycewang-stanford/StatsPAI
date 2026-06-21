"""Smoke tests for v0.10 Cluster RCT × Interference suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def matched_pair_data():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(20):
        # Each pair: 2 clusters, 1 treated
        for cl_idx, treated in enumerate([0, 1]):
            cluster_id = f"c{p * 2 + cl_idx}"
            for i in range(15):
                y = 0.3 * p + 1.0 * treated + rng.standard_normal()
                rows.append(
                    {
                        "y": y,
                        "cluster": cluster_id,
                        "treat": treated,
                        "pair": f"p{p}",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def cross_cluster_data():
    rng = np.random.default_rng(1)
    rows = []
    n_clusters = 30
    for c in range(n_clusters):
        treated = int(rng.uniform() < 0.5)
        nshare = float(rng.uniform())  # share of treated neighbours
        for i in range(20):
            y = 0.5 * treated + 0.3 * nshare + rng.standard_normal()
            rows.append(
                {
                    "y": y,
                    "cluster": f"c{c}",
                    "treat": treated,
                    "nshare": nshare,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_cluster_data():
    rng = np.random.default_rng(2)
    rows = []
    for c in range(15):
        first_t = 0 if c < 5 else (3 if c < 10 else 5)
        for t in range(8):
            post = first_t > 0 and t >= first_t
            y = 0.1 * c + 0.2 * t + (1.0 if post else 0.0) + rng.standard_normal()
            rows.append(
                {
                    "y": y,
                    "cluster": f"c{c}",
                    "time": t,
                    "first_treat": first_t,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_matched_pair(matched_pair_data):
    res = sp.cluster_matched_pair(
        matched_pair_data,
        y="y",
        cluster="cluster",
        treat="treat",
        pair="pair",
    )
    cl = (
        matched_pair_data.groupby(["pair", "cluster", "treat"])["y"]
        .mean()
        .reset_index()
    )
    pair_diffs = []
    for _, sub in cl.groupby("pair"):
        yt = sub.loc[sub["treat"] == 1, "y"].iloc[0]
        yc = sub.loc[sub["treat"] == 0, "y"].iloc[0]
        pair_diffs.append(yt - yc)
    pair_diffs = np.asarray(pair_diffs)
    np.testing.assert_allclose(res.estimate, pair_diffs.mean())
    np.testing.assert_allclose(
        res.se, pair_diffs.std(ddof=1) / np.sqrt(len(pair_diffs))
    )
    assert isinstance(res, sp.MatchedPairResult)
    # True effect ≈ 1.0
    assert 0.0 < res.estimate < 2.0


def test_cluster_cross_interference(cross_cluster_data):
    res = sp.cluster_cross_interference(
        cross_cluster_data,
        y="y",
        cluster="cluster",
        treat="treat",
        neighbour_treat_share="nshare",
    )
    assert isinstance(res, sp.CrossClusterRCTResult)
    # True direct ≈ 0.5
    assert -0.5 < res.direct_effect < 1.5
    np.testing.assert_allclose(
        [
            res.direct_effect,
            res.direct_se,
            res.spillover_effect,
            res.spillover_se,
            res.n_clusters,
        ],
        [
            0.667983323682481,
            0.08876320648575868,
            0.1988955917957267,
            0.13478308532337718,
            30,
        ],
        atol=1e-12,
    )


def test_cluster_staggered_rollout(staggered_cluster_data):
    res = sp.cluster_staggered_rollout(
        staggered_cluster_data,
        y="y",
        cluster="cluster",
        time="time",
        first_treat="first_treat",
        leads=2,
        lags=2,
    )
    assert isinstance(res, sp.StaggeredClusterRCTResult)
    np.testing.assert_allclose(
        res.overall_att,
        res.event_study.loc[res.event_study["rel_time"] >= 0, "att"].mean(),
    )
    # True post-ATT ≈ 1.0
    assert -0.5 < res.overall_att < 2.5


def test_dnc_gnn_did(staggered_cluster_data):
    df = staggered_cluster_data.copy()
    rng = np.random.default_rng(3)
    df["nc_y1"] = df["y"] * 0.3 + rng.standard_normal(len(df))
    df["nc_x1"] = rng.standard_normal(len(df))
    res = sp.dnc_gnn_did(
        df,
        y="y",
        treat="first_treat",
        time="time",
        id="cluster",
        nc_outcome=["nc_y1"],
        nc_exposure=["nc_x1"],
        n_boot=20,
    )
    treated_mask = df["first_treat"] > 0
    ref_time = float(df.loc[treated_mask, "first_treat"].median())
    manual = df.copy()
    manual["_post"] = np.where(
        treated_mask,
        (manual["time"] >= manual["first_treat"]).astype(int),
        (manual["time"] >= ref_time).astype(int),
    )
    manual["_treat"] = treated_mask.astype(int)
    unit_means = (
        manual.groupby(["cluster", "_treat", "_post"])["y"].mean().reset_index()
    )
    pivot = unit_means.pivot_table(
        index="cluster", columns="_post", values="y"
    ).dropna()
    pivot.columns = ["y_pre", "y_post"]
    pivot["delta"] = pivot["y_post"] - pivot["y_pre"]
    pivot["_treat_unit"] = (manual.groupby("cluster")["_treat"].max() > 0).astype(int)
    pivot = pivot.join(manual.groupby("cluster")[["nc_y1", "nc_x1"]].mean()).dropna()
    X = np.column_stack(
        [
            np.ones(len(pivot)),
            pivot["_treat_unit"].to_numpy(float),
            pivot[["nc_y1", "nc_x1"]].to_numpy(float),
        ]
    )
    beta = np.linalg.solve(X.T @ X, X.T @ pivot["delta"].to_numpy(float))
    np.testing.assert_allclose(res.estimate, beta[1])
    assert isinstance(res, sp.DNCGNNDiDResult)
    assert np.isfinite(res.estimate)


def test_matched_pair_too_few_pairs():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0],
            "cluster": ["c1", "c2"],
            "treat": [0, 1],
            "pair": ["p1", "p1"],
        }
    )
    with pytest.raises(ValueError, match="at least 2"):
        sp.cluster_matched_pair(
            df, y="y", cluster="cluster", treat="treat", pair="pair"
        )
