"""Equivalence test for the vectorized ``cluster_bootstrap_draw``.

The draw was rewritten from a per-cluster boolean-scan + ``pd.concat`` loop to
a pre-grouped fancy-index build. With an identically-seeded RNG the resampled
rows and the collision-avoiding relabel suffixes must be byte-for-byte the same
as the old implementation (reproduced here as the reference).
"""
import numpy as np
import pandas as pd

from statspai.did._core import cluster_bootstrap_draw


def _reference_draw(df, *, cluster_col, rng, relabel_cols=None, sep="_b"):
    """Original per-cluster loop implementation, kept as the oracle."""
    if relabel_cols is None:
        relabel_cols = [cluster_col]
    clusters = df[cluster_col].unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    frames = []
    for j, c in enumerate(sampled):
        chunk = df[df[cluster_col] == c].copy()
        suffix = f"{sep}{j}"
        for col in relabel_cols:
            chunk[col] = chunk[col].astype(str) + suffix
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def _panel(n_clusters=40, periods=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        for t in range(periods):
            rows.append({"unit": c, "time": t,
                         "y": rng.normal(), "x": rng.normal()})
    return pd.DataFrame(rows)


def test_matches_reference_default_relabel():
    df = _panel()
    new = cluster_bootstrap_draw(
        df, cluster_col="unit", rng=np.random.default_rng(123))
    ref = _reference_draw(
        df, cluster_col="unit", rng=np.random.default_rng(123))
    pd.testing.assert_frame_equal(new, ref)


def test_matches_reference_multi_relabel_cols():
    df = _panel(seed=11)
    new = cluster_bootstrap_draw(
        df, cluster_col="unit", rng=np.random.default_rng(7),
        relabel_cols=["unit", "time"], sep="__d")
    ref = _reference_draw(
        df, cluster_col="unit", rng=np.random.default_rng(7),
        relabel_cols=["unit", "time"], sep="__d")
    pd.testing.assert_frame_equal(new, ref)


def test_row_count_preserved():
    df = _panel()
    out = cluster_bootstrap_draw(
        df, cluster_col="unit", rng=np.random.default_rng(1))
    assert len(out) == len(df)  # balanced panel: equal cluster sizes
