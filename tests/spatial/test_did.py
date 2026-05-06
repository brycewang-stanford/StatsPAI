"""Spatial DiD workflow tests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from statspai.spatial.did import spatial_did
from statspai.spatial.weights import W


def _panel(seed: int = 0, n: int = 24, t_periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        unit_fe = rng.normal(scale=0.2)
        for t in range(t_periods):
            d = int(i < n // 3 and t >= 3)
            y = unit_fe + 0.15 * t + 1.0 * d + rng.normal(scale=0.25)
            rows.append({"i": i, "t": t, "d": d, "y": y, "x": rng.normal()})
    return pd.DataFrame(rows)


def _chain_w(n: int) -> np.ndarray:
    Wm = np.zeros((n, n))
    for i in range(n - 1):
        Wm[i, i + 1] = 1.0
        Wm[i + 1, i] = 1.0
    return Wm


def test_spatial_did_w_object_and_exports():
    df = _panel()
    n = df["i"].nunique()
    neighbors = {
        i: [j for j in (i - 1, i + 1) if 0 <= j < n]
        for i in range(n)
    }
    w = W(neighbors, id_order=list(range(n)))
    w.transform = "R"

    res = spatial_did(df, y="y", treat="d", unit="i", time="t", W=w)

    assert np.isfinite(res.direct_effect)
    assert np.isfinite(res.spillover_effect)
    assert {"direct", "spillover", "total"} <= set(res.tidy()["term"])
    assert res.glance().loc[0, "nobs"] == len(df)
    assert "direct" in res.to_csv()
    assert "spillover" in res.to_markdown()
    assert json.loads(res.to_json())["se_type"] == "cluster"


def test_spatial_did_conley_distance_matrix():
    df = _panel(seed=1)
    n = df["i"].nunique()
    Wm = _chain_w(n)
    dist = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)

    res = spatial_did(
        df,
        y="y",
        treat="d",
        unit="i",
        time="t",
        W=Wm,
        distance_matrix=dist,
        conley_cutoff=1.5,
        se_type="conley",
    )

    assert res.se_type == "conley"
    assert np.isfinite(res.se_direct)
    assert res.model_info["conley_cutoff"] == 1.5


def test_spatial_did_event_study_and_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _panel(seed=2)
    n = df["i"].nunique()
    res = spatial_did(
        df,
        y="y",
        treat="d",
        unit="i",
        time="t",
        W=_chain_w(n),
        event_study=True,
        event_window=(-2, 2),
    )

    es = res.detail["event_study"]
    assert set(es["effect"]) == {"direct", "spillover"}
    assert set(es["relative_time"]) == {-2, 0, 1, 2}
    assert "direct" in res.detail["pretrend_test"]

    fig1, ax1 = res.plot(kind="coef")
    fig2, ax2 = res.plot(kind="exposure")
    fig3, axes = res.plot(kind="event_study")
    assert ax1 is not None and ax2 is not None
    assert len(axes) == 2
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
