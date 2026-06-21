"""Spatial DiD workflow tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
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


def _ring_w(n: int) -> np.ndarray:
    Wm = np.zeros((n, n))
    for i in range(n):
        Wm[i, (i - 1) % n] = 1.0
        Wm[i, (i + 1) % n] = 1.0
    return Wm / Wm.sum(axis=1, keepdims=True)


def test_spatial_did_w_object_and_exports():
    df = _panel()
    n = df["i"].nunique()
    neighbors = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}
    w = W(neighbors, id_order=list(range(n)))
    w.transform = "R"

    res = spatial_did(df, y="y", treat="d", unit="i", time="t", W=w, covariates="x")

    assert np.isfinite(res.direct_effect)
    assert np.isfinite(res.spillover_effect)
    assert {"direct", "spillover", "total"} <= set(res.tidy()["term"])
    assert res.glance().loc[0, "nobs"] == len(df)
    assert "direct" in res.to_csv()
    assert "spillover" in res.to_markdown()
    assert json.loads(res.to_json())["se_type"] == "cluster"
    assert res.model_info["covariates"] == ["x"]


def test_spatial_did_recovers_noiseless_direct_and_spillover_effects():
    n, t_periods = 8, 5
    Wm = _ring_w(n)
    unit_fe = np.linspace(-0.2, 0.3, n)
    time_fe = np.linspace(0.0, 0.4, t_periods)
    rows = []
    for t in range(t_periods):
        d = np.array(
            [
                int((i in [0, 2, 5] and t >= 2) or (i in [3, 6] and t >= 3))
                for i in range(n)
            ],
            dtype=float,
        )
        wd = Wm @ d
        for i in range(n):
            y = unit_fe[i] + time_fe[t] + 1.25 * d[i] + 0.75 * wd[i]
            rows.append({"i": i, "t": t, "d": d[i], "y": y})
    df = pd.DataFrame(rows)

    res = spatial_did(
        df,
        y="y",
        treat="d",
        unit="i",
        time="t",
        W=Wm,
        se_type="robust",
    )

    np.testing.assert_allclose(res.direct_effect, 1.25, atol=1e-12)
    np.testing.assert_allclose(res.spillover_effect, 0.75, atol=1e-12)
    np.testing.assert_allclose(res.total_effect, 2.0, atol=1e-12)


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


def test_spatial_did_rejects_invalid_weights_and_unit_order():
    df = _panel()
    n = df["i"].nunique()

    with pytest.raises(MethodIncompatibility, match="square spatial weights"):
        spatial_did(df, y="y", treat="d", unit="i", time="t", W=np.ones((n, n + 1)))

    with pytest.raises(MethodIncompatibility, match="unit order"):
        spatial_did(
            df,
            y="y",
            treat="d",
            unit="i",
            time="t",
            W=_chain_w(n),
            unit_order=list(range(n - 1)),
        )


def test_spatial_did_rejects_duplicate_cells_and_missing_columns():
    df = _panel()
    n = df["i"].nunique()

    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="one row per unit-period"):
        spatial_did(duplicated, y="y", treat="d", unit="i", time="t", W=_chain_w(n))

    with pytest.raises(MethodIncompatibility, match="Columns not found"):
        spatial_did(df, y="missing", treat="d", unit="i", time="t", W=_chain_w(n))


def test_spatial_did_conley_input_validation():
    df = _panel()
    n = df["i"].nunique()

    with pytest.raises(MethodIncompatibility, match="requires conley_cutoff"):
        spatial_did(
            df, y="y", treat="d", unit="i", time="t", W=_chain_w(n), se_type="conley"
        )

    with pytest.raises(MethodIncompatibility, match="distance_matrix"):
        spatial_did(
            df,
            y="y",
            treat="d",
            unit="i",
            time="t",
            W=_chain_w(n),
            se_type="conley",
            conley_cutoff=1.0,
            distance_matrix=np.ones((n - 1, n - 1)),
        )


def test_spatial_did_result_taxonomy_for_exports_and_plots():
    import matplotlib

    matplotlib.use("Agg")
    df = _panel()
    n = df["i"].nunique()
    res = spatial_did(df, y="y", treat="d", unit="i", time="t", W=_chain_w(n))

    with pytest.raises(MethodIncompatibility, match="detail must"):
        res.to_dict(detail="verbose")

    with pytest.raises(MethodIncompatibility, match="event-study"):
        res.plot(kind="event_study")

    with pytest.raises(MethodIncompatibility, match="kind must"):
        res.plot(kind="unknown")


def test_spatial_did_rejects_empty_complete_panel():
    df = _panel()
    n = df["i"].nunique()
    df["y"] = np.nan

    with pytest.raises(DataInsufficient, match="No complete observations"):
        spatial_did(df, y="y", treat="d", unit="i", time="t", W=_chain_w(n))
