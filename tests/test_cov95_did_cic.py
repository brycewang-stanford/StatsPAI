"""Coverage tests for statspai.did.cic (Changes-in-Changes, Athey & Imbens 2006)."""

import numpy as np
import pandas as pd
import pytest

import importlib

import statspai as sp

cicmod = importlib.import_module("statspai.did.cic")


def _cic_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    rows = []
    for g in (0, 1):
        for t in (0, 1):
            base = rng.normal(0, 1, n)
            # treated-post shifted up by ~2
            shift = 2.0 if (g == 1 and t == 1) else 0.0
            shift += 0.5 * t + 0.3 * g
            for v in base + shift:
                rows.append({"y": v, "group": g, "time": t})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cic_df():
    return _cic_data()


def test_cic_att_only(cic_df):
    r = sp.cic(cic_df, y="y", group="group", time="time", n_boot=60, seed=1)
    assert np.isfinite(r.estimate)
    assert r.se >= 0
    assert 0.0 <= r.pvalue <= 1.0
    assert r.ci[0] <= r.ci[1]
    # ATT should be positive (treated-post shifted up)
    assert r.estimate > 0.5
    assert r.model_info["n_boot"] == 60
    assert "qte" not in r.model_info


def test_cic_with_quantiles(cic_df):
    r = sp.cic(
        cic_df,
        y="y",
        group="group",
        time="time",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=60,
        seed=2,
    )
    assert "qte" in r.model_info
    det = r.model_info["qte"]
    assert len(det) == 3
    assert set(["quantile", "qte", "se", "ci_lower", "ci_upper", "pvalue"]).issubset(
        det.columns
    )
    assert np.all(det["ci_lower"] <= det["ci_upper"])


def test_cic_summary_att(cic_df, capsys):
    r = sp.cic(cic_df, y="y", group="group", time="time", n_boot=40, seed=3)
    out = r.summary()
    assert "Changes-in-Changes" in out
    assert "ATT" in out


def test_cic_summary_with_qte(cic_df):
    r = sp.cic(
        cic_df, y="y", group="group", time="time", quantiles=[0.5], n_boot=40, seed=4
    )
    out = r.summary()
    assert "Quantile Treatment Effects" in out


def test_cic_plot_att(cic_df):
    import matplotlib

    matplotlib.use("Agg")
    r = sp.cic(cic_df, y="y", group="group", time="time", n_boot=30, seed=5)
    fig, ax = r.plot()
    assert ax is not None


def test_cic_plot_qte(cic_df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = sp.cic(
        cic_df,
        y="y",
        group="group",
        time="time",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=30,
        seed=6,
    )
    fig, ax = plt.subplots()
    fig2, ax2 = r.plot(ax=ax)
    assert ax2 is not None


def test_cic_too_few_obs_raises():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "group": [0, 1, 1],
            "time": [0, 0, 1],
        }
    )  # control-post cell empty
    with pytest.raises(ValueError):
        sp.cic(df, y="y", group="group", time="time", n_boot=10)


def test_cic_helper_quantile_func():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    q = cicmod._quantile_func(x, np.array([0.25, 0.5, 0.75]))
    assert q[0] <= q[1] <= q[2]


def test_cic_helper_ecdf():
    x = np.array([1.0, 2.0, 3.0])
    vals = cicmod._ecdf(x, np.array([0.0, 2.0, 5.0]))
    assert vals[0] == 0.0
    assert vals[-1] == 1.0
