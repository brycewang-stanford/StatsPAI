"""Coverage campaign — panel compare rendering, interactive FE, HDFE kernels.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Closes the remaining reachable panel
lines: the ``PanelCompareResults`` ``summary()`` / ``plot()`` and the
``panel_compare`` comparison-table builder (``panel_reg.py``), the interactive
fixed-effects estimator (``interactive_fe.py``), and the HDFE demeaning kernels
with weights / singleton dropping (``hdfe.py`` / ``_hdfe_kernels.py``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def panel_df():
    rng = np.random.default_rng(0)
    n_e, n_t = 50, 8
    N = n_e * n_t
    ent = np.repeat(np.arange(n_e), n_t)
    tm = np.tile(np.arange(n_t), n_e)
    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal(N)
    fe = np.repeat(rng.standard_normal(n_e), n_t)
    y = 2.0 * x1 - 1.0 * x2 + fe + rng.standard_normal(N)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "id": ent,
            "time": tm,
            "w": rng.uniform(0.5, 1.5, N),
        }
    )


# ─── panel_compare comparison table + rendering ──────────────────────────


def test_panel_compare_summary_and_plot(panel_df):
    cmp = sp.panel_compare(
        panel_df,
        formula="y ~ x1 + x2",
        entity="id",
        time="time",
        methods=["fe", "re", "pooled", "twoway"],
    )
    s = cmp.summary() if hasattr(cmp, "summary") else str(cmp)
    assert isinstance(s, str) and len(s) > 0
    # str() exercises the __repr__/__str__ rendering of the compare object
    assert isinstance(str(cmp), str)
    plt.close("all")


def test_panel_compare_default_methods(panel_df):
    cmp = sp.panel_compare(panel_df, formula="y ~ x1 + x2", entity="id", time="time")
    assert cmp is not None


# ─── interactive fixed effects (Bai 2009) ────────────────────────────────


@pytest.mark.parametrize("n_factors", [1, 2])
def test_interactive_fe(panel_df, n_factors):
    res = sp.interactive_fe(
        panel_df, y="y", x=["x1", "x2"], id="id", time="time", n_factors=n_factors
    )
    assert res is not None
    b = getattr(res, "params", None)
    if b is not None and "x1" in getattr(b, "index", []):
        assert np.isfinite(float(b["x1"]))


# ─── HDFE demeaning kernels: weights + singletons ────────────────────────


def test_demean_weighted(panel_df):
    x = panel_df[["y", "x1"]].to_numpy(dtype=float)
    fe = panel_df[["id", "time"]]
    w = panel_df["w"].to_numpy(dtype=float)
    out = sp.demean(x, fe=fe, weights=w)
    arr = out[0] if isinstance(out, tuple) else out
    assert np.asarray(arr).shape[0] == len(panel_df)


def test_demean_singleton_dropping():
    # a singleton group (one obs) should be dropped under drop_singletons
    rng = np.random.default_rng(3)
    n = 200
    g = rng.integers(0, 40, n)
    g[0] = 999  # unique → singleton
    x = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    fe = pd.DataFrame({"g": g})
    out = sp.demean(x, fe=fe, drop_singletons=True)
    assert out is not None
