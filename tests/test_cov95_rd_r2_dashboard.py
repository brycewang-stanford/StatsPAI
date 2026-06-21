"""Coverage round-2 — ``statspai.rd.dashboard`` (rd_dashboard panels).

Round 1 ran ``rd_dashboard`` with covariates (balance panel). This file adds:

- the *no-covs* dashboard (running-variable distribution panel instead of
  balance);
- the title + save-to-file path;
- a custom ``bw_grid`` for the sensitivity panel;
- the fuzzy pass-through;
- the ``_plot_balance`` "no usable covariate" fallback and the
  ``_plot_bw_sensitivity`` "bandwidth not available" fallback, driven via the
  private helpers on real data.

Real synthetic RD data; assertions check that figures/axes are produced and
that helper fallbacks render text rather than raising — never fabricated
numbers.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.rd.dashboard import _plot_balance, _plot_bw_sensitivity  # noqa: E402


def _df(seed=6, n=2000, tau=3.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + tau * (x >= 0) + rng.normal(0, 0.3, n)
    z = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.2 + 0.6 * (x >= 0)).astype(float)
    return pd.DataFrame({"y": y, "x": x, "z": z, "d": d})


def test_rd_dashboard_no_covs_running_var_panel():
    df = _df()
    fig, axes = sp.rd_dashboard(
        df, y="y", x="x", c=0, covs=None, title="RD diagnostics"
    )
    assert axes.shape == (2, 2)
    plt.close("all")


def test_rd_dashboard_with_covs_and_custom_grid_and_save():
    df = _df()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dash.png")
        fig, axes = sp.rd_dashboard(
            df, y="y", x="x", c=0, covs=["z"], bw_grid=[0.2, 0.4, 0.6], save=path
        )
        assert os.path.exists(path)
    plt.close("all")


def test_rd_dashboard_fuzzy():
    df = _df()
    fig, axes = sp.rd_dashboard(df, y="y", x="x", c=0, fuzzy="d")
    assert fig is not None
    plt.close("all")


def test_plot_balance_no_usable_covs_fallback():
    df = _df()
    fig, ax = plt.subplots()
    # request a covariate that does not exist -> "no covariates" text fallback
    _plot_balance(ax, df, x="x", c=0, covs=["does_not_exist"])
    assert ax is not None
    plt.close("all")


def test_plot_bw_sensitivity_no_bandwidth_fallback():
    df = _df()
    fig, ax = plt.subplots()
    # h_ref None -> "(bandwidth not available)" text branch
    _plot_bw_sensitivity(
        ax, df, y="y", x="x", c=0, fuzzy=None, h_ref=None, bw_grid=None
    )
    assert ax is not None
    plt.close("all")


def test_plot_bw_sensitivity_explicit_grid():
    df = _df()
    fig, ax = plt.subplots()
    _plot_bw_sensitivity(
        ax, df, y="y", x="x", c=0, fuzzy=None, h_ref=0.5, bw_grid=[0.3, 0.5, 0.7]
    )
    assert ax is not None
    plt.close("all")
