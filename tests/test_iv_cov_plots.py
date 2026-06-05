"""Coverage campaign — ``sp.iv`` plotting layer (``iv/plot.py``).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Drives every public plotting helper in
``statspai/iv/plot.py`` — first-stage fit, AR confidence set, MTE curve,
plausibly-exogenous β(γ) panel, the IV forest plot (raw table + from-diag),
the weak-IV CI overlay, and the four-panel diagnostic figure — on real fitted
results under the headless Agg backend.

Rendering smoke tests: assert each helper returns a Matplotlib Axes/Figure and
does not raise; pixel content is not pinned (the estimators' numerics are
covered by the parity suites).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def iv_df():
    rng = np.random.default_rng(21)
    n = 400
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = 0.8 * z1 + 0.6 * z2 + 0.3 * x + 0.6 * u + 0.5 * rng.standard_normal(n)
    y = 1.0 + 2.0 * d + 0.4 * x + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})


@pytest.fixture(scope="module")
def diag_result(iv_df):
    return sp.iv.iv_diag(
        iv_df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=50,
        random_state=0,
    )


def _close(out):
    fig = out[0] if isinstance(out, tuple) else getattr(out, "figure", out)
    assert isinstance(fig, (Figure, Axes)) or isinstance(out, (Figure, Axes))
    plt.close("all")


def test_plot_first_stage(iv_df):
    ax = sp.iv.plot.plot_first_stage(
        endog="d", instruments=["z1", "z2"], exog=["x"], data=iv_df
    )
    _close(ax)


def test_plot_ar_confidence_set(iv_df):
    ax = sp.iv.plot.plot_ar_confidence_set(
        y="y", endog="d", instruments=["z1", "z2"], exog=["x"], data=iv_df
    )
    _close(ax)


def test_plot_mte_curve(iv_df):
    rng = np.random.default_rng(5)
    n = 700
    z = rng.uniform(-2, 2, n)
    x = rng.standard_normal(n)
    v = rng.standard_normal(n)
    d = ((0.9 * z + 0.3 * x - 0.4 * v) > 0).astype(float)
    y = 1.0 + 1.2 * d + 0.4 * x + 0.5 * v + rng.standard_normal(n)
    res = sp.iv.mte(
        y=y, treatment=d, instruments=z.reshape(-1, 1), exog=x.reshape(-1, 1)
    )
    _close(sp.iv.plot.plot_mte_curve(res))


def test_plot_plausibly_exogenous(iv_df):
    res = sp.iv(
        method="plausibly_exog_uci",
        data=iv_df,
        y="y",
        endog="d",
        instruments=["z1"],
        gamma_grid=np.linspace(-0.4, 0.4, 9),
    )
    _close(sp.iv.plot.plot_plausibly_exogenous(res))


def test_plot_iv_forest_from_table(iv_df):
    table = sp.iv.iv_compare(
        formula="y ~ (d ~ z1 + z2) + x", data=iv_df, methods=("2sls", "liml")
    )
    _close(sp.iv.plot.plot_iv_forest(table, reference=2.0, sort_by="estimate"))


def test_plot_iv_forest_from_diag(diag_result):
    _close(sp.iv.plot.plot_iv_forest_from_diag(diag_result, title="forest"))


def test_plot_weak_iv_ci_overlay(diag_result):
    _close(sp.iv.plot.plot_weak_iv_ci_overlay(diag_result, title="overlay"))


def test_plot_iv_diagnostics_panel(diag_result):
    fig = sp.iv.plot.plot_iv_diagnostics(diag_result, suptitle="diag")
    assert isinstance(fig, (Figure, tuple))
    plt.close("all")
