"""Coverage campaign — branch coverage for ``statspai.panel.panel_plots``.

The dispatcher-level tests in ``test_cov95_panel_reg.py`` route through
``PanelResults.plot(type=...)`` but skip the optional-argument branches and
never reach ``plot_within_between`` (a standalone data-level helper). Here we
call the plotting functions directly on a real fitted FE model + real panel
data, exercising:

- ``plot_coef``      — explicit ``variables=`` subset and caller-supplied ``ax``
- ``plot_effects``   — ``kind='both'`` (hist+KDE) and both "no effects" guards
- ``plot_residuals`` — the 2x2 diagnostic grid on a real result
- ``plot_within_between`` — the full within/between variance decomposition,
  including the degenerate constant-column branch and a supplied ``ax``
- ``plot_compare``   — multi-method overlay with a caller-supplied ``ax``
- ``plot_hausman``   — the FE-vs-RE visual + Hausman annotation

These are rendering smoke tests under the headless ``Agg`` backend: they assert
each call returns a Matplotlib ``Figure``/``Axes`` and produces the expected
artists (e.g. the within/between bar groups), not pixel output. The numerical
content (coefficients, effects) is pinned by the estimator parity suites.
"""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.panel import panel_plots as pp  # noqa: E402


@pytest.fixture
def panel_df():
    rng = np.random.default_rng(11)
    n_id, T = 40, 6
    rows = []
    for i in range(n_id):
        alpha = rng.normal(4, 2)
        for t in range(T):
            x1 = rng.normal() + 0.4 * alpha
            x2 = rng.normal()
            y = alpha + 2.0 * x1 + 1.0 * x2 + rng.normal(0, 0.5)
            rows.append({"id": i, "year": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


@pytest.fixture
def fe_result(panel_df):
    return sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")


def teardown_function(_):
    plt.close("all")


# ── plot_coef: variables subset + supplied ax ───────────────────────────


def test_plot_coef_variables_and_supplied_ax(fe_result):
    fig0, ax0 = plt.subplots()
    fig, ax = pp.plot_coef(fe_result, variables=["x1"], ax=ax0, color="#E74C3C")
    assert fig is fig0 and ax is ax0  # reused the supplied axes
    # exactly one coefficient (x1) was plotted → one y-tick label
    assert [t.get_text() for t in ax.get_yticklabels()] == ["x1"]


# ── plot_effects: hist+KDE branch and both unavailable guards ───────────


def test_plot_effects_hist_and_kde(fe_result):
    fig, ax = pp.plot_effects(fe_result, kind="both", bins=15)
    assert isinstance(fig, Figure)
    # 'both' draws histogram bars AND a KDE line
    assert len(ax.patches) > 0
    assert len(ax.lines) >= 1


def test_plot_effects_supplied_ax(fe_result):
    fig0, ax0 = plt.subplots()
    fig, ax = pp.plot_effects(fe_result, ax=ax0, kind="hist")
    assert ax is ax0


def test_plot_effects_raises_without_lm_result(fe_result):
    fe_result._lm_result = None
    with pytest.raises(ValueError, match="Entity effects not available"):
        pp.plot_effects(fe_result)


def test_plot_effects_raises_without_estimated_effects(fe_result):
    # lm_result present but missing the ``estimated_effects`` attribute.
    fe_result._lm_result = types.SimpleNamespace(resids=np.zeros(3))
    with pytest.raises(ValueError, match="Entity effects not available"):
        pp.plot_effects(fe_result)


# ── plot_residuals: full 2x2 diagnostic grid ────────────────────────────


def test_plot_residuals_grid(fe_result):
    fig, axes = pp.plot_residuals(fe_result)
    assert isinstance(fig, Figure)
    assert axes.shape == (2, 2)


def test_plot_residuals_raises_without_residuals(fe_result):
    fe_result.data_info["residuals"] = None
    with pytest.raises(ValueError, match="Residuals/fitted"):
        pp.plot_residuals(fe_result)


# ── plot_within_between: full function incl degenerate column + ax ──────


def test_plot_within_between_full(panel_df):
    df = panel_df.assign(constcol=1.0)  # zero-variance column → degenerate branch
    fig0, ax0 = plt.subplots()
    fig, ax = pp.plot_within_between(
        df,
        variables=["x1", "x2", "constcol"],
        entity="id",
        ax=ax0,
    )
    assert ax is ax0
    # one Between bar-group + one Within bar-group across 3 variables
    assert len(ax.containers) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x1", "x2", "constcol"]


def test_plot_within_between_creates_own_axes(panel_df):
    fig, ax = pp.plot_within_between(panel_df, variables=["x1"], entity="id")
    assert isinstance(fig, Figure)


# ── plot_compare: multi-method overlay with supplied ax ─────────────────


def test_plot_compare_supplied_ax(panel_df):
    fe = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    re = sp.panel(panel_df, "y ~ x1 + x2", entity="id", time="year", method="re")
    fig0, ax0 = plt.subplots()
    fig, ax = pp.plot_compare({"FE": fe, "RE": re}, ax=ax0)
    assert ax is ax0
    # one errorbar container per method
    assert len(ax.containers) == 2


# ── plot_hausman: FE-vs-RE visual + annotation ──────────────────────────


def test_plot_hausman(fe_result):
    fig, ax = pp.plot_hausman(fe_result)
    assert isinstance(fig, Figure)
    # the Hausman annotation text should be present somewhere on the axes
    texts = [t.get_text() for t in ax.texts]
    assert any("Hausman" in t for t in texts)
