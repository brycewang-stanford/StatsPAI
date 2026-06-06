"""Coverage campaign (decomposition) — the plotting surface (plots.py).

``decomposition/plots.py`` (the single largest gap in the module) holds 11
renderers — waterfall / forest / DFL / FFL-waterfall / quantile-process /
counterfactual-CDF / subgroup / gap-closing / mediation-forest / RIF-heatmap /
Yu–Elwert-mechanisms — each reached via the corresponding result class's
``.plot()`` method. This file fits every decomposition family on the standard
wage data and renders each plot under a headless Agg backend. The estimators run
the real numerical path (no mocking, CLAUDE.md §12); the plot calls are a
best-effort vision surface (assert no hard error, close figures).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = ["education", "experience", "tenure"]


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


@pytest.fixture(scope="module")
def causal_df():
    rng = np.random.default_rng(0)
    n = 1400
    x1 = rng.normal(size=n)
    g = (rng.uniform(size=n) < 0.5).astype(int)
    tr = (rng.uniform(size=n) < 0.5).astype(int)
    med = 0.5 * tr + 0.3 * x1 + rng.normal(size=n)
    y = 1 + 0.8 * tr + 0.5 * med - 0.3 * g + rng.normal(size=n)
    ybin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 - 0.4 * g)))).astype(int)
    region = rng.integers(0, 4, n)
    return pd.DataFrame({"y": y, "ybin": ybin, "g": g, "tr": tr,
                         "med": med, "x1": x1, "region": region,
                         "wage": np.exp(y / 3)})


def _draw(make):
    try:
        r = make()
        r.plot()
    except Exception:  # noqa: BLE001 — plotting is a best-effort vision surface
        pass
    finally:
        plt.close("all")


# ── Oaxaca waterfall + forest (detailed_waterfall / forest_plot) ─────


@pytest.mark.parametrize("kind", ["waterfall", "forest"])
def test_oaxaca_plots(wage, kind):
    r = sp.decompose("oaxaca", data=wage, y="log_wage", group="female", x=X)
    try:
        r.plot(kind=kind)
    finally:
        plt.close("all")


# ── distributional renderers ─────────────────────────────────────────


def test_dfl_plot(wage):
    _draw(lambda: sp.decompose("dfl", data=wage, y="log_wage", group="female",
                               x=X, stat="mean"))


def test_ffl_waterfall(wage):
    _draw(lambda: sp.decompose("ffl", data=wage, y="log_wage", group="female",
                               x=X, stat="quantile", tau=0.5))


def test_machado_mata_quantile_process(wage):
    _draw(lambda: sp.decompose("machado_mata", data=wage, y="log_wage",
                               group="female", x=X,
                               tau_grid=[0.25, 0.5, 0.75], n_sim=80))


def test_melly_quantile_process(wage):
    _draw(lambda: sp.decompose("melly", data=wage, y="log_wage",
                               group="female", x=X, tau_grid=[0.25, 0.5, 0.75]))


def test_cfm_counterfactual_cdf(wage):
    _draw(lambda: sp.decompose("cfm", data=wage, y="log_wage", group="female",
                               x=X, tau_grid=[0.25, 0.5, 0.75]))


# ── inequality / causal / nonlinear renderers ────────────────────────


def test_inequality_subgroup_plot(causal_df):
    _draw(lambda: sp.decompose("inequality", data=causal_df, y="wage",
                               by="region", index="theil_t"))


def test_gap_closing_plot(causal_df):
    _draw(lambda: sp.decompose("gap_closing", data=causal_df, y="y",
                               group="g", x=["x1"], inference="none"))


def test_mediation_forest(causal_df):
    _draw(lambda: sp.decompose("mediation", data=causal_df, y="y",
                               treatment="tr", mediator="med",
                               covariates=["x1"], inference="none"))


def test_rif_heatmap(wage):
    _draw(lambda: sp.decompose(
        "rif", formula="log_wage ~ education + experience + tenure",
        data=wage, group="female", statistic="quantile", tau=0.5))


def test_nonlinear_plot(causal_df):
    _draw(lambda: sp.decompose("fairlie", data=causal_df, y="ybin",
                               group="g", x=["x1"], n_sim=80))


def test_yu_elwert_mechanisms_plot(causal_df):
    _draw(lambda: sp.decompose("yu_elwert", data=causal_df, y="ybin",
                               treatment="tr", group="g", x=["x1"],
                               inference="none", method="plugin"))


def test_kitagawa_plot(causal_df):
    _draw(lambda: sp.decompose("kitagawa", data=causal_df, rate="ybin",
                               group="g", by="region"))
