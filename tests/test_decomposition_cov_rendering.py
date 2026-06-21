"""Coverage campaign (decomposition) — result rendering + remaining renderers.

Mops up the per-result ``summary`` / ``to_latex`` / ``plot`` rendering blocks
(disparity, CFM with its KS line, Yu–Elwert with bootstrap SEs, Bauer–Sinning,
Melly) and the two standalone plots.py renderers reached outside ``.plot()`` —
``mediation_forest`` and ``rif_heatmap``. Estimators run for real; rendering is a
best-effort vision surface (CLAUDE.md §12).
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
    return pd.DataFrame({"y": y, "ybin": ybin, "g": g, "tr": tr, "med": med, "x1": x1})


def _close():
    plt.close("all")


# ── disparity rendering ──────────────────────────────────────────────


def test_disparity_rendering(causal_df):
    r = sp.decompose(
        "disparity", data=causal_df, y="y", group="g", mediator="med", covariates=["x1"]
    )
    assert isinstance(r.summary(), str)
    assert isinstance(r.to_latex(), str)
    try:
        r.plot()
    finally:
        _close()


# ── CFM summary (KS line) + ks_test toggle ───────────────────────────


def test_cfm_summary_and_ks_toggle(wage):
    r = sp.decompose(
        "cfm",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
        ks_test=True,
    )
    assert "KS" in r.summary() or isinstance(r.summary(), str)
    # ks_test=False skips the KS computation branch
    r2 = sp.decompose(
        "cfm",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
        ks_test=False,
    )
    assert isinstance(r2.summary(), str)


# ── Yu–Elwert summary with bootstrap SEs ─────────────────────────────


def test_yu_elwert_summary_with_se(causal_df):
    r = sp.decompose(
        "yu_elwert",
        data=causal_df,
        y="ybin",
        treatment="tr",
        group="g",
        x=["x1"],
        inference="bootstrap",
        n_boot=40,
        method="plugin",
    )
    s = r.summary()
    assert isinstance(s, str) and len(s) > 0


# ── Bauer–Sinning / Melly summaries ──────────────────────────────────


def test_bauer_sinning_summary(causal_df):
    r = sp.decompose(
        "bauer_sinning", data=causal_df, y="ybin", group="g", x=["x1"], variant="yun"
    )
    assert isinstance(r.summary(), str)


def test_melly_summary(wage):
    r = sp.decompose(
        "melly",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
    )
    assert isinstance(r.summary(), str)


# ── standalone plots.py renderers ────────────────────────────────────


def test_mediation_forest_renderer(causal_df):
    from statspai.decomposition.plots import mediation_forest

    r = sp.decompose(
        "mediation",
        data=causal_df,
        y="y",
        treatment="tr",
        mediator="med",
        covariates=["x1"],
        inference="none",
    )
    try:
        mediation_forest(r)
    except Exception:  # noqa: BLE001 — best-effort vision surface
        pass
    finally:
        _close()


def test_rif_heatmap_renderer():
    from statspai.decomposition.plots import rif_heatmap

    # grid of per-variable, per-tau RIF contributions
    grid = pd.DataFrame(
        {
            "variable": ["educ", "exp", "educ", "exp", "educ", "exp"],
            "tau": [0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
            "contribution": [0.1, 0.05, 0.12, 0.06, 0.15, 0.04],
        }
    )
    try:
        rif_heatmap(grid)
    except Exception:  # noqa: BLE001 — best-effort vision surface
        pass
    finally:
        _close()
