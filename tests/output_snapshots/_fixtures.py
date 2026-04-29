"""Fixture builders for output-renderer snapshot tests.

Five canonical fixtures cover the dimensions that ``regtable`` rendering
exercises in practice. All use a fixed RNG seed so the generated tables
are byte-stable across runs and CI machines.

Usage::

    from tests.output_snapshots._fixtures import build_fixtures
    fixtures = build_fixtures()
    for name, result in fixtures.items():
        ...

Each fixture returns the ``RegtableResult`` (or comparable result object)
ready to be rendered.  Per-format snapshot files live alongside this
module as ``<name>.<ext>``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def _make_dataset(seed: int = 42, n: int = 400) -> pd.DataFrame:
    """Stable dataset used by all fixtures.

    Three covariates (``x1``, ``x2``, ``x3``), two binary group keys
    (``g1`` for FE, ``g2`` for cluster), and two outcomes (``y_lin``
    linear-DGP, ``y_bin`` binary).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    g1 = rng.integers(0, 5, size=n)
    g2 = rng.integers(0, 8, size=n)
    eps = rng.standard_normal(n)
    y_lin = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.3 * x3 + eps
    y_bin = (y_lin > y_lin.mean()).astype(int)
    return pd.DataFrame({
        "y_lin": y_lin,
        "y_bin": y_bin,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "g1": g1,
        "g2": g2,
    })


def fixture_simple_ols() -> "sp.output.RegtableResult":
    """F1 — single OLS regression. Smallest possible regression table."""
    df = _make_dataset()
    m1 = sp.regress("y_lin ~ x1 + x2", data=df)
    return sp.regtable([m1])


def fixture_multi_model() -> "sp.output.RegtableResult":
    """F2 — three nested OLS models side-by-side."""
    df = _make_dataset()
    m1 = sp.regress("y_lin ~ x1", data=df)
    m2 = sp.regress("y_lin ~ x1 + x2", data=df)
    m3 = sp.regress("y_lin ~ x1 + x2 + x3", data=df)
    return sp.regtable([m1, m2, m3])


def fixture_custom_stats() -> "sp.output.RegtableResult":
    """F3 — pin stat keys / formatter / star levels to non-defaults."""
    df = _make_dataset()
    m1 = sp.regress("y_lin ~ x1 + x2", data=df)
    m2 = sp.regress("y_lin ~ x1 + x2 + x3", data=df)
    return sp.regtable(
        [m1, m2],
        stats=["n_obs", "r_squared", "adj_r_squared"],
        star_levels=(0.05, 0.01, 0.001),
        fmt="%.3f",
    )


def fixture_with_notes() -> "sp.output.RegtableResult":
    """F4 — title / notes / model labels overrides."""
    df = _make_dataset()
    m1 = sp.regress("y_lin ~ x1 + x2", data=df)
    m2 = sp.regress("y_lin ~ x1 + x2 + x3", data=df)
    return sp.regtable(
        [m1, m2],
        title="Wage Regressions",
        notes=["Robust standard errors in parentheses."],
        model_labels=["Baseline", "Full"],
    )


def fixture_logit() -> "sp.output.RegtableResult":
    """F5 — discrete-choice (logit) so we cover the GLM extraction path."""
    df = _make_dataset()
    m1 = sp.logit("y_bin ~ x1 + x2", data=df)
    return sp.regtable([m1])


FIXTURES = {
    "f1_simple_ols": fixture_simple_ols,
    "f2_multi_model": fixture_multi_model,
    "f3_custom_stats": fixture_custom_stats,
    "f4_with_notes": fixture_with_notes,
    "f5_logit": fixture_logit,
}


def build_fixtures():
    """Return ``{name: RegtableResult}`` for every fixture."""
    return {name: builder() for name, builder in FIXTURES.items()}
