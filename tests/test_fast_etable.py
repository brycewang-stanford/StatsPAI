"""Tests for ``sp.fast.etable`` (Phase 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _poisson_data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    fe = rng.integers(0, 50, size=n).astype(np.int32)
    eta = 0.5 + 0.30 * x1 - 0.20 * x2 + 0.1 * (fe % 5)
    y = rng.poisson(np.exp(np.clip(eta, -10, 10))).astype(np.int64)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "fe": fe})


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_etable_single_fepois():
    df = _poisson_data(seed=1)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab = sp.fast.etable(fit)
    assert isinstance(tab, pd.DataFrame)
    assert "x1" in tab.index
    assert "x2" in tab.index
    assert "N" in tab.index


def test_etable_multi_models_aligned_by_var():
    df = _poisson_data(seed=2)
    fit_full = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    fit_only_x1 = sp.fast.fepois("y ~ x1 | fe", df)
    tab = sp.fast.etable(fit_full, fit_only_x1, names=["full", "x1-only"])
    # Both columns present
    assert list(tab.columns) == ["full", "x1-only"]
    # x1 row populated in both columns; x2 row only in 'full'
    assert tab.loc["x1", "full"] != ""
    assert tab.loc["x1", "x1-only"] != ""
    assert tab.loc["x2", "full"] != ""
    assert tab.loc["x2", "x1-only"] == ""


def test_etable_keep_filter():
    df = _poisson_data(seed=3)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab = sp.fast.etable(fit, keep=["x1"])
    assert "x1" in tab.index
    assert "x2" not in tab.index


def test_etable_drop_filter():
    df = _poisson_data(seed=4)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab = sp.fast.etable(fit, drop=["x2"])
    assert "x1" in tab.index
    assert "x2" not in tab.index


def test_etable_below_format_doubles_rows():
    df = _poisson_data(seed=5)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab_paren = sp.fast.etable(fit, se_format="paren")
    tab_below = sp.fast.etable(fit, se_format="below")
    # below format adds an SE row beneath each coef row → strictly more rows
    assert len(tab_below) > len(tab_paren)


def test_etable_latex_output():
    df = _poisson_data(seed=6)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    s = sp.fast.etable(fit, format="latex")
    assert isinstance(s, str)
    assert "tabular" in s
    assert "x1" in s


def test_etable_html_output():
    df = _poisson_data(seed=7)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    s = sp.fast.etable(fit, format="html")
    assert "<table" in s
    assert "x1" in s


def test_etable_stars_appear_when_significant():
    """β=0.30 with small SE on n=2000 should be highly significant."""
    df = _poisson_data(seed=8)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab = sp.fast.etable(fit, stars=True)
    # x1 should have stars (true β=0.3, large t-stat); x2 too (β=-0.2)
    assert "*" in tab.loc["x1", "(1)"]


def test_etable_stars_off():
    df = _poisson_data(seed=9)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe", df)
    tab = sp.fast.etable(fit, stars=False)
    assert "*" not in tab.loc["x1", "(1)"]


def test_etable_works_with_event_study():
    """Cross-class compatibility: event-study results have coef()/se()/n_obs."""
    rng = np.random.default_rng(10)
    n_units, n_periods = 30, 10
    rows = []
    for u in range(n_units):
        treat_t = 5 if u < 15 else None
        for t in range(n_periods):
            et = (t - treat_t) if treat_t is not None else np.nan
            d = 1.0 if (treat_t is not None and t >= treat_t) else 0.0
            y = 0.4 * d + rng.normal()
            rows.append((u, t, y, et))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "event_time"])
    es = sp.fast.event_study(
        df, y="y", unit="unit", time="time", event_time="event_time",
        window=(-2, 2),
    )

    # event_study returns coefs / ses but indexes them by integer event-time;
    # we need a dict-like .coef() / .se(). Adapt:
    class _ESAdapter:
        def __init__(self, e):
            self.e = e
            self.n_obs = e.n_obs
        def coef(self):
            return pd.Series(self.e.coefs,
                              index=[f"et_{int(t)}" for t in self.e.event_times])
        def se(self):
            return pd.Series(self.e.ses,
                              index=[f"et_{int(t)}" for t in self.e.event_times])
    tab = sp.fast.etable(_ESAdapter(es))
    assert "N" in tab.index
    assert any(idx.startswith("et_") for idx in tab.index)


def test_etable_unknown_format_rejected():
    df = _poisson_data(seed=11)
    fit = sp.fast.fepois("y ~ x1 | fe", df)
    with pytest.raises(ValueError, match="format"):
        sp.fast.etable(fit, format="bogus")


def test_etable_no_models_rejected():
    with pytest.raises(ValueError, match="at least one"):
        sp.fast.etable()


# ---------------------------------------------------------------------------
# t-distribution stars (P2 #11 follow-up)
# ---------------------------------------------------------------------------

def test_etable_uses_t_distribution_when_df_residual_present():
    """Stars should use the t-distribution when the fit exposes df_residual.

    On a small-sample fit with df < ~30, the t critical values are
    materially larger than the Normal-z fallback (e.g. for df=10
    two-sided 5%: t = 2.228 vs z = 1.960). A coefficient with z just
    above 1.960 should NOT pick up two stars under the t-stars path.
    """
    rng = np.random.default_rng(100)
    n = 25  # small sample → df ~= 25 - p - FE rank
    df = pd.DataFrame({
        "y": rng.poisson(2.0, size=n).astype(np.int64),
        "x1": rng.normal(size=n),
        "fe": rng.integers(0, 3, size=n).astype(np.int32),
    })
    fit = sp.fast.fepois("y ~ x1 | fe", df)
    assert fit.df_residual > 0

    # Hand-compute z-stat to compare with what etable gets
    coef_x1 = float(fit.coef()["x1"])
    se_x1 = float(fit.se()["x1"])
    z = abs(coef_x1 / se_x1)

    tab = sp.fast.etable(fit, stars=True)
    cell = tab.loc["x1", "(1)"]

    # Manual t critical values for fit.df_residual
    from scipy import stats
    t5 = stats.t.ppf(0.975, fit.df_residual)
    if z > t5:
        assert "**" in cell
    else:
        assert cell.count("*") < 2, (
            f"got {cell!r} but z={z:.3f} <= t5={t5:.3f}"
        )


def test_etable_falls_back_to_z_when_no_df_residual():
    """If a fit object lacks df_residual we should still produce a table —
    using Normal-z thresholds rather than crashing."""

    class _MinimalFit:
        n_obs = 1000
        coef_names = ["x1"]
        coef_vec = np.array([0.30])

        def coef(self):
            return pd.Series(self.coef_vec, index=self.coef_names)

        def se(self):
            return pd.Series([0.10], index=self.coef_names)

    tab = sp.fast.etable(_MinimalFit())
    # 0.30 / 0.10 = 3.0 > 2.576 (z critical at 1%)
    assert "***" in tab.loc["x1", "(1)"]
