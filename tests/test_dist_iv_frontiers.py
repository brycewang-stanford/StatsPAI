"""Smoke tests for v0.10 distributional IV / panel QTE frontier."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def iv_data():
    """Valid LATE design: true complier effect 1.5, exclusion satisfied.

    The previous fixture had ``Y = 1.5*D + 0.5*Z + eps`` -- a direct Z->Y
    path, i.e. a violated exclusion restriction -- so no IV estimator could
    have recovered 1.5 from it, and the frozen expectations it produced were
    numbers from a misspecified design.  The ``0.5 * Z`` term is gone.
    """
    rng = np.random.default_rng(0)
    n = 4000
    Z = (rng.uniform(size=n) < 0.5).astype(int)
    D = (rng.uniform(size=n) < 0.4 + 0.4 * Z).astype(int)
    Y = 1.5 * D + rng.standard_normal(n)
    return pd.DataFrame({"y": Y, "treat": D, "z": Z})


@pytest.fixture
def panel_qte_data():
    rng = np.random.default_rng(1)
    n_units, n_t = 30, 4
    rows = []
    for u in range(n_units):
        for t in range(n_t):
            d = int(u >= 15 and t >= 2)
            x = rng.standard_normal(5)
            y = u * 0.05 + t * 0.1 + 1.0 * d + 0.5 * x[0] + rng.standard_normal()
            rows.append(
                {
                    "y": y,
                    "treat": d,
                    "unit": f"u{u}",
                    "time": t,
                    **{f"x{i}": x[i] for i in range(5)},
                }
            )
    return pd.DataFrame(rows)


def test_dist_iv(iv_data):
    """Constant complier effect 1.5 => flat QTE at 1.5.

    Asserts against the DGP truth rather than frozen digits: the previous
    expectations (2.59 / 2.57 / 2.29) came from the pre-1.21 quantile-Wald
    ratio, which was inconsistent for this estimand.
    """
    res = sp.dist_iv(
        iv_data,
        y="y",
        treat="treat",
        instrument="z",
        quantiles=np.array([0.25, 0.5, 0.75]),
    )
    assert isinstance(res, sp.DistIVResult)
    assert len(res.late_q) == 3
    np.testing.assert_allclose(res.late_q, 1.5, atol=0.25)
    assert np.all(res.se_q > 0)
    # first stage is P(D=1|Z=1) - P(D=1|Z=0) = 0.4 by construction
    assert abs(res.complier_share - 0.4) < 0.05


def test_kan_dlate(iv_data):
    """Deprecated alias: warns, and returns exactly what dist_iv returns."""
    with pytest.warns(DeprecationWarning, match="kan_dlate"):
        res = sp.kan_dlate(
            iv_data,
            y="y",
            treat="treat",
            instrument="z",
            quantiles=np.array([0.5]),
        )
    assert isinstance(res, sp.DistIVResult)
    direct = sp.dist_iv(
        iv_data, y="y", treat="treat", instrument="z", quantiles=np.array([0.5])
    )
    np.testing.assert_allclose(res.late_q, direct.late_q)
    np.testing.assert_allclose(res.late_q, 1.5, atol=0.25)


def test_qte_hd_panel(panel_qte_data):
    """True effect is a constant 1.0; assert recovery, not frozen digits.

    The previous expectations came from the pre-1.21 within-demeaning path,
    which is inconsistent off pure location shifts. This fixture has only
    T = 4, so Canay's first step is noisy; the accompanying short-panel
    warning is asserted in test_hd_panel_qte.py::test_short_panel_warns
    (Python emits each warn() call site only once per session, so it cannot
    be asserted in two places).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = sp.qte_hd_panel(
            panel_qte_data,
            y="y",
            treat="treat",
            unit="unit",
            time="time",
            covariates=[f"x{i}" for i in range(5)],
            quantiles=np.array([0.25, 0.5, 0.75]),
            se="none",
        )
    assert isinstance(res, sp.HDPanelQTEResult)
    assert len(res.qte) == 3
    # Wide band: 30 units x 4 periods is a small, short panel.
    np.testing.assert_allclose(res.qte, 1.0, atol=0.6)
    # SEs must be absent, not fabricated, when se='none'.
    assert np.all(np.isnan(res.se))


def test_beyond_average(iv_data):
    res = sp.beyond_average_late(
        iv_data,
        y="y",
        treat="treat",
        instrument="z",
        quantiles=np.array([0.25, 0.5, 0.75]),
        n_boot=50,
    )
    assert isinstance(res, sp.BeyondAverageResult)
    assert 0 < res.complier_share < 1


def test_beyond_average_invalid_instrument():
    df = pd.DataFrame(
        {
            "y": np.random.randn(50),
            "treat": np.random.randint(0, 2, 50),
            "z": np.random.randn(50),  # continuous, should fail
        }
    )
    with pytest.raises(ValueError, match="binary"):
        sp.beyond_average_late(df, y="y", treat="treat", instrument="z")


class TestDistIVDegenerateInstrument:
    """A degenerate instrument must fail loudly, not return numbers (§7)."""

    def test_constant_instrument_raises(self):
        """Pre-1.21 this warned and returned an all-NaN ``late_q``.

        A constant instrument has no first stage at all, so there is nothing
        to estimate: raise instead of handing back a result object whose
        fields are silently NaN.
        """
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(n),
                "treat": (rng.uniform(size=n) < 0.5).astype(int),
                "z": np.ones(n, dtype=int),  # degenerate: no variation
            }
        )
        with pytest.raises(ValueError, match="first stage|complier share"):
            sp.dist_iv(df, y="y", treat="treat", instrument="z")

    def test_healthy_instrument_returns_finite_estimates(self, iv_data):
        res = sp.dist_iv(iv_data, y="y", treat="treat", instrument="z")
        assert np.isfinite(res.late_q).all()
        assert np.isfinite(res.se_q).all()
