"""``sp.xtlsdvc(plug_in=)``: the first-order correction against its exact limit.

For an AR(1) panel with unit effects, no regressors and a stationary start,
the inconsistency of the within estimator has a closed form (Nickell 1981).
In the notation of Bun and Kiviet (2003) it is ``tr(P) / (v' M v / (1 -
rho^2) + tr(P'P))`` with ``P = M C``, ``C = L (I - rho L)^{-1}`` and
``v_t = rho^t``. The first-order term of the LSDVC correction estimates that
quantity. With ``plug_in='bruno'`` (Stata's ``xtlsdvc``) the deterministic
part of the lagged outcome is built from within-demeaned residuals, which
adds ``iota' C' M C iota / T`` to the denominator: the correction converges
to a smaller number. ``plug_in='consistent'`` removes that excess.

Found by the SE calibration screen (``tests/reliability/se_screen/panel.py``,
row ``xtlsdvc/default``: bias of -0.55 SD at 500 units).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RHO = 0.5
N_UNITS = 3000
N_PERIODS = 8  # seven usable periods after the lag


def _limits(rho: float, T: int = N_PERIODS - 1):
    """(exact inconsistency, limit of the Stata plug-in) of LSDV at ``rho``."""
    L = np.eye(T, k=-1)
    C = L @ np.linalg.inv(np.eye(T) - rho * L)
    iota = np.ones((T, 1))
    M = np.eye(T) - iota @ iota.T / T
    P = M @ C
    v = np.array([rho**t for t in range(T)])[:, None]
    den = (v.T @ M @ v).item() / (1 - rho**2) + np.trace(P.T @ P)
    excess = ((C @ iota).T @ M @ (C @ iota)).item() / T
    return np.trace(P) / den, np.trace(P) / (den + excess)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    burn = 30
    a = rng.normal(size=N_UNITS)
    y = np.zeros((N_UNITS, N_PERIODS + burn))
    y[:, 0] = a / (1 - RHO) + rng.normal(size=N_UNITS) / np.sqrt(1 - RHO**2)
    for j in range(1, N_PERIODS + burn):
        y[:, j] = RHO * y[:, j - 1] + a + rng.normal(size=N_UNITS)
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(N_UNITS), N_PERIODS),
            "time": np.tile(np.arange(N_PERIODS), N_UNITS),
            "y": y[:, burn:].ravel(),
        }
    )


def _fit(panel, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtlsdvc(panel, y="y", bias_order=1, **kw)


def test_closed_form_is_the_nickell_inconsistency():
    """The exact limit used below is Nickell's formula, and the two differ."""
    T = N_PERIODS - 1
    num = -(1 + RHO) / (T - 1) * (1 - (1 - RHO**T) / (T * (1 - RHO)))
    den = 1 - 2 * RHO / ((1 - RHO) * (T - 1)) * (1 - (1 - RHO**T) / (T * (1 - RHO)))
    exact, stata = _limits(RHO)
    assert exact == pytest.approx(num / den, rel=1e-12)
    assert exact == pytest.approx(-0.2353372434, abs=1e-9)
    assert stata == pytest.approx(-0.2194644024, abs=1e-9)


def test_consistent_plug_in_recovers_the_exact_inconsistency(panel):
    r = _fit(panel, plug_in="consistent")
    exact, _ = _limits(r.model_info["initial_rho"])
    # Sampling error of W'MW and sigma^2 at 3,000 units: at most 0.0045 over
    # twelve seeds; the two limits are 0.016 apart.
    assert r.detail["bias_subtracted"].iloc[0] == pytest.approx(exact, abs=0.006)
    # 3,000 units put the LSDVC estimate within 0.01 of the truth (its
    # standard deviation there is about 0.008).
    assert r.estimate == pytest.approx(RHO, abs=0.02)


def test_default_plug_in_is_stata_and_converges_to_the_smaller_limit(panel):
    r = _fit(panel)
    assert r.model_info["plug_in"] == "bruno"
    _, stata = _limits(r.model_info["initial_rho"])
    assert r.detail["bias_subtracted"].iloc[0] == pytest.approx(stata, abs=0.006)
    explicit = _fit(panel, plug_in="bruno")
    assert explicit.estimate == r.estimate


def test_uncorrected_estimate_does_not_depend_on_the_plug_in(panel):
    a = _fit(panel).detail["lsdv_uncorrected"].iloc[0]
    b = _fit(panel, plug_in="consistent").detail["lsdv_uncorrected"].iloc[0]
    assert a == b
    exact, _ = _limits(RHO)
    # LSDV itself sits at the Nickell value (sd about 0.006 at 3,000 units).
    assert a - RHO == pytest.approx(exact, abs=0.02)


def test_unknown_plug_in_is_rejected(panel):
    with pytest.raises(ValueError, match="plug_in must be"):
        sp.xtlsdvc(panel, y="y", plug_in="exact")


def test_bootstrap_refits_with_the_same_plug_in(panel):
    """The bootstrap of the consistent estimator is seeded and reports its source."""
    small = panel[panel["id"] < 150]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.xtlsdvc(small, y="y", plug_in="consistent", bootstrap=20, seed=3)
        again = sp.xtlsdvc(small, y="y", plug_in="consistent", bootstrap=20, seed=3)
    assert r.se == again.se and np.isfinite(r.se) and r.se > 0
    assert r.model_info["bootstrap_failed"] == 0
    assert r.model_info["plug_in"] == "consistent"
    assert r.model_info["se_source"] == "parametric bootstrap (20 replications)"
    lo, hi = r.ci
    assert lo == pytest.approx(r.estimate - 1.959963984540054 * r.se, rel=1e-12)
    assert hi == pytest.approx(r.estimate + 1.959963984540054 * r.se, rel=1e-12)
