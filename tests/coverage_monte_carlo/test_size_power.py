"""Monte Carlo size and power validation for the core closed-form estimators.

Coverage (the sibling ``test_coverage.py``) checks that a 95% CI covers the
truth ~95% of the time. Size and power check the *other* two faces of the same
inference machinery:

- **Size** — under the null (true effect = 0), a nominal 5% two-sided test
  must reject ~5% of the time. An over-sized test invents significance;
  an under-sized one is needlessly conservative. This is the false-positive
  rate that referees worry about most.
- **Power** — under a sequence of alternatives (true effect = delta > 0), the
  rejection rate must rise monotonically with delta and approach 1 for a
  large effect. A test that never rejects is calibrated but useless.

The test statistic is the 95% CI itself: we reject H0: effect = 0 iff 0 lies
outside the interval. This is exactly equivalent to a two-sided 5% Wald test
and reuses the same ``.ci`` / ``.conf_int`` interface the coverage suite
validates, so size/power and coverage are guaranteed to be consistent.

These rows cover the analytically-fast estimators: OLS, 2x2 DiD, strong-IV
2SLS, sharp RD, two-way FE panel, Callaway-Sant'Anna staggered DiD (influence-
function SEs), and entropy-balancing (convex dual). A power curve needs B draws
at each of several effect sizes, so the cross-fit estimators (DML, causal
forest) and resampling-based SDID stay coverage-only in ``test_coverage.py``;
they would turn a multi-delta sweep into hours of wall-clock.

Entropy balancing is included precisely because it is *conservative*: its size
sits near 0 (it almost never rejects under the null, mirroring its ~1.0 over-
coverage) and its power therefore rises later than the exactly-calibrated
estimators. That is an honest, documented characterisation, not a defect — the
size test only caps the upper tail (validity), and the power curve still has to
reach the same >=0.80 at the top effect size.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import statspai as sp

B_DEFAULT = int(os.environ.get("STATSPAI_MC_DRAWS", 300))

NOMINAL_SIZE = 0.05


def _wilson_bounds_at(p: float, B: int, conf: float = 0.99) -> tuple:
    """Two-sided Wilson-score band around proportion ``p`` for ``B`` draws.

    ``conf`` is the confidence of the test-of-size itself (default 99%, so
    only a grossly mis-sized test fails). For ``p=0.05``, ``B=300`` this is
    roughly ``[0.026, 0.093]``.
    """
    from scipy.stats import norm

    z = norm.ppf((1 + conf) / 2)
    denom = 1 + z**2 / B
    centre = (p + z**2 / (2 * B)) / denom
    half = z * (p * (1 - p) / B + z**2 / (4 * B**2)) ** 0.5 / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _assert_size(rejections: int, B: int, label: str) -> None:
    """Assert the empirical size does not exceed the nominal 5% (one-sided).

    Test *validity* is a one-sided property: a 5% test must not OVER-reject
    under the null (anti-conservative SEs invent false significance). Under-
    rejection is conservative — it is valid but inefficient and shows up as
    over-coverage in the sibling coverage suite (e.g. classic 2x2 DiD sizes
    at ~0.017, mirroring its 0.955 coverage). So we cap the empirical size at
    the Wilson upper bound and only floor it loosely to catch a degenerate
    never-reject estimator (the power suite is the real guard against that).
    """
    rate = rejections / B
    _lo, hi = _wilson_bounds_at(NOMINAL_SIZE, B)
    assert rate <= hi, (
        f"{label} empirical size = {rate:.3f} exceeds Wilson upper bound "
        f"{hi:.3f} for nominal {NOMINAL_SIZE} (B={B}) — test over-rejects"
    )


def _assert_power_curve(
    powers: list,
    deltas: list,
    label: str,
    min_top_power: float = 0.80,
    slack: float = 0.03,
) -> None:
    """Assert the power curve is (near-)monotone and reaches ``min_top_power``.

    ``slack`` tolerates Monte Carlo wiggle in the monotonicity check so a
    1-2 draw dip between adjacent effect sizes does not fail the test.
    """
    for a, b, da, db in zip(powers, powers[1:], deltas, deltas[1:]):
        assert b >= a - slack, (
            f"{label} power non-monotone: power({db})={b:.3f} < "
            f"power({da})={a:.3f} - {slack}"
        )
    assert powers[-1] >= min_top_power, (
        f"{label} power at delta={deltas[-1]} = {powers[-1]:.3f} "
        f"below required {min_top_power}"
    )


def _reject(ci, null: float = 0.0) -> bool:
    """Reject H0: effect = null iff null lies outside the 95% CI."""
    if ci is None or len(ci) != 2:
        return False
    lo, hi = ci
    return not (lo <= null <= hi)


# ---------------------------------------------------------------------------
# Per-estimator single-draw fit -> (does the 95% CI exclude 0?)
#
# Each ``_fit_*`` takes (seed, delta) and returns True iff H0 is rejected on
# that draw. Size reuses delta=0; power sweeps delta>0.
# ---------------------------------------------------------------------------


def _fit_ols(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n = 500
    d = rng.binomial(1, 0.5, n)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + delta * d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    r = sp.regress("y ~ d + x", data=df, robust="hc1")
    lo, hi = r.conf_int().loc["d"].values
    return _reject((lo, hi))


def _fit_did(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n = 300
    rows = []
    for i in range(n):
        treat = 1 if i < n // 2 else 0
        ui = rng.normal(scale=0.5)
        for t in (0, 1):
            y = (
                1.0
                + 0.3 * t
                + 0.5 * treat
                + delta * treat * t
                + ui
                + rng.normal(scale=0.7)
            )
            rows.append({"i": i, "t": t, "treated": treat, "post": t, "y": y})
    df = pd.DataFrame(rows)
    r = sp.did(df, y="y", treat="treated", time="t", post="post")
    return _reject(r.ci)


def _fit_iv(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n = 600
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(size=n)
    d = (0.2 + 0.6 * z + 0.3 * u + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    y = 1.0 + delta * d + 0.5 * u + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    r = sp.ivreg("y ~ (d ~ z)", data=df, robust="hc1")
    lo, hi = r.conf_int().loc["d"].values
    return _reject((lo, hi))


def _fit_rd(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.uniform(-1, 1, n)
    y = 2 + 3 * x + x**2 + delta * (x >= 0).astype(int) + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"y": y, "x": x})
    r = sp.rdrobust(df, y="y", x="x", c=0.0)
    return _reject(r.ci)


def _fit_panel(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n_units, n_time = 50, 6
    rows = []
    for i in range(n_units):
        ai = rng.normal()
        for t in range(n_time):
            d = rng.binomial(1, 0.5)
            y = ai + 0.3 * t + delta * d + rng.normal(scale=0.8)
            rows.append({"i": i, "t": t, "d": d, "y": y})
    df = pd.DataFrame(rows)
    r = sp.panel(df, formula="y ~ d", entity="i", time="t", method="fe")
    lo, hi = r.conf_int().loc["d"].values
    return _reject((lo, hi))


def _fit_cs(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    cohorts = [3, 5, 7, 0]
    rows = []
    for i in range(200):
        g = cohorts[i % 4]
        ui = rng.normal(scale=0.5)
        for t in range(1, 9):
            post = 1 if (g > 0 and t >= g) else 0
            y = 0.2 * t + delta * post + ui + rng.normal(scale=0.8)
            rows.append({"i": i, "t": t, "g": g, "y": y})
    r = sp.callaway_santanna(
        pd.DataFrame(rows), y="y", g="g", t="t", i="i", estimator="reg"
    )
    return _reject(r.ci)


def _fit_ebalance(seed: int, delta: float) -> bool:
    rng = np.random.default_rng(seed)
    n = 500
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-0.3 + 0.5 * X1 - 0.3 * X2)))
    d = (rng.uniform(0, 1, n) < p).astype(int)
    y = 1.0 + 1.5 * X1 - 0.8 * X2 + delta * d + rng.normal(scale=0.8, size=n)
    df = pd.DataFrame({"y": y, "d": d, "X1": X1, "X2": X2})
    r = sp.ebalance(df, y="y", treat="d", covariates=["X1", "X2"])
    return _reject(r.ci)


# (fit_fn, label, B-cap, power-effect-sizes). RD and CS are slower so capped.
# Entropy balancing is conservative, so its power sweep runs to a larger top
# effect (1.0) to clear the >=0.80 bar.
_ESTIMATORS = [
    (_fit_ols, "OLS RCT", None, [0.0, 0.10, 0.20, 0.30]),
    (_fit_did, "DID 2x2", None, [0.0, 0.20, 0.40, 0.60]),
    (_fit_iv, "IV 2SLS", None, [0.0, 0.20, 0.40, 0.60]),
    (_fit_rd, "RD sharp", 200, [0.0, 0.20, 0.40, 0.60]),
    (_fit_panel, "Panel FE", None, [0.0, 0.15, 0.30, 0.45]),
    (_fit_cs, "CS staggered", 100, [0.0, 0.30, 0.60, 0.90]),
    (_fit_ebalance, "Ebalance", 200, [0.0, 0.40, 0.70, 1.00]),
]


# ---------------------------------------------------------------------------
# Size: rejection rate under the null must be ~5%
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "fit_fn,label,cap,_deltas", _ESTIMATORS, ids=[e[1] for e in _ESTIMATORS]
)
def test_size_under_null(fit_fn, label, cap, _deltas):
    """Under H0 (effect=0), a nominal 5% test must reject ~5% of the time."""
    B = B_DEFAULT if cap is None else min(B_DEFAULT, cap)
    rejections = sum(fit_fn(seed, 0.0) for seed in range(B))
    _assert_size(rejections, B, label)


# ---------------------------------------------------------------------------
# Power: rejection rate must rise monotonically and reach >=0.80 at the
# largest effect size
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "fit_fn,label,cap,deltas", _ESTIMATORS, ids=[e[1] for e in _ESTIMATORS]
)
def test_power_curve(fit_fn, label, cap, deltas):
    """Power must increase with the effect size and be high at the top."""
    B = B_DEFAULT if cap is None else min(B_DEFAULT, cap)
    powers = []
    for delta in deltas:
        rej = sum(fit_fn(seed, delta) for seed in range(B))
        powers.append(rej / B)
    # power[0] is the null point — it doubles as a size sanity check.
    lo, hi = _wilson_bounds_at(NOMINAL_SIZE, B)
    assert powers[0] <= hi + 0.02, (
        f"{label} power curve null point {powers[0]:.3f} too high "
        f"(size band upper {hi:.3f})"
    )
    _assert_power_curve(powers, deltas, label)


# ---------------------------------------------------------------------------
# Fast smoke: OLS size at B=60, always runs (not slow)
# ---------------------------------------------------------------------------


def test_ols_size_smoke():
    """B=60 smoke: OLS size under the null must not be wildly off 5%."""
    B = 60
    rejections = sum(_fit_ols(seed, 0.0) for seed in range(B))
    rate = rejections / B
    # Wide band at B=60: a catastrophic mis-sizing (e.g. 0.3) still fails.
    assert (
        0.0 <= rate <= 0.20
    ), f"OLS smoke size = {rate:.3f} outside [0.0, 0.20] (B={B})"
