"""Pinning tests for the v1.0.1 NEEDS_VERIFICATION fixes.

These close out the two items deferred in the v1.0.0 review pass:

- ``beyond_average_late`` now uses the Abadie (2002) κ-weighted
  complier-CDF inversion instead of an ad-hoc quantile-range rescaling.
- ``bridge.surrogate_pci`` path B uses a treated-arm counterfactual
  bridge (distinct identifying assumption from Path A's surrogacy),
  not the tautological OLS on (D, S, X).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# beyond_average_late — complier QTE via Abadie CDF inversion.
# ---------------------------------------------------------------------------


def _simulate_late_complier_data(
    n: int = 3000,
    effect_low: float = 0.0,
    effect_high: float = 1.0,
    seed: int = 7,
):
    """
    Simulate a simple LATE-setup with heterogeneous complier effect:
    lower quantile compliers get `effect_low`, upper quantile compliers
    get `effect_high`.  Always-takers / never-takers are noise.
    """
    rng = np.random.default_rng(seed)
    # Types: 0 = never-taker, 1 = always-taker, 2 = complier
    types = rng.choice([0, 1, 2], size=n, p=[0.2, 0.2, 0.6])
    Z = rng.binomial(1, 0.5, n)
    D = np.where(
        types == 0, 0,
        np.where(types == 1, 1, Z)
    )
    u = rng.uniform(0, 1, n)
    y0 = rng.normal(loc=u, scale=0.5)
    # Heterogeneous treatment: effect grows with the latent u
    tau = effect_low + (effect_high - effect_low) * u
    y1 = y0 + tau
    Y = np.where(D == 1, y1, y0)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


def test_beyond_average_late_recovers_heterogeneity():
    df = _simulate_late_complier_data(n=3000, effect_low=0.0,
                                        effect_high=1.0)
    res = sp.beyond_average_late(
        df, y="y", treat="d", instrument="z",
        quantiles=np.array([0.25, 0.5, 0.75]),
        n_boot=30, seed=1,
    )
    # QTE monotonically increasing in q (low-u compliers get 0,
    # high-u compliers get 1).  Check Q75 > Q25.
    q25, q50, q75 = res.late_q
    assert q75 > q25, (
        f"Complier QTE should grow with quantile; got q25={q25:.3f}, "
        f"q75={q75:.3f}"
    )
    # Complier share recovered reasonably (true ≈ 0.6).
    assert 0.3 < res.complier_share < 0.9


def test_beyond_average_late_zero_effect_gives_near_zero_qte():
    df = _simulate_late_complier_data(n=3000, effect_low=0.0,
                                        effect_high=0.0)
    res = sp.beyond_average_late(
        df, y="y", treat="d", instrument="z",
        quantiles=np.array([0.25, 0.5, 0.75]),
        n_boot=20, seed=2,
    )
    # Every QTE level should be near zero when there is no effect.
    for q in res.late_q:
        assert abs(q) < 0.7, f"No-effect QTE should be ~0, got {q:.3f}"


def test_beyond_average_late_rejects_non_binary_instrument():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "d": [0, 1, 0],
                        "z": [0, 1, 2]})
    with pytest.raises(ValueError):
        sp.beyond_average_late(df, y="y", treat="d", instrument="z",
                                n_boot=5)


# ---------------------------------------------------------------------------
# surrogate_pci bridge — path B is genuinely distinct from OLS.
# ---------------------------------------------------------------------------


def test_surrogate_pci_bridge_path_b_differs_from_ols():
    """Path B should NOT collapse to plain OLS on (D, S, X).

    Under heterogeneity, the treated-arm counterfactual bridge gives
    a materially different number from the pooled OLS coefficient on D.
    """
    rng = np.random.default_rng(11)
    n = 500
    D = rng.binomial(1, 0.5, n)
    S = rng.normal(0, 1, n) + 0.3 * D  # surrogate correlated with D
    X = rng.normal(0, 1, n)
    # Heterogeneous outcome: slope on S differs by arm.
    Y = (2.0 * D
         + (0.5 + 1.5 * D) * S
         + 0.3 * X
         + rng.normal(0, 0.5, n))
    df = pd.DataFrame({"y": Y, "d": D, "s": S, "x": X})
    res = sp.bridge(
        kind="surrogate_pci",
        data=df, long_term="y", treat="d",
        short_term=["s"], covariates=["x"],
        n_boot=30, seed=5,
    )
    # For reference: plain OLS coefficient on D
    from sklearn.linear_model import LinearRegression
    Z = np.column_stack([D, S, X])
    ols = LinearRegression().fit(Z, Y)
    ols_d = float(ols.coef_[0])
    # Path B should NOT equal the pooled OLS D-coefficient when
    # outcome heterogeneity exists across arms.
    assert abs(res.estimate_b - ols_d) > 0.05, (
        f"Path B ({res.estimate_b:.3f}) collapsed to OLS on D "
        f"({ols_d:.3f}) — dual-path bridge is broken."
    )


def test_surrogate_pci_bridge_reports_agreement_stats():
    rng = np.random.default_rng(13)
    n = 400
    D = rng.binomial(1, 0.5, n)
    S = rng.normal(0, 1, n)
    Y = 1.5 * D + 0.5 * S + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "d": D, "s": S})
    res = sp.bridge(
        kind="surrogate_pci",
        data=df, long_term="y", treat="d",
        short_term=["s"],
        n_boot=25, seed=3,
    )
    # Both paths report finite estimates in a well-behaved linear
    # simulation, and the agreement test returns a p-value in [0, 1].
    assert np.isfinite(res.estimate_a)
    assert np.isfinite(res.estimate_b)
    assert 0.0 <= res.diff_p <= 1.0
