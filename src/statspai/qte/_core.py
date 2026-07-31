"""Shared primitives for the QTE / distributional-effects family.

Everything in this module is private (``_``-prefixed at the call sites that
matter) and exists so that the estimators in ``qte/`` stop re-implementing
the same four things.  Per CLAUDE.md §4, module-level shared primitives live
here rather than being copy-pasted across ``qte.py`` / ``dist_iv.py`` /
``beyond_average.py`` / ``hd_panel.py``.

Contents
--------
Weighted empirical distribution
    :func:`weighted_quantile`, :func:`weighted_ecdf`, :func:`invert_cdf`
Abadie (2002, 2003) complier weighting
    :func:`abadie_kappa`, :func:`complier_cdfs`
Density estimation (for analytic influence-function standard errors)
    :func:`kernel_density_at`
Monotonicity
    :func:`rearrange`

References
----------
abadie2002bootstrap, abadie2003semiparametric, frolich2013unconditional,
chernozhukov2010quantile
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "weighted_quantile",
    "weighted_ecdf",
    "invert_cdf",
    "abadie_kappa",
    "complier_cdfs",
    "kernel_density_at",
    "rearrange",
    "multiplier_bootstrap",
    "uniform_band",
    "functional_test",
]


# ══════════════════════════════════════════════════════════════════════
#  Weighted empirical distribution
# ══════════════════════════════════════════════════════════════════════


def weighted_quantile(
    values: np.ndarray,
    taus: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted empirical quantiles.

    Uses the *left-continuous inverse* of the weighted ECDF,
    ``Q(tau) = inf{y : F(y) >= tau}``, which is the definition that makes
    the weighted check-function minimiser and the CDF inverse agree.  This
    matches ``quantile(type=1)`` in R, NOT the default ``type=7``: for a
    weighted sample there is no interpolation convention that stays
    consistent with the influence function, so we take the step function.

    Parameters
    ----------
    values : ndarray
        Sample points.
    taus : ndarray
        Probability levels in (0, 1).
    weights : ndarray, optional
        Non-negative weights. ``None`` means equal weights.

    Returns
    -------
    ndarray
        Quantiles, same shape as ``taus``.
    """
    values = np.asarray(values, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if values.size == 0:
        return np.full(taus.shape, np.nan)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(taus.shape, np.nan)
    cdf = np.cumsum(w) / total
    # inf{y : F(y) >= tau}  ->  first index whose running CDF reaches tau
    idx = np.searchsorted(cdf, taus, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return np.asarray(v[idx])


def weighted_ecdf(
    values: np.ndarray,
    grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted empirical CDF evaluated on ``grid``.

    Vectorised via ``searchsorted`` on the sorted sample (the naive
    ``[sum(w * (vals <= g)) for g in grid]`` loop is O(n * n_grid) and
    dominates runtime once the bootstrap wraps it).
    """
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if values.size == 0:
        return np.zeros(grid.shape)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    total = w.sum()
    if not np.isfinite(total) or total == 0:
        return np.zeros(grid.shape)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    idx = np.searchsorted(v, grid, side="right")
    return np.asarray(cum[idx] / total)


def invert_cdf(grid: np.ndarray, cdf: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Invert a CDF tabulated on ``grid`` at levels ``taus``.

    ``Q(tau) = inf{y in grid : F(y) >= tau}``.  ``cdf`` must be
    non-decreasing (call :func:`rearrange` first if it may not be).
    """
    grid = np.asarray(grid, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if grid.size == 0:
        return np.full(taus.shape, np.nan)
    idx = np.searchsorted(cdf, taus, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    return np.asarray(grid[idx])


# ══════════════════════════════════════════════════════════════════════
#  Abadie complier weighting
# ══════════════════════════════════════════════════════════════════════


def abadie_kappa(
    D: np.ndarray,
    Z: np.ndarray,
    pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Abadie's one-sided kappa weights for the complier subpopulation.

    With ``pi(X) = P(Z = 1 | X)``,

    .. math::

        \\kappa_1 = D \\frac{Z - \\pi(X)}{\\pi(X)(1 - \\pi(X))}, \\qquad
        \\kappa_0 = (1 - D) \\frac{\\pi(X) - Z}{\\pi(X)(1 - \\pi(X))}

    these satisfy, for any integrable ``g``,

    .. math::

        E[\\kappa_1 g(Y)] = E[g(Y_1) \\mathbf{1}\\{complier\\}], \\qquad
        E[\\kappa_0 g(Y)] = E[g(Y_0) \\mathbf{1}\\{complier\\}]

    so ``E[kappa_1] = E[kappa_0] = P(complier)``.  Always-takers and
    never-takers drop out in expectation because ``E[Z - pi(X) | X] = 0``.

    Individual weights CAN be negative — that is a feature of Abadie's
    identification result, not a bug.  Downstream CDF estimators must
    therefore enforce monotonicity and the [0, 1] range explicitly; see
    :func:`complier_cdfs`.

    Returns
    -------
    (kappa_1, kappa_0) : tuple of ndarray

    References
    ----------
    abadie2003semiparametric
    """
    D = np.asarray(D, dtype=float)
    Z = np.asarray(Z, dtype=float)
    pi = np.asarray(pi, dtype=float)
    denom = pi * (1.0 - pi)
    if np.any(denom <= 0):
        raise ValueError(
            "Abadie kappa requires 0 < P(Z=1|X) < 1 for every observation; "
            f"{int(np.sum(denom <= 0))} observation(s) violate this. "
            "Trim or re-specify the instrument propensity model."
        )
    kappa_1 = D * (Z - pi) / denom
    kappa_0 = (1.0 - D) * (pi - Z) / denom
    return kappa_1, kappa_0


def complier_cdfs(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    pi: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Abadie-weighted marginal potential-outcome CDFs for compliers.

    Estimates ``F_{Y(1)|complier}`` and ``F_{Y(0)|complier}`` on a shared
    grid, using :func:`abadie_kappa`.  Without covariates (``pi=None``)
    ``pi`` is the constant sample share ``P(Z=1)`` and the estimator
    collapses to the per-CDF Wald identity

    .. math::

        F_{Y(1)|c}(y) = \\frac{P(Y \\le y, D=1 | Z=1)
                              - P(Y \\le y, D=1 | Z=0)}{\\Delta p}

    which is the Imbens-Angrist/Abadie (2002) form.  With covariates this
    is the Frolich & Melly (2013) unconditional IV-QTE weighting.

    **This is the correct object for a distributional LATE.**  The
    "Wald ratio of quantiles",
    ``[Q(tau|Z=1) - Q(tau|Z=0)] / [E(D|Z=1) - E(D|Z=0)]``, is NOT — the
    quantile operator is not linear, so the mean-Wald scaling does not
    carry over.  See the WP-1 regression test in
    ``tests/reference_parity/test_dist_iv_parity.py``.

    Parameters
    ----------
    Y, D, Z : ndarray
        Outcome, binary treatment, binary instrument.
    pi : ndarray, optional
        ``P(Z=1|X)`` per observation. ``None`` uses the constant sample share.
    grid : ndarray, optional
        Evaluation grid. Defaults to the sorted unique values of ``Y``.

    Returns
    -------
    (grid, F1, F0, complier_share) or None
        ``None`` when the first stage is degenerate (``|E[kappa_1]| < 1e-10``).
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = len(Y)
    if n == 0:
        return None

    if pi is None:
        p1 = float(np.mean(Z))
        if p1 <= 0.0 or p1 >= 1.0:
            return None
        pi_vec = np.full(n, p1)
    else:
        pi_vec = np.asarray(pi, dtype=float)

    try:
        kappa_1, kappa_0 = abadie_kappa(D, Z, pi_vec)
    except ValueError:
        return None

    share_1 = float(np.mean(kappa_1))
    share_0 = float(np.mean(kappa_0))
    # Both estimate P(complier); use kappa_1's for the reported share but
    # bail if either normaliser is degenerate (an F0 built on a ~0 denominator
    # is numerically meaningless even when F1 looks fine).
    if abs(share_1) < 1e-10 or abs(share_0) < 1e-10:
        return None

    if grid is None:
        grid = np.unique(Y)
    grid = np.asarray(grid, dtype=float)

    # E[kappa_j 1{Y <= y}] / E[kappa_j], vectorised over the grid.
    order = np.argsort(Y, kind="mergesort")
    y_s = Y[order]
    idx = np.searchsorted(y_s, grid, side="right")
    cum1 = np.concatenate([[0.0], np.cumsum(kappa_1[order])])
    cum0 = np.concatenate([[0.0], np.cumsum(kappa_0[order])])
    F1 = (cum1[idx] / n) / share_1
    F0 = (cum0[idx] / n) / share_0

    # Negative kappas can make the raw curves non-monotone; project back.
    F1 = np.clip(np.maximum.accumulate(F1), 0.0, 1.0)
    F0 = np.clip(np.maximum.accumulate(F0), 0.0, 1.0)
    return grid, F1, F0, share_1


# ══════════════════════════════════════════════════════════════════════
#  Density estimation (analytic influence-function SEs)
# ══════════════════════════════════════════════════════════════════════


def kernel_density_at(
    values: np.ndarray,
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Weighted Gaussian kernel density evaluated at ``points``.

    Needed because every quantile influence function carries a ``1/f(Q(tau))``
    factor.  Bandwidth defaults to Silverman's rule on the *effective* sample
    size ``(sum w)^2 / sum w^2``, which is the right n when weights are
    non-uniform.

    Returns
    -------
    ndarray
        Density values; floored at a small positive number so callers get a
        large-but-finite SE rather than a divide-by-zero, in the far tails
        where no data support the quantile.
    """
    values = np.asarray(values, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    if values.size < 2:
        return np.full(points.shape, np.nan)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    # Abadie weights can be negative; density weighting needs the positive part.
    w = np.clip(weights, 0.0, None)
    sw = w.sum()
    if sw <= 0:
        return np.full(points.shape, np.nan)
    w = w / sw

    mean = float(np.sum(w * values))
    var = float(np.sum(w * (values - mean) ** 2))
    sd = float(np.sqrt(max(var, 0.0)))
    if sd <= 0:
        return np.full(points.shape, np.nan)

    if bandwidth is None:
        n_eff = 1.0 / float(np.sum(w**2))  # (sum w)^2 / sum w^2 with sum w = 1
        # Silverman with the IQR guard, matching stats::bw.nrd0 conventions.
        q75, q25 = weighted_quantile(values, np.array([0.75, 0.25]), w)
        iqr = float(q75 - q25)
        scale = min(sd, iqr / 1.349) if iqr > 0 else sd
        bandwidth = 0.9 * scale * n_eff ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return np.full(points.shape, np.nan)

    u = (points[:, None] - values[None, :]) / bandwidth
    dens = (w[None, :] * np.exp(-0.5 * u**2)).sum(axis=1) / (
        bandwidth * np.sqrt(2.0 * np.pi)
    )
    return np.asarray(np.maximum(dens, 1e-12))


# ══════════════════════════════════════════════════════════════════════
#  Monotonicity
# ══════════════════════════════════════════════════════════════════════


def rearrange(curve: np.ndarray) -> np.ndarray:
    """Rearrange a curve into its non-decreasing version.

    Chernozhukov, Fernandez-Val & Galichon (2010): sorting a monotone-in-
    population estimate is a projection that never increases the ``L^p``
    distance to the truth, so rearranging a crossing quantile curve weakly
    improves it.  Used to enforce monotonicity of estimated quantile and
    CDF curves.

    References
    ----------
    chernozhukov2010quantile
    """
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"rearrange expects a 1-D curve, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not finite.any():
        return arr.copy()
    out = arr.copy()
    out[finite] = np.sort(arr[finite])
    return out


# ══════════════════════════════════════════════════════════════════════
#  Uniform (simultaneous) inference over a quantile grid
# ══════════════════════════════════════════════════════════════════════


def multiplier_bootstrap(
    influence: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    weights: str = "rademacher",
) -> np.ndarray:
    """Multiplier bootstrap of a standardised sup-statistic over a grid.

    Parameters
    ----------
    influence : ndarray, shape (n, K)
        Influence-function values: row ``i`` is observation ``i``'s
        contribution at each of the ``K`` grid points, so that
        ``theta_hat_k - theta_k = mean_i(influence[i, k]) + o_p(n^-1/2)``.
    n_boot : int
    seed : int
    weights : {'rademacher', 'gaussian', 'mammen'}
        Multiplier distribution; all have mean 0 and variance 1. Rademacher
        is the default as the most robust to heavy tails.

    Returns
    -------
    ndarray, shape (n_boot,)
        Draws of ``max_k |n^-1/2 sum_i xi_i psi_ik| / sd_k``, whose quantiles
        give uniform critical values.

    Notes
    -----
    Multiplying the influence function rather than resampling the data avoids
    refitting the estimator per replication.
    """
    psi = np.asarray(influence, dtype=float)
    if psi.ndim != 2:
        raise ValueError(f"influence must be 2-D (n, K), got shape {psi.shape}")
    n, _ = psi.shape
    if n < 2:
        return np.full(n_boot, np.nan)

    psi = psi - psi.mean(axis=0, keepdims=True)
    sd = psi.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)

    rng = np.random.default_rng(seed)
    if weights == "rademacher":
        xi = rng.integers(0, 2, size=(n_boot, n)).astype(float) * 2.0 - 1.0
    elif weights == "gaussian":
        xi = rng.standard_normal((n_boot, n))
    elif weights == "mammen":
        p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
        a, b = -(np.sqrt(5.0) - 1.0) / 2.0, (np.sqrt(5.0) + 1.0) / 2.0
        xi = np.where(rng.random((n_boot, n)) < p, a, b)
    else:
        raise ValueError(
            f"weights must be 'rademacher', 'gaussian' or 'mammen', got {weights!r}"
        )

    dev = (xi @ psi) / np.sqrt(n)
    return np.asarray(np.nanmax(np.abs(dev / sd[None, :]), axis=1))


def uniform_band(
    estimate: np.ndarray,
    influence: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Simultaneous confidence band for a curve estimated on a grid.

    A pointwise band covers each ``theta_k`` separately; the probability it
    covers the whole curve at once is much lower. Curve-level claims -- "the
    effect is zero at every quantile", "the effect is constant" -- need this.

    Returns
    -------
    (lower, upper, se, crit) : tuple
        ``crit`` exceeds ``z_{1-alpha/2}``; the ratio is how much wider the
        honest band is.
    """
    est = np.asarray(estimate, dtype=float)
    psi = np.asarray(influence, dtype=float)
    n = psi.shape[0]
    se = psi.std(axis=0, ddof=1) / np.sqrt(n)

    draws = multiplier_bootstrap(psi, n_boot=n_boot, seed=seed)
    good = np.isfinite(draws)
    crit = (
        float("nan")
        if good.sum() < 10
        else float(np.quantile(draws[good], 1.0 - alpha))
    )
    return est - crit * se, est + crit * se, se, crit


def functional_test(
    estimate: np.ndarray,
    influence: np.ndarray,
    null: Optional[np.ndarray] = None,
    kind: str = "ks",
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Test a hypothesis about the whole curve, not one grid point.

    Parameters
    ----------
    estimate, influence
        As in :func:`uniform_band`.
    null : ndarray, optional
        The curve under H0. ``None`` means the zero curve.
    kind : {'ks', 'cvm'}
        Sup-statistic (sharper against a localised departure) or integrated
        squared deviation (sharper against a broad shallow one).

    Returns
    -------
    (statistic, pvalue)
    """
    est = np.asarray(estimate, dtype=float)
    psi = np.asarray(influence, dtype=float)
    n, K = psi.shape
    if null is None:
        null = np.zeros(K)
    dev = est - np.asarray(null, dtype=float)

    psi_c = psi - psi.mean(axis=0, keepdims=True)
    sd = psi_c.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    se = sd / np.sqrt(n)

    rng = np.random.default_rng(seed)
    xi = rng.integers(0, 2, size=(n_boot, n)).astype(float) * 2.0 - 1.0
    boot_dev = (xi @ psi_c) / n

    if kind == "ks":
        stat = float(np.nanmax(np.abs(dev / se)))
        draws = np.nanmax(np.abs(boot_dev / se[None, :]), axis=1)
    elif kind == "cvm":
        stat = float(np.nansum((dev / se) ** 2))
        draws = np.nansum((boot_dev / se[None, :]) ** 2, axis=1)
    else:
        raise ValueError(f"kind must be 'ks' or 'cvm', got {kind!r}")

    good = np.isfinite(draws)
    if good.sum() < 10:
        return stat, float("nan")
    return stat, float(np.mean(draws[good] >= stat))


# ══════════════════════════════════════════════════════════════════════
#  Uniform (simultaneous) inference over a quantile grid
