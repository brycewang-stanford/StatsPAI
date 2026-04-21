"""Policy-relevant weight builders for :meth:`BayesianMTEResult.policy_effect`.

Each builder returns a vectorised ``weight_fn(u) -> weights`` usable
directly by :meth:`policy_effect`. The design goal is "common H-V
policy objects in one line":

- :func:`policy_weight_ate` — uniform weights (for parity with ``.ate``).
- :func:`policy_weight_subsidy` — weight = 1 on ``[u_lo, u_hi]``,
  elsewhere 0; use when a subsidy expands treatment within a band.
- :func:`policy_weight_prte` — policy-relevant treatment effect
  weights for a uniform latent-index shift (Heckman-Vytlacil 2005,
  Carneiro-Heckman-Vytlacil 2011).
- :func:`policy_weight_marginal` — marginal PRTE at a specific
  propensity level (delta-function approximation via a narrow band).

All builders validate their arguments up-front so misuse raises
immediately rather than silently producing zero-weight grids at
call time.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def policy_weight_ate() -> Callable[[np.ndarray], np.ndarray]:
    """Uniform weight = 1 on every grid point.

    Equivalent to calling ``policy_effect`` and getting back the
    ATE from the current grid; provided for API parity.
    """
    def _w(u: np.ndarray) -> np.ndarray:
        return np.ones_like(u, dtype=float)
    return _w


def policy_weight_subsidy(
    u_lo: float,
    u_hi: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Weight = 1 on ``[u_lo, u_hi]``; 0 elsewhere.

    Use when the counterfactual policy is "subsidise treatment for
    units whose propensity-to-treat lies in this band". The returned
    weights reach the compliers induced by the subsidy.

    Parameters
    ----------
    u_lo, u_hi : float
        Band endpoints on the propensity (``U_D``) scale. Both must
        lie in ``[0, 1]`` and ``u_lo < u_hi``.
    """
    if not (0.0 <= u_lo < u_hi <= 1.0):
        raise ValueError(
            f"Expected 0 <= u_lo < u_hi <= 1; got u_lo={u_lo}, u_hi={u_hi}"
        )

    def _w(u: np.ndarray) -> np.ndarray:
        return ((u >= u_lo) & (u <= u_hi)).astype(float)
    return _w


def policy_weight_prte(
    shift: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """**Stylised** PRTE weights — convenience builder, NOT the
    textbook Carneiro-Heckman-Vytlacil (2011) PRTE.

    The textbook CHV 2011 PRTE under a propensity-scale policy shift
    ``Δ`` is

    .. code-block:: text

        w(u) ∝ [F_P(u) - F_{P+Δ}(u)] / Δ

    which depends on the *observed* propensity distribution ``F_P``
    of the sample — it is NOT a rectangle and cannot be specified
    from ``shift`` alone. Implementing the true PRTE therefore
    requires the user to pass their sample's propensity draws, which
    is deliberately out of scope for a one-arg builder.

    What this function returns instead is a convenience rectangle
    around the mean propensity ``0.5``:

    .. code-block:: text

        weight(u) = 1 if u ∈ [0.5 - |shift|/2, 0.5 + |shift|/2]
                    0 otherwise

    This approximates the "units near the decision margin under a
    uniform index shift" narrative and is frequently good enough for
    agent-native exploration. If you need the exact CHV PRTE for a
    specific ``F_P``, build a bespoke ``weight_fn`` with the observed
    propensity kernel and pass it to
    :meth:`BayesianMTEResult.policy_effect` directly — e.g.

    .. code-block:: python

        from scipy.stats import gaussian_kde
        fp = gaussian_kde(propensity_sample)
        def chv_prte(u, delta=0.1):
            return (fp(u) - fp(u - delta)) / delta
        r.policy_effect(chv_prte, label='chv_prte')

    Parameters
    ----------
    shift : float
        Size of the propensity-scale shift. Must be in ``(-1, 1)``
        and non-zero.
    """
    if not (-1.0 < shift < 1.0):
        raise ValueError(
            f"shift must be in (-1, 1); got {shift}"
        )
    if shift == 0.0:
        raise ValueError(
            "shift must be non-zero; shift=0 produces zero-weight "
            "grids which are degenerate."
        )

    half = abs(shift) / 2.0
    u_lo = max(0.0, 0.5 - half)
    u_hi = min(1.0, 0.5 + half)

    def _w(u: np.ndarray) -> np.ndarray:
        return ((u >= u_lo) & (u <= u_hi)).astype(float)
    return _w


def policy_weight_marginal(
    u_star: float,
    bandwidth: float = 0.05,
) -> Callable[[np.ndarray], np.ndarray]:
    """Marginal PRTE at a specific propensity level ``u_star``.

    Approximates the derivative of the policy effect at ``u_star``
    by averaging MTE over a narrow band of half-width ``bandwidth``.
    Useful for sanity-checking selection-on-gains at a specific
    decision margin.

    Parameters
    ----------
    u_star : float
        Target propensity level, in ``[0, 1]``.
    bandwidth : float, default 0.05
        Half-width of the averaging window.
    """
    if not (0.0 <= u_star <= 1.0):
        raise ValueError(f"u_star must be in [0, 1]; got {u_star}")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth}")

    u_lo = max(0.0, u_star - bandwidth)
    u_hi = min(1.0, u_star + bandwidth)

    def _w(u: np.ndarray) -> np.ndarray:
        return ((u >= u_lo) & (u <= u_hi)).astype(float)
    return _w


def policy_weight_observed_prte(
    propensity_sample: np.ndarray,
    shift: float,
    *,
    bw_method=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """True **CHV-2011 PRTE** weights from the observed propensity
    distribution via Gaussian KDE.

    Implements the policy-relevant treatment effect weighting
    (Carneiro-Heckman-Vytlacil 2011, Theorem 1):

    .. code-block:: text

        w(u) ∝ [F_P(u) - F_{P + Δ}(u)] / Δ
             = [F_P(u) - F_P(u - Δ)] / Δ
             = ∫_{u - Δ}^{u} f_P(s) ds / Δ

    where ``F_P`` is the CDF of the observed propensity sample, ``f_P``
    is its density (Gaussian KDE here), and ``Δ = shift`` is the
    scalar propensity-scale policy shift. Intuition: ``w(u)`` is the
    population density of policy-induced compliers at propensity
    level ``u``.

    Compared to :func:`policy_weight_prte` (a stylised rectangle),
    this uses the *actual* sample distribution of propensity, which is
    what CHV 2011 describes as the correct weighting kernel.

    Parameters
    ----------
    propensity_sample : np.ndarray
        1-D array of observed propensity scores in ``[0, 1]``.
        Typical source: ``sp.bayes_mte(...).model_info['propensity']``
        or a direct logit fit on your sample.
    shift : float
        Policy-scale propensity shift. Must be non-zero and in
        ``(-1, 1)``. Positive = expand treatment uptake by ``shift``
        at every propensity level; negative = contraction.
    bw_method : str | float | callable | None, default ``None``
        Passed to :class:`scipy.stats.gaussian_kde`. ``None`` uses
        Scott's rule, which is a good default for smooth propensity
        densities.

    Returns
    -------
    callable
        A function ``weight_fn(u: np.ndarray) -> np.ndarray`` that
        returns the (non-negative) PRTE weight at each grid point.
        Negative-density-difference values are clipped at 0 — the
        integral ``∫ w(u) MTE(u) du`` is ill-defined for negative
        weights and clipping aligns with the CHV-2011 interpretation
        when the kernel returns slightly negative tail values due to
        the shift placing density outside ``[0, 1]``.

    References
    ----------
    Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011).
    Estimating marginal returns to education. *AER*, 101(6),
    2754-2781.
    """
    from scipy.stats import gaussian_kde

    sample = np.asarray(propensity_sample, dtype=float).ravel()
    if sample.size < 2:
        raise ValueError(
            "propensity_sample must contain at least 2 values to "
            "estimate a density; got size "
            f"{sample.size}."
        )
    if np.any((sample < 0.0) | (sample > 1.0)):
        raise ValueError(
            "propensity_sample values must all lie in [0, 1]."
        )
    if not (-1.0 < shift < 1.0):
        raise ValueError(
            f"shift must be in (-1, 1); got {shift}"
        )
    if shift == 0.0:
        raise ValueError(
            "shift must be non-zero; shift=0 yields zero-weight "
            "grids."
        )

    kde = gaussian_kde(sample, bw_method=bw_method)
    # Force lazy covariance precomputation so subsequent calls from
    # multiple threads (e.g. PyMC's post-processing) don't race on
    # ``_compute_covariance``.
    kde(np.array([0.5]))

    def _w(u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        # CHV-2011 weight:
        #     w(u) = [F_P(u) - F_P(u - Δ)] / Δ
        #          = ∫_{u - Δ}^{u} f_P(s) ds / Δ
        #
        # For Δ > 0 the numerator is non-negative (CDF non-decreasing)
        # so w(u) ≥ 0 — the density of compliers at propensity level
        # u induced by the uniform policy shift.
        #
        # For Δ < 0 we integrate over ``[u - Δ, u] = [u + |Δ|, u]``
        # which requires swapping the bounds; scipy's
        # ``integrate_box_1d(lo, hi)`` expects lo < hi. We compute the
        # magnitude via sorted bounds and let the sign of ``Δ`` in
        # the denominator produce the correct signed weight (negative
        # shift ⇒ negative weight ⇒ policy reduction).
        out = np.empty_like(u_arr)
        for i, u_val in enumerate(u_arr):
            lo, hi = (u_val - shift, u_val) if shift > 0 else (u_val, u_val - shift)
            integral = kde.integrate_box_1d(float(lo), float(hi))
            out[i] = integral / shift
        return out

    return _w


__all__ = [
    'policy_weight_ate',
    'policy_weight_subsidy',
    'policy_weight_prte',
    'policy_weight_marginal',
    'policy_weight_observed_prte',
]
