"""Abadie-Imbens (2016) ATT variance for matching on an *estimated* score.

Stata ``teffects psmatch ..., atet vce(robust, nn(h))`` reports the
Abadie-Imbens (2016) variance, which starts from the Abadie-Imbens (2006)
population-ATT variance and corrects it for the fact that the propensity
score was estimated by maximum likelihood.  The formulas implemented here
are the ones printed under "PSM, ATE, and ATET variance adjustment" in
``[CAUSAL] teffects nnmatch`` (Stata 18 manual), which the ``teffects
psmatch`` entry delegates to.  Every symbol below is that manual's.

Writing ``delta`` for the ATT, ``N1`` for the number of matched treated,
``p_i`` for the fitted score, ``f_i = p_i (1 - p_i)`` for the logit
density, ``z_i`` for the treatment-model design row (constant included),
``V_gamma`` for the treatment model's observed-information covariance and
``K_i`` / ``K'_i`` for the matching weight a control receives (the sum of
the ``1/m_j`` shares over the treated that use it, and the sum of their
squares)::

    base   = [ sum_{i treated} (y_i - yhat0_i - delta)^2
               + sum_i xi2_i (K_i^2 - K'_i) ] / N1^2

    c1     = (1/N1) sum_i z_i f_i (ytilde1_i - ytilde0_i - delta)
    c2     = (1/N1) sum_i [ f_i cov(z_i, yhat1_i) + p_i^2 cov(z_i, yhat0_i) ]
    d      = (1/N1) sum_i z_i f_i { (2 t_i - 1)(y_i - ybar^z_i) - delta }

    var    = base - (c1 + c2)' V_gamma (c1 + c2) + d' V_gamma d

where the per-unit ingredients come from four neighbour sets found on the
score:

* ``Psi_h(i)``   -- the ``h`` nearest *same-arm* units **counting unit i
  itself** (distance zero), all ties included; ``xi2_i`` is the ddof-1
  sample variance of the outcome over this set and ``cov(z_i, yhat_{t_i,i})``
  the ddof-1 sample covariance of (design row, outcome) over it.
* ``Psi_h(-i)``  -- the same, excluding unit i; ``ytilde_{t_i,i}`` is its
  mean outcome.
* ``Omega_h(i)`` -- the ``h`` nearest *opposite-arm* units, ties included;
  ``ytilde_{1-t_i,i}`` is its mean outcome and ``cov(z_i, yhat_{1-t_i,i})``
  the sample covariance over it.
* ``Omega^z_m(i)`` -- the ``m`` nearest opposite-arm units by **Euclidean
  distance on the raw treatment-model covariates**, ties included;
  ``ybar^z_i`` is its mean outcome.  (This is the extra clustering the
  manual introduces for ``d(delta)/d(gamma)``; the metric is not stated
  there and was identified numerically -- Mahalanobis and inverse-variance
  scalings are ruled out at rel 4e-4, first-match-only tie handling at
  2e-4, while raw Euclidean with ties reproduces ``e(V)`` at 5e-9.)

Stata's ``nn(h)`` therefore counts the unit itself: ``nn(2)`` (the
default) is one same-arm neighbour plus the unit, which makes ``xi2_i``
equal to the Abadie-Imbens (2006) ``J = 1`` estimate ``(y_i - y_nn)^2 / 2``
whenever the neighbour is unique.  This module takes ``h`` directly, so
callers map StatsPAI's ``ai_matches = J`` to ``h = J + 1``.

Ties are decided by exact floating-point equality of the distance, as in
Stata; the sets are found with the *unclipped* fitted score, because
clipping the score to ``[1e-6, 1 - 1e-6]`` manufactures ties among the
near-zero controls that shift ``c1`` and ``c2`` (rel 1e-6 on the NSW-DW
replica, where 1,758 controls have ``p < 1e-6``).

Every reduction is chunked over rows so memory stays at
``chunk x max(n_treated, n_control)`` floats; the cost is ``O(n * n_arm)``
distance evaluations per arm, which is what the same-arm search costs in
Stata too.

References
----------
abadie2006large, abadie2016matching
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["abadie_imbens_2016_se", "abadie_imbens_2016_ate_se"]


def _tie_inclusive_mask(dist: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` nearest columns per row, ties included.

    A column is selected when its distance is ``<=`` the ``k``-th smallest
    distance in that row, so every unit tied at the boundary comes in.
    ``k`` is clamped to the row length.
    """
    k = int(min(max(k, 1), dist.shape[1]))
    thr = np.partition(dist, k - 1, axis=1)[:, k - 1]
    return dist <= thr[:, None]


def _set_stats(
    mask: np.ndarray,
    y_pool: np.ndarray,
    Z_pool: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (count, mean, ddof-1 variance, ddof-1 cov with Z) over a mask.

    ``cov`` is ``sum_j M_ij (z_j - zbar_i)(y_j - ybar_i) / (n_i - 1)``,
    computed through the identity ``sum_j M_ij z_j (y_j - ybar_i)`` (the
    centring of ``z`` drops out because ``sum_j M_ij (y_j - ybar_i) = 0``).
    Rows with fewer than two members get ``nan`` variance / covariance.
    """
    m = mask.astype(float)
    cnt = m.sum(axis=1)
    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    ybar = (m @ y_pool) / safe_cnt
    resid = (y_pool[None, :] - ybar[:, None]) * m
    denom = np.where(cnt > 1, cnt - 1.0, np.nan)
    var = (resid**2).sum(axis=1) / denom
    cov = (resid @ Z_pool) / denom[:, None]
    ybar = np.where(cnt > 0, ybar, np.nan)
    return cnt, ybar, var, cov


def abadie_imbens_2016_se(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    design: np.ndarray,
    vcov_gamma: np.ndarray,
    covariates: np.ndarray,
    idx_t: np.ndarray,
    idx_c: np.ndarray,
    matches: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    n_matches: int,
    h: int = 2,
    chunk: int = 256,
    density: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    """ATT standard error of Abadie & Imbens (2016) for logit-score matching.

    Reproduces Stata ``teffects psmatch (y) (t x, logit), atet
    nneighbor(m) vce(robust, nn(h))`` -- see the module docstring for the
    formulas and the neighbour-set conventions.

    Parameters
    ----------
    outcome, treated, pscore : ndarray, shape (n,)
        Outcome, 0/1 treatment and the **unclipped** fitted propensity
        score over the estimation sample, in row order.
    design : ndarray, shape (n, k)
        Treatment-model design matrix at which the score was fitted
        (constant column included, any column order).
    vcov_gamma : ndarray, shape (k, k)
        Observed-information covariance of the treatment-model
        coefficients, ``inv(Z' diag(p(1-p)) Z)``, in the same column order
        as ``design``.
    covariates : ndarray, shape (n, q)
        Raw treatment-model covariates (no constant) for the Euclidean
        clustering behind ``d(delta)/d(gamma)``.
    idx_t, idx_c : ndarray
        Row positions of the treated and control units.
    matches, weights : sequence of ndarray
        Per-treated arrays of *positions into* ``idx_c`` and their
        weights (summing to one); an empty array marks an unmatched
        treated unit, which is dropped from the ATT and ``N1`` exactly as
        the point estimate drops it.
    n_matches : int
        The ``m`` of the ATT matching (``nneighbor(m)``); used for the
        covariate-space clustering.
    h : int, default 2
        Stata's ``nn(h)``: the same-arm set is the unit plus ``h - 1``
        nearest same-arm units; the opposite-arm sets for ``c1``/``c2``
        take ``h`` units.  Must be ``>= 2``.
    chunk : int, default 256
        Rows per block in the neighbour searches.
    density : ndarray, shape (n,), optional
        Derivative of the score with respect to the treatment-model index,
        ``f_i``. Omitted, it is the logit's ``p (1 - p)``; for a probit
        score pass the normal density at the index (and the probit's
        ``vcov_gamma``).

    Returns
    -------
    se : float
        ``sqrt(var)``; ``nan`` when ``var`` is negative or undefined.
    components : dict
        ``base_var``, ``base_se``, ``c_V_c``, ``d_V_d``, ``var``, ``n1``,
        ``h``, ``att`` -- the terms of the formula so the number can be
        audited against Stata's ``e(V)`` term by term.

    References
    ----------
    abadie2006large, abadie2016matching
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=int)
    p = np.asarray(pscore, dtype=float)
    Z = np.asarray(design, dtype=float)
    Vg = np.asarray(vcov_gamma, dtype=float)
    Xz = np.asarray(covariates, dtype=float)
    idx_t = np.asarray(idx_t, dtype=int)
    idx_c = np.asarray(idx_c, dtype=int)
    n = len(y)
    h = int(h)
    if h < 2:
        raise ValueError("abadie_imbens_2016_se: h must be >= 2 (Stata nn(#))")
    if Z.shape != (n, Vg.shape[0]) or Vg.shape[0] != Vg.shape[1]:
        raise ValueError("abadie_imbens_2016_se: design / vcov_gamma shapes differ")

    nan_components: Dict[str, Any] = {
        "base_var": float("nan"),
        "base_se": float("nan"),
        "c_V_c": float("nan"),
        "d_V_d": float("nan"),
        "var": float("nan"),
        "n1": 0,
        "h": h,
        "att": float("nan"),
    }
    if len(idx_t) < 2 or len(idx_c) < 2:
        return float("nan"), nan_components

    # ---- matched-set quantities: delta, K, K', yhat0 --------------------
    K = np.zeros(n)
    Kp = np.zeros(n)
    tau_i: List[float] = []
    for pos, (m_idx, w) in enumerate(zip(matches, weights)):
        if len(m_idx) == 0:
            continue
        w = np.asarray(w, dtype=float)
        rows = idx_c[np.asarray(m_idx, dtype=int)]
        tau_i.append(float(y[idx_t[pos]] - np.sum(w * y[rows])))
        np.add.at(K, rows, w)
        np.add.at(Kp, rows, w**2)
    n1 = len(tau_i)
    if n1 < 2:
        return float("nan"), nan_components
    tau = np.asarray(tau_i, dtype=float)
    delta = float(tau.mean())

    # ---- score-space neighbour sets, per arm ------------------------------
    xi2 = np.full(n, np.nan)
    ytil_same = np.full(n, np.nan)
    ytil_opp = np.full(n, np.nan)
    cov_same = np.full((n, Z.shape[1]), np.nan)
    cov_opp = np.full((n, Z.shape[1]), np.nan)
    ybar_z = np.full(n, np.nan)
    arms = {1: idx_t, 0: idx_c}
    for g in (1, 0):
        own = arms[g]
        other = arms[1 - g]
        p_own, y_own, Z_own = p[own], y[own], Z[own]
        p_oth, y_oth, Z_oth = p[other], y[other], Z[other]
        X_oth = Xz[other]
        for start in range(0, len(own), chunk):
            rows = own[start : start + chunk]
            sel = slice(start, start + len(rows))
            # same arm, self included (distance zero)
            d_same = np.abs(p[rows][:, None] - p_own[None, :])
            mask = _tie_inclusive_mask(d_same, h)
            _, _, var_s, cov_s = _set_stats(mask, y_own, Z_own)
            xi2[rows] = var_s
            cov_same[rows] = cov_s
            # same arm, self excluded
            d_minus = d_same.copy()
            d_minus[np.arange(len(rows)), np.arange(start, start + len(rows))] = np.inf
            mask_m = _tie_inclusive_mask(d_minus, h)
            _, ybar_m, _, _ = _set_stats(mask_m, y_own, Z_own)
            ytil_same[rows] = ybar_m
            # opposite arm, h units
            d_opp = np.abs(p[rows][:, None] - p_oth[None, :])
            mask_o = _tie_inclusive_mask(d_opp, h)
            _, ybar_o, _, cov_o = _set_stats(mask_o, y_oth, Z_oth)
            ytil_opp[rows] = ybar_o
            cov_opp[rows] = cov_o
            # opposite arm, m units, Euclidean on the raw covariates
            diff = Xz[rows][:, None, :] - X_oth[None, :, :]
            d_z = np.sqrt((diff**2).sum(axis=2))
            mask_z = _tie_inclusive_mask(d_z, n_matches)
            _, ybar_zz, _, _ = _set_stats(mask_z, y_oth, Z_oth)
            ybar_z[rows] = ybar_zz
            del sel

    # ---- base (Abadie-Imbens 2006 population-ATT) variance ---------------
    reuse = float(np.nansum(xi2 * (K**2 - Kp)))
    hetero = float(np.sum((tau - delta) ** 2))
    base_var = (hetero + reuse) / n1**2

    # ---- estimated-score adjustment (Abadie-Imbens 2016) -----------------
    if density is None:
        f = p * (1.0 - p)
        odds_f = p**2  # f_i p_i / (1 - p_i): finite as p -> 1
    else:
        f = np.asarray(density, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            odds_f = np.where(p < 1.0, f * p / (1.0 - p), 0.0)
    y1til = np.where(t == 1, ytil_same, ytil_opp)
    y0til = np.where(t == 1, ytil_opp, ytil_same)
    cov1 = np.where((t == 1)[:, None], cov_same, cov_opp)
    cov0 = np.where((t == 1)[:, None], cov_opp, cov_same)
    c1 = (Z * (f * (y1til - y0til - delta))[:, None]).sum(axis=0) / n1
    c2 = (f[:, None] * cov1 + odds_f[:, None] * cov0).sum(axis=0) / n1
    d = (Z * (f * ((2 * t - 1) * (y - ybar_z) - delta))[:, None]).sum(axis=0) / n1
    c = c1 + c2
    c_V_c = float(c @ Vg @ c)
    d_V_d = float(d @ Vg @ d)
    var = base_var - c_V_c + d_V_d

    components: Dict[str, Any] = {
        "base_var": float(base_var),
        "base_se": float(np.sqrt(base_var)) if base_var >= 0 else float("nan"),
        "c_V_c": c_V_c,
        "d_V_d": d_V_d,
        "var": float(var),
        "n1": int(n1),
        "h": h,
        "att": delta,
    }
    if not np.isfinite(var) or var < 0:
        return float("nan"), components
    return float(np.sqrt(var)), components


def abadie_imbens_2016_ate_se(
    outcome: np.ndarray,
    treated: np.ndarray,
    pscore: np.ndarray,
    design: np.ndarray,
    vcov_gamma: np.ndarray,
    n_matches: int = 1,
    h: int = 2,
    chunk: int = 256,
    density: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    """ATE standard error of Abadie & Imbens (2016) for score matching.

    Reproduces Stata ``teffects psmatch (y) (t x), ate nneighbor(m)
    vce(robust, nn(h))``. In the notation of the module docstring and of
    "PSM, ATE, and ATET variance adjustment" in ``[CAUSAL] teffects
    nnmatch``, with ``N`` the sample size, ``yhat1_i`` / ``yhat0_i`` the
    imputed potential outcomes (the unit's own outcome in its arm, the mean
    of its ``m`` nearest opposite-arm units in the other) and ``tau`` their
    mean difference::

        base = sum_i [ (yhat1_i - yhat0_i - tau)^2
                       + xi2_i (K_i^2 + 2 K_i - K'_i) ] / N^2

        c    = (1/N) sum_i f_i [ cov(z_i, yhat1_i) / p_i
                                 + cov(z_i, yhat0_i) / (1 - p_i) ]

        var  = base - c' V_gamma c

    ``K_i`` is the matching weight unit ``i`` receives from the opposite
    arm (both arms are matched for the ATE) and ``K'_i`` the sum of the
    squared shares. ``xi2_i`` and ``cov(z_i, yhat_{t_i,i})`` are taken over
    the ``h`` nearest same-arm units counting the unit itself, and
    ``cov(z_i, yhat_{1-t_i,i})`` over the ``h`` nearest opposite-arm units,
    ties included, as for the ATT. There is no ``d(tau)/d(gamma)`` term:
    estimating the score can only lower the variance of the ATE (Abadie
    and Imbens 2016, Theorem 1). The Stata 18 manual prints the adjustment
    with a plus sign; ``e(V)`` is reproduced with the minus sign of the
    paper (five reference fits, logit and probit, to the seven digits
    ``teffects`` prints).

    Parameters are those of :func:`abadie_imbens_2016_se` that apply; the
    matches are found here, on the unclipped score, for both arms.

    Returns
    -------
    se : float
        ``sqrt(var)``; ``nan`` when undefined.
    components : dict
        ``base_var``, ``base_se``, ``c_V_c``, ``var``, ``n``, ``h``, ``ate``.

    References
    ----------
    abadie2006large, abadie2016matching
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=int)
    p = np.asarray(pscore, dtype=float)
    Z = np.asarray(design, dtype=float)
    Vg = np.asarray(vcov_gamma, dtype=float)
    n = len(y)
    h = int(h)
    if h < 2:
        raise ValueError("abadie_imbens_2016_ate_se: h must be >= 2 (Stata nn(#))")
    if Z.shape != (n, Vg.shape[0]) or Vg.shape[0] != Vg.shape[1]:
        raise ValueError("abadie_imbens_2016_ate_se: design / vcov_gamma shapes differ")
    nan_components: Dict[str, Any] = {
        "base_var": float("nan"),
        "base_se": float("nan"),
        "c_V_c": float("nan"),
        "var": float("nan"),
        "n": n,
        "h": h,
        "ate": float("nan"),
    }
    idx_t = np.flatnonzero(t == 1)
    idx_c = np.flatnonzero(t == 0)
    if len(idx_t) < 2 or len(idx_c) < 2:
        return float("nan"), nan_components

    K = np.zeros(n)
    Kp = np.zeros(n)
    y_opp = np.full(n, np.nan)  # mean outcome of the m nearest opposite units
    xi2 = np.full(n, np.nan)
    cov_same = np.full((n, Z.shape[1]), np.nan)
    cov_opp = np.full((n, Z.shape[1]), np.nan)
    arms = {1: idx_t, 0: idx_c}
    for g in (1, 0):
        own = arms[g]
        other = arms[1 - g]
        p_own, y_own, Z_own = p[own], y[own], Z[own]
        p_oth, y_oth, Z_oth = p[other], y[other], Z[other]
        for start in range(0, len(own), chunk):
            rows = own[start : start + chunk]
            # the m nearest opposite-arm units: the imputed outcome, and the
            # weight each of them receives
            d_opp = np.abs(p[rows][:, None] - p_oth[None, :])
            mask_m = _tie_inclusive_mask(d_opp, n_matches)
            cnt, ybar_m, _, _ = _set_stats(mask_m, y_oth, Z_oth)
            y_opp[rows] = ybar_m
            share = mask_m / cnt[:, None]
            np.add.at(K, other, share.sum(axis=0))
            np.add.at(Kp, other, (share**2).sum(axis=0))
            # h nearest opposite-arm units: the covariance of (z, y)
            mask_o = _tie_inclusive_mask(d_opp, h)
            _, _, _, cov_o = _set_stats(mask_o, y_oth, Z_oth)
            cov_opp[rows] = cov_o
            # same arm, self included
            d_same = np.abs(p[rows][:, None] - p_own[None, :])
            mask_s = _tie_inclusive_mask(d_same, h)
            _, _, var_s, cov_s = _set_stats(mask_s, y_own, Z_own)
            xi2[rows] = var_s
            cov_same[rows] = cov_s

    y1 = np.where(t == 1, y, y_opp)
    y0 = np.where(t == 1, y_opp, y)
    tau = float(np.mean(y1 - y0))
    hetero = float(np.sum((y1 - y0 - tau) ** 2))
    reuse = float(np.nansum(xi2 * (K**2 + 2.0 * K - Kp)))
    base_var = (hetero + reuse) / n**2

    f = p * (1.0 - p) if density is None else np.asarray(density, dtype=float)
    cov1 = np.where((t == 1)[:, None], cov_same, cov_opp)
    cov0 = np.where((t == 1)[:, None], cov_opp, cov_same)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = (f / p)[:, None] * cov1 + (f / (1.0 - p))[:, None] * cov0
    c = np.nansum(term, axis=0) / n
    c_V_c = float(c @ Vg @ c)
    var = base_var - c_V_c
    components: Dict[str, Any] = {
        "base_var": float(base_var),
        "base_se": float(np.sqrt(base_var)) if base_var >= 0 else float("nan"),
        "c_V_c": c_V_c,
        "var": float(var),
        "n": int(n),
        "h": h,
        "ate": tau,
    }
    if not np.isfinite(var) or var < 0:
        return float("nan"), components
    return float(np.sqrt(var)), components
