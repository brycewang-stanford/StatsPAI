"""Specification diagnostics for dynamic-panel GMM.

* **Arellano-Bond serial-correlation tests** (:func:`serial_correlation_test`):
  the AR(l) test of Arellano and Bond (1991) on the first-differenced
  residuals. Clean-room implementation (2026-10-20) from the paper and from
  Roodman (2009, Section 3.5), with Stata's ``xtabond`` / ``xtabond2`` used
  only as black boxes (their outputs on our data). Specification:
  ``docs/dev/cleanroom/dynpanel_ar.md``.
* **Over-identification** (:func:`overid_test`): Sargan and Hansen J.
* **Difference-in-Hansen** (:func:`difference_in_hansen`) and the
  instrument-count advisory (:func:`check_instrument_count`).

References
----------
[@arellano1991some]; [@roodman2009xtabond].
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "serial_correlation_test",
    "overid_test",
    "check_instrument_count",
    "difference_in_hansen",
]


def overid_test(
    Z: np.ndarray, resid: np.ndarray, weight: np.ndarray, df: int, scale: float = 1.0
) -> Dict[str, float]:
    """Generic over-identification statistic ``g' A g / scale``.

    With ``weight = A₁`` and ``scale = σ̂²`` this is the Sargan statistic
    (valid under homoskedasticity); with ``weight = A₂`` and ``scale = 1``
    it is the heteroskedasticity-robust Hansen J.
    """
    if df <= 0:
        return {"stat": float("nan"), "df": int(df), "pvalue": float("nan")}
    g = Z.T @ resid
    stat = float(g @ weight @ g / scale)
    return {
        "stat": stat,
        "df": int(df),
        "pvalue": float(stats.chi2.sf(stat, df)),
    }


def check_instrument_count(
    n_instruments: int, n_units: int, stacklevel: int = 3
) -> Optional[str]:
    """Warn when the moment count rivals or exceeds the number of units.

    Returns the warning message (also emitted as a ``UserWarning``) or
    ``None``.  The message is stored in ``model_info`` so an agent reading
    the result object sees the same caveat a human reads off the console.
    """
    if n_units <= 0 or n_instruments <= 0:
        return None
    if n_instruments < n_units:
        return None
    msg = (
        f"{n_instruments} instruments for {n_units} units. With at least as "
        "many moment conditions as cross-sectional units the two-step weight "
        "matrix is (near-)singular, the Hansen test loses power — its p-value "
        "is pushed toward 1.0, which looks reassuring but is not — and the "
        "estimate is biased toward the within-groups estimator. Reduce the "
        "instrument set with collapse=True or a tighter gmm_lags window "
        "(Roodman 2009, Stata Journal 9(1), Sec. 5)."
    )
    warnings.warn(msg, stacklevel=stacklevel)
    return msg


def difference_in_hansen(
    design,
    hansen_full: float,
    n_params: int,
    omega: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Difference-in-Hansen (C) tests for each instrument subset.

    For a subset ``S`` of moment conditions, refit the model *without*
    ``S``, take its Hansen J, and difference:

        ``C = J_full − J_excluding``   with   ``df = |S|``,

    which is asymptotically chi-squared under the null that ``S`` is
    exogenous.

    The restricted fit must reuse the **full** model's moment covariance —
    ``A_restricted = (Ω[keep, keep])^{-1}`` rather than a freshly estimated
    ``Ω`` on the reduced instrument set.  That is what guarantees the two J
    statistics are on a common scale and therefore that ``C ≥ 0`` in
    population (Hayashi 2000; Baum, Schaffer & Stillman 2003).  Re-estimating
    ``Ω`` instead is a natural-looking mistake that inflates the statistic
    badly — on ``abdata`` it turns xtabond2's 5.32 (p = 0.62, level
    instruments look fine) into 18.63 (p = 0.009, level instruments look
    rejected), i.e. it reverses the conclusion.

    This is the only way to interrogate the extra assumptions a moment set
    brings — most importantly system GMM's level moments, whose validity
    requires that each unit's deviation from its long-run mean be
    uncorrelated with the fixed effect. Reporting a system-GMM estimate
    without it is reporting an untested identifying assumption.

    Matches ``xtabond2``'s "Difference-in-Hansen tests of exogeneity of
    instrument subsets" block.

    Notes
    -----
    The statistic can come out negative in finite samples when the
    excluded-instrument weight matrix is ill-conditioned; ``xtabond2``
    warns about the same thing.  Negative values are reported as-is rather
    than clipped — silently flooring at zero would hide the conditioning
    problem that produced them.
    """
    from ._estimate import safe_inv

    Z, W, dy = design.Z, design.W, design.dy

    # Everything the restricted fit needs is a *submatrix* of three
    # cross-products, so the big design matrices are touched once rather
    # than sliced and re-solved per subset.
    ZW = Z.T @ W
    Zy = Z.T @ dy
    m = Z.shape[1]
    out: Dict[str, Dict[str, float]] = {}

    for name, cols in design.instrument_groups.items():
        cols = np.asarray(cols, dtype=int)
        keep = np.ones(m, dtype=bool)
        keep[cols] = False
        n_keep = int(keep.sum())
        if n_keep <= n_params:
            # Dropping the subset would leave the model under-identified, so
            # the test is undefined rather than zero. Say so.
            out[name] = {
                "hansen_excluding": float("nan"),
                "df_excluding": 0,
                "statistic": float("nan"),
                "df": int(cols.size),
                "pvalue": float("nan"),
                "note": "under-identified without this subset",
            }
            continue

        ZWk, Zyk = ZW[keep], Zy[keep]
        with warnings.catch_warnings():
            # A rank-deficient reduced weight matrix is expected for some
            # subsets and is reported through the statistic itself.
            warnings.simplefilter("ignore")
            A = safe_inv(omega[np.ix_(keep, keep)], "reduced moment covariance")
            b = safe_inv(ZWk.T @ (A @ ZWk), "reduced moment matrix") @ (
                ZWk.T @ (A @ Zyk)
            )
        g = Zyk - ZWk @ b
        j_excl = float(g @ A @ g)
        stat = float(hansen_full - j_excl)
        df = int(cols.size)
        out[name] = {
            "hansen_excluding": j_excl,
            "df_excluding": n_keep - n_params,
            "statistic": stat,
            "df": df,
            "pvalue": float(stats.chi2.sf(stat, df)) if stat > 0 else float("nan"),
        }
    return out


def _lag_index(units: np.ndarray, periods: np.ndarray, use: np.ndarray, order: int):
    """Row of the same unit's residual ``order`` periods earlier, or -1.

    Only rows flagged in ``use`` (first-differenced equations) count, on
    both ends of the pair.
    """
    where = {
        (u, p): r
        for r, (u, p, ok) in enumerate(zip(units.tolist(), periods.tolist(), use))
        if ok
    }
    lag = np.full(units.size, -1, dtype=np.int64)
    for r in np.flatnonzero(use):
        lag[r] = where.get((int(units[r]), int(periods[r]) - order), -1)
    return lag


def _group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``values`` (1-D or 2-D) within each group code."""
    shape = (n_groups,) + values.shape[1:]
    out = np.zeros(shape)
    np.add.at(out, codes, values)
    return out


def serial_correlation_test(
    order: int,
    *,
    resid_test: np.ndarray,
    X_test: np.ndarray,
    units_test: np.ndarray,
    periods_test: np.ndarray,
    use_test: np.ndarray,
    groups_test: np.ndarray,
    Z: np.ndarray,
    resid_est: np.ndarray,
    groups_est: np.ndarray,
    XZ: np.ndarray,
    weight: np.ndarray,
    Minv: np.ndarray,
    coef_vcov: np.ndarray,
    n_groups: int,
    classical_blocks: Optional[Sequence[tuple]] = None,
) -> Dict[str, float]:
    """Arellano-Bond (1991) test for AR(``order``) in first-differenced errors.

    With ``e`` the first-differenced residuals of the test rows and ``q``
    the same unit's residual ``order`` periods earlier (zero when that row
    does not exist), the statistic is ``z = q'e / sqrt(V)`` with (Arellano
    and Bond 1991; Roodman 2009, Section 3.5)

    ``V = S_qq - 2 q'X M^{-1} X'Z A S_Zq + q'X Avar(b) X'q``,

    ``M = X'Z A Z'X`` the estimator's moment matrix, ``Avar(b)`` the
    reported coefficient covariance (robust or not, Windmeijer-corrected or
    not), and the "meat" terms

    * robust: ``S_qq = sum_g (q_g' e_g)^2`` and
      ``S_Zq = sum_g Z_g' u_g (e_g' q_g)``, summed over the groups ``g``,
      ``u`` the estimation residuals;
    * classical (one-step, non-robust): ``S_qq = sum_i q_i' C_tt,i q_i`` and
      ``S_Zq = sum_i Z_i' C_et,i q_i`` with ``C`` the error covariances
      implied by i.i.d. level errors of variance ``s^2``; each entry of
      ``classical_blocks`` is ``(est_rows, test_rows, C_et, C_tt)`` for one
      unit.

    The test basis (``*_test``: always first differences) may differ from
    the estimation basis (``Z``, ``resid_est``: e.g. forward orthogonal
    deviations); ``groups_*`` give each row's group code in
    ``0..n_groups-1`` so that the two bases line up.

    Returns ``{'z', 'pvalue'}`` (two-sided normal p-value); both are NaN
    when no lagged pair exists or the variance is not positive.
    """
    use = np.asarray(use_test, dtype=bool)
    e = np.where(use, np.asarray(resid_test, dtype=float), 0.0)
    lag = _lag_index(np.asarray(units_test), np.asarray(periods_test), use, order)
    has = lag >= 0
    q = np.zeros_like(e)
    q[has] = e[lag[has]]
    if not has.any():
        return {"z": float("nan"), "pvalue": float("nan")}

    numer = float(q @ e)
    qX = q @ X_test
    proj = qX @ Minv @ XZ @ weight
    if classical_blocks is None:
        per_group = _group_sums(q * e, groups_test, n_groups)
        s_qq = float(per_group @ per_group)
        zu = _group_sums(
            Z * np.asarray(resid_est, dtype=float)[:, None], groups_est, n_groups
        )
        s_zq = zu.T @ per_group
    else:
        s_qq = 0.0
        s_zq = np.zeros(Z.shape[1])
        for est_rows, test_rows, c_et, c_tt in classical_blocks:
            qt = q[test_rows]
            s_qq += float(qt @ c_tt @ qt)
            s_zq += Z[est_rows].T @ (c_et @ qt)
    var = s_qq - 2.0 * float(proj @ s_zq) + float(qX @ coef_vcov @ qX)
    if not var > 0:
        return {"z": float("nan"), "pvalue": float("nan")}
    z = numer / np.sqrt(var)
    return {"z": float(z), "pvalue": float(2.0 * stats.norm.sf(abs(z)))}
