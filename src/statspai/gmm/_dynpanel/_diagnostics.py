"""Specification diagnostics for dynamic-panel GMM.

Three families, and all three are load-bearing rather than decorative:

* **Arellano-Bond AR(1)/AR(2)**.  First-differencing induces MA(1) by
  construction, so AR(1) *should* reject; AR(2) rejecting is evidence that
  the level errors are serially correlated, which invalidates the lagged
  levels used as instruments.
* **Sargan** (one-step, homoskedastic) and **Hansen J** (two-step,
  heteroskedasticity-robust) over-identification tests.
* **Instrument-count guardrail**.  Instrument proliferation is the dominant
  practical failure mode: with full lag depth the moment count grows as
  O(T²), overfits the endogenous regressor, biases the estimate toward the
  (Nickell-biased) within estimator, and drives the Hansen p-value toward
  1.0 — where it looks *reassuring* while being uninformative.  Roodman
  (2009) recommends keeping the instrument count below the number of units
  and always reporting it.

References
----------
Arellano, M. and Bond, S. (1991). *Review of Economic Studies* 58(2).
[@arellano1991some]
Roodman, D. (2009). How to do xtabond2. *Stata Journal* 9(1), 86-136.
[@roodman2009xtabond]
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

from ._moments import first_difference_H

__all__ = [
    "arellano_bond_ar_test",
    "overid_test",
    "check_instrument_count",
    "difference_in_hansen",
]


def arellano_bond_ar_test(
    resid: np.ndarray,
    unit_rows: Sequence[np.ndarray],
    periods: np.ndarray,
    order: int,
    Z: np.ndarray,
    W: np.ndarray,
    weight: np.ndarray,
    Minv: np.ndarray,
    robust: bool,
    sigma2: float,
    eq_mask: Optional[np.ndarray] = None,
    coef_vcov: Optional[np.ndarray] = None,
    naive_vcov: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Test for AR(``order``) in the first-differenced errors.

    ``m = (q'ê) / sqrt(Var)`` where ``q`` carries, for each row at period
    ``p``, the same unit's residual at ``p - order`` (0 when unavailable).
    The variance uses the influence-function adjustment for the estimated
    coefficients,

        ``c = q − Z A (Z'W) Minv (W'q)``,

    with a clustered outer-product variance ``Σ_i (c_i'ê_i)²`` when
    ``robust``, else the homoskedastic ``σ̂² Σ_i c_i' H_i c_i``.

    ``eq_mask`` restricts which stacked rows may supply a residual pair.
    Under system GMM the test still concerns serial correlation in the
    *first-differenced* errors, so only the transformed rows participate;
    the influence-function adjustment nevertheless runs over the full
    stacked system because that is what the coefficients were estimated on.

    **Coefficient-variance term.**  Expanding the clustered sum shows that
    the third of Arellano & Bond's (1991) three variance terms is exactly
    ``(W'q)' Avar(β̂) (W'q)`` with ``Avar(β̂)`` the *uncorrected* robust
    sandwich.  Whenever the reported VCE is a different one — the two-step
    Windmeijer correction, or the conventional two-step ``(W'ZA₂Z'W)^{-1}``
    — that term has to be swapped for the reported VCE, or the test
    silently uses a variance the estimate does not:

        ``Var = Σ_i (c_i'ê_i)² − (W'q)' V_naive (W'q) + (W'q)' V_reported (W'q)``.

    Pass ``coef_vcov`` (reported) and ``naive_vcov`` (the robust sandwich at
    the final weight and residuals) to enable it; when they coincide — the
    one-step robust case — the swap is identically zero.  Without it, the
    two-step AR statistic on ``abdata`` is -4.32 where Stata reports -3.10.

    Validated to machine precision against Stata's ``estat abond`` for
    one-step robust, one-step classical, two-step conventional and two-step
    Windmeijer-corrected estimation.
    """
    resid = np.asarray(resid, dtype=float)
    q = np.zeros(resid.shape[0])
    for rows in unit_rows:
        use = rows if eq_mask is None else rows[eq_mask[rows]]
        pos = {int(p): int(i) for p, i in zip(periods[use], use)}
        for p, i in pos.items():
            src = pos.get(p - order)
            if src is not None:
                q[i] = resid[src]

    if not np.any(q):
        return {"z": float("nan"), "pvalue": float("nan")}

    num = float(q @ resid)
    u = Minv @ (W.T @ q)
    c = q - Z @ (weight @ ((Z.T @ W) @ u))

    if robust:
        var = float(sum((c[rows] @ resid[rows]) ** 2 for rows in unit_rows))
        if coef_vcov is not None and naive_vcov is not None:
            Wq = W.T @ q
            var += float(Wq @ (np.asarray(coef_vcov) - np.asarray(naive_vcov)) @ Wq)
    else:
        total = 0.0
        for rows in unit_rows:
            use = rows if eq_mask is None else rows[eq_mask[rows]]
            if use.size:
                total += float(c[use] @ first_difference_H(periods[use]) @ c[use])
        var = sigma2 * total

    if not np.isfinite(var) or var <= 0:
        return {"z": float("nan"), "pvalue": float("nan")}
    z = num / np.sqrt(var)
    return {"z": float(z), "pvalue": float(2 * stats.norm.sf(abs(z)))}


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
    from ._estimate import gmm_solve, safe_inv

    Z, W, dy = design.Z, design.W, design.dy
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

        Zr = Z[:, keep]
        with warnings.catch_warnings():
            # A rank-deficient reduced weight matrix is expected for some
            # subsets and is reported through the statistic itself.
            warnings.simplefilter("ignore")
            A = safe_inv(omega[np.ix_(keep, keep)], "reduced moment covariance")
            b, _, _ = gmm_solve(W, Zr, dy, A)
        r = dy - W @ b
        g = Zr.T @ r
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
