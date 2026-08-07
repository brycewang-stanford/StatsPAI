"""Stata ``pstest``-faithful post-matching balance statistics.

``sp.psmatch2(...).balance()`` reports StatsPAI's own balance diagnostics
(SMD, variance ratio, KS, effective sample size).  Those are the right
numbers, but they are not *Stata's* numbers, and an analyst checking a
ported result against a printed ``pstest`` table needs the latter.

This module reproduces ``pstest`` exactly, following ``pstest.ado`` v4.2.2
(Leuven & Sianesi 2003).  The conventions that are easy to get wrong:

* **The standardised bias after matching keeps the *unmatched* denominator.**
  Both rows divide by ``sqrt((v1u + v0u)/2)`` computed on the raw sample, so
  the "after" bias is comparable to the "before" bias rather than being
  rescaled by whatever matching did to the variances::

      bias_before = 100 (m1u - m0u) / sqrt((v1u + v0u)/2)
      bias_after  = 100 (m1m - m0m) / sqrt((v1u + v0u)/2)
                                      ^^^^^^^^^^^^^^^^^^ unmatched

* **The matched moments use Stata importance weights.**  ``summarize x
  [iw=w]`` gives ``mean = Σwx/Σw`` and ``Var = Σw(x-mean)²/(Σw - 1)`` — the
  denominator is the weight *total* minus one, not the row count minus one.

* **Rubin's B and R are computed on pstest's own probit index, not on
  psmatch2's propensity score.**  ``pstest`` fits ``probit treated <x>``
  itself — twice, once unweighted on the full sample and once
  importance-weighted on the matched sample — and uses each fit's linear
  predictor ``xb``.  Reusing ``_pscore`` (or its logit) instead gets Rubin's
  B wrong by several percent and the pseudo-R² wrong outright, because
  psmatch2's score may come from a *logit* while pstest always probits.

* ``%reduct`` is the reduction in *absolute* bias, so a covariate whose bias
  flips sign but shrinks still shows a positive reduction.

Rubin's (2001) rules of thumb, which ``pstest`` stars: balance is adequate
when ``B < 25`` and ``R`` lies in ``[0.5, 2]``.

References
----------
Leuven, E. and Sianesi, B. (2003). PSMATCH2: Stata module to perform full
    Mahalanobis and propensity score matching, common support graphing, and
    covariate imbalance testing.  Statistical Software Components S432001,
    Boston College Department of Economics.
Rosenbaum, P.R. and Rubin, D.B. (1985). The American Statistician, 39(1),
    33-38.  [``rosenbaum1985constructing``]
Rubin, D.B. (2001). Health Services and Outcomes Research Methodology,
    2(3-4), 169-188.  [``rubin2001using``]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _stats

#: Rubin (2001): B below this (in %) is considered balanced.
RUBIN_B_THRESHOLD = 25.0
#: Rubin (2001): R inside this interval is considered balanced.
RUBIN_R_BOUNDS = (0.5, 2.0)


def _iw_mean_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Stata ``summarize x [iw=w]`` mean and variance.

    With importance weights Stata treats ``Σw`` as the sample size, so the
    variance divides by ``Σw - 1`` rather than ``n - 1``.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w != 0)
    x, w = x[keep], w[keep]
    sw = float(w.sum())
    if sw <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(w * x) / sw)
    if sw <= 1:
        return mean, float("nan")
    var = float(np.sum(w * (x - mean) ** 2) / (sw - 1.0))
    return mean, var


def _plain_mean_var(x: np.ndarray) -> Tuple[float, float]:
    """Stata ``summarize x`` mean and (ddof=1) variance."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan")
    return float(np.mean(x)), float(np.var(x, ddof=1))


def _probit_index_and_fit(
    X: np.ndarray,
    treated: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Fit ``probit treated X [iw=w]`` and return (index, pseudo-R2, chi2, p).

    ``pstest`` does **not** reuse psmatch2's propensity score.  It fits its
    own **probit** — twice, once on the full sample for the "before" row and
    once on the matched sample (importance-weighted) for the "after" row —
    and takes the linear predictor ``xb`` of each as the index that Rubin's
    B and R are computed on.  Using the logit of ``_pscore`` instead gets
    Rubin's B wrong by several percent, and the pseudo-R² wrong outright.

    The likelihood is Stata's Bernoulli one,
    ``ll = Σ w [y log Φ(xb) + (1-y) log(1-Φ(xb))]``, so that
    ``e(r2_p) = 1 - ll/ll0`` agrees digit for digit.
    """
    nan = float("nan")
    empty = np.full(len(treated), nan)
    try:
        import statsmodels.api as sm
    except ImportError:  # pragma: no cover - statsmodels is a core dependency
        return empty, nan, nan, nan

    y = np.asarray(treated, dtype=float)
    Xd = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    weights = None if w is None else np.asarray(w, dtype=float)
    fam = sm.families.Binomial(link=sm.families.links.Probit())
    # A probit on a matched sample can be genuinely unfittable -- a singular
    # design, or perfect separation once the weights concentrate on a
    # sub-sample. Those are expected and reported as missing, exactly as
    # Stata leaves e(r2_p) missing. Anything else is a bug and must surface.
    fit_errors: tuple = (np.linalg.LinAlgError, ValueError, ZeroDivisionError)
    try:  # pragma: no cover - statsmodels always ships this symbol
        from statsmodels.tools.sm_exceptions import PerfectSeparationError

        fit_errors = fit_errors + (PerfectSeparationError,)
    except ImportError:
        pass
    try:
        full = sm.GLM(y, Xd, family=fam, freq_weights=weights).fit()
        null = sm.GLM(y, Xd[:, :1], family=fam, freq_weights=weights).fit()
    except fit_errors:
        return empty, nan, nan, nan

    def _ll(fit: Any) -> float:
        p = np.clip(np.asarray(fit.fittedvalues, dtype=float), 1e-300, 1 - 1e-16)
        contrib = y * np.log(p) + (1.0 - y) * np.log1p(-p)
        if weights is not None:
            contrib = weights * contrib
        return float(np.sum(contrib))

    ll, ll0 = _ll(full), _ll(null)
    if not np.isfinite(ll) or not np.isfinite(ll0) or ll0 == 0:
        return empty, nan, nan, nan

    index = np.asarray(Xd @ full.params, dtype=float)
    r2 = 1.0 - ll / ll0
    chi2 = 2.0 * (ll - ll0)
    dof = Xd.shape[1] - 1
    p_chi2 = float(_stats.chi2.sf(chi2, dof)) if dof > 0 else nan
    return index, float(r2), float(chi2), p_chi2


def _weighted_ttest(
    x: np.ndarray,
    treated: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Two-sample t on the (optionally weighted) group means.

    pstest runs ``regress x treated`` (weighted after matching) and reports
    that coefficient's t; for a two-group regression this is the equal-variance
    two-sample t statistic, which is what we form here.
    """
    t_mask = treated == 1
    c_mask = treated == 0
    if w is None:
        m1, v1 = _plain_mean_var(x[t_mask])
        m0, v0 = _plain_mean_var(x[c_mask])
        n1 = float(np.sum(np.isfinite(x[t_mask])))
        n0 = float(np.sum(np.isfinite(x[c_mask])))
    else:
        m1, v1 = _iw_mean_var(x[t_mask], w[t_mask])
        m0, v0 = _iw_mean_var(x[c_mask], w[c_mask])
        n1 = float(np.nansum(w[t_mask]))
        n0 = float(np.nansum(w[c_mask]))
    if not np.isfinite(v1) or not np.isfinite(v0) or n1 + n0 <= 2:
        return float("nan"), float("nan")
    dof = n1 + n0 - 2.0
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / dof
    denom = np.sqrt(pooled * (1.0 / n1 + 1.0 / n0))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan"), float("nan")
    tstat = (m1 - m0) / denom
    pval = float(2 * _stats.t.sf(abs(tstat), dof))
    return float(tstat), pval


def pstest_table(
    data: pd.DataFrame,
    *,
    treat: str,
    covariates: Sequence[str],
    weight_col: str,
    support_col: str,
    pscore_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reproduce ``pstest <covariates>, both``.

    Parameters
    ----------
    data : DataFrame
        The matched frame (input data plus the psmatch2 columns).
    treat : str
        Treatment indicator column.
    covariates : sequence of str
        Covariates to test.
    weight_col, support_col, pscore_col : str
        Names of the ``_weight`` / ``_support`` / ``_pscore`` columns.
        ``pscore_col`` is accepted for symmetry and recorded in the result;
        the Rubin statistics deliberately do **not** use it (pstest refits
        its own probit -- see :func:`_probit_index_and_fit`).

    Returns
    -------
    (DataFrame, dict)
        The per-covariate table (one row per covariate, in the order given)
        and the sample-level summary block (``Ps R2``, ``LR chi2``,
        ``p>chi2``, ``MeanBias``, ``MedBias``, ``B``, ``R``) for the
        unmatched and matched samples.
    """
    treated = data[treat].to_numpy(dtype=float)
    support = data[support_col].to_numpy(dtype=float)
    raw_w = data[weight_col].to_numpy(dtype=float)

    # The matched sample: on support with a non-missing matching weight.
    # pstest's `[iw=_weight] if _support==1` drops everything else.
    m_ok = np.isfinite(raw_w) & (support == 1)
    w = np.where(m_ok, raw_w, 0.0)

    t_u = treated == 1
    c_u = treated == 0
    t_m = t_u & m_ok
    c_m = c_u & m_ok

    rows: List[Dict[str, Any]] = []
    for name in covariates:
        x = data[name].to_numpy(dtype=float)
        m1u, v1u = _plain_mean_var(x[t_u])
        m0u, v0u = _plain_mean_var(x[c_u])
        m1m, v1m = _iw_mean_var(x[t_m], w[t_m])
        m0m, v0m = _iw_mean_var(x[c_m], w[c_m])

        pooled_sd = np.sqrt((v1u + v0u) / 2.0)
        bias_u = 100.0 * (m1u - m0u) / pooled_sd if pooled_sd > 0 else np.nan
        bias_m = 100.0 * (m1m - m0m) / pooled_sd if pooled_sd > 0 else np.nan
        reduct = (
            -100.0 * (abs(bias_m) - abs(bias_u)) / abs(bias_u)
            if np.isfinite(bias_u) and abs(bias_u) > 0
            else np.nan
        )

        t_before, p_before = _weighted_ttest(x, treated, None)
        # The matched t uses only the matched rows, weighted.
        x_m = np.where(m_ok, x, np.nan)
        t_after, p_after = _weighted_ttest(x_m, treated, w)

        rows.append(
            {
                "variable": name,
                "mean_treated_unmatched": m1u,
                "mean_control_unmatched": m0u,
                "mean_treated_matched": m1m,
                "mean_control_matched": m0m,
                "pct_bias_unmatched": bias_u,
                "pct_bias_matched": bias_m,
                "pct_reduction_abs_bias": reduct,
                "t_unmatched": t_before,
                "p_unmatched": p_before,
                "t_matched": t_after,
                "p_matched": p_after,
                "variance_ratio_unmatched": (
                    v1u / v0u if np.isfinite(v0u) and v0u > 0 else np.nan
                ),
                "variance_ratio_matched": (
                    v1m / v0m if np.isfinite(v0m) and v0m > 0 else np.nan
                ),
            }
        )

    table = pd.DataFrame(rows).set_index("variable")

    # --- sample-level block ------------------------------------------
    # pstest fits its own probit twice, and Rubin's B / R are computed on
    # each fit's linear predictor -- not on psmatch2's propensity score.
    X = data[list(covariates)].to_numpy(dtype=float)
    index_u, r2_u, chi2_u, p_u = _probit_index_and_fit(X, treated, None)
    index_m_sub, r2_m, chi2_m, p_m = _probit_index_and_fit(
        X[m_ok], treated[m_ok], w[m_ok]
    )
    index_m = np.full(len(treated), np.nan)
    index_m[m_ok] = index_m_sub

    mi1u, vi1u = _plain_mean_var(index_u[t_u])
    mi0u, vi0u = _plain_mean_var(index_u[c_u])
    mi1m, vi1m = _iw_mean_var(index_m[t_m], w[t_m])
    mi0m, vi0m = _iw_mean_var(index_m[c_m], w[c_m])

    def _rubin_b(m1: float, v1: float, m0: float, v0: float) -> float:
        sd = np.sqrt((v1 + v0) / 2.0)
        return float(100.0 * (m1 - m0) / sd) if sd > 0 else float("nan")

    abs_bias_u = table["pct_bias_unmatched"].abs()
    abs_bias_m = table["pct_bias_matched"].abs()

    summary: Dict[str, Any] = {
        "unmatched": {
            "ps_r2": r2_u,
            "lr_chi2": chi2_u,
            "p_chi2": p_u,
            "mean_bias": float(abs_bias_u.mean()),
            "median_bias": float(abs_bias_u.median()),
            "rubin_b": _rubin_b(mi1u, vi1u, mi0u, vi0u),
            "rubin_r": float(vi1u / vi0u) if vi0u > 0 else float("nan"),
        },
        "matched": {
            "ps_r2": r2_m,
            "lr_chi2": chi2_m,
            "p_chi2": p_m,
            "mean_bias": float(abs_bias_m.mean()),
            "median_bias": float(abs_bias_m.median()),
            "rubin_b": _rubin_b(mi1m, vi1m, mi0m, vi0m),
            "rubin_r": float(vi1m / vi0m) if vi0m > 0 else float("nan"),
        },
    }
    for key in ("unmatched", "matched"):
        b = summary[key]["rubin_b"]
        r = summary[key]["rubin_r"]
        summary[key]["rubin_balanced"] = bool(
            np.isfinite(b)
            and np.isfinite(r)
            and abs(b) < RUBIN_B_THRESHOLD
            and RUBIN_R_BOUNDS[0] <= r <= RUBIN_R_BOUNDS[1]
        )
    return table, summary
