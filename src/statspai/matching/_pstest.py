"""Stata ``pstest``-style post-matching balance statistics.

``sp.psmatch2(...).pstest()`` reports the balance table that Stata users
print after ``psmatch2`` (Leuven and Sianesi's ``pstest``), so a ported
analysis can be checked against it. ``sp.psmatch2(...).balance()`` keeps
StatsPAI's own diagnostics (SMD, variance ratio, KS, effective sample size).

Clean-room rewrite (2026-10-20) from the ``pstest`` help file, Rosenbaum and
Rubin (1985) and Rubin (2001); Stata ``pstest`` 4.2.2 was used only as a
black box (``docs/dev/cleanroom/pstest.md``). The statistics, per the help
file and the papers:

* **Standardised % bias** (Rosenbaum and Rubin 1985): ``100 (m1 - m0) /
  sqrt((v1u + v0u) / 2)``. The denominator is the *unmatched* treated and
  control variances, before and after matching, so the two rows are on the
  same scale; ``%reduct`` is the reduction in the absolute bias.
* **Matched moments** use the matching weights as Stata importance
  weights: ``mean = sum(w x) / sum(w)``, ``var = sum(w (x - mean)^2) /
  (sum(w) - 1)``.
* **t-tests** come from a regression of the covariate on the treatment
  indicator: unweighted on the full sample before matching, weighted by
  the matching weights on the on-support matched sample after.
* **Pseudo-R2, LR chi2 and its p-value** come from a probit of the
  treatment on the covariates, fitted by ``pstest`` itself on the full
  sample and, importance-weighted, on the matched sample (not psmatch2's
  propensity score); covariates that are constant or collinear in a sample
  are dropped from that sample's probit and from the LR degrees of freedom.
* **Rubin's B and R** (Rubin 2001) are computed on that probit's linear
  index: ``B = 100 |mean_t - mean_c| / sqrt((var_t + var_c) / 2)``,
  ``R = var_t / var_c`` (matched moments weighted). Rubin's rule of thumb:
  balance is adequate when ``B < 25`` and ``0.5 <= R <= 2``.

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

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_RUBIN_B_MAX = 25.0
_RUBIN_R_RANGE = (0.5, 2.0)
_COLUMNS = [
    "mean_treated_unmatched",
    "mean_control_unmatched",
    "mean_treated_matched",
    "mean_control_matched",
    "pct_bias_unmatched",
    "pct_bias_matched",
    "pct_reduction_abs_bias",
    "t_unmatched",
    "p_unmatched",
    "t_matched",
    "p_matched",
    "variance_ratio_unmatched",
    "variance_ratio_matched",
]


def _iw_mean_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Mean and variance under Stata importance weights.

    The weight total plays the role of the sample size, so the variance is
    ``sum(w (x - mean)^2) / (sum(w) - 1)``.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    total = np.float64(w.sum())
    mean = np.float64(np.sum(w * x) / total)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.float64(np.sum(w * (x - mean) ** 2) / (total - 1.0))
    return mean, var  # NumPy floats: a constant covariate gives NaN, not an error


def _plain_mean_var(x: np.ndarray) -> Tuple[float, float]:
    """Mean and sample variance (divisor ``n - 1``)."""
    return _iw_mean_var(x, np.ones(len(x)))


def _independent_columns(Z: np.ndarray, w: np.ndarray) -> list:
    """Columns of ``Z`` kept left to right, dropping any in the span of the
    ones before it (Stata's omission order for collinear regressors)."""
    zw = Z * np.sqrt(w)[:, None]
    keep: list = []
    for j in range(Z.shape[1]):
        cand = zw[:, keep + [j]]
        if np.linalg.matrix_rank(cand) == len(keep) + 1:
            keep.append(j)
    return keep


def _probit_index_and_fit(
    X: np.ndarray,
    treated: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Probit of ``treated`` on ``[1, X]`` (importance weights ``w``).

    Returns the linear index ``xb``, the pseudo-R2 ``1 - ll / ll0``, the
    likelihood-ratio statistic ``2 (ll - ll0)`` and its chi-square p-value
    with as many degrees of freedom as covariates kept. Maximum likelihood
    by Newton steps on the expected information.
    """
    y = np.asarray(treated, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    Z = np.column_stack(
        [np.ones(len(y)), np.asarray(X, dtype=float).reshape(len(y), -1)]
    )
    Z = Z[:, _independent_columns(Z, w)]

    def loglik(b: np.ndarray) -> float:
        xb = Z @ b
        return float(
            np.sum(w * (y * stats.norm.logcdf(xb) + (1 - y) * stats.norm.logsf(xb)))
        )

    b = np.zeros(Z.shape[1])
    converged = False
    for _ in range(300):
        xb = Z @ b
        log_pdf = stats.norm.logpdf(xb)
        mills_1 = np.exp(log_pdf - stats.norm.logcdf(xb))
        mills_0 = np.exp(log_pdf - stats.norm.logsf(xb))
        score = Z.T @ (w * (y * mills_1 - (1 - y) * mills_0))
        info_w = w * mills_1 * mills_0  # phi^2 / (Phi (1 - Phi))
        step = np.linalg.lstsq(Z.T @ (Z * info_w[:, None]), score, rcond=None)[0]
        b = b + step
        if np.max(np.abs(step)) <= 1e-13 * max(1.0, float(np.max(np.abs(b)))):
            converged = True
            break
    if not converged:
        warnings.warn(
            "pstest: the probit for Rubin's B / R and the pseudo-R2 did not "
            "converge (perfect prediction?); the summary block is unreliable.",
            RuntimeWarning,
            stacklevel=3,
        )
    ll = loglik(b)
    p_bar = float(np.sum(w * y) / np.sum(w))
    ll0 = float(np.sum(w * (y * np.log(p_bar) + (1 - y) * np.log1p(-p_bar))))
    lr = 2.0 * (ll - ll0)
    df = Z.shape[1] - 1
    return Z @ b, 1.0 - ll / ll0, lr, float(stats.chi2.sf(lr, df))


def _weighted_ttest(
    x: np.ndarray,
    treated: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """t and two-sided p of the treatment coefficient in ``x ~ 1 + treated``.

    With importance weights the residual degrees of freedom are
    ``sum(w) - 2``; this is the equal-variance two-sample t statistic.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(treated, dtype=float)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    m1, v1 = _iw_mean_var(x[d == 1], w[d == 1])
    m0, v0 = _iw_mean_var(x[d == 0], w[d == 0])
    w1, w0 = float(w[d == 1].sum()), float(w[d == 0].sum())
    dof = w1 + w0 - 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = ((w1 - 1.0) * v1 + (w0 - 1.0) * v0) / dof
        t = (m1 - m0) / np.sqrt(pooled * (1.0 / w1 + 1.0 / w0))
    return float(t), float(2.0 * stats.t.sf(abs(t), dof))


def _rubin_b_r(index: np.ndarray, d: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    mt, vt = _iw_mean_var(index[d == 1], w[d == 1])
    mc, vc = _iw_mean_var(index[d == 0], w[d == 0])
    return 100.0 * abs(mt - mc) / np.sqrt((vt + vc) / 2.0), vt / vc


def pstest_table(
    data: pd.DataFrame,
    *,
    treat: str,
    covariates: Sequence[str],
    weight_col: str,
    support_col: str,
    pscore_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """``pstest <covariates>, both`` on a psmatch2 matched frame.

    Parameters
    ----------
    data : DataFrame
        The matched frame (input data plus the psmatch2 columns).
    treat : str
        Treatment indicator column.
    covariates : sequence of str
        Covariates to test.
    weight_col, support_col, pscore_col : str
        Names of the ``_weight`` / ``_support`` / ``_pscore`` columns. The
        matched sample is the on-support rows with a positive weight.
        ``pscore_col`` is accepted for symmetry; the Rubin statistics use
        ``pstest``'s own probit, not psmatch2's score.

    Returns
    -------
    (DataFrame, dict)
        The per-covariate table (index ``variable``, covariates in the order
        given) and the sample-level block (``ps_r2``, ``lr_chi2``,
        ``p_chi2``, ``mean_bias``, ``median_bias``, ``rubin_b``,
        ``rubin_r``, ``rubin_balanced``) for ``"unmatched"`` and
        ``"matched"``.
    """
    covs = list(covariates)
    d = data[treat].to_numpy(dtype=float)
    weight = data[weight_col].to_numpy(dtype=float)
    support = data[support_col].to_numpy(dtype=float)
    matched = (support == 1) & np.isfinite(weight) & (weight > 0)
    ones = np.ones(len(d))
    unmatched_w = np.where(matched, weight, 0.0)

    rows = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in covs:
            x = data[name].to_numpy(dtype=float)
            m1u, v1u = _plain_mean_var(x[d == 1])
            m0u, v0u = _plain_mean_var(x[d == 0])
            t1, t0 = matched & (d == 1), matched & (d == 0)
            m1m, v1m = _iw_mean_var(x[t1], weight[t1])
            m0m, v0m = _iw_mean_var(x[t0], weight[t0])
            scale = np.sqrt((v1u + v0u) / 2.0)
            bias_u = 100.0 * (m1u - m0u) / scale
            bias_m = 100.0 * (m1m - m0m) / scale
            t_u, p_u = _weighted_ttest(x, d)
            t_m, p_m = _weighted_ttest(x[matched], d[matched], weight[matched])
            rows[name] = [
                m1u, m0u, m1m, m0m, bias_u, bias_m,
                -100.0 * (abs(bias_m) - abs(bias_u)) / abs(bias_u),
                t_u, p_u, t_m, p_m, v1u / v0u, v1m / v0m,
            ]  # fmt: skip
    table = pd.DataFrame.from_dict(rows, orient="index", columns=_COLUMNS)
    table.index.name = "variable"

    summary: Dict[str, Any] = {}
    X = data[covs].to_numpy(dtype=float)
    for label, mask, w in (
        ("unmatched", np.ones(len(d), dtype=bool), ones),
        ("matched", matched, unmatched_w),
    ):
        index, r2, lr, p = _probit_index_and_fit(X[mask], d[mask], w[mask])
        b, r = _rubin_b_r(index, d[mask], w[mask])
        abs_bias = np.abs(table[f"pct_bias_{label}"].to_numpy(dtype=float))
        abs_bias = abs_bias[np.isfinite(abs_bias)]
        summary[label] = {
            "ps_r2": float(r2),
            "lr_chi2": float(lr),
            "p_chi2": float(p),
            "mean_bias": float(np.mean(abs_bias)) if abs_bias.size else float("nan"),
            "median_bias": (
                float(np.median(abs_bias)) if abs_bias.size else float("nan")
            ),
            "rubin_b": float(b),
            "rubin_r": float(r),
            "rubin_balanced": bool(
                b < _RUBIN_B_MAX and _RUBIN_R_RANGE[0] <= r <= _RUBIN_R_RANGE[1]
            ),
        }
    return table, summary
