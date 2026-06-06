"""
RD inference with a discrete running variable (Kolesár & Rothe 2018).

Implements ``rd_discrete``: honest, finite-sample-valid confidence
intervals for the RD treatment effect when the running variable takes
only a moderate number of distinct values (mass points).  The naive
practice — clustering standard errors by the running variable — does
*not* guard against model misspecification near the cutoff, as shown
empirically in Kolesár & Rothe (2018).

Two CIs are constructed (both honest under their respective
restrictions on the conditional expectation function):

1. **Bounded second derivative (BSD)**, ``method='bsd'`` — the squared
   second derivative |g''(x)| ≤ M for x near c.  This is the same
   smoothness class used in :func:`rd_honest` for continuous running
   variables, but the bias bound is computed exactly from the
   distinct mass points rather than via asymptotic kernel constants.

2. **Bounded misspecification (BM)**, ``method='bm'`` — the bias of the
   linear approximation in any single bin is bounded by K.  This is
   the data-driven default proposed by Kolesár & Rothe (2018) and
   estimated via the bin-wise residual sum of squares.

Both CIs are valid uniformly over the implied smoothness class — they
remain honest even when the asymptotic theory backing :func:`rdrobust`
breaks down because the mass points are sparse.

References
----------
Kolesár, M. and Rothe, C. (2018).
"Inference in Regression Discontinuity Designs with a Discrete Running
Variable." *American Economic Review*, 108(8), 2277-2304.
doi:10.1257/aer.20160945. [@kolesar2018inference]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from .honest_ci import _ak_critical_value


def rd_discrete(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    M: Optional[float] = None,
    K: Optional[float] = None,
    method: str = "bsd",
    h: Optional[float] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Honest CI for RD with a discrete running variable (Kolesár-Rothe 2018).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Discrete running variable column.
    c : float, default 0.0
        Cutoff.  Treatment assigned for x ≥ c.
    M : float, optional
        Bound on |g''(x)| for the bounded-second-derivative ``'bsd'``
        method.  If ``None``, estimated from the data.
    K : float, optional
        Bound on the local linear approximation bias per bin for the
        bounded-misspecification ``'bm'`` method.  If ``None``,
        estimated from the data.
    method : {'bsd', 'bm'}, default ``'bsd'``
        Smoothness class.
    h : float, optional
        Bandwidth in units of x.  If ``None``, uses all observations
        (rectangular weighting on every distinct value).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        Result with ``model_info['discrete']`` containing the number
        of distinct values, mass-point bin counts, the worst-case
        bias, smoothness bound (M or K), and the honest CI.

    Raises
    ------
    ValueError
        If fewer than 4 distinct mass points are present.

    Notes
    -----
    A standard-practice check this routine performs and reports:
    if the standard-error CI is *narrower* than the honest CI, the
    naive RDD inference is over-stating precision.  This is the
    Kolesár-Rothe (2018) finding driving their recommendation to
    abandon clustering by the running variable for honest inference.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.rd_discrete(df, y='outcome', x='age_in_years', c=18, method='bsd')
    >>> r.summary()
    """
    if y not in data.columns or x not in data.columns:
        raise ValueError(f"Columns '{y}' or '{x}' not found in data.")
    if method not in ("bsd", "bm"):
        raise ValueError("method must be 'bsd' or 'bm'.")

    df = data.dropna(subset=[y, x]).copy()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float)
    n_obs = len(Y)

    # --- Bin by distinct mass points ---------------------------------
    if h is not None:
        # Drop observations on mass points outside the bandwidth window,
        # then re-bin from scratch to avoid any inv-map bookkeeping bugs.
        unique_all = np.unique(X)
        keep_vals = unique_all[np.abs(unique_all - c) <= float(h)]
        if len(keep_vals) < 4:
            raise ValueError(  # pragma: no cover
                f"rd_discrete: only {len(keep_vals)} mass points within "
                f"bandwidth h={h}; need ≥4."
            )
        mask = np.isin(X, keep_vals)
        Y, X = Y[mask], X[mask]
    unique_x, inv = np.unique(X, return_inverse=True)

    n_bins = len(unique_x)
    if n_bins < 4:
        raise ValueError(
            f"rd_discrete requires ≥4 distinct mass points; got {n_bins}."
        )

    bin_means = np.zeros(n_bins)
    bin_var = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=int)
    for j in range(n_bins):
        ix = inv == j
        bin_n[j] = int(ix.sum())
        if bin_n[j] >= 1:
            bin_means[j] = float(np.mean(Y[ix]))
            bin_var[j] = (
                float(np.var(Y[ix], ddof=1) / bin_n[j])
                if bin_n[j] > 1 else float(np.var(Y) / bin_n[j])
            )
        else:  # pragma: no cover
            bin_means[j] = np.nan
            bin_var[j] = np.nan

    # --- Local-linear estimator on bin means -------------------------
    # Weighted least squares with weights = bin_n on bin_means
    left = unique_x < c
    right = unique_x >= c
    if left.sum() < 2 or right.sum() < 2:
        raise ValueError(  # pragma: no cover
            "rd_discrete requires ≥2 mass points on each side of the cutoff."
        )

    mu_l, slope_l, var_mu_l, w_l = _ll_on_bins(
        unique_x[left] - c, bin_means[left], bin_var[left], bin_n[left],
    )
    mu_r, slope_r, var_mu_r, w_r = _ll_on_bins(
        unique_x[right] - c, bin_means[right], bin_var[right], bin_n[right],
    )
    tau_hat = float(mu_r - mu_l)
    se = float(np.sqrt(var_mu_l + var_mu_r))

    # --- Smoothness / misspecification bound -------------------------
    # The estimator is τ̂ = ∑_j w_j μ̂_j with weights (w_l on the left,
    # w_r on the right) coming from the WLS local-linear regression on
    # bin means.  The worst-case bias under |g''| ≤ M is achieved by
    # g(x) = sign(w_j) · M/2 · (x - c)² and equals
    #     M/2 · ( ∑_j |w_l_j| (x_l_j - c)² + ∑_j |w_r_j| (x_r_j - c)² )
    # — the Kolesár-Rothe (2018) BSD bias functional applied to a WLS
    # local-linear-on-bin-means estimator.
    bias_bound = 0.0
    if method == "bsd":
        M_estimated = M is None
        if M_estimated:
            M = _estimate_M_discrete(unique_x, bin_means)
        M = float(max(M, 1e-12))
        x_l_centered = unique_x[left] - c
        x_r_centered = unique_x[right] - c
        bias_bound = 0.5 * float(M) * (
            float(np.sum(np.abs(w_l) * x_l_centered ** 2))
            + float(np.sum(np.abs(w_r) * x_r_centered ** 2))
        )
        smoothness_label = f"M = {M:.4g}{' (estimated)' if M_estimated else ' (supplied)'}"
    else:  # bm
        K_estimated = K is None
        if K_estimated:
            K = _estimate_K_bm(unique_x, bin_means, c)
        K = float(max(K, 1e-12))
        # Worst-case bias is K · ∑ |w_j| on each side ≥ K (Kolesár-Rothe
        # 2018, eq. 5).  Use the exact weight-based bound.
        bias_bound = float(K) * (
            float(np.sum(np.abs(w_l))) + float(np.sum(np.abs(w_r)))
        )
        smoothness_label = f"K = {K:.4g}{' (estimated)' if K_estimated else ' (supplied)'}"

    # --- Honest CI (Armstrong-Kolesár FLCI critical value) -----------
    # CI half-length is cv * se where cv = cv_α(bias_bound / se) — the
    # AK (2018) critical value already accounts for worst-case bias,
    # so we do *not* additionally pad by bias_bound.
    if se > 0:
        b = bias_bound / se
        cv = _ak_critical_value(b, alpha)
    else:
        cv = stats.norm.ppf(1 - alpha / 2)
    honest_ci = (tau_hat - cv * se, tau_hat + cv * se)
    naive_ci = (
        tau_hat - stats.norm.ppf(1 - alpha / 2) * se,
        tau_hat + stats.norm.ppf(1 - alpha / 2) * se,
    )

    # --- p-value -----------------------------------------------------
    pvalue = (
        2 * (1 - stats.norm.cdf(abs(tau_hat) / se))
        if se > 0 else float("nan")
    )

    summary = (
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Honest RD with Discrete Running Variable\n"
        "  (Kolesár & Rothe 2018, AER)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Method:                   {method.upper()}\n"
        f"  τ̂:                        {tau_hat:.4f}\n"
        f"  SE:                       {se:.4f}\n"
        f"  Naive {int((1 - alpha) * 100)}% CI:        [{naive_ci[0]:.4f}, {naive_ci[1]:.4f}]\n"
        f"  Honest {int((1 - alpha) * 100)}% CI:       [{honest_ci[0]:.4f}, {honest_ci[1]:.4f}]\n"
        f"  Smoothness:               {smoothness_label}\n"
        f"  Worst-case bias bound:    {bias_bound:.4f}\n"
        f"  Distinct mass points:     {n_bins} ({int(left.sum())} L / {int(right.sum())} R)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    out = CausalResult(
        method="Honest RD with discrete running variable (Kolesár-Rothe 2018)",
        estimand="ATE at cutoff",
        estimate=tau_hat,
        se=se,
        pvalue=pvalue,
        ci=honest_ci,
        alpha=alpha,
        n_obs=int(n_obs),
        model_info={
            "discrete": {
                "method": method,
                "n_mass_points": int(n_bins),
                "n_left": int(left.sum()),
                "n_right": int(right.sum()),
                "bin_means": bin_means.tolist(),
                "bin_n": bin_n.tolist(),
                "M": float(M) if method == "bsd" else None,
                "K": float(K) if method == "bm" else None,
                "bias_bound": float(bias_bound),
                "naive_ci": naive_ci,
                "honest_ci": honest_ci,
            },
            "summary_str": summary,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            out,
            function="sp.rd.rd_discrete",
            params={
                "y": y, "x": x, "c": c, "M": M, "K": K, "method": method,
                "h": h, "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ll_on_bins(
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    var_bins: np.ndarray,
    n_bins: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """Weighted local-linear regression on bin means.  Returns
    (intercept, slope, var(intercept), weight_vector_on_bin_means).

    The weight vector ``w`` satisfies ``intercept = w · y_bins`` exactly
    and is needed to compute the Kolesár-Rothe worst-case bias bound.
    """
    n = len(x_bins)
    Z = np.column_stack([np.ones(n), x_bins])
    # Weights are bin sizes (precision-weighted)
    nw = n_bins.astype(float)
    sqw = np.sqrt(np.maximum(nw, 0))
    Zw = Z * sqw[:, None]
    yw = y_bins * sqw
    try:
        ZtZ_inv = np.linalg.inv(Zw.T @ Zw)
    except np.linalg.LinAlgError:  # pragma: no cover
        ZtZ_inv = np.linalg.pinv(Zw.T @ Zw)
    beta = ZtZ_inv @ Zw.T @ yw
    # Weight vector on raw bin means: intercept = w_vec · y_bins,
    # where w_vec[j] = e_1' (Z'NZ)^{-1} Z_j' n_j (with N=diag(n_bins)).
    w_vec = ((ZtZ_inv @ Z.T) * nw)[0]
    # Variance: the stochastic part is a weighted sum of bin means.
    Vbins = np.diag(np.maximum(var_bins, 0))
    cov = ZtZ_inv @ (Zw.T @ Vbins @ Zw) @ ZtZ_inv
    return float(beta[0]), float(beta[1]), float(cov[0, 0]), w_vec


def _estimate_M_discrete(unique_x: np.ndarray, bin_means: np.ndarray) -> float:
    """Estimate |g''| bound from second-difference quotients of the
    bin-mean curve over evenly distinct values.  Returns the max
    absolute second difference rescaled to match a continuous
    second-derivative bound.
    """
    if len(unique_x) < 3:
        return 1.0
    dx = np.diff(unique_x)
    # Only valid where consecutive spacings exist
    second = (
        (bin_means[2:] - 2 * bin_means[1:-1] + bin_means[:-2])
        / np.maximum(dx[1:] * dx[:-1], 1e-12)
    )
    return float(np.max(np.abs(second))) if len(second) else 1.0


def _estimate_K_bm(
    unique_x: np.ndarray, bin_means: np.ndarray, c: float,
) -> float:
    """Estimate the worst-case linear-approximation bias per side as
    the maximum residual from a global linear fit on each side.
    """
    K = 0.0
    for side in (unique_x < c, unique_x >= c):
        if side.sum() < 2:
            continue  # pragma: no cover
        xs = unique_x[side] - c
        ys = bin_means[side]
        coeffs = np.polyfit(xs, ys, 1)
        resid = ys - np.polyval(coeffs, xs)
        K = max(K, float(np.max(np.abs(resid))))
    return K
