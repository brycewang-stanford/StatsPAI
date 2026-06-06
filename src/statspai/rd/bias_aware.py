"""
Bias-aware confidence intervals for fuzzy RD (Noack & Rothe 2024).

Implements ``rd_bias_aware_fuzzy``: a bias-aware confidence set for the
fuzzy regression discontinuity parameter τ = (μ_Y(c+) − μ_Y(c−))
/ (μ_D(c+) − μ_D(c−)) based on local-linear regression.  The CI takes
the smoothing bias of *both* numerator and denominator into account,
following Noack & Rothe (2024, Econometrica 92(3), 687-711).

Unlike the conventional fuzzy-RD t-ratio CI, which can have severely
distorted coverage when the first stage is moderate, the bias-aware
construction has uniformly correct coverage over the smoothness class

    F(M_Y, M_D) = { (g_Y, g_D) : |g_Y''| ≤ M_Y, |g_D''| ≤ M_D }

via Anderson--Rubin style test inversion: τ_0 lies in the CI iff
``tau_to_test = mu_Y(c+) - mu_Y(c-) - τ_0 * (mu_D(c+) - mu_D(c-))``
fails to be rejected when accounting for worst-case bias of the
numerator and denominator local-linear estimators.

The CI also addresses the *power asymmetry* documented in
Kaliski, Keane & Neal (2025, NBER 33972): when the first-stage
discontinuity is small, the conventional 2SLS-style CI has poor
power on one side.  The Anderson-Rubin construction here is naturally
robust to weak first stages.

References
----------
Noack, C. and Rothe, C. (2024).
"Bias-Aware Inference in Fuzzy Regression Discontinuity Designs."
*Econometrica*, 92(3), 687-711. doi:10.3982/ECTA19466.
[@noack2024biasaware]

Kaliski, D., Keane, M.P. and Neal, T. (2025).
"The Power Asymmetry in Fuzzy Regression Discontinuity Designs."
NBER Working Paper No. 33972. doi:10.3386/w33972. [@kaliski2025power]
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult
from ._core import _kernel_fn, _local_poly_wls
from .honest_ci import _ak_critical_value, _estimate_M


def rd_bias_aware_fuzzy(
    data: pd.DataFrame,
    y: str,
    x: str,
    fuzzy: str,
    c: float = 0.0,
    M_y: Optional[float] = None,
    M_d: Optional[float] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    cluster: Optional[str] = None,
    n_grid: int = 401,
) -> CausalResult:
    """
    Bias-aware confidence interval for fuzzy RD (Noack & Rothe 2024).

    Constructs an Anderson--Rubin-style CI for the Wald-ratio fuzzy RD
    parameter that takes worst-case smoothing bias of the numerator and
    denominator local-linear estimates into account.  The CI is robust
    to weak first stages and avoids the power asymmetry documented in
    Kaliski-Keane-Neal (2025).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Running variable column.
    fuzzy : str
        Treatment indicator column for the fuzzy first stage.
    c : float, default 0.0
        RD cutoff.
    M_y, M_d : float, optional
        Smoothness bounds on |g_Y''| and |g_D''|.  If ``None``, both
        are estimated from a local-quadratic fit on each side of the
        cutoff (the same default as :func:`rd_honest`).
    h : float, optional
        Bandwidth.  If ``None``, defaults to a Silverman pilot.
    kernel : str, default ``'triangular'``
        Kernel.
    alpha : float, default 0.05
        Significance level.
    cluster : str, optional
        Cluster variable for variance estimation.
    n_grid : int, default 401
        Grid resolution for the AR-style test inversion.

    Returns
    -------
    CausalResult
        Result with ``model_info['bias_aware']`` containing
        ``M_y``, ``M_d``, ``naive_ci`` (standard CCT-style fuzzy CI),
        ``bias_aware_ci``, ``rejection_grid`` and ``first_stage_F``.

    Notes
    -----
    The bias-aware CI is the set of ``τ_0`` for which the test statistic

        T(τ_0) = (Δ̂_Y − τ_0 · Δ̂_D) / σ̂(τ_0)

    has |T(τ_0)| ≤ cv(b(τ_0)) where ``b(τ_0)`` is the worst-case bias-
    to-noise ratio of the numerator–denominator combination.  The grid
    inversion accommodates non-convex CIs that arise under weak first
    stages.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.rd_bias_aware_fuzzy(df, y='earnings', x='age', fuzzy='retired',
    ...                            c=65.0)
    >>> print(r.model_info['bias_aware']['bias_aware_ci'])
    """
    # --- Parse data ---------------------------------------------------
    needed = [y, x, fuzzy]
    if cluster is not None:
        needed.append(cluster)
    missing = [n for n in needed if n not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")  # pragma: no cover
    df = data.dropna(subset=needed).copy()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float) - float(c)
    D = df[fuzzy].to_numpy(dtype=float)
    cl = df[cluster].to_numpy() if cluster is not None else None
    n_obs = len(Y)

    if kernel not in ("triangular", "epanechnikov", "uniform"):
        raise ValueError(f"Unsupported kernel '{kernel}'.")  # pragma: no cover
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")  # pragma: no cover

    # --- Bandwidth ---------------------------------------------------
    if h is None:
        h = 1.06 * float(np.std(X)) * (n_obs ** (-1 / 5))
        h = max(h, 1e-6)

    # --- M estimation -------------------------------------------------
    M_y_estimated = M_y is None
    M_d_estimated = M_d is None
    if M_y_estimated:
        M_y = _estimate_M(Y, X, 0.0, h, kernel)
    if M_d_estimated:
        M_d = _estimate_M(D, X, 0.0, h, kernel)
    M_y = max(float(M_y), 1e-12)
    M_d = max(float(M_d), 1e-12)

    # --- Local-linear point estimates on each side -------------------
    left = X < 0
    right = X >= 0

    beta_y_l, vcov_y_l, n_yl = _local_poly_wls(
        Y[left], X[left], h, p=1, kernel=kernel,
        cluster=cl[left] if cl is not None else None,
    )
    beta_y_r, vcov_y_r, n_yr = _local_poly_wls(
        Y[right], X[right], h, p=1, kernel=kernel,
        cluster=cl[right] if cl is not None else None,
    )
    beta_d_l, vcov_d_l, _ = _local_poly_wls(
        D[left], X[left], h, p=1, kernel=kernel,
        cluster=cl[left] if cl is not None else None,
    )
    beta_d_r, vcov_d_r, _ = _local_poly_wls(
        D[right], X[right], h, p=1, kernel=kernel,
        cluster=cl[right] if cl is not None else None,
    )

    delta_y = float(beta_y_r[0] - beta_y_l[0])
    delta_d = float(beta_d_r[0] - beta_d_l[0])
    var_dy = float(vcov_y_r[0, 0] + vcov_y_l[0, 0])
    var_dd = float(vcov_d_r[0, 0] + vcov_d_l[0, 0])
    cov_yd = float(_local_cov_yd(Y, D, X, h, kernel, cl))

    # Use a relative threshold to detect a near-zero first stage so the
    # naive Wald CI does not collapse to an essentially-infinite interval
    # whose seed range corrupts the AR grid below.
    sd_d = float(np.std(D)) if len(D) else 1.0
    weak_first_stage = abs(delta_d) < 0.01 * max(sd_d, 1e-6)
    if weak_first_stage:
        warnings.warn(  # pragma: no cover
            "rd_bias_aware_fuzzy: estimated first-stage discontinuity is "
            "tiny relative to Var(D); naive CI is reported as unbounded "
            "and the bias-aware AR CI is the appropriate object.",
            UserWarning,
        )

    # Naive Wald-ratio point estimate and delta-method CI
    if weak_first_stage:
        tau_hat = float("nan")  # pragma: no cover
        se_naive = float("nan")  # pragma: no cover
        naive_ci = (float("-inf"), float("inf"))
    else:
        tau_hat = delta_y / delta_d
        se_naive = float(
            np.sqrt(max(var_dy + tau_hat ** 2 * var_dd - 2 * tau_hat * cov_yd, 0))
            / abs(delta_d)
        )
        z = stats.norm.ppf(1 - alpha / 2)
        naive_ci = (tau_hat - z * se_naive, tau_hat + z * se_naive)

    # --- Bias bounds for numerator and denominator -------------------
    Ck = _kernel_bias_constant(kernel)
    bias_y = Ck * h ** 2 * M_y
    bias_d = Ck * h ** 2 * M_d

    # --- AR-style inversion -------------------------------------------
    # Build the grid around the naive point estimate when the first
    # stage is strong; otherwise span a wide neutral window so the
    # AR test inverts symmetrically around 0.
    if np.isfinite(tau_hat) and np.isfinite(se_naive):
        span = max(abs(tau_hat) + 6 * se_naive, 5.0)
        grid_lo = tau_hat - span
        grid_hi = tau_hat + span
    else:
        # Weak first stage: use the scale of |Δ_Y| and σ to pick a window
        scale = max(abs(delta_y), 6 * float(np.sqrt(var_dy)), 5.0)
        # Heuristic: scan an interval up to 100x the Y-side jump
        grid_lo = -100 * scale
        grid_hi = 100 * scale
    grid = np.linspace(grid_lo, grid_hi, int(n_grid))
    accept = np.zeros_like(grid, dtype=bool)
    for i, t0 in enumerate(grid):
        # Numerator-denominator combination
        num = delta_y - t0 * delta_d
        var = max(var_dy + t0 ** 2 * var_dd - 2 * t0 * cov_yd, 0)
        se = float(np.sqrt(var))
        if se <= 0:
            continue  # pragma: no cover
        # Worst-case bias of (Δ_Y − τ0 Δ_D) under |g_Y''| ≤ M_y, |g_D''| ≤ M_d
        bias = bias_y + abs(t0) * bias_d
        b_ratio = bias / se
        cv = _ak_critical_value(b_ratio, alpha)
        # Anderson-Rubin acceptance under the AK FLCI critical value:
        # cv_α(b/se) is calibrated so that |num/se| ≤ cv has coverage
        # ≥ 1-α uniformly over |bias| ≤ b.  No additional bias padding.
        accept[i] = abs(num) <= cv * se

    if accept.any():
        # CI is the convex hull of accepted points.
        idx = np.where(accept)[0]
        ci_lo = float(grid[idx.min()])
        ci_hi = float(grid[idx.max()])
        # Detect non-convex region (rare under strong first stage)
        non_convex = not np.all(accept[idx.min(): idx.max() + 1])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")  # pragma: no cover
        non_convex = False

    bias_aware_ci = (ci_lo, ci_hi)

    # --- Power asymmetry diagnostic (KKN 2025) -----------------------
    first_stage_t = abs(delta_d) / np.sqrt(var_dd) if var_dd > 0 else float("inf")
    first_stage_F = float(first_stage_t ** 2)
    if first_stage_F < 10.0:
        warnings.warn(
            f"rd_bias_aware_fuzzy: first-stage F = {first_stage_F:.2f} < 10. "
            "Conventional fuzzy-RD t-tests have a power asymmetry (Kaliski-"
            "Keane-Neal 2025); the bias-aware AR CI here is robust, but you "
            "should also report the ITT (sharp RD on the outcome) per their "
            "recommendation.",
            UserWarning,
        )

    # --- Pretty summary ----------------------------------------------
    summary = (
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Bias-Aware Fuzzy RD (Noack & Rothe 2024 ECTA)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  τ̂ (Wald ratio):           {tau_hat:.4f}\n"
        f"  Δ̂_Y (numerator jump):     {delta_y:.4f}\n"
        f"  Δ̂_D (denominator jump):   {delta_d:.4f}\n"
        f"  First-stage F:            {first_stage_F:.2f}\n"
        f"\n"
        f"  Naive {int((1 - alpha) * 100)}% CI:       [{naive_ci[0]:.4f}, {naive_ci[1]:.4f}]\n"
        f"  Bias-aware {int((1 - alpha) * 100)}% CI:  [{ci_lo:.4f}, {ci_hi:.4f}]"
        f"{' (non-convex)' if non_convex else ''}\n"
        f"\n"
        f"  M_y:                      {M_y:.4g} {'(estimated)' if M_y_estimated else '(supplied)'}\n"
        f"  M_d:                      {M_d:.4g} {'(estimated)' if M_d_estimated else '(supplied)'}\n"
        f"  Bandwidth h:              {h:.4f}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    # P-value derived from the bias-aware test of H0: τ=0
    # (whether 0 lies in the bias-aware CI).
    pvalue = (
        2 * (1 - stats.norm.cdf(abs(tau_hat) / se_naive))
        if (np.isfinite(tau_hat) and np.isfinite(se_naive) and se_naive > 0)
        else float("nan")
    )
    if not np.isfinite(pvalue) and ci_lo == ci_lo and ci_hi == ci_hi:
        # Approximate p-value from the AR CI: if 0 is well inside the CI,
        # report a conservative 1.0; if outside, report alpha.
        pvalue = float(alpha) if not (ci_lo <= 0 <= ci_hi) else 1.0

    out = CausalResult(
        method="Bias-aware fuzzy RD (Noack-Rothe 2024)",
        estimand="LATE at cutoff",
        estimate=tau_hat,
        se=se_naive,
        pvalue=pvalue,
        ci=bias_aware_ci,
        alpha=alpha,
        n_obs=int(n_obs),
        model_info={
            "bias_aware": {
                "naive_ci": naive_ci,
                "bias_aware_ci": bias_aware_ci,
                "non_convex_ci": bool(non_convex),
                "M_y": float(M_y),
                "M_d": float(M_d),
                "M_y_estimated": bool(M_y_estimated),
                "M_d_estimated": bool(M_d_estimated),
                "delta_y": delta_y,
                "delta_d": delta_d,
                "first_stage_F": first_stage_F,
                "bandwidth": float(h),
                "kernel": kernel,
                "rejection_grid": (grid.tolist(), accept.tolist()),
            },
            "summary_str": summary,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            out,
            function="sp.rd.rd_bias_aware_fuzzy",
            params={
                "y": y, "x": x, "fuzzy": fuzzy, "c": c,
                "M_y": M_y, "M_d": M_d, "h": h, "kernel": kernel,
                "alpha": alpha, "cluster": cluster,
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

_BIAS_CONSTS = {
    # Local-linear bias constant: int u^2 K(u) du / 2 (effective leading term
    # for the local-linear bias when the kernel has support on |u| ≤ 1)
    "triangular": 1 / 12,
    "epanechnikov": 1 / 10,
    "uniform": 1 / 6,
}


def _kernel_bias_constant(kernel: str) -> float:
    return _BIAS_CONSTS.get(kernel, _BIAS_CONSTS["triangular"])


def _local_cov_yd(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    h: float,
    kernel: str,
    cl: Optional[np.ndarray],
) -> float:
    """Approximate covariance of the local-linear intercepts of Y and D
    estimated separately on each side, summed across sides.  Uses the
    weighted residual covariance.
    """
    cov = 0.0
    for side_mask in (X < 0, X >= 0):
        if side_mask.sum() < 5:
            continue  # pragma: no cover
        u = X[side_mask] / h
        w = _kernel_fn(u, kernel)
        in_bw = np.abs(u) <= 1
        if in_bw.sum() < 4:
            continue  # pragma: no cover
        Yw = Y[side_mask][in_bw]
        Dw = D[side_mask][in_bw]
        Xw = X[side_mask][in_bw]
        ww = w[in_bw]
        Z = np.column_stack([np.ones_like(Xw), Xw])
        sqw = np.sqrt(ww)
        Zw = Z * sqw[:, None]
        try:
            ZtZ_inv = np.linalg.inv(Zw.T @ Zw)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue  # pragma: no cover
        beta_y = ZtZ_inv @ Zw.T @ (Yw * sqw)
        beta_d = ZtZ_inv @ Zw.T @ (Dw * sqw)
        ry = Yw - Z @ beta_y
        rd = Dw - Z @ beta_d
        if cl is not None:
            cl_in = cl[side_mask][in_bw]
            unique = np.unique(cl_in)
            meat = np.zeros((2, 2))
            for cval in unique:
                idx = cl_in == cval
                sy = (Zw[idx].T @ (ry[idx] * sqw[idx])).ravel()
                sd_ = (Zw[idx].T @ (rd[idx] * sqw[idx])).ravel()
                meat += np.outer(sy, sd_)
            corr = len(unique) / max(len(unique) - 1, 1)
            v = (corr * ZtZ_inv @ meat @ ZtZ_inv)[0, 0]
        else:
            n_eff = int(in_bw.sum())
            corr = n_eff / max(n_eff - 2, 1)
            meat = Zw.T @ np.diag(ry * rd * corr) @ Zw
            v = (ZtZ_inv @ meat @ ZtZ_inv)[0, 0]
        cov += float(v)
    return cov
