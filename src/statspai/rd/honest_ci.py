"""
Honest Confidence Intervals for Regression Discontinuity
=========================================================

Implements the Armstrong & Kolesár (2018, 2020) honest confidence intervals
for RD designs. Standard RD CIs can have poor coverage because they ignore
smoothing bias. Honest CIs are valid uniformly over a class of regression
functions characterised by a bound M on the second derivative.

References
----------
Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of
    regression models. Econometrica, 86(2), 655-683.
Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence
    intervals in nonparametric regression. Quantitative Economics, 11(1), 1-39. [@armstrong2018optimal]
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #

# The kernels rd_honest accepts. Only the keys are used: the worst-case bias
# is computed in rd/_rdhonest.py from the realised kernel weights.
_KERNEL_BIAS_CONSTANTS = {
    "triangular": 1.0 / 6.0,
    "epanechnikov": 1.0 / 5.0,
    "uniform": 1.0 / 3.0,
}


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #


def _honest_pvalue(tau: float, se: float, bias: float) -> float:
    """Smallest ``alpha`` at which the honest CI excludes zero.

    Solved by bisection on ``|tau| = cv_{1-alpha}(bias/se) * se``. Reporting
    the naive ``2*(1 - Phi(|tau|/se))`` here would ignore the very bias the
    rest of this function exists to bound, so a result could show an honest
    CI containing zero next to a p-value below 0.05.
    """
    from ._rdhonest import cv_bias

    t = abs(tau) / se
    b = bias / se
    if cv_bias(b, 1 - 1e-12) >= t:
        return 1.0
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cv_bias(b, mid) > t:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def rd_honest(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    M: Optional[float] = None,
    kernel: str = "triangular",
    h: Optional[float] = None,
    alpha: float = 0.05,
    opt_criterion: str = "mse",
    sclass: str = "H",
    cluster: Optional[str] = None,
) -> CausalResult:
    """
    Honest confidence intervals for regression discontinuity designs.

    Implements Armstrong & Kolesár (2018, 2020): CIs that are valid uniformly
    over the class of regression functions whose second derivative is bounded
    by *M*.  These "honest" CIs account for the smoothing bias that standard
    local-polynomial CIs ignore, yielding correct coverage even in finite
    samples.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff.
    M : float, optional
        Upper bound on |f''(c)|. If ``None``, estimated from a local
        quadratic fit on each side of the cutoff.
    kernel : str, default "triangular"
        Kernel function: ``"triangular"``, ``"epanechnikov"``, or
        ``"uniform"``.
    h : float, optional
        Bandwidth. If ``None``, chosen by the criterion in *opt_criterion*.
    alpha : float, default 0.05
        Significance level for the confidence interval.
    opt_criterion : str, default "mse"
        Bandwidth selection criterion when *h* is ``None``:
        ``"mse"`` (MSE-optimal, Imbens-Kalyanaraman style),
        ``"flci"`` (minimises honest CI length), or
        ``"oci"`` (one-sided CI length).
    sclass : str, default "H"
        Smoothness class the bound *M* constrains, matching ``RDHonest``.
        ``"H"`` (Holder — ``f'`` is ``M``-Lipschitz) is the default and lets
        the two sides' curvature contributions cancel. ``"T"`` (Taylor) only
        requires ``|f(x) - f(0) - f'(0)x| <= M x^2 / 2`` on each side, which
        permits no cancellation, so its worst-case bias — and hence the
        interval — is always at least as wide for the same *M*.
    cluster : str, optional
        Cluster column, as ``RDHonest(clusterid=)``: the standard error sums
        ``w_i e_i`` within clusters, and the bandwidth search uses the
        preliminary variance plus a Moulton within-cluster correlation.

    Returns
    -------
    CausalResult
        Result object with ``model_info`` containing:
        - ``honest_ci`` : tuple – honest confidence interval
        - ``naive_ci``  : tuple – standard CI for comparison
        - ``M``         : float – smoothness bound used
        - ``bias_bound``: float – estimated worst-case bias
        - ``bandwidth`` : float – bandwidth used

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.5 * x + 2.0 * (x >= 0) + rng.normal(0, 0.4, n)
    >>> df = pd.DataFrame({'y': y, 'x': x})
    >>> result = sp.rd_honest(df, y='y', x='x', c=0)
    >>> bool(abs(result.estimate - 2.0) < 0.3)  # true jump is 2.0
    True
    >>> lo, hi = result.model_info['honest_ci']
    >>> nlo, nhi = result.model_info['naive_ci']
    >>> bool((hi - lo) > (nhi - nlo))  # honest CI widens for worst-case bias
    True

    References
    ----------
    Armstrong, T. B. and Kolesár, M. (2018). Optimal Inference in a Class of
    Regression Models. *Econometrica*. [@armstrong2018optimal]

    Armstrong, T. B. and Kolesár, M. (2020). Simple and honest confidence
    intervals in nonparametric regression. *Quantitative Economics*.
    [@armstrong2020simple]
    """
    kernel = kernel.lower()
    if kernel not in _KERNEL_BIAS_CONSTANTS:
        raise ValueError(  # pragma: no cover
            f"Unknown kernel '{kernel}'. Choose from {list(_KERNEL_BIAS_CONSTANTS)}"
        )
    opt_criterion = opt_criterion.lower()
    if opt_criterion not in ("mse", "flci", "oci"):
        raise ValueError(
            "opt_criterion must be 'mse', 'flci' or 'oci', got " f"{opt_criterion!r}"
        )
    if str(sclass).upper() not in ("H", "T", "HOLDER", "TAYLOR"):
        raise ValueError(f"sclass must be 'H' (Holder) or 'T' (Taylor), got {sclass!r}")

    df = data.dropna(subset=[y, x] + ([cluster] if cluster else []))
    y_arr = df[y].values.astype(float)
    x_arr = df[x].values.astype(float)
    cl_arr = df[cluster].to_numpy() if cluster else None
    n_obs = len(y_arr)

    # Everything below delegates to rd/_rdhonest.py, a port of R RDHonest.
    # The previous implementation got the point estimate right but built the
    # interval from two approximations that are not Armstrong-Kolesar's:
    #
    #   * a closed-form bias ``M h^2 C_k`` instead of the worst-case bias of
    #     the estimator actually computed, which depends on the realised
    #     kernel weights and so cannot be a function of ``h`` alone; and
    #   * ``tau +/- (cv * se + bias)``, which DOUBLE-COUNTS the bias --
    #     ``cv = cv_{1-alpha}(bias/se)`` already accounts for it.
    #
    # Together those made intervals up to ~2x wider than RDHonest's. Wider is
    # not "safer" here: it is a different, less informative procedure being
    # reported under Armstrong-Kolesar's name.
    from ._rdhonest import honest_rd

    sclass_code = "H" if str(sclass).upper().startswith("H") else "T"
    fit, M_estimated = honest_rd(
        x_arr,
        y_arr,
        c=c,
        M=M,
        h=h,
        kernel=kernel,
        alpha=alpha,
        opt_criterion=opt_criterion.upper(),
        sclass=sclass_code,
        cluster=cl_arr,
    )
    tau_hat = fit["estimate"]
    se = fit["se"]
    M_value = fit["M"]
    h_value = fit["bandwidth"]
    bias_bound = fit["bias"]
    cv = fit["cv"]
    b = bias_bound / se if se > 0 else 0.0
    honest_ci = (fit["ci_lower"], fit["ci_upper"])

    z_naive = stats.norm.ppf(1.0 - alpha / 2.0)
    naive_ci = (tau_hat - z_naive * se, tau_hat + z_naive * se)

    n_l = int(np.sum((x_arr < c) & (np.abs(x_arr - c) <= h_value)))
    n_r = int(np.sum((x_arr >= c) & (np.abs(x_arr - c) <= h_value)))

    # Honest p-value: invert the honest CI rather than reporting the naive
    # z-test, which ignores the bias the rest of this function exists to
    # bound. Reject at level alpha iff 0 lies outside the honest interval.
    if se > 0:
        pvalue = _honest_pvalue(tau_hat, se, bias_bound)
    else:
        pvalue = np.nan  # pragma: no cover

    # ---- Build summary string ---- #
    M_label = (
        f"{M_value:.4g} (estimated)" if M_estimated else f"{M_value:.4g} (supplied)"
    )
    summary_str = (
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Honest CI for RD (Armstrong & Kolesar, 2020)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  RD estimate:          {tau_hat:.4f}\n"
        f"  Standard SE:          {se:.4f}\n"
        f"  Naive {int((1 - alpha) * 100)}% CI:        [{naive_ci[0]:.4f}, "
        f"{naive_ci[1]:.4f}]\n"
        f"\n"
        f"  Honest {int((1 - alpha) * 100)}% CI:       [{honest_ci[0]:.4f}, "
        f"{honest_ci[1]:.4f}]\n"
        f"  Smoothness bound M:   {M_label}\n"
        f"  Bias bound:           {bias_bound:.4f}\n"
        f"  Bandwidth:            {h_value:.4f}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    _result = CausalResult(
        method="Honest CI for RD (Armstrong & Kolesar, 2020)",
        estimand="LATE",
        estimate=tau_hat,
        se=se,
        pvalue=pvalue,
        ci=honest_ci,
        alpha=alpha,
        n_obs=n_obs,
        model_info={
            "honest_ci": honest_ci,
            "naive_ci": naive_ci,
            "M": M_value,
            "M_estimated": M_estimated,
            "bias_bound": bias_bound,
            "bandwidth": h_value,
            "kernel": kernel,
            "cutoff": c,
            "opt_criterion": opt_criterion,
            "n_left": int(n_l),
            "n_right": int(n_r),
            "ak_critical_value": cv,
            "bias_noise_ratio": b,
            "sclass": sclass_code,
            "cluster": cluster,
            "n_clusters": None if cluster is None else int(pd.unique(cl_arr).size),
            "eff_obs": fit["eff_obs"],
            "reference_backend": "RDHonest",
            "validation_tier": "T2_native_reference_parity",
            "summary_str": summary_str,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.rd.rd_honest",
            params={
                "y": y,
                "x": x,
                "c": c,
                "M": M_value,
                "kernel": kernel,
                "h": h_value,
                "alpha": alpha,
                "opt_criterion": opt_criterion,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
