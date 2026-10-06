"""
Tobit (1958) censored regression model.

For outcomes censored at a lower and/or upper limit (e.g., wages
observed only if employed, expenditure ≥ 0).

    Y_i* = X_i'β + ε_i,    ε_i ~ N(0, σ²)
    Y_i  = max(Y_i*, L)     (left-censored at L)

References
----------
Tobin, J. (1958).
"Estimation of Relationships for Limited Dependent Variables."
*Econometrica*, 26(1), 24-36. [@tobin1958estimation]

Amemiya, T. (1984).
"Tobit Models: A Survey."
*Journal of Econometrics*, 24(1-2), 3-61. [@amemiya1984tobit]
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases, accepts_formula_first
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._limited_dep_result import LimitedDepResult
from ._optim_helpers import robust_convergence


@accepts_formula_first()
@accepts_aliases(robust="vce", covariates="x")
def tobit(
    data: pd.DataFrame,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    ll: Optional[float] = 0,
    ul: Optional[float] = None,
    alpha: float = 0.05,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    method: str = "mle",
    *,
    formula: Optional[str] = None,
) -> CausalResult:
    """
    Tobit (Type I) censored regression via MLE.

    Equivalent to Stata's ``tobit y x, ll(0)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Censored outcome variable.
    x : list of str
        Regressors.
    method : {'mle', 'scls'}, default 'mle'
        ``'scls'`` is Powell's symmetrically censored least squares. It
        needs only a symmetric error around ``x'b`` and stays consistent
        under heteroskedasticity and non-normality, where the maximum
        likelihood estimator does not. It uses only observations with
        ``x'b`` above the limit, so it is less precise when the model is
        right, reports no ``sigma``, and is defined for a lower limit
        only. Its standard errors are Powell's sandwich (cluster-robust
        with ``cluster=``). Use it when :func:`cmtest` rejects.
    ll : float or None, default 0
        Lower censoring limit. Observations with Y ≤ ll are censored.
        ``None`` means no lower limit (Stata's ``tobit y x, ul(#)``); the
        default of 0 is this function's, not Stata's, where a limit applies
        only when it is written.
        Set to ``-np.inf`` for no lower censoring.
    ul : float, optional
        Upper censoring limit. Default: no upper censoring.
    alpha : float, default 0.05
    vce : str, optional
        Standard errors: ``None`` / ``'oim'`` (observed information, Stata's
        default), ``'robust'`` (``vce(robust)``, with Stata's ``N/(N-1)``)
        or ``'cluster'``; ``vce="cluster firm"`` also works. ``robust=`` is
        accepted as an alias.
    cluster : str, optional
        Cluster column (Stata ``vce(cluster c)``, factor ``G/(G-1)``).
    weights : str, optional
        Sampling-weight column (Stata ``[pw=]``): the log-likelihood is
        weighted and, as in Stata, the standard errors are robust.
    formula : str, optional
        ``"y ~ x1 + I(x1**2) + C(g)"`` in place of ``y=`` and ``x=``.
        Transformed, factor and interaction terms are built as columns
        and named as ``sp.regress`` names them.

    Returns
    -------
    CausalResult
        MLE coefficients, sigma, and marginal effects.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> wage = rng.uniform(5, 25, n)
    >>> education = rng.integers(8, 18, n)
    >>> children = rng.integers(0, 4, n)
    >>> latent = (-20 + 1.5 * wage + 2.0 * education
    ...           - 5.0 * children + rng.normal(0, 8, n))
    >>> df = pd.DataFrame({
    ...     'hours': np.maximum(latent, 0),  # weekly hours, censored at 0
    ...     'wage': wage, 'education': education, 'children': children,
    ... })
    >>> result = sp.tobit(df, y='hours',
    ...                   x=['wage', 'education', 'children'], ll=0)
    >>> print(result.summary())  # doctest: +SKIP

    Notes
    -----
    The Tobit log-likelihood for left-censoring at L:

    .. math::
        \\ell = \\sum_{y_i > L} \\left[
            -\\frac{1}{2}\\log(2\\pi\\sigma^2)
            - \\frac{(y_i - x_i'\\beta)^2}{2\\sigma^2}
        \\right]
        + \\sum_{y_i = L} \\log \\Phi\\left(
            \\frac{L - x_i'\\beta}{\\sigma}
        \\right)

    **Marginal effects**: The coefficient β does NOT directly give the
    marginal effect on E[Y|X]. The marginal effect on the observed
    (uncensored) mean is β × Φ(X'β/σ).

    See Tobin (1958, *Econometrica*).
    """
    from ..core._vcov_spec import parse_se_request

    if ll is None:
        ll = -np.inf  # no lower limit
    se_req = parse_se_request(
        vce,
        cluster,
        function="tobit",
        supported=("nonrobust", "robust", "cluster"),
    )
    se_kind, cluster = se_req.kind, se_req.cluster
    if weights is not None and se_kind == "nonrobust":
        # Stata: pweights imply vce(robust); the OIM variance is not valid
        # for a pseudo-likelihood.
        se_kind = "robust"
    if formula is not None:
        if y is not None or x is not None:
            raise MethodIncompatibility(
                "tobit: pass a formula or y= and x=, not both.",
                diagnostics={"formula": formula},
            )
        from ..core.utils import formula_to_columns

        data, y, x = formula_to_columns(formula, data)
    if y is None or x is None:
        raise MethodIncompatibility(
            "tobit: the outcome and the regressors are needed.",
            recovery_hint="Pass formula='y ~ x1 + x2', or y= and x=.",
        )
    extra = [c for c in (cluster, weights) if isinstance(c, str)]
    missing_cols = [c for c in [y] + list(x) + extra if c not in data]
    if missing_cols:
        raise MethodIncompatibility(
            f"tobit: columns not found in data: {missing_cols}",
            diagnostics={"missing": missing_cols},
        )
    df = data[list(dict.fromkeys([y] + list(x) + extra))].dropna()
    from ..core._collinear import drop_collinear_names

    x, collinear_omitted = drop_collinear_names(df, x, "tobit")
    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(len(df))] + [df[v].values.astype(float) for v in x])
    n, k = X.shape
    method_key = str(method).lower()
    if method_key not in ("mle", "ml", "scls"):
        raise MethodIncompatibility(
            f"tobit: method={method!r} is not available; use 'mle' or 'scls'."
        )
    if method_key == "scls":
        return _tobit_scls(df, Y, X, y, list(x), ll, ul, cluster, weights, alpha)
    wt = np.ones(n)
    if weights is not None:
        wt = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wt)) or np.any(wt <= 0):
            raise MethodIncompatibility(
                "tobit: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        wt = wt * (n / wt.sum())

    if ul is None:
        ul = np.inf

    censored_low = Y <= ll
    if np.isfinite(ul):
        censored_high = Y >= ul
    else:
        censored_high = np.zeros(len(Y), dtype=bool)
    uncensored = ~censored_low & ~censored_high

    n_censored = int(censored_low.sum() + censored_high.sum())
    n_uncensored = uncensored.sum()

    if n_uncensored < k + 1:
        raise DataInsufficient("Not enough uncensored observations.")

    # Initial values from OLS on uncensored
    beta_init = np.linalg.lstsq(X[uncensored], Y[uncensored], rcond=None)[0]
    resid_init = Y[uncensored] - X[uncensored] @ beta_init
    log_sigma_init = np.log(max(np.std(resid_init), 0.01))

    theta0 = np.concatenate([beta_init, [log_sigma_init]])

    # MLE
    def neg_loglik(theta: np.ndarray) -> float:
        beta = theta[:k]
        sigma = np.exp(theta[k])
        sigma = max(sigma, 1e-6)

        xb = X @ beta
        ll_val = 0.0

        # Uncensored
        if uncensored.any():
            resid = Y[uncensored] - xb[uncensored]
            ll_val += np.sum(
                wt[uncensored]
                * (-0.5 * np.log(2 * np.pi * sigma**2) - resid**2 / (2 * sigma**2))
            )

        # Left-censored
        if censored_low.any():
            z = (ll - xb[censored_low]) / sigma
            ll_val += np.sum(
                wt[censored_low] * np.log(np.maximum(stats.norm.cdf(z), 1e-20))
            )

        # Upper-censored
        if isinstance(censored_high, np.ndarray) and censored_high.any():
            z = (ul - xb[censored_high]) / sigma
            ll_val += np.sum(
                wt[censored_high] * np.log(np.maximum(stats.norm.sf(z), 1e-20))
            )

        return -ll_val

    result = optimize.minimize(
        neg_loglik, theta0, method="BFGS", options={"maxiter": 1000, "gtol": 1e-6}
    )

    # BFGS often reports status-2 ("precision loss") at a good Tobit optimum;
    # derive ``converged`` from the gradient norm so the flag does not
    # spuriously distrust correct estimates (see robust_convergence).
    converged, grad_norm = robust_convergence(result)

    # Newton polish on complex-step scores (the shared ML path of truncreg /
    # biprobit). BFGS stops at gtol=1e-6, which left coefficients ~1e-6 and
    # standard errors ~5e-5 (relative) from Stata's tobit; a few exact
    # Newton steps reach the optimum.
    from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation weighted log-likelihood, complex-step safe."""
        xb = X @ theta[:k]
        ln_s = theta[k]
        s = np.exp(ln_s)
        out = np.zeros(n, dtype=np.result_type(theta, float))
        mid = ~censored_low & ~censored_high
        r = (Y[mid] - xb[mid]) / s
        out[mid] = -0.5 * np.log(2 * np.pi) - ln_s - 0.5 * r * r
        if censored_low.any():
            out[censored_low] = special.log_ndtr((ll - xb[censored_low]) / s)
        if censored_high.any():
            out[censored_high] = special.log_ndtr((xb[censored_high] - ul) / s)
        return wt * out

    theta_hat, scores, H, _ = ml_newton_polish(
        obs_loglik, np.asarray(result.x, dtype=float)
    )
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    converged = bool(converged or grad_norm < 1e-6)
    beta = theta_hat[:k]
    sigma = np.exp(theta_hat[k])

    # Standard errors from the observed information of the polished fit.
    # Before 1.32 the second difference of the log-likelihood (~1e-5
    # accurate); earlier still `result.hess_inv` from BFGS, 13-30% off
    # R censReg::censReg and Stata `tobit` (parity finding #9).
    from ..core._vcov import ml_vcov

    clusters = df[cluster].to_numpy() if se_kind == "cluster" else None
    V_full = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se_full = se_from_vcov(V_full)

    se_beta = se_full[:k]
    se_sigma = se_full[k] * sigma  # delta method for exp transform

    var_names = ["const"] + x
    z_stats = beta / se_beta
    pvals = 2 * stats.norm.sf(np.abs(z_stats))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    detail = pd.DataFrame(
        {
            "variable": var_names + ["sigma"],
            "coefficient": np.append(beta, sigma),
            "se": np.append(se_beta, se_sigma),
            "z": np.append(z_stats, np.nan),
            "pvalue": np.append(pvals, np.nan),
        }
    )

    # Main estimate: first regressor
    main_coef = float(beta[1])
    main_se = float(se_beta[1])
    main_p = float(pvals[1])
    ci = (main_coef - z_crit * main_se, main_coef + z_crit * main_se)

    model_info = {
        "omitted": collinear_omitted,
        "method": "Tobit MLE",
        "sigma": float(sigma),
        "n_censored": int(n_censored),
        "n_uncensored": int(n_uncensored),
        "censor_pct": round(n_censored / n * 100, 1),
        "lower_limit": ll,
        "upper_limit": ul if np.isfinite(ul) else None,
        "log_likelihood": float(-result.fun),
        "converged": converged,
        "gradient_norm": grad_norm,
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "n_clusters": (int(df[cluster].nunique()) if se_kind == "cluster" else None),
        "weights": weights,
    }

    fit = LimitedDepResult(
        method="Tobit (Censored Regression)",
        estimand=f"beta_{x[0]}",
        estimate=main_coef,
        se=main_se,
        pvalue=main_p,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="tobit",
    )
    if weights is None:
        # What sp.cmtest needs to rebuild the likelihood and its moments.
        fit._cm_design = {
            "model": "tobit",
            "y": Y,
            "X": X,
            "names": var_names,
            "theta": theta_hat,
            "ll": ll if np.isfinite(ll) else None,
            "ul": ul if np.isfinite(ul) else None,
        }
        # Unweighted per-observation log-likelihood (sp.vuong).
        fit._llobs = np.real(obs_loglik(theta_hat))
    return fit


def _tobit_scls(
    df: pd.DataFrame,
    Y: np.ndarray,
    X: np.ndarray,
    y: str,
    x: List[str],
    ll: float,
    ul: Optional[float],
    cluster: Optional[str],
    weights: Optional[str],
    alpha: float,
) -> CausalResult:
    """Powell's (1986) symmetrically censored least squares.

    With the limit at zero, an observation with ``x'b > 0`` has its error
    censored from below at ``-x'b``. Censoring it from above at ``x'b`` as
    well, i.e. replacing ``y`` by ``min(y, 2 x'b)``, restores the symmetry
    of the error around ``x'b``, and least squares on those observations is
    consistent. The estimator is the fixed point of that regression.
    """
    if ul is not None and np.isfinite(ul):
        raise MethodIncompatibility(
            "tobit: method='scls' is defined for a lower limit only; set ul=None."
        )
    if ll is None or not np.isfinite(ll):
        raise MethodIncompatibility("tobit: method='scls' needs a finite ll.")
    if weights is not None:
        raise MethodIncompatibility("tobit: method='scls' does not take weights.")
    n, k = X.shape
    Ys = Y - ll  # limit at zero; the shift goes back into the constant
    mid = Ys > 0
    if int(mid.sum()) < k + 1:
        raise DataInsufficient("Not enough uncensored observations.")
    b = np.linalg.lstsq(X[mid], Ys[mid], rcond=None)[0]
    converged = False
    for n_iter in range(1, 5001):
        xb = X @ b
        keep = xb > 0
        if int(keep.sum()) < k + 1:
            raise DataInsufficient(
                "tobit: symmetrically censored least squares ran out of "
                f"observations with a positive index ({int(keep.sum())} left "
                f"for {k} coefficients); the data are too heavily censored."
            )
        Xk = X[keep]
        b_new = np.linalg.solve(Xk.T @ Xk, Xk.T @ np.minimum(Ys[keep], 2.0 * xb[keep]))
        done = np.max(np.abs(b_new - b)) < 1e-12 * (1.0 + np.max(np.abs(b)))
        b = b_new
        if done:
            converged = True
            break
    if not converged:
        from ..exceptions import ConvergenceFailure

        raise ConvergenceFailure(
            "tobit: the symmetrically censored least squares iteration did "
            "not settle in 5000 steps."
        )
    xb = X @ b
    u = Ys - xb
    keep = xb > 0
    inner = keep & (np.abs(u) < xb)
    C = X[inner].T @ X[inner]
    psi = X * (keep * (np.minimum(Ys, 2.0 * xb) - xb))[:, None]
    if cluster is not None:
        codes = pd.factorize(df[cluster].to_numpy())[0]
        sums = np.zeros((codes.max() + 1, k))
        np.add.at(sums, codes, psi)
        g = sums.shape[0]
        D = sums.T @ sums * (g / (g - 1.0))
    else:
        D = psi.T @ psi
    C_inv = np.linalg.inv(C)
    V = C_inv @ D @ C_inv
    se = np.sqrt(np.diag(V))
    beta = b.copy()
    beta[0] += ll
    var_names = ["const"] + x
    z_stats = beta / se
    pvals = 2 * stats.norm.sf(np.abs(z_stats))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    detail = pd.DataFrame(
        {
            "variable": var_names,
            "coefficient": beta,
            "se": se,
            "z": z_stats,
            "pvalue": pvals,
        }
    )
    fit = LimitedDepResult(
        method="Tobit (symmetrically censored least squares)",
        estimand=f"beta_{x[0]}",
        estimate=float(beta[1]),
        se=float(se[1]),
        pvalue=float(pvals[1]),
        ci=(float(beta[1] - z_crit * se[1]), float(beta[1] + z_crit * se[1])),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info={
            "method": "Tobit SCLS",
            "n_censored": int((~mid).sum()),
            "n_uncensored": int(mid.sum()),
            "n_used": int(keep.sum()),
            "n_symmetric": int(inner.sum()),
            "censor_pct": round(float((~mid).mean()) * 100, 1),
            "lower_limit": ll,
            "upper_limit": None,
            "converged": True,
            "n_iter": n_iter,
            "vce": "cluster" if cluster is not None else "robust",
            "cluster": cluster,
            "n_clusters": (int(df[cluster].nunique()) if cluster is not None else None),
            "var_cov": V,
        },
        _citation_key="tobit_scls",
    )
    return fit


CausalResult._CITATIONS["tobit_scls"] = (
    "@article{powell1986symmetrically,\n"
    "  title={Symmetrically Trimmed Least Squares Estimation for Tobit Models},\n"
    "  author={Powell, James L.},\n"
    "  journal={Econometrica},\n"
    "  volume={54},\n"
    "  number={6},\n"
    "  pages={1435--1460},\n"
    "  year={1986},\n"
    "  doi={10.2307/1914308}\n"
    "}"
)

# Citation
CausalResult._CITATIONS["tobit"] = (
    "@article{tobin1958estimation,\n"
    "  title={Estimation of Relationships for Limited Dependent Variables},\n"
    "  author={Tobin, James},\n"
    "  journal={Econometrica},\n"
    "  volume={26},\n"
    "  number={1},\n"
    "  pages={24--36},\n"
    "  year={1958},\n"
    "  publisher={Wiley}\n"
    "}"
)
