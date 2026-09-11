"""
Sample selection and treatment effect models.

Bivariate probit, treatment effects model (endogenous treatment),
and switching regression.

Equivalent to Stata's ``biprobit``, ``etregress``, ``movestay``.

References
----------
Heckman, J.J. (1978).
"Dummy Endogenous Variables in a Simultaneous Equation System."
*Econometrica*, 46(4), 931-959. [@heckman1978dummy]

Maddala, G.S. (1983).
"Limited-Dependent and Qualitative Variables in Econometrics."
*Cambridge University Press*. [@maddala1983limited]
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from ..core.results import EconometricResults
from ._optim_helpers import robust_convergence


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


@accepts_aliases(vce="robust")
def biprobit(
    data: pd.DataFrame,
    y1: str,
    y2: str,
    x1: List[str],
    x2: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Bivariate probit model.

    Jointly estimates two binary outcomes with correlated errors:
        y1* = x1'β1 + ε1,  y1 = 1(y1* > 0)
        y2* = x2'β2 + ε2,  y2 = 1(y2* > 0)
        (ε1, ε2) ~ BVN(0, 0, 1, 1, ρ)

    Equivalent to Stata's ``biprobit y1 y2 = x1, x2``.

    Parameters
    ----------
    data : pd.DataFrame
    y1 : str
        First binary outcome.
    y2 : str
        Second binary outcome.
    x1 : list of str
        Regressors for equation 1.
    x2 : list of str, optional
        Regressors for equation 2. If None, same as x1.
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> age = rng.normal(40, 10, n)
    >>> education = rng.normal(13, 3, n)
    >>> e1 = rng.normal(0, 1, n)
    >>> e2 = 0.5 * e1 + rng.normal(0, 1, n)  # correlated errors
    >>> employed = ((-1.0 + 0.03 * age + 0.1 * education + e1) > 0).astype(int)
    >>> married = ((-0.5 + 0.02 * age + e2) > 0).astype(int)
    >>> df = pd.DataFrame({"employed": employed, "married": married,
    ...                    "age": age, "education": education})
    >>> result = sp.biprobit(df, y1='employed', y2='married',
    ...                      x1=['age', 'education'])
    >>> bool('rho' in result.model_info)  # error correlation reported
    True
    """
    x2_names = list(x1) if x2 is None else list(x2)

    df = data.dropna(subset=[y1, y2] + x1 + x2_names)
    n = len(df)

    y1_data = df[y1].values.astype(float)
    y2_data = df[y2].values.astype(float)
    X1 = np.column_stack([np.ones(n), df[x1].values.astype(float)])
    X2 = np.column_stack([np.ones(n), df[x2_names].values.astype(float)])
    k1 = X1.shape[1]
    k2 = X2.shape[1]

    def _bvn_cdf(h: np.ndarray, k: np.ndarray, rho: Any) -> np.ndarray:
        """Bivariate standard-normal CDF P(X<=h, Y<=k; corr=rho).

        Vectorised over observations via the Drezner-Wesolowsky (1990)
        identity

            Phi2(h, k; rho) = Phi(h) Phi(k) + integral_0^rho phi2(h, k; r) dr,

        with the inner integral evaluated by a 24-node Gauss-Legendre rule
        (smooth in every argument, accurate to ~1e-10). The previous
        implementation looped over observations calling
        ``multivariate_normal.cdf`` per point: ~280x slower and — fatally —
        precise only to ~1e-8, which corrupted BFGS's finite-difference
        gradient and pinned ``rho`` at its starting value of 0 (so the model
        always reported zero error correlation regardless of the data).
        """
        h = np.asarray(h, dtype=float)
        k = np.asarray(k, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 0:
            rho = np.full(h.shape, float(rho))
        rho = np.clip(rho, -0.999999, 0.999999)
        base = stats.norm.cdf(h) * stats.norm.cdf(k)
        nodes, weights = np.polynomial.legendre.leggauss(24)
        s = 0.5 * (nodes + 1.0)  # map [-1, 1] -> [0, 1]
        w = 0.5 * weights
        integral = np.zeros(h.shape)
        for s_j, w_j in zip(s, w):
            r = rho * s_j
            denom = 1.0 - r * r
            integral += (
                w_j
                * np.exp(-(h * h - 2.0 * r * h * k + k * k) / (2.0 * denom))
                / (2.0 * np.pi * np.sqrt(denom))
            )
        return _as_float_array(base + rho * integral)

    def neg_ll(theta: np.ndarray) -> float:
        beta1 = theta[:k1]
        beta2 = theta[k1 : k1 + k2]
        atanh_rho = theta[-1]
        rho = np.tanh(atanh_rho)

        xb1 = X1 @ beta1
        xb2 = X2 @ beta2

        # Adjust signs for different (y1, y2) combinations
        q1 = 2 * y1_data - 1  # +1 if y=1, -1 if y=0
        q2 = 2 * y2_data - 1

        h = q1 * xb1
        k_ = q2 * xb2
        rho_adj = q1 * q2 * rho

        probs = _bvn_cdf(h, k_, rho_adj)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return float(-np.sum(np.log(probs)))

    # Initialize with separate probits
    beta1_init = np.linalg.lstsq(X1, y1_data, rcond=None)[0]
    beta2_init = np.linalg.lstsq(X2, y2_data, rcond=None)[0]
    theta0 = np.concatenate([beta1_init, beta2_init, [0.0]])

    try:
        result = minimize(
            neg_ll, theta0, method="BFGS", options={"maxiter": maxiter, "gtol": tol}
        )
        theta_hat = _as_float_array(result.x)
        converged, grad_norm = robust_convergence(result)
    except Exception:
        theta_hat = theta0
        converged, grad_norm = False, float("inf")

    beta1 = theta_hat[:k1]
    beta2 = theta_hat[k1 : k1 + k2]
    rho = float(np.tanh(theta_hat[-1]))

    # Numerical Hessian for SE
    k_total = len(theta_hat)
    eps = 1e-5
    H = np.zeros((k_total, k_total))
    f0 = neg_ll(theta_hat)
    for i in range(k_total):
        ei = np.zeros(k_total)
        ei[i] = eps
        fp = neg_ll(theta_hat + ei)
        fm = neg_ll(theta_hat - ei)
        H[i, i] = (fp - 2 * f0 + fm) / eps**2
        for j in range(i + 1, k_total):
            ej = np.zeros(k_total)
            ej[j] = eps
            fpp = neg_ll(theta_hat + ei + ej)
            fpm = neg_ll(theta_hat + ei - ej)
            fmp = neg_ll(theta_hat - ei + ej)
            fmm = neg_ll(theta_hat - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

    try:
        var_cov = np.linalg.inv(H)
        se = _as_float_array(np.sqrt(np.abs(np.diag(var_cov))))
    except np.linalg.LinAlgError:
        se = np.full(k_total, np.nan)

    # Delta method for rho SE
    rho_se = float(se[-1] * (1 - rho**2))  # d(tanh)/d(atanh) = 1-tanh^2

    names = (
        ["eq1._cons"]
        + [f"eq1.{v}" for v in x1]
        + ["eq2._cons"]
        + [f"eq2.{v}" for v in x2_names]
        + ["rho"]
    )
    param_vals = np.concatenate([beta1, beta2, [rho]])
    se_vals = np.concatenate([se[:k1], se[k1 : k1 + k2], [rho_se]])

    params = pd.Series(param_vals, index=names)
    std_errors = pd.Series(se_vals, index=names)

    ll = float(-neg_ll(theta_hat))

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": "Bivariate Probit",
            "converged": converged,
            "gradient_norm": grad_norm,
            "rho": rho,
            "rho_se": rho_se,
            "rho_test_p": (
                float(2 * stats.norm.sf(abs(rho / rho_se))) if rho_se > 0 else np.nan
            ),
        },
        data_info={
            "n_obs": n,
            "dep_var_1": y1,
            "dep_var_2": y2,
            "df_resid": n - k_total,
        },
        diagnostics={
            "log_likelihood": ll,
            "aic": -2 * ll + 2 * k_total,
            "bic": -2 * ll + np.log(n) * k_total,
        },
    )


# ══════════════════════════════════════════════════════════════════════
#  etregress: likelihood, score and the two variance estimators
# ══════════════════════════════════════════════════════════════════════


def _probit_coefficients(Z: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Probit MLE coefficients for the selection equation."""

    def _neg(g):
        zg = Z @ g
        ll = np.where(D == 1, stats.norm.logcdf(zg), stats.norm.logcdf(-zg))
        q = 2.0 * D - 1.0
        lam = q * np.exp(stats.norm.logpdf(zg) - stats.norm.logcdf(q * zg))
        return -float(np.sum(ll)), -(Z * lam[:, None]).sum(0)

    start = np.linalg.lstsq(Z, D - 0.5, rcond=None)[0]
    res = minimize(
        _neg, start, jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 500}
    )
    return res.x


def _etregress_scores(
    theta: np.ndarray,
    yv: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
):
    """Log-likelihood and **per-observation** scores for Stata ``etregress``.

    The model is the common-slopes endogenous-treatment regression

        y  = w'b + eps           (w already carries the treatment column)
        D* = z'g + u,  D = 1(D* > 0)
        (eps, u) ~ BVN(0, 0, sigma^2, 1, rho*sigma)

    with Stata's parameterisation ``athrho = atanh(rho)``,
    ``lnsigma = log(sigma)`` -- both unbounded, which is what keeps the
    optimiser away from the |rho| = 1 boundary.

    Writing ``e = (y - w'b)/sigma``, ``a = (z'g + rho e)/sqrt(1 - rho^2)``
    and ``q = 2D - 1``, the contribution is

        l_i = log Phi(q_i a_i) - e_i^2/2 - log sigma - log(2 pi)/2

    The scores are returned per observation rather than summed because the
    robust and cluster covariance estimators need them that way; the caller
    sums when it wants the gradient.
    """
    kb, kg = W.shape[1], Z.shape[1]
    beta = theta[:kb]
    gamma = theta[kb : kb + kg]
    athrho, lnsigma = theta[kb + kg], theta[kb + kg + 1]
    rho = np.tanh(athrho)
    sigma = np.exp(lnsigma)
    one_m = 1.0 - rho**2
    root = np.sqrt(one_m)
    e = (yv - W @ beta) / sigma
    zg = Z @ gamma
    a = (zg + rho * e) / root
    q = 2.0 * D - 1.0
    m = q * a
    logcdf = stats.norm.logcdf(m)
    ll_i = logcdf - 0.5 * e**2 - lnsigma - 0.5 * np.log(2 * np.pi)
    # exp(logpdf - logcdf) rather than pdf/cdf: the ratio underflows to 0/0
    # in the far tail, where the log form is still exact.
    ql = q * np.exp(stats.norm.logpdf(m) - logcdf)
    s_beta = W * ((e - ql * rho / root) / sigma)[:, None]
    s_gamma = Z * (ql / root)[:, None]
    s_lnsigma = -ql * rho * e / root + e**2 - 1.0
    da_drho = e / root + (zg + rho * e) * rho / one_m**1.5
    s_athrho = ql * da_drho * one_m
    scores = np.column_stack([s_beta, s_gamma, s_athrho, s_lnsigma])
    return float(np.sum(ll_i)), scores


def _etregress_hessian(theta, yv, W, Z, D):
    """Observed information: the analytic score differenced once.

    Stata reports ``vce(oim)`` here. Differencing the *score* rather than
    the log-likelihood costs one power of eps -- second differences of the
    objective lose about half the digits, which showed up directly as
    standard errors an order of magnitude further from Stata's.
    """

    def _g(t):
        return _etregress_scores(t, yv, W, Z, D)[1].sum(0)

    n_ = len(theta)
    H = np.zeros((n_, n_))
    step = np.maximum(np.abs(theta), 1.0) * (np.finfo(float).eps ** (1 / 3))
    for i in range(n_):
        tp = theta.copy()
        tp[i] += step[i]
        tm = theta.copy()
        tm[i] -= step[i]
        H[:, i] = -(_g(tp) - _g(tm)) / (2 * step[i])
    return 0.5 * (H + H.T)


def _etregress_fit_mle(yv, W, Z, D, maxiter: int, tol: float):
    """Full-information ML, matching ``etregress`` (no ``twostep`` option).

    Returns ``(theta, hessian, loglik)``.

    The residual disagreement with Stata is ~5e-7 on the parameters and
    ~7e-16 on the log-likelihood, and that ordering is the whole story: the
    likelihood is flat to machine precision across the parameter gap.
    Starting BFGS from Stata's own reported theta neither raises the
    likelihood nor reduces the gradient, so neither optimiser is "closer to
    the optimum" than the other -- there is no further optimum to reach in
    float64. See tests/reference_parity/test_etregress_stata_parity.py.
    """
    beta0 = np.linalg.lstsq(W, yv, rcond=None)[0]
    resid0 = yv - W @ beta0
    theta = np.concatenate(
        [
            beta0,
            np.linalg.lstsq(Z, D - 0.5, rcond=None)[0],
            [0.0],
            [np.log(np.sqrt(np.mean(resid0**2)))],
        ]
    )

    def _neg(t):
        ll, s = _etregress_scores(t, yv, W, Z, D)
        return -ll, -s.sum(0)

    res = minimize(
        _neg,
        theta,
        jac=True,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": max(tol, 1e-12)},
    )
    theta = res.x
    best_ll = _etregress_scores(theta, yv, W, Z, D)[0]
    # Newton polish, but only accepting steps that raise the likelihood.
    # On a surface this flat an unguarded Newton step drifts sideways: an
    # earlier version accepted every step and landed 2.5e-4 from Stata with
    # a *lower* likelihood than where it started.
    for _ in range(25):
        g = _etregress_scores(theta, yv, W, Z, D)[1].sum(0)
        H = _etregress_hessian(theta, yv, W, Z, D)
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
            break
        moved = False
        for shrink in (1.0, 0.5, 0.25, 0.125):
            cand = theta + shrink * step
            ll_c = _etregress_scores(cand, yv, W, Z, D)[0]
            if ll_c > best_ll:
                theta, best_ll, moved = cand, ll_c, True
                break
        if not moved:
            break
    return theta, _etregress_hessian(theta, yv, W, Z, D), best_ll


def _sandwich(H, scores, cluster_vals):
    """Robust / cluster covariance from the observed information and scores."""
    Hinv = np.linalg.inv(H)
    n, k = scores.shape
    if cluster_vals is None:
        # Stata's ML robust VCE carries N/(N-1) on the meat. Omitting it is
        # a uniform 1/(2N) shortfall on every standard error -- 2.5e-4 at
        # N = 2000, which reads as noise until you notice it is the same
        # 2.5e-4 on all of them.
        meat = (n / (n - 1.0)) * (scores.T @ scores)
        return Hinv @ meat @ Hinv
    codes = pd.Categorical(cluster_vals).codes
    g = int(codes.max()) + 1
    agg = np.zeros((g, k))
    np.add.at(agg, codes, scores)
    meat = agg.T @ agg
    # Stata's ML cluster factor: g/(g-1) only. Not (n-1)/(n-k)*g/(g-1),
    # which is the regress-family factor.
    return (g / (g - 1.0)) * (Hinv @ meat @ Hinv)


@accepts_aliases(vce="robust")
def etregress(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    treatment: str,
    z: List[str],
    method: str = "mle",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Endogenous treatment effects model.

    Estimates the effect of an endogenous binary treatment on a
    continuous outcome, accounting for selection on unobservables.

        Outcome:  y = x'β + δ*D + ε
        Selection: D* = z'γ + u, D = 1(D* > 0)
        (ε, u) ~ BVN(0, 0, σ², 1, ρσ)

    Equivalent to Stata's ``etregress y x, treat(D = z)``, and pinned
    against it: the two-step agrees to 5e-9 on every coefficient and every
    standard error, and the ML fit's likelihood, score and observed
    information reproduce Stata's standard errors to 1e-10 when evaluated
    at Stata's own parameter vector. See
    ``tests/reference_parity/test_etregress_stata_parity.py``.

    .. versionchanged:: 1.27.0
       Three fixes, all of which change published numbers:

       * ``method='mle'`` now runs a maximum likelihood fit. It previously
         executed a verbatim copy of the two-step branch — the two code
         paths were identical — while the docstring named Stata's MLE.
       * ``method='twostep'`` standard errors now carry Heckman's
         correction for the estimated first stage. They were the naive OLS
         standard errors of the hazard-augmented regression, ~11% too
         small on a two-instrument design.
       * ``robust`` and ``cluster`` were accepted and then never
         referenced. Both now select the variance estimator.

       See MIGRATION.md — treatment effects and their inference should be
       recomputed.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : list of str
        Exogenous regressors.
    treatment : str
        Binary treatment variable (0/1, both values present).
    z : list of str
        Instruments for the selection equation.
    method : str, default 'mle'
        ``'mle'`` for full-information maximum likelihood (Stata's
        default) or ``'twostep'`` for the control-function estimator
        (Stata's ``twostep`` option). They are different estimators, not
        two routes to the same numbers.
    robust : str, default 'nonrobust'
        ``'nonrobust'`` (observed information), ``'robust'`` (sandwich,
        with Stata's ``N/(N-1)`` factor) or ``'cluster'``. Passing
        ``cluster=`` implies ``'cluster'``. ``vce=`` is accepted as an
        alias.
    cluster : str, optional
        Cluster column. Uses Stata's ML cluster factor ``g/(g-1)``.
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` carries the outcome equation, then the selection
        equation as ``<treatment>:<name>``, then ``athrho`` and
        ``lnsigma`` (ML) or ``hazard:lambda`` (two-step).
        ``diagnostics`` carries ``ate``, ``ate_se``, ``rho``, ``sigma``,
        ``lambda`` and, for ML, ``loglik``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> experience = rng.normal(15, 7, n)
    >>> education = rng.normal(13, 3, n)
    >>> father_union = rng.integers(0, 2, n)
    >>> region = rng.integers(0, 4, n)
    >>> u = rng.normal(0, 1, n)  # selection error, correlated with wage
    >>> union = ((-0.5 + 0.4 * father_union + 0.1 * region + u) > 0).astype(int)
    >>> wage = (8.0 + 0.3 * experience + 0.5 * education + 2.0 * union
    ...         + 1.5 * u + rng.normal(0, 2, n))
    >>> df = pd.DataFrame({"wage": wage, "experience": experience,
    ...                    "education": education, "union": union,
    ...                    "father_union": father_union, "region": region})
    >>> result = sp.etregress(df, y='wage', x=['experience', 'education'],
    ...                       treatment='union', z=['father_union', 'region'])
    >>> bool('ate' in result.diagnostics)  # endogenous treatment effect
    True
    """
    if method not in ("mle", "twostep"):
        raise ValueError(f"method must be 'mle' or 'twostep', got {method!r}")
    robust_kind = str(robust).lower()
    if robust_kind not in ("nonrobust", "robust", "cluster"):
        raise ValueError(
            "robust must be 'nonrobust', 'robust' or 'cluster', " f"got {robust!r}"
        )
    subset = [y, treatment] + x + z + ([cluster] if cluster else [])
    df = data.dropna(subset=subset)
    n = len(df)

    y_data = df[y].values.astype(float)
    D_data = df[treatment].values.astype(float)
    uniq = np.unique(D_data)
    if not np.all(np.isin(uniq, (0.0, 1.0))) or uniq.size < 2:
        raise ValueError(
            f"treatment '{treatment}' must be binary 0/1 with both values "
            f"present; got {uniq[:5]}"
        )
    W = np.column_stack([np.ones(n), df[x].values.astype(float), D_data])
    Z = np.column_stack([np.ones(n), df[z].values.astype(float)])
    k_out = W.shape[1]
    out_names = ["_cons"] + list(x) + [treatment]
    sel_names = [f"{treatment}:_cons"] + [f"{treatment}:{v}" for v in z]
    cluster_vals = df[cluster].values if cluster else None
    if robust_kind == "cluster" and cluster_vals is None:
        raise ValueError("robust='cluster' requires cluster=<column name>")
    if cluster_vals is not None:
        robust_kind = "cluster"

    if method == "twostep":
        # Stata's `etregress ..., twostep`: probit, then OLS on the
        # treatment-specific hazard, with Heckman's corrected covariance.
        gamma = _probit_coefficients(Z, D_data)
        zg = Z @ gamma
        hazard = np.where(
            D_data == 1,
            np.exp(stats.norm.logpdf(zg) - stats.norm.logcdf(zg)),
            -np.exp(stats.norm.logpdf(zg) - stats.norm.logcdf(-zg)),
        )
        delta_i = hazard * (hazard + zg)
        Wa = np.column_stack([W, hazard])
        XtX_inv = np.linalg.inv(Wa.T @ Wa)
        beta = XtX_inv @ (Wa.T @ y_data)
        resid = y_data - Wa @ beta
        # sigma^2 is NOT the residual variance: the hazard term absorbs part
        # of it, so Heckman adds rho^2 * sum(delta_i) back before dividing
        # by n. Using the plain residual variance understates sigma and,
        # through it, every standard error.
        sigma2 = (resid @ resid + beta[-1] ** 2 * np.sum(delta_i)) / n
        sigma = np.sqrt(sigma2)
        rho = beta[-1] / sigma
        probit_info = np.linalg.inv((Z * delta_i[:, None]).T @ Z)
        cross = (Wa * delta_i[:, None]).T @ Z
        Q = (rho**2) * cross @ probit_info @ cross.T
        vcov = (
            sigma2
            * XtX_inv
            @ ((Wa * (1.0 - rho**2 * delta_i)[:, None]).T @ Wa + Q)
            @ XtX_inv
        )
        se_all = np.sqrt(np.diag(vcov))
        params = pd.Series(beta, index=out_names + ["hazard:lambda"])
        std_errors = pd.Series(se_all, index=params.index)
        loglik = None
        converged = True
        extra = {}
    else:
        theta, hessian, loglik = _etregress_fit_mle(
            y_data, W, Z, D_data, maxiter=maxiter, tol=tol
        )
        scores = _etregress_scores(theta, y_data, W, Z, D_data)[1]
        if robust_kind == "nonrobust":
            vcov = np.linalg.inv(hessian)
        else:
            vcov = _sandwich(
                hessian, scores, cluster_vals if robust_kind == "cluster" else None
            )
        se_all = np.sqrt(np.diag(vcov))
        kg = Z.shape[1]
        rho = float(np.tanh(theta[k_out + kg]))
        sigma = float(np.exp(theta[k_out + kg + 1]))
        names = out_names + sel_names + ["athrho", "lnsigma"]
        params = pd.Series(theta, index=names)
        std_errors = pd.Series(se_all, index=names)
        converged = bool(np.all(np.isfinite(se_all)))
        extra = {
            "athrho": float(theta[k_out + kg]),
            "lnsigma": float(theta[k_out + kg + 1]),
            "loglik": float(loglik),
        }

    ate = float(params.iloc[k_out - 1])
    ate_se = float(std_errors.iloc[k_out - 1])
    beta = params.values
    se = std_errors.values

    diagnostics = {
        "ate": ate,
        "ate_se": ate_se,
        "rho": float(rho),
        "sigma": float(sigma),
        "lambda": float(rho * sigma),
        "converged": converged,
    }
    diagnostics.update(extra)
    # ``selection_corr`` is rho under both methods -- it is the parameter the
    # name always meant. ``mills_coef`` / ``mills_se`` are two-step objects
    # (the hazard coefficient rho*sigma and its SE) and have no counterpart
    # in the MLE, which never forms a Mills ratio, so they appear only there
    # rather than being faked from rho*sigma.
    diagnostics["selection_corr"] = float(rho)
    if method == "twostep":
        diagnostics["mills_coef"] = float(params.iloc[-1])
        diagnostics["mills_se"] = float(std_errors.iloc[-1])

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": "Endogenous Treatment Effects",
            "method": method,
            "vce": robust_kind,
            "treatment_effect": ate,
            "treatment_var": treatment,
            "rho": float(rho),
            "sigma": float(sigma),
            "log_likelihood": None if loglik is None else float(loglik),
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "df_resid": n - len(beta),
        },
        diagnostics=diagnostics,
    )
