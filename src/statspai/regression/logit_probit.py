"""
Logit, probit, and complementary log-log discrete choice models.

Maximum-likelihood estimation with analytical gradients and Hessians,
robust/clustered standard errors, marginal effects, and full diagnostics.

    P(Y=1|X) = F(X'β)

where F is the logistic CDF (logit), standard normal CDF (probit),
or 1 - exp(-exp(·)) (cloglog).

References
----------
Cameron, A.C. & Trivedi, P.K. (2005).
    *Microeconometrics: Methods and Applications*. Cambridge.

Greene, W.H. (2018).
    *Econometric Analysis*, 8th ed. Pearson.

Hosmer, D.W. & Lemeshow, S. (2000).
    *Applied Logistic Regression*, 2nd ed. Wiley. [@hosmer2000applied]
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from ..core.results import EconometricResults
from ..core.utils import parse_formula, create_design_matrices, prepare_data


# =========================================================================
# Link functions: CDF, PDF, and derivatives
# =========================================================================

def _logit_cdf(z: np.ndarray) -> np.ndarray:
    """Logistic CDF  Λ(z) = 1/(1+exp(-z))  (numerically stable)."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def _logit_pdf(z: np.ndarray) -> np.ndarray:
    """Logistic PDF  λ(z) = Λ(z)(1-Λ(z))."""
    p = _logit_cdf(z)
    return p * (1.0 - p)


def _logit_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of logistic PDF:  λ'(z) = λ(z)(1-2Λ(z))."""
    p = _logit_cdf(z)
    return p * (1.0 - p) * (1.0 - 2.0 * p)


def _probit_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF  Φ(z)."""
    return stats.norm.cdf(z)


def _probit_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF  φ(z)."""
    return stats.norm.pdf(z)


def _probit_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of normal PDF:  φ'(z) = -z φ(z)."""
    return -z * stats.norm.pdf(z)


def _cloglog_cdf(z: np.ndarray) -> np.ndarray:
    """Complementary log-log CDF  1 - exp(-exp(z))."""
    # Clip to prevent overflow
    z_clip = np.clip(z, -30, 30)
    return 1.0 - np.exp(-np.exp(z_clip))


def _cloglog_pdf(z: np.ndarray) -> np.ndarray:
    """Complementary log-log PDF  exp(z) * exp(-exp(z))."""
    z_clip = np.clip(z, -30, 30)
    return np.exp(z_clip) * np.exp(-np.exp(z_clip))


def _cloglog_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of cloglog PDF."""
    z_clip = np.clip(z, -30, 30)
    ez = np.exp(z_clip)
    return np.exp(-ez) * ez * (1.0 - ez)


_LINKS = {
    'logit': (_logit_cdf, _logit_pdf, _logit_pdf_deriv),
    'probit': (_probit_cdf, _probit_pdf, _probit_pdf_deriv),
    'cloglog': (_cloglog_cdf, _cloglog_pdf, _cloglog_pdf_deriv),
}


# =========================================================================
# Core MLE engine
# =========================================================================

def _log_likelihood(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Bernoulli log-likelihood  Σ w_i [y_i log F(Xβ) + (1-y_i) log(1-F(Xβ))]."""
    z = X @ beta
    p = cdf_func(z)
    # Clip probabilities for numerical safety
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    if weights is not None:
        ll = ll * weights
    return np.sum(ll)


def _score(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func,
    pdf_func,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gradient (score) vector  ∂ℓ/∂β."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    # generalized residual
    gen_resid = (y - p) * f / (p * (1.0 - p))
    if weights is not None:
        gen_resid = gen_resid * weights
    return X.T @ gen_resid


def _hessian(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func,
    pdf_func,
    pdf_deriv_func,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Analytical Hessian  ∂²ℓ/∂β∂β'."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    fp = pdf_deriv_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    pq = p * (1.0 - p)
    # d²ℓ/dz² for each obs
    d2 = (
        (y - p) * (fp * pq - f**2 * (1.0 - 2.0 * p)) / pq**2
        - f**2 / pq
    )
    if weights is not None:
        d2 = d2 * weights
    return X.T @ (d2[:, np.newaxis] * X)


def _newton_raphson(
    y: np.ndarray,
    X: np.ndarray,
    link: str,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Newton-Raphson MLE for binary choice models.

    Returns
    -------
    beta : parameter vector
    H : Hessian at convergence
    ll : log-likelihood at convergence
    n_iter : iterations used
    """
    cdf_func, pdf_func, pdf_deriv_func = _LINKS[link]
    n, k = X.shape

    # Starting values via OLS on y (clipped to 0.01–0.99)
    y_star = np.clip(y, 0.01, 0.99)
    try:
        beta = np.linalg.lstsq(X, y_star, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(k)

    ll_old = -np.inf
    converged = False

    for iteration in range(maxiter):
        ll_val = _log_likelihood(beta, y, X, cdf_func, weights)
        if np.abs(ll_val - ll_old) < tol:
            converged = True
            break
        ll_old = ll_val

        g = _score(beta, y, X, cdf_func, pdf_func, weights)
        H = _hessian(beta, y, X, cdf_func, pdf_func, pdf_deriv_func, weights)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]

        # Line search with step halving
        step_size = 1.0
        for _ in range(20):
            beta_new = beta - step_size * step
            ll_new = _log_likelihood(beta_new, y, X, cdf_func, weights)
            if ll_new > ll_val - 1e-4:
                break
            step_size *= 0.5
        beta = beta_new

    if not converged:
        warnings.warn(
            f"Newton-Raphson did not converge after {maxiter} iterations. "
            "Consider increasing maxiter or checking data.",
            stacklevel=3,
        )

    # Final Hessian for variance estimation
    H = _hessian(beta, y, X, cdf_func, pdf_func, pdf_deriv_func, weights)
    ll_val = _log_likelihood(beta, y, X, cdf_func, weights)

    return beta, H, ll_val, iteration + 1


# =========================================================================
# Variance-covariance estimators
# =========================================================================

def _mle_vcov(H: np.ndarray) -> np.ndarray:
    """MLE (observed information) variance: V = -H^{-1}."""
    try:
        return np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular; using pseudo-inverse.", stacklevel=3)
        return np.linalg.pinv(-H)


def _score_obs(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func,
    pdf_func,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-observation score vectors (n × k)."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    gen_resid = (y - p) * f / (p * (1.0 - p))
    if weights is not None:
        gen_resid = gen_resid * weights
    return gen_resid[:, np.newaxis] * X


def _robust_vcov(
    H: np.ndarray,
    score_obs: np.ndarray,
) -> np.ndarray:
    """
    Sandwich (Huber-White) robust variance.
    V = H^{-1} B H^{-1}  where B = Σ s_i s_i'.
    """
    A_inv = _mle_vcov(H)
    B = score_obs.T @ score_obs
    return A_inv @ B @ A_inv


def _cluster_vcov(
    H: np.ndarray,
    score_obs: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """
    Clustered sandwich variance.
    Score vectors summed within clusters before forming outer product.
    """
    A_inv = _mle_vcov(H)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    k = score_obs.shape[1]
    B = np.zeros((k, k))
    for c in unique_clusters:
        s_c = score_obs[clusters == c].sum(axis=0)
        B += np.outer(s_c, s_c)

    # Finite-sample correction: G/(G-1)
    correction = n_clusters / (n_clusters - 1)
    return correction * A_inv @ B @ A_inv


# =========================================================================
# Marginal effects
# =========================================================================

def _marginal_effects(
    beta: np.ndarray,
    X: np.ndarray,
    pdf_func,
    var_names: List[str],
    kind: str = 'average',
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute marginal effects ∂P/∂x_j = f(X'β) β_j.

    Parameters
    ----------
    kind : 'average' (AME), 'mean' (MEM), 'at' (MER)
    at_values : dict of variable -> value (for kind='at')
    """
    if kind == 'average':
        # AME: average of f(x_i'β) across all obs
        z = X @ beta
        f = pdf_func(z)
        me = np.mean(f) * beta
    elif kind == 'mean':
        # MEM: f evaluated at sample means
        x_bar = X.mean(axis=0)
        z_bar = x_bar @ beta
        f_bar = pdf_func(np.array([z_bar]))[0]
        me = f_bar * beta
    elif kind == 'at':
        # MER: at representative values
        if at_values is None:
            # Default to means
            x_rep = X.mean(axis=0)
        else:
            x_rep = X.mean(axis=0).copy()
            for vname, val in at_values.items():
                if vname in var_names:
                    idx = var_names.index(vname)
                    x_rep[idx] = val
        z_rep = x_rep @ beta
        f_rep = pdf_func(np.array([z_rep]))[0]
        me = f_rep * beta
    else:
        raise ValueError(f"Unknown marginal_effects kind: {kind}")

    return pd.DataFrame({
        'dy/dx': me,
    }, index=var_names)


# =========================================================================
# Diagnostics
# =========================================================================

def _hosmer_lemeshow(y: np.ndarray, p_hat: np.ndarray, n_groups: int = 10) -> Tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Returns (chi2_stat, p_value).
    """
    n = len(y)
    order = np.argsort(p_hat)
    y_sorted = y[order]
    p_sorted = p_hat[order]

    # Create groups (approximately equal sized)
    groups = np.array_split(np.arange(n), n_groups)

    chi2 = 0.0
    for grp_idx in groups:
        n_g = len(grp_idx)
        if n_g == 0:
            continue
        o_g = y_sorted[grp_idx].sum()
        e_g = p_sorted[grp_idx].sum()
        pi_g = e_g / n_g
        denom = n_g * pi_g * (1.0 - pi_g)
        if denom > 1e-15:
            chi2 += (o_g - e_g) ** 2 / denom

    df = n_groups - 2
    p_value = 1.0 - stats.chi2.cdf(chi2, df) if df > 0 else np.nan
    return chi2, p_value


def _roc_auc(y: np.ndarray, p_hat: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U statistic)."""
    pos = p_hat[y == 1]
    neg = p_hat[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Efficient computation via rank sums
    n1 = len(pos)
    n0 = len(neg)
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n1), np.zeros(n0)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=float)

    # Handle ties
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (ranks[order[i:j]].sum()) / (j - i)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum = ranks[labels == 1].sum()
    u = rank_sum - n1 * (n1 + 1) / 2
    return u / (n1 * n0)


def _classification_table(
    y: np.ndarray,
    p_hat: np.ndarray,
    cutoff: float = 0.5,
) -> Dict[str, Any]:
    """Confusion matrix with sensitivity/specificity."""
    y_pred = (p_hat >= cutoff).astype(int)
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    pcp = (tp + tn) / len(y) * 100

    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'pcp': pcp,
        'cutoff': cutoff,
    }


# =========================================================================
# Prediction helper
# =========================================================================

def _predict(
    beta: np.ndarray,
    X: np.ndarray,
    cdf_func,
    pred_type: str = 'response',
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Predict from a fitted binary choice model.

    Parameters
    ----------
    pred_type : 'response' (probabilities), 'link' (xβ), 'class' (0/1)
    """
    xb = X @ beta
    if pred_type == 'link':
        return xb
    elif pred_type == 'response':
        return cdf_func(xb)
    elif pred_type == 'class':
        return (cdf_func(xb) >= cutoff).astype(int)
    else:
        raise ValueError(f"Unknown predict type: {pred_type}")


# =========================================================================
# Public API
# =========================================================================

def _fit_binary(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
    link: str,
    robust: str,
    cluster: Optional[str],
    weights: Optional[str],
    marginal_effects: Optional[str],
    odds_ratio: bool,
    maxiter: int,
    tol: float,
    alpha: float,
    at_values: Optional[Dict[str, float]] = None,
) -> EconometricResults:
    """
    Internal workhorse for logit / probit / cloglog estimation.
    """
    # ── Prepare data ────────────────────────────────────────────────────
    if formula is not None and data is not None:
        y_df, X_df = create_design_matrices(formula, data)
        y_vec = y_df.values.ravel()
        X_mat = X_df.values
        var_names = list(X_df.columns)
        dep_var = y_df.columns[0]
    elif y is not None and x is not None and data is not None:
        cols = [y] + list(x)
        clean = data[cols].dropna()
        y_vec = clean[y].values.astype(float)
        dep_var = y
        X_raw = clean[x].values.astype(float)
        X_mat = np.column_stack([np.ones(len(X_raw)), X_raw])
        var_names = ['Intercept'] + list(x)
    else:
        raise ValueError("Provide either (formula, data) or (y, x, data).")

    # Validate binary outcome
    unique_vals = np.unique(y_vec)
    if not np.array_equal(np.sort(unique_vals), np.array([0.0, 1.0])):
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            pass  # only one category present — unusual but proceed
        else:
            raise ValueError(
                f"Dependent variable must be binary (0/1). "
                f"Found values: {unique_vals[:10]}"
            )

    n, k = X_mat.shape

    # Weights
    w = None
    if weights is not None and data is not None:
        w = data.loc[X_df.index if formula else clean.index, weights].values.astype(float)

    # Cluster variable
    cluster_arr = None
    if cluster is not None and data is not None:
        cluster_arr = data.loc[
            X_df.index if formula else clean.index, cluster
        ].values

    cdf_func, pdf_func, pdf_deriv_func = _LINKS[link]

    # ── Estimation ──────────────────────────────────────────────────────
    beta, H, ll, n_iter = _newton_raphson(y_vec, X_mat, link, w, maxiter, tol)

    # ── Variance-covariance ─────────────────────────────────────────────
    s_obs = _score_obs(beta, y_vec, X_mat, cdf_func, pdf_func, w)

    if cluster_arr is not None:
        vcov = _cluster_vcov(H, s_obs, cluster_arr)
        se_type = f"Clustered ({cluster})"
    elif robust != 'nonrobust':
        vcov = _robust_vcov(H, s_obs)
        se_type = "Robust (sandwich)"
    else:
        vcov = _mle_vcov(H)
        se_type = "MLE (observed information)"

    std_errors = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    # ── Null model log-likelihood ───────────────────────────────────────
    X_null = np.ones((n, 1))
    _, _, ll_null, _ = _newton_raphson(y_vec, X_null, link, w, maxiter=50, tol=1e-8)

    # ── Diagnostics ─────────────────────────────────────────────────────
    p_hat = cdf_func(X_mat @ beta)
    lr_chi2 = 2.0 * (ll - ll_null)
    lr_df = k - 1
    lr_pvalue = 1.0 - stats.chi2.cdf(lr_chi2, lr_df) if lr_df > 0 else np.nan
    pseudo_r2 = 1.0 - ll / ll_null if ll_null != 0 else np.nan
    aic = -2.0 * ll + 2.0 * k
    bic = -2.0 * ll + np.log(n) * k
    cls_table = _classification_table(y_vec, p_hat)
    hl_chi2, hl_pval = _hosmer_lemeshow(y_vec, p_hat)
    auc = _roc_auc(y_vec, p_hat)

    # ── Marginal effects ────────────────────────────────────────────────
    me_df = None
    if marginal_effects is not None:
        kind_map = {'average': 'average', 'mean': 'mean', 'at': 'at'}
        kind = kind_map.get(marginal_effects, 'average')
        me_df = _marginal_effects(beta, X_mat, pdf_func, var_names, kind, at_values)

    # ── Odds ratios (logit only) ────────────────────────────────────────
    or_series = None
    if odds_ratio and link == 'logit':
        or_vals = np.exp(beta)
        or_se = or_vals * std_errors  # delta-method
        or_series = pd.DataFrame({
            'OR': or_vals,
            'Std. Err.': or_se,
            f'[{alpha/2:.3f}': np.exp(beta - stats.norm.ppf(1 - alpha / 2) * std_errors),
            f'{1-alpha/2:.3f}]': np.exp(beta + stats.norm.ppf(1 - alpha / 2) * std_errors),
        }, index=var_names)

    # ── Build result ────────────────────────────────────────────────────
    params = pd.Series(beta, index=var_names)
    se_series = pd.Series(std_errors, index=var_names)

    link_label = {'logit': 'Logit', 'probit': 'Probit', 'cloglog': 'Complementary log-log'}
    model_info = {
        'model_type': link_label[link],
        'method': 'Maximum Likelihood (Newton-Raphson)',
        'family': 'binomial',
        'link': link,
        'll': ll,
        'll_null': ll_null,
        'lr_chi2': lr_chi2,
        'lr_df': lr_df,
        'lr_pvalue': lr_pvalue,
        'pseudo_r2': pseudo_r2,
        'aic': aic,
        'bic': bic,
        'pcp': cls_table['pcp'],
        'robust': robust,
        'cluster': cluster,
        'se_type': se_type,
        'n_iter': n_iter,
        'odds_ratio': or_series,
        'marginal_effects': me_df,
        'classification': cls_table,
        'hosmer_lemeshow': {'chi2': hl_chi2, 'p_value': hl_pval},
        'auc': auc,
    }

    data_info = {
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'dependent_var': dep_var,
        'fitted_values': p_hat,
        'residuals': y_vec - p_hat,
        'X': X_mat,
        'y': y_vec,
        'var_cov': vcov,
        'var_names': var_names,
    }

    diagnostics = {
        'Pseudo R-squared': pseudo_r2,
        'Log-Likelihood': ll,
        'Log-Lik. (null)': ll_null,
        'LR chi2': lr_chi2,
        'Prob > chi2': lr_pvalue,
        'AIC': aic,
        'BIC': bic,
        'PCP': cls_table['pcp'],
        'AUC (ROC)': auc,
    }

    result = EconometricResults(
        params=params,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    # Attach extra methods
    result.predict = lambda X_new=None, pred_type='response', cutoff=0.5: (
        _predict(beta, X_new if X_new is not None else X_mat, cdf_func, pred_type, cutoff)
    )
    result.classification_table = lambda cutoff=0.5: _classification_table(
        y_vec, p_hat if cutoff == 0.5 else cdf_func(X_mat @ beta), cutoff
    )

    return result


# =========================================================================
# Public functions
# =========================================================================

def logit(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    marginal_effects: str = None,
    odds_ratio: bool = False,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: dict = None,
) -> EconometricResults:
    """
    Logit (logistic) regression via maximum likelihood.

    Equivalent to Stata's ``logit y x1 x2`` or ``logistic`` (with ``or=True``).

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    odds_ratio : bool, default False
        Report odds ratios instead of log-odds coefficients.
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.logit("admit ~ gre + gpa + rank", data=df)
    >>> print(result.summary())

    >>> # With odds ratios and robust SE
    >>> result = sp.logit("admit ~ gre + gpa", data=df,
    ...                   robust='hc1', odds_ratio=True)

    >>> # Marginal effects at the mean
    >>> result = sp.logit("y ~ x1 + x2", data=df, marginal_effects='mean')
    >>> print(result.model_info['marginal_effects'])
    """
    return _fit_binary(
        formula=formula, data=data, y=y, x=x,
        link='logit', robust=robust, cluster=cluster, weights=weights,
        marginal_effects=marginal_effects, odds_ratio=odds_ratio,
        maxiter=maxiter, tol=tol, alpha=alpha, at_values=at_values,
    )


def probit(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    marginal_effects: str = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: dict = None,
) -> EconometricResults:
    """
    Probit regression via maximum likelihood.

    Equivalent to Stata's ``probit y x1 x2``.

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.probit("admit ~ gre + gpa + rank", data=df)
    >>> print(result.summary())

    >>> # Average marginal effects with clustered SE
    >>> result = sp.probit("y ~ x1 + x2", data=df,
    ...                    cluster='state', marginal_effects='average')
    """
    return _fit_binary(
        formula=formula, data=data, y=y, x=x,
        link='probit', robust=robust, cluster=cluster, weights=weights,
        marginal_effects=marginal_effects, odds_ratio=False,
        maxiter=maxiter, tol=tol, alpha=alpha, at_values=at_values,
    )


def cloglog(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    marginal_effects: str = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: dict = None,
) -> EconometricResults:
    """
    Complementary log-log regression via maximum likelihood.

    Appropriate when P(Y=1) is small (rare events) or when the latent
    distribution is asymmetric (extreme value type I).

    Equivalent to Stata's ``cloglog y x1 x2``.

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.cloglog("default ~ income + balance", data=df)
    >>> print(result.summary())
    """
    return _fit_binary(
        formula=formula, data=data, y=y, x=x,
        link='cloglog', robust=robust, cluster=cluster, weights=weights,
        marginal_effects=marginal_effects, odds_ratio=False,
        maxiter=maxiter, tol=tol, alpha=alpha, at_values=at_values,
    )
