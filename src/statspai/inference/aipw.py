"""
Augmented Inverse Probability Weighting (AIPW) estimator.

Doubly robust estimator for average treatment effects that combines
outcome regression and inverse probability weighting. Consistent if
EITHER the outcome model OR the propensity score model is correctly
specified (but not necessarily both).

References
----------
Robins, J.M., Rotnitzky, A. and Zhao, L.P. (1994).
"Estimation of Regression Coefficients When Some Regressors Are Not
Always Observed."
*Journal of the American Statistical Association*, 89(427), 846-866.

Glynn, A.N. and Quinn, K.M. (2010).
"An Introduction to the Augmented Inverse Propensity Weighted Estimator."
*Political Analysis*, 18(1), 36-56.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
Hansen, C., Newey, W. and Robins, J. (2018).
"Double/Debiased Machine Learning for Treatment and Structural Parameters."
*The Econometrics Journal*, 21(1), C1-C68.
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def aipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = 'ATE',
    n_folds: int = 5,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Augmented Inverse Probability Weighting (AIPW) estimator.

    Doubly robust estimator: consistent if either the outcome model
    or the propensity score model is correctly specified.

    Uses cross-fitting (Chernozhukov et al. 2018) to avoid overfitting
    bias from flexible first-stage models.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment (0/1).
    covariates : list of str
        Covariates for both propensity score and outcome models.
    estimand : str, default 'ATE'
        Target: 'ATE' (average treatment effect) or
        'ATT' (average treatment effect on the treated).
    n_folds : int, default 5
        Number of cross-fitting folds.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    CausalResult
        Doubly robust treatment effect estimate.

    Examples
    --------
    >>> result = aipw(df, y='outcome', treat='treatment',
    ...              covariates=['age', 'income', 'education'])
    >>> print(result.summary())

    Notes
    -----
    The AIPW estimator is:

    .. math::
        \\hat{\\tau}_{AIPW} = \\frac{1}{n} \\sum_i \\left[
            \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{D_i (Y_i - \\hat{\\mu}_1(X_i))}{\\hat{e}(X_i)}
            - \\frac{(1-D_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-\\hat{e}(X_i)}
        \\right]

    where :math:`\\hat{e}(X)` is the propensity score and
    :math:`\\hat{\\mu}_d(X)` is the outcome regression for treatment
    status :math:`d`.

    See Glynn & Quinn (2010) for an introduction, and
    Chernozhukov et al. (2018) for the cross-fitting procedure.
    """
    rng = np.random.default_rng(seed)

    df = data[[y, treat] + covariates].dropna()
    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    X = df[covariates].values.astype(float)
    n = len(Y)

    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError(f"Treatment must be binary (0/1)")

    # Cross-fitted predictions
    mu1_hat = np.zeros(n)  # E[Y|X, D=1]
    mu0_hat = np.zeros(n)  # E[Y|X, D=0]
    e_hat = np.zeros(n)    # P(D=1|X)

    fold_ids = rng.choice(n_folds, size=n)

    for fold in range(n_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask

        X_tr, Y_tr, D_tr = X[train_mask], Y[train_mask], D[train_mask]
        X_te = X[test_mask]

        # Propensity score (logistic regression)
        e_hat[test_mask] = _fit_propensity(X_tr, D_tr, X_te)

        # Outcome regressions (OLS on treated and control separately)
        mu1_hat[test_mask] = _fit_outcome(
            X_tr[D_tr == 1], Y_tr[D_tr == 1], X_te)
        mu0_hat[test_mask] = _fit_outcome(
            X_tr[D_tr == 0], Y_tr[D_tr == 0], X_te)

    # Clip propensity scores
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # AIPW influence function
    if estimand == 'ATE':
        psi = (
            mu1_hat - mu0_hat
            + D * (Y - mu1_hat) / e_hat
            - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        )
    elif estimand == 'ATT':
        p_treat = np.mean(D)
        psi = (
            D * (Y - mu0_hat) / p_treat
            - (1 - D) * e_hat * (Y - mu0_hat) / ((1 - e_hat) * p_treat)
        )
    else:
        raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

    tau = float(np.mean(psi))
    se = float(np.sqrt(np.var(psi, ddof=1) / n))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = tau / se if se > 0 else 0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (tau - z_crit * se, tau + z_crit * se)

    model_info = {
        'estimator': 'AIPW (Doubly Robust)',
        'n_folds': n_folds,
        'n_treated': int(D.sum()),
        'n_control': int((1 - D).sum()),
        'mean_propensity': round(float(e_hat.mean()), 4),
    }

    return CausalResult(
        method='AIPW (Doubly Robust)',
        estimand=estimand,
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='aipw',
    )


def _fit_propensity(X_train, D_train, X_test):
    """Logistic regression propensity score."""
    try:
        import statsmodels.api as sm
        X_tr = sm.add_constant(X_train)
        X_te = sm.add_constant(X_test)
        logit = sm.Logit(D_train, X_tr)
        res = logit.fit(disp=0, maxiter=300, warn_convergence=False)
        return np.clip(res.predict(X_te), 0.01, 0.99)
    except Exception:
        return np.full(len(X_test), np.mean(D_train))


def _fit_outcome(X_train, Y_train, X_test):
    """OLS outcome regression."""
    if len(X_train) < 3:
        return np.full(len(X_test), np.mean(Y_train) if len(Y_train) > 0 else 0)
    try:
        import statsmodels.api as sm
        X_tr = sm.add_constant(X_train)
        X_te = sm.add_constant(X_test)
        ols = sm.OLS(Y_train, X_tr)
        res = ols.fit()
        return res.predict(X_te)
    except Exception:
        return np.full(len(X_test), np.mean(Y_train))


# Citation
CausalResult._CITATIONS['aipw'] = (
    "@article{glynn2010introduction,\n"
    "  title={An Introduction to the Augmented Inverse Propensity "
    "Weighted Estimator},\n"
    "  author={Glynn, Adam N. and Quinn, Kevin M.},\n"
    "  journal={Political Analysis},\n"
    "  volume={18},\n"
    "  number={1},\n"
    "  pages={36--56},\n"
    "  year={2010},\n"
    "  publisher={Cambridge University Press}\n"
    "}"
)
