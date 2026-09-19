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
[@robins1994estimation]

Glynn, A.N. and Quinn, K.M. (2010). "An Introduction to the Augmented Inverse
Propensity Weighted Estimator." *Political Analysis*, 18(1), 36-56.
[@glynn2010introduction]

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
Hansen, C., Newey, W. and Robins, J. (2018).
"Double/Debiased Machine Learning for Treatment and Structural Parameters."
*The Econometrics Journal*, 21(1), C1-C68. [@chernozhukov2018double]
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning


@accepts_aliases(_strict=True, controls="covariates")
def aipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    n_folds: int = 5,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
    cross_fit: bool = True,
    se_method: str = "influence",
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
    seed : int or None, default 42
        Seed for the cross-fitting fold assignment.  The default is a
        fixed integer so repeated calls on the same data return the same
        estimate — matching ``sp.tmle``, ``sp.bcf`` and
        ``sp.super_learner``, which already default to 42.

        Before 1.21.0 this defaulted to ``None``, which seeds
        ``np.random.default_rng`` from OS entropy: the point estimate
        moved by hundreds of dollars between identical calls and could
        not be pinned even with ``np.random.seed(...)``, because
        ``default_rng(None)`` ignores the legacy global RNG.

        Pass ``seed=None`` explicitly to opt back into a fresh random
        fold split per call (useful for Monte-Carlo studies of
        fold-assignment sensitivity); the seed actually used is always
        recorded in ``result.model_info['seed']``.
    cross_fit : bool, default True
        Cross-fit the nuisance models over ``n_folds`` random folds
        (Chernozhukov et al. 2018). ``False`` fits the logit propensity
        and the two per-arm OLS outcome regressions once on the full
        sample -- the classical parametric AIPW estimator, which is
        what Stata's ``teffects aipw`` and R ``AIPW`` with
        ``k_split = 1`` compute. ``n_folds`` and ``seed`` are then
        unused.
    se_method : {'influence', 'sandwich'}, default 'influence'
        ``'influence'`` reports ``sd(psi) / sqrt(n)`` from the
        estimated efficient influence function, treating the nuisance
        fits as known (the cross-fitting / DML convention; R ``AIPW``
        reports the same quantity). ``'sandwich'`` stacks the logit
        score, the two OLS normal equations and the AIPW moment and
        reports the M-estimation sandwich (divisor ``n``), which
        accounts for the estimated nuisance parameters at finite ``n``.
        This is the robust standard error of Stata's
        ``teffects aipw``. Requires ``cross_fit=False`` and
        ``estimand='ATE'``. The two agree asymptotically when both
        nuisance models are correctly specified.

    Returns
    -------
    CausalResult
        Doubly robust treatment effect estimate.

    References
    ----------
    Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of
    Regression Coefficients When Some Regressors are not Always Observed.
    *Journal of the American Statistical Association*.
    doi:10.1080/01621459.1994.10476818 [@robins1994estimation]

    Glynn, A. N. and Quinn, K. M. (2010). An Introduction to the
    Augmented Inverse Propensity Weighted Estimator. *Political
    Analysis*. doi:10.1093/pan/mpp036 [@glynn2010introduction]

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> age = rng.normal(40, 10, n)
    >>> income = rng.normal(50, 15, n)
    >>> education = rng.integers(10, 18, n).astype(float)
    >>> ps = 1 / (1 + np.exp(-(0.03 * (age - 40) + 0.02 * (income - 50))))
    >>> treatment = rng.binomial(1, ps)
    >>> outcome = (2.0 * treatment + 0.05 * age + 0.03 * income
    ...            + 0.1 * education + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({'outcome': outcome, 'treatment': treatment,
    ...                    'age': age, 'income': income,
    ...                    'education': education})
    >>> result = sp.aipw(df, y='outcome', treat='treatment',
    ...                  covariates=['age', 'income', 'education'], seed=0)
    >>> bool(result.estimate > 0)
    True

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
    if se_method not in ("influence", "sandwich"):
        raise ValueError(
            f"se_method must be 'influence' or 'sandwich', got {se_method!r}"
        )
    if se_method == "sandwich" and (cross_fit or estimand != "ATE"):
        raise ValueError(
            "se_method='sandwich' is the stacked M-estimation variance of "
            "the full-sample parametric AIPW ATE; it requires "
            "cross_fit=False and estimand='ATE'."
        )
    rng = np.random.default_rng(seed)

    df = data[[y, treat] + covariates].dropna()
    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    X = df[covariates].values.astype(float)
    n = len(Y)

    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError("Treatment must be binary (0/1)")

    # Cross-fitted predictions
    mu1_hat = np.zeros(n)  # E[Y|X, D=1]
    mu0_hat = np.zeros(n)  # E[Y|X, D=0]
    e_hat = np.zeros(n)  # P(D=1|X)

    if cross_fit:
        fold_ids = rng.choice(n_folds, size=n)
        splits = [(fold_ids != f, fold_ids == f) for f in range(n_folds)]
    else:
        everyone = np.ones(n, dtype=bool)
        splits = [(everyone, everyone)]

    for train_mask, test_mask in splits:
        X_tr, Y_tr, D_tr = X[train_mask], Y[train_mask], D[train_mask]
        X_te = X[test_mask]

        # Propensity score (logistic regression)
        e_hat[test_mask] = _fit_propensity(X_tr, D_tr, X_te)

        # Outcome regressions (OLS on treated and control separately)
        mu1_hat[test_mask] = _fit_outcome(X_tr[D_tr == 1], Y_tr[D_tr == 1], X_te)
        mu0_hat[test_mask] = _fit_outcome(X_tr[D_tr == 0], Y_tr[D_tr == 0], X_te)

    # Clip propensity scores
    n_clipped = int(np.sum((e_hat < 0.01) | (e_hat > 0.99)))
    np.clip(e_hat, 0.01, 0.99, out=e_hat)

    po_means = None
    po_means_se = None
    # AIPW influence function
    if estimand == "ATE":
        phi1 = mu1_hat + D * (Y - mu1_hat) / e_hat
        phi0 = mu0_hat + (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        psi = phi1 - phi0
        tau = float(np.mean(psi))
        m1, m0 = float(np.mean(phi1)), float(np.mean(phi0))
        po_means = {1: m1, 0: m0}
        if se_method == "sandwich":
            if n_clipped:
                warnings.warn(
                    f"aipw: {n_clipped} propensity score(s) were clipped to "
                    "[0.01, 0.99]; the stacked sandwich differentiates the "
                    "unclipped logit and is only approximate for those rows.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if_1, if_0 = _aipw_stacked_if(
                X, D, Y, e_hat, mu1_hat, mu0_hat, phi1 - m1, phi0 - m0
            )
            # M-estimation sandwich, divisor n (Stata teffects convention).
            se = float(np.sqrt(np.mean((if_1 - if_0) ** 2) / n))
            po_means_se = {
                1: float(np.sqrt(np.mean(if_1**2) / n)),
                0: float(np.sqrt(np.mean(if_0**2) / n)),
            }
        else:
            se = float(np.sqrt(np.var(psi, ddof=1) / n))
            po_means_se = {
                1: float(np.sqrt(np.var(phi1, ddof=1) / n)),
                0: float(np.sqrt(np.var(phi0, ddof=1) / n)),
            }
    elif estimand == "ATT":
        p_treat = np.mean(D)
        psi = D * (Y - mu0_hat) / p_treat - (1 - D) * e_hat * (Y - mu0_hat) / (
            (1 - e_hat) * p_treat
        )
        tau = float(np.mean(psi))
        # ATT = mean(N_i) / mean(D_i) is a ratio, so its influence function
        # is (N_i - tau * D_i) / p, i.e. psi_i - tau * D_i / p. Before
        # 1.29 the ``- tau * D / p`` centring term was missing and the SE
        # was the standard deviation of psi itself, which is wrong unless
        # tau = 0 (it overstated the SE on every non-null ATT).
        se = float(np.sqrt(np.var(psi - tau * D / p_treat, ddof=1) / n))
    else:
        raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = tau / se if se > 0 else 0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    ci = (tau - z_crit * se, tau + z_crit * se)

    model_info = {
        "estimator": "AIPW (Doubly Robust)",
        "n_folds": n_folds,
        "n_treated": int(D.sum()),
        "n_control": int((1 - D).sum()),
        "mean_propensity": float(e_hat.mean()),
        "cross_fit": bool(cross_fit),
        "se_method": se_method,
        "n_propensity_clipped": n_clipped,
        "potential_outcome_means": po_means,
        "potential_outcome_means_se": po_means_se,
        # Provenance for the cross-fitting split: None means the caller
        # explicitly opted into a fresh entropy-seeded split, so this
        # estimate is not reproducible by re-running.
        "seed": seed,
    }

    _result = CausalResult(
        method="AIPW (Doubly Robust)",
        estimand=estimand,
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key="aipw",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.aipw",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "n_folds": n_folds,
                "alpha": alpha,
                "seed": seed,
                "cross_fit": cross_fit,
                "se_method": se_method,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _aipw_stacked_if(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    phi1_c: np.ndarray,
    phi0_c: np.ndarray,
) -> tuple:
    """Influence functions of the two AIPW potential-outcome means.

    Stacks the full-sample logit score ``X (D - e)``, the per-arm OLS
    normal equations ``D X (Y - X b1)`` / ``(1-D) X (Y - X b0)`` and the
    AIPW moments, and returns row ``i`` of ``-A^{-1} psi_i`` for the two
    mean parameters. ``phi1_c`` / ``phi0_c`` are the centred AIPW
    moments. The correction terms are ``E[d psi_mu / d gamma] IF_gamma``
    and ``E[d psi_mu / d beta] IF_beta``; they vanish in expectation only
    when both nuisance models are correct, so at finite ``n`` they move
    the variance.
    """
    n = len(Y)
    Xc = np.column_stack([np.ones(n), X])
    r1, r0 = Y - mu1, Y - mu0
    w = e * (1 - e)
    if_gamma = np.linalg.solve(
        (Xc * w[:, None]).T @ Xc / n, (Xc * (D - e)[:, None]).T
    ).T
    if_b1 = np.linalg.solve((Xc * D[:, None]).T @ Xc / n, (Xc * (D * r1)[:, None]).T).T
    if_b0 = np.linalg.solve(
        (Xc * (1 - D)[:, None]).T @ Xc / n, (Xc * ((1 - D) * r0)[:, None]).T
    ).T
    # d/d gamma of D r1 / e is -D r1 (1 - e) / e * x; of (1-D) r0 / (1-e)
    # it is (1-D) r0 e / (1 - e) * x. d/d beta of the AIPW moments is
    # x (1 - D / e) and x (1 - (1-D) / (1-e)).
    g1 = (Xc * (-D * r1 * (1 - e) / e)[:, None]).mean(axis=0)
    g0 = (Xc * ((1 - D) * r0 * e / (1 - e))[:, None]).mean(axis=0)
    b1 = (Xc * (1 - D / e)[:, None]).mean(axis=0)
    b0 = (Xc * (1 - (1 - D) / (1 - e))[:, None]).mean(axis=0)
    if_1 = phi1_c + if_gamma @ g1 + if_b1 @ b1
    if_0 = phi0_c + if_gamma @ g0 + if_b0 @ b0
    return if_1, if_0


def _fit_propensity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Logistic regression propensity score."""
    try:
        import statsmodels.api as sm

        X_tr = sm.add_constant(X_train)
        X_te = sm.add_constant(X_test)
        logit = sm.Logit(D_train, X_tr)
        res = logit.fit(disp=0, maxiter=300, warn_convergence=False)
        return np.asarray(np.clip(res.predict(X_te), 0.01, 0.99), dtype=float)
    except Exception as exc:
        # A constant propensity turns AIPW into regression-adjustment-only,
        # and the influence-function SE (which assumes both nuisances were
        # fit) is then wrong. Do not degrade silently (CLAUDE.md §3.7).
        warnings.warn(
            "aipw: the propensity-score logit failed "
            f"({type(exc).__name__}: {exc}); falling back to a CONSTANT "
            "propensity. The AIPW estimate reduces to outcome-regression "
            "adjustment and its influence-function standard error is no "
            "longer valid. Check for separated or collinear covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return np.full(len(X_test), np.mean(D_train))


def _fit_outcome(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """OLS outcome regression."""
    if len(X_train) < 3:
        return np.full(len(X_test), np.mean(Y_train) if len(Y_train) > 0 else 0)
    try:
        import statsmodels.api as sm

        X_tr = sm.add_constant(X_train)
        X_te = sm.add_constant(X_test)
        ols = sm.OLS(Y_train, X_tr)
        res = ols.fit()
        return np.asarray(res.predict(X_te), dtype=float)
    except Exception as exc:
        # A constant outcome model turns AIPW into IPW-only; the
        # influence-function SE is then wrong. Warn (CLAUDE.md §3.7).
        warnings.warn(
            "aipw: the outcome regression failed "
            f"({type(exc).__name__}: {exc}); falling back to a CONSTANT "
            "outcome model. The AIPW estimate reduces to inverse-propensity "
            "weighting and its influence-function standard error is no "
            "longer valid. Check for collinear or degenerate covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return np.full(len(X_test), np.mean(Y_train))


# Citation
CausalResult._CITATIONS["aipw"] = (
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
