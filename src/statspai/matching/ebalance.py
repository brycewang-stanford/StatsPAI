"""
Entropy Balancing (Hainmueller 2012).

Reweights the control group so that weighted covariate moments (mean,
variance, skewness) exactly match the treated group, without dropping
observations or relying on propensity score models.

More robust than PSM because it directly targets balance rather than
modeling the selection process.

References
----------
Hainmueller, J. (2012).
"Entropy Balancing for Causal Effects: A Multivariate Reweighting
Method to Produce Balanced Samples in Observational Studies."
*Political Analysis*, 20(1), 25-46. [@hainmueller2012entropy]
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult


def ebalance(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    moments: int = 1,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Entropy Balancing treatment effect estimator.

    Reweights control units to exactly match treated covariate moments,
    then estimates ATT via weighted difference in means.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Covariates to balance on.
    moments : int, default 1
        Number of moments to balance:
        - 1: means only
        - 2: means and variances
        - 3: means, variances, and skewness
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ATT estimate with entropy-balanced weights and balance table.

    Examples
    --------
    >>> result = sp.ebalance(df, y='outcome', treat='treated',
    ...                       covariates=['age', 'income', 'education'])
    >>> print(result.summary())
    >>> # Check balance improvement
    >>> print(result.model_info['balance'])

    Notes
    -----
    Entropy balancing solves:

    .. math::
        \\min_w \\sum_i w_i \\log(w_i / q_i)

    subject to balance constraints (weighted moments match) and
    normalization (weights sum to 1).

    Unlike PSM, this guarantees exact balance on specified moments
    without iteration or caliper tuning.

    See Hainmueller (2012, *Political Analysis*).
    """
    df = data[[y, treat] + covariates].dropna()
    D = df[treat].values.astype(float)
    Y = df[y].values.astype(float)
    X = df[covariates].values.astype(float)

    t_mask = D == 1
    c_mask = D == 0
    n_t = t_mask.sum()
    n_c = c_mask.sum()

    if n_t < 2 or n_c < 2:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "Need at least 2 treated and 2 control units.",
            recovery_hint=(
                "Check the treatment variable coding or relax the sample "
                "filter; entropy balancing needs at least a 2/2 split."
            ),
            diagnostics={"n_treated": int(n_t), "n_control": int(n_c)},
            alternative_functions=["sp.match", "sp.cbps"],
        )

    X_t = X[t_mask]
    X_c = X[c_mask]
    Y_t = Y[t_mask]
    Y_c = Y[c_mask]

    # Build moment constraint targets (from treated group)
    targets, C_matrix = _build_constraints(X_t, X_c, covariates, moments)

    # Solve for entropy-balanced weights
    weights = _solve_ebalance(C_matrix, targets, n_c)

    # Verify balance constraints are satisfied
    achieved = C_matrix.T @ weights
    max_imbalance = np.max(np.abs(achieved - targets))
    if max_imbalance > 0.01:
        import warnings
        warnings.warn(
            f"Entropy balancing did not fully converge "
            f"(max moment imbalance = {max_imbalance:.4f}). "
            f"Consider reducing the number of covariates or moments.",
            UserWarning,
        )

    # ATT = mean(Y_t) - weighted_mean(Y_c)
    att = float(np.mean(Y_t) - np.average(Y_c, weights=weights))

    # SE via weighted variance
    var_t = np.var(Y_t, ddof=1) / n_t
    var_c = np.average((Y_c - np.average(Y_c, weights=weights)) ** 2,
                       weights=weights) / n_c
    se = float(np.sqrt(var_t + var_c))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = att / se if se > 0 else 0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (att - z_crit * se, att + z_crit * se)

    # Balance check
    balance = _balance_check(X_t, X_c, weights, covariates)

    model_info = {
        'method': 'Entropy Balancing',
        'moments_balanced': moments,
        'n_treated': int(n_t),
        'n_control': int(n_c),
        'max_weight': float(np.max(weights)),
        'eff_sample_size': float(1 / np.sum(weights ** 2)),
        'balance': balance,
        'weights': weights,
    }

    return CausalResult(
        method='Entropy Balancing (Hainmueller 2012)',
        estimand='ATT',
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(df),
        detail=balance,
        model_info=model_info,
        _citation_key='ebalance',
    )


def _build_constraints(X_t, X_c, covariates, moments):
    """Build moment targets and constraint matrix."""
    k = len(covariates)
    targets = []
    C_cols = []

    # First moments (means)
    for j in range(k):
        targets.append(np.mean(X_t[:, j]))
        C_cols.append(X_c[:, j])

    # Second moments (variances) if requested
    if moments >= 2:
        for j in range(k):
            targets.append(np.mean(X_t[:, j] ** 2))
            C_cols.append(X_c[:, j] ** 2)

    # Third moments (skewness) if requested
    if moments >= 3:
        for j in range(k):
            targets.append(np.mean(X_t[:, j] ** 3))
            C_cols.append(X_c[:, j] ** 3)

    C_matrix = np.column_stack(C_cols)
    targets = np.array(targets)

    return targets, C_matrix


def _solve_ebalance(C, targets, n_c, max_iter=200):
    """Solve entropy balancing via Lagrange dual (Newton's method)."""
    m = len(targets)

    # Dual: maximize L(λ) = -log(Σ exp(C λ)) + λ' targets
    def neg_dual(lam):
        Cl = C @ lam
        Cl = np.clip(Cl, -500, 500)  # prevent overflow
        log_sum_exp = np.log(np.sum(np.exp(Cl)))
        return -(lam @ targets - log_sum_exp)

    def grad(lam):
        Cl = C @ lam
        Cl = np.clip(Cl, -500, 500)
        exp_Cl = np.exp(Cl)
        w = exp_Cl / np.sum(exp_Cl)
        return -(targets - C.T @ w)

    lam0 = np.zeros(m)

    try:
        result = optimize.minimize(
            neg_dual, lam0, jac=grad, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12},
        )
        lam = result.x
    except Exception:
        return np.ones(n_c) / n_c

    # Recover weights
    Cl = C @ lam
    Cl = np.clip(Cl, -500, 500)
    w = np.exp(Cl)
    w = w / np.sum(w)

    return w


def _balance_check(X_t, X_c, weights, covariates):
    """Check balance before/after reweighting."""
    rows = []
    for j, cov in enumerate(covariates):
        mean_t = np.mean(X_t[:, j])
        mean_c_raw = np.mean(X_c[:, j])
        mean_c_w = np.average(X_c[:, j], weights=weights)

        sd_pooled = np.sqrt(
            (np.var(X_t[:, j], ddof=1) + np.var(X_c[:, j], ddof=1)) / 2
        )
        sd_pooled = max(sd_pooled, 1e-10)

        smd_before = (mean_t - mean_c_raw) / sd_pooled
        smd_after = (mean_t - mean_c_w) / sd_pooled

        rows.append({
            'covariate': cov,
            'mean_treated': round(mean_t, 4),
            'mean_control_raw': round(mean_c_raw, 4),
            'mean_control_balanced': round(mean_c_w, 4),
            'smd_before': round(smd_before, 4),
            'smd_after': round(smd_after, 4),
        })

    return pd.DataFrame(rows)


# Citation
CausalResult._CITATIONS['ebalance'] = (
    "@article{hainmueller2012entropy,\n"
    "  title={Entropy Balancing for Causal Effects: A Multivariate "
    "Reweighting Method to Produce Balanced Samples in Observational "
    "Studies},\n"
    "  author={Hainmueller, Jens},\n"
    "  journal={Political Analysis},\n"
    "  volume={20},\n"
    "  number={1},\n"
    "  pages={25--46},\n"
    "  year={2012},\n"
    "  publisher={Cambridge University Press}\n"
    "}"
)
