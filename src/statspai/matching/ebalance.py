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

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

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
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(40, 10, n)
    >>> income = rng.normal(50, 15, n)
    >>> education = rng.integers(8, 20, n).astype(float)
    >>> ps = 1 / (1 + np.exp(-(0.03 * (age - 40) + 0.02 * (income - 50))))
    >>> treated = (rng.uniform(size=n) < ps).astype(int)
    >>> outcome = (2.0 * treated + 0.1 * age + 0.05 * income
    ...            + 0.2 * education + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({'outcome': outcome, 'treated': treated,
    ...                    'age': age, 'income': income,
    ...                    'education': education})
    >>> result = sp.ebalance(df, y='outcome', treat='treated',
    ...                      covariates=['age', 'income', 'education'])
    >>> bool(np.isfinite(result.estimate))
    True
    >>> 'balance' in result.model_info  # post-weighting balance table
    True

    Notes
    -----
    Entropy balancing solves:

    .. math::
        \\min_w \\sum_i w_i \\log(w_i / q_i)

    subject to balance constraints (weighted moments match) and
    normalization (weights sum to 1).

    Unlike PSM, this guarantees exact balance on specified moments
    without iteration or caliper tuning.

    If the dual optimizer raises, uniform control weights are used as a
    fallback; a ``ConvergenceWarning`` is emitted and
    ``model_info['weights_fallback']`` is set to True.

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
    weights, weights_fallback = _solve_ebalance(C_matrix, targets, n_c)

    # Verify balance constraints are satisfied. The check is on the
    # *standardised* moment gap: an absolute threshold is meaningless when
    # one constraint is a 0/1 indicator and the next is annual earnings in
    # dollars, and would either never fire or always fire depending on
    # the units the caller happened to use.
    achieved = C_matrix.T @ weights
    moment_scale = C_matrix.std(axis=0)
    moment_scale = np.where(moment_scale > 0, moment_scale, 1.0)
    max_imbalance = float(np.max(np.abs(achieved - targets) / moment_scale))
    if max_imbalance > 1e-6:
        import warnings

        warnings.warn(
            f"Entropy balancing did not fully converge (max standardised "
            f"moment imbalance = {max_imbalance:.2e}). Entropy balancing "
            f"is supposed to match the targeted moments exactly, so treat "
            f"this result as unbalanced. Consider reducing the number of "
            f"covariates or moments.",
            UserWarning,
        )

    # ATT = mean(Y_t) - weighted_mean(Y_c)
    att = float(np.mean(Y_t) - np.average(Y_c, weights=weights))

    # SE via weighted variance
    var_t = np.var(Y_t, ddof=1) / n_t
    var_c = (
        np.average((Y_c - np.average(Y_c, weights=weights)) ** 2, weights=weights) / n_c
    )
    se = float(np.sqrt(var_t + var_c))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = att / se if se > 0 else 0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (att - z_crit * se, att + z_crit * se)

    # Balance check
    balance = _balance_check(X_t, X_c, weights, covariates)

    # ``weights`` holds the CONTROL weights only (length n_control), which
    # is the ebal convention. ``weights_full`` is the same solution laid
    # out over every retained row (treated units carry weight 1), which is
    # what callers need to join weights back onto the input frame.
    weights_full = np.ones(len(df), dtype=float)
    weights_full[c_mask] = weights * n_t / weights.sum()

    model_info = {
        "method": "Entropy Balancing",
        "moments_balanced": moments,
        "n_treated": int(n_t),
        "n_control": int(n_c),
        "max_weight": float(np.max(weights)),
        "eff_sample_size": float(1 / np.sum(weights**2)),
        "balance": balance,
        "weights": weights,
        "weights_full": weights_full,
        "max_standardized_moment_gap": max_imbalance,
        "weights_fallback": weights_fallback,
    }

    return CausalResult(
        method="Entropy Balancing (Hainmueller 2012)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(df),
        detail=balance,
        model_info=model_info,
        _citation_key="ebalance",
    )


def _build_constraints(
    X_t: np.ndarray,
    X_c: np.ndarray,
    covariates: List[str],
    moments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build moment targets and constraint matrix."""
    k = len(covariates)
    targets: List[float] = []
    C_cols: List[np.ndarray] = []

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

    C_matrix = np.asarray(np.column_stack(C_cols), dtype=float)
    targets_arr = np.asarray(targets, dtype=float)

    return targets_arr, C_matrix


def _solve_ebalance(
    C: np.ndarray,
    targets: np.ndarray,
    n_c: int,
    max_iter: int = 200,
    tol: float = 1e-12,
    base_weights: "np.ndarray | None" = None,
) -> Tuple[np.ndarray, bool]:
    """Solve entropy balancing via its Lagrange dual (Newton + line search).

    Entropy balancing's defining property is that the reweighted moments
    match the targets *exactly*, so the solver must be driven to a true
    stationary point rather than merely to a good objective value.

    Writing ``A_i = C_i - targets``, the weights are
    ``w_i ∝ q_i exp(-A_i'λ)`` and the dual objective

        F(λ) = log Σ_i q_i exp(-A_i'λ)

    is convex with ``∇F = -A'w`` and ``∇²F = A'diag(w)A - (A'w)(A'w)'``.
    Exact balance is precisely ``∇F = 0``, so Newton's method with a
    backtracking line search is run until ``‖A'w‖_∞`` is at tolerance.

    The constraint columns are additionally divided by their standard
    deviation before solving. Without that rescaling the dual Hessian is
    badly conditioned whenever covariates live on different scales (a
    dollar-denominated earnings variable next to a 0/1 indicator), and a
    quasi-Newton method stops early — leaving moment gaps of order 1e-3
    relative, which silently breaks the estimator's contract.

    Returns ``(weights, fallback)`` where ``fallback=True`` means no
    balancing solution was found and uniform control weights were
    returned.
    """
    m = len(targets)
    A_raw = np.asarray(C, dtype=float) - np.asarray(targets, dtype=float)
    scale = A_raw.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    A = A_raw / scale

    if base_weights is None:
        q = np.full(n_c, 1.0 / n_c)
    else:
        q = np.asarray(base_weights, dtype=float)
        q = q / q.sum()

    log_q = np.log(np.maximum(q, 1e-300))

    def _weights(lam: np.ndarray) -> np.ndarray:
        z = log_q - A @ lam
        z -= z.max()  # log-sum-exp stabilisation
        w = np.exp(z)
        return w / w.sum()

    def _objective(lam: np.ndarray) -> float:
        z = log_q - A @ lam
        zmax = z.max()
        return float(zmax + np.log(np.sum(np.exp(z - zmax))))

    lam = np.zeros(m)
    w = _weights(lam)
    fallback = False

    for _ in range(max_iter):
        grad = -(A.T @ w)
        if np.max(np.abs(grad)) < tol:
            break
        # Hessian of the log-sum-exp dual (the weighted covariance of A).
        Aw = A * w[:, None]
        hess = A.T @ Aw - np.outer(A.T @ w, A.T @ w)
        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hess) @ grad
        if not np.all(np.isfinite(step)):
            fallback = True
            break
        # Backtracking: the dual is convex, so any descent direction with a
        # short enough step decreases F.
        f0 = _objective(lam)
        t = 1.0
        for _ls in range(60):
            cand = lam + t * step
            if _objective(cand) <= f0:
                break
            t *= 0.5
        else:  # pragma: no cover - only on a numerically hopeless problem
            fallback = True
            break
        lam = lam + t * step
        w = _weights(lam)
    else:
        # Ran out of iterations without hitting the gradient tolerance.
        fallback = np.max(np.abs(A.T @ w)) > 1e-6

    if fallback:
        from ..exceptions import ConvergenceWarning
        from ..exceptions import warn as _sp_warn

        _sp_warn(
            ConvergenceWarning,
            "ebalance: the entropy-balancing dual did not converge; the "
            "treated moments are probably outside the convex hull of the "
            "control moments. Falling back to uniform control weights.",
            recovery_hint=(
                "Reduce the number of covariates or moments, drop control "
                "units far outside the treated covariate range, or check "
                "for a covariate with no overlap between the groups."
            ),
            stacklevel=4,
        )
        return np.ones(n_c, dtype=float) / n_c, True

    return np.asarray(w, dtype=float), False


def _balance_check(
    X_t: np.ndarray,
    X_c: np.ndarray,
    weights: np.ndarray,
    covariates: List[str],
) -> pd.DataFrame:
    """Check balance before/after reweighting."""
    rows = []
    for j, cov in enumerate(covariates):
        mean_t = np.mean(X_t[:, j])
        mean_c_raw = np.mean(X_c[:, j])
        mean_c_w = np.average(X_c[:, j], weights=weights)

        sd_pooled = np.sqrt((np.var(X_t[:, j], ddof=1) + np.var(X_c[:, j], ddof=1)) / 2)
        sd_pooled = max(sd_pooled, 1e-10)

        smd_before = (mean_t - mean_c_raw) / sd_pooled
        smd_after = (mean_t - mean_c_w) / sd_pooled

        rows.append(
            {
                "covariate": cov,
                "mean_treated": round(mean_t, 4),
                "mean_control_raw": round(mean_c_raw, 4),
                "mean_control_balanced": round(mean_c_w, 4),
                "smd_before": round(smd_before, 4),
                "smd_after": round(smd_after, 4),
            }
        )

    return pd.DataFrame(rows)


# Citation
CausalResult._CITATIONS["ebalance"] = (
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
