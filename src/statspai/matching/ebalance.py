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

import warnings
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult


@accepts_aliases(_strict=True, controls="covariates")
def ebalance(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    moments: Union[int, Sequence[int]] = 1,
    alpha: float = 0.05,
    vce: str = "mestimation",
    dof_adjust: bool = False,
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
    moments : int or sequence of int, default 1
        Number of moments to balance:
        - 1: means only
        - 2: means and variances
        - 3: means, variances, and skewness

        A sequence gives the order covariate by covariate, in the order of
        ``covariates`` (Stata ``ebalance ..., targets(2 2 1)``). A moment
        that repeats another one in the sample is balanced by it and is
        left out of the problem: the square of a 0/1 indicator is the
        indicator, so asking for its variance adds nothing. Such moments
        are listed in ``model_info['redundant_moments']``.
    alpha : float, default 0.05
    vce : {'mestimation', 'naive'}, default 'mestimation'
        Standard error of the ATT.

        - ``'mestimation'``: the sandwich variance of the stacked
          estimating equations -- treated moment means, the entropy-balancing
          dual conditions, and the two outcome means -- so the uncertainty
          from estimating the weights is propagated. This is the variance
          R ``WeightIt::lm_weightit(..., vcov = "asympt")`` reports for
          ``method = "ebal", estimand = "ATT"``, and it agrees with it to
          ~1e-9 (``tests/reference_parity/test_ebalance_weightit_parity.py``).
          Because the weights balance the covariates exactly, outcome
          variation that the covariates explain cancels out of the ATT, and
          this variance reflects that.
        - ``'naive'``: the pre-1.31 formula, a weighted two-sample variance
          that treats the weights as fixed. It ignores the balancing, so it
          overstates the standard error whenever the covariates predict the
          outcome (by about 2x on the Track B design: coverage 1.000, size
          0.000). Retained only to reproduce earlier numbers.

    dof_adjust : bool, default False
        How a second or third moment is matched. By default the raw
        moments of the controls are set to those of the treated, which is
        what R ``ebal`` and ``WeightIt`` do when handed ``x**2``. Stata's
        ``ebalance`` instead matches the *sample* variance: it scales the
        centred moment ``(x - mean_treated)**p`` by ``(n / (n - 1))**(p/2)``
        within each group, so the reweighted controls reproduce the treated
        variance with its ``n - 1`` divisor. ``dof_adjust=True`` follows
        Stata; with ``moments=1`` the two agree. The factor differs from
        one by ``O(1/n)`` and does not enter the first-order variance, so
        the standard error is computed the same way.

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
    spec = _moment_spec(covariates, moments)
    spec, redundant = _drop_redundant(X, spec)
    C_all = _constraint_functions(X, spec)
    if dof_adjust:
        C_all = _sample_moment_functions(X, spec, t_mask)
    targets = C_all[t_mask].mean(axis=0)
    C_matrix = C_all[c_mask]

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

    vce_l = str(vce).lower()
    if vce_l not in {"mestimation", "naive"}:
        from statspai.exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"vce must be 'mestimation' or 'naive', got {vce!r}.",
            recovery_hint="Use vce='mestimation' (default).",
            diagnostics={"vce": vce},
        )
    if vce_l == "mestimation" and weights_fallback:
        import warnings

        warnings.warn(
            "Entropy balancing did not find balancing weights, so the "
            "M-estimation standard error (which assumes the balance conditions "
            "hold) is unavailable; reporting the naive weighted-difference SE.",
            UserWarning,
            stacklevel=2,
        )
        vce_l = "naive"
    if vce_l == "mestimation":
        se = _mestimation_se(Y, D, C_all, weights)
    else:
        # Weighted two-sample variance with the weights held fixed.
        var_t = np.var(Y_t, ddof=1) / n_t
        var_c = (
            np.average((Y_c - np.average(Y_c, weights=weights)) ** 2, weights=weights)
            / n_c
        )
        se = float(np.sqrt(var_t + var_c))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = att / se if se > 0 else 0
    pvalue = float(2 * stats.norm.sf(abs(z)))
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
        "moments_balanced": moments if isinstance(moments, int) else list(moments),
        "dof_adjust": bool(dof_adjust),
        "redundant_moments": [f"{covariates[j]}^{power}" for j, power in redundant],
        "n_treated": int(n_t),
        "n_control": int(n_c),
        "max_weight": float(np.max(weights)),
        "eff_sample_size": float(1 / np.sum(weights**2)),
        "balance": balance,
        "weights": weights,
        "weights_full": weights_full,
        "max_standardized_moment_gap": max_imbalance,
        "weights_fallback": weights_fallback,
        "vce": vce_l,
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


def ebalance_weights(
    data: pd.DataFrame,
    treat: str,
    covariates: List[str],
    moments: Union[int, Sequence[int]] = 1,
    dof_adjust: bool = False,
) -> np.ndarray:
    """The entropy-balancing weights alone, one per row of ``data``.

    Treated units carry weight 1 and the control weights sum to the number
    of treated units (the layout of ``model_info['weights_full']`` of
    :func:`ebalance`, and of the variable Stata ``ebalance, generate()``
    creates). ``data`` must be free of missing values in the columns used.
    Raises ``NumericalInstability`` when no balancing weights exist.
    """
    from ..exceptions import NumericalInstability

    D = data[treat].to_numpy(dtype=float)
    X = data[covariates].to_numpy(dtype=float)
    t_mask = D == 1
    c_mask = D == 0
    spec, _ = _drop_redundant(X, _moment_spec(covariates, moments))
    C_all = _constraint_functions(X, spec)
    if dof_adjust:
        C_all = _sample_moment_functions(X, spec, t_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights, fallback = _solve_ebalance(
            C_all[c_mask], C_all[t_mask].mean(axis=0), int(c_mask.sum())
        )
    if fallback:
        raise NumericalInstability(
            "ebalance: no weights balance the requested moments; the treated "
            "moments are probably outside the range the controls can reach.",
            recovery_hint="Balance fewer moments or covariates, or check for "
            "a covariate with no overlap between the groups.",
            diagnostics={"covariates": list(covariates)},
        )
    out = np.ones(len(D), dtype=float)
    out[c_mask] = weights * t_mask.sum() / weights.sum()
    return out


def _moment_spec(
    covariates: Sequence[str], moments: Union[int, Sequence[int]]
) -> List[Tuple[int, int]]:
    """The balanced moments as ``(covariate index, power)``: every mean
    first, then every second moment, then every third."""
    from ..exceptions import MethodIncompatibility

    if isinstance(moments, (int, np.integer)):
        orders = [int(moments)] * len(covariates)
    else:
        orders = [int(m) for m in moments]
        if len(orders) != len(covariates):
            raise MethodIncompatibility(
                f"ebalance: moments has {len(orders)} entries for "
                f"{len(covariates)} covariates.",
                recovery_hint="Give one order per covariate, or a single "
                "integer for all of them.",
                diagnostics={"moments": orders, "covariates": list(covariates)},
            )
    if any(m not in (1, 2, 3) for m in orders):
        raise MethodIncompatibility(
            f"ebalance: moments must be 1, 2 or 3, got {orders}.",
            recovery_hint="Use moments=1 (means), 2 (and variances) or 3 "
            "(and skewness).",
            diagnostics={"moments": orders},
        )
    return [
        (j, power)
        for power in (1, 2, 3)
        for j, order in enumerate(orders)
        if order >= power
    ]


def _drop_redundant(
    X: np.ndarray, spec: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Leave out a moment that is a linear combination of the earlier ones
    (and the constant) in the whole sample.

    Such a moment is balanced as soon as the others are, so it carries no
    restriction; kept, it makes the dual Hessian singular and the solver
    fail. The check runs on treated and controls together: a relation that
    holds only among the controls is a real restriction and stays.
    """
    kept: List[Tuple[int, int]] = []
    dropped: List[Tuple[int, int]] = []
    basis = np.ones((X.shape[0], 1))
    for j, power in spec:
        col = X[:, j] ** power
        scale = float(np.linalg.norm(col - col.mean()))
        resid = col - basis @ np.linalg.lstsq(basis, col, rcond=None)[0]
        if scale == 0.0 or np.linalg.norm(resid) <= 1e-9 * scale:
            dropped.append((j, power))
            continue
        kept.append((j, power))
        basis = np.column_stack([basis, col])
    return kept, dropped


def _constraint_functions(X: np.ndarray, spec: List[Tuple[int, int]]) -> np.ndarray:
    """Raw-moment constraint functions ``c(X)``, one column per entry of
    ``spec``."""
    cols = [X[:, j] ** power for j, power in spec]
    return np.asarray(np.column_stack(cols), dtype=float)


def _mestimation_se(
    Y: np.ndarray, D: np.ndarray, C: np.ndarray, weights_c: np.ndarray
) -> float:
    """Sandwich SE of the entropy-balancing ATT from its estimating equations.

    Parameters are ``theta = (m, lambda, mu1, mu0)`` with, per unit,

    * ``D (c - m)``                      -- ``m``: treated means of ``c(X)``
    * ``(1 - D) w (c - m)``              -- balance, ``w = exp(lambda'(c - m))``
    * ``D (Y - mu1)``                    -- treated outcome mean
    * ``(1 - D) w (Y - mu0)``            -- reweighted control outcome mean

    and ``ATT = mu1 - mu0``. The variance is ``A^{-1} B A^{-T} / n`` with
    ``A`` the mean Jacobian (analytic) and ``B`` the mean outer product of
    the estimating functions. ``lambda`` is recovered from the solver's
    weights, which are exactly log-linear in ``c(X)``.
    """
    t = D == 1
    ctrl = ~t
    n, k = C.shape
    m = C[t].mean(axis=0)
    R = C - m
    # log w = lambda' R + const on the controls; recover lambda exactly.
    design = np.column_stack([np.ones(int(ctrl.sum())), R[ctrl]])
    coef, *_ = np.linalg.lstsq(design, np.log(weights_c), rcond=None)
    lam = coef[1:]
    w = np.zeros(n)
    w[ctrl] = np.exp(R[ctrl] @ lam)
    d = D.astype(float)
    mu1 = float(Y[t].mean())
    mu0 = float(np.sum(w * Y) / w.sum())

    psi = np.column_stack(
        [d[:, None] * R, w[:, None] * R, d * (Y - mu1), w * (Y - mu0)]
    )
    p = 2 * k + 2
    a = np.zeros((p, p))
    wr_sum = (w[:, None] * R).sum(axis=0)
    a[:k, :k] = -d.sum() * np.eye(k)
    a[k : 2 * k, :k] = -(w.sum() * np.eye(k) + np.outer(wr_sum, lam))
    a[k : 2 * k, k : 2 * k] = (w[:, None] * R).T @ R
    a[2 * k, 2 * k] = -d.sum()
    resid0 = w * (Y - mu0)
    a[2 * k + 1, :k] = -resid0.sum() * lam
    a[2 * k + 1, k : 2 * k] = resid0 @ R
    a[2 * k + 1, 2 * k + 1] = -w.sum()
    a /= n
    b = psi.T @ psi / n
    a_inv = np.linalg.inv(a)
    v = a_inv @ b @ a_inv.T / n
    g = np.zeros(p)
    g[2 * k], g[2 * k + 1] = 1.0, -1.0
    return float(np.sqrt(g @ v @ g))


def _sample_moment_functions(
    X: np.ndarray, spec: List[Tuple[int, int]], treated: np.ndarray
) -> np.ndarray:
    """Constraint functions in the scaling of Stata ``ebalance``: a moment
    of order ``p >= 2`` is ``(x - mean_treated)**p * (n_g / (n_g - 1))**(p/2)``
    with ``n_g`` the size of the unit's own group."""
    n_t = float(treated.sum())
    n_c = float((~treated).sum())
    ratio = np.where(treated, n_t / (n_t - 1.0), n_c / (n_c - 1.0))
    cols = []
    for j, power in spec:
        x = X[:, j]
        if power == 1:
            cols.append(x)
        else:
            cols.append((x - x[treated].mean()) ** power * ratio ** (power / 2.0))
    return np.asarray(np.column_stack(cols), dtype=float)


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
