"""
Causal inference with noisy covariates (Agarwal and Singh).

The covariates that drive treatment are observed with corruption::

    Z_ij = (X_ij + H_ij) * pi_ij,     E[H_ij | X] = 0,   pi_ij in {NA, 1}

where ``H`` may be heteroscedastic measurement error, randomised
rounding or a differential-privacy mechanism (Laplace noise), and
``pi`` marks missing entries. The key assumption is that the true
covariate matrix ``X`` is approximately low rank: the ``p`` columns are
approximate repeated measurements of a few latent signals.

The procedure has four steps, run with the roles of the folds reversed
(cross-fitting):

1. Data cleaning on the training fold. Missing entries are filled with
   0 and observed ones are divided by the column's observed fraction
   ``rho_j``; the filled matrix is truncated to its top ``k`` singular
   components.
2. Error-in-variable regression. OLS of ``Y`` on the dictionary
   ``b(D, X_hat) = (D X_hat, (1 - D) X_hat)`` in the training fold.
3. Error-in-variable balancing weight. The same normal equations with
   the counterfactual moment ``M = sum_i [b(1, X_hat_i) - b(0, X_hat_i)]``
   in place of ``B'Y``; the weight is ``alpha(D, Z) = b(D, Z) eta``.
4. On the test fold, without cleaning it ("implicit" cleaning: the
   filled test row times the training coefficients), the doubly robust
   score ``psi_i = gamma(1, Z_i) - gamma(0, Z_i) + alpha(D_i, Z_i)
   (Y_i - gamma(D_i, Z_i))`` is evaluated. The estimate is the mean of
   ``psi`` and the standard error ``sd(psi) / sqrt(n)``.

References
----------
[@agarwal2026causal]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._validate import require_binary_treatment
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from ..matrix_completion._core import _svt
from ._noisy_functionals import (
    FUNCTIONALS,
    functional_dictionary,
    functional_moment,
    parse_policy,
    user_folds,
)

__all__ = ["noisy_covariates", "gavish_donoho_rank"]

_ESTIMANDS = (
    "ATE",
    "LATE",
    "partially_linear",
    "partially_linear_iv",
    "elasticity",
    "policy_effect",
)
_IV = ("LATE", "partially_linear_iv")
_PLR = ("partially_linear", "partially_linear_iv")


# --------------------------------------------------------------------
# Cleaning primitives
# --------------------------------------------------------------------


def gavish_donoho_rank(M: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Rank chosen by the Gavish-Donoho hard threshold for unknown noise.

    Singular values above ``omega(beta) * median(s)`` are kept, with
    ``beta`` the aspect ratio (at most 1) and ``omega`` the cubic
    approximation ``0.56 b^3 - 0.95 b^2 + 1.82 b + 1.43``
    [@gavish2014optimal]. Returns the rank (at least 1), the threshold
    and the singular values.
    """
    M = np.asarray(M, dtype=float)
    s = np.linalg.svd(M, compute_uv=False)
    beta = min(M.shape) / max(M.shape)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = float(omega * np.median(s))
    k = int(np.sum(s > tau))
    return max(k, 1), tau, s


def _fill(Z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Missing entries to 0, observed ones divided by their column's rho."""
    out = np.where(np.isnan(Z), 0.0, Z / rho)
    return np.asarray(out)


def _pinv(B: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Pseudoinverse that discards singular values below ``rtol * s_max``.

    The cleaned matrix has exact rank ``k``; rounding leaves singular
    values of order ``1e-15 s_max`` beyond it, which numpy's default
    cutoff can keep and invert.
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((B.shape[1], B.shape[0]))
    keep = s > rtol * s[0]
    return np.asarray((Vt[keep].T / s[keep]) @ U[:, keep].T)


def _dictionary(
    estimand: str, d: np.ndarray, X: np.ndarray, add_constant: bool
) -> np.ndarray:
    """``b(D, X)``: arm-interacted for ATE / LATE, ``(D, X)`` for PLR."""
    if add_constant:
        X = np.column_stack([np.ones(X.shape[0]), X])
    if estimand in _PLR:
        return np.column_stack([d, X])
    return np.column_stack([d[:, None] * X, (1.0 - d)[:, None] * X])


def _cross_fit(
    estimand: str,
    Y: np.ndarray,
    D: Optional[np.ndarray],
    U: Optional[np.ndarray],
    Z: np.ndarray,
    ell: np.ndarray,
    folds: np.ndarray,
    rank: Optional[int],
    add_constant: bool,
    clean: bool,
    weighted_fit: bool = False,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-fitted doubly robust scores (``psi``) for one estimand.

    For LATE and the partially linear IV the scores of the reduced form
    (outcome on instrument) and of the first stage (treatment on
    instrument) are both returned. With ``weighted_fit`` the regression
    and the balancing-weight normal equations are weighted by ``ell``
    and so is the correction term of the score.
    """
    n = Y.shape[0]
    a = U if estimand in _IV else D  # variable the dictionary is built on
    targets = [Y, D] if estimand in _IV else [Y]
    psi = np.zeros((n, len(targets)))
    alpha_hat = np.zeros(n)
    ranks: List[int] = []
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        m = int(tr.sum())
        obs = ~np.isnan(Z[tr])
        rho = np.maximum(obs.mean(axis=0), 1.0 / m)
        F_tr = _fill(Z[tr], rho)
        F_te = _fill(Z[te], rho)
        if clean:
            k = rank if rank is not None else gavish_donoho_rank(F_tr)[0]
            k = min(k, min(F_tr.shape))
            X_hat = _svt(F_tr, 0.0, max_rank=k)[0]
        else:
            k = min(F_tr.shape)
            X_hat = F_tr
        ranks.append(int(k))
        a_tr = None if a is None else a[tr]
        a_te = None if a is None else a[te]
        if estimand in FUNCTIONALS:
            B = functional_dictionary(estimand, a_tr, X_hat, add_constant)
        else:
            B = _dictionary(estimand, a_tr, X_hat, add_constant)
        sw = np.sqrt(ell[tr])
        P = _pinv(B * sw[:, None]) if weighted_fit else _pinv(B)
        if estimand in FUNCTIONALS:
            m_tr = functional_moment(estimand, a_tr, X_hat, add_constant, policy)
            M = m_tr.T @ ell[tr]
        elif estimand in _PLR:
            M = np.zeros(B.shape[1])
            M[0] = float(np.sum(ell[tr]))
        else:
            M = (
                _dictionary(estimand, np.ones(m), X_hat, add_constant)
                - _dictionary(estimand, np.zeros(m), X_hat, add_constant)
            ).T @ ell[tr]
        eta = P @ (P.T @ M)
        n_te = int(te.sum())
        if estimand in FUNCTIONALS:
            b_obs = functional_dictionary(estimand, a_te, F_te, add_constant)
            m_te = functional_moment(estimand, a_te, F_te, add_constant, policy)
        else:
            b_obs = _dictionary(estimand, a_te, F_te, add_constant)
        alpha = b_obs @ eta
        alpha_hat[te] = alpha
        for t, target in enumerate(targets):
            beta = P @ (target[tr] * sw) if weighted_fit else P @ target[tr]
            fitted = b_obs @ beta
            if estimand in FUNCTIONALS:
                contrast = m_te @ beta
            elif estimand in _PLR:
                contrast = np.full(n_te, beta[0])
            else:
                b1 = _dictionary(estimand, np.ones(n_te), F_te, add_constant)
                b0 = _dictionary(estimand, np.zeros(n_te), F_te, add_constant)
                contrast = (b1 - b0) @ beta
            if weighted_fit:
                psi[te, t] = ell[te] * (contrast + alpha * (target[te] - fitted))
            else:
                psi[te, t] = ell[te] * contrast + alpha * (target[te] - fitted)
    return {"psi": psi, "alpha": alpha_hat, "ranks": ranks}


def _summarise(
    estimand: str, psi: np.ndarray
) -> Tuple[float, float, np.ndarray, Dict[str, float]]:
    """Point estimate, SE, centred influence function and LATE pieces."""
    n = psi.shape[0]
    extra: Dict[str, float] = {}
    if estimand in _IV:
        num, den = psi[:, 0].mean(), psi[:, 1].mean()
        theta = float(num / den)
        inf = (psi[:, 0] - theta * psi[:, 1]) / den
        extra["reduced_form"] = float(num)
        extra["first_stage"] = float(den)
        extra["first_stage_se"] = float(np.sqrt(np.mean((psi[:, 1] - den) ** 2) / n))
    else:
        theta = float(psi[:, 0].mean())
        inf = psi[:, 0] - theta
    se = float(np.sqrt(np.mean(inf**2) / n))
    return theta, se, inf, extra


# --------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------


def noisy_covariates(
    data: pd.DataFrame,
    y: str,
    treat: Optional[str],
    covariates: Sequence[str],
    estimand: Literal[
        "ATE",
        "LATE",
        "partially_linear",
        "partially_linear_iv",
        "elasticity",
        "policy_effect",
    ] = "ATE",
    instrument: Optional[str] = None,
    rank: Optional[int] = None,
    weights: Optional[str] = None,
    add_constant: bool = True,
    n_folds: int = 2,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
    policy: Optional[Dict[str, Any]] = None,
    weighted_fit: bool = False,
    folds: Optional[Any] = None,
) -> CausalResult:
    """
    Treatment effects when the covariates are noisy, missing or privatised.

    Agarwal-Singh data cleaning-adjusted inference: principal-component
    cleaning of the corrupted covariates, an error-in-variable regression
    and an error-in-variable balancing weight, combined in a cross-fitted
    doubly robust score. The confidence interval accounts for the
    cleaning step. No model of the noise is needed; the key assumption
    is that the true covariates are approximately low rank.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str or None
        Treatment column: 0/1 for ``estimand="ATE"`` and ``"LATE"``, any
        numeric dose for ``"partially_linear"``, ``"partially_linear_iv"``
        and ``"elasticity"``. For ``"policy_effect"`` it is optional: an
        exactly observed regressor that enters the regression linearly
        and that ``policy`` may move; pass ``None`` to regress on the
        covariates alone, as in the paper's Example E.3.
    covariates : list of str
        The corrupted covariates ``Z``. Missing values (NaN) are allowed
        and treated as missing at random within each column.
    estimand : str, default "ATE"
        ``"ATE"``: average treatment effect of a binary treatment.
        ``"LATE"``: local average treatment effect with a binary
        ``instrument``, the ratio of two cleaned doubly robust contrasts.
        ``"partially_linear"``: the coefficient on ``treat`` in
        ``Y = theta D + g(X) + e``.
        ``"partially_linear_iv"``: the partially linear IV coefficient
        with a continuous ``instrument`` U, the ratio of the partially
        linear coefficients of ``y`` on U and of ``treat`` on U (Example
        E.6; the paper's Autor-Dorn-Hanson application).
        ``"elasticity"``: the average derivative ``E[d E[Y|D,X] / dD]``
        of a continuous ``treat`` (Example E.4), with the dictionary
        ``(X, D X, D^2 X)``; an average elasticity when ``y`` and
        ``treat`` are in logs.
        ``"policy_effect"``: ``E[E[Y|t(X)] - E[Y|X]]`` for the affine
        counterfactual ``t`` given in ``policy`` (Example E.3), with a
        regression linear in the covariates.
    instrument : str, optional
        Instrument column, required for ``estimand="LATE"`` (0/1) and
        ``"partially_linear_iv"`` (numeric).
    rank : int, optional
        Number of principal components ``k`` kept when cleaning. ``None``
        chooses it in each training fold with the Gavish-Donoho hard
        threshold for unknown noise. The paper reports that
        over-choosing ``k`` costs little, under-choosing it biases.
    weights : str, optional
        Column of known unit weights (e.g. population) for a weighted
        estimand; they are normalised to mean one. By default the
        regression stays unweighted and only the functional is weighted.
        For the partially linear estimands the functional's moment is
        constant, so the default estimates (up to the weights' mean in
        each training fold) the unweighted coefficient; pass
        ``weighted_fit=True`` for the weighted-regression coefficient
        that Stata's ``[aw=]`` reports.
    add_constant : bool, default True
        Append an exactly observed constant column after cleaning, so
        each arm gets its own intercept. The paper's dictionary has no
        separate constant; with centred covariates and an outcome without
        an intercept the two coincide.
    n_folds : int, default 2
        Cross-fitting folds (the paper uses two, swapping train and test).
    seed : int, optional
        Seed for the fold assignment.
    alpha : float, default 0.05
    policy : dict, optional
        Required for ``estimand="policy_effect"``: ``{column: c}`` moves
        that column to ``x + c``, ``{column: (a, c)}`` to ``a x + c``.
        Columns not named stay as they are. Keys are covariates or
        ``treat``. The counterfactual applies to the true covariates;
        their cleaned values stand in for them.
    weighted_fit : bool, default False
        With ``weights``: also weight the regression and the balancing
        weight's normal equations, and multiply the score's correction
        term by the weight. This is the authors' replication code for
        the Autor-Dorn-Hanson application. The default keeps the
        regression unweighted and weights only the functional; both are
        doubly robust for the weighted estimand and agree when the
        regression is correctly specified.
    folds : str or array-like, optional
        Fold label of each row (a column name or a length-n array), to
        reproduce an external cross-fitting split. Overrides ``n_folds``
        and ``seed``. Labels refer to the rows that remain after rows
        with missing ``y``, ``treat``, ``instrument`` or ``weights`` are
        dropped.

    Returns
    -------
    CausalResult
        ``model_info`` holds the rank used in each fold, the singular
        values of the filled covariate matrix, the observed fraction of
        each covariate, summaries of the balancing weight, and
        ``estimate_without_cleaning`` (the same cross-fitted score with
        no truncation) for comparison. For LATE it also holds the
        reduced form and first stage.

    Notes
    -----
    The rank rule for ``rank=None`` is not part of the paper, which
    treats ``k`` as a hyperparameter read off the scree plot. The
    Gavish-Donoho threshold assumes white noise; under strong
    heteroscedasticity or heavy missingness, inspect
    ``model_info["singular_values"]`` and pass ``rank``.

    Each unit's own noise inside the ``k`` retained directions is not
    removed; it is averaged over the ``p`` covariates. The cleaning error
    therefore has a term of order ``1/p``, and the paper's inference
    result asks for ``n^(2/3) <= p <= n^(3/2)`` up to constants.
    With few covariates per latent factor and strong confounding the
    estimate keeps an errors-in-variables bias that the interval does
    not cover. ``model_info["p_in_theory_range"]`` reports the check.

    Only the training fold is cleaned. Test rows are filled and scaled
    but not projected; their predictions use the training coefficients,
    which keeps test scores independent given the training fold.

    The variance is the sample variance of the scores. When the
    covariate distribution differs across units it is conservative.
    Missingness must not depend on the outcome (no sample selection);
    deterministic rounding (interval censoring) is outside the model.

    The published Econometrica version is a shorter version of the
    working paper. The procedure here follows the working paper's
    algorithm, including missing values and the LATE and
    partially linear examples. Taken literally, the paper's ATE
    dictionary has no intercept, which cannot fit its own simulation
    design (the treatment shifts the outcome by a constant); the
    constant is therefore appended after cleaning. Inside the cleaning
    step it would be truncated away. On the paper's simulation design
    (n = p = 100, rank 5, 20% noise, k = 5), 300 replications give a
    mean of 2.24 (truth 2.2), a mean standard error of 0.38 and 92%
    coverage at the 95% level; the paper reports 2.22, 0.35 and 96%.
    Coverage reaches 94-95% at n = 400-800.

    On the paper's Autor-Dorn-Hanson application
    (``estimand="partially_linear_iv"``, ``rank=5``, population weights
    with ``weighted_fit=True`` and the authors' folds) the estimate
    equals the authors' replication code: -0.61799 (SE 0.06331) on the
    raw covariates and -0.48986 (SE 0.07193) on standardised ones, and
    the same code on noisy, missing and privatised covariates agrees to
    1e-10 or better. With the default (unweighted) fit the same data give
    -0.448 (SE 0.154): there the two weightings target different
    projections.

    ``"policy_effect"`` is identified only for counterfactuals that stay
    in the span of the low-rank covariates. Shifting a single covariate
    of a low-rank matrix leaves that span, and the estimate then depends
    on the minimum-norm regression coefficients; shift along a loading
    direction (or rescale all covariates) instead.

    ``estimate_without_cleaning`` uses the full, untruncated filled
    covariates. When ``p`` is close to the number of training rows it is
    numerically erratic, which is the failure the cleaning step fixes.

    References
    ----------
    [@agarwal2026causal]
    [@gavish2014optimal]
    [@autor2013china]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, p, r = 400, 80, 3
    >>> X = rng.normal(size=(n, r)) @ rng.normal(size=(r, p))
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-0.3 * X[:, 0])))
    >>> yv = 2.0 * d + X[:, 0] + rng.normal(size=n)
    >>> Z = X + rng.laplace(scale=0.5, size=(n, p))  # privacy noise
    >>> Z[rng.uniform(size=(n, p)) < 0.2] = np.nan  # 20% missing
    >>> df = pd.DataFrame(Z, columns=[f"z{j}" for j in range(p)])
    >>> df["y"], df["d"] = yv, d
    >>> res = sp.noisy_covariates(df, "y", "d", [f"z{j}" for j in range(p)])
    >>> bool(abs(res.estimate - 2.0) < 4 * res.se)
    True

    Average derivative of a continuous dose (an elasticity in logs):

    >>> dose = X[:, 0] / 3 + rng.normal(size=n)
    >>> df["y2"] = 0.5 * dose + 0.2 * dose**2 + X[:, 1] + rng.normal(size=n)
    >>> df["dose"] = dose
    >>> el = sp.noisy_covariates(df, "y2", "dose", [f"z{j}" for j in range(p)],
    ...                          estimand="elasticity", rank=3)
    >>> truth = float(np.mean(0.5 + 0.4 * dose))
    >>> bool(abs(el.estimate - truth) < 4 * el.se)
    True
    """
    if estimand not in _ESTIMANDS:
        raise MethodIncompatibility(
            f"estimand must be one of {_ESTIMANDS}; got {estimand!r}.",
            recovery_hint="Use 'ATE' for a 0/1 treatment, 'LATE' with an "
            "instrument, or 'partially_linear' for a continuous treatment.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1).", recovery_hint="Pass e.g. alpha=0.05."
        )
    if int(n_folds) != n_folds or n_folds < 2:
        raise MethodIncompatibility(
            f"n_folds must be an integer >= 2; got {n_folds!r}.",
            recovery_hint="Use n_folds=2 as in the paper.",
        )
    covariates = list(covariates)
    if not covariates:
        raise MethodIncompatibility(
            "at least one covariate is required.",
            recovery_hint="Pass the noisy covariate columns in covariates=.",
        )
    if estimand in _IV and instrument is None:
        raise MethodIncompatibility(
            f"estimand={estimand!r} needs an instrument column.",
            recovery_hint="Pass instrument='<column>'.",
        )
    if estimand not in _IV and instrument is not None:
        raise MethodIncompatibility(
            "instrument= is only used with estimand='LATE' or "
            "'partially_linear_iv'.",
            recovery_hint="Set one of those estimands or drop instrument=.",
        )
    if treat is None and estimand != "policy_effect":
        raise MethodIncompatibility(
            f"estimand={estimand!r} needs a treat column.",
            recovery_hint="treat=None is only allowed for " "estimand='policy_effect'.",
        )
    if (policy is None) == (estimand == "policy_effect"):
        raise MethodIncompatibility(
            "policy= is required with, and only used with, "
            "estimand='policy_effect'.",
            recovery_hint="Pass policy={'column': shift} with "
            "estimand='policy_effect', or drop policy=.",
        )
    if weighted_fit and weights is None:
        raise MethodIncompatibility(
            "weighted_fit=True needs weights=.",
            recovery_hint="Pass the weight column, or drop weighted_fit.",
        )
    cols: Dict[str, Any] = {"y": y, "covariates": covariates}
    if treat is not None:
        cols["treat"] = treat
    if instrument is not None:
        cols["instrument"] = instrument
    if weights is not None:
        cols["weights"] = weights
    if isinstance(folds, str):
        cols["folds"] = folds
    check_columns(data, cols, function="sp.noisy_covariates")
    pol = parse_policy(policy, covariates, treat) if policy is not None else None

    core = (
        [y]
        + ([treat] if treat is not None else [])
        + ([instrument] if instrument else [])
        + ([weights] if weights else [])
    )
    keep = data[core].notna().all(axis=1).to_numpy()
    n_dropped = int((~keep).sum())
    df = data.iloc[np.flatnonzero(keep)]
    n = df.shape[0]
    Y = df[y].to_numpy(dtype=float)
    D = None if treat is None else df[treat].to_numpy(dtype=float)
    Z = df[covariates].to_numpy(dtype=float)
    if not (np.all(np.isfinite(Y)) and (D is None or np.all(np.isfinite(D)))):
        raise MethodIncompatibility(
            "y and treat must be finite.",
            recovery_hint="Drop or recode infinite values first.",
        )
    if np.any(np.isinf(Z)):
        raise MethodIncompatibility(
            "covariates contain infinite values.",
            recovery_hint="Recode infinite values as NaN (missing) or drop them.",
        )
    empty = [c for c, ok in zip(covariates, (~np.isnan(Z)).any(axis=0)) if not ok]
    if empty:
        raise DataInsufficient(
            f"covariates with no observed value: {empty}.",
            recovery_hint="Drop these columns; they carry no information.",
        )

    U: Optional[np.ndarray] = None
    if estimand in ("ATE", "LATE"):
        require_binary_treatment(df[treat], function="noisy_covariates")
    if estimand == "LATE":
        require_binary_treatment(
            df[instrument], function="noisy_covariates", argument="instrument"
        )
    if estimand in _IV:
        U = df[instrument].to_numpy(dtype=float)
        if not np.all(np.isfinite(U)):
            raise MethodIncompatibility(
                "instrument must be finite.",
                recovery_hint="Drop or recode infinite values first.",
            )
    split = U if estimand in _IV else D
    binary = estimand in ("ATE", "LATE")
    if binary:
        n1 = int(split.sum())
        if min(n1, n - n1) < 2 * n_folds:
            raise DataInsufficient(
                f"each arm needs at least {2 * n_folds} units; "
                f"got {n1} and {n - n1}.",
                recovery_hint="Use fewer folds or more data.",
            )
    elif split is not None:
        what = "instrument" if estimand in _IV else "treat"
        if float(np.std(split)) <= 1e-12 * max(1.0, float(np.max(np.abs(split)))):
            raise DataInsufficient(
                f"{what} does not vary.",
                recovery_hint=f"The {estimand} estimand needs variation in "
                f"the {what}.",
            )
        if estimand == "elasticity" and np.unique(split).size < 3:
            raise DataInsufficient(
                "treat takes fewer than three values; the quadratic "
                "dictionary of the elasticity estimand is not identified.",
                recovery_hint="Use estimand='ATE' for a binary treatment or "
                "'partially_linear' for a constant effect.",
            )
    if n < 2 * n_folds:
        raise DataInsufficient(
            f"{n} complete rows are too few for {n_folds} folds.",
            recovery_hint="Use fewer folds or more data.",
        )

    if weights is not None:
        ell = df[weights].to_numpy(dtype=float)
        if np.any(ell < 0) or not np.all(np.isfinite(ell)) or ell.sum() <= 0:
            raise MethodIncompatibility(
                "weights must be finite, non-negative and not all zero.",
                recovery_hint="Check the weight column.",
            )
        ell = ell / ell.mean()
    else:
        ell = np.ones(n)

    if folds is not None:
        fold_id = user_folds(data, folds, keep, split if binary else None)
        n_folds_used = int(np.unique(fold_id).size)
        m_min = n - int(np.max(np.bincount(fold_id)))
    else:
        # Folds stratified on the binary arm so every training fold has both.
        rng = np.random.default_rng(seed)
        fold_id = np.empty(n, dtype=int)
        groups = [np.arange(n)]
        if binary:
            groups = [np.flatnonzero(split == 1), np.flatnonzero(split == 0)]
        for g in groups:
            perm = rng.permutation(g)
            fold_id[perm] = np.arange(perm.size) % n_folds
        n_folds_used = int(n_folds)
        m_min = n - int(np.ceil(n / n_folds))
    max_rank = min(m_min, Z.shape[1])
    if rank is not None:
        if int(rank) != rank or rank < 1:
            raise MethodIncompatibility(
                f"rank must be a positive integer; got {rank!r}.",
                recovery_hint="Pass rank=None to choose it from the data.",
            )
        if rank > max_rank:
            raise MethodIncompatibility(
                f"rank={rank} exceeds min(training rows, covariates) = " f"{max_rank}.",
                recovery_hint="Choose a rank below the number of covariates; "
                "the method needs the covariates to be approximately low rank.",
            )
        rank = int(rank)

    opts = dict(weighted_fit=bool(weighted_fit), policy=pol)
    fit = _cross_fit(
        estimand, Y, D, U, Z, ell, fold_id, rank, add_constant, True, **opts
    )
    theta, se, inf, extra = _summarise(estimand, fit["psi"])
    raw = _cross_fit(
        estimand, Y, D, U, Z, ell, fold_id, None, add_constant, False, **opts
    )
    theta_raw, se_raw, _, _ = _summarise(estimand, raw["psi"])

    obs_frac = (~np.isnan(Z)).mean(axis=0)
    rho_full = np.maximum(obs_frac, 1.0 / n)
    k_full, tau, s_full = gavish_donoho_rank(_fill(Z, rho_full))
    ranks = fit["ranks"]
    k_used = max(ranks)
    if k_used > 0.5 * min(m_min, Z.shape[1]) and Z.shape[1] >= 4:
        warnings.warn(
            f"noisy_covariates: the cleaning step kept {k_used} of "
            f"{Z.shape[1]} components; the covariates do not look low rank, "
            "which is the method's key assumption.",
            AssumptionWarning,
            stacklevel=2,
        )
    if estimand in _IV:
        fs, fs_se = extra["first_stage"], extra["first_stage_se"]
        if abs(fs) < 3 * fs_se:
            warnings.warn(
                f"noisy_covariates: weak first stage ({fs:.3g}, SE "
                f"{fs_se:.3g}); the {estimand} interval is unreliable.",
                AssumptionWarning,
                stacklevel=2,
            )

    crit = float(stats.norm.ppf(1 - alpha / 2))
    if np.isfinite(se) and se > 0:
        pvalue = float(2 * stats.norm.sf(abs(theta / se)))
        ci = (theta - crit * se, theta + crit * se)
    else:
        pvalue, ci = np.nan, (np.nan, np.nan)

    a_hat = fit["alpha"]
    model_info: Dict[str, Any] = {
        "estimand": estimand,
        "rank_by_fold": ranks,
        "rank_rule": "user" if rank is not None else "gavish_donoho",
        "rank_full_sample_rule": int(k_full),
        "singular_values": s_full,
        "threshold": tau,
        "observed_fraction": dict(zip(covariates, obs_frac.tolist())),
        "rho_min": float(obs_frac.min()),
        "n_covariates": len(covariates),
        "p_in_theory_range": bool(n ** (2 / 3) <= len(covariates) <= n ** (3 / 2)),
        "n_folds": n_folds_used,
        "fold_assignment": "user" if folds is not None else "random",
        "weighted_fit": bool(weighted_fit),
        "n_dropped_rows": n_dropped,
        "add_constant": bool(add_constant),
        "weighted": weights is not None,
        "balancing_weight_mean_abs": float(np.mean(np.abs(a_hat))),
        "balancing_weight_max_abs": float(np.max(np.abs(a_hat))),
        "balancing_weight_share_negative": float(np.mean(a_hat < 0)),
        "estimate_without_cleaning": theta_raw,
        "se_without_cleaning": se_raw,
        "citation_key": "agarwal2026causal",
    }
    if pol is not None:
        model_info["policy"] = pol["record"]
    model_info.update(extra)
    label = {
        "ATE": "ATE",
        "LATE": "LATE",
        "partially_linear": "partially linear coefficient",
        "partially_linear_iv": "partially linear IV coefficient",
        "elasticity": "average derivative (elasticity)",
        "policy_effect": "average policy effect",
    }[estimand]
    return CausalResult(
        method="Noisy-covariate DR (Agarwal-Singh)",
        estimand=label,
        estimate=theta,
        se=se,
        pvalue=pvalue,
        ci=(float(ci[0]), float(ci[1])),
        alpha=float(alpha),
        n_obs=int(n),
        model_info=model_info,
        _influence_funcs=inf,
        _citation_key="agarwal2026causal",
    )


# Kept in sync with paper.bib (key agarwal2026causal).
CausalResult._CITATIONS["agarwal2026causal"] = (
    "@article{agarwal2026causal,\n"
    "  title={Causal Inference With Noisy Covariates: Heteroscedastic "
    "Measurement Error and Differential Privacy},\n"
    "  author={Agarwal, Anish and Singh, Rahul},\n"
    "  journal={Econometrica},\n"
    "  volume={94},\n"
    "  number={5},\n"
    "  pages={1887--1906},\n"
    "  year={2026},\n"
    "  doi={10.3982/ecta23897}\n"
    "}"
)
