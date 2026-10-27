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

__all__ = ["noisy_covariates", "gavish_donoho_rank"]

_ESTIMANDS = ("ATE", "LATE", "partially_linear")


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
    if estimand == "partially_linear":
        return np.column_stack([d, X])
    return np.column_stack([d[:, None] * X, (1.0 - d)[:, None] * X])


def _cross_fit(
    estimand: str,
    Y: np.ndarray,
    D: np.ndarray,
    U: Optional[np.ndarray],
    Z: np.ndarray,
    ell: np.ndarray,
    folds: np.ndarray,
    rank: Optional[int],
    add_constant: bool,
    clean: bool,
) -> Dict[str, Any]:
    """Cross-fitted doubly robust scores (``psi``) for one estimand.

    For LATE the scores of the reduced form (outcome on instrument) and
    of the first stage (treatment on instrument) are both returned.
    """
    n = Y.shape[0]
    a = U if estimand == "LATE" else D  # variable the dictionary is built on
    targets = [Y, D] if estimand == "LATE" else [Y]
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
        B = _dictionary(estimand, a[tr], X_hat, add_constant)
        P = _pinv(B)
        if estimand == "partially_linear":
            M = np.zeros(B.shape[1])
            M[0] = float(np.sum(ell[tr]))
        else:
            M = (
                _dictionary(estimand, np.ones(m), X_hat, add_constant)
                - _dictionary(estimand, np.zeros(m), X_hat, add_constant)
            ).T @ ell[tr]
        eta = P @ (P.T @ M)
        n_te = int(te.sum())
        b_obs = _dictionary(estimand, a[te], F_te, add_constant)
        alpha = b_obs @ eta
        alpha_hat[te] = alpha
        for t, target in enumerate(targets):
            beta = P @ target[tr]
            fitted = b_obs @ beta
            if estimand == "partially_linear":
                contrast = np.full(n_te, beta[0])
            else:
                b1 = _dictionary(estimand, np.ones(n_te), F_te, add_constant)
                b0 = _dictionary(estimand, np.zeros(n_te), F_te, add_constant)
                contrast = (b1 - b0) @ beta
            psi[te, t] = ell[te] * contrast + alpha * (target[te] - fitted)
    return {"psi": psi, "alpha": alpha_hat, "ranks": ranks}


def _summarise(
    estimand: str, psi: np.ndarray
) -> Tuple[float, float, np.ndarray, Dict[str, float]]:
    """Point estimate, SE, centred influence function and LATE pieces."""
    n = psi.shape[0]
    extra: Dict[str, float] = {}
    if estimand == "LATE":
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
    treat: str,
    covariates: Sequence[str],
    estimand: Literal["ATE", "LATE", "partially_linear"] = "ATE",
    instrument: Optional[str] = None,
    rank: Optional[int] = None,
    weights: Optional[str] = None,
    add_constant: bool = True,
    n_folds: int = 2,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
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
    treat : str
        Treatment column: 0/1 for ``estimand="ATE"`` and ``"LATE"``, any
        numeric dose for ``"partially_linear"``.
    covariates : list of str
        The corrupted covariates ``Z``. Missing values (NaN) are allowed
        and treated as missing at random within each column.
    estimand : {"ATE", "LATE", "partially_linear"}, default "ATE"
        ``"ATE"``: average treatment effect of a binary treatment.
        ``"LATE"``: local average treatment effect with a binary
        ``instrument``, the ratio of two cleaned doubly robust contrasts.
        ``"partially_linear"``: the coefficient on ``treat`` in
        ``Y = theta D + g(X) + e``.
    instrument : str, optional
        0/1 instrument column, required for ``estimand="LATE"``.
    rank : int, optional
        Number of principal components ``k`` kept when cleaning. ``None``
        chooses it in each training fold with the Gavish-Donoho hard
        threshold for unknown noise. The paper reports that
        over-choosing ``k`` costs little, under-choosing it biases.
    weights : str, optional
        Column of known unit weights (e.g. population) for a weighted
        estimand; they are normalised to mean one. The regression stays
        unweighted, as in the paper's weighted functionals.
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

    ``estimate_without_cleaning`` uses the full, untruncated filled
    covariates. When ``p`` is close to the number of training rows it is
    numerically erratic, which is the failure the cleaning step fixes.

    References
    ----------
    [@agarwal2026causal]
    [@gavish2014optimal]

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
    if estimand == "LATE" and instrument is None:
        raise MethodIncompatibility(
            "estimand='LATE' needs an instrument column.",
            recovery_hint="Pass instrument='<0/1 column>'.",
        )
    if estimand != "LATE" and instrument is not None:
        raise MethodIncompatibility(
            "instrument= is only used with estimand='LATE'.",
            recovery_hint="Set estimand='LATE' or drop instrument=.",
        )
    cols: Dict[str, Any] = {"y": y, "treat": treat, "covariates": covariates}
    if instrument is not None:
        cols["instrument"] = instrument
    if weights is not None:
        cols["weights"] = weights
    check_columns(data, cols, function="sp.noisy_covariates")

    core = (
        [y, treat]
        + ([instrument] if instrument else [])
        + ([weights] if weights else [])
    )
    keep = data[core].notna().all(axis=1).to_numpy()
    n_dropped = int((~keep).sum())
    df = data.iloc[np.flatnonzero(keep)]
    n = df.shape[0]
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    Z = df[covariates].to_numpy(dtype=float)
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(D))):
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
        U = df[instrument].to_numpy(dtype=float)
    split = U if estimand == "LATE" else D
    if estimand != "partially_linear":
        n1 = int(split.sum())
        if min(n1, n - n1) < 2 * n_folds:
            raise DataInsufficient(
                f"each arm needs at least {2 * n_folds} units; "
                f"got {n1} and {n - n1}.",
                recovery_hint="Use fewer folds or more data.",
            )
    elif float(np.std(D)) <= 1e-12 * max(1.0, float(np.max(np.abs(D)))):
        raise DataInsufficient(
            "treat does not vary.",
            recovery_hint="The partially linear coefficient needs variation "
            "in the treatment.",
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

    # Folds stratified on the binary arm so every training fold has both.
    rng = np.random.default_rng(seed)
    folds = np.empty(n, dtype=int)
    groups = [np.arange(n)]
    if estimand != "partially_linear":
        groups = [np.flatnonzero(split == 1), np.flatnonzero(split == 0)]
    for g in groups:
        perm = rng.permutation(g)
        folds[perm] = np.arange(perm.size) % n_folds

    fit = _cross_fit(estimand, Y, D, U, Z, ell, folds, rank, add_constant, True)
    theta, se, inf, extra = _summarise(estimand, fit["psi"])
    raw = _cross_fit(estimand, Y, D, U, Z, ell, folds, None, add_constant, False)
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
    if estimand == "LATE":
        fs, fs_se = extra["first_stage"], extra["first_stage_se"]
        if abs(fs) < 3 * fs_se:
            warnings.warn(
                f"noisy_covariates: weak first stage ({fs:.3g}, SE "
                f"{fs_se:.3g}); the LATE interval is unreliable.",
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
        "n_folds": int(n_folds),
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
    model_info.update(extra)
    label = {
        "ATE": "ATE",
        "LATE": "LATE",
        "partially_linear": "partially linear coefficient",
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
