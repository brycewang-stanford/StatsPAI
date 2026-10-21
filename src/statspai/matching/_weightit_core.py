"""Estimation kernels behind :func:`sp.weightit`.

One treatment model for every categorical case: a binary treatment is a
two-level multinomial one, so the multinomial-logit fit, the
just-identified CBPS root and the per-group entropy balancing below serve
both. Continuous treatments have their own two kernels (normal GPS and
entropy balancing of treatment-covariate covariances).

Conventions were fixed against R ``WeightIt`` 1.7.0 output on shared
fixtures (WeightIt is GPL: used as a black box, documentation and outputs
only):

* multinomial ATE / ATT weights are ``1 / P(A = a | X)`` and
  ``P(A = focal | X) / P(A = a | X)``;
* just-identified CBPS makes the weighted sums of ``(1, X)`` equal across
  the treatment groups (ATE) or equal to the focal group's sums (ATT); the
  ATT solution is the entropy-balancing one;
* entropy balancing weights each group to the full-sample means (ATE) or
  the focal-group means (ATT), each group's weights summing to ``n`` (ATE)
  or ``n_focal`` (ATT);
* continuous GPS weights are ``phi(z_A) / phi(r / sigma)``: the standard
  normal density of the standardized treatment over that of the
  standardized residual, both scales the maximum-likelihood ones
  (divisor ``n``). This is the density ratio ``f(A) / f(A | X)`` times the
  constant ``sd(A) / sigma``, so weighted means are unchanged by it;
* continuous entropy balancing sets the weighted covariance of the
  treatment with each balanced covariate term to zero while holding the
  weighted means (and, with ``d_moments``, higher powers) of the treatment
  and the covariates at their sample values; weights sum to ``n``.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import ConvergenceWarning, NumericalInstability
from ..exceptions import warn as _sp_warn
from .ebalance import _drop_redundant, _solve_ebalance


def balance_terms(
    X: np.ndarray, names: Sequence[str], moments: int, interactions: bool
) -> Tuple[np.ndarray, List[str]]:
    """Covariate terms to balance: powers up to ``moments`` of each
    non-binary column and, with ``interactions``, all pairwise products.

    Powers of a 0/1 column equal the column and are skipped (WeightIt
    ignores ``moments > 1`` for categorical covariates the same way).
    """
    cols = [X[:, j] for j in range(X.shape[1])]
    labels = list(names)
    binary = [np.isin(X[:, j], (0.0, 1.0)).all() for j in range(X.shape[1])]
    for power in range(2, int(moments) + 1):
        for j in range(X.shape[1]):
            if not binary[j]:
                cols.append(X[:, j] ** power)
                labels.append(f"{names[j]}^{power}")
    if interactions:
        for j in range(X.shape[1]):
            for k in range(j + 1, X.shape[1]):
                cols.append(X[:, j] * X[:, k])
                labels.append(f"{names[j]}*{names[k]}")
    M = np.column_stack(cols) if cols else np.empty((X.shape[0], 0))
    spec, _ = _drop_redundant(M, [(j, 1) for j in range(M.shape[1])])
    keep = [j for j, _p in spec]
    return M[:, keep], [labels[j] for j in keep]


def _standardize(Z: np.ndarray) -> np.ndarray:
    """``[1, (Z - mean) / sd]``: an invertible reparametrisation of the
    span of ``[1, Z]``, used only for conditioning."""
    sd = Z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return np.column_stack([np.ones(len(Z)), (Z - Z.mean(axis=0)) / sd])


def _probs(Xt: np.ndarray, beta: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial-logit probabilities and linear predictors (level 0 is
    the base)."""
    p = Xt.shape[1]
    eta = np.column_stack([np.zeros(len(Xt)), Xt @ beta.reshape(K - 1, p).T])
    eta_c = eta - eta.max(axis=1, keepdims=True)
    P = np.exp(eta_c)
    P /= P.sum(axis=1, keepdims=True)
    return P, eta


def mlogit_fit(
    Xt: np.ndarray, G: np.ndarray, K: int, *, tol: float = 1e-11, max_iter: int = 200
) -> np.ndarray:
    """Maximum-likelihood multinomial logit by damped Newton.

    ``Xt`` holds the intercept; ``G`` the 0..K-1 group codes. Returns the
    stacked coefficients ``beta`` (``(K-1) * p``). Separation (no finite
    maximum) is reported as :class:`NumericalInstability`.
    """
    n, p = Xt.shape
    Y = (G[:, None] == np.arange(K)[None, :]).astype(float)
    beta = np.zeros((K - 1) * p)

    def nll(b: np.ndarray) -> float:
        _, eta = _probs(Xt, b, K)
        m = eta.max(axis=1)
        return float(
            -(Y * eta).sum() + (m + np.log(np.exp(eta - m[:, None]).sum(1))).sum()
        )

    f = nll(beta)
    for _ in range(max_iter):
        P, _ = _probs(Xt, beta, K)
        grad = ((Y - P)[:, 1:].T @ Xt).ravel()  # score, block k = X'(y_k - p_k)
        if np.max(np.abs(grad)) <= tol * n:
            return beta
        H = np.zeros(((K - 1) * p, (K - 1) * p))
        for a in range(1, K):
            for b in range(a, K):
                wgt = P[:, a] * ((a == b) - P[:, b])
                blk = (Xt * wgt[:, None]).T @ Xt
                H[(a - 1) * p : a * p, (b - 1) * p : b * p] = blk
                H[(b - 1) * p : b * p, (a - 1) * p : a * p] = blk.T
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        for _ls in range(60):
            cand = beta + t * step
            fc = nll(cand)
            if fc <= f + 1e-12 * abs(f):
                break
            t *= 0.5
        beta, f = cand, fc
        if np.max(np.abs(beta)) > 50:
            break
    P, _ = _probs(Xt, beta, K)
    if np.max(np.abs(beta)) > 50 or P.min() < 1e-12:
        raise NumericalInstability(
            "weightit: the multinomial-logit propensity model has no finite "
            "maximum (a covariate separates a treatment group).",
            recovery_hint="Drop or coarsen the separating covariate, or use "
            "method='ebal', which does not fit a propensity model.",
            diagnostics={"max_abs_coef": float(np.max(np.abs(beta)))},
        )
    _sp_warn(
        ConvergenceWarning,
        "weightit: the multinomial-logit fit stopped before its score "
        "reached tolerance.",
        recovery_hint="Rescale the covariates or drop near-collinear ones.",
        stacklevel=4,
    )
    return beta


def group_weights(
    P: np.ndarray, G: np.ndarray, estimand: str, focal: Optional[int]
) -> np.ndarray:
    """IPW weights from generalized propensity scores."""
    pa = P[np.arange(len(G)), G]
    w: np.ndarray
    if estimand == "ATE":
        w = 1.0 / pa
    elif estimand in ("ATT", "ATC"):  # ATC: binary only, focal is the control
        w = P[:, focal] / pa
    elif estimand == "ATO":
        w = (1.0 / pa) / (1.0 / P).sum(axis=1)
    elif estimand == "ATM":
        w = P.min(axis=1) / pa
    else:
        raise ValueError(estimand)  # pragma: no cover - validated upstream
    return w


def cbps_fit(
    Xt: np.ndarray,
    G: np.ndarray,
    K: int,
    estimand: str,
    focal: Optional[int],
    beta0: np.ndarray,
    *,
    tol: float = 1e-11,
    max_iter: int = 200,
) -> np.ndarray:
    """Just-identified CBPS for a categorical treatment.

    Solves ``sum_i (1[A_i = k] - 1[A_i = r]) w_i (1, X_i) = 0`` for every
    ``k != r`` (``r`` = base level for the ATE, the focal level for the
    ATT), with ``w`` the multinomial-logit weights of :func:`group_weights`.
    That is ``(K - 1) * p`` equations in as many coefficients; Newton on
    the analytic Jacobian with a merit-function line search.
    """
    n, p = Xt.shape
    r = 0 if estimand == "ATE" else int(focal)  # type: ignore[arg-type]
    others = [k for k in range(K) if k != r]
    sign = np.zeros((n, K - 1))
    for c, k in enumerate(others):
        sign[:, c] = (G == k).astype(float) - (G == r).astype(float)

    def equations(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P, _ = _probs(Xt, beta, K)
        w = group_weights(P, G, estimand, focal)
        g = (sign * w[:, None]).T @ Xt  # (K-1, p)
        # d log w_i / d eta_m, m = 1..K-1
        onehot = (G[:, None] == np.arange(K)[None, :]).astype(float)
        if estimand == "ATE":
            dlog = P[:, 1:] - onehot[:, 1:]
        else:
            fvec = np.zeros(K)
            fvec[r] = 1.0
            dlog = fvec[None, 1:] - onehot[:, 1:]
        J = np.zeros(((K - 1) * p, (K - 1) * p))
        for c in range(K - 1):
            for m in range(K - 1):
                coef = sign[:, c] * w * dlog[:, m]
                J[c * p : (c + 1) * p, m * p : (m + 1) * p] = (
                    Xt * coef[:, None]
                ).T @ Xt
        return g.ravel() / n, J / n

    beta = beta0.copy()
    g, J = equations(beta)
    for _ in range(max_iter):
        if np.max(np.abs(g)) <= tol:
            return beta
        try:
            step = -np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(J, g, rcond=None)[0]
        merit = float(g @ g)
        t = 1.0
        for _ls in range(60):
            cand = beta + t * step
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                gc, Jc = equations(cand)
            if np.all(np.isfinite(gc)) and float(gc @ gc) < merit:
                break
            t *= 0.5
        else:
            break
        beta, g, J = cand, gc, Jc
    if np.max(np.abs(g)) > 1e-7:
        raise NumericalInstability(
            "weightit(method='cbps'): the balance equations have no solution "
            "the solver could reach (largest standardized imbalance "
            f"{float(np.max(np.abs(g))):.2e}).",
            recovery_hint="Balance fewer terms (moments=1, interactions="
            "False), drop a covariate with little overlap, or use "
            "method='glm'.",
            diagnostics={"max_abs_moment": float(np.max(np.abs(g)))},
        )
    return beta


def ebal_groups(
    M: np.ndarray, G: np.ndarray, K: int, estimand: str, focal: Optional[int]
) -> np.ndarray:
    """Entropy-balancing weights for a categorical treatment.

    ATE: each group is weighted to the full-sample means of ``M`` and its
    weights sum to ``n``. ATT: each non-focal group is weighted to the
    focal-group means and sums to ``n_focal``; focal units keep weight 1.
    """
    n = len(G)
    w = np.ones(n)
    if estimand == "ATE":
        target, total, groups = M.mean(axis=0), float(n), list(range(K))
    else:
        sel = G == focal
        target, total = M[sel].mean(axis=0), float(sel.sum())
        groups = [k for k in range(K) if k != focal]
    for k in groups:
        sel = G == k
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wk, fallback = _solve_ebalance(M[sel], target, int(sel.sum()))
        if fallback:
            raise NumericalInstability(
                "weightit(method='ebal'): no positive weights move treatment "
                f"group {k} to the target means; the target lies outside the "
                "convex hull of that group's covariates.",
                recovery_hint="Balance fewer terms, drop a covariate with "
                "little overlap, or use method='glm'.",
                diagnostics={"group_code": int(k)},
            )
        w[sel] = wk * total / wk.sum()
    return w


def continuous_glm(
    a: np.ndarray, Xt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normal-GPS weights ``phi(z_A) / phi(r / sigma)``; returns the
    weights, the fitted means and ``sigma`` (ML scale)."""
    beta = np.linalg.lstsq(Xt, a, rcond=None)[0]
    mu = Xt @ beta
    r = a - mu
    sigma = float(np.sqrt(np.mean(r**2)))
    sd_a = float(a.std())
    if sigma <= 1e-12 * max(sd_a, 1.0):
        raise NumericalInstability(
            "weightit: the covariates predict the continuous treatment "
            "exactly, so its conditional density is degenerate.",
            recovery_hint="Remove covariates that determine the treatment.",
        )
    z_a = (a - a.mean()) / sd_a
    log_w = -0.5 * z_a**2 + 0.5 * (r / sigma) ** 2
    return np.exp(log_w), mu, sigma


def continuous_ebal(
    a: np.ndarray, X: np.ndarray, M: np.ndarray, d_moments: int
) -> np.ndarray:
    """Entropy balancing for a continuous treatment.

    Constraints (weights normalized to sum ``n``): the weighted means of
    ``a`` and of every balance term in ``M`` equal their sample means; the
    weighted covariance of ``a`` with every column of ``M`` is zero; and
    for ``d_moments = k > 1`` the weighted central moments 2..k of ``a``
    and of each non-binary raw covariate in ``X`` equal the sample ones.
    """
    n = len(a)
    a_c = a - a.mean()
    cols: List[np.ndarray] = [a]
    for power in range(2, int(d_moments) + 1):
        cols.append(a_c**power)
    for j in range(M.shape[1]):
        cols.append(M[:, j])
    for j in range(X.shape[1]):
        x = X[:, j]
        if not np.isin(x, (0.0, 1.0)).all():
            for power in range(2, int(d_moments) + 1):
                cols.append((x - x.mean()) ** power)
    first_cov = len(cols)
    for j in range(M.shape[1]):
        x = M[:, j]
        cols.append(a_c * (x - x.mean()))
    C = np.column_stack(cols)
    spec, _ = _drop_redundant(C, [(j, 1) for j in range(C.shape[1])])
    kept = [j for j, _p in spec]
    C = C[:, kept]
    # moment targets are the sample values; covariance targets are zero
    target = np.array([0.0 if j >= first_cov else float(cols[j].mean()) for j in kept])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, fallback = _solve_ebalance(C, target, n)
    if fallback:
        raise NumericalInstability(
            "weightit(method='ebal'): no positive weights remove the "
            "treatment-covariate covariances while keeping the moments.",
            recovery_hint="Lower moments / d_moments, or drop a covariate "
            "that is nearly a function of the treatment.",
        )
    w_out: np.ndarray = w * n / w.sum()
    return w_out


def describe_weights(w: np.ndarray, G: Optional[np.ndarray], labels: Sequence) -> Dict:
    """Weight ranges and effective sample sizes (overall and by group)."""

    def _ess(v: np.ndarray) -> float:
        return float(v.sum() ** 2 / (v**2).sum())

    out: Dict = {
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "coef_of_variation": float(w.std() / w.mean()),
        "ess": _ess(w),
    }
    if G is not None:
        out["ess_by_group"] = {
            str(lab): _ess(w[G == k]) for k, lab in enumerate(labels)
        }
        out["n_by_group"] = {
            str(lab): int((G == k).sum()) for k, lab in enumerate(labels)
        }
    return out
