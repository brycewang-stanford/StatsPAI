"""Natural effects when the outcome model is not linear.

Extends :mod:`statspai.mediation._po_means` to a probit, logit or Poisson
outcome and to a treatment contrast other than 1 against 0. The mean of the
potential outcome ``Y(d, M(d'))`` is no longer a linear function of the
mediator's mean, so the mediator has to be integrated out:

* binary mediator (logit or probit model): a two-point mixture,
  ``p G(eta + c) + (1 - p) G(eta)`` with ``p = P(M = 1 | d', x)``;
* continuous mediator (linear model): the mediator given ``(d', x)`` is taken
  to be normal with the regression's mean and its maximum-likelihood residual
  variance ``s2``, and ``G`` is averaged over that distribution. For a probit
  outcome the average is ``Phi((eta + c m) / sqrt(1 + c^2 s2))``, for a
  Poisson outcome ``exp(eta + c m + c^2 s2 / 2)``; for a logit outcome there
  is no closed form and Gauss-Hermite quadrature is used.

Here ``eta`` is the part of the outcome index that does not involve the
mediator and ``c`` its coefficient (``b_m + b_dm d``).

Every parameter (outcome coefficients, mediator coefficients, ``s2`` and the
four means) solves one system of estimating equations, and the covariance is
the sandwich of that system, as in the linear case. This is the estimator of
Stata 18's ``mediate`` for the model pairs Stata allows; Stata refuses a
logit outcome with a linear mediator, which is the pair that needs the
quadrature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility
from ._po_means import (
    _PO_LABELS,
    _expand_categorical_covariates,
    _fit_binary,
    _mean_and_slope,
)

__all__ = ["potential_outcome_mediation_nonlinear"]

_OUTCOME_MODELS = ("linear", "logit", "probit", "poisson")
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(40)


def _outcome_mean(eta: np.ndarray, family: str) -> np.ndarray:
    if family == "linear":
        return eta
    if family == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    if family == "probit":
        return np.asarray(stats.norm.cdf(eta))
    return np.exp(eta)


def _outcome_score_weight(Y: np.ndarray, eta: np.ndarray, family: str) -> np.ndarray:
    """``s`` such that the score of the outcome model is ``W' s``."""
    if family in ("linear", "logit", "poisson"):
        return Y - _outcome_mean(eta, family)
    q = 2.0 * Y - 1.0
    return np.asarray(q * stats.norm.pdf(q * eta) / stats.norm.cdf(q * eta))


def _fit_outcome(W: np.ndarray, Y: np.ndarray, family: str) -> np.ndarray:
    if family == "linear":
        return np.asarray(np.linalg.solve(W.T @ W, W.T @ Y))
    if family in ("logit", "probit"):
        return _fit_binary(W, Y, family)
    beta = np.zeros(W.shape[1])
    beta[0] = np.log(max(Y.mean(), 1e-8))
    for _ in range(100):
        mu = np.exp(W @ beta)
        try:
            step = np.linalg.solve((W * mu[:, None]).T @ W, W.T @ (Y - mu))
        except np.linalg.LinAlgError as exc:
            raise ConvergenceFailure(
                "mediate: the Poisson outcome model is not identified "
                "(singular information matrix).",
            ) from exc
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
    raise ConvergenceFailure(
        "mediate: the Poisson outcome model did not converge.",
        recovery_hint="Check the outcome for extreme counts.",
    )


def _integrated_mean(
    eta: np.ndarray, c: float, m_mean: np.ndarray, s2: float, family: str
) -> np.ndarray:
    """``E[G(eta + c M)]`` for ``M ~ N(m_mean, s2)``."""
    if family == "linear":
        return eta + c * m_mean
    if family == "probit":
        return np.asarray(
            stats.norm.cdf((eta + c * m_mean) / np.sqrt(1.0 + c * c * s2))
        )
    if family == "poisson":
        return np.exp(eta + c * m_mean + 0.5 * c * c * s2)
    # logit: Gauss-Hermite over the normal mediator
    nodes = m_mean[:, None] + np.sqrt(2.0 * max(s2, 0.0)) * _GH_NODES[None, :]
    vals = 1.0 / (1.0 + np.exp(-(eta[:, None] + c * nodes)))
    return np.asarray(vals @ _GH_WEIGHTS / np.sqrt(np.pi))


def potential_outcome_mediation_nonlinear(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    *,
    interaction: bool = True,
    mediator_model: str = "linear",
    outcome_model: str = "logit",
    treat_values: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Potential-outcome means and natural effects, any supported model pair.

    Same return structure as
    :func:`statspai.mediation._po_means.potential_outcome_mediation`, plus
    ``outcome_model``, ``treat_values`` and, for a linear mediator,
    ``mediator_variance``.
    """
    family = str(outcome_model).lower()
    link = str(mediator_model).lower()
    if family not in _OUTCOME_MODELS:
        raise MethodIncompatibility(
            f"mediate: outcome_model must be one of {_OUTCOME_MODELS}, "
            f"got {outcome_model!r}.",
        )
    if link not in ("linear", "logit", "probit"):
        raise MethodIncompatibility(
            "mediate: mediator_model must be 'linear', 'logit' or 'probit', "
            f"got {mediator_model!r}.",
        )
    data, covs = _expand_categorical_covariates(data, list(covariates or []))
    cols = [y, treat, mediator, *covs]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"mediate: column(s) not found in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    clean = data[cols].dropna()
    n = len(clean)
    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    M = clean[mediator].to_numpy(dtype=float)
    X = clean[covs].to_numpy(dtype=float) if covs else np.empty((n, 0))

    if treat_values is None:
        if not np.isin(D, (0.0, 1.0)).all():
            raise MethodIncompatibility(
                f"mediate: {treat!r} is not coded 0/1, so say which two "
                "values of it to contrast.",
                recovery_hint="Pass treat_values=(control, treated), e.g. "
                "treat_values=(1, 3) for a continuous treatment.",
                diagnostics={"values": np.unique(D)[:10].tolist()},
            )
        d0, d1 = 0.0, 1.0
    else:
        try:
            d0, d1 = (float(v) for v in treat_values)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "mediate: treat_values must be a pair (control, treated)."
            ) from exc
        if d0 == d1:
            raise MethodIncompatibility("mediate: the two treat_values are the same.")
    if family in ("logit", "probit") and not np.isin(Y, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"mediate: outcome_model={family!r} needs a 0/1 outcome.",
            diagnostics={"values": np.unique(Y)[:10].tolist()},
        )
    if family == "poisson" and (Y < 0).any():
        raise MethodIncompatibility(
            "mediate: outcome_model='poisson' needs a non-negative outcome."
        )
    if link != "linear" and not np.isin(M, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"mediate: mediator_model={link!r} needs a 0/1 mediator.",
            recovery_hint="Use mediator_model='linear' for a continuous mediator.",
        )

    ones = np.ones(n)
    W = np.column_stack([ones, D, M] + ([D * M] if interaction else []) + [X])
    Z = np.column_stack([ones, D, X])
    kb, ka = W.shape[1], Z.shape[1]
    if n <= kb + 4:
        raise DataInsufficient(
            "mediate: too few observations for the outcome equation.",
            diagnostics={"n_obs": n, "n_parameters": kb},
        )
    if np.linalg.matrix_rank(W) < kb or np.linalg.matrix_rank(Z) < ka:
        raise MethodIncompatibility(
            "mediate: the outcome or mediator design matrix is rank deficient.",
            recovery_hint="Drop collinear covariates.",
        )

    beta = _fit_outcome(W, Y, family)
    if link == "linear":
        a = np.asarray(np.linalg.solve(Z.T @ Z, Z.T @ M))
        s2 = float(np.mean((M - Z @ a) ** 2))
    else:
        a = _fit_binary(Z, M, link)
        s2 = 0.0
    # The residual variance of a linear mediator enters the means as a
    # known constant, as in Stata's mediate: it has no estimating equation
    # of its own, so its sampling variation is not in the standard errors.
    has_s2 = False
    x_start = 4 if interaction else 3
    arms = ((d0, d0), (d1, d0), (d0, d1), (d1, d1))  # (d, d') per label

    def po_matrix(b: np.ndarray, al: np.ndarray, var: float) -> np.ndarray:
        """n x 4 matrix of g_dd'(x_i)."""
        x_part = X @ b[x_start:] if X.shape[1] else np.zeros(n)
        b_dm = b[3] if interaction else 0.0
        out = np.empty((n, 4))
        for j, (d, d_prime) in enumerate(arms):
            eta = b[0] + b[1] * d + x_part
            c = b[2] + b_dm * d
            idx = al[0] + al[1] * d_prime + (X @ al[2:] if X.shape[1] else 0.0)
            idx = np.broadcast_to(idx, (n,)).astype(float)
            if link == "linear":
                out[:, j] = _integrated_mean(eta, c, idx, var, family)
            else:
                p, _ = _mean_and_slope(idx, link)
                out[:, j] = p * _outcome_mean(eta + c, family) + (
                    1.0 - p
                ) * _outcome_mean(eta, family)
        return out

    mu = po_matrix(beta, a, s2).mean(axis=0)
    k = kb + ka + int(has_s2) + 4
    theta = np.concatenate([beta, a, [s2] if has_s2 else [], mu])

    def moments(th: np.ndarray) -> np.ndarray:
        """n x k matrix of the stacked estimating functions."""
        b = th[:kb]
        al = th[kb : kb + ka]
        var = th[kb + ka] if has_s2 else s2
        m_ = th[-4:]
        psi_b = W * _outcome_score_weight(Y, W @ b, family)[:, None]
        idx = Z @ al
        if link == "linear":
            r = M - idx
            psi_a = Z * r[:, None]
            psi_s = (r * r - var)[:, None] if has_s2 else np.empty((n, 0))
        else:
            psi_a = Z * _outcome_score_weight(M, idx, link)[:, None]
            psi_s = np.empty((n, 0))
        psi_mu = po_matrix(b, al, var) - m_[None, :]
        return np.column_stack([psi_b, psi_a, psi_s, psi_mu])

    psi = moments(theta)
    # A = - d mean(psi) / d theta, by central differences. The moment
    # functions are smooth in theta, so a step of 1e-6 leaves an error near
    # 1e-10 in each entry.
    A = np.empty((k, k))
    for j in range(k):
        h = 1e-6 * max(1.0, abs(theta[j]))
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        A[:, j] = -(moments(up).mean(axis=0) - moments(dn).mean(axis=0)) / (2 * h)
    B = psi.T @ psi / n
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceFailure(
            "mediate: the estimating equations are singular at the solution."
        ) from exc
    V = A_inv @ B @ A_inv.T / n
    V_mu = V[-4:, -4:]

    crit = stats.norm.ppf(1 - alpha / 2)

    def _table(names: List[str], est: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, est / se, np.nan)
        return pd.DataFrame(
            {
                "effect": names,
                "estimate": est,
                "se": se,
                "z": z,
                "pvalue": 2 * stats.norm.sf(np.abs(z)),
                "ci_lower": est - crit * se,
                "ci_upper": est + crit * se,
            }
        )

    contrasts = {
        "NIE": np.array([0.0, -1.0, 0.0, 1.0]),
        "NDE": np.array([-1.0, 1.0, 0.0, 0.0]),
        "PNIE": np.array([-1.0, 0.0, 1.0, 0.0]),
        "TNDE": np.array([0.0, 0.0, -1.0, 1.0]),
        "TE": np.array([-1.0, 0.0, 0.0, 1.0]),
    }
    # With a non-linear outcome the two decompositions differ even without
    # the treatment-mediator interaction, so all five are always reported.
    nonlinear = family != "linear"
    order = (
        ["NIE", "NDE", "PNIE", "TNDE", "TE"]
        if (interaction or nonlinear)
        else ["NIE", "NDE", "TE"]
    )
    C = np.vstack([contrasts[name] for name in order])
    effects = _table(order, C @ mu, C @ V_mu @ C.T)

    nie, te = float(contrasts["NIE"] @ mu), float(contrasts["TE"] @ mu)
    if te != 0:
        grad = contrasts["NIE"] / te - nie * contrasts["TE"] / te**2
        pm, pm_var = nie / te, float(grad @ V_mu @ grad)
    else:  # pragma: no cover - a total effect of exactly zero
        pm, pm_var = float("nan"), float("nan")
    prop = _table(["Prop. Mediated"], np.array([pm]), np.array([[pm_var]]))

    out_names = (
        ["_cons", treat, mediator]
        + ([f"{treat}:{mediator}"] if interaction else [])
        + covs
    )
    med_names = ["_cons", treat] + covs
    return {
        "effects": effects,
        "po_means": _table(list(_PO_LABELS), mu, V_mu),
        "prop_mediated": prop,
        "outcome_coef": pd.Series(beta, index=out_names),
        "outcome_se": pd.Series(np.sqrt(np.diag(V[:kb, :kb])), index=out_names),
        "mediator_coef": pd.Series(a, index=med_names),
        "mediator_se": pd.Series(
            np.sqrt(np.diag(V[kb : kb + ka, kb : kb + ka])), index=med_names
        ),
        "vcov_po": pd.DataFrame(V_mu, index=_PO_LABELS, columns=_PO_LABELS),
        "n_obs": n,
        "interaction": bool(interaction),
        "mediator_model": link,
        "outcome_model": family,
        "treat_values": (d0, d1),
        "mediator_variance": s2 if link == "linear" else None,
    }
