"""Natural direct and indirect effects from potential-outcome means, with
standard errors from the stacked estimating equations.

The estimator behind ``sp.mediate(inference='robust')`` and Stata 18's
``mediate``. For a binary treatment ``D``, mediator ``M`` and outcome ``Y``::

    outcome   E[Y | D, M, X] = b0 + bd D + bm M + bdm D M + X'g
    mediator  E[M | D, X]    = h(a0 + ad D + X'a)

with ``h`` the identity, the logistic or the normal distribution function.
Writing ``m_d'(x) = h(a0 + ad d' + x'a)``, the mean of the potential outcome
``Y(d, M(d'))`` is the sample average of

    g_dd'(x) = b0 + bd d + (bm + bdm d) m_d'(x) + x'g

and the effects are contrasts of the four means::

    NIE  = Y1M1 - Y1M0      natural indirect effect
    NDE  = Y1M0 - Y0M0      natural direct effect
    PNIE = Y0M1 - Y0M0      pure natural indirect effect
    TNDE = Y1M1 - Y0M1      total natural direct effect
    TE   = Y1M1 - Y0M0      total effect  (= NIE + NDE = PNIE + TNDE)

Without the treatment-mediator interaction ``NIE = PNIE`` and
``NDE = TNDE``.

All parameters -- outcome coefficients, mediator coefficients and the four
means -- solve one system of estimating equations, so their joint covariance
is the sandwich ``A^-1 B A^-T / n`` of that system. That is what makes the
standard errors valid for the *sample-averaged* means (the covariate
distribution is estimated too) and robust to heteroskedasticity; the
delta-method standard errors of the product of coefficients are neither.

References
----------
imai2010general, vanderweele2014unification
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

__all__ = ["potential_outcome_mediation"]

_MEDIATOR_MODELS = ("linear", "logit", "probit")
_PO_LABELS = ("Y0M0", "Y1M0", "Y0M1", "Y1M1")
# (d, d') behind each label: outcome under d, mediator under d'.
_PO_ARMS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _fit_binary(
    Z: np.ndarray, m: np.ndarray, link: str, max_iter: int = 100, tol: float = 1e-12
) -> np.ndarray:
    """Maximum likelihood for a logit or probit mediator, by Newton steps."""
    a = np.zeros(Z.shape[1])
    q = 2.0 * m - 1.0
    for _ in range(max_iter):
        eta = Z @ a
        if link == "logit":
            p = 1.0 / (1.0 + np.exp(-eta))
            score = Z.T @ (m - p)
            hess = (Z * (p * (1.0 - p))[:, None]).T @ Z
        else:
            lam = q * stats.norm.pdf(q * eta) / stats.norm.cdf(q * eta)
            score = Z.T @ lam
            hess = (Z * (lam * (lam + eta))[:, None]).T @ Z
        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceFailure(
                f"mediate: the {link} mediator model is not identified "
                "(singular information matrix).",
                recovery_hint=(
                    "Check for perfect prediction or collinear covariates in "
                    "the mediator equation."
                ),
            ) from exc
        a = a + step
        if np.max(np.abs(step)) < tol:
            return a
    raise ConvergenceFailure(
        f"mediate: the {link} mediator model did not converge in "
        f"{max_iter} Newton steps.",
        recovery_hint="Check for perfect prediction in the mediator equation.",
        diagnostics={"link": link, "max_iter": max_iter},
    )


def _mean_and_slope(eta: np.ndarray, link: str) -> Tuple[np.ndarray, np.ndarray]:
    """``h(eta)`` and ``h'(eta)`` for the mediator's mean function."""
    if link == "linear":
        return eta, np.ones_like(eta)
    if link == "logit":
        p = 1.0 / (1.0 + np.exp(-eta))
        return p, p * (1.0 - p)
    return stats.norm.cdf(eta), stats.norm.pdf(eta)


_CATEGORICAL = re.compile(r"^C\(\s*([^\W\d]\w*)\s*\)$")


def _expand_categorical_covariates(
    data: pd.DataFrame, covariates: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Replace ``C(g)`` in the covariate list by indicator columns.

    One indicator per level except the lowest, named ``g[level]`` -- the
    coding ``i.g`` has in Stata and ``C(g)`` in a formula. Rows where ``g``
    is missing get missing indicators and drop out with the rest.
    """
    if not any(_CATEGORICAL.match(c) for c in covariates):
        return data, covariates
    frame = data.copy()
    out: List[str] = []
    for cov in covariates:
        m = _CATEGORICAL.match(cov)
        if not m:
            out.append(cov)
            continue
        name = m.group(1)
        if name not in frame.columns:
            raise MethodIncompatibility(
                f"mediate: column {name!r} (from {cov!r}) not found in data.",
                diagnostics={"missing_columns": [name]},
            )
        levels = sorted(frame[name].dropna().unique().tolist())
        for level in levels[1:]:
            label = int(level) if float(level).is_integer() else level
            col = f"{name}[{label}]"
            frame[col] = np.where(
                frame[name].isna(), np.nan, (frame[name] == level).astype(float)
            )
            out.append(col)
    return frame, out


def potential_outcome_mediation(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    *,
    interaction: bool = True,
    mediator_model: str = "linear",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Potential-outcome means, natural effects and their joint covariance.

    Returns a dict with ``effects`` and ``po_means`` (DataFrames with
    ``estimate``, ``se``, ``z``, ``pvalue``, ``ci_lower``, ``ci_upper``),
    ``prop_mediated`` (``NIE / TE`` with its delta-method standard error),
    ``outcome_coef`` / ``mediator_coef`` (Series), ``vcov_po`` (4x4),
    ``n_obs``.
    """
    link = str(mediator_model).lower()
    if link not in _MEDIATOR_MODELS:
        raise MethodIncompatibility(
            f"mediate: mediator_model must be one of {_MEDIATOR_MODELS}, "
            f"got {mediator_model!r}.",
            diagnostics={"mediator_model": mediator_model},
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
    if not np.isin(D, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            "mediate(inference='robust') contrasts potential outcomes under "
            f"treatment 1 and 0, so {treat!r} must be coded 0/1.",
            recovery_hint=(
                "Recode the treatment, or use inference='delta' / "
                "'bootstrap' for a continuous treatment."
            ),
            diagnostics={"values": np.unique(D)[:10].tolist()},
        )
    if link != "linear" and not np.isin(M, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"mediate: mediator_model={link!r} needs a 0/1 mediator.",
            recovery_hint="Use mediator_model='linear' for a continuous mediator.",
            diagnostics={"values": np.unique(M)[:10].tolist()},
        )

    ones = np.ones(n)
    out_cols = [ones, D, M] + ([D * M] if interaction else [])
    W = np.column_stack(out_cols + [X])
    Z = np.column_stack([ones, D, X])
    kb, ka = W.shape[1], Z.shape[1]
    if n <= kb + 4:
        raise DataInsufficient(
            "mediate: too few observations for the outcome equation.",
            diagnostics={"n_obs": n, "n_parameters": kb},
        )
    if np.linalg.matrix_rank(W) < kb or np.linalg.matrix_rank(Z) < ka:
        raise MethodIncompatibility(
            "mediate: the outcome or mediator design matrix is rank " "deficient.",
            recovery_hint=(
                "Drop collinear covariates (for a set of category dummies, "
                "leave one category out)."
            ),
        )

    beta = np.linalg.solve(W.T @ W, W.T @ Y)
    if link == "linear":
        a = np.linalg.solve(Z.T @ Z, Z.T @ M)
    else:
        a = _fit_binary(Z, M, link)

    b_m = beta[2]
    b_dm = beta[3] if interaction else 0.0
    x_start = 4 if interaction else 3

    # Mediator mean under each treatment level, and its slope in the index.
    m_hat: Dict[int, np.ndarray] = {}
    m_slope: Dict[int, np.ndarray] = {}
    Z_cf: Dict[int, np.ndarray] = {}
    for d_prime in (0, 1):
        Zd = np.column_stack([ones, np.full(n, float(d_prime)), X])
        Z_cf[d_prime] = Zd
        m_hat[d_prime], m_slope[d_prime] = _mean_and_slope(Zd @ a, link)

    g = np.empty((n, 4))
    dg_dbeta = np.empty((4, kb))
    dg_dalpha = np.empty((4, ka))
    x_part = X @ beta[x_start:] if X.shape[1] else np.zeros(n)
    x_bar = X.mean(axis=0) if X.shape[1] else np.empty(0)
    for j, (d, d_prime) in enumerate(_PO_ARMS):
        mh = m_hat[d_prime]
        g[:, j] = beta[0] + beta[1] * d + (b_m + b_dm * d) * mh + x_part
        grad_b = [1.0, float(d), float(mh.mean())]
        if interaction:
            grad_b.append(float(d * mh.mean()))
        dg_dbeta[j] = np.concatenate([grad_b, x_bar])
        dg_dalpha[j] = (b_m + b_dm * d) * (
            Z_cf[d_prime] * m_slope[d_prime][:, None]
        ).mean(axis=0)
    mu = g.mean(axis=0)

    # ---- stacked estimating equations ------------------------------------
    resid_y = Y - W @ beta
    psi_b = W * resid_y[:, None]
    eta = Z @ a
    if link == "linear":
        psi_a = Z * (M - eta)[:, None]
        A_aa = Z.T @ Z / n
    elif link == "logit":
        p = 1.0 / (1.0 + np.exp(-eta))
        psi_a = Z * (M - p)[:, None]
        A_aa = (Z * (p * (1.0 - p))[:, None]).T @ Z / n
    else:
        q = 2.0 * M - 1.0
        lam = q * stats.norm.pdf(q * eta) / stats.norm.cdf(q * eta)
        psi_a = Z * lam[:, None]
        A_aa = (Z * (lam * (lam + eta))[:, None]).T @ Z / n
    psi_mu = g - mu[None, :]

    k = kb + ka + 4
    A = np.zeros((k, k))
    A[:kb, :kb] = W.T @ W / n
    A[kb : kb + ka, kb : kb + ka] = A_aa
    A[kb + ka :, :kb] = -dg_dbeta
    A[kb + ka :, kb : kb + ka] = -dg_dalpha
    A[kb + ka :, kb + ka :] = np.eye(4)
    psi = np.column_stack([psi_b, psi_a, psi_mu])
    B = psi.T @ psi / n
    A_inv = np.linalg.inv(A)
    V = A_inv @ B @ A_inv.T / n
    V_mu = V[kb + ka :, kb + ka :]

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

    # Contrasts on (Y0M0, Y1M0, Y0M1, Y1M1).
    contrasts = {
        "NIE": np.array([0.0, -1.0, 0.0, 1.0]),
        "NDE": np.array([-1.0, 1.0, 0.0, 0.0]),
        "PNIE": np.array([-1.0, 0.0, 1.0, 0.0]),
        "TNDE": np.array([0.0, 0.0, -1.0, 1.0]),
        "TE": np.array([-1.0, 0.0, 0.0, 1.0]),
    }
    order = (
        ["NIE", "NDE", "PNIE", "TNDE", "TE"] if interaction else ["NIE", "NDE", "TE"]
    )
    C = np.vstack([contrasts[name] for name in order])
    eff = C @ mu
    V_eff = C @ V_mu @ C.T
    effects = _table(order, eff, V_eff)

    # Proportion mediated NIE / TE, delta method on the two contrasts.
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
    }
