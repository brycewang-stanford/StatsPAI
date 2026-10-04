"""
Poisson regression with an endogenous binary treatment.

    y | x, t, eps  ~  Poisson(exp(x'b + delta * t + eps))
    t = 1[w'g + u > 0]
    (eps, u) ~ N(0, [[sigma^2, rho * sigma], [rho * sigma, 1]])

The outcome error ``eps`` is what makes the count overdispersed, and its
correlation with ``u`` is what makes the treatment endogenous. The
likelihood integrates ``eps`` out by Gauss-Hermite quadrature, as Stata's
``etpoisson`` does.

References
----------
[@terza1998estimating]
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._vcov import ml_vcov
from ..core.results import CausalResult, EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

__all__ = ["etpoisson"]


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@accepts_aliases(robust="vce", covariates="x", treatment="treat")
def etpoisson(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    treat: Optional[str] = None,
    z: Union[str, Sequence[str], None] = None,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    intpoints: int = 24,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Poisson regression with an endogenous binary treatment, by ML.

    Equivalent to Stata's ``etpoisson y x, treat(treat = z)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Count outcome (non-negative integers).
    x : str or list of str, optional
        Exogenous regressors of the outcome equation. A constant is added.
    treat : str
        Binary (0/1) endogenous treatment. ``treatment=`` is an alias.
    z : str or list of str
        Regressors of the treatment equation. They may overlap with ``x``;
        a variable in ``z`` and not in ``x`` is an exclusion restriction.
        Without one the model is identified by the normality of the
        errors alone.
    vce : {None, 'robust', 'cluster'}, optional
        Observed information (default), sandwich, or cluster sandwich.
    cluster : str, optional
    intpoints : int, default 24
        Gauss-Hermite points for the integral over the outcome error
        (Stata's default). Estimates move in the fifth digit between 24
        and 64 points on moderately overdispersed data.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params``: the outcome equation (``y:var``, the treatment
        coefficient under its own name), the treatment equation
        (``treatment:var``), ``/athrho`` and ``/lnsigma``.
        ``model_info`` carries ``rho``, ``sigma``, the Wald test of
        ``rho = 0`` (``independence_chi2``, ``independence_pvalue``) and
        the average treatment effect on the count scale (``ate``,
        ``ate_se``, by the delta method with the covariates held fixed),
        with the two potential-outcome means.

    Notes
    -----
    The coefficient on the treatment is the effect on the log of the
    conditional mean; ``exp`` of it is an incidence-rate ratio. ``ate`` is
    the difference in expected counts,
    ``mean(exp(x'b + sigma^2 / 2) * (exp(delta) - 1))``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.etpoisson(df, y="visits", x=["age", "income"],
    ...                    treat="insured", z=["age", "employer_offer"])
    >>> res.model_info["ate"]  # doctest: +SKIP

    References
    ----------
    [@terza1998estimating]
    """
    from ..core._vcov_spec import parse_se_request

    req = parse_se_request(
        vce, cluster, function="etpoisson", supported=("nonrobust", "robust", "cluster")
    )
    se_kind, cl = req.kind, req.cluster
    treatment = treat
    xs, zs = _as_list(x), _as_list(z)
    if treatment is None or not zs:
        raise MethodIncompatibility(
            "etpoisson: treat= and z= (the treatment equation) are required."
        )
    if treatment in xs:
        raise MethodIncompatibility(
            f"etpoisson: {treatment!r} is the treatment; leave it out of x=."
        )
    if int(intpoints) < 2:
        raise MethodIncompatibility("etpoisson: intpoints must be at least 2.")
    extra = [cl] if isinstance(cl, str) else []
    cols = [y, treatment] + xs + zs + extra
    missing = [c for c in cols if c not in data]
    if missing:
        raise MethodIncompatibility(
            f"etpoisson: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    D = df[treatment].to_numpy(dtype=float)
    if np.any(Y < 0) or np.any(Y != np.round(Y)):
        raise MethodIncompatibility(
            f"etpoisson: y={y!r} must hold non-negative integers."
        )
    if not np.all(np.isin(D, (0.0, 1.0))) or np.unique(D).size < 2:
        raise MethodIncompatibility(
            f"etpoisson: treat={treatment!r} must be 0/1 with both values present."
        )
    one = np.ones(n)
    X = np.column_stack([df[v].to_numpy(dtype=float) for v in xs] + [D, one])
    W = np.column_stack([df[v].to_numpy(dtype=float) for v in zs] + [one])
    kx, kw = X.shape[1], W.shape[1]
    if n <= kx + kw + 2:
        raise DataInsufficient(
            f"etpoisson: {n} observations for {kx + kw + 2} parameters."
        )
    if np.linalg.matrix_rank(X) < kx or np.linalg.matrix_rank(W) < kw:
        raise MethodIncompatibility(
            "etpoisson: the regressors of one equation are collinear."
        )

    nodes, weights = np.polynomial.hermite.hermgauss(int(intpoints))
    log_w = np.log(weights) - 0.5 * np.log(np.pi)
    q = 2.0 * D - 1.0
    lgam = special.gammaln(Y + 1.0)

    def obs(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        xb = X @ theta[:kx]
        wg = W @ theta[kx : kx + kw]
        rho = np.tanh(theta[kx + kw])
        sigma = np.exp(theta[kx + kw + 1])
        # eps = sigma * sqrt(2) * node; u | eps ~ N(rho * eps / sigma, 1 - rho^2)
        a = np.sqrt(2.0) * nodes[None, :]
        eta = xb[:, None] + sigma * a
        log_pois = Y[:, None] * eta - np.exp(eta) - lgam[:, None]
        arg = q[:, None] * (wg[:, None] + rho * a) / np.sqrt(1.0 - rho * rho)
        terms = log_pois + special.log_ndtr(arg) + log_w[None, :]
        top = np.max(np.real(terms), axis=1, keepdims=True)
        return np.asarray(top[:, 0] + np.log(np.sum(np.exp(terms - top), axis=1)))

    # Start: Poisson for the outcome, probit for the treatment.
    b: np.ndarray = np.zeros(kx)
    b[-1] = np.log(max(float(Y.mean()), 1e-8))
    for _ in range(50):
        mu = np.exp(X @ b)
        step = np.linalg.solve((X * mu[:, None]).T @ X, X.T @ (Y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    g: np.ndarray = np.zeros(kw)
    for _ in range(50):
        idx = W @ g
        lam = q * np.exp(stats.norm.logpdf(q * idx) - special.log_ndtr(q * idx))
        Hm = (W * (lam * (idx + lam))[:, None]).T @ W
        step = np.linalg.solve(Hm, W.T @ lam)
        g = g + step
        if np.max(np.abs(step)) < 1e-10:
            break
    start = np.concatenate([b, g, [0.0, np.log(0.5)]])

    def neg(t: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = float(np.sum(np.real(obs(t))))
        return -v if np.isfinite(v) else 1e300

    res = optimize.minimize(neg, start, method="BFGS", options={"gtol": 1e-6})
    theta, scores, H, _ = ml_newton_polish(obs, np.asarray(res.x, dtype=float))
    ll = float(np.sum(np.real(obs(theta))))
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    clusters = df[cl].to_numpy() if se_kind == "cluster" else None
    V = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se = se_from_vcov(V)

    names = (
        [f"{y}:{v}" for v in xs]
        + [f"{y}:{treatment}", f"{y}:_cons"]
        + [f"{treatment}:{v}" for v in zs]
        + [f"{treatment}:_cons", "/athrho", "/lnsigma"]
    )
    i_rho, i_sig = kx + kw, kx + kw + 1
    rho, sigma = float(np.tanh(theta[i_rho])), float(np.exp(theta[i_sig]))
    chi2 = float((theta[i_rho] / se[i_rho]) ** 2)

    # Potential-outcome means and their difference, covariates held fixed.
    X0, X1 = X.copy(), X.copy()
    X0[:, kx - 2], X1[:, kx - 2] = 0.0, 1.0

    def means(t: np.ndarray) -> np.ndarray:
        half = 0.5 * np.exp(2.0 * t[i_sig])
        m0 = np.mean(np.exp(X0 @ t[:kx] + half))
        m1 = np.mean(np.exp(X1 @ t[:kx] + half))
        return np.array([m0, m1, m1 - m0])

    h = 1e-30
    J = np.column_stack(
        [
            np.imag(means(theta + 1j * h * np.eye(theta.size)[j])) / h
            for j in range(theta.size)
        ]
    )
    m = np.real(means(theta.astype(complex)))
    m_se = np.sqrt(np.diag(J @ V @ J.T))

    info: Dict[str, Any] = {
        "alpha": alpha,
        "model_type": "Poisson with endogenous treatment",
        "citation_key": "etpoisson",
        "method": "Maximum likelihood (Gauss-Hermite quadrature)",
        "intpoints": int(intpoints),
        "vce": se_kind,
        "cluster": cl if se_kind == "cluster" else None,
        "treatment": treatment,
        "ll": ll,
        "log_likelihood": ll,
        "aic": -2.0 * ll + 2.0 * theta.size,
        "bic": -2.0 * ll + np.log(n) * theta.size,
        "gradient_norm": grad_norm,
        "converged": bool(grad_norm < 1e-5 * max(1.0, n**0.5)),
        "rho": rho,
        "sigma": sigma,
        "independence_chi2": chi2,
        "independence_pvalue": float(stats.chi2.sf(chi2, 1)),
        "pomean_0": float(m[0]),
        "pomean_1": float(m[1]),
        "ate": float(m[2]),
        "ate_se": float(m_se[2]),
        "irr": float(np.exp(theta[kx - 2])),
    }
    if clusters is not None:
        info["n_clusters"] = int(pd.unique(clusters).size)
    return EconometricResults(
        params=pd.Series(theta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": n,
            "df_model": kx - 1,
            "df_resid": n - theta.size,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "inference": "z",
            "y": Y,
            "llobs": np.real(obs(theta)),
        },
        diagnostics={
            "Log-Likelihood": ll,
            "rho": rho,
            "sigma": sigma,
            "Wald test of indep. eqns. (chi2)": chi2,
            "Prob > chi2": info["independence_pvalue"],
            "ATE (count scale)": info["ate"],
        },
    )


# Citation. Mirrors paper.bib.
CausalResult._CITATIONS["etpoisson"] = (
    "@article{terza1998estimating,\n"
    "  title={Estimating Count Data Models with Endogenous Switching: Sample "
    "Selection and Endogenous Treatment Effects},\n"
    "  author={Terza, Joseph V.},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={84},\n"
    "  number={1},\n"
    "  pages={129--154},\n"
    "  year={1998},\n"
    "  doi={10.1016/S0304-4076(97)00082-1}\n"
    "}"
)
