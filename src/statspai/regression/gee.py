"""
Generalized estimating equations (Liang and Zeger 1986).

``sp.gee`` fits a marginal mean model ``g(E[y_it | x_it]) = x_it'b`` to
clustered or longitudinal data. The regression coefficients are consistent
whenever the mean model is right; the working correlation only buys
efficiency, and the sandwich covariance stays valid when it is wrong. It is
the population-averaged counterpart of the random-effects models in
``sp.multilevel``: Stata ``xtgee`` / ``xtreg, pa``, R ``gee::gee`` and
``geepack::geeglm``.

The moment estimators of the scale and of the working correlation follow
Liang and Zeger [@liang1986longitudinal]. The two reference implementations
differ in one divisor, exposed as ``dof_correction``: R ``gee`` divides by
``N - p`` (and by ``pairs - p``), Stata ``xtgee`` by ``N`` (and ``pairs``)
unless its ``nmp`` option is given.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility
from .glm import FAMILIES, _get_link

_CORSTR = {
    "independence": "independence",
    "independent": "independence",
    "ind": "independence",
    "exchangeable": "exchangeable",
    "exch": "exchangeable",
    "exc": "exchangeable",
    "ar1": "ar1",
    "ar": "ar1",
    "ar-m": "ar-m",
    "arm": "ar-m",
    "unstructured": "unstructured",
    "uns": "unstructured",
}


def _working_correlation(kind: str, alpha: Any, m: int) -> np.ndarray:
    """The ``m x m`` working correlation matrix of one cluster."""
    if kind == "independence" or m == 1:
        return np.eye(m)
    if kind == "exchangeable":
        R = np.full((m, m), float(alpha))
        np.fill_diagonal(R, 1.0)
        return R
    if kind in ("ar1", "ar-m"):
        lag = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
        return np.asarray(float(alpha) ** lag, dtype=float)
    return np.asarray(alpha, dtype=float)[:m, :m]


def _estimate_alpha(
    kind: str,
    resid: List[np.ndarray],
    phi: float,
    p: int,
    correct: bool,
) -> Any:
    """Moment estimator of the working-correlation parameters.

    ``resid`` holds the Pearson residuals cluster by cluster.
    """
    if kind == "independence":
        return 0.0
    if kind == "exchangeable":
        num = 0.0
        pairs = 0.0
        for r in resid:
            m = len(r)
            num += (r.sum() ** 2 - (r**2).sum()) / 2.0
            pairs += m * (m - 1) / 2.0
        denom = pairs - (p if correct else 0)
        if denom <= 0:
            raise DataInsufficient(
                "gee: no within-cluster pairs left to estimate the "
                "exchangeable correlation.",
                recovery_hint="Use corstr='independence'.",
            )
        return float(num / (denom * phi))
    if kind == "ar-m":
        # R gee's "AR-M" with M = 1: each cluster's mean lag-one product
        # over each cluster's mean square, summed across clusters. The
        # scale cancels and there is no degrees-of-freedom term. On a
        # balanced panel it is the pooled moment below without the
        # correction; with unequal cluster sizes short clusters count for
        # more here than there.
        top = 0.0
        bottom = 0.0
        for r in resid:
            if len(r) > 1:
                top += float(np.sum(r[:-1] * r[1:])) / (len(r) - 1)
            bottom += float(np.sum(r**2)) / len(r)
        if top == 0.0 and all(len(r) < 2 for r in resid):
            raise DataInsufficient(
                "gee: no adjacent within-cluster pairs to estimate the AR(1) "
                "correlation.",
                recovery_hint="Use corstr='independence'.",
            )
        return float(top / bottom)
    if kind == "ar1":
        num = 0.0
        pairs = 0.0
        for r in resid:
            if len(r) > 1:
                num += float(np.sum(r[:-1] * r[1:]))
                pairs += len(r) - 1
        denom = pairs - (p if correct else 0)
        if denom <= 0:
            raise DataInsufficient(
                "gee: no adjacent within-cluster pairs to estimate the AR(1) "
                "correlation.",
                recovery_hint="Use corstr='independence'.",
            )
        return float(num / (denom * phi))
    # unstructured: one correlation per pair of positions; clusters shorter
    # than the longest contribute to the positions they have
    m_max = max(len(r) for r in resid)
    num_m = np.zeros((m_max, m_max))
    cnt = np.zeros((m_max, m_max))
    for r in resid:
        m = len(r)
        num_m[:m, :m] += np.outer(r, r)
        cnt[:m, :m] += 1.0
    denom_m = cnt - (p if correct else 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(denom_m > 0, num_m / (denom_m * phi), 0.0)
    np.fill_diagonal(R, 1.0)
    return (R + R.T) / 2.0


def gee(
    formula: str,
    data: pd.DataFrame,
    id: str,
    family: str = "gaussian",
    link: Optional[str] = None,
    corstr: str = "independence",
    time: Optional[str] = None,
    vce: str = "robust",
    dof_correction: bool = True,
    scale: Optional[float] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Generalized estimating equations for clustered and longitudinal data.

    Fits the population-averaged model ``g(E[y | x]) = x'b`` with a working
    correlation inside each cluster and reports the cluster-robust sandwich
    covariance, which is valid whether or not the working correlation is
    right. Equivalent to R ``gee::gee(formula, id=, family=, corstr=)`` and
    Stata ``xtgee y x, family() link() corr()``.

    Parameters
    ----------
    formula : str
        Model formula, e.g. ``"y ~ treat + x"``. Factors (``C(g)``),
        interactions and transformations are allowed.
    data : pd.DataFrame
        One row per observation. Rows with a missing value in the formula,
        ``id`` or ``time`` are dropped.
    id : str
        Cluster identifier (subject, village, mouse). Observations of a
        cluster need not be contiguous.
    family : {"gaussian", "binomial", "poisson", "gamma"}, default "gaussian"
        Variance function of the marginal model.
    link : str, optional
        Link function; the family's canonical link when omitted.
    corstr : {"independence", "exchangeable", "ar1", "ar-m", "unstructured"}
        Working correlation. ``"independence"`` gives the pooled GLM point
        estimates with a cluster-robust covariance. ``"ar1"`` and
        ``"ar-m"`` are the same AR(1) structure with Stata's and R
        ``gee``'s estimator of its parameter (see Notes).
    time : str, optional
        Column that orders observations within a cluster. Needed for
        ``"ar1"`` and ``"unstructured"`` unless the rows are already in
        time order; it is used to sort, so unequally spaced times are
        treated as consecutive positions.
    vce : {"robust", "model"}, default "robust"
        ``"robust"`` is the Liang-Zeger sandwich (R ``gee``'s "Robust
        S.E.", Stata ``vce(robust)``); ``"model"`` the naive covariance
        that takes the working correlation at its word. Both are returned
        in ``model_info`` whichever is chosen.
    dof_correction : bool, default True
        Divisor of the scale and correlation moments. ``True`` uses
        ``N - p`` (R ``gee``; Stata ``xtgee, nmp``), ``False`` uses ``N``
        (Stata ``xtgee`` default). The coefficients of an exchangeable or
        AR(1) fit and every model-based standard error depend on it; with
        ``corstr="independence"`` only the model-based standard errors do.
    scale : float, optional
        Fix the scale of the model-based covariance at this value instead
        of estimating it. ``scale=1`` reproduces the model-based standard
        errors Stata ``xtgee`` prints for the binomial and Poisson
        families, where it holds the scale at one; R ``gee`` estimates it.
        The coefficients, the working correlation and the robust
        covariance do not depend on it.
    maxiter : int, default 100
        Maximum number of iterations.
    tol : float, default 1e-8
        Convergence tolerance on the largest relative coefficient change.
    alpha : float, default 0.05
        Significance level of the confidence intervals.

    Returns
    -------
    EconometricResults
        Coefficients with normal-based inference. ``model_info`` carries
        ``scale``, ``working_correlation``, ``corr_alpha`` (the scalar
        parameter for exchangeable and AR(1)), ``se_model``, ``se_robust``,
        ``n_clusters`` and ``iterations``.

    Notes
    -----
    Each iteration alternates a Fisher-scoring step on ``b`` given the
    working covariance ``V_i = phi A_i^{1/2} R(a) A_i^{1/2}`` with moment
    updates of ``phi`` (mean squared Pearson residual) and ``a`` (mean
    product of Pearson residuals over within-cluster pairs, all pairs for
    exchangeable and adjacent pairs for AR(1)).

    The sandwich has no small-sample factor, as in R ``gee``. Stata
    ``xtgee, vce(robust)`` multiplies the same matrix by ``G / (G - 1)``,
    so its robust standard errors are larger by ``sqrt(G / (G - 1))``.
    With few clusters (under roughly 40) either version is biased
    downward; treat the standard errors as a lower bound there.

    Two moment estimators of the AR(1) correlation are in use and both are
    offered. ``corstr="ar1"`` pools the products of adjacent Pearson
    residuals over all clusters, divided by the number of pairs (minus
    ``p`` under ``dof_correction``) and the scale: Stata ``xtgee, corr(ar
    1)``. ``corstr="ar-m"`` is R ``gee``'s ``"AR-M"`` with ``Mv = 1``: the
    sum over clusters of each cluster's mean adjacent product, over the
    sum of each cluster's mean square. The two agree on a balanced panel
    up to the degrees-of-freedom term and part ways when cluster sizes
    differ, where the second gives short clusters more weight. Both are
    consistent; the coefficients differ by a few percent of a standard
    error.

    A cluster-level working correlation requires every cluster to be small
    enough to factor: memory grows with the square of the largest cluster.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(100), 5)
    >>> u = rng.normal(size=100)[g]
    >>> x = rng.normal(size=500)
    >>> df = pd.DataFrame({"g": g, "x": x,
    ...                    "y": 1 + 0.5 * x + u + rng.normal(size=500)})
    >>> res = sp.gee("y ~ x", df, id="g", corstr="exchangeable")
    >>> bool(abs(res.params["x"] - 0.5) < 0.15)
    True
    >>> bool(0.3 < res.model_info["corr_alpha"] < 0.7)
    True

    References
    ----------
    [@liang1986longitudinal]
    """
    kind = _CORSTR.get(str(corstr).lower())
    if kind is None:
        raise MethodIncompatibility(
            f"gee: corstr={corstr!r} is not one of 'independence', "
            "'exchangeable', 'ar1', 'ar-m', 'unstructured'.",
            diagnostics={"corstr": corstr},
        )
    vce_key = str(vce).lower()
    if vce_key in ("naive", "conventional", "nonrobust"):
        vce_key = "model"
    if vce_key not in ("robust", "model"):
        raise MethodIncompatibility(
            f"gee: vce={vce!r} is not 'robust' or 'model'.",
            diagnostics={"vce": vce},
        )
    fam_key = str(family).lower()
    if fam_key not in FAMILIES:
        raise MethodIncompatibility(
            f"gee: unknown family {family!r}. Choose from: "
            f"{', '.join(FAMILIES.keys())}.",
            diagnostics={"family": family},
        )
    for col in [id] + ([time] if time is not None else []):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"gee: column {col!r} not found in data.",
                diagnostics={"column": col},
            )
    fam = FAMILIES[fam_key]()
    lnk = _get_link(link, fam)

    keep = data.dropna(subset=[id] + ([time] if time is not None else []))
    y_df, X_df = create_design_matrices(formula, keep)
    rows = X_df.index
    names = [str(c) for c in X_df.columns]
    y = np.asarray(y_df, dtype=float).reshape(len(rows), -1)[:, -1]
    X = np.asarray(X_df, dtype=float)
    n, p = X.shape
    groups = keep.loc[rows, id]
    codes = pd.factorize(groups, sort=True)[0]
    if time is not None:
        order = np.lexsort((keep.loc[rows, time].to_numpy(), codes))
    else:
        order = np.argsort(codes, kind="stable")
    y, X, codes = y[order], X[order], codes[order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [n]])
    G = len(starts)
    if G < 2:
        raise DataInsufficient(
            "gee: at least two clusters are needed.",
            diagnostics={"n_clusters": int(G)},
        )
    if n <= p:
        raise DataInsufficient(
            f"gee: {n} observations for {p} coefficients.",
        )

    # start from the pooled GLM fit (independence working correlation)
    mu = fam.initialize_mu(y)
    eta = lnk.link(mu)
    beta: np.ndarray = np.zeros(p)
    for _ in range(200):
        gp = lnk.deriv(mu)
        w = 1.0 / (fam.variance(mu) * gp**2)
        z = eta + (y - mu) * gp
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        done = np.max(np.abs(beta_new - beta)) <= 1e-12 * max(
            1.0, float(np.max(np.abs(beta_new)))
        )
        beta = beta_new
        eta = X @ beta
        mu = lnk.inverse(eta)
        if done:
            break

    def moments(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Any]:
        mu_b = lnk.inverse(X @ b)
        v = fam.variance(mu_b)
        pearson = (y - mu_b) / np.sqrt(v)
        phi_b = float(np.sum(pearson**2) / ((n - p) if dof_correction else n))
        a_b = _estimate_alpha(
            kind,
            [pearson[s:e] for s, e in zip(starts, stops)],
            phi_b,
            p,
            dof_correction,
        )
        return mu_b, v, pearson, phi_b, a_b

    def pieces(
        b: np.ndarray, a_now: Any, with_meat: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``sum D'V^-1 D``, ``sum D'V^-1 (y - mu)`` and the sandwich meat,
        with the scale left out of ``V`` (it cancels in the update)."""
        mu_b = lnk.inverse(X @ b)
        sd = np.sqrt(fam.variance(mu_b))
        Dm = X / lnk.deriv(mu_b)[:, None]
        e = y - mu_b
        bread = np.zeros((p, p))
        score = np.zeros(p)
        meat = np.zeros((p, p))
        if kind == "independence":
            Dw = Dm / sd[:, None] ** 2
            bread = Dw.T @ Dm
            score = Dw.T @ e
            if with_meat:
                u = Dw * e[:, None]
                per = np.add.reduceat(u, starts, axis=0)
                meat = per.T @ per
            return bread, score, meat
        for s, t in zip(starts, stops):
            m = t - s
            Di = Dm[s:t] / sd[s:t, None]
            ei = e[s:t] / sd[s:t]
            if kind == "exchangeable" and m > 1:
                # R^-1 = [I - a/(1 + (m-1)a) 11'] / (1 - a)
                a = float(a_now)
                c = a / (1.0 + (m - 1) * a)
                RiD = (Di - c * Di.sum(axis=0)) / (1.0 - a)
                Rie = (ei - c * ei.sum()) / (1.0 - a)
            else:
                Rm = _working_correlation(kind, a_now, m)
                sol = np.linalg.solve(Rm, np.column_stack([Di, ei]))
                RiD, Rie = sol[:, :p], sol[:, p]
            bread += Di.T @ RiD
            ui = Di.T @ Rie
            score += ui
            if with_meat:
                meat += np.outer(ui, ui)
        return bread, score, meat

    converged = kind == "independence"
    iterations = 0
    _, _, _, phi, a_hat = moments(beta)
    if not converged:
        for iterations in range(1, maxiter + 1):
            _check_alpha(kind, a_hat, int(max(stops - starts)))
            bread, score, _ = pieces(beta, a_hat, with_meat=False)
            step = np.linalg.solve(bread, score)
            beta = np.asarray(beta + step, dtype=float)
            _, _, _, phi, a_hat = moments(beta)
            if np.max(np.abs(step)) <= tol * max(1.0, float(np.max(np.abs(beta)))):
                converged = True
                break
    if not converged:
        raise ConvergenceFailure(
            f"gee: no convergence in {maxiter} iterations.",
            recovery_hint="Raise maxiter, or try corstr='independence'.",
            diagnostics={"iterations": iterations, "corstr": kind},
        )
    _check_alpha(kind, a_hat, int(max(stops - starts)))

    bread, _, meat = pieces(beta, a_hat, with_meat=True)
    bread_inv = np.linalg.inv(bread)
    if scale is not None:
        if not np.isfinite(scale) or scale <= 0:
            raise MethodIncompatibility(
                f"gee: scale must be a positive number, got {scale!r}.",
            )
    cov_model = (phi if scale is None else float(scale)) * bread_inv
    cov_robust = bread_inv @ meat @ bread_inv
    cov = cov_robust if vce_key == "robust" else cov_model
    se = np.sqrt(np.diag(cov))

    sizes = stops - starts
    m_max = int(sizes.max())
    wc = _working_correlation(kind, a_hat, m_max)
    params = pd.Series(beta, index=names)
    std_errors = pd.Series(se, index=names)
    mu_hat = lnk.inverse(X @ beta)

    if G < 40 and vce_key == "robust":
        warnings.warn(
            f"gee: the sandwich covariance rests on {G} clusters; with fewer "
            "than about 40 it understates the sampling variance.",
            UserWarning,
            stacklevel=2,
        )

    model_info: Dict[str, Any] = {
        "model_type": "GEE",
        "method": f"GEE ({fam_key}, {lnk.name} link, {kind})",
        "family": fam_key,
        "link": lnk.name,
        "corstr": kind,
        "vce": vce_key,
        "robust": "cluster" if vce_key == "robust" else "nonrobust",
        "cluster": id,
        "scale": phi,
        "scale_fixed": None if scale is None else float(scale),
        "corr_alpha": None if isinstance(a_hat, np.ndarray) else float(a_hat),
        "working_correlation": wc,
        "se_model": pd.Series(np.sqrt(np.diag(cov_model)), index=names),
        "se_robust": pd.Series(np.sqrt(np.diag(cov_robust)), index=names),
        "vcov_model": pd.DataFrame(cov_model, index=names, columns=names),
        "vcov_robust": pd.DataFrame(cov_robust, index=names, columns=names),
        "vcov": pd.DataFrame(cov, index=names, columns=names),
        "n_clusters": int(G),
        "iterations": int(iterations),
        "dof_correction": bool(dof_correction),
        "formula": formula,
        "alpha": alpha,
    }
    data_info = {
        "nobs": int(n),
        "n_clusters": int(G),
        "dependent_var": str(y_df.columns[-1]) if hasattr(y_df, "columns") else "y",
        "df_resid": np.inf,
        "cluster_size_min": int(sizes.min()),
        "cluster_size_max": m_max,
        "fitted_values": mu_hat,
        "residuals": y - mu_hat,
    }
    diagnostics = {
        "Scale": phi,
        "Clusters": int(G),
        "Cluster size (min)": int(sizes.min()),
        "Cluster size (max)": m_max,
        "Iterations": int(iterations),
    }
    if model_info["corr_alpha"] is not None and kind != "independence":
        diagnostics["Working correlation"] = model_info["corr_alpha"]
    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _check_alpha(kind: str, a_hat: Any, m_max: int) -> None:
    """Refuse a working correlation that is not positive definite."""
    if kind == "exchangeable":
        lo = -1.0 / max(m_max - 1, 1)
        ok = lo < float(a_hat) < 1.0
    elif kind in ("ar1", "ar-m"):
        ok = abs(float(a_hat)) < 1.0
    elif kind == "unstructured":
        ok = bool(np.min(np.linalg.eigvalsh(np.asarray(a_hat))) > 1e-10)
    else:
        return
    if not ok:
        raise MethodIncompatibility(
            f"gee: the estimated {kind} working correlation is not positive "
            "definite.",
            recovery_hint=(
                "Use corstr='independence' (still consistent, with the same "
                "robust covariance) or a simpler structure."
            ),
            diagnostics={"corstr": kind},
        )
