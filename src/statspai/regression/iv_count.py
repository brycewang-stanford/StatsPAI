"""
Exponential-mean (Poisson) regression with endogenous regressors by GMM.

    E[y | x] = exp(x'beta)        with some x correlated with the error

Two error structures give two sets of moment conditions on the
instruments ``z``:

* additive,        ``y = exp(x'b) + e``:      ``E[z (y - exp(x'b))] = 0``
* multiplicative,  ``y = exp(x'b) * e``:      ``E[z (y / exp(x'b) - 1)] = 0``

The multiplicative form is Mullahy's (1997). It is the one that follows
from an omitted variable inside the exponential, which is the usual
story for endogeneity in a count or trade-flow model, and it is why the
Poisson first-order conditions with instruments in place of regressors
(the additive form) are not consistent in that case.

Follows Stata's ``ivpoisson gmm``: two-step by default, with a
heteroskedasticity-robust weight matrix and covariance.

References
----------
[@mullahy1997instrumental], [@windmeijer1997endogeneity]
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult, EconometricResults
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

__all__ = ["ivpoisson"]


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _moment_cov(
    Zu: np.ndarray,
    kind: str,
    clusters: Optional[np.ndarray],
    Z: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Covariance of the moment contributions, divided by n (Stata's S)."""
    n = Zu.shape[0]
    if kind == "unadjusted":
        return np.asarray(float(u @ u / n) * (Z.T @ Z) / n)
    if kind == "cluster":
        assert clusters is not None
        codes = pd.factorize(clusters)[0]
        sums = np.zeros((codes.max() + 1, Zu.shape[1]))
        np.add.at(sums, codes, Zu)
        return np.asarray(sums.T @ sums / n)
    return np.asarray(Zu.T @ Zu / n)


@accepts_aliases(robust="vce", covariates="x")
def ivpoisson(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    endog: Union[str, Sequence[str], None] = None,
    instruments: Union[str, Sequence[str], None] = None,
    errors: str = "additive",
    method: str = "twostep",
    vce: Optional[str] = "robust",
    cluster: Optional[str] = None,
    wmatrix: Optional[str] = None,
    alpha: float = 0.05,
    maxiter: int = 200,
    tol: float = 1e-10,
) -> EconometricResults:
    """
    Poisson (exponential mean) regression with endogenous regressors, GMM.

    Equivalent to Stata's ``ivpoisson gmm y x (endog = instruments)``;
    ``errors='multiplicative'`` is the ``multiplicative`` option.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Non-negative outcome. It need not be an integer: the model is the
        conditional mean ``exp(x'b)``, as for Poisson pseudo-likelihood.
    x : str or list of str, optional
        Exogenous regressors. A constant is added.
    endog : str or list of str
        Endogenous regressors.
    instruments : str or list of str
        Excluded instruments, at least as many as ``endog``.
    errors : {'additive', 'multiplicative'}, default 'additive'
        Which moment condition to use (see the module notes). The default
        is Stata's. ``'multiplicative'`` is the model in which an omitted
        variable enters the exponential; with it the constant absorbs the
        mean of the error, so the intercept is not comparable across the
        two.
    method : {'twostep', 'onestep', 'igmm'}, default 'twostep'
        ``'onestep'`` weights the moments by ``(Z'Z / n)^{-1}``;
        ``'twostep'`` re-weights once by the inverse covariance of the
        moments; ``'igmm'`` iterates that to convergence.
    vce : {'robust', 'cluster', 'unadjusted'}, default 'robust'
        Covariance of the moments, used for both the weight matrix and
        the standard errors. ``'unadjusted'`` assumes homoskedastic
        errors, under which two-step GMM reduces to one-step.
    cluster : str, optional
        Cluster variable (implies ``vce='cluster'``).
    wmatrix : {'robust', 'cluster', 'unadjusted'}, optional
        Covariance used for the weight matrix when it should differ from
        ``vce``. By default the two agree, as in Stata, where
        ``wmatrix()`` follows ``vce()`` unless it is given.
    alpha : float, default 0.05
    maxiter : int, default 200
    tol : float, default 1e-10

    Returns
    -------
    EconometricResults
        ``model_info`` carries Hansen's J (``j_stat``, ``j_df``,
        ``j_pvalue``), reported when the model is over-identified.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.ivpoisson(df, y="trips", x=["income"], endog="cost",
    ...                    instruments=["distance", "tolls"],
    ...                    errors="multiplicative")
    >>> print(res.summary())  # doctest: +SKIP
    >>> res.model_info["j_pvalue"]  # doctest: +SKIP

    References
    ----------
    [@mullahy1997instrumental], [@windmeijer1997endogeneity]
    """
    from ..core._vcov_spec import parse_se_request

    err = str(errors).lower()
    if err not in ("additive", "multiplicative"):
        raise MethodIncompatibility(
            f"ivpoisson: errors={errors!r}; use 'additive' or 'multiplicative'."
        )
    meth = {"two-step": "twostep", "one-step": "onestep", "iterated": "igmm"}.get(
        str(method).lower(), str(method).lower()
    )
    if meth not in ("twostep", "onestep", "igmm"):
        raise MethodIncompatibility(
            f"ivpoisson: method={method!r}; use 'twostep', 'onestep' or 'igmm'."
        )
    if isinstance(vce, str) and vce.lower() in ("unadjusted", "iid", "homoskedastic"):
        if cluster is not None:
            raise MethodIncompatibility(
                "ivpoisson: vce='unadjusted' and cluster= contradict each other."
            )
        kind, cl = "unadjusted", None
    else:
        req = parse_se_request(
            vce, cluster, function="ivpoisson", supported=("robust", "cluster")
        )
        kind, cl = req.kind, req.cluster

    w_kind = kind
    if wmatrix is not None:
        w_kind = {"iid": "unadjusted", "homoskedastic": "unadjusted"}.get(
            str(wmatrix).lower(), str(wmatrix).lower()
        )
        if w_kind not in ("robust", "cluster", "unadjusted"):
            raise MethodIncompatibility(
                f"ivpoisson: wmatrix={wmatrix!r}; use 'robust', 'cluster' or "
                "'unadjusted'."
            )
        if w_kind == "cluster" and cl is None:
            raise MethodIncompatibility("ivpoisson: wmatrix='cluster' needs cluster=.")

    xs, en, iv = _as_list(x), _as_list(endog), _as_list(instruments)
    if not en:
        raise MethodIncompatibility(
            "ivpoisson: endog= is empty. Without an endogenous regressor "
            "this is sp.poisson with robust standard errors."
        )
    overlap = sorted((set(en) & set(xs)) | (set(en) & set(iv)) | (set(xs) & set(iv)))
    if overlap:
        raise MethodIncompatibility(
            f"ivpoisson: {overlap} appear in more than one of x=, endog=, "
            "instruments=.",
            diagnostics={"overlap": overlap},
        )
    extra = [cl] if isinstance(cl, str) else []
    cols = [y] + en + xs + iv + extra
    missing = [c for c in cols if c not in data]
    if missing:
        raise MethodIncompatibility(
            f"ivpoisson: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    if len(iv) < len(en):
        raise MethodIncompatibility(
            f"ivpoisson: {len(en)} endogenous regressor(s) but only {len(iv)} "
            "excluded instrument(s)."
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    if np.any(Y < 0):
        raise MethodIncompatibility(
            f"ivpoisson: y={y!r} has negative values; an exponential mean "
            "cannot fit them."
        )
    one = np.ones(n)
    ex = [df[v].to_numpy(dtype=float) for v in xs]
    X = np.column_stack([df[v].to_numpy(dtype=float) for v in en] + ex + [one])
    Z = np.column_stack(ex + [df[v].to_numpy(dtype=float) for v in iv] + [one])
    k, q = X.shape[1], Z.shape[1]
    if n <= q:
        raise DataInsufficient(f"ivpoisson: {n} observations for {q} moments.")
    if np.linalg.matrix_rank(Z) < q:
        raise MethodIncompatibility(
            "ivpoisson: the exogenous variables and instruments are collinear."
        )
    clusters = df[cl].to_numpy() if isinstance(cl, str) else None
    names = en + xs + ["_cons"]
    multiplicative = err == "multiplicative"

    def resid(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Residual u and its derivative with respect to the index."""
        xb = X @ b
        if multiplicative:
            r = Y * np.exp(-xb)
            return r - 1.0, -r
        mu = np.exp(xb)
        return Y - mu, -mu

    def solve(b: np.ndarray, Wm: np.ndarray) -> np.ndarray:
        """Gauss-Newton on g(b)' W g(b) with step halving."""

        def crit(bb: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                g = Z.T @ resid(bb)[0] / n
                v = float(g @ Wm @ g)
            return v if np.isfinite(v) else np.inf

        cur = crit(b)
        for _ in range(maxiter):
            u, du = resid(b)
            g = Z.T @ u / n
            G = Z.T @ (X * du[:, None]) / n
            step = np.linalg.solve(G.T @ Wm @ G, G.T @ Wm @ g)
            scale = 1.0
            for _half in range(50):
                cand = b - scale * step
                new = crit(cand)
                if new <= cur:
                    break
                scale /= 2.0
            else:
                break
            b, cur = cand, new
            if np.max(np.abs(scale * step)) < tol * (1.0 + np.max(np.abs(b))):
                return b
        u, du = resid(b)
        G = Z.T @ (X * du[:, None]) / n
        grad = G.T @ Wm @ (Z.T @ u / n)
        if np.max(np.abs(grad)) > 1e-6:
            raise ConvergenceFailure(
                "ivpoisson: the GMM criterion did not converge "
                f"(gradient {np.max(np.abs(grad)):.2e}). With multiplicative "
                "errors this usually means a regressor on a very large scale; "
                "rescale it."
            )
        return b

    # Start from the exponential fit that ignores endogeneity.
    b: np.ndarray = np.zeros(k)
    b[-1] = np.log(max(float(Y.mean()), 1e-8))
    for _ in range(50):
        mu = np.exp(X @ b)
        step = np.linalg.solve((X * mu[:, None]).T @ X, X.T @ (Y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break

    W = np.linalg.inv(Z.T @ Z / n)
    b = solve(b, W)
    n_weight_updates = 0
    if meth != "onestep":
        for _ in range(1 if meth == "twostep" else maxiter):
            u, _ = resid(b)
            W = np.linalg.inv(_moment_cov(Z * u[:, None], w_kind, clusters, Z, u))
            b_new = solve(b, W)
            n_weight_updates += 1
            done = np.max(np.abs(b_new - b)) < 1e-9 * (1.0 + np.max(np.abs(b)))
            b = b_new
            if done:
                break

    u, du = resid(b)
    g = Z.T @ u / n
    G = Z.T @ (X * du[:, None]) / n
    S = _moment_cov(Z * u[:, None], kind, clusters, Z, u)
    bread = np.linalg.inv(G.T @ W @ G)
    V = bread @ (G.T @ W @ S @ W @ G) @ bread / n
    j_stat = float(n * g @ W @ g)
    if meth == "onestep":
        # (Z'Z/n)^{-1} is the efficient weight only up to the error variance
        # and only under homoskedasticity; scale as Stata does.
        j_stat /= float(u @ u / n)
    j_df = q - k
    se = np.sqrt(np.diag(V))
    info: Dict[str, Any] = {
        "alpha": alpha,
        "model_type": "IV Poisson (GMM)",
        "citation_key": "ivpoisson",
        "method": meth,
        "errors": err,
        "vce": kind,
        "wmatrix": w_kind,
        "cluster": cl if kind == "cluster" else None,
        "endog": en,
        "instruments": iv,
        "j_stat": j_stat if j_df > 0 else None,
        "j_df": j_df,
        "j_pvalue": float(stats.chi2.sf(j_stat, j_df)) if j_df > 0 else None,
        "n_weight_updates": n_weight_updates,
    }
    if meth == "onestep" and j_df > 0:
        info["j_note"] = (
            "One-step J is chi-squared only if the moments are homoskedastic; "
            "use method='twostep' for a test that is robust."
        )
    if kind == "cluster" and clusters is not None:
        info["n_clusters"] = int(pd.unique(clusters).size)
    mu_hat = np.exp(X @ b)
    return EconometricResults(
        params=pd.Series(b, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": n,
            "df_model": k - 1,
            "df_resid": n - k,
            "dependent_var": y,
            "fitted_values": mu_hat,
            "residuals": Y - mu_hat,
            "var_cov": V,
            "var_names": names,
            "inference": "z",
        },
        diagnostics=(
            {"Hansen J": j_stat, "Prob > chi2 (J)": info["j_pvalue"]}
            if j_df > 0
            else {}
        ),
    )


# Citation. Mirrors paper.bib.
CausalResult._CITATIONS["ivpoisson"] = (
    "@article{mullahy1997instrumental,\n"
    "  title={Instrumental-Variable Estimation of Count Data Models: "
    "Applications to Models of Cigarette Smoking Behavior},\n"
    "  author={Mullahy, John},\n"
    "  journal={Review of Economics and Statistics},\n"
    "  volume={79},\n"
    "  number={4},\n"
    "  pages={586--593},\n"
    "  year={1997},\n"
    "  doi={10.1162/003465397557169}\n"
    "}"
)
