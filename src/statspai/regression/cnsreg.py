"""Linear regression under linear equality constraints.

The model is ``y = x'b + e`` with ``R b = c``. Two estimators are offered.

*Constrained least squares* minimises the sum of squared residuals subject
to the constraints. It is computed by writing every admissible coefficient
vector as ``b = b0 + T g``, with ``b0`` one solution of ``R b0 = c`` and the
columns of ``T`` a basis of the null space of ``R``, and regressing
``y - X b0`` on ``X T``. The covariance of ``g`` is that of an ordinary
regression with ``k - q`` parameters, and ``V(b) = T V(g) T'``.

*Efficient minimum distance* starts from the unconstrained estimate and its
robust covariance ``V`` and moves to the closest admissible point in the
metric ``V^{-1}``: ``b = b_ols - V R' (R V R')^{-1} (R b_ols - c)``. Under
heteroskedasticity it has the smaller asymptotic variance of the two; under
homoskedasticity they coincide.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core._vcov import sandwich_vcov
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["cnsreg"]

_VCE = ("ols", "hc1", "hc2", "hc3", "cluster")


def _null_space(R: np.ndarray) -> np.ndarray:
    """Orthonormal basis of ``{b : R b = 0}`` (columns)."""
    _, s, vt = np.linalg.svd(R)
    rank = int((s > s.max() * max(R.shape) * np.finfo(float).eps).sum())
    return vt[rank:].T


def _robust_cov(
    X: np.ndarray,
    resid: np.ndarray,
    vce: str,
    clusters: Optional[np.ndarray],
    n_params: int,
) -> np.ndarray:
    """Covariance of the least-squares coefficients of ``X`` with residuals
    ``resid``, counting ``n_params`` estimated parameters."""
    n = X.shape[0]
    bread = np.linalg.inv(X.T @ X)
    if vce == "ols":
        return bread * float(resid @ resid) / (n - n_params)
    if vce == "cluster":
        return sandwich_vcov(
            bread,
            X * resid[:, None],
            clusters=clusters,
            correction="stata",
            n_params=n_params,
        )
    if vce == "hc1":
        weight = np.full(n, n / (n - n_params))
    else:
        h = np.einsum("ij,jk,ik->i", X, bread, X)
        weight = 1.0 / (1.0 - h) if vce == "hc2" else 1.0 / (1.0 - h) ** 2
    meat = (X * (resid**2 * weight)[:, None]).T @ X
    return bread @ meat @ bread


def cnsreg(
    formula: str,
    data: pd.DataFrame,
    constraints: Union[str, Sequence[str]],
    *,
    method: str = "cls",
    vce: str = "ols",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """Linear regression with linear equality constraints on the coefficients.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 + x3"``.
    data : pandas.DataFrame
        The data. Rows with a missing value in a model variable are dropped.
    constraints : str or list of str
        Linear restrictions on the coefficients, each written as in
        :func:`statspai.test`: ``"x1 + x2 + x3 = 0"``, ``"x1 = x2"``,
        ``"x1 = 0.5"``. ``_cons`` names the intercept. Redundant
        restrictions are dropped; contradictory ones are refused.
    method : {'cls', 'emd'}, default 'cls'
        ``'cls'`` is constrained least squares (Stata's ``cnsreg``).
        ``'emd'`` is the efficient minimum distance estimator, which weights
        the distance from the unconstrained estimate by the inverse of its
        robust covariance.
    vce : {'ols', 'robust', 'hc2', 'hc3'}, default 'ols'
        Covariance estimator; ``'robust'`` is HC1. With ``method='emd'`` the
        default is read as ``'robust'``, because with the classical
        covariance the two estimators are the same.
    cluster : str, optional
        Column to cluster on. Overrides ``vce``.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        Constrained coefficients for every regressor, with standard errors
        that respect the constraints (a coefficient fixed by a constraint
        has standard error zero). ``model_info`` holds the restriction
        matrix ``R`` and vector ``c``, the unconstrained coefficients and
        ``constraint_test``: the Wald statistic of the constraints at the
        unconstrained estimate, which is also the minimised distance of
        ``method='emd'``, with its chi-squared p-value.

    Notes
    -----
    Degrees of freedom are ``n - k + q`` for ``k`` regressors and ``q``
    independent constraints: the constrained model has ``k - q`` free
    parameters. The HC1 and cluster factors use the same count, as Stata's
    ``cnsreg`` does.

    The covariance of ``method='emd'`` is
    ``V - V R' (R V R')^{-1} R V`` with ``V`` the robust covariance of the
    unconstrained estimator evaluated at the *constrained* residuals and
    scaled by ``n / (n - k + q)`` [@hansen2022econometrics, chapter 8].

    A constraint that the data reject (see ``constraint_test``) makes both
    estimators inconsistent for the unconstrained coefficients.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({"k": rng.normal(size=n), "l": rng.normal(size=n)})
    >>> df["q"] = 1 + 0.3 * df.k + 0.7 * df.l + rng.normal(scale=0.5, size=n)
    >>> crs = sp.cnsreg("q ~ k + l", df, "k + l = 1", vce="robust")
    >>> round(float(crs.params["k"] + crs.params["l"]), 12)
    1.0
    >>> emd = sp.cnsreg("q ~ k + l", df, "k + l = 1", method="emd")
    >>> emd.model_info["constraint_test"]["df"]
    1

    References
    ----------
    hansen2022econometrics
    """
    from ..postestimation.hypothesis import _independent_restrictions, _parse_hypothesis
    from .ols import regress

    method = str(method).lower()
    if method not in ("cls", "emd"):
        raise MethodIncompatibility(
            f"sp.cnsreg: method={method!r} is not 'cls' or 'emd'.",
            recovery_hint="Use method='cls' (constrained least squares) or "
            "'emd' (efficient minimum distance).",
        )
    vce = {"robust": "hc1"}.get(str(vce).lower(), str(vce).lower())
    if cluster is not None:
        vce = "cluster"
    if vce not in _VCE or (vce == "cluster" and cluster is None):
        raise MethodIncompatibility(
            f"sp.cnsreg: vce={vce!r} is not 'ols', 'robust', 'hc2' or 'hc3'.",
            recovery_hint="Use vce='robust', or cluster='<column>'.",
        )
    if method == "emd" and vce == "ols":
        vce = "hc1"
    if isinstance(constraints, str):
        constraints = [constraints]
    constraints = list(constraints)
    if not constraints:
        raise MethodIncompatibility(
            "sp.cnsreg: no constraint given.",
            recovery_hint="Pass constraints='x1 + x2 = 1', or use sp.regress.",
        )

    ols = regress(formula, data=data)
    info = ols.data_info
    X = np.asarray(info["X"], dtype=float)
    y = np.asarray(info["y"], dtype=float)
    names: List[str] = [str(v) for v in info["var_names"]]
    n, k = X.shape
    clusters = None
    if cluster is not None:
        if cluster not in data.columns:
            raise MethodIncompatibility(
                f"sp.cnsreg: cluster={cluster!r} is not a column.",
                recovery_hint="Pass the name of the cluster variable.",
            )
        used = data.loc[info["sample_index"]]
        if used[cluster].isna().any():
            raise MethodIncompatibility(
                "sp.cnsreg: the cluster variable is missing on rows of the "
                "estimation sample.",
                recovery_hint="Drop the rows with a missing cluster first.",
            )
        clusters = pd.factorize(used[cluster])[0]

    b_ols = np.asarray(ols.params, dtype=float)
    R, c = _parse_hypothesis(constraints, pd.Series(b_ols, index=names))
    R, c = _independent_restrictions(R, c, "; ".join(constraints))
    q = R.shape[0]
    if q >= k:
        raise MethodIncompatibility(
            f"sp.cnsreg: {q} independent constraints on {k} coefficients "
            "leave nothing to estimate.",
            recovery_hint="Remove constraints.",
        )
    free = k - q
    if n <= free:
        raise DataInsufficient(
            "sp.cnsreg: not enough observations.",
            recovery_hint="Use fewer regressors.",
        )
    df_resid = n - free

    V_ols = _robust_cov(X, y - X @ b_ols, vce, clusters, k)
    gap = R @ b_ols - c
    wald = float(gap @ np.linalg.solve(R @ V_ols @ R.T, gap))

    if method == "cls":
        T = _null_space(R)
        b0 = np.linalg.lstsq(R, c, rcond=None)[0]
        XT = X @ T
        g = np.linalg.lstsq(XT, y - X @ b0, rcond=None)[0]
        beta = b0 + T @ g
        resid = y - X @ beta
        cov = T @ _robust_cov(XT, resid, vce, clusters, free) @ T.T
    else:
        beta = b_ols - V_ols @ R.T @ np.linalg.solve(R @ V_ols @ R.T, gap)
        resid = y - X @ beta
        V = _robust_cov(X, resid, vce, clusters, free)
        cov = V - V @ R.T @ np.linalg.solve(R @ V @ R.T, R @ V)
    cov = (cov + cov.T) / 2.0
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    rss = float(resid @ resid)
    const = next((j for j in range(k) if np.ptp(X[:, j]) == 0 and X[0, j] != 0), None)
    tss = float(((y - y.mean()) ** 2).sum()) if const is not None else float(y @ y)
    # The model F: the slopes are tested one after another and a slope
    # whose test is implied by the earlier ones under the constraints (its
    # variance adds no rank) is left out, which is how Stata's `test`
    # handles a singular covariance. With a constraint that fixes a
    # coefficient away from zero the statistic is a convention, not a test
    # of a hypothesis the model allows.
    tested: List[int] = []
    for col in (c_ for c_ in range(k) if c_ != const):
        block = cov[np.ix_(tested + [col], tested + [col])]
        scale = np.sqrt(np.clip(np.diag(block), 0.0, None))
        if scale.min() <= 0:
            continue
        unit = block / np.outer(scale, scale)
        if np.linalg.matrix_rank(unit, tol=1e-9) > len(tested):
            tested.append(col)
    if tested:
        b_t = beta[tested]
        f_stat = float(b_t @ np.linalg.solve(cov[np.ix_(tested, tested)], b_t)) / len(
            tested
        )
        df_f = df_resid if clusters is None else int(clusters.max())
        f_p = float(stats.f.sf(f_stat, len(tested), df_f))
    else:
        f_stat, f_p = float("nan"), float("nan")

    label = {"ols": "nonrobust", "hc1": "robust"}.get(vce, vce)
    model_info: Dict[str, Any] = {
        "model_type": (
            "Constrained least squares"
            if method == "cls"
            else "Efficient minimum distance"
        ),
        "method": method,
        "constraints": constraints,
        "R": R,
        "c": c,
        "n_constraints": q,
        "unconstrained_params": pd.Series(b_ols, index=names),
        "constraint_test": {
            "chi2": wald,
            "df": q,
            "pvalue": float(stats.chi2.sf(wald, q)),
        },
        "robust": label,
        "cluster": cluster,
        "alpha": alpha,
    }
    if clusters is not None:
        model_info["n_clusters"] = int(clusters.max()) + 1
    data_info: Dict[str, Any] = {
        "nobs": n,
        "df_model": free - (const is not None),
        "df_resid": df_resid if clusters is None else int(clusters.max()),
        "dependent_var": info.get("dependent_var"),
        "var_names": names,
        "var_cov": cov,
        "X": X,
        "y": y,
        "residuals": resid,
        "fitted_values": X @ beta,
        "rss": rss,
        "tss": tss,
    }
    diagnostics = {
        "F-statistic": f_stat,
        "Prob (F-statistic)": f_p,
        "Root MSE": float(np.sqrt(rss / df_resid)),
        "Constraint chi2": wald,
        "Constraint p-value": model_info["constraint_test"]["pvalue"],
    }
    index = pd.Index(names)
    return EconometricResults(
        params=pd.Series(beta, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
