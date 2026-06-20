"""Network regression for ``sp.network``.

Two complementary tools for regressing relational / dyadic outcomes:

* :func:`netlm` / :func:`netlogit` — QAP / MRQAP regression of one network
  matrix on others, with permutation inference robust to network
  autocorrelation (Krackhardt 1988; Dekker-Krackhardt-Snijders 2007
  double-semi-partialling).  These mirror ``sna::netlm`` / ``sna::netlogit``.

* :func:`dyadic_regression` — OLS on a dyadic data set with
  dyadic-cluster-robust standard errors that account for the dependence
  between dyads sharing a node (Fafchamps-Gubert 2007; Aronow-Samii-Assenova
  2015).

References
----------
Krackhardt, D. (1988). "Predicting with networks: nonparametric multiple
regression analysis of dyadic data." *Social Networks*, 10(4), 359-381.
[@krackhardt1988predicting]

Dekker, D., Krackhardt, D. & Snijders, T. A. B. (2007). "Sensitivity of
MRQAP tests to collinearity and autocorrelation conditions."
*Psychometrika*, 72(4), 563-581. [@dekker2007sensitivity]

Fafchamps, M. & Gubert, F. (2007). "The formation of risk sharing
networks." *Journal of Development Economics*, 83(2), 326-350.
[@fafchamps2007formation]

Aronow, P. M., Samii, C. & Assenova, V. A. (2015). "Cluster-robust variance
estimation for dyadic data." *Political Analysis*, 23(4), 564-577.
[@aronow2015cluster]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

__all__ = [
    "netlm",
    "netlogit",
    "QAPResult",
    "dyadic_regression",
    "DyadicRegressionResult",
]


# ====================================================================== #
#  Shared: turn network matrices into dyadic design vectors
# ====================================================================== #


def _to_matrix(obj: Any, n: Optional[int] = None) -> np.ndarray:
    g_like = obj
    if hasattr(obj, "adjacency_matrix"):
        return np.asarray(obj.adjacency_matrix(), dtype=float)
    M: np.ndarray = np.asarray(g_like, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("network matrices must be square 2-D arrays")
    return M


def _dyad_mask(n: int, directed: bool) -> np.ndarray:
    if directed:
        m = ~np.eye(n, dtype=bool)
    else:
        m = np.triu(np.ones((n, n), dtype=bool), k=1)
    return m


def _stack_predictors(
    predictors: Union[np.ndarray, Sequence, Mapping],
    n: int,
    mask: np.ndarray,
) -> "tuple[np.ndarray, List[str]]":
    """Return ``(design_without_intercept, names)`` from network matrices."""
    if isinstance(predictors, Mapping):
        items = list(predictors.items())
        names = [str(k) for k, _ in items]
        mats = [_to_matrix(v, n) for _, v in items]
    elif isinstance(predictors, np.ndarray) and predictors.ndim == 2:
        names = ["x1"]
        mats = [predictors.astype(float)]
    else:
        seq = list(predictors)
        names = [f"x{i + 1}" for i in range(len(seq))]
        mats = [_to_matrix(v, n) for v in seq]
    cols = [M[mask] for M in mats]
    return np.column_stack(cols) if cols else np.zeros((int(mask.sum()), 0)), names


def _permute_matrix(M: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(M[np.ix_(perm, perm)], dtype=float)


# ====================================================================== #
#  QAP / MRQAP
# ====================================================================== #


@dataclass
class QAPResult(ResultProtocolMixin):
    """QAP / MRQAP network-regression result (``sp.netlm`` / ``sp.netlogit``).

    Attributes
    ----------
    coefficients : pandas.DataFrame
        Columns ``variable``, ``coef``, ``se``, ``z``/``t``, ``p_qap``.
    p_qap : dict
        Permutation p-value per coefficient (the headline inference).
    r_squared : float
        OLS R^2 (``netlm``) or McFadden pseudo-R^2 (``netlogit``).
    n_dyads : int
    nperm : int
    method : str
        ``"netlm"`` or ``"netlogit"``; ``permutation`` records ``"dsp"`` or
        ``"y-permutation"``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 25
    >>> X = (rng.random((n, n)) < 0.3).astype(float); np.fill_diagonal(X, 0)
    >>> noise = rng.normal(0, 0.1, (n, n))
    >>> Y = 2.0 * X + noise
    >>> res = sp.netlm(Y, X, nperm=200, seed=1)
    >>> bool(1.5 < res.coefficients.loc[1, "coef"] < 2.5)
    True
    """

    _citation_keys = ("krackhardt1988predicting", "dekker2007sensitivity")

    coefficients: pd.DataFrame
    p_qap: Dict[str, float]
    r_squared: float
    n_dyads: int
    nperm: int
    method: str
    permutation: str = "dsp"
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.method} (QAP, {self.permutation}, {self.nperm} perms)\n"
            f"  n_dyads   : {self.n_dyads}\n"
            f"  R^2/pseudo: {self.r_squared:.4f}\n"
            f"{self.coefficients.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"QAPResult(method={self.method!r}, n_dyads={self.n_dyads})"


def _ols(y: np.ndarray, X: np.ndarray) -> "tuple[np.ndarray, np.ndarray, float]":
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = sse / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return beta, se, r2


def netlm(
    y: Any,
    predictors: Union[Any, Sequence, Mapping],
    directed: Optional[bool] = None,
    nperm: int = 1000,
    method: str = "dsp",
    seed: Optional[int] = None,
    intercept: bool = True,
) -> QAPResult:
    """MRQAP linear network regression (``sna::netlm`` analogue).

    Regress a dependent network matrix ``y`` on one or more predictor network
    matrices, with permutation inference that respects the dyadic dependence
    structure.  For multiple predictors the default ``method="dsp"`` is
    Dekker-Krackhardt-Snijders double-semi-partialling, which is robust to
    collinearity and network autocorrelation.

    Parameters
    ----------
    y : Graph or (n, n) array
    predictors : (n, n) array, sequence of arrays, or ``{name: array}``
    directed : bool, optional
        Whether dyads ``(i, j)`` and ``(j, i)`` are distinct.  Inferred from
        the symmetry of ``y`` when ``None``.
    nperm : int, default 1000
        Number of QAP permutations.
    method : {"dsp", "y"}, default "dsp"
        ``"dsp"`` = Dekker double-semi-partialling (recommended);
        ``"y"`` = permute the dependent matrix (classic Krackhardt QAP).
    seed : int, optional
    intercept : bool, default True

    Returns
    -------
    QAPResult

    References
    ----------
    dekker2007sensitivity

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    >>> Y = 1.0 + 2.0 * X
    >>> res = sp.netlm(Y, X, nperm=10, seed=0)
    >>> res.method
    'netlm'
    """
    Y = _to_matrix(y)
    n = Y.shape[0]
    if directed is None:
        directed = not np.allclose(Y, Y.T)
    mask = _dyad_mask(n, directed)
    yv = Y[mask]
    Xpred, names = _stack_predictors(predictors, n, mask)

    if intercept:
        X = np.column_stack([np.ones(len(yv)), Xpred])
        col_names = ["(Intercept)"] + names
    else:
        X = Xpred
        col_names = list(names)

    beta, se, r2 = _ols(yv, X)
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    perm_ge = np.zeros(p)

    if method == "dsp" and Xpred.shape[1] >= 1:
        # Double-semi-partialling: residualise each predictor on the others,
        # permute the residual matrix, refit, collect coefficient null.
        for perm_i in range(nperm):
            perm = rng.permutation(n)
            b_perm = np.empty(p)
            for kcol in range(p):
                if intercept and kcol == 0:
                    b_perm[kcol] = 0.0
                    continue
                pred_idx = kcol - 1 if intercept else kcol
                others = [c for c in range(X.shape[1]) if c != kcol]
                # residual of predictor kcol on the other design columns
                Xo = X[:, others]
                bk = np.linalg.pinv(Xo.T @ Xo) @ Xo.T @ Xpred[:, pred_idx]
                # residual back in matrix form, then permute by nodes
                Rk_vec = Xpred[:, pred_idx] - Xo @ bk
                Rk_mat = np.zeros((n, n))
                Rk_mat[mask] = Rk_vec
                if not directed:
                    Rk_mat = Rk_mat + Rk_mat.T
                Rk_perm = _permute_matrix(Rk_mat, perm)[mask]
                Xnew = X.copy()
                Xnew[:, kcol] = Rk_perm
                bb = np.linalg.pinv(Xnew.T @ Xnew) @ Xnew.T @ yv
                b_perm[kcol] = bb[kcol]
            perm_ge += (np.abs(b_perm) >= np.abs(beta) - 1e-12).astype(float)
        permutation = "dsp"
    else:
        # Classic QAP: permute the dependent matrix.
        for perm_i in range(nperm):
            perm = rng.permutation(n)
            yperm = _permute_matrix(Y, perm)[mask]
            bb = np.linalg.pinv(X.T @ X) @ X.T @ yperm
            perm_ge += (np.abs(bb) >= np.abs(beta) - 1e-12).astype(float)
        permutation = "y-permutation"

    p_qap = (perm_ge + 1.0) / (nperm + 1.0)
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    coef_df = pd.DataFrame(
        {
            "variable": col_names,
            "coef": beta,
            "se": se,
            "z": z,
            "p_qap": p_qap,
        }
    )
    return QAPResult(
        coefficients=coef_df,
        p_qap={col_names[i]: float(p_qap[i]) for i in range(p)},
        r_squared=float(r2),
        n_dyads=int(len(yv)),
        nperm=nperm,
        method="netlm",
        permutation=permutation,
        detail={"directed": directed, "intercept": intercept},
    )


def netlogit(
    y: Any,
    predictors: Union[Any, Sequence, Mapping],
    directed: Optional[bool] = None,
    nperm: int = 1000,
    seed: Optional[int] = None,
    intercept: bool = True,
    max_iter: int = 100,
) -> QAPResult:
    """QAP logistic network regression for a *binary* dependent network.

    Fits a logistic regression of the binarised dependent matrix on predictor
    matrices, with QAP (dependent-matrix-permutation) inference.  Analogue of
    ``sna::netlogit``.

    Parameters
    ----------
    y : Graph or (n, n) array
        Binary (0/1) dependent network (non-zero entries are treated as 1).
    predictors, directed, nperm, seed, intercept
        As in :func:`netlm`.
    max_iter : int, default 100
        IRLS iterations for the logistic fit.

    Returns
    -------
    QAPResult

    References
    ----------
    krackhardt1988predicting

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    >>> Y = (X > 0).astype(float)
    >>> res = sp.netlogit(Y, X, nperm=5, seed=0)
    >>> res.method
    'netlogit'
    """
    Y = (_to_matrix(y) != 0).astype(float)
    n = Y.shape[0]
    if directed is None:
        directed = not np.allclose(Y, Y.T)
    mask = _dyad_mask(n, directed)
    yv = Y[mask]
    Xpred, names = _stack_predictors(predictors, n, mask)
    if intercept:
        X = np.column_stack([np.ones(len(yv)), Xpred])
        col_names = ["(Intercept)"] + names
    else:
        X = Xpred
        col_names = list(names)

    beta, se, ll, ll0 = _irls_logit(yv, X, max_iter)
    pseudo_r2 = 1.0 - ll / ll0 if ll0 != 0 else 0.0

    rng = np.random.default_rng(seed)
    p = X.shape[1]
    perm_ge = np.zeros(p)
    for _ in range(nperm):
        perm = rng.permutation(n)
        yperm = _permute_matrix(Y, perm)[mask]
        bb, _, _, _ = _irls_logit(yperm, X, max_iter)
        perm_ge += (np.abs(bb) >= np.abs(beta) - 1e-12).astype(float)
    p_qap = (perm_ge + 1.0) / (nperm + 1.0)
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    coef_df = pd.DataFrame(
        {
            "variable": col_names,
            "coef": beta,
            "se": se,
            "z": z,
            "p_qap": p_qap,
        }
    )
    return QAPResult(
        coefficients=coef_df,
        p_qap={col_names[i]: float(p_qap[i]) for i in range(p)},
        r_squared=float(pseudo_r2),
        n_dyads=int(len(yv)),
        nperm=nperm,
        method="netlogit",
        permutation="y-permutation",
        detail={"directed": directed, "intercept": intercept},
    )


def _irls_logit(
    y: np.ndarray, X: np.ndarray, max_iter: int
) -> "tuple[np.ndarray, np.ndarray, float, float]":
    """Newton-Raphson / IRLS logistic fit; returns (beta, se, ll, ll_null)."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        W = mu * (1.0 - mu)
        W = np.clip(W, 1e-10, None)
        XtWX = X.T @ (W[:, None] * X)
        grad = X.T @ (y - mu)
        try:
            step = np.linalg.solve(XtWX, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(XtWX) @ grad
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    pbar = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    ll0 = float(len(y) * (pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar)))
    W = np.clip(mu * (1 - mu), 1e-10, None)
    cov = np.linalg.pinv(X.T @ (W[:, None] * X))
    se = np.sqrt(np.diag(cov))
    return beta, se, ll, ll0


# ====================================================================== #
#  Dyadic regression with dyadic-cluster-robust SEs
# ====================================================================== #


@dataclass
class DyadicRegressionResult(ResultProtocolMixin):
    """Dyadic OLS with dyadic-cluster-robust standard errors.

    Attributes
    ----------
    coefficients : pandas.DataFrame
        Columns ``variable``, ``coef``, ``se_dyadic``, ``se_classical``,
        ``z``, ``p``, ``ci_low``, ``ci_high``.
    n_dyads, n_nodes : int
    r_squared : float

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(20):
    ...     for j in range(i + 1, 20):
    ...         x = rng.normal()
    ...         rows.append((i, j, x, 1.0 + 0.5 * x + rng.normal(0, 0.5)))
    >>> df = pd.DataFrame(rows, columns=["i", "j", "x", "y"])
    >>> res = sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")
    >>> res.coefficients.loc[1, "variable"]
    'x'
    """

    _citation_keys = ("fafchamps2007formation", "aronow2015cluster")

    coefficients: pd.DataFrame
    n_dyads: int
    n_nodes: int
    r_squared: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            "Dyadic regression (Aronow-Samii-Assenova dyadic-robust SE)\n"
            f"  n_dyads / n_nodes : {self.n_dyads} / {self.n_nodes}\n"
            f"  R^2               : {self.r_squared:.4f}\n"
            f"{self.coefficients.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DyadicRegressionResult(n_dyads={self.n_dyads})"


def dyadic_regression(
    data: pd.DataFrame,
    y: str,
    covariates: Sequence[str],
    i: str,
    j: str,
    intercept: bool = True,
    alpha: float = 0.05,
) -> DyadicRegressionResult:
    """OLS on dyadic data with dyadic-cluster-robust standard errors.

    Estimates ``y_ij = x_ij' beta + e_ij`` by OLS and reports the
    Aronow-Samii-Assenova (2015) dyadic-cluster-robust variance, which
    allows arbitrary correlation between any two dyads that share a node —
    the dependence structure that invalidates classical / one-way clustered
    SEs in network data (Fafchamps-Gubert 2007).

    Parameters
    ----------
    data : pandas.DataFrame
        One row per dyad.
    y : str
        Outcome column.
    covariates : sequence of str
        Dyadic regressor columns.
    i, j : str
        Columns identifying the two nodes of each dyad.
    intercept : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    DyadicRegressionResult

    References
    ----------
    aronow2015cluster

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "i": [0, 0, 1, 1],
    ...     "j": [1, 2, 2, 3],
    ...     "x": [0.0, 1.0, 0.5, 1.5],
    ...     "y": [1.0, 2.0, 1.5, 2.5],
    ... })
    >>> res = sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")
    >>> res.n_dyads
    4
    """
    cols = list(covariates)
    needed = [y] + cols + [i, j]
    df = data[needed].dropna().reset_index(drop=True)
    yv = df[y].to_numpy(dtype=float)
    Xc = df[cols].to_numpy(dtype=float) if cols else np.zeros((len(df), 0))
    if intercept:
        X = np.column_stack([np.ones(len(df)), Xc])
        names = ["(Intercept)"] + cols
    else:
        X = Xc
        names = list(cols)

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    resid = yv - X @ beta
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0

    # classical (homoskedastic) SE for comparison
    dof = max(len(yv) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se_classical = np.sqrt(np.diag(sigma2 * XtX_inv))

    # ----- Aronow-Samii-Assenova dyadic-robust meat -----
    # score per dyad
    g = X * resid[:, None]  # (D, p)
    nodes = pd.unique(pd.concat([df[i], df[j]], ignore_index=True))
    p = X.shape[1]
    meat = np.zeros((p, p))
    # S_n = sum of scores over dyads incident to node n
    S = {nd: np.zeros(p) for nd in nodes}
    iv = df[i].to_numpy()
    jv = df[j].to_numpy()
    for d in range(len(df)):
        S[iv[d]] += g[d]
        S[jv[d]] += g[d]
    for nd in nodes:
        meat += np.outer(S[nd], S[nd])
    # subtract the diagonal (dyads share both their nodes -> double counted)
    meat -= g.T @ g
    V = XtX_inv @ meat @ XtX_inv
    se_dyadic = np.sqrt(np.clip(np.diag(V), 0, None))

    crit = float(stats.norm.ppf(1 - alpha / 2))
    z = np.divide(beta, se_dyadic, out=np.zeros_like(beta), where=se_dyadic > 0)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    coef_df = pd.DataFrame(
        {
            "variable": names,
            "coef": beta,
            "se_dyadic": se_dyadic,
            "se_classical": se_classical,
            "z": z,
            "p": pvals,
            "ci_low": beta - crit * se_dyadic,
            "ci_high": beta + crit * se_dyadic,
        }
    )
    return DyadicRegressionResult(
        coefficients=coef_df,
        n_dyads=int(len(df)),
        n_nodes=int(len(nodes)),
        r_squared=float(r2),
        detail={"intercept": intercept, "alpha": alpha},
    )
