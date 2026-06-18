"""Geographically Weighted Regression (Fotheringham, Brunsdon, Charlton 2002).

Core model — for each observation ``i`` fit a locally weighted least squares:

    β̂_i = (X' W_i X)^{-1} X' W_i y

where ``W_i`` is a diagonal matrix of kernel-distance weights centred on ``i``.

Kernels: Gaussian, bisquare, exponential.
Bandwidth: fixed metric distance OR adaptive (k-nearest neighbours).

The result object mirrors mgwr's ``GWR.fit()`` return: per-observation
parameter matrix ``params (n × k)``, predictions, residuals, local R²,
effective degrees of freedom (trace of the hat matrix), AICc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

KernelName = Literal["gaussian", "bisquare", "exponential"]


# --------------------------------------------------------------------- #
#  Kernel functions
# --------------------------------------------------------------------- #


def _kernel(u: np.ndarray, kernel: KernelName) -> np.ndarray:
    """Kernel evaluated at the scaled distance ``u = d / bw``.

    Gaussian has infinite support; bisquare and exponential clip at u=1
    (return 0 beyond). The bisquare form is the default in mgwr.
    """
    u = np.asarray(u, dtype=float)
    if kernel == "gaussian":
        return np.asarray(np.exp(-0.5 * u * u), dtype=float)
    if kernel == "bisquare":
        return np.asarray(np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0), dtype=float)
    if kernel == "exponential":
        return np.asarray(np.where(u < 1.0, np.exp(-u), 0.0), dtype=float)
    raise ValueError(f"unknown kernel {kernel!r}")


# --------------------------------------------------------------------- #
#  Weight construction (one row of W_i)
# --------------------------------------------------------------------- #


def _weights_row(
    coords: np.ndarray,
    i: int,
    bw: float,
    kernel: KernelName,
    fixed: bool,
    tree: cKDTree,
) -> np.ndarray:
    n = coords.shape[0]
    if fixed:
        d = np.linalg.norm(coords - coords[i], axis=1)
        return _kernel(d / bw, kernel)
    # adaptive: bw is number of nearest neighbours (can be float from
    # golden-section search — use floor + fractional interpolation).
    k = int(np.ceil(bw))
    k = max(min(k, n), 2)
    dists, idx = tree.query(coords[i], k=k)
    w = np.zeros(n)
    # use the k-th distance as bandwidth scale
    scale = dists[-1] if dists[-1] > 0 else 1.0
    u = dists / scale
    w[idx] = _kernel(u, kernel)
    return w


# --------------------------------------------------------------------- #
#  Result dataclass
# --------------------------------------------------------------------- #


@dataclass
class GWRResult:
    params: np.ndarray  # (n, k)
    predicted: np.ndarray  # (n,)
    residuals: np.ndarray  # (n,)
    bw: float
    kernel: KernelName
    fixed: bool
    local_R2: np.ndarray  # (n,)
    aicc: float
    aic: float
    bic: float
    R2: float
    resid_ss: float
    tr_S: float  # effective df = tr(H)
    n: int
    k: int
    se: Optional[np.ndarray] = None  # (n, k) local coefficient standard errors
    tvals: Optional[np.ndarray] = None  # (n, k) local t-statistics

    def summary(self) -> str:
        lines = [
            "Geographically Weighted Regression",
            "-" * 40,
            f"Bandwidth     : {self.bw:.3f}"
            f"  ({'fixed' if self.fixed else 'adaptive / NN'})",
            f"Kernel        : {self.kernel}",
            f"n             : {self.n}",
            f"k             : {self.k}",
            f"Effective DF  : {self.tr_S:.3f}",
            f"R²            : {self.R2:.4f}",
            f"AICc          : {self.aicc:.2f}",
            f"Residual SS   : {self.resid_ss:.3f}",
            "",
            "Local coefficient summary:",
        ]
        for j in range(self.k):
            col = self.params[:, j]
            lines.append(
                f"  β{j}  mean={col.mean(): .4f}  std={col.std(): .4f}"
                f"  min={col.min(): .4f}  max={col.max(): .4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  GWR fit
# --------------------------------------------------------------------- #


def gwr(
    coords: Any,
    y: Any,
    X: Any,
    bw: float,
    kernel: KernelName = "bisquare",
    fixed: bool = False,
    add_constant: bool = True,
) -> GWRResult:
    """Fit Geographically Weighted Regression.

    Parameters
    ----------
    coords : (n, 2) array-like
        Point coordinates (projected — use UTM or equivalent; lat/lon
        will bias distances in most regions).
    y : (n,) or (n, 1) array-like
    X : (n, p) array-like
        Design matrix WITHOUT constant unless ``add_constant=False``.
    bw : float
        Bandwidth. Interpretation depends on ``fixed``:
        - ``fixed=True``: metric distance.
        - ``fixed=False`` (default, "adaptive"): number of nearest
          neighbours. Non-integer values are rounded up to include the
          fractional neighbour, consistent with ``mgwr``.
    kernel : {"bisquare", "gaussian", "exponential"}
        Bisquare is mgwr's default.
    fixed : bool, default False
    add_constant : bool, default True
        Prepend a column of ones to ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> coords = rng.uniform(0, 10, size=(n, 2))
    >>> x = rng.normal(size=n)
    >>> beta = 0.5 + 0.1 * coords[:, 1]  # slope varies across space
    >>> y = 1.0 + beta * x + rng.normal(0, 0.3, n)
    >>> res = sp.gwr(coords, y, x.reshape(-1, 1), bw=30, fixed=False)
    >>> res.params.shape  # one local coefficient vector per observation
    (80, 2)
    >>> bool(0.0 <= res.R2 <= 1.0)
    True
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if add_constant:
        X = np.column_stack([np.ones(X.shape[0]), X])
    n, k = X.shape
    tree = cKDTree(coords)

    params = np.empty((n, k))
    predicted = np.empty(n)
    S_diag = np.empty(n)  # diagonal of the hat matrix
    CCt_diag = np.empty((n, k))  # diag(C_i C_i'), C_i = (X'W_iX)^-1 X'W_i
    tr_StS = 0.0  # tr(S'S) = Σ_i ||s_i||² (effective resid df)

    for i in range(n):
        w = _weights_row(coords, i, bw, kernel, fixed, tree)
        WX = X * w[:, None]
        XtWX = X.T @ WX
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        C_i = XtWX_inv @ (X.T * w)  # (k, n): β̂_i = C_i y
        beta_i = C_i @ y
        params[i] = beta_i
        predicted[i] = float(X[i] @ beta_i)
        CCt_diag[i] = np.sum(C_i**2, axis=1)
        # Hat-row: s_i = x_i' C_i  → S[i, i] = s_i[i]; accumulate tr(S'S)
        s_i = X[i] @ C_i
        S_diag[i] = float(s_i[i])
        tr_StS += float(s_i @ s_i)

    residuals = y - predicted
    resid_ss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    R2 = 1.0 - resid_ss / tss if tss > 0 else np.nan
    tr_S = float(S_diag.sum())

    # Local standard errors (Fotheringham, Brunsdon & Charlton 2002, §2.4):
    #   Var(β̂_i) = σ̂² C_i C_i',  σ̂² = RSS / (n - 2 tr(S) + tr(S'S)).
    delta1 = max(n - 2.0 * tr_S + tr_StS, 1e-6)
    sigma2_se = resid_ss / delta1
    se = np.sqrt(np.maximum(sigma2_se * CCt_diag, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, params / se, np.nan)

    sigma2 = resid_ss / n
    # AIC / AICc per Fotheringham et al. (2002) eq. (2.34)
    aic = 2 * n * np.log(np.sqrt(sigma2)) + n * np.log(2 * np.pi) + n + tr_S
    denom = max(n - tr_S - 2, 1e-6)
    aicc = (
        2 * n * np.log(np.sqrt(sigma2)) + n * np.log(2 * np.pi) + n * (n + tr_S) / denom
    )
    bic = 2 * n * np.log(np.sqrt(sigma2)) + n * np.log(2 * np.pi) + tr_S * np.log(n)

    # Local R²: Leung, Mei & Zhang (2000) definition
    local_R2 = np.empty(n)
    for i in range(n):
        w = _weights_row(coords, i, bw, kernel, fixed, tree)
        ybar_local = float(w @ y) / float(w.sum()) if w.sum() > 0 else 0.0
        tss_local = float((w * (y - ybar_local) ** 2).sum())
        rss_local = float((w * residuals**2).sum())
        local_R2[i] = 1.0 - rss_local / tss_local if tss_local > 0 else np.nan

    return GWRResult(
        params=params,
        predicted=predicted,
        residuals=residuals,
        bw=float(bw),
        kernel=kernel,
        fixed=fixed,
        local_R2=local_R2,
        aicc=float(aicc),
        aic=float(aic),
        bic=float(bic),
        R2=float(R2),
        resid_ss=resid_ss,
        tr_S=tr_S,
        n=n,
        k=k,
        se=se,
        tvals=tvals,
    )
