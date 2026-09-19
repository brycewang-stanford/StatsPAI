"""Multiscale GWR (Fotheringham, Yang & Kang 2017).

Each regression coefficient is allowed its own bandwidth. Fitted by
back-fitting: cycle over covariates, regressing partial residuals on each
covariate via a GWR with a covariate-specific bandwidth selected to
minimise AICc, until convergence.

This is a first-version implementation; it tracks mgwr's algorithm but
uses the same golden-section AICc selector as our standard GWR. For
tight numerical parity with the mgwr package on the Georgia benchmark
you may want to compare against mgwr.MGWR directly after calling this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from ..._result_serialize import ResultProtocolMixin
from .bandwidth import gwr_bandwidth
from .gwr import KernelName, gwr


@dataclass
class MGWRResult(ResultProtocolMixin):
    params: np.ndarray  # (n, k)
    predicted: np.ndarray  # (n,)
    residuals: np.ndarray  # (n,)
    bws: List[float]  # one per coefficient
    kernel: KernelName
    fixed: bool
    R2: float
    resid_ss: float
    n: int
    k: int
    n_iter: int

    def summary(self) -> str:
        lines = [
            "Multiscale Geographically Weighted Regression (MGWR)",
            "-" * 52,
            f"n            : {self.n}",
            f"k            : {self.k}",
            f"Iterations   : {self.n_iter}",
            f"R²           : {self.R2:.4f}",
            f"Residual SS  : {self.resid_ss:.3f}",
            "",
            "Coefficient-specific bandwidths:",
        ]
        for j, bw in enumerate(self.bws):
            lines.append(f"  β{j}  bw = {bw:.3f}")
        lines += ["", "Local coefficient summary:"]
        for j in range(self.k):
            col = self.params[:, j]
            lines.append(f"  β{j}  mean={col.mean(): .4f}  std={col.std(): .4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def mgwr(
    coords: Any,
    y: Any,
    X: Any,
    kernel: KernelName = "bisquare",
    fixed: bool = False,
    add_constant: bool = True,
    max_iter: int = 200,
    tol: float = 1e-5,
    bw_init: Optional[float] = None,
    bws: Optional[Sequence[float]] = None,
) -> MGWRResult:
    """Multiscale GWR via back-fitting.

    Algorithm
    ---------
    1. Initialise with standard GWR at bandwidth selected via AICc.
    2. For each iteration:
       a. For each covariate ``j``:
          - Form partial residuals ``ε_j = y - Σ_{m≠j} f_m(x_m)``.
          - Fit a univariate GWR of ``ε_j`` on ``x_j`` with AICc bandwidth.
          - Replace ``f_j``.
       b. Terminate when max |Δf_j| < tol.

    Parameters
    ----------
    coords : (n, 2) array-like
    y : (n,) array-like
    X : (n, p) array-like
        Regressors, without a constant unless ``add_constant=False``.
    kernel, fixed, add_constant
        As in :func:`gwr`.
    max_iter : int, default 200
        Maximum number of back-fitting sweeps.
    tol : float, default 1e-5
        Convergence threshold on ``max_ij |f_j(i)_new - f_j(i)_old|``.
    bw_init : float, optional
        Bandwidth of the initialising GWR (selected by AICc if omitted).
    bws : sequence of float, optional
        Fixed covariate-specific bandwidths (length ``k``, intercept first
        when ``add_constant=True``). When given, no bandwidth is searched:
        the back-fitting solves the additive model at these bandwidths,
        which is ``GWmodel::gwr.multiscale(bws0 = bws, bw.seled = TRUE)``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> coords = rng.uniform(0, 10, size=(n, 2))
    >>> x = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x + 0.1 * coords[:, 0] + rng.normal(0, 0.3, size=n)
    >>> res = sp.mgwr(coords, y, x.reshape(-1, 1))
    >>> res.params.shape  # (n, intercept + 1 covariate)
    (60, 2)
    >>> bool(np.isfinite(res.R2))
    True
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if add_constant:
        X = np.column_stack([np.ones(X.shape[0]), X])
    n, k = X.shape

    # Initial standard GWR for a decent warm start
    if bw_init is None:
        bw_init = gwr_bandwidth(
            coords,
            y,
            X[:, 1:] if add_constant else X,
            kernel=kernel,
            fixed=fixed,
            add_constant=add_constant,
        )
    init = gwr(
        coords,
        y,
        X[:, 1:] if add_constant else X,
        bw_init,
        kernel=kernel,
        fixed=fixed,
        add_constant=add_constant,
    )

    # Partial predictors f_j(x_j) = β_j(i) * x_{ij}
    f = init.params * X  # shape (n, k)
    params = init.params.copy()
    if bws is not None:
        fixed_bws = [float(b) for b in bws]
        if len(fixed_bws) != k:
            raise ValueError(f"bws must have length {k}; got {len(fixed_bws)}")
    else:
        fixed_bws = None
    bw_list: List[float] = list(fixed_bws) if fixed_bws else [bw_init] * k

    for iteration in range(max_iter):
        max_delta = 0.0
        for j in range(k):
            partial = y - (f.sum(axis=1) - f[:, j])
            xj = X[:, j : j + 1]
            # Select bw for this covariate alone (unless fixed by the caller)
            if fixed_bws is not None:
                bw_j = fixed_bws[j]
            else:
                bw_j = gwr_bandwidth(
                    coords,
                    partial,
                    xj,
                    kernel=kernel,
                    fixed=fixed,
                    add_constant=False,
                )
            res_j = gwr(
                coords,
                partial,
                xj,
                bw_j,
                kernel=kernel,
                fixed=fixed,
                add_constant=False,
            )
            params[:, j] = res_j.params.ravel()
            new_f_j = params[:, j] * X[:, j]
            delta = float(np.abs(new_f_j - f[:, j]).max())
            max_delta = max(max_delta, delta)
            f[:, j] = new_f_j
            bw_list[j] = float(bw_j)
        if max_delta < tol:
            break
    else:
        import warnings

        warnings.warn(
            f"MGWR back-fitting did not converge in {max_iter} iterations "
            f"(last max |delta f| = {max_delta:.3g} >= tol = {tol:g}).",
            RuntimeWarning,
            stacklevel=2,
        )

    predicted = f.sum(axis=1)
    residuals = y - predicted
    resid_ss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    R2 = 1.0 - resid_ss / tss if tss > 0 else np.nan

    return MGWRResult(
        params=params,
        predicted=predicted,
        residuals=residuals,
        bws=bw_list,
        kernel=kernel,
        fixed=fixed,
        R2=float(R2),
        resid_ss=resid_ss,
        n=n,
        k=k,
        n_iter=iteration + 1,
    )
