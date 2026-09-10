"""
Shared low-level primitives for the RD module.

Centralizes kernel functions, kernel-specific constants, and the
canonical weighted-least-squares local polynomial regression used
across rdrobust/rd2d. Keeping them in one place guarantees numerical
consistency and removes a class of drift bugs.

Public surface (module-internal, underscore-prefixed to stay private
to statspai.rd):

    _kernel_fn(u, kernel)           -> kernel weights K(u)
    _kernel_constants(kernel)       -> dict with C_K, mu_2, nu_0
    _kernel_mse_constant(kernel)    -> float C_K (local-linear MSE constant)
    _local_poly_wls(y, x, h, p, kernel, cluster=None, covs=None)
        -> (beta, vcov, n_eff) with HC1 or cluster-robust variance;
           optional additive covariate augmentation; returned beta/vcov
           correspond to the polynomial part only (first p+1 entries).
    _complete_cases(frame, columns)  -> (cleaned, n_dropped)
        drops rows with a non-finite value in any named column, so the
        subpackage's entry points cannot disagree about what a missing
        outcome means.
    _check_covariate_rank(x_centered, covs, p, names=, where=)
        raises when the covariate-augmented local design is numerically
        rank deficient, i.e. when a covariate is collinear with the
        polynomial basis the estimator already fits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy import stats as _sp_stats

_KERNEL_TABLE = {
    "triangular": {"C_K": 3.4375, "mu_2": 1 / 6, "nu_0": 2 / 3},
    "epanechnikov": {"C_K": 3.0, "mu_2": 1 / 5, "nu_0": 3 / 5},
    "uniform": {"C_K": 2.7, "mu_2": 1 / 3, "nu_0": 1 / 2},
}


def _kernel_fn(u: np.ndarray, kernel: str) -> np.ndarray:
    """
    Kernel K(u). Triangular / epanechnikov / uniform have compact support
    on |u| <= 1; gaussian is the standard normal pdf (full support).
    """
    u = np.asarray(u, dtype=float)
    if kernel == "triangular":
        return cast(np.ndarray, np.maximum(1 - np.abs(u), 0))
    elif kernel == "uniform":
        return cast(np.ndarray, 0.5 * (np.abs(u) <= 1).astype(float))
    elif kernel == "epanechnikov":
        return cast(np.ndarray, 0.75 * np.maximum(1 - u**2, 0))
    elif kernel == "gaussian":
        return cast(np.ndarray, _sp_stats.norm.pdf(u))
    raise ValueError(f"Unknown kernel: {kernel}")  # pragma: no cover


def _kernel_constants(kernel: str) -> dict:
    """
    Return kernel-specific constants for bandwidth selection.

    C_K  : MSE-optimal bandwidth constant for local linear (p=1).
    mu_2 : second moment of the kernel, int u^2 K(u) du.
    nu_0 : int K(u)^2 du (roughness).
    """
    return _KERNEL_TABLE[kernel]


def _kernel_mse_constant(kernel: str) -> float:
    """C_{1,1}: MSE-optimal bandwidth constant for local linear."""
    return _KERNEL_TABLE.get(kernel, _KERNEL_TABLE["triangular"])["C_K"]


def _sandwich_variance(
    Xw: np.ndarray,
    yw: np.ndarray,
    beta: np.ndarray,
    resid: np.ndarray,
    n_eff: int,
    k: int,
    cluster_in_bw: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    HC1 or cluster-robust sandwich variance for a WLS fit.

    Parameters
    ----------
    Xw : (n, k) array
        Square-root-weighted design matrix, i.e. X * sqrt(w).
    yw : (n,) array
        Square-root-weighted response, i.e. y * sqrt(w).
    beta : (k,) array
        Estimated coefficients from the weighted LS.
    resid : (n,) array
        Raw (unweighted) residuals y - X @ beta.
    n_eff : int
        Effective sample size (observations inside the bandwidth).
    k : int
        Number of columns in the design matrix.
    cluster_in_bw : (n,) array or None
        Cluster identifiers for observations inside the bandwidth.
        When None, the heteroskedasticity-robust (HC) meat is used.
    weights : (n,) array or None
        Raw in-bandwidth kernel weights ``w`` (NOT their square root). The
        Calonico-Cattaneo-Titiunik (2014) HC meat is ``X' W diag(e^2) W X``,
        which carries the kernel weight *squared*; since ``Xw = X * sqrt(w)``
        only supplies one power, ``weights`` provides the missing factor. If
        None, ``w`` is recovered from the weighted intercept column
        ``Xw[:, 0] = sqrt(w)``. Ignored for the cluster-robust meat (whose
        per-observation score ``Xw' (yw - Xw beta)`` already equals
        ``w x e``).
    """
    XtWX = Xw.T @ Xw
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    if cluster_in_bw is not None:
        unique_cl = np.unique(cluster_in_bw)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cluster_in_bw == c_val
            score = (Xw[idx].T @ (yw[idx] - Xw[idx] @ beta)).ravel()
            meat += np.outer(score, score)
        corr = n_cl / (n_cl - 1) if n_cl > 1 else 1.0
        return cast(np.ndarray, corr * bread @ meat @ bread)
    else:
        corr = n_eff / (n_eff - k) if n_eff > k else 1.0
        # CCT (2014) heteroskedasticity-robust local-polynomial meat:
        #   X' W diag(e^2) W X = sum_i w_i^2 x_i x_i' e_i^2.
        # Xw = X * sqrt(w) carries only ONE power of the kernel weight on each
        # side (w_i total), so we multiply by an extra w_i to reach w_i^2.
        # Omitting this factor (the historical bug) inflated every HC-robust RD
        # standard error -- ~1.4x for a uniform kernel versus R rdrobust
        # vce='hc0'. The cluster branch above is unaffected (its score already
        # carries w_i x_i e_i).
        if weights is not None:
            w_vec = np.asarray(weights, dtype=float)
        else:
            w_vec = Xw[:, 0] ** 2  # Xw[:, 0] = sqrt(w) (weighted intercept)
        meat = Xw.T @ np.diag(w_vec * resid**2 * corr) @ Xw
        return cast(np.ndarray, bread @ meat @ bread)


def _local_poly_wls(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    p: int,
    kernel: str,
    cluster: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    WLS local polynomial regression evaluated at x = 0.

    When covs is provided, the design matrix is augmented:
    [1, x, x^2, ..., x^p, z1, z2, ..., zk]
    The treatment effect is still beta[0] (intercept) for deriv=0
    or beta[1] (slope) for deriv=1, etc. Covariates enter additively.

    Returns (beta, vcov, n_effective).
    The returned beta and vcov correspond to the polynomial part only
    (first p+1 elements), with covariate effects absorbed.
    """
    u = x / h
    w = _kernel_fn(u, kernel)
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())

    k_poly = p + 1

    if n_eff < k_poly + 2:
        return np.zeros(k_poly), np.eye(k_poly) * 1e10, 0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = w[in_bw]

    # Design matrix [1, x, x^2, ..., x^p]
    X_poly = np.column_stack([x_bw**j for j in range(k_poly)])

    # Augment with covariates if provided
    if covs is not None:
        Z_bw = covs[in_bw]
        X = np.column_stack([X_poly, Z_bw])
    else:
        X = X_poly

    k_total = X.shape[1]

    # WLS via square-root weights
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta_full = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - X @ beta_full

    cl_in_bw = cluster[in_bw] if cluster is not None else None
    vcov_full = _sandwich_variance(
        Xw, yw, beta_full, resid, n_eff, k_total, cl_in_bw, weights=w_bw
    )

    # Return only the polynomial part (first k_poly elements)
    beta = beta_full[:k_poly]
    vcov = vcov_full[:k_poly, :k_poly]

    return beta, vcov, n_eff


def _complete_cases(
    frame,
    columns: Sequence[Union[str, Sequence[str], None]],
) -> Tuple[object, int]:
    """Drop rows with a non-finite value in any of ``columns``.

    Returns ``(cleaned, n_dropped)``.

    ``columns`` accepts ``None`` entries and nested sequences so callers can
    pass optional arguments through unchanged::

        _complete_cases(df_w, [y, x, covs, fuzzy])

    Why this is shared rather than inlined: the local-randomization entry
    points (``rdrandinf``, ``rdwinselect``, ``rdsensitivity``, ``rdrbounds``)
    had no missing-data handling at all, while ``rdrobust`` in the same
    subpackage has always dropped incomplete rows. On the Lee 2008 senate
    replica -- which carries 93 missing outcomes -- that difference made
    ``sp.rdrandinf`` return a NaN difference in means and, because every
    comparison against a NaN is False, a permutation p-value of exactly
    0.000. A failed statistic was being reported as the most significant
    result the test can produce. Two entry points in one subpackage must not
    disagree about what a missing outcome means.

    Note that ``np.isfinite`` is deliberately stricter than ``notna``: an
    infinite outcome breaks a difference in means exactly as a missing one
    does, and silently propagates just as far.
    """
    import numpy as _np
    import pandas as _pd

    flat: list[str] = []
    for col in columns:
        if col is None:
            continue
        if isinstance(col, str):
            flat.append(col)
        else:
            flat.extend(c for c in col if c is not None)

    if not flat:
        return frame, 0

    keep = _np.ones(len(frame), dtype=bool)
    for name in flat:
        values = _pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        keep &= _np.isfinite(values)

    n_dropped = int((~keep).sum())
    return (frame if n_dropped == 0 else frame.loc[keep].copy()), n_dropped


#: Relative singular-value cutoff below which a design is treated as rank
#: deficient. This is ``sqrt(.Machine$double.eps)``, the same threshold
#: ``MASS::ginv`` uses and therefore the one ``rdrobust`` inherits when its
#: Cholesky fails and it falls back to a pseudo-inverse.
_RANK_TOL = float(np.sqrt(np.finfo(float).eps))


def _check_covariate_rank(
    x_centered: "np.ndarray",
    covs: "Optional[np.ndarray]",
    p: int,
    *,
    names: "Optional[Sequence[str]]" = None,
    where: str = "rd",
) -> None:
    """Warn when covariates are collinear with the local polynomial basis.

    An RD covariate that is a smooth function of the running variable is
    absorbed by the polynomial terms the estimator already fits, so the
    augmented design ``[1, x, ..., x^p, Z]`` loses rank and the estimate
    stops being identified. Nothing about that is visible in the output.

    NumPy's Cholesky is the reason it stays invisible here: on the exactly
    collinear case it *succeeds* where R's refuses, so the pseudo-inverse
    fallback both implementations carry never fires on our side and a
    numerically meaningless solve is returned as an ordinary answer. The
    smallest relative singular value in that case is ~4e-15, seven orders
    below the cutoff, and the returned bandwidth sits 3.8e-3 away from
    ``rdrobust``'s. Two implementations disagreeing is the visible symptom;
    the defect is that neither the user nor the caller was told the design
    was singular.

    The check is global rather than per-window: the bandwidth is not known
    until after selection runs, so a window-local test would have to be
    deferred past the point where the damage is done. A covariate collinear
    on the whole support is collinear in every window, which is the case
    that matters.
    """
    if covs is None:
        return
    Z = np.asarray(covs, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.size == 0:
        return

    basis = np.column_stack(
        [np.ones(len(x_centered))] + [x_centered**k for k in range(1, p + 1)]
    )
    design = np.column_stack([basis, Z])
    ok = np.isfinite(design).all(axis=1)
    design = design[ok]
    if design.shape[0] <= design.shape[1]:
        return

    # Scale columns to unit norm so the ratio measures collinearity rather
    # than the units the covariates happen to be recorded in.
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1.0
    sv = np.linalg.svd(design / norms, compute_uv=False)
    if sv[0] <= 0:
        return
    rel = sv[-1] / sv[0]
    if rel >= _RANK_TOL:
        return

    label = (
        f" ({', '.join(str(n) for n in names)})"
        if names is not None and len(list(names)) == Z.shape[1]
        else ""
    )
    # Raise rather than warn. The estimate is not identified, so the number
    # that would come back is whichever generalised inverse the linear
    # algebra happened to pick -- and downstream it does not even survive
    # intact: on this design `sp.rdrobust` goes on to report a NaN standard
    # error, and at other covariate scalings the solve raises a bare
    # `LinAlgError: Singular matrix` with nothing to tell the caller which
    # covariate caused it. Warning and continuing would leave all three
    # outcomes in play and none of them meaningful. R refuses this design
    # too under `covs_drop=FALSE`.
    raise ValueError(
        f"{where}: the covariate-augmented local design is numerically rank "
        f"deficient (smallest relative singular value {rel:.2e}, below the "
        f"{_RANK_TOL:.2e} cutoff). At least one covariate{label} is collinear "
        f"with the polynomial basis in the running variable, which the "
        f"estimator already fits, so the covariate adjustment is not "
        f"identified. Drop the offending covariate, or lower p if the "
        f"collinearity comes from a high-order term."
    )
