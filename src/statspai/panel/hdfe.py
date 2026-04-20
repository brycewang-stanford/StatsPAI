"""
High-Dimensional Fixed Effects (HDFE) absorber.

Native Python/NumPy implementation of within-transformation (mean-sweep) for
regressions absorbing many high-cardinality fixed effects. Functionally
mirrors Stata's ``reghdfe`` (Correia 2017) and R's ``fixest::feols``
(Bergé 2018) core routine.

Algorithm
---------
The within-transformation on multiple FE groups ``G_1, ..., G_K`` is the
orthogonal projection onto the complement of the span of K group-indicator
matrices. When K = 1 this reduces to de-meaning; when K >= 2 no closed
form exists and alternating projections (method of alternating
projections, von Neumann 1933) are used:

    repeat:
        for k in 1..K:
            x <- x - mean_{G_k}(x)
    until ||dx|| < tol

This converges linearly. To get industrial speed we use the Irons-Tuck /
Aitken scalar acceleration (Δ²), which cuts iteration count by 3-10x for
the typical panel case. See Correia (2017) §2.3 for details.

Singleton detection
-------------------
Observations whose FE group has only one observation do not contribute to
within-variation and bias degrees-of-freedom. Iterative singleton pruning
(Correia 2015) removes them. We run a single-pass prune; subsequent rounds
only matter for K > 3 in pathological data.

Exported API
------------
``Absorber``: class that holds FE columns + weights, with ``demean`` /
``residualize`` methods.
``demean``: functional convenience wrapper.
``absorb_ols``: solve an OLS with absorbed FEs in one call (returns
coefs, SE, absorbed residuals, FE-adjusted R²).

References
----------
Correia, S. (2015). "Singletons, Cluster-Robust Standard Errors and
Fixed Effects." Working paper.
Correia, S. (2017). "Linear Models with High-Dimensional Fixed Effects."
Working paper. https://scorreia.com/research/hdfe.pdf
Bergé, L. (2018). "Efficient estimation of maximum likelihood models
with multiple fixed-effects: the R package FENmlm." CREA DP 13.
Gaure, S. (2013). "OLS with multiple high dimensional category variables."
Computational Statistics & Data Analysis, 66, 8-18.
Guimarães, P. and Portugal, P. (2010). "A simple feasible procedure to
fit models with high-dimensional fixed effects." Stata Journal, 10(4).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ======================================================================
# Core demean kernel
# ======================================================================


def _factorize(fe: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return integer codes in [0, G) and the group count G.

    Handles both numeric and object/string arrays. ``pd.factorize`` is
    used for speed and NaN-safety.
    """
    codes, uniq = pd.factorize(fe, sort=False, use_na_sentinel=True)
    if (codes < 0).any():
        raise ValueError("HDFE: NaN values in fixed-effect column are not allowed.")
    return codes.astype(np.int64), len(uniq)


def _group_mean_sweep(
    x: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
    weights: Optional[np.ndarray] = None,
    wsum: Optional[np.ndarray] = None,
) -> None:
    """In-place de-mean x by group codes. 2D x supported (column-wise)."""
    if x.ndim == 1:
        if weights is None:
            sums = np.bincount(codes, weights=x, minlength=counts.size)
            means = sums / counts
        else:
            sums = np.bincount(codes, weights=x * weights, minlength=counts.size)
            means = sums / wsum
        x -= means[codes]
    else:
        G = counts.size
        n, p = x.shape
        if weights is None:
            for j in range(p):
                sums = np.bincount(codes, weights=x[:, j], minlength=G)
                x[:, j] -= (sums / counts)[codes]
        else:
            for j in range(p):
                sums = np.bincount(codes, weights=x[:, j] * weights, minlength=G)
                x[:, j] -= (sums / wsum)[codes]


def _map_ap(
    x: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: Optional[np.ndarray],
    wsum_list: Optional[List[np.ndarray]],
) -> None:
    """One full alternating-projection sweep over all FE dimensions (in place)."""
    K = len(fe_codes)
    for k in range(K):
        _group_mean_sweep(
            x,
            fe_codes[k],
            counts_list[k],
            weights=weights,
            wsum=wsum_list[k] if wsum_list is not None else None,
        )


def _aitken_accelerate(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Irons-Tuck / vector Aitken-like Δ² acceleration.

    Given three successive iterates x0, x1, x2 of a contraction map
    ``T``, return the accelerated value

        x_acc = x0 - α · (x1 - x0)

    where ``α = <dx1, d2> / <d2, d2>`` (the Minimum-Polynomial
    Extrapolation / Irons-Tuck scalar, Smith-Ford-Sidi 1987). This is
    numerically safer than element-wise Aitken (no per-element
    division blow-up) and equivalent in the limit. See Varadhan-Roland
    (2008) §3.

    Falls back to x2 if the denominator is near-zero (no acceleration
    signal).
    """
    dx1 = x1 - x0
    d2 = x2 - 2.0 * x1 + x0
    denom = float(d2 @ d2)
    if denom < 1e-30:
        return x2
    alpha = float(dx1 @ d2) / denom
    return x0 - alpha * dx1


# ======================================================================
# Singleton pruning
# ======================================================================


def _detect_singletons(
    fe: np.ndarray,
    fe_codes_raw: List[np.ndarray],
) -> np.ndarray:
    """Iteratively drop singleton observations (groups of size 1).

    Each pass: mark obs whose code in any dimension appears only once; if
    any found, drop them and recount. Repeat until stable. Returns the
    keep mask (True = keep).

    Worst-case quadratic in number of passes but in practice converges in
    <= 3 passes even on nested panels (Correia 2015).
    """
    n = fe.shape[0]
    keep = np.ones(n, dtype=bool)
    K = len(fe_codes_raw)

    while True:
        dropped = False
        for k in range(K):
            codes_k = fe_codes_raw[k][keep]
            counts_k = np.bincount(codes_k)
            single_groups = np.where(counts_k == 1)[0]
            if single_groups.size == 0:
                continue
            single_mask_local = np.isin(codes_k, single_groups)
            if single_mask_local.any():
                # Map back to global indices
                global_idx = np.where(keep)[0]
                keep[global_idx[single_mask_local]] = False
                dropped = True
        if not dropped:
            break
    return keep


# ======================================================================
# Absorber class
# ======================================================================


class Absorber:
    """Reusable HDFE demean operator.

    Build once from a DataFrame's FE columns; reuse ``demean`` to sweep any
    outcome / regressor vector or matrix. Useful when fitting many models
    that share the same absorbing FEs (e.g. event-study coefficient paths).

    Parameters
    ----------
    fe_data : DataFrame or ndarray (n, K)
        FE columns. Must have no NaN.
    weights : ndarray (n,), optional
        Observation weights. If given, weighted means are used.
    drop_singletons : bool, default True
        If True, singleton observations (FE groups of size 1) are pruned
        before building the absorber. ``keep_mask`` stores the surviving
        rows.
    tol : float, default 1e-8
        Convergence threshold on max |dx| per iteration.
    maxiter : int, default 10000
        Maximum alternating-projection iterations.
    accelerate : bool, default True
        Enable Irons-Tuck Δ² acceleration.

    Attributes
    ----------
    keep_mask : ndarray of bool
        Rows retained after singleton pruning. Callers must apply this
        mask to ``y``, ``X``, and any weights before passing to
        ``demean``.
    n_kept : int
        Number of surviving observations.
    n_dropped : int
        Number of singleton observations removed.
    n_fe : list of int
        Number of groups per FE dimension (post-prune).
    """

    __slots__ = (
        "fe_codes",
        "counts_list",
        "wsum_list",
        "weights",
        "keep_mask",
        "n_kept",
        "n_dropped",
        "n_fe",
        "tol",
        "maxiter",
        "accelerate",
        "_converged",
        "_iters",
    )

    def __init__(
        self,
        fe_data: Union[pd.DataFrame, np.ndarray],
        weights: Optional[np.ndarray] = None,
        drop_singletons: bool = True,
        tol: float = 1e-8,
        maxiter: int = 10_000,
        accelerate: bool = True,
    ) -> None:
        if isinstance(fe_data, pd.DataFrame):
            fe_arr = fe_data.values
        else:
            fe_arr = np.asarray(fe_data)
            if fe_arr.ndim == 1:
                fe_arr = fe_arr.reshape(-1, 1)

        n, K = fe_arr.shape
        if K == 0:
            raise ValueError("HDFE: at least one fixed-effect column required.")

        # Factorize each FE column
        fe_codes_raw: List[np.ndarray] = []
        for k in range(K):
            codes_k, _ = _factorize(fe_arr[:, k])
            fe_codes_raw.append(codes_k)

        # Singleton pruning
        if drop_singletons:
            keep_mask = _detect_singletons(fe_arr, fe_codes_raw)
        else:
            keep_mask = np.ones(n, dtype=bool)
        n_kept = int(keep_mask.sum())
        n_dropped = n - n_kept

        # Re-factorize on kept rows so codes are dense in [0, G)
        fe_codes: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        n_fe: List[int] = []
        for codes_k in fe_codes_raw:
            codes_kept = codes_k[keep_mask]
            dense, uniq = pd.factorize(codes_kept, sort=False, use_na_sentinel=True)
            dense = dense.astype(np.int64)
            G = len(uniq)
            counts = np.bincount(dense, minlength=G).astype(np.float64)
            fe_codes.append(dense)
            counts_list.append(counts)
            n_fe.append(G)

        # Weighted group sums, pre-compute wsum per group
        if weights is not None:
            w_kept = np.ascontiguousarray(weights[keep_mask], dtype=np.float64)
            wsum_list: Optional[List[np.ndarray]] = []
            for codes_k, G in zip(fe_codes, n_fe):
                wsum_list.append(np.bincount(codes_k, weights=w_kept, minlength=G))
            self.weights = w_kept
        else:
            wsum_list = None
            self.weights = None

        self.fe_codes = fe_codes
        self.counts_list = counts_list
        self.wsum_list = wsum_list
        self.keep_mask = keep_mask
        self.n_kept = n_kept
        self.n_dropped = n_dropped
        self.n_fe = n_fe
        self.tol = tol
        self.maxiter = maxiter
        self.accelerate = accelerate
        self._converged = False
        self._iters = 0

    # ------------------------------------------------------------------
    # Demean
    # ------------------------------------------------------------------

    def demean(
        self,
        x: np.ndarray,
        copy: bool = True,
        already_masked: bool = False,
    ) -> np.ndarray:
        """Within-transform ``x`` by sweeping out all absorbed FEs.

        Parameters
        ----------
        x : ndarray, shape (n,) or (n, p)
            Variable(s) to residualize. ``n`` must equal either the full
            input size (then ``keep_mask`` is applied) or ``n_kept``
            (when ``already_masked=True``).
        copy : bool, default True
            If True, operate on a copy; if False, modify ``x`` in place.
            Callers passing fresh arrays can set False to save memory.
        already_masked : bool, default False
            Skip application of ``keep_mask``.

        Returns
        -------
        ndarray
            Residualized ``x`` with shape ``(n_kept,)`` or ``(n_kept, p)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if not already_masked:
            x = x[self.keep_mask]
        if copy:
            x = x.copy()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False

        fe_codes = self.fe_codes
        counts_list = self.counts_list
        wsum_list = self.wsum_list
        weights = self.weights
        K = len(fe_codes)

        # K=1: closed-form
        if K == 1:
            _group_mean_sweep(x, fe_codes[0], counts_list[0], weights, wsum_list[0] if wsum_list else None)
            self._converged = True
            self._iters = 1
            return x.ravel() if squeeze else x

        # K>=2: alternating projections with optional Irons-Tuck acceleration
        tol = self.tol
        maxiter = self.maxiter
        accelerate = self.accelerate

        # Per-column AP loop
        for j in range(x.shape[1]):
            col = x[:, j].copy()  # work on a copy to avoid in-place surprises
            base_scale = np.max(np.abs(col)) + 1e-30

            # Standard AP loop with periodic Irons-Tuck acceleration.
            # Every ACCEL_PERIOD sweeps, apply the vector-Aitken jump built
            # from three consecutive iterates (classic SQUAREM layout).
            accel_period = 5
            col_hist: list = []
            converged = False
            for it in range(maxiter):
                col_before = col.copy()
                _group_mean_sweep_seq(col, fe_codes, counts_list, weights, wsum_list)
                dx = np.max(np.abs(col - col_before)) / base_scale
                if dx < tol:
                    converged = True
                    self._iters = max(self._iters, it + 1)
                    break
                if accelerate:
                    col_hist.append(col.copy())
                    if len(col_hist) >= 3 and (it + 1) % accel_period == 0:
                        col_acc = _aitken_accelerate(
                            col_hist[-3], col_hist[-2], col_hist[-1]
                        )
                        # Only accept the jump if it does not blow up
                        if np.max(np.abs(col_acc)) < 10 * base_scale:
                            col = col_acc
                        col_hist = []
            self._converged = self._converged or converged
            if not converged:
                self._iters = maxiter
            x[:, j] = col

        return x.ravel() if squeeze else x

    # ------------------------------------------------------------------
    # Residualize (alias + sanity)
    # ------------------------------------------------------------------

    def residualize(self, x: np.ndarray, copy: bool = True) -> np.ndarray:
        """Alias for ``demean`` — returns FE-residualized version of x."""
        return self.demean(x, copy=copy)

    def __repr__(self) -> str:
        return (
            f"Absorber(K={len(self.fe_codes)}, n_kept={self.n_kept}, "
            f"n_dropped={self.n_dropped}, groups={self.n_fe})"
        )


def _group_mean_sweep_seq(
    col: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: Optional[np.ndarray],
    wsum_list: Optional[List[np.ndarray]],
) -> None:
    """One full sequential sweep over all K dimensions (in place, 1D)."""
    K = len(fe_codes)
    if weights is None:
        for k in range(K):
            codes = fe_codes[k]
            counts = counts_list[k]
            sums = np.bincount(codes, weights=col, minlength=counts.size)
            col -= (sums / counts)[codes]
    else:
        for k in range(K):
            codes = fe_codes[k]
            wsum = wsum_list[k]
            sums = np.bincount(codes, weights=col * weights, minlength=wsum.size)
            col -= (sums / wsum)[codes]


# ======================================================================
# Functional API
# ======================================================================


def demean(
    x: np.ndarray,
    fe: Union[pd.DataFrame, np.ndarray],
    weights: Optional[np.ndarray] = None,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the within-transformed ``x`` and the singleton keep mask.

    Convenience wrapper around :class:`Absorber`.
    """
    ab = Absorber(
        fe, weights=weights, drop_singletons=drop_singletons,
        tol=tol, maxiter=maxiter,
    )
    xw = ab.demean(x)
    return xw, ab.keep_mask


# ======================================================================
# High-level API: absorb_ols
# ======================================================================


def absorb_ols(
    y: np.ndarray,
    X: np.ndarray,
    fe: Union[pd.DataFrame, np.ndarray],
    weights: Optional[np.ndarray] = None,
    cluster: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    return_absorber: bool = False,
) -> dict:
    """OLS with absorbed high-dimensional fixed effects (reghdfe-style).

    Solves ``y = X β + Σ_k α_{g_k} + ε`` by sweeping out the FEs from
    both y and X (Frisch-Waugh-Lovell) and running OLS on residuals.

    Parameters
    ----------
    y : ndarray, shape (n,)
    X : ndarray, shape (n, p)
        Regressors *excluding* the absorbed FEs and the constant (the
        constant is absorbed by any FE dimension).
    fe : DataFrame or ndarray (n, K)
        Fixed-effect columns.
    weights : ndarray (n,), optional
        Observation weights.
    cluster : ndarray or list of ndarrays, optional
        One-way or multi-way cluster variables for robust SEs. If
        provided, returns cluster-robust SEs (one-way: Liang-Zeger
        sandwich; multi-way: inclusion-exclusion Cameron-Gelbach-Miller).
    drop_singletons : bool, default True
    tol, maxiter : float, int
        Demean convergence controls.
    return_absorber : bool, default False
        If True, also return the ``Absorber`` object for reuse.

    Returns
    -------
    dict with keys:
        ``coef`` (p,), ``se`` (p,), ``vcov`` (p,p), ``resid`` (n_kept,),
        ``n`` (n_kept), ``df_resid``, ``dof_fe``, ``r2_within``,
        ``n_singletons_dropped``, ``converged``, ``iters``,
        ``absorber`` (if requested)
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X length mismatch.")

    ab = Absorber(
        fe, weights=weights, drop_singletons=drop_singletons,
        tol=tol, maxiter=maxiter,
    )
    yw = ab.demean(y)
    Xw = ab.demean(X)
    w = ab.weights

    # Weighted OLS on residuals
    if w is None:
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
    else:
        Xw_w = Xw * w[:, None]
        XtX = Xw.T @ Xw_w
        Xty = Xw.T @ (yw * w)
    # Solve (use pinv fallback if near-singular)
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    resid = yw - Xw @ coef

    # DOF: n - p - Σ(G_k - 1) - 1_if_first_fe (the "one lost for mean" already in
    # each FE's G-1; common intercept overlap handled by subtracting (K-1))
    dof_fe = sum(G for G in ab.n_fe) - (len(ab.n_fe) - 1)
    df_resid = ab.n_kept - p - dof_fe
    if df_resid <= 0:
        raise ValueError(
            f"Degrees of freedom exhausted: n_kept={ab.n_kept}, p={p}, dof_fe={dof_fe}. "
            "Reduce regressors or drop a FE dimension."
        )

    # Within R² (FE already swept)
    if w is None:
        ss_res = float((resid ** 2).sum())
        y_demeaned = yw - yw.mean()
        ss_tot = float((y_demeaned ** 2).sum())
    else:
        ss_res = float((resid ** 2 * w).sum())
        y_bar_w = (yw * w).sum() / w.sum()
        ss_tot = float(((yw - y_bar_w) ** 2 * w).sum())
    r2_within = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Variance-covariance matrix
    XtX_inv = np.linalg.inv(XtX)
    if cluster is None:
        # Classical (with DOF adjustment)
        sigma2 = ss_res / df_resid
        vcov = sigma2 * XtX_inv
    else:
        # Clip cluster arrays to the singleton-pruned subsample so shapes
        # match the within-transformed X and residuals.
        keep = ab.keep_mask
        if isinstance(cluster, list):
            cluster_sub = [np.asarray(c)[keep] for c in cluster]
        else:
            cluster_sub = np.asarray(cluster)[keep]
        vcov = _cluster_sandwich(
            Xw, resid, coef, XtX_inv, cluster_sub, df_resid=df_resid,
            weights=w, n_absorbed=dof_fe + p,
        )
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    out = {
        "coef": coef,
        "se": se,
        "vcov": vcov,
        "resid": resid,
        "fitted_within": Xw @ coef,  # within-prediction (excludes FE)
        "n": ab.n_kept,
        "df_resid": df_resid,
        "dof_fe": dof_fe,
        "r2_within": r2_within,
        "n_singletons_dropped": ab.n_dropped,
        "converged": ab._converged,
        "iters": ab._iters,
        "n_fe": ab.n_fe,
    }
    if return_absorber:
        out["absorber"] = ab
    return out


# ======================================================================
# Clustered sandwich (one-way + N-way inclusion-exclusion)
# ======================================================================


def _cluster_sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    coef: np.ndarray,
    XtX_inv: np.ndarray,
    cluster: Union[np.ndarray, List[np.ndarray]],
    df_resid: int,
    weights: Optional[np.ndarray] = None,
    n_absorbed: int = 0,
) -> np.ndarray:
    """Cluster-robust variance (one-way or multi-way, PSD-corrected)."""
    if not isinstance(cluster, list):
        clusters_list = [np.asarray(cluster)]
    else:
        clusters_list = [np.asarray(c) for c in cluster]

    n, k = X.shape
    scores = X * resid[:, None] if weights is None else X * (resid * weights)[:, None]

    def _one_way(c: np.ndarray) -> np.ndarray:
        codes, _ = _factorize(c)
        G = int(codes.max()) + 1
        # Aggregate scores by cluster
        agg = np.zeros((G, k))
        np.add.at(agg, codes, scores)
        meat = agg.T @ agg
        scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - n_absorbed, 1))
        return scale * XtX_inv @ meat @ XtX_inv

    if len(clusters_list) == 1:
        V = _one_way(clusters_list[0])
    else:
        # N-way CGM via inclusion-exclusion over all non-empty subsets.
        from itertools import combinations
        V = np.zeros((k, k))
        M = len(clusters_list)
        for r in range(1, M + 1):
            for combo in combinations(range(M), r):
                # intersection cluster: tuple of labels
                inter = np.stack([clusters_list[i] for i in combo], axis=1)
                inter_codes, _ = _factorize(
                    pd.DataFrame(inter).astype(str).agg("\0".join, axis=1).values
                )
                V += ((-1) ** (r + 1)) * _one_way(inter_codes)
        # PSD correction
        eigvals, eigvecs = np.linalg.eigh(V)
        if (eigvals < 0).any():
            eigvals = np.maximum(eigvals, 0.0)
            V = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return V


__all__ = [
    "Absorber",
    "demean",
    "absorb_ols",
]
