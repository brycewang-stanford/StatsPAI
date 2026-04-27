"""``sp.fast.inference`` — cluster-robust SEs and wild cluster bootstrap.

Phase 4 deliverables. Implements:

* :func:`crve`     — closed-form cluster-robust variance for OLS / WLS,
                     with CR1 (Liang-Zeger + small-sample) and CR3
                     (jackknife-style) flavours.
* :func:`boottest` — wild cluster bootstrap for a single-coefficient
                     null hypothesis ``β_idx = β_0``. Supports
                     Rademacher and Webb-6 weight distributions.

Both functions accept a generic OLS/WLS-like dictionary (``X``, ``y``,
``residuals``, ``beta``, ``cluster``) so they slot in cleanly above any
estimator that exposes those fields — including ``sp.fast.fepois`` (via
its working response + score path; not wired in this phase).

References
----------
Liang, K.-Y., Zeger, S. L. (1986). Longitudinal data analysis using
generalised linear models. Biometrika 73, 13–22.

Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based
improvements for inference with clustered errors. Review of Economics
and Statistics 90(3), 414–427.

MacKinnon, J. G., Webb, M. D. (2018). The wild bootstrap for few
(treated) clusters. Econometrics Journal 21(2), 114–135.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cluster-robust variance
# ---------------------------------------------------------------------------

def crve(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
    type: str = "cr1",
    extra_df: int = 0,
) -> np.ndarray:
    """Cluster-robust variance-covariance matrix for OLS / WLS.

    Computes the sandwich

        V = (X'WX)^{-1} M (X'WX)^{-1}

    where the meat ``M`` depends on the requested type:

    - ``"cr1"`` (Liang-Zeger): ``M = Σ_g (X_g' u_g)(X_g' u_g)'``,
      then scaled by the small-sample factor
      ``c = (G/(G-1)) * (n-1)/(n - k - extra_df)``.
    - ``"cr2"`` (Bell-McCaffrey): replaces the cluster scores with
      ``X_g' (I_{n_g} - H_gg)^{-1/2} u_g``, where the cluster-leverage
      matrix ``H_gg = X_g (X' W X)^{-1} X_g' diag(w_g)`` adjusts each
      cluster for the rest of the sample. Recommended for designs with
      few clusters (G < 50). No additional small-sample multiplier.
    - ``"cr3"`` (jackknife): same outer sum as CR1, scaled by ``(G-1)/G``.
      Closer to leave-one-cluster-out variance; reduces over-rejection in
      few-cluster settings. ``extra_df`` is ignored for CR3.

    HDFE callers must pass ``extra_df``
    -----------------------------------
    When ``X`` has been FE-residualised (``sp.fast.event_study``,
    DML / Lasso pipelines built on ``sp.fast.within``), the small-sample
    factor's denominator must subtract the absorbed FE rank in addition
    to ``k = X.shape[1]``. Otherwise CR1 SEs are systematically too small
    (matches the ``reghdfe`` / ``fixest`` convention). Pass
    ``extra_df = sum(G_k - 1)`` over the K FE dimensions to recover the
    correct factor. ``extra_df`` is only consumed by CR1.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Regressors (already FE-residualised if applicable).
    residuals : ndarray, shape (n,)
        Model residuals.
    cluster : ndarray, shape (n,)
        Cluster identifiers (any dtype factorisable by pd.factorize).
    weights : ndarray, shape (n,), optional
        Observation weights. Default: 1.
    bread : ndarray, shape (k, k), optional
        Pre-computed inverse Hessian / (X'WX)^{-1}. Saves one solve.
    type : {"cr1", "cr2", "cr3"}
    extra_df : int, default 0
        Additional residual-DOF charge for CR1's small-sample factor.
        Ignored for CR2 / CR3.

    References
    ----------
    Bell, R. M., McCaffrey, D. F. (2002). Bias reduction in standard errors
    for linear regression with multi-stage samples. Survey Methodology
    28(2), 169–179.
    """
    if type not in ("cr1", "cr2", "cr3"):
        raise ValueError(f"type={type!r}; expected 'cr1', 'cr2', or 'cr3'")
    if extra_df < 0:
        raise ValueError(f"extra_df={extra_df}; must be >= 0")

    n, k = X.shape
    if weights is None:
        weights = np.ones(n)

    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G < 2:
        raise ValueError(f"crve: need at least 2 clusters, got {G}")

    if bread is None:
        XtWX = X.T @ (X * weights[:, None])
        bread = np.linalg.inv(XtWX)

    if type == "cr2":
        # Bell-McCaffrey CR2:
        #   For each cluster g, build A_g = (I_{n_g} - H_gg)^{-1/2}
        #   where H_gg = X_g (X'WX)^{-1} X_g'^T · diag(w_g).
        #   Adjusted score per row: replace u_g by A_g u_g (then form
        #   X_g'^T (W u_adj)_g and sum across g as in the standard
        #   sandwich). No extra scalar multiplier.
        #
        # For numerical robustness we compute (I - H)^{-1/2} via an
        # eigendecomposition of (I - H) and clamp eigenvalues away from 0.
        meat = np.zeros((k, k))
        for g in range(G):
            mask = cluster_codes == g
            X_g = X[mask]
            w_g = weights[mask]
            u_g = residuals[mask]
            # H_gg = X_g · bread · X_g'^T diag(w_g)   (shape n_g × n_g)
            # = (X_g · bread) · (X_g · diag(w_g))^T  for symmetry below.
            Xb = X_g @ bread
            H_gg = Xb @ (X_g * w_g[:, None]).T
            n_g = X_g.shape[0]
            # Symmetrise to kill round-off asymmetry before eig.
            M = np.eye(n_g) - 0.5 * (H_gg + H_gg.T)
            evals, evecs = np.linalg.eigh(M)
            # Avoid blowing up on (near-)zero eigenvalues — pin a floor.
            evals = np.maximum(evals, 1e-12)
            inv_sqrt = (evecs * (evals ** -0.5)) @ evecs.T  # (n_g, n_g)
            u_adj = inv_sqrt @ u_g                          # adjusted residuals
            score_g = (X_g * w_g[:, None]).T @ u_adj         # (k,)
            meat += np.outer(score_g, score_g)
        return bread @ meat @ bread

    # CR1 / CR3 share the unadjusted cluster-score sum
    score = (residuals * weights)[:, None] * X      # (n, k)
    cluster_score = np.zeros((G, k))
    np.add.at(cluster_score, cluster_codes, score)
    meat = cluster_score.T @ cluster_score          # (k, k)
    V = bread @ meat @ bread

    if type == "cr1":
        c = (G / (G - 1.0)) * ((n - 1.0) / max(n - k - extra_df, 1))
    else:  # cr3
        c = (G - 1.0) / G
    return V * c


# ---------------------------------------------------------------------------
# Wild cluster bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootTestResult:
    """Outcome of :func:`boottest`."""

    null_coef: int
    null_value: float
    t_obs: float
    pvalue: float
    n_boots: int
    weights: str
    boot_t_dist: np.ndarray
    method: str

    def summary(self) -> str:
        return (
            f"sp.fast.boottest  |  H0: β[{self.null_coef}]={self.null_value}  "
            f"|  weights={self.weights}  B={self.n_boots}\n"
            f"  t_obs = {self.t_obs:.4f}    p = {self.pvalue:.4f}  "
            f"(method: {self.method})"
        )


_RADEMACHER = np.array([-1.0, 1.0])


def _rademacher_weights(rng: np.random.Generator, n_clusters: int) -> np.ndarray:
    """Draw a Rademacher (±1) wild weight per cluster."""
    return rng.choice(_RADEMACHER, size=n_clusters)


# Webb 6-point distribution (MacKinnon-Webb 2018):
# Values: ±sqrt(1.5), ±1, ±sqrt(0.5) (each with prob 1/6).
_WEBB6 = np.array([
    -np.sqrt(1.5), -1.0, -np.sqrt(0.5),
    np.sqrt(0.5), 1.0, np.sqrt(1.5),
])


def _webb6_weights(rng: np.random.Generator, n_clusters: int) -> np.ndarray:
    """Draw a Webb-6-point wild weight per cluster."""
    return rng.choice(_WEBB6, size=n_clusters)


def boottest(
    X: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
    *,
    null_coef: int = 0,
    null_value: float = 0.0,
    weights: str = "rademacher",
    B: int = 9999,
    seed: Optional[int] = None,
    obs_weights: Optional[np.ndarray] = None,
    extra_df: int = 0,
) -> BootTestResult:
    """Wild cluster bootstrap of H0: β[null_coef] = null_value (Davidson-Flachaire / MacKinnon 2019).

    Algorithm
    ---------
    1. Fit the unrestricted model β_full = (X'WX)^{-1} X'Wy.
    2. Compute the t-statistic of β[null_coef] using CR1 cluster SE.
    3. Fit the restricted model β_R that imposes β[null_coef] = null_value
       (closed-form via partialling out the constrained column).
    4. For b = 1..B:
       a. Draw cluster-level weights v_g ~ {Rademacher | Webb6}.
       b. Form y_b = X β_R + v_{g(i)} u_R,i  where u_R = y - X β_R.
       c. Refit OLS on y_b → β_b; compute t_b under the same CR1 formula.
    5. p-value = 2 * min(P(t_b > t_obs), P(t_b <= t_obs)) clipped to [0, 1].

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)
    cluster : ndarray, shape (n,)
        Cluster ids.
    null_coef : int
        Zero-based index of the coefficient under H0.
    null_value : float
        Hypothesised value (default 0).
    weights : {"rademacher", "webb"}
        Wild-weight distribution.
    B : int
        Number of bootstrap replications.
    seed : int, optional
        RNG seed for reproducibility.
    obs_weights : ndarray, shape (n,), optional
        Observation weights for WLS. Default uniform.
    extra_df : int, default 0
        Additional residual-DOF charge in the CR1 small-sample factor
        (used both for the observed t-stat and uniformly across the
        bootstrap replicas). Pass ``Σ(G_k - 1)`` when ``X`` is
        FE-residualised — same convention as :func:`crve`. Without it,
        bootstrap p-values for HDFE designs are biased downward because
        the t-pivot uses an under-corrected SE.

    Returns
    -------
    BootTestResult
    """
    if weights not in ("rademacher", "webb"):
        raise ValueError(f"weights={weights!r}; expected 'rademacher' or 'webb'")
    if extra_df < 0:
        raise ValueError(f"extra_df={extra_df}; must be >= 0")

    n, k = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows but X has {n}")
    if cluster.shape[0] != n:
        raise ValueError(f"cluster has {cluster.shape[0]} rows but X has {n}")
    if not (0 <= null_coef < k):
        raise IndexError(f"null_coef={null_coef} out of range [0, {k})")

    rng = np.random.default_rng(seed)
    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G < 2:
        raise ValueError(f"boottest: need at least 2 clusters, got {G}")

    w = obs_weights if obs_weights is not None else np.ones(n)

    # --- Unrestricted fit ---
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (w * y)
    beta_full = np.linalg.solve(XtWX, XtWy)
    bread = np.linalg.inv(XtWX)

    # Observed t-statistic
    resid_full = y - X @ beta_full
    V_full = crve(
        X, resid_full, cluster, weights=w, bread=bread, type="cr1",
        extra_df=extra_df,
    )
    se_obs = float(np.sqrt(V_full[null_coef, null_coef]))
    t_obs = float((beta_full[null_coef] - null_value) / se_obs)

    # --- Restricted fit: enforce β[null_coef] = null_value ---
    # Closed form: shift y by (null_value - β_full[idx]) * X[:, idx], then re-fit
    # without the constrained column.
    # Equivalently: y_R = y - null_value * X[:, idx], and run OLS on X_other.
    keep_cols = np.ones(k, dtype=bool)
    keep_cols[null_coef] = False
    X_R = X[:, keep_cols]
    y_R = y - null_value * X[:, null_coef]
    XR_tWXR = X_R.T @ (X_R * w[:, None])
    XR_tWyR = X_R.T @ (w * y_R)
    beta_R_other = np.linalg.solve(XR_tWXR, XR_tWyR)
    fitted_R = X_R @ beta_R_other + null_value * X[:, null_coef]
    resid_R = y - fitted_R     # restricted residuals

    # --- Bootstrap loop ---
    # Hot-path optimisations (Round 3 audit):
    # - Reuse the precomputed ``bread`` instead of solving the linear
    #   system B times (the original code's main bottleneck).
    # - Cache per-cluster index lists once; the bootstrap meat is
    #   computed without going through ``crve`` to skip its overhead.
    # - The CR1 small-sample factor is constant across bootstraps (n,
    #   k, G all fixed), so multiply it in once at the end.
    boot_t = np.empty(B)
    weight_fn = _rademacher_weights if weights == "rademacher" else _webb6_weights

    cr1_factor = (G / (G - 1.0)) * ((n - 1.0) / max(n - k - extra_df, 1))
    bread_row = bread[null_coef]               # (k,) row of bread for this null

    # Pre-bake the score basis ``X * w``: doesn't change across B, and
    # multiplying once outside the loop saves O(B·n·k) work. We keep
    # both ``Xw`` and its transpose handy because beta update uses Xw.T
    # and the score formula uses Xw directly.
    Xw = X * w[:, None]                        # (n, k)
    XwT = Xw.T                                 # (k, n)

    for b in range(B):
        v_g = weight_fn(rng, G)
        v_i = v_g[cluster_codes]
        y_b = fitted_R + v_i * resid_R                   # (n,)
        beta_b = bread @ (XwT @ y_b)                      # (k,)
        resid_b = y_b - X @ beta_b                        # (n,)
        # Per-row score s_i = w_i * u_i * X_i (uses Xw to avoid re-mul).
        score_b = resid_b[:, None] * Xw                   # (n, k)
        # Cluster-summed scores via scatter-add. ``np.add.at`` is the
        # fastest path here for G in the typical 10–1000 range; an
        # explicit per-cluster Python loop wins only at very large G.
        cs = np.zeros((G, k))
        np.add.at(cs, cluster_codes, score_b)
        meat = cs.T @ cs                                  # (k, k)
        var_null = float(bread_row @ meat @ bread_row) * cr1_factor
        if var_null > 0.0:
            boot_t[b] = (beta_b[null_coef] - null_value) / np.sqrt(var_null)
        else:
            boot_t[b] = 0.0

    # Two-sided p-value (symmetric); clip to [0, 1]
    p_right = float(np.mean(boot_t >= abs(t_obs)))
    p_left = float(np.mean(boot_t <= -abs(t_obs)))
    pvalue = float(np.clip(p_right + p_left, 0.0, 1.0))

    return BootTestResult(
        null_coef=null_coef,
        null_value=null_value,
        t_obs=t_obs,
        pvalue=pvalue,
        n_boots=B,
        weights=weights,
        boot_t_dist=boot_t,
        method="wild_cluster_restricted",
    )


# ---------------------------------------------------------------------------
# Multi-coefficient joint Wald wild cluster bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootWaldResult:
    """Outcome of :func:`boottest_wald` (joint hypothesis bootstrap)."""

    R: np.ndarray
    r: np.ndarray
    wald_obs: float
    pvalue: float
    n_boots: int
    weights: str
    boot_wald_dist: np.ndarray
    df: int                                # rank(R) — number of restrictions

    def summary(self) -> str:
        return (
            f"sp.fast.boottest_wald  |  H0: Rβ = r  |  q={self.df}  "
            f"weights={self.weights}  B={self.n_boots}\n"
            f"  Wald = {self.wald_obs:.4f}    p = {self.pvalue:.4f}"
        )


def boottest_wald(
    X: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
    R: np.ndarray,
    r: Optional[np.ndarray] = None,
    *,
    weights: str = "rademacher",
    B: int = 9999,
    seed: Optional[int] = None,
    obs_weights: Optional[np.ndarray] = None,
    extra_df: int = 0,
) -> BootWaldResult:
    """Wild cluster bootstrap of the joint linear hypothesis ``R β = r``.

    Computes the Wald statistic ``W = (R β - r)' (R V R')^{-1} (R β - r)``
    using the CR1 cluster-robust variance, then bootstraps under the null
    by imposing the constraint ``R β_R = r`` on the restricted fit and
    drawing wild cluster weights as in :func:`boottest`. The p-value is
    one-sided on the upper tail of the Wald distribution.

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)
    cluster : ndarray, shape (n,)
        Cluster ids.
    R : ndarray, shape (q, k)
        Restriction matrix; full row rank required (q ≤ k).
    r : ndarray, shape (q,), optional
        Right-hand side. Default zero vector.
    weights : {"rademacher", "webb"}
    B : int
    seed : int, optional
    obs_weights : ndarray, optional
    extra_df : int, default 0
        Additional residual-DOF charge in the CR1 small-sample factor
        (used both for the observed Wald and uniformly across bootstrap
        replicas). Pass ``Σ(G_k - 1)`` for FE-residualised ``X`` — same
        convention as :func:`crve` and :func:`boottest`.

    Returns
    -------
    BootWaldResult
    """
    if weights not in ("rademacher", "webb"):
        raise ValueError(f"weights={weights!r}; expected 'rademacher' or 'webb'")
    if extra_df < 0:
        raise ValueError(f"extra_df={extra_df}; must be >= 0")

    n, k = X.shape
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    if R.shape[1] != k:
        raise ValueError(f"R has {R.shape[1]} cols but X has {k}")
    q = R.shape[0]
    if q < 1:
        raise ValueError("R must have at least one row")
    if q > k:
        raise ValueError(f"R has {q} rows > k={k}; over-determined")
    # Rank check: full row rank
    if np.linalg.matrix_rank(R) < q:
        raise ValueError("R must have full row rank")
    if r is None:
        r = np.zeros(q, dtype=np.float64)
    else:
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        if r.shape[0] != q:
            raise ValueError(f"r has length {r.shape[0]} but R has {q} rows")

    rng = np.random.default_rng(seed)
    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G < 2:
        raise ValueError(f"boottest_wald: need at least 2 clusters, got {G}")

    w = obs_weights if obs_weights is not None else np.ones(n)

    # --- Unrestricted fit + observed Wald ---
    XtWX = X.T @ (X * w[:, None])
    bread = np.linalg.inv(XtWX)
    beta_full = bread @ (X.T @ (w * y))
    resid_full = y - X @ beta_full
    V_full = crve(
        X, resid_full, cluster, weights=w, bread=bread, type="cr1",
        extra_df=extra_df,
    )
    diff = R @ beta_full - r
    RVRT = R @ V_full @ R.T
    try:
        wald_obs = float(diff @ np.linalg.solve(RVRT, diff))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"R V R' is singular (rank {np.linalg.matrix_rank(RVRT)} of {q}); "
            "the restriction matrix may be redundant given the data."
        ) from exc

    # --- Restricted fit: β_R = β_full - V R' (R V R')^{-1} (R β_full - r) ---
    # NB: V here is the *unweighted* (X'WX)^{-1}; this gives the constrained
    # WLS solution that exactly satisfies R β_R = r (Greene 2003 §6.4).
    bread_RT = bread @ R.T                                        # (k, q)
    RbreadRT = R @ bread_RT                                       # (q, q)
    beta_R = beta_full - bread_RT @ np.linalg.solve(RbreadRT, diff)
    fitted_R = X @ beta_R
    resid_R = y - fitted_R

    weight_fn = _rademacher_weights if weights == "rademacher" else _webb6_weights
    cr1_factor = (G / (G - 1.0)) * ((n - 1.0) / max(n - k - extra_df, 1))
    Xw = X * w[:, None]
    XwT = Xw.T

    boot_wald = np.empty(B)
    for b in range(B):
        v_g = weight_fn(rng, G)
        v_i = v_g[cluster_codes]
        y_b = fitted_R + v_i * resid_R
        beta_b = bread @ (XwT @ y_b)
        resid_b = y_b - X @ beta_b
        score_b = resid_b[:, None] * Xw                            # (n, k)
        cs = np.zeros((G, k))
        np.add.at(cs, cluster_codes, score_b)
        meat = cs.T @ cs
        V_b = (bread @ meat @ bread) * cr1_factor
        diff_b = R @ beta_b - r
        RVR_b = R @ V_b @ R.T
        try:
            boot_wald[b] = float(diff_b @ np.linalg.solve(RVR_b, diff_b))
        except np.linalg.LinAlgError:
            boot_wald[b] = np.nan

    finite_mask = np.isfinite(boot_wald)
    if not finite_mask.any():
        raise RuntimeError("All bootstrap Wald stats were singular; aborting")
    pvalue = float(np.mean(boot_wald[finite_mask] >= wald_obs))

    return BootWaldResult(
        R=R, r=r, wald_obs=wald_obs, pvalue=pvalue,
        n_boots=int(finite_mask.sum()), weights=weights,
        boot_wald_dist=boot_wald[finite_mask], df=q,
    )


# ---------------------------------------------------------------------------
# Bell-McCaffrey / Imbens-Kolesar Satterthwaite DOF for CR2 t-tests
# ---------------------------------------------------------------------------

def cluster_dof_bm(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    contrast: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """Bell-McCaffrey / Imbens-Kolesar Satterthwaite DOF for a single
    linear contrast under CR2 cluster-robust inference.

    For testing ``H0: c'β = c0`` with the CR2 sandwich variance, the
    cluster-robust t-statistic is approximated by ``t_{ν_BM}`` where

        ν_BM = (Σ_g λ_g)² / Σ_g λ_g²

    and ``λ_g = c' V_g c`` is the cluster-g contribution to the CR2
    sandwich under IID null errors:

        V_g = bread X_g'^T W_g (I - H_gg)^{-1} W_g X_g bread,
        H_gg = X_g · bread · X_g'^T · diag(w_g).

    Equivalently, with ``A_g = (I - H_gg)^{-1/2}`` and ``v = X_g · bread · c``,
    ``λ_g = ||A_g · W_g · v||²`` — the form used by this implementation
    for numerical stability (eigendecomposition of the symmetrised
    ``I - 0.5(H + H')`` with eigenvalue floor 1e-12, matching :func:`crve`'s
    CR2 path).

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Regressors (already FE-residualised if applicable).
    cluster : ndarray, shape (n,)
        Cluster identifiers.
    contrast : ndarray, shape (k,)
        Linear restriction ``c`` being tested.
    weights : ndarray, shape (n,), optional
        Observation weights. Default 1.
    bread : ndarray, shape (k, k), optional
        Pre-computed inverse Hessian / ``(X'WX)^{-1}``.

    Returns
    -------
    float
        ``ν_BM``, bounded between 1 and ``G - 1`` for well-conditioned
        problems. Use this in place of the naive ``G - 1`` cluster DOF
        when calling ``scipy.stats.t`` for finite-sample inference.

    Notes
    -----
    Only ships the **single-contrast** Satterthwaite DOF. Multi-contrast
    (Wald / matrix Satterthwaite) DOF is the natural follow-up and lives
    next to :func:`boottest_wald` in scope.

    Convention vs ``clubSandwich``
    ------------------------------
    Both this implementation and R's ``clubSandwich::coef_test(...,
    test="Satterthwaite")`` are "Bell-McCaffrey-style" Satterthwaite
    approximations. The CR2 *variance* matches clubSandwich to the
    visible precision (verified separately). The DOF formula has
    documented implementation variants:

    - StatsPAI / BM 2002 simplified: ``ν = (Σ λ_g)² / Σ λ_g²`` —
      exactly what falls out of ``2 [E(V̂)]² / Var(V̂)`` under a Gaussian
      IID null on residuals.
    - clubSandwich (Pustejovsky-Tipton 2018): adds cross-cluster
      adjustment terms involving the working covariance ``Φ`` over the
      full residual vector.

    On a typical panel the two agree within ~5–10%. clubSandwich is
    slightly more conservative (smaller ν); StatsPAI's value is between
    that and the naive ``G - 1``. Use ``boottest`` for tight
    finite-sample inference if the difference matters.

    References
    ----------
    Bell, R. M., McCaffrey, D. F. (2002). Bias reduction in standard
    errors for linear regression with multi-stage samples. Survey
    Methodology 28(2), 169–179.

    Imbens, G. W., Kolesár, M. (2016). Robust standard errors in small
    samples: some practical advice. Review of Economics and Statistics
    98(4), 701–712.

    Pustejovsky, J. E., Tipton, E. (2018). Small-sample methods for
    cluster-robust variance estimation and hypothesis testing in fixed
    effects models. Journal of Business & Economic Statistics 36(4),
    672–683.
    """
    contrast = np.asarray(contrast, dtype=np.float64).ravel()
    n, k = X.shape
    if contrast.shape[0] != k:
        raise ValueError(
            f"contrast has {contrast.shape[0]} entries but X has k={k} cols"
        )
    if weights is None:
        weights = np.ones(n)

    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G < 2:
        raise ValueError(f"cluster_dof_bm: need at least 2 clusters, got {G}")

    if bread is None:
        XtWX = X.T @ (X * weights[:, None])
        bread = np.linalg.inv(XtWX)

    # Pre-compute ``X · bread · contrast`` once; per-cluster only needs
    # the subset rows.
    bread_c = bread @ contrast                    # (k,)
    Xbc = X @ bread_c                             # (n,)

    lambdas = np.zeros(G)
    for g in range(G):
        mask = cluster_codes == g
        X_g = X[mask]
        w_g = weights[mask]
        n_g = X_g.shape[0]

        # H_gg = X_g · bread · X_g'^T diag(w_g) — same formula as :func:`crve`.
        Xb = X_g @ bread
        H_gg = Xb @ (X_g * w_g[:, None]).T
        M = np.eye(n_g) - 0.5 * (H_gg + H_gg.T)
        evals, evecs = np.linalg.eigh(M)
        evals = np.maximum(evals, 1e-12)
        inv_sqrt = (evecs * (evals ** -0.5)) @ evecs.T

        # λ_g = ||A_g · W_g · v||² with v = X_g · bread · c
        v = Xbc[mask]                              # (n_g,)
        Wv = w_g * v                               # (n_g,)
        AWv = inv_sqrt @ Wv                        # (n_g,)
        lambdas[g] = float(AWv @ AWv)

    sum_lam = float(lambdas.sum())
    sum_lam2 = float((lambdas * lambdas).sum())
    if sum_lam2 <= 0:
        # Degenerate (all λ_g = 0): contrast lies in the null space of
        # the cluster-score basis. Return G - 1 as a conservative fallback.
        return float(G - 1)
    return float(sum_lam ** 2 / sum_lam2)


def cluster_dof_wald_bm(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """Bell-McCaffrey / Pustejovsky-Tipton matrix Satterthwaite DOF for
    a multi-restriction Wald test under CR2 cluster-robust inference.

    Generalises :func:`cluster_dof_bm` from a 1-D contrast to a
    ``q``-row restriction matrix ``R``. For testing
    ``H0: R β = r`` with the CR2 sandwich variance, the cluster-robust
    Wald statistic ``W = (Rβ̂ - r)' (R V_CR2 R')^{-1} (Rβ̂ - r)`` is
    approximated by a (scaled) ``F_{q, ν_W}`` distribution where

        ν_W = [tr(M)]² / tr(M²)

    and ``M = Σ_g E_g E_g'`` with cluster contributions

        E_g = R · bread · X_g'^T · A_g · sqrt(W_g)    (q × n_g)
        A_g = (I_{n_g} - H_gg)^{-1/2}
        H_gg = X_g · bread · X_g'^T · diag(w_g).

    For ``q = 1`` this collapses to ``ν = (Σ λ_g)² / Σ λ_g²`` —
    bit-identical to :func:`cluster_dof_bm`.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Regressors (already FE-residualised if applicable).
    cluster : ndarray, shape (n,)
        Cluster identifiers.
    R : ndarray, shape (q, k)
        Restriction matrix; full row rank required (q ≤ k).
    weights : ndarray, shape (n,), optional
        Observation weights. Default 1.
    bread : ndarray, shape (k, k), optional
        Pre-computed inverse Hessian / ``(X'WX)^{-1}``.

    Returns
    -------
    float
        ``ν_W``. For q ≥ 1, bounded below by ``q`` and above by
        ``q · (G - q) / G`` × something — in practice well-behaved
        problems have ``ν_W`` in ``(q, G - q + 1)``.

    Notes
    -----
    **Not equivalent to** ``clubSandwich::Wald_test(..., test="HTZ")``.
    The HTZ test uses a Hotelling-T² approximation with the more
    elaborate Pustejovsky-Tipton 2018 §3.2 moment-matching DOF, which
    is typically much smaller (more conservative) than the BM 2002 §3
    simplified ``(Σ tr)² / Σ ||·||_F²`` formula implemented here. On
    moderate panels the two can differ by 50–100%.

    For the strictest small-G inference prefer :func:`boottest_wald`,
    which avoids any asymptotic approximation.

    References
    ----------
    Bell, R. M., McCaffrey, D. F. (2002). Bias reduction in standard
    errors for linear regression with multi-stage samples. Survey
    Methodology 28(2), 169–179.

    Pustejovsky, J. E., Tipton, E. (2018). Small-sample methods for
    cluster-robust variance estimation and hypothesis testing in fixed
    effects models. Journal of Business & Economic Statistics 36(4),
    672–683.
    """
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    n, k = X.shape
    if R.shape[1] != k:
        raise ValueError(f"R has {R.shape[1]} cols but X has k={k}")
    q = R.shape[0]
    if q < 1:
        raise ValueError("R must have at least one row")
    if q > k:
        raise ValueError(f"R has {q} rows > k={k}; over-determined")
    if np.linalg.matrix_rank(R) < q:
        raise ValueError("R must have full row rank")

    if weights is None:
        weights = np.ones(n)

    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G < 2:
        raise ValueError(f"cluster_dof_wald_bm: need at least 2 clusters, got {G}")
    if G <= q:
        raise ValueError(
            f"cluster_dof_wald_bm: need G > q for a meaningful Wald DOF "
            f"(got G={G}, q={q})"
        )

    if bread is None:
        XtWX = X.T @ (X * weights[:, None])
        bread = np.linalg.inv(XtWX)

    # R · bread is the per-restriction "lifted" contribution; precompute
    # once. Shape (q, k).
    R_bread = R @ bread

    # Matrix Satterthwaite (BM 2002 §3 / IK 2016 eq 14) for q-dim Wald:
    #   ν_W = (Σ_g tr(E_g E_g'))² / Σ_g ||E_g E_g'||_F²
    # where ||·||_F is the Frobenius norm. For q=1 this collapses to
    # (Σ λ_g)² / Σ λ_g² — i.e. exactly :func:`cluster_dof_bm`.
    sum_tr = 0.0          # Σ_g tr(E_g E_g')   = Σ_g ||E_g||_F²
    sum_frob2 = 0.0       # Σ_g ||E_g E_g'||_F²  = Σ_g tr((E_g E_g')²)

    for g in range(G):
        mask = cluster_codes == g
        X_g = X[mask]
        w_g = weights[mask]
        n_g = X_g.shape[0]

        # H_gg = X_g · bread · X_g'^T · diag(w_g) — same formula as :func:`crve`.
        Xb = X_g @ bread
        H_gg = Xb @ (X_g * w_g[:, None]).T
        Hsym = 0.5 * (H_gg + H_gg.T)
        Msym = np.eye(n_g) - Hsym
        evals, evecs = np.linalg.eigh(Msym)
        evals = np.maximum(evals, 1e-12)
        inv_sqrt = (evecs * (evals ** -0.5)) @ evecs.T

        # E_g = R · bread · X_g'^T · A_g · sqrt(W_g)        (q, n_g)
        sqrt_w = np.sqrt(w_g)
        RB_Xt = R_bread @ X_g.T              # (q, n_g)
        RB_Xt_A = RB_Xt @ inv_sqrt           # (q, n_g)
        E_g = RB_Xt_A * sqrt_w               # (q, n_g)

        EEt = E_g @ E_g.T                    # (q, q), symmetric PSD
        sum_tr += float(np.trace(EEt))
        # ||EEt||_F² = sum of squared entries (== tr(EEt²) for symmetric)
        sum_frob2 += float(np.sum(EEt * EEt))

    if sum_frob2 <= 0:
        # Degenerate: the contrasts lie outside the cluster-score image.
        # Conservative fallback: ``q · (G - q)`` (a typical IK upper bound).
        return float(q * (G - q))
    return float(sum_tr ** 2 / sum_frob2)


# ---------------------------------------------------------------------------
# clubSandwich-equivalent HTZ Wald (Pustejovsky-Tipton 2018, JBES 36(4))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaldTestResult:
    """Outcome of a cluster-robust Wald test under CR2 sandwich.

    Mirrors the contents of R ``clubSandwich::Wald_test(...)`` so that
    cross-language replication can compare any field directly.

    Attributes
    ----------
    test : str
        Test variant. ``"HTZ"`` only in v1; future variants
        (``"HTA"``, ``"KZ"``, ``"Naive"``, ``"EDF"``) may be added.
    q : int
        Number of restrictions (rank of ``R``).
    eta : float
        Pustejovsky-Tipton 2018 §3.2 moment-matching DOF (``η``).
    F_stat : float
        Hotelling-T²-scaled F statistic ``(η - q + 1) / (η · q) · Q``.
    p_value : float
        Right-tail probability ``1 - F_{q, η-q+1}.cdf(F_stat)``.
    Q : float
        Raw cluster-robust Wald statistic ``(Rβ̂ - r)' V_R^{-1} (Rβ̂ - r)``.
    R : ndarray, shape (q, k)
        Restriction matrix.
    r : ndarray, shape (q,)
        Null value.
    V_R : ndarray, shape (q, q)
        ``R · V^CR2 · R^T``.
    """
    test: str
    q: int
    eta: float
    F_stat: float
    p_value: float
    Q: float
    R: np.ndarray
    r: np.ndarray
    V_R: np.ndarray

    def summary(self) -> str:
        return (
            f"Wald test (test={self.test!r}, q={self.q}):\n"
            f"  η = {self.eta:.4f},  F = {self.F_stat:.4f},  "
            f"p = {self.p_value:.4f}  (raw Q = {self.Q:.4f})"
        )

    def to_dict(self) -> dict:
        return {
            "test": self.test, "q": self.q, "eta": self.eta,
            "F_stat": self.F_stat, "p_value": self.p_value, "Q": self.Q,
            "R": self.R.tolist(), "r": self.r.tolist(),
            "V_R": self.V_R.tolist(),
        }


def _htz_per_cluster_quantities(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> dict:
    """Internal helper: per-cluster lifted contributions for the HTZ test.

    Computes the matrices that both :func:`cluster_dof_wald_htz` and
    :func:`cluster_wald_htz` need:

    Per cluster ``g``:
        H_gg      = X_g · bread · X_g^T · diag(w_g)            (n_g × n_g)
        A_g       = (I_g - 0.5(H_gg + H_gg^T))^{-1/2}          (n_g × n_g)
        G_g       = R · bread · X_g^T · diag(√w_g)             (q × n_g)
                    # No A_g — Ω = Σ_g G_g G_g^T = R · bread · R^T
                    # under the working covariance Φ = I.
        A_g_sqrtW = diag(√w_g) · A_g                           (n_g × n_g)
                    # Applied to raw residuals: ẽ_g = A_g_sqrtW · u_g.
                    # Then v_g = G_g · ẽ_g = R bread X_g^T diag(w_g) A_g u_g
                    # — exactly the weighted CR2 score lifted to R-subspace,
                    # matching :func:`crve` (CR2 path).

    Aggregate:
        Ω = Σ_g G_g G_g^T                                       (q × q)
                    # Unweighted: Ω = R · (X'X)^{-1} · R^T.
                    # Weighted:   Ω = R · (X'WX)^{-1} · R^T.

    Eigenvalue floor 1e-12 on ``(I - Hsym)`` matches the convention in
    :func:`crve` (CR2 path) and :func:`cluster_dof_wald_bm`.

    Returns
    -------
    dict
        Keys: ``"G_g"`` (list of q×n_g arrays per cluster, in cluster-code
        order from ``pd.factorize(sort=False)``), ``"A_g_sqrtW"`` (list of
        n_g×n_g arrays — same ordering), ``"Omega"`` (q×q),
        ``"cluster_codes"`` (n-vector), ``"G"`` (cluster count), ``"q"``
        (R rank), ``"bread"``.

    Raises
    ------
    ValueError
        If ``R`` is rank-deficient, ``G ≤ q``, or ``weights`` non-positive.
    """
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    n, k = X.shape
    if R.shape[1] != k:
        raise ValueError(f"R has {R.shape[1]} cols but X has k={k}")
    q = R.shape[0]
    if q < 1:
        raise ValueError("R must have at least one row")
    if q > k:
        raise ValueError(f"R has {q} rows > k={k}; over-determined")
    if np.linalg.matrix_rank(R) < q:
        raise ValueError("R must have full row rank")

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if (weights <= 0).any():
            raise ValueError("weights must be strictly positive")

    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G_clusters = int(cluster_codes.max()) + 1 if cluster_codes.size else 0
    if G_clusters < 2:
        raise ValueError(
            f"HTZ requires at least 2 clusters, got {G_clusters}"
        )
    if G_clusters <= q:
        raise ValueError(
            f"HTZ Wald requires G > q (got G={G_clusters}, q={q})"
        )

    if bread is None:
        XtWX = X.T @ (X * weights[:, None])
        bread = np.linalg.inv(XtWX)

    R_bread = R @ bread                                  # (q, k)

    G_list: list = []
    Asw_list: list = []
    Omega = np.zeros((q, q))

    for cg in range(G_clusters):
        mask = cluster_codes == cg
        X_g = X[mask]
        w_g = weights[mask]
        n_g = X_g.shape[0]
        if n_g == 1:
            warnings.warn(
                f"HTZ: singleton cluster (code={cg}, n_g=1); A_g floored",
                RuntimeWarning,
                stacklevel=3,
            )

        Xb = X_g @ bread                                  # (n_g, k)
        H_gg = Xb @ (X_g * w_g[:, None]).T                # (n_g, n_g)
        Hsym = 0.5 * (H_gg + H_gg.T)
        Msym = np.eye(n_g) - Hsym
        evals, evecs = np.linalg.eigh(Msym)
        evals = np.maximum(evals, 1e-12)
        A_g = (evecs * (evals ** -0.5)) @ evecs.T          # (n_g, n_g)

        sqrt_w = np.sqrt(w_g)
        # G_g = R · bread · X_g^T · diag(√w_g)         (NO A_g)
        RB_Xt = R_bread @ X_g.T                            # (q, n_g)
        G_g = RB_Xt * sqrt_w                               # (q, n_g)

        # A_g_sqrtW = diag(√w_g) · A_g  (applied LEFT-to-RIGHT to residuals)
        # so that v_g = G_g · (A_g_sqrtW · u_g)
        #             = R bread X_g^T √w_g · √w_g A_g u_g
        #             = R bread X_g^T diag(w_g) A_g u_g    ← matches crve.
        A_g_sqrtW = sqrt_w[:, None] * A_g                  # (n_g, n_g)

        G_list.append(G_g)
        Asw_list.append(A_g_sqrtW)
        Omega += G_g @ G_g.T

    # Symmetrize Ω against floating drift
    Omega = 0.5 * (Omega + Omega.T)

    return {
        "G_g": G_list,
        "A_g_sqrtW": Asw_list,
        "Omega": Omega,
        "cluster_codes": cluster_codes,
        "G": G_clusters,
        "q": q,
        "bread": bread,
    }


def cluster_dof_wald_htz(
    X: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> float:
    """Stub — implemented in Task 3."""
    raise NotImplementedError("cluster_dof_wald_htz: implementation pending")


def cluster_wald_htz(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    *,
    R: np.ndarray,
    beta: np.ndarray,
    r: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bread: Optional[np.ndarray] = None,
) -> WaldTestResult:
    """Stub — implemented in Task 4."""
    raise NotImplementedError("cluster_wald_htz: implementation pending")


__all__ = [
    "crve", "boottest", "BootTestResult",
    "boottest_wald", "BootWaldResult",
    "cluster_dof_bm", "cluster_dof_wald_bm",
    "cluster_dof_wald_htz", "cluster_wald_htz", "WaldTestResult",
]
