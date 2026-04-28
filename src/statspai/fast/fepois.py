"""Native StatsPAI Poisson HDFE estimator.

Implements the PPML-HDFE algorithm (Correia, Guimarães, Zylkin 2020,
"Fast Poisson estimation with high-dimensional fixed effects", Stata
Journal 20(1)). Python IRLS outer loop that delegates the within-
transformation to ``sp.fast.demean`` (unweighted) and to the Phase A
Rust kernel ``statspai_hdfe.demean_2d_weighted`` (weighted, used by
the IRLS-internal demean) when the compiled extension is available;
falls back to a pure-NumPy ``np.bincount`` weighted path otherwise.

Goals
-----
1. Numerically match ``fixest::fepois`` and Stata's ``ppmlhdfe`` on the
   coefficient vector to ~1e-6 across our medium synthetic panel.
2. Be **independent of pyfixest** — this is the path that lets us keep
   shipping ``sp.fepois`` even if pyfixest goes away or its API drifts.
3. Hand back a result object with the same surface (``coef()``,
   ``se()``, ``tidy()``, ``summary()``) so downstream code that expects
   the old shape keeps working.

What it does NOT (yet) do
-------------------------
- Anderson(m) acceleration on the IRLS outer loop. Phase 2 ships with
  step-halving only; Anderson is a v1.7.1 tracked item.
- Cluster-robust standard errors. The IID and HC1 sandwiches ship; CR1
  arrives in Phase 4 alongside wild bootstrap.
- Negative-Binomial / Logit / Gaussian families. Poisson only here;
  ``feglm`` is a follow-up that swaps the family object.
- Native Rust IRLS. Producing one is a multi-week task and was
  intentionally split off the Phase 2 critical path. The Phase 1 demean
  kernel is exposed so a future Rust IRLS can call it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .demean import demean as _fast_demean

# Optional Rust kernel for the weighted IRLS-internal demean. When available,
# the dispatcher below routes to it; when not, the pure-NumPy fallback runs.
try:
    import statspai_hdfe as _rust_hdfe  # type: ignore
    _HAS_RUST_HDFE = True
except ImportError:  # pragma: no cover  - exercised in CI on no-Rust wheels
    _rust_hdfe = None  # type: ignore
    _HAS_RUST_HDFE = False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FePoisResult:
    """Outcome of :func:`fepois`.

    Attributes mirror pyfixest / fixest naming so that downstream code
    that reaches for ``.coef()`` / ``.se()`` / ``.tidy()`` keeps working.
    """

    formula: str
    coef_names: List[str]
    coef_vec: np.ndarray
    vcov_matrix: np.ndarray
    n_obs: int
    n_kept: int
    n_dropped_singletons: int
    n_dropped_separation: int
    deviance: float
    log_likelihood: float
    iterations: int
    converged: bool
    fe_names: List[str]
    fe_cardinality: List[int]
    vcov_type: str
    df_residual: int = 0           # n_kept - p - Σ(G_k - 1); used for t-stat tails
    backend: str = "statspai-native"

    # ------------------------------------------------------------------
    # pyfixest-compatible accessors
    # ------------------------------------------------------------------

    def coef(self) -> pd.Series:
        return pd.Series(self.coef_vec, index=self.coef_names, name="Estimate")

    def se(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.vcov_matrix)),
                         index=self.coef_names, name="Std. Error")

    def vcov(self) -> pd.DataFrame:
        return pd.DataFrame(self.vcov_matrix, index=self.coef_names,
                            columns=self.coef_names)

    def tidy(self) -> pd.DataFrame:
        b = self.coef_vec
        s = np.sqrt(np.diag(self.vcov_matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(s > 0, b / s, np.nan)
        # Two-sided z-test; large-sample for fepois.
        from scipy.stats import norm
        p = 2.0 * (1.0 - norm.cdf(np.abs(t)))
        return pd.DataFrame({
            "Estimate": b,
            "Std. Error": s,
            "z value": t,
            "Pr(>|z|)": p,
        }, index=self.coef_names)

    def summary(self) -> str:
        lines: List[str] = []
        lines.append(
            f"sp.fast.fepois  |  {self.formula}  |  Poisson, log link, vcov={self.vcov_type}"
        )
        lines.append(
            f"N={self.n_obs:,}  kept={self.n_kept:,}  "
            f"singletons={self.n_dropped_singletons}  "
            f"separation={self.n_dropped_separation}  "
            f"iters={self.iterations}  converged={self.converged}"
        )
        if self.fe_names:
            fe_desc = ", ".join(
                f"{n}({c:,})" for n, c in zip(self.fe_names, self.fe_cardinality)
            )
            lines.append(f"Fixed effects: {fe_desc}")
        lines.append("")
        lines.append(self.tidy().to_string(float_format=lambda x: f"{x:.6f}"))
        lines.append("")
        lines.append(
            f"Deviance: {self.deviance:.4f}    Log-likelihood: {self.log_likelihood:.4f}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover  - cosmetic
        return self.summary()


# ---------------------------------------------------------------------------
# Formula parsing  ("y ~ x1 + x2 | fe1 + fe2")
# ---------------------------------------------------------------------------

def _parse_fepois_formula(formula: str) -> Tuple[str, List[str], List[str]]:
    """Split ``y ~ x | fe`` into (lhs, rhs_terms, fe_terms).

    Intentionally minimal — no ``i()``, no ``^``, no IV — those land in
    Phase 3's full DSL. Here we only need the three-part split so we can
    extract the columns and run the IRLS.
    """
    if "~" not in formula:
        raise ValueError(f"formula {formula!r} missing '~'")
    lhs_str, rest = formula.split("~", 1)
    lhs = lhs_str.strip()
    if not lhs:
        raise ValueError("LHS of formula is empty")

    fe_terms: List[str] = []
    if "|" in rest:
        rhs_str, fe_str = rest.split("|", 1)
        fe_terms = [t.strip() for t in fe_str.split("+") if t.strip()]
    else:
        rhs_str = rest

    rhs_terms = [t.strip() for t in rhs_str.split("+") if t.strip()]
    if not rhs_terms:
        raise ValueError("formula must have at least one regressor on the RHS")
    return lhs, rhs_terms, fe_terms


# ---------------------------------------------------------------------------
# Weighted demean (one full sweep over K FEs, in place on a column)
# ---------------------------------------------------------------------------

def _weighted_sweep(
    col: np.ndarray,
    codes: np.ndarray,
    wsum: np.ndarray,
    weights: np.ndarray,
) -> None:
    """In-place: subtract weighted group means.

    ``mean_g(col) = sum_{i in g} w_i * col_i / sum_{i in g} w_i``.
    """
    weighted_sums = np.bincount(
        codes, weights=col * weights, minlength=wsum.size
    )
    means = np.divide(
        weighted_sums, wsum,
        out=np.zeros_like(weighted_sums), where=wsum > 0
    )
    col -= means[codes]


def _aitken_extrapolate(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """Vector Irons-Tuck extrapolation; returns x2 if denominator degenerate.

    Identical formula to the Rust ``demean::aitken_step``. Pure NumPy because
    we run inside the (already-Python) IRLS outer loop.
    """
    d1 = x1 - x0
    d2 = x2 - 2.0 * x1 + x0
    den = float(d2 @ d2)
    if den < 1e-30:
        return x2
    alpha = float(d1 @ d2) / den
    return x0 - alpha * d1


def _weighted_ap_demean_numpy(
    arr: np.ndarray,
    fe_codes_list: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
    accelerate: bool = True,
    accel_period: int = 5,
) -> Tuple[np.ndarray, int, bool]:
    """Pure-NumPy weighted alternating-projection demean (canonical reference + Rust-unavailable fallback).

    Returns the residualised array (copy), iter count, and convergence
    flag. Used inside the IRLS loop where the weight vector ``w = mu``
    changes each iteration.

    Mirrors the Rust ``demean_column_inplace`` algorithm: AP sweep +
    Irons-Tuck extrapolation every ``accel_period`` sweeps, with the
    "max|x_acc| < 10*base_scale" safeguard against blow-up.
    """
    if arr.ndim == 1:
        squeeze = True
        arr = arr.reshape(-1, 1)
    else:
        squeeze = False
    arr = arr.astype(np.float64, copy=True)
    n, p = arr.shape

    K = len(fe_codes_list)
    if K == 0:
        if squeeze:
            return arr.ravel(), 0, True
        return arr, 0, True

    # Pre-compute weighted group sums per FE
    wsum_list: List[np.ndarray] = []
    for k in range(K):
        wsum_list.append(np.bincount(
            fe_codes_list[k], weights=weights, minlength=counts_list[k].size
        ))

    # K=1 closed form
    if K == 1:
        for j in range(p):
            _weighted_sweep(arr[:, j], fe_codes_list[0], wsum_list[0], weights)
        if squeeze:
            return arr.ravel(), 1, True
        return arr, 1, True

    iters = 0
    converged_all = True
    for j in range(p):
        col = arr[:, j]
        base_scale = float(np.max(np.abs(col))) + 1e-30
        stop_threshold = tol * base_scale
        hist: List[np.ndarray] = []
        col_iters = 0
        col_conv = False
        for it in range(max_iter):
            before = col.copy()
            for k in range(K):
                _weighted_sweep(col, fe_codes_list[k], wsum_list[k], weights)
            mdx = float(np.max(np.abs(col - before)))
            col_iters = it + 1
            if mdx <= stop_threshold:
                col_conv = True
                break
            if accelerate:
                hist.append(col.copy())
                if len(hist) >= 3 and (it + 1) % accel_period == 0:
                    acc = _aitken_extrapolate(hist[-3], hist[-2], hist[-1])
                    # Safeguard: only accept the jump if it doesn't blow up.
                    if float(np.max(np.abs(acc))) < 10.0 * base_scale:
                        col[:] = acc
                    hist.clear()
        arr[:, j] = col
        iters = max(iters, col_iters)
        if not col_conv:
            converged_all = False

    if squeeze:
        return arr.ravel(), iters, converged_all
    return arr, iters, converged_all


def _weighted_ap_demean(
    arr: np.ndarray,
    fe_codes_list: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
    accelerate: bool = True,
    accel_period: int = 5,
) -> Tuple[np.ndarray, int, bool]:
    """Dispatcher: Rust when available, NumPy fallback otherwise.

    Routes to ``statspai_hdfe.demean_2d_weighted`` when the compiled
    extension is loadable (Phase A); falls back to
    ``_weighted_ap_demean_numpy`` (the canonical reference) otherwise.
    Output is bit-for-bit identical between paths within float-rounding;
    parity is verified by ``test_rust_weighted_demean_matches_numpy_kernel``
    at atol ≤ 1e-14.

    K=0 (no-op) and K=1 (closed form) stay on the NumPy path because FFI
    overhead would dominate. K ≥ 2 is where the Rust win lives.
    """
    K = len(fe_codes_list)
    if not _HAS_RUST_HDFE or K < 2:
        return _weighted_ap_demean_numpy(
            arr, fe_codes_list, counts_list, weights,
            max_iter=max_iter, tol=tol,
            accelerate=accelerate, accel_period=accel_period,
        )

    squeeze = arr.ndim == 1
    src = arr.reshape(-1, 1) if squeeze else arr
    # Single allocation directly into F-order f64 (avoids the previous
    # astype-then-asfortranarray double copy on the C-order hot path).
    arr_F = np.array(src, dtype=np.float64, order="F", copy=True)

    # Per-FE precompute of weighted group sums. K bincount calls per IRLS
    # iter — O(n) each, negligible vs the sweep.
    wsum_list = [
        np.bincount(fe_codes_list[k], weights=weights,
                    minlength=counts_list[k].size).astype(np.float64)
        for k in range(K)
    ]

    infos = _rust_hdfe.demean_2d_weighted(
        arr_F, list(fe_codes_list), wsum_list, weights,
        int(max_iter), 0.0, float(tol), bool(accelerate), int(accel_period),
    )

    iters = max(int(d["iters"]) for d in infos) if infos else 0
    converged_all = all(bool(d["converged"]) for d in infos) if infos else True

    if squeeze:
        return arr_F.ravel(), iters, converged_all
    return arr_F, iters, converged_all


# ---------------------------------------------------------------------------
# Separation detection (Correia 2020 §3.4 informal heuristic)
# ---------------------------------------------------------------------------

def _drop_separation(
    y: np.ndarray, fe_codes_list: List[np.ndarray]
) -> np.ndarray:
    """Identify rows in zero-only FE clusters (Poisson separation).

    Returns a keep-mask. Iterative: if dropping rows in one FE creates
    a new zero-only cluster in another, repeat. Mirrors what
    ``pyfixest`` and ``ppmlhdfe`` do as their first pre-pass.
    """
    n = y.size
    keep = np.ones(n, dtype=bool)
    while True:
        dropped = False
        for codes in fe_codes_list:
            ck = codes[keep]
            yk = y[keep]
            G = int(ck.max()) + 1 if ck.size else 0
            if G == 0:
                continue
            sums = np.bincount(ck, weights=yk, minlength=G)
            zero_groups = np.where(sums == 0)[0]
            if zero_groups.size == 0:
                continue
            mask_local = np.isin(ck, zero_groups)
            if mask_local.any():
                gidx = np.where(keep)[0]
                keep[gidx[mask_local]] = False
                dropped = True
        if not dropped:
            break
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fepois(
    formula: str,
    data: pd.DataFrame,
    *,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    maxiter: int = 50,
    tol: float = 1e-8,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1000,
    drop_singletons: bool = True,
    drop_separation: bool = True,
) -> FePoisResult:
    """Poisson regression with high-dimensional fixed effects.

    Native StatsPAI implementation of the PPML-HDFE algorithm
    (Correia-Guimarães-Zylkin 2020). The public interface mirrors
    pyfixest's ``fepois`` so existing user code keeps running.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 | fe1 + fe2"``. The DSL here is intentionally
        minimal — see ``sp.fepois`` (the pyfixest-backed entry point) if
        you need ``i()``, ``^``, IV, etc.
    data : pd.DataFrame
        Must contain all columns referenced in ``formula``.
    vcov : {"iid", "hc1", "cr1"}
        Variance-covariance estimator. ``"cr1"`` is one-way cluster-
        robust (Liang-Zeger) with the FE-rank-aware small-sample factor
        ``(G/(G-1)) * (n-1)/(n - p - Σ(G_k - 1))``. CR2/CR3 are not yet
        wired in for fepois — the WLS leverage adjustment requires
        weighting the H_gg matrix by ``μ`` (the IRLS working weight),
        which the generic :func:`crve` doesn't separate from the score
        weights. See :func:`crve` for OLS / linear cluster sandwich.
    cluster : str, optional
        Column name of cluster identifiers. Required when
        ``vcov="cr1"``; rejected otherwise. NaN cluster values raise.
    weights : str, optional
        Column name of observation weights (e.g. survey / frequency
        weights). Each obs's contribution to the log-likelihood is
        scaled by ``w_i``; the IRLS working weight becomes ``w_i * mu_i``.
        ``None`` (default) is the unweighted MLE.
    maxiter : int
        Outer IRLS iteration cap.
    tol : float
        Relative deviance change for IRLS convergence.
    fe_tol, fe_maxiter : float, int
        Inner alternating-projection demean controls.
    drop_singletons : bool
    drop_separation : bool
        Drop rows whose FE level has all-zero outcomes (Poisson cannot
        identify them anyway).

    Returns
    -------
    FePoisResult
    """
    if vcov not in ("iid", "hc1", "cr1"):
        raise ValueError(f"vcov={vcov!r}; supported: 'iid', 'hc1', or 'cr1'")
    if vcov == "cr1" and cluster is None:
        raise ValueError("vcov='cr1' requires cluster=<column name>")
    if cluster is not None and vcov in ("iid", "hc1"):
        raise ValueError(
            f"cluster={cluster!r} provided but vcov={vcov!r}; "
            "set vcov='cr1' to compute cluster-robust SE"
        )

    lhs, rhs_terms, fe_terms = _parse_fepois_formula(formula)

    # Pull the literal "1" out of RHS as an intercept request. ``fixest``
    # auto-adds an intercept when there is no FE block; we match that.
    user_intercept = "1" in rhs_terms
    rhs_terms = [t for t in rhs_terms if t != "1"]
    add_intercept = user_intercept or not fe_terms

    needed_cols = [lhs] + rhs_terms + fe_terms
    if weights is not None:
        needed_cols = needed_cols + [weights]
    if cluster is not None:
        needed_cols = needed_cols + [cluster]
    missing = [c for c in needed_cols if c not in data.columns]
    if missing:
        raise KeyError(f"columns missing from data: {missing}")

    n_obs = len(data)
    y = data[lhs].to_numpy(dtype=np.float64).copy()
    if (y < 0).any():
        raise ValueError("Poisson y must be non-negative")
    if weights is not None:
        obs_weights = data[weights].to_numpy(dtype=np.float64).copy()
        if (obs_weights < 0).any():
            raise ValueError(f"weights column {weights!r} contains negative values")
        if not np.isfinite(obs_weights).all():
            raise ValueError(f"weights column {weights!r} contains non-finite values")
    else:
        obs_weights = None

    if cluster is not None:
        cluster_arr_full = data[cluster].to_numpy()
        # Reject NaN: ambiguous treatment, force the user to handle upstream.
        cluster_codes_check, _ = pd.factorize(
            cluster_arr_full, sort=False, use_na_sentinel=True,
        )
        if (cluster_codes_check < 0).any():
            raise ValueError(
                f"cluster column {cluster!r} contains NaN; drop or impute upstream"
            )
    else:
        cluster_arr_full = None
    X_user = data[rhs_terms].to_numpy(dtype=np.float64).copy() if rhs_terms else \
        np.empty((n_obs, 0), dtype=np.float64)
    if X_user.ndim == 1:
        X_user = X_user.reshape(-1, 1)
    if add_intercept:
        X = np.column_stack([np.ones(n_obs, dtype=np.float64), X_user])
        coef_names_full = ["(Intercept)"] + list(rhs_terms)
    else:
        X = X_user
        coef_names_full = list(rhs_terms)

    # Factorise FEs
    fe_codes_raw: List[np.ndarray] = []
    fe_card_raw: List[int] = []
    for fe_name in fe_terms:
        codes, uniq = pd.factorize(data[fe_name], sort=False, use_na_sentinel=True)
        if (codes < 0).any():
            raise ValueError(f"NaN in fixed effect column {fe_name!r}")
        fe_codes_raw.append(codes.astype(np.int64))
        fe_card_raw.append(len(uniq))

    # Iterative pre-passes:
    # 1. Singleton drop  → reduces residual DOF correctly
    # 2. Separation drop → removes Poisson-unidentified rows (zero-only FE
    #    clusters can't be fit by exp(eta) > 0).
    # We track each pass's contribution separately so users can see
    # which kind of row got removed (was a debt left over from the
    # original Phase 2 ship).
    keep = np.ones(n_obs, dtype=bool)
    n_dropped_singletons = 0
    n_dropped_separation = 0

    if drop_singletons and fe_terms:
        from .demean import _detect_singletons as _ds
        keep_s = _ds(fe_codes_raw, n_obs)
        n_dropped_singletons = int((~keep_s).sum())
        keep &= keep_s

    if drop_separation and fe_terms:
        # Always run on the *current* keep subset — singletons may have
        # already pruned rows, so we must respect that mask. One unified
        # path handles both "no singletons dropped" and "some dropped"
        # cleanly.
        sub_idx = np.where(keep)[0]
        sub_codes = [c[keep] for c in fe_codes_raw]
        sub_keep = _drop_separation(y[keep], sub_codes)
        n_dropped_separation = int((~sub_keep).sum())
        keep[sub_idx[~sub_keep]] = False

    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(
            "All rows dropped by singleton + separation pre-passes. Common "
            "causes: too-fine FE structure relative to N, or all-zero outcomes "
            "in every FE cluster. Disable the drops with drop_singletons=False "
            "/ drop_separation=False to inspect what happens."
        )
    n, p = X[keep].shape if n_kept < n_obs else X.shape
    if p == 0:
        raise ValueError(
            "No regressors after parsing — formula must include at least one "
            "RHS term (or '1' for an intercept)."
        )

    if n_kept < n_obs:
        # Re-densify FE codes on the kept subset
        fe_codes: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        for codes_k in fe_codes_raw:
            ck = codes_k[keep]
            dense, uniq = pd.factorize(ck, sort=False)
            dense = dense.astype(np.int64)
            G = len(uniq)
            fe_codes.append(dense)
            counts_list.append(
                np.bincount(dense, minlength=G).astype(np.float64)
            )
        y = y[keep]
        X = X[keep]
        if obs_weights is not None:
            obs_weights = obs_weights[keep]
        if cluster_arr_full is not None:
            cluster_arr_full = cluster_arr_full[keep]
    else:
        fe_codes = []
        counts_list = []
        for codes_k, G in zip(fe_codes_raw, fe_card_raw):
            fe_codes.append(codes_k.copy())
            counts_list.append(
                np.bincount(codes_k, minlength=G).astype(np.float64)
            )

    # ----- IRLS / PPML-HDFE main loop -----
    # n, p already computed and guarded above; reassign in case X was masked
    n, p = X.shape
    # Effective per-obs weight in the IRLS working step is ``mu * obs_weights``.
    # Default to 1 so the unweighted MLE path is unchanged.
    if obs_weights is None:
        obs_weights = np.ones(n, dtype=np.float64)
    # Initialisation: add a small jitter so log(y) is well-defined for y=0 rows.
    mu = np.maximum(y, 1.0) + 0.1
    eta = np.log(mu)
    deviance = float("inf")
    converged = False
    iters_used = 0

    for it in range(maxiter):
        z = eta + (y - mu) / mu             # working response (no weights here)
        w = mu * obs_weights                 # working weight: μ_i * w_i

        # Weighted within-transform of z and X by FE
        z_tilde, _, _ = _weighted_ap_demean(
            z, fe_codes, counts_list, w,
            max_iter=fe_maxiter, tol=fe_tol,
        )
        X_tilde, _, _ = _weighted_ap_demean(
            X, fe_codes, counts_list, w,
            max_iter=fe_maxiter, tol=fe_tol,
        )

        # WLS on demeaned: solve (X_tilde' W X_tilde) beta = X_tilde' W z_tilde
        Xw = X_tilde * w[:, None]
        XtWX = X_tilde.T @ Xw
        XtWz = X_tilde.T @ (w * z_tilde)
        try:
            beta = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"WLS step failed at iter {it}: {exc}. Check for "
                "perfect collinearity or all-zero weights."
            ) from exc

        # eta_new such that eta_new = X*beta + alpha (FE) (Correia 2020, §3.1)
        # Identity: P_FE^w(z - X*beta) = (z - X*beta) - (z_tilde - X_tilde*beta)
        # so eta_new = X*beta + P_FE^w(z - X*beta) = z - (z_tilde - X_tilde @ beta)
        eta_new = z - (z_tilde - X_tilde @ beta)
        # Numerical guard on eta to keep mu in float64 range.
        np.clip(eta_new, -30.0, 30.0, out=eta_new)
        mu_new = np.exp(eta_new)

        # Step-halving if deviance non-decrease (rare but happens near boundary).
        # Use the weighted deviance when ``weights=`` was supplied so the
        # scalar tracks the actual log-likelihood we're optimising.
        new_dev = _poisson_deviance(y, mu_new, obs_weights)
        halvings = 0
        while new_dev > deviance and halvings < 10 and np.isfinite(deviance):
            eta_new = 0.5 * (eta_new + eta)
            np.clip(eta_new, -30.0, 30.0, out=eta_new)
            mu_new = np.exp(eta_new)
            new_dev = _poisson_deviance(y, mu_new, obs_weights)
            halvings += 1

        rel = abs(new_dev - deviance) / max(1.0, abs(new_dev))
        eta = eta_new
        mu = mu_new
        deviance = new_dev
        iters_used = it + 1
        if rel < tol:
            converged = True
            break

    # ----- Variance-covariance -----
    # IID:  Var(beta) = sigma^2 * (X_tilde' W X_tilde)^-1 with sigma^2 = 1
    # for canonical Poisson. We use the more conservative HC1 by default.
    # (X_tilde, w from the final iter are still in scope.)
    XtWX = X_tilde.T @ (X_tilde * w[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"vcov inversion failed: {exc}") from exc

    # fixest / pyfixest small-sample correction: scale by n/(n-p-Σ(G_k-1))
    # so that absorbed FE coefficients are charged against degrees of
    # freedom. Matches ``ssc(adj=TRUE)`` in fixest, which is the default.
    fe_dof = sum(int(c.size) - 1 for c in counts_list)
    df_resid = n - p - fe_dof

    if vcov == "iid":
        # Poisson canonical: dispersion = 1, but apply the fixest-style
        # SSC adjustment so SEs line up with pyfixest / fixest defaults.
        vcov_mat = XtWX_inv
        if df_resid > 0:
            vcov_mat = vcov_mat * (n / df_resid)
    elif vcov == "hc1":
        # HC1 sandwich: meat = Σ_i s_i s_i' where s_i is the score row.
        # Score = X_tilde * (y - mu).
        u = (y - mu)[:, None] * X_tilde
        meat = u.T @ u
        vcov_mat = XtWX_inv @ meat @ XtWX_inv
        if df_resid > 0:
            vcov_mat = vcov_mat * (n / df_resid)
    else:  # cr1 cluster-robust
        # Poisson cluster meat: M = Σ_g (Σ_i w_i x̃_i (y_i - μ_i))²
        # where w_i are the **observation** weights (frequency / survey),
        # not the IRLS working weights μ. For unweighted MLE w_i ≡ 1.
        # bread = (X̃' diag(μ * w) X̃)^{-1} (already in XtWX_inv).
        # crve's score formula is `(u * weights)[:,None] * X` so passing
        # weights=obs_weights produces obs_weights * (y - μ) * X̃ —
        # exactly the weighted Poisson score row.
        from .inference import crve as _crve
        score_weights = (
            obs_weights if obs_weights is not None
            else np.ones(n, dtype=np.float64)
        )
        vcov_mat = _crve(
            X_tilde, y - mu, cluster_arr_full,
            weights=score_weights,
            bread=XtWX_inv,
            type="cr1",
            extra_df=fe_dof,
        )

    log_lik = float(np.sum(obs_weights * (
        y * np.log(np.maximum(mu, 1e-30)) - mu
    )))

    return FePoisResult(
        formula=formula,
        coef_names=coef_names_full,
        coef_vec=beta,
        vcov_matrix=vcov_mat,
        n_obs=n_obs,
        n_kept=n_kept,
        n_dropped_singletons=n_dropped_singletons,
        n_dropped_separation=n_dropped_separation,
        deviance=deviance,
        log_likelihood=log_lik,
        iterations=iters_used,
        converged=converged,
        fe_names=list(fe_terms),
        fe_cardinality=[int(np.unique(c).size) for c in fe_codes],
        vcov_type=vcov,
        df_residual=int(max(df_resid, 0)),
    )


def _poisson_deviance(
    y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None,
) -> float:
    """Poisson deviance: 2 * Σ w_i [ y log(y/mu) - (y - mu) ] (y log y → 0 at y=0).

    ``weights`` defaults to 1; passing a per-obs weight vector returns the
    weighted deviance so step-halving tracks the weighted log-likelihood
    we're actually optimising under ``weights=``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-30)), 0.0)
    contributions = ratio - (y - mu)
    if weights is None:
        return float(2.0 * np.sum(contributions))
    return float(2.0 * np.sum(weights * contributions))


__all__ = ["fepois", "FePoisResult"]
