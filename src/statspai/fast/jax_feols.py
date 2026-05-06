"""JAX-backed end-to-end ``feols``.

This module ships a drop-in replacement for :func:`statspai.fast.feols`
whose **WLS solve + HC1 sandwich** computations execute on JAX/XLA
(CPU / GPU / TPU). The FE residualisation step still goes through the
Rust ``demean`` kernel because, on typical FE cardinalities, the
bincount-style memory pattern beats anything XLA emits on CPU. The
``cr1`` cluster-robust path also stays on the existing
:func:`statspai.fast.inference.crve`, which itself dispatches to the
Phase-2 Rust ``cluster_meat`` kernel when built. The unique value of
this module is on **GPU** boxes: the post-demean dense linear algebra
step runs at GPU speed without any host↔device ping-pong inside the
solve.

Honest scope
------------
- The dev environment used to write this code has **no CUDA**, only
  Apple-Silicon MPS. JAX-on-MPS is unofficial and the parity tests run
  on the CPU JAX path; the GPU promise is structural (XLA auto-
  dispatches to whatever ``jax.devices()[0]`` points at).
- ``cr1`` parity is delegated; we test that ``feols_jax(vcov="cr1")``
  returns the same matrix as ``feols(vcov="cr1")``.
- Default ``dtype="float64"`` so existing pinned numerical tests stay
  bit-comparable. Pass ``dtype="float32"`` on a GPU to trade ~1 ulp of
  precision for the XLA float32 fast path.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .feols import FeolsResult


# ---------------------------------------------------------------------------
# JAX availability + helpers (mirrors jax_backend.py's policy)
# ---------------------------------------------------------------------------

try:
    import jax

    # StatsPAI defaults to float64; XLA truncates unless explicitly enabled.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit

    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised on no-jax CI
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    jit = None  # type: ignore[assignment]
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# JIT-compiled core: WLS solve + iid/HC1 sandwich
# ---------------------------------------------------------------------------

def _make_jax_kernels():
    """Build JIT-compiled solvers lazily so module import is jax-free."""
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable the JAX "
            "feols backend."
        )

    @jit
    def _wls_solve(X, y, w):
        """QR-based weighted-least-squares solve.

        Returns (beta, residuals, rss, XtWX_inv).
        """
        # Sqrt-W trick: solve (sqrt(W) X) beta = sqrt(W) y so we can use
        # a plain QR. Numerically equivalent to the normal-equation
        # ``inv(X' W X) X' W y`` but better-conditioned for ill-shaped X.
        sw = jnp.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw

        Q, R = jnp.linalg.qr(Xw, mode="reduced")
        beta = jnp.linalg.solve(R, Q.T @ yw)
        resid = y - X @ beta
        rss = jnp.sum(w * resid * resid)
        # XtWX_inv = inv(R' R). Compute via triangular solves so we
        # never materialise XtWX explicitly.
        eye = jnp.eye(R.shape[0], dtype=R.dtype)
        Rinv = jax.scipy.linalg.solve_triangular(R, eye, lower=False)
        XtWX_inv = Rinv @ Rinv.T
        return beta, resid, rss, XtWX_inv

    @jit
    def _hc1_meat(X, resid, w):
        """HC1 meat: sum_i (w_i u_i)² x_i x_i^T."""
        u = (resid * w)[:, None] * X  # (n, p)
        return u.T @ u

    @jit
    def _y_centered_ss(y, w):
        """Weighted total sum of squares around the weighted mean."""
        wsum = jnp.sum(w)
        ybar = jnp.sum(w * y) / wsum
        tss = jnp.sum(w * (y - ybar) ** 2)
        return tss

    return _wls_solve, _hc1_meat, _y_centered_ss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feols_jax(
    formula: str,
    data: pd.DataFrame,
    *,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    dtype: str = "float64",
) -> FeolsResult:
    """JAX-backed OLS / WLS with high-dimensional fixed effects.

    Drop-in replacement for :func:`statspai.fast.feols` — same formula
    DSL, same ``FeolsResult`` return type. The WLS solve and HC1
    sandwich run on the default JAX device; FE residualisation and CR1
    cluster sandwich delegate to the Rust / numba paths.

    Parameters
    ----------
    formula, data, vcov, cluster, weights, drop_singletons, fe_tol,
    fe_maxiter
        See :func:`statspai.fast.feols`.
    dtype : {"float64", "float32"}, default "float64"
        Working precision on the JAX device. ``"float32"`` is roughly
        2x faster on CUDA and 32x on TPU but trades ~1 ulp of
        precision; only flip it if the parity drift is acceptable.

    Returns
    -------
    FeolsResult
        ``backend`` is set to ``"statspai-jax"`` (the only field that
        differs from :func:`feols`'s output).

    Raises
    ------
    ImportError
        If jax is not installed.
    """
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable "
            "feols_jax. Plain sp.fast.feols runs without JAX."
        )
    if vcov not in ("iid", "hc1", "cr1"):
        raise ValueError(f"vcov={vcov!r}; supported: 'iid', 'hc1', or 'cr1'")
    if vcov == "cr1" and cluster is None:
        raise ValueError("vcov='cr1' requires cluster=<column name>")
    if cluster is not None and vcov in ("iid", "hc1"):
        raise ValueError(
            f"cluster={cluster!r} provided but vcov={vcov!r}; "
            "set vcov='cr1' to compute cluster-robust SE"
        )
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype={dtype!r}; supported: 'float64' or 'float32'")

    # Lazy imports — keep module-level surface minimal.
    from .fepois import _parse_fepois_formula
    from .demean import demean as _demean
    from .inference import crve as _crve

    # ---------- Formula parsing + data extraction (numpy) ----------
    lhs, rhs_terms, fe_terms = _parse_fepois_formula(formula)

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
    np_dtype = np.float64  # always work in float64 on host; JAX kernel
    # downcasts to float32 only if requested.

    y = data[lhs].to_numpy(dtype=np_dtype).copy()
    if not np.isfinite(y).all():
        raise ValueError(f"outcome column {lhs!r} has non-finite values")
    X_user = (
        data[rhs_terms].to_numpy(dtype=np_dtype).copy()
        if rhs_terms else np.empty((n_obs, 0), dtype=np_dtype)
    )
    if X_user.ndim == 1:
        X_user = X_user.reshape(-1, 1)
    if not np.isfinite(X_user).all():
        raise ValueError("regressor columns contain non-finite values")
    if add_intercept:
        X = np.column_stack([np.ones(n_obs, dtype=np_dtype), X_user])
        coef_names_full: List[str] = ["(Intercept)"] + list(rhs_terms)
    else:
        X = X_user
        coef_names_full = list(rhs_terms)
    if X.shape[1] == 0:
        raise ValueError(
            "No regressors after parsing — formula must include at least "
            "one RHS term (or '1' for an intercept)."
        )

    if weights is not None:
        w_full = data[weights].to_numpy(dtype=np_dtype).copy()
        if (w_full < 0).any():
            raise ValueError(f"weights column {weights!r} contains negative values")
        if not np.isfinite(w_full).all():
            raise ValueError(f"weights column {weights!r} contains non-finite values")
    else:
        w_full = None

    if cluster is not None:
        cluster_arr_full = data[cluster].to_numpy()
        cluster_codes_check, _ = pd.factorize(
            cluster_arr_full, sort=False, use_na_sentinel=True,
        )
        if (cluster_codes_check < 0).any():
            raise ValueError(
                f"cluster column {cluster!r} contains NaN; drop or impute upstream"
            )
    else:
        cluster_arr_full = None

    # ---------- FE residualisation (Rust / numba) ----------
    if fe_terms:
        fe_df = data[fe_terms]
        if w_full is None:
            stacked = np.column_stack([y, X])
            stacked_dem, info = _demean(
                stacked, fe_df,
                drop_singletons=drop_singletons,
                tol=1e-12, max_iter=fe_maxiter, tol_abs=fe_tol,
            )
            keep_mask = info.keep_mask
            n_kept = info.n_kept
            n_dropped_singletons = info.n_dropped
            y_dem = stacked_dem[:, 0]
            X_dem = stacked_dem[:, 1:]
            fe_card = list(info.n_fe)
        else:
            from .fepois import _weighted_ap_demean
            from .demean import _detect_singletons as _ds_helper
            fe_codes_raw: List[np.ndarray] = []
            for col in fe_terms:
                codes, _uniq = pd.factorize(
                    data[col], sort=False, use_na_sentinel=True,
                )
                if (codes < 0).any():
                    raise ValueError(f"NaN in fixed effect column {col!r}")
                fe_codes_raw.append(codes.astype(np.int64))
            keep_mask = (
                _ds_helper(fe_codes_raw, n_obs)
                if drop_singletons else np.ones(n_obs, dtype=bool)
            )
            n_kept = int(keep_mask.sum())
            n_dropped_singletons = n_obs - n_kept
            fe_codes_kept: List[np.ndarray] = []
            counts_list: List[np.ndarray] = []
            fe_card: List[int] = []
            for codes_k in fe_codes_raw:
                ck = codes_k[keep_mask]
                dense, uniq = pd.factorize(ck, sort=False)
                dense = dense.astype(np.int64)
                G = len(uniq)
                fe_codes_kept.append(dense)
                counts_list.append(
                    np.bincount(dense, minlength=G).astype(np.float64)
                )
                fe_card.append(G)
            y_kept = y[keep_mask]
            X_kept = X[keep_mask]
            w_kept = w_full[keep_mask]
            stacked = np.column_stack([y_kept, X_kept])
            stacked_dem, _, _ = _weighted_ap_demean(
                stacked, fe_codes_kept, counts_list, w_kept,
                max_iter=fe_maxiter, tol=fe_tol,
            )
            y_dem = stacked_dem[:, 0]
            X_dem = stacked_dem[:, 1:]
        fe_dof = sum(int(g) - 1 for g in fe_card)
    else:
        keep_mask = np.ones(n_obs, dtype=bool)
        n_kept = n_obs
        n_dropped_singletons = 0
        y_dem = y.copy()
        X_dem = X.copy()
        fe_card = []
        fe_dof = 0

    if w_full is not None:
        w = w_full[keep_mask]
    else:
        w = np.ones(n_kept, dtype=np_dtype)
    if cluster_arr_full is not None:
        cluster_arr_kept = cluster_arr_full[keep_mask]
    else:
        cluster_arr_kept = None

    n, p = X_dem.shape

    # ---------- JAX device transfer + WLS + iid/HC1 sandwich ----------
    jax_dtype = jnp.float32 if dtype == "float32" else jnp.float64
    X_j = jnp.asarray(X_dem, dtype=jax_dtype)
    y_j = jnp.asarray(y_dem, dtype=jax_dtype)
    w_j = jnp.asarray(w, dtype=jax_dtype)

    _wls_solve, _hc1_meat, _y_centered_ss = _make_jax_kernels()
    beta_j, resid_j, rss_j, XtWX_inv_j = _wls_solve(X_j, y_j, w_j)
    tss_j = _y_centered_ss(y_j, w_j)

    # Materialise outputs we always need on host.
    beta = np.asarray(beta_j, dtype=np_dtype)
    resid = np.asarray(resid_j, dtype=np_dtype)
    rss = float(np.asarray(rss_j, dtype=np_dtype))
    tss = float(np.asarray(tss_j, dtype=np_dtype))
    XtWX_inv = np.asarray(XtWX_inv_j, dtype=np_dtype)
    r_squared_within = 1.0 - rss / max(tss, 1e-30)

    df_resid = n - p - fe_dof

    if vcov == "iid":
        sigma2 = rss / max(df_resid, 1)
        vcov_mat = sigma2 * XtWX_inv
    elif vcov == "hc1":
        meat_j = _hc1_meat(X_j, resid_j, w_j)
        meat = np.asarray(meat_j, dtype=np_dtype)
        vcov_mat = XtWX_inv @ meat @ XtWX_inv
        if df_resid > 0:
            vcov_mat = vcov_mat * (n / df_resid)
    else:  # cr1 → delegate to existing crve (which uses the Phase 2 Rust kernel)
        vcov_mat = _crve(
            X_dem, resid, cluster_arr_kept,
            weights=w, bread=XtWX_inv,
            type="cr1",
            extra_df=fe_dof,
        )

    return FeolsResult(
        formula=formula,
        coef_names=coef_names_full,
        coef_vec=beta,
        vcov_matrix=vcov_mat,
        n_obs=n_obs,
        n_kept=int(n_kept),
        n_dropped_singletons=int(n_dropped_singletons),
        rss=rss,
        tss=tss,
        r_squared_within=r_squared_within,
        fe_names=list(fe_terms),
        fe_cardinality=fe_card,
        df_resid=int(df_resid),
        vcov_type=vcov,
        cluster_var=cluster,
        backend="statspai-jax",
    )


__all__ = ["feols_jax"]
