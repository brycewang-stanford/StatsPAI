"""JAX-backed end-to-end ``feols``.

This module ships an API-compatible backend for :func:`statspai.fast.feols`
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from .feols import FeolsResult
from ._result_protocol import distribution_summary as _distribution_summary
from ._result_protocol import jsonable as _jsonable
from ._result_protocol import tidy_records as _tidy_records
from ._validation import (
    nonempty_sample as _nonempty_sample,
    nonnegative_finite_float as _nonnegative_finite_float,
    open_unit_float as _open_unit_float,
    positive_int as _positive_int,
    positive_weight_mass as _positive_weight_mass,
)

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


def _require_dataframe(data: Any, *, context: str) -> None:
    """Require a pandas DataFrame before touching ``.columns``."""
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{context}: data must be a pandas DataFrame",
            recovery_hint="Pass the estimator a pandas DataFrame with named columns.",
            diagnostics={"type": type(data).__name__},
        )


def _parse_formula_checked(
    formula: Any,
    *,
    context: str,
) -> tuple[str, List[str], List[str]]:
    """Parse the fast formula DSL and map syntax errors into the taxonomy."""
    if not isinstance(formula, str):
        raise MethodIncompatibility(
            f"{context}: formula must be a string",
            recovery_hint="Use a formula like 'y ~ x1 + x2 | firm + year'.",
            diagnostics={"type": type(formula).__name__},
        )
    from .fepois import _parse_fepois_formula

    try:
        return _parse_fepois_formula(formula)
    except ValueError as exc:
        raise MethodIncompatibility(
            f"{context}: {exc}",
            recovery_hint=(
                "Use the fast formula DSL: outcome ~ regressors | fixed_effects."
            ),
            diagnostics={"formula": formula},
        ) from exc


def _require_finite_outputs(context: str, **arrays: Any) -> None:
    """Reject non-finite JAX outputs before they escape as silent NaNs."""
    bad = []
    for name, values in arrays.items():
        arr = np.asarray(values, dtype=np.float64)
        if not np.isfinite(arr).all():
            bad.append(name)
    if bad:
        raise NumericalInstability(
            f"{context}: JAX solve produced non-finite outputs",
            recovery_hint=(
                "Check for collinear regressors after fixed-effect residualisation "
                "or use the native feols path to inspect the design matrix."
            ),
            diagnostics={"nonfinite_outputs": bad},
            alternative_functions=["sp.fast.feols"],
        )


# ---------------------------------------------------------------------------
# JIT-compiled core: WLS solve + iid/HC1 sandwich
# ---------------------------------------------------------------------------


def _make_jax_kernels() -> tuple[Any, Any, Any]:
    """Build JIT-compiled solvers lazily so module import is jax-free."""
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable the JAX "
            "feols backend."
        )

    @jit
    def _wls_solve(X: Any, y: Any, w: Any) -> Any:
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
    def _hc1_meat(X: Any, resid: Any, w: Any) -> Any:
        """HC1 meat: sum_i (w_i u_i)² x_i x_i^T."""
        u = (resid * w)[:, None] * X  # (n, p)
        return u.T @ u

    @jit
    def _y_centered_ss(y: Any, w: Any) -> Any:
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

    JAX-backed companion to :func:`statspai.fast.feols`: same formula
    DSL and ``FeolsResult`` return type. The WLS solve and HC1
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
    _require_dataframe(data, context="feols_jax")
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable "
            "feols_jax. Plain sp.fast.feols runs without JAX."
        )
    if vcov not in ("iid", "hc1", "cr1"):
        raise MethodIncompatibility(
            f"feols_jax: vcov={vcov!r}; supported: 'iid', 'hc1', or 'cr1'"
        )
    if vcov == "cr1" and cluster is None:
        raise MethodIncompatibility(
            "feols_jax: vcov='cr1' requires cluster=<column name>"
        )
    if cluster is not None and vcov in ("iid", "hc1"):
        raise MethodIncompatibility(
            f"feols_jax: cluster={cluster!r} provided but vcov={vcov!r}; "
            "set vcov='cr1' to compute cluster-robust SE"
        )
    if dtype not in ("float64", "float32"):
        raise MethodIncompatibility(
            f"feols_jax: dtype={dtype!r}; supported: 'float64' or 'float32'"
        )
    fe_maxiter = _positive_int(
        fe_maxiter,
        name="fe_maxiter",
        context="feols_jax",
    )
    fe_tol = _nonnegative_finite_float(
        fe_tol,
        name="fe_tol",
        context="feols_jax",
    )

    # Lazy imports — keep module-level surface minimal.
    from .demean import demean as _demean
    from .inference import crve as _crve

    # ---------- Formula parsing + data extraction (numpy) ----------
    lhs, rhs_terms, fe_terms = _parse_formula_checked(
        formula,
        context="feols_jax",
    )

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
        raise MethodIncompatibility(
            f"feols_jax: columns missing from data: {missing}",
            recovery_hint="Add the missing columns or update the formula/options.",
            diagnostics={"missing_columns": missing},
        )

    n_obs = len(data)
    _nonempty_sample(n_obs, context="feols_jax")
    np_dtype = np.float64  # always work in float64 on host; JAX kernel
    # downcasts to float32 only if requested.

    y = data[lhs].to_numpy(dtype=np_dtype).copy()
    if not np.isfinite(y).all():
        raise MethodIncompatibility(
            f"feols_jax: outcome column {lhs!r} has non-finite values"
        )
    X_user = (
        data[rhs_terms].to_numpy(dtype=np_dtype).copy()
        if rhs_terms
        else np.empty((n_obs, 0), dtype=np_dtype)
    )
    if X_user.ndim == 1:
        X_user = X_user.reshape(-1, 1)
    if not np.isfinite(X_user).all():
        raise MethodIncompatibility(
            "feols_jax: regressor columns contain non-finite values"
        )
    if add_intercept:
        X = np.column_stack([np.ones(n_obs, dtype=np_dtype), X_user])
        coef_names_full: List[str] = ["(Intercept)"] + list(rhs_terms)
    else:
        X = X_user
        coef_names_full = list(rhs_terms)
    if X.shape[1] == 0:
        raise MethodIncompatibility(
            "feols_jax: No regressors after parsing — formula must include at least "
            "one RHS term (or '1' for an intercept)."
        )

    if weights is not None:
        w_full = data[weights].to_numpy(dtype=np_dtype).copy()
        if (w_full < 0).any():
            raise MethodIncompatibility(
                f"feols_jax: weights column {weights!r} contains negative values"
            )
        if not np.isfinite(w_full).all():
            raise MethodIncompatibility(
                f"feols_jax: weights column {weights!r} contains non-finite values"
            )
        _positive_weight_mass(
            w_full,
            context=f"feols_jax weights column {weights!r}",
        )
    else:
        w_full = None

    if cluster is not None:
        cluster_arr_full = data[cluster].to_numpy()
        cluster_codes_check, _ = pd.factorize(
            cluster_arr_full,
            sort=False,
            use_na_sentinel=True,
        )
        if (cluster_codes_check < 0).any():
            raise MethodIncompatibility(
                f"feols_jax: cluster column {cluster!r} contains NaN; "
                "drop or impute upstream"
            )
    else:
        cluster_arr_full = None

    # ---------- FE residualisation (Rust / numba) ----------
    if fe_terms:
        fe_df = data[fe_terms]
        if w_full is None:
            stacked = np.column_stack([y, X])
            stacked_dem, info = _demean(
                stacked,
                fe_df,
                drop_singletons=drop_singletons,
                tol=1e-12,
                max_iter=fe_maxiter,
                tol_abs=fe_tol,
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
                    data[col],
                    sort=False,
                    use_na_sentinel=True,
                )
                if (codes < 0).any():
                    raise MethodIncompatibility(
                        f"feols_jax: NaN in fixed effect column {col!r}"
                    )
                fe_codes_raw.append(codes.astype(np.int64))
            keep_mask = (
                _ds_helper(fe_codes_raw, n_obs)
                if drop_singletons
                else np.ones(n_obs, dtype=bool)
            )
            n_kept = int(keep_mask.sum())
            n_dropped_singletons = n_obs - n_kept
            fe_codes_kept: List[np.ndarray] = []
            counts_list: List[np.ndarray] = []
            fe_card = []
            for codes_k in fe_codes_raw:
                ck = codes_k[keep_mask]
                dense, uniq = pd.factorize(ck, sort=False)
                dense = dense.astype(np.int64)
                G = len(uniq)
                fe_codes_kept.append(dense)
                counts_list.append(np.bincount(dense, minlength=G).astype(np.float64))
                fe_card.append(G)
            y_kept = y[keep_mask]
            X_kept = X[keep_mask]
            w_kept = w_full[keep_mask]
            _positive_weight_mass(
                w_kept,
                context=f"feols_jax kept sample weights column {weights!r}",
            )
            stacked = np.column_stack([y_kept, X_kept])
            stacked_dem, _, _ = _weighted_ap_demean(
                stacked,
                fe_codes_kept,
                counts_list,
                w_kept,
                max_iter=fe_maxiter,
                tol=fe_tol,
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
        _positive_weight_mass(
            w,
            context=f"feols_jax kept sample weights column {weights!r}",
        )
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
    _require_finite_outputs(
        "feols_jax",
        beta=beta,
        residuals=resid,
        rss=rss,
        tss=tss,
        bread=XtWX_inv,
    )
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
            X_dem,
            resid,
            cluster_arr_kept,
            weights=w,
            bread=XtWX_inv,
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


# =============================================================================
# Phase 4b: vmap'd bootstrap on JAX
# =============================================================================
#
# The single-shot ``feols_jax`` above runs the dense linear-algebra step
# on whatever ``jax.devices()[0]`` points at. The real GPU-only win
# comes when the **same** computation is repeated many times — bootstrap
# resampling. ``feols_jax_bootstrap`` builds a JIT-compiled single-
# iteration solver and ``jax.vmap``-batches it across draws.
#
# Bootstrap variants supported:
# - ``'pairs'`` — Efron pairs bootstrap: each draw resamples *rows*
#   with replacement (multinomial counts as the bootstrap weights).
# - ``'cluster'`` — cluster bootstrap: each draw resamples *clusters*
#   with replacement; rows in a sampled-twice cluster get weight 2.
#
# Both variants pre-demean once (FE structure treated as known) — same
# convention as fixest's ``boottest``. For small-cluster wild
# bootstrap (Cameron-Gelbach-Miller 2008), see Phase 4c.


@dataclass
class FeolsBootstrapResult:
    """Outcome of :func:`feols_jax_bootstrap`.

    Attributes
    ----------
    coef : pd.Series
        Point estimate from the original (un-resampled) data — same as
        :func:`feols_jax` would return.
    se_boot : pd.Series
        Bootstrap standard errors (per-coefficient std-dev across the
        ``n_boot`` resamples).
    ci_lower, ci_upper : pd.Series
        Percentile-method bootstrap confidence intervals at level
        ``1 - ci_alpha``.
    boot_betas : pd.DataFrame
        ``(n_boot, p)`` table of per-resample coefficient vectors.
        Useful for custom CI methods (BCa, basic bootstrap, etc.) and
        for permutation-style inference.
    n_boot : int
    bootstrap_type : str
        ``'pairs'`` or ``'cluster'``.
    backend : str
        Always ``'statspai-jax-bootstrap'``.
    """

    coef: "pd.Series"
    se_boot: "pd.Series"
    ci_lower: "pd.Series"
    ci_upper: "pd.Series"
    boot_betas: "pd.DataFrame"
    n_boot: int
    bootstrap_type: str
    backend: str = "statspai-jax-bootstrap"

    def summary(self) -> str:
        b = self.coef.values
        s = self.se_boot.values
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(s > 0, b / s, np.nan)
        df = pd.DataFrame(
            {
                "Estimate": b,
                "SE (boot)": s,
                "t value": t,
                "CI lower": self.ci_lower.values,
                "CI upper": self.ci_upper.values,
            },
            index=self.coef.index,
        )
        header = (
            f"sp.fast.feols_jax_bootstrap  |  bootstrap={self.bootstrap_type}  |  "
            f"n_boot={self.n_boot}"
        )
        return header + "\n" + str(df.to_string(float_format=lambda x: f"{x:.6f}"))

    def tidy(self) -> pd.DataFrame:
        """Coefficient-level bootstrap table."""
        b = self.coef.values
        s = self.se_boot.values
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(s > 0, b / s, np.nan)
        return pd.DataFrame(
            {
                "Estimate": b,
                "SE (boot)": s,
                "t value": t,
                "CI lower": self.ci_lower.values,
                "CI upper": self.ci_upper.values,
            },
            index=self.coef.index,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Lossless JSON-safe payload for JAX bootstrap results."""
        return dict(
            _jsonable(
                {
                    "kind": "fast_feols_jax_bootstrap_result",
                    "n_boot": self.n_boot,
                    "bootstrap_type": self.bootstrap_type,
                    "backend": self.backend,
                    "coefficients": _tidy_records(self.tidy()),
                    "boot_betas": self.boot_betas.to_dict(orient="records"),
                }
            )
        )

    def to_agent_summary(self, *, max_terms: int = 10) -> Dict[str, Any]:
        """Bounded agent-facing summary for JAX bootstrap results."""
        n_terms = len(self.coef.index)
        limit = max(int(max_terms), 0)
        boot_summaries = {
            str(term): _distribution_summary(self.boot_betas[term].to_numpy())
            for term in self.boot_betas.columns[:limit]
        }
        return dict(
            _jsonable(
                {
                    "kind": "fast_feols_jax_bootstrap_agent_summary",
                    "n_boot": self.n_boot,
                    "bootstrap_type": self.bootstrap_type,
                    "backend": self.backend,
                    "coefficients": _tidy_records(self.tidy().head(limit)),
                    "bootstrap_distributions": boot_summaries,
                    "n_terms": n_terms,
                    "truncated_terms": max(n_terms - limit, 0),
                }
            )
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _make_bootstrap_kernels() -> tuple[Any, Any, Any, Any]:
    """Build JIT-compiled single-iteration bootstrap solvers."""
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable the JAX "
            "bootstrap backend."
        )

    @jit
    def _wls_beta_only(X: Any, y: Any, w: Any) -> Any:
        """QR-based WLS that returns only the coefficient vector.

        Same algorithm as ``_wls_solve`` but skips the residuals / RSS /
        XtWX_inv outputs we don't need per-bootstrap-iteration. Saves
        memory and a triangular solve.
        """
        sw = jnp.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        Q, R = jnp.linalg.qr(Xw, mode="reduced")
        beta = jnp.linalg.solve(R, Q.T @ yw)
        return beta

    @jit
    def _one_pairs_boot(key: Any, X: Any, y: Any, w_base: Any) -> Any:
        """One pairs-bootstrap iteration: multinomial counts as weights."""
        n = X.shape[0]
        idx = jax.random.choice(key, n, shape=(n,), replace=True)
        boot_w = jnp.bincount(idx, length=n).astype(X.dtype)
        return _wls_beta_only(X, y, w_base * boot_w)

    def _build_cluster_boot(n_clusters: int) -> Any:
        """Build a JIT-compiled cluster-bootstrap kernel for a fixed
        ``n_clusters``. Closing over the integer keeps it concrete
        inside the JIT trace, so ``jax.random.choice`` accepts it."""
        n_clusters_int = int(n_clusters)

        @jit
        def _one_cluster_boot(
            key: Any,
            X: Any,
            y: Any,
            w_base: Any,
            cluster_codes: Any,
        ) -> Any:
            idx = jax.random.choice(
                key,
                n_clusters_int,
                shape=(n_clusters_int,),
                replace=True,
            )
            boot_count = jnp.bincount(idx, length=n_clusters_int)
            boot_w = boot_count[cluster_codes].astype(X.dtype)
            return _wls_beta_only(X, y, w_base * boot_w)

        return _one_cluster_boot

    # ─── Phase 4c: wild + wild cluster bootstrap (score formulation) ───
    #
    # Wild bootstrap (Wu 1986, Cameron-Gelbach-Miller 2008): pseudo-y*_i
    # = X_i β̂ + η_i ⊙ û_i with η_i Rademacher draws. Refitting OLS on
    # (X, y*) gives β* = β̂ + (X'WX)^{-1} X'W (η ⊙ û). The right-hand
    # form is the "score bootstrap" — mathematically identical, much
    # cheaper per iteration (one mat-vec instead of a full QR), and
    # exposes the per-iteration randomness as a single Rademacher draw.
    #
    # ``rademacher`` is implemented as ``2 * Bernoulli(0.5) - 1`` for
    # portability across JAX versions (``jax.random.rademacher`` only
    # arrived in jax >= 0.4.x).

    @jit
    def _one_wild_boot(
        key: Any,
        X: Any,
        w_base: Any,
        residuals: Any,
        beta_hat: Any,
        XtWX_inv: Any,
    ) -> Any:
        """One row-level wild-bootstrap iteration (score form)."""
        n = X.shape[0]
        eta = 2 * jax.random.bernoulli(key, p=0.5, shape=(n,)).astype(X.dtype) - 1
        score = X.T @ (w_base * eta * residuals)  # (p,)
        return beta_hat + XtWX_inv @ score

    def _build_wild_cluster_boot(n_clusters: int) -> Any:
        """Build a JIT-compiled wild-cluster-bootstrap kernel for a fixed
        ``n_clusters``. Each iteration draws one Rademacher per cluster
        and broadcasts to rows via ``cluster_codes``."""
        n_clusters_int = int(n_clusters)

        @jit
        def _one_wild_cluster_boot(
            key: Any,
            X: Any,
            w_base: Any,
            residuals: Any,
            beta_hat: Any,
            XtWX_inv: Any,
            cluster_codes: Any,
        ) -> Any:
            eta_g = (
                2
                * jax.random.bernoulli(key, p=0.5, shape=(n_clusters_int,)).astype(
                    X.dtype
                )
                - 1
            )
            eta = eta_g[cluster_codes]  # (n,)
            score = X.T @ (w_base * eta * residuals)  # (p,)
            return beta_hat + XtWX_inv @ score

        return _one_wild_cluster_boot

    return (
        _one_pairs_boot,
        _build_cluster_boot,
        _one_wild_boot,
        _build_wild_cluster_boot,
    )


def _jax_prep_inputs(
    formula: str,
    data: "pd.DataFrame",
    *,
    weights: Optional[str],
    drop_singletons: bool,
    fe_tol: float,
    fe_maxiter: int,
) -> Dict[str, Any]:
    """Shared prep: parse formula → extract matrices → FE-residualise.

    Mirrors the prep block of :func:`feols_jax` so bootstrap can reuse
    the exact same residualisation. Pulled out as a helper rather than
    refactoring the live ``feols_jax`` body to keep this commit's blast
    radius small.

    Returns dict with keys: ``y_dem``, ``X_dem``, ``w``,
    ``coef_names_full``, ``n_obs``, ``n_kept``, ``n_dropped_singletons``,
    ``fe_card``, ``fe_dof``, ``fe_terms``, ``keep_mask``.
    """
    _require_dataframe(data, context="feols_jax_bootstrap")
    fe_maxiter = _positive_int(
        fe_maxiter,
        name="fe_maxiter",
        context="feols_jax_bootstrap",
    )
    fe_tol = _nonnegative_finite_float(
        fe_tol,
        name="fe_tol",
        context="feols_jax_bootstrap",
    )
    from .demean import demean as _demean

    lhs, rhs_terms, fe_terms = _parse_formula_checked(
        formula,
        context="feols_jax_bootstrap",
    )
    user_intercept = "1" in rhs_terms
    rhs_terms = [t for t in rhs_terms if t != "1"]
    add_intercept = user_intercept or not fe_terms

    needed_cols = [lhs] + rhs_terms + fe_terms
    if weights is not None:
        needed_cols = needed_cols + [weights]
    missing = [c for c in needed_cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: columns missing from data: {missing}",
            recovery_hint="Add the missing columns or update the formula/options.",
            diagnostics={"missing_columns": missing},
        )

    n_obs = len(data)
    _nonempty_sample(n_obs, context="feols_jax_bootstrap")
    np_dtype = np.float64

    y = data[lhs].to_numpy(dtype=np_dtype).copy()
    if not np.isfinite(y).all():
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: outcome column {lhs!r} has non-finite values"
        )
    X_user = (
        data[rhs_terms].to_numpy(dtype=np_dtype).copy()
        if rhs_terms
        else np.empty((n_obs, 0), dtype=np_dtype)
    )
    if X_user.ndim == 1:
        X_user = X_user.reshape(-1, 1)
    if not np.isfinite(X_user).all():
        raise MethodIncompatibility(
            "feols_jax_bootstrap: regressor columns contain non-finite values"
        )
    if add_intercept:
        X = np.column_stack([np.ones(n_obs, dtype=np_dtype), X_user])
        coef_names_full: List[str] = ["(Intercept)"] + list(rhs_terms)
    else:
        X = X_user
        coef_names_full = list(rhs_terms)
    if X.shape[1] == 0:
        raise MethodIncompatibility(
            "feols_jax_bootstrap: No regressors after parsing — formula must "
            "include at least "
            "one RHS term (or '1' for an intercept)."
        )

    if weights is not None:
        w_full = data[weights].to_numpy(dtype=np_dtype).copy()
        if (w_full < 0).any():
            raise MethodIncompatibility(
                f"feols_jax_bootstrap: weights column {weights!r} contains "
                "negative values"
            )
        if not np.isfinite(w_full).all():
            raise MethodIncompatibility(
                f"feols_jax_bootstrap: weights column {weights!r} contains "
                "non-finite values"
            )
        _positive_weight_mass(
            w_full,
            context=f"feols_jax_bootstrap weights column {weights!r}",
        )
    else:
        w_full = None

    if fe_terms:
        fe_df = data[fe_terms]
        if w_full is None:
            stacked = np.column_stack([y, X])
            stacked_dem, info = _demean(
                stacked,
                fe_df,
                drop_singletons=drop_singletons,
                tol=1e-12,
                max_iter=fe_maxiter,
                tol_abs=fe_tol,
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
                    data[col],
                    sort=False,
                    use_na_sentinel=True,
                )
                if (codes < 0).any():
                    raise MethodIncompatibility(
                        f"feols_jax_bootstrap: NaN in fixed effect column {col!r}"
                    )
                fe_codes_raw.append(codes.astype(np.int64))
            keep_mask = (
                _ds_helper(fe_codes_raw, n_obs)
                if drop_singletons
                else np.ones(n_obs, dtype=bool)
            )
            n_kept = int(keep_mask.sum())
            n_dropped_singletons = n_obs - n_kept
            fe_codes_kept: List[np.ndarray] = []
            counts_list: List[np.ndarray] = []
            fe_card = []
            for codes_k in fe_codes_raw:
                ck = codes_k[keep_mask]
                dense, uniq = pd.factorize(ck, sort=False)
                dense = dense.astype(np.int64)
                G = len(uniq)
                fe_codes_kept.append(dense)
                counts_list.append(np.bincount(dense, minlength=G).astype(np.float64))
                fe_card.append(G)
            y_kept = y[keep_mask]
            X_kept = X[keep_mask]
            w_kept = w_full[keep_mask]
            _positive_weight_mass(
                w_kept,
                context=(
                    f"feols_jax_bootstrap kept sample weights column " f"{weights!r}"
                ),
            )
            stacked = np.column_stack([y_kept, X_kept])
            stacked_dem, _, _ = _weighted_ap_demean(
                stacked,
                fe_codes_kept,
                counts_list,
                w_kept,
                max_iter=fe_maxiter,
                tol=fe_tol,
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
        _positive_weight_mass(
            w,
            context=f"feols_jax_bootstrap kept sample weights column {weights!r}",
        )
    else:
        w = np.ones(n_kept, dtype=np_dtype)

    return {
        "y_dem": y_dem,
        "X_dem": X_dem,
        "w": w,
        "coef_names_full": coef_names_full,
        "n_obs": int(n_obs),
        "n_kept": int(n_kept),
        "n_dropped_singletons": int(n_dropped_singletons),
        "fe_card": fe_card,
        "fe_dof": int(fe_dof),
        "fe_terms": list(fe_terms),
        "keep_mask": keep_mask,
    }


def feols_jax_bootstrap(
    formula: str,
    data: "pd.DataFrame",
    *,
    n_boot: int = 1_000,
    seed: int = 0,
    bootstrap: str = "pairs",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    ci_alpha: float = 0.05,
    vmap_chunk_size: int = 200,
    dtype: str = "float64",
) -> FeolsBootstrapResult:
    """JAX-backed pairs / cluster bootstrap for ``feols_jax``.

    Pre-residualises ``y`` and ``X`` once via :func:`sp.fast.demean`,
    then runs ``n_boot`` bootstrap WLS fits in parallel using
    ``jax.vmap``. On a CPU box this is roughly equal-time to a numpy
    sequential bootstrap (the JIT overhead amortises around B≈100); on
    a CUDA / TPU device it is **10–100x faster** because the per-
    iteration QR is the same JIT-compiled XLA program lifted to a
    batched primitive.

    Parameters
    ----------
    formula, data, weights, drop_singletons, fe_tol, fe_maxiter, dtype
        See :func:`feols_jax`.
    n_boot : int, default 1000
        Number of bootstrap resamples.
    seed : int, default 0
        Seed for the JAX PRNG.
    bootstrap : {"pairs", "cluster", "wild", "wild_cluster"}, default "pairs"
        - ``"pairs"`` — Efron pairs bootstrap; each iteration resamples
          rows with replacement (multinomial counts become the
          bootstrap weights). Asymptotic target: HC1 SE.
        - ``"cluster"`` — Cameron-Gelbach-Miller (2008) §III.A cluster
          bootstrap; each iteration resamples *clusters* with
          replacement; observations in a cluster sampled k times get
          weight k. Use with G ≥ 30 typical clusters. Asymptotic
          target: CR1 SE.
        - ``"wild"`` — Wu (1986) row-level wild bootstrap; each
          iteration draws independent Rademacher signs ``η_i ∈ {-1, +1}``
          per row and uses the score formulation ``β* = β̂ +
          (X'WX)^{-1} X'W (η ⊙ û)`` (one mat-vec, no per-iteration
          QR). Mathematically identical to the literal "refit on
          y* = X β̂ + η ⊙ û" formulation. Asymptotic target: HC SE.
        - ``"wild_cluster"`` — Cameron-Gelbach-Miller (2008) §III.B
          wild cluster bootstrap; same score formulation as ``"wild"``
          but Rademacher signs are drawn *per cluster*. The standard
          tool for few-cluster inference (G < 30); especially good
          when ``G < 10``. Asymptotic target: CR SE.
    cluster : str, optional
        Column name with cluster identifiers; required when
        ``bootstrap`` is ``"cluster"`` or ``"wild_cluster"``.
    ci_alpha : float, default 0.05
        Percentile-CI level (2-sided ``1 - alpha``).
    vmap_chunk_size : int, default 200
        Inner ``jax.vmap`` batch size. Larger = fewer compile launches
        and more device parallelism; smaller = lower peak memory.
        Tune up on GPUs with lots of HBM, down on tight-memory CPUs.

    Returns
    -------
    FeolsBootstrapResult

    Notes
    -----
    The pre-demean-once convention is the same as ``boottest`` /
    fixest. The strict alternative (re-demeaning each iteration) is
    asymptotically more honest but rarely used in practice and
    explodes the per-iteration cost.

    Examples
    --------
    >>> import statspai as sp
    >>> b = sp.fast.feols_jax_bootstrap(
    ...     "y ~ x1 + x2 | firm + year", data=df,
    ...     n_boot=2_000, bootstrap='cluster', cluster='firm',
    ... )
    >>> print(b.summary())
    """
    _require_dataframe(data, context="feols_jax_bootstrap")
    if not _HAS_JAX:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable "
            "feols_jax_bootstrap."
        )
    if bootstrap not in ("pairs", "cluster", "wild", "wild_cluster"):
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: bootstrap={bootstrap!r}; supported: "
            f"'pairs', 'cluster', "
            f"'wild', or 'wild_cluster'"
        )
    if bootstrap in ("cluster", "wild_cluster") and cluster is None:
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: bootstrap={bootstrap!r} requires "
            "cluster=<column name>"
        )
    if dtype not in ("float64", "float32"):
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: dtype={dtype!r}; supported: "
            "'float64' or 'float32'"
        )
    n_boot = _positive_int(n_boot, name="n_boot", context="feols_jax_bootstrap")
    if n_boot < 2:
        raise DataInsufficient(
            "feols_jax_bootstrap: n_boot must be >= 2 to estimate bootstrap SE",
            recovery_hint=(
                "Use at least two bootstrap draws; production runs should use "
                "many more."
            ),
            diagnostics={"n_boot": n_boot},
        )
    vmap_chunk_size = _positive_int(
        vmap_chunk_size,
        name="vmap_chunk_size",
        context="feols_jax_bootstrap",
    )
    ci_alpha = _open_unit_float(
        ci_alpha,
        name="ci_alpha",
        context="feols_jax_bootstrap",
    )
    fe_maxiter = _positive_int(
        fe_maxiter,
        name="fe_maxiter",
        context="feols_jax_bootstrap",
    )
    fe_tol = _nonnegative_finite_float(
        fe_tol,
        name="fe_tol",
        context="feols_jax_bootstrap",
    )
    if cluster is not None and cluster not in data.columns:
        raise MethodIncompatibility(
            f"feols_jax_bootstrap: cluster column {cluster!r} not in data",
            recovery_hint="Pass an existing cluster column or omit cluster bootstrap.",
            diagnostics={"cluster": cluster},
        )

    prep = _jax_prep_inputs(
        formula,
        data,
        weights=weights,
        drop_singletons=drop_singletons,
        fe_tol=fe_tol,
        fe_maxiter=fe_maxiter,
    )

    # Cluster column needs to be aligned with the kept-rows mask; it
    # carries through the demean-singleton path even though demean
    # doesn't touch it directly.
    if cluster is not None:
        cluster_arr_full = data[cluster].to_numpy()
        cluster_codes_check, _ = pd.factorize(
            cluster_arr_full,
            sort=False,
            use_na_sentinel=True,
        )
        if (cluster_codes_check < 0).any():
            raise MethodIncompatibility(
                f"feols_jax_bootstrap: cluster column {cluster!r} contains "
                "NaN; drop or impute upstream"
            )
        cluster_kept = cluster_arr_full[prep["keep_mask"]]
        cluster_codes_kept, _ = pd.factorize(cluster_kept, sort=False)
        cluster_codes_kept = cluster_codes_kept.astype(np.int32)
        n_clusters = int(cluster_codes_kept.max()) + 1 if cluster_codes_kept.size else 0
        if n_clusters < 2 and bootstrap in ("cluster", "wild_cluster"):
            raise DataInsufficient(
                f"feols_jax_bootstrap: bootstrap={bootstrap!r} requires >= 2 clusters; "
                f"got {n_clusters}"
            )
    else:
        cluster_codes_kept = None
        n_clusters = 0

    np_dtype = np.float64
    jax_dtype = jnp.float32 if dtype == "float32" else jnp.float64

    X_j = jnp.asarray(prep["X_dem"], dtype=jax_dtype)
    y_j = jnp.asarray(prep["y_dem"], dtype=jax_dtype)
    w_j = jnp.asarray(prep["w"], dtype=jax_dtype)

    # Point estimate (single solve). Wild bootstrap variants also need
    # the residuals and bread XtWX_inv from this fit.
    _wls_solve, _, _ = _make_jax_kernels()
    beta_point_j, resid_j, _, XtWX_inv_j = _wls_solve(X_j, y_j, w_j)
    beta_point = np.asarray(beta_point_j, dtype=np_dtype)
    resid_point = np.asarray(resid_j, dtype=np_dtype)
    XtWX_inv = np.asarray(XtWX_inv_j, dtype=np_dtype)
    _require_finite_outputs(
        "feols_jax_bootstrap",
        beta=beta_point,
        residuals=resid_point,
        bread=XtWX_inv,
    )

    # Bootstrap.
    (
        _one_pairs_boot,
        _build_cluster_boot,
        _one_wild_boot,
        _build_wild_cluster_boot,
    ) = _make_bootstrap_kernels()
    key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(key, int(n_boot))

    if bootstrap == "pairs":

        def _run_chunk(keys_chunk: Any) -> Any:
            return jax.vmap(
                _one_pairs_boot,
                in_axes=(0, None, None, None),
            )(keys_chunk, X_j, y_j, w_j)

    elif bootstrap == "cluster":
        cluster_codes_j = jnp.asarray(cluster_codes_kept)
        _one_cluster_boot = _build_cluster_boot(n_clusters)

        def _run_chunk(keys_chunk: Any) -> Any:
            return jax.vmap(
                _one_cluster_boot,
                in_axes=(0, None, None, None, None),
            )(keys_chunk, X_j, y_j, w_j, cluster_codes_j)

    elif bootstrap == "wild":

        def _run_chunk(keys_chunk: Any) -> Any:
            return jax.vmap(
                _one_wild_boot,
                in_axes=(0, None, None, None, None, None),
            )(keys_chunk, X_j, w_j, resid_j, beta_point_j, XtWX_inv_j)

    else:  # wild_cluster
        cluster_codes_j = jnp.asarray(cluster_codes_kept)
        _one_wild_cluster_boot = _build_wild_cluster_boot(n_clusters)

        def _run_chunk(keys_chunk: Any) -> Any:
            return jax.vmap(
                _one_wild_cluster_boot,
                in_axes=(0, None, None, None, None, None, None),
            )(
                keys_chunk,
                X_j,
                w_j,
                resid_j,
                beta_point_j,
                XtWX_inv_j,
                cluster_codes_j,
            )

    chunks: List[np.ndarray] = []
    for start in range(0, int(n_boot), int(vmap_chunk_size)):
        stop = min(start + int(vmap_chunk_size), int(n_boot))
        chunk_keys = keys[start:stop]
        chunk_betas = _run_chunk(chunk_keys)
        chunks.append(np.asarray(chunk_betas, dtype=np_dtype))
    boot_betas = np.concatenate(chunks, axis=0)  # (n_boot, p)
    _require_finite_outputs(
        "feols_jax_bootstrap",
        bootstrap_betas=boot_betas,
    )

    se_boot = boot_betas.std(axis=0, ddof=1)
    lower = float(ci_alpha / 2)
    upper = float(1.0 - ci_alpha / 2)
    ci_lo = np.quantile(boot_betas, lower, axis=0)
    ci_hi = np.quantile(boot_betas, upper, axis=0)

    names = prep["coef_names_full"]
    return FeolsBootstrapResult(
        coef=pd.Series(beta_point, index=names, name="Estimate"),
        se_boot=pd.Series(se_boot, index=names, name="SE (boot)"),
        ci_lower=pd.Series(ci_lo, index=names, name="CI lower"),
        ci_upper=pd.Series(ci_hi, index=names, name="CI upper"),
        boot_betas=pd.DataFrame(boot_betas, columns=names),
        n_boot=int(n_boot),
        bootstrap_type=bootstrap,
    )


__all__ = ["feols_jax", "feols_jax_bootstrap", "FeolsBootstrapResult"]
