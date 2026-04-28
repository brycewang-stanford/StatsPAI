"""Native StatsPAI OLS HDFE estimator (``sp.fast.feols``).

Closes the ``sp.fast.*`` symmetry: ``fepois`` (Phase 2) ships native
Poisson HDFE; ``feols`` ships the linear analogue. Pure-Python
orchestration on top of the Phase 1 demean kernel + Phase 4 inference
primitives — independent of pyfixest.

What this is
------------
- ``y ~ x1 + x2 | fe1 + fe2`` minimal formula DSL (same as ``fepois``).
- Closed-form WLS solve on FE-residualised regressors / outcome.
- Full DOF accounting: ``fe_dof = sum(G_k - 1)`` (matches ``fepois``,
  matches ``fixest::ssc(fixef.K='full')`` to a uniform 1-DOF off-by-true-
  rank for K≥2 — same convention as the rest of fast/*).
- vcov: ``iid`` (homoscedastic), ``hc1`` (heteroscedasticity-robust),
  ``cr1`` (one-way cluster-robust, FE-rank-aware via ``extra_df``).
- Optional observation weights → WLS.

What this is NOT (yet)
----------------------
- IV / 2SLS. Use ``sp.iv`` / ``sp.fast.fepois`` for analogues; ``feols``
  here is single-equation OLS only.
- ``i()`` / ``sw()`` / ``csw()`` formula expansion. Build the dummies
  yourself with the Phase 3 DSL helpers and pass them in as columns.
- CR2 / CR3 cluster SE. Wired through :func:`crve` directly if needed.

References
----------
Bergé, L. (2018). Efficient estimation of maximum likelihood models with
multiple fixed-effects: the R package FENmlm. CREA DP 13.
Correia, S. (2017). Linear models with high-dimensional fixed effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FeolsResult:
    """Outcome of :func:`feols`.

    Attribute names mirror :class:`FePoisResult` so downstream code
    using ``.coef()`` / ``.se()`` / ``.tidy()`` keeps working.
    """

    formula: str
    coef_names: List[str]
    coef_vec: np.ndarray
    vcov_matrix: np.ndarray
    n_obs: int
    n_kept: int
    n_dropped_singletons: int
    rss: float                  # residual sum of squares (in residualised space)
    tss: float                  # total sum of squares (after FE residualisation)
    r_squared_within: float     # within R²
    fe_names: List[str]
    fe_cardinality: List[int]
    df_resid: int               # n_kept - p - sum(G_k - 1)
    vcov_type: str
    cluster_var: Optional[str] = None
    backend: str = "statspai-native"

    def coef(self) -> pd.Series:
        return pd.Series(self.coef_vec, index=self.coef_names, name="Estimate")

    def se(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self.vcov_matrix)),
            index=self.coef_names, name="Std. Error",
        )

    def vcov(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.vcov_matrix, index=self.coef_names, columns=self.coef_names,
        )

    def tidy(self) -> pd.DataFrame:
        b = self.coef_vec
        s = np.sqrt(np.diag(self.vcov_matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(s > 0, b / s, np.nan)
        # Two-sided t-test (df = df_resid for iid; large-sample
        # approximation otherwise — we still report t-style stats since
        # users will compare to fixest's tidy output)
        from scipy.stats import t as _t
        df = max(self.df_resid, 1)
        p = 2.0 * (1.0 - _t.cdf(np.abs(t), df=df))
        return pd.DataFrame({
            "Estimate": b,
            "Std. Error": s,
            "t value": t,
            "Pr(>|t|)": p,
        }, index=self.coef_names)

    def summary(self) -> str:
        lines: List[str] = []
        lines.append(
            f"sp.fast.feols  |  {self.formula}  |  vcov={self.vcov_type}"
            + (f", cluster={self.cluster_var}" if self.cluster_var else "")
        )
        lines.append(
            f"N={self.n_obs:,}  kept={self.n_kept:,}  "
            f"singletons={self.n_dropped_singletons}  "
            f"df_resid={self.df_resid}  R²(within)={self.r_squared_within:.4f}"
        )
        if self.fe_names:
            fe_desc = ", ".join(
                f"{n}({c:,})" for n, c in zip(self.fe_names, self.fe_cardinality)
            )
            lines.append(f"Fixed effects: {fe_desc}")
        lines.append("")
        lines.append(self.tidy().to_string(float_format=lambda x: f"{x:.6f}"))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover  - cosmetic
        return self.summary()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feols(
    formula: str,
    data: pd.DataFrame,
    *,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
) -> FeolsResult:
    """OLS / WLS with high-dimensional fixed effects.

    Native StatsPAI implementation — independent of pyfixest. Mirrors
    the public surface of :func:`sp.fast.fepois` (formula DSL, ``vcov``,
    ``cluster``, ``weights`` arguments).

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 | fe1 + fe2"``. The ``| fe`` block is optional.
    data : pd.DataFrame
        Must contain all columns referenced in ``formula``.
    vcov : {"iid", "hc1", "cr1"}, default "iid"
        Variance-covariance estimator.

        - ``"iid"``: classical homoscedastic OLS variance,
          ``σ² (X̃'WX̃)^{-1}`` with ``σ² = RSS / df_resid``.
        - ``"hc1"``: heteroscedasticity-robust HC1 sandwich.
        - ``"cr1"``: one-way cluster-robust (Liang-Zeger) with
          FE-rank-aware small-sample factor.

        The DOF convention is ``df_resid = n_kept - p - Σ(G_k - 1)``
        — same as :func:`sp.fast.fepois` and :func:`sp.fast.event_study`.
        This matches ``fixest::ssc(fixef.K="full")`` to a uniform 1-DOF
        off-by-true-rank for K≥2, and pyfixest's default
        ``ssc(fixef.K="nested")`` to within ~1% on iid/hc1 (always
        slightly smaller SE because StatsPAI charges all FEs even when
        nested in cluster). Use the bootstrap (``sp.fast.boottest``)
        for tight finite-sample inference; use ``vcov="iid"`` /
        ``"hc1"`` for the canonical analogues of fixest's defaults.
    cluster : str, optional
        Column name for cluster identifiers. Required when
        ``vcov="cr1"``; rejected otherwise. NaN cluster values raise.
    weights : str, optional
        Column name of observation weights. Each obs's contribution to
        the objective is scaled by ``w_i`` (frequency / survey weights).
    drop_singletons : bool, default True
        Iteratively drop FE-singleton rows before fitting.
    fe_tol, fe_maxiter : float, int
        Inner alternating-projection demean controls; passed through to
        ``sp.fast.demean``.

    Returns
    -------
    FeolsResult

    Examples
    --------
    >>> import statspai as sp
    >>> fit = sp.fast.feols("y ~ x1 + x2 | firm + year", data=df)
    >>> fit.summary()
    >>> fit.coef()                                # pd.Series
    >>> fit.se()                                  # pd.Series
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

    # Lazy imports to keep top-level `sp.fast` cheap.
    from .fepois import _parse_fepois_formula
    from .demean import demean as _demean
    from .inference import crve as _crve

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
    y = data[lhs].to_numpy(dtype=np.float64).copy()
    if not np.isfinite(y).all():
        raise ValueError(f"outcome column {lhs!r} has non-finite values")
    X_user = (
        data[rhs_terms].to_numpy(dtype=np.float64).copy()
        if rhs_terms else np.empty((n_obs, 0), dtype=np.float64)
    )
    if X_user.ndim == 1:
        X_user = X_user.reshape(-1, 1)
    if not np.isfinite(X_user).all():
        raise ValueError("regressor columns contain non-finite values")
    if add_intercept:
        X = np.column_stack([np.ones(n_obs, dtype=np.float64), X_user])
        coef_names_full: List[str] = ["(Intercept)"] + list(rhs_terms)
    else:
        X = X_user
        coef_names_full = list(rhs_terms)
    if X.shape[1] == 0:
        raise ValueError(
            "No regressors after parsing — formula must include at least "
            "one RHS term (or '1' for an intercept)."
        )

    # Observation weights
    if weights is not None:
        w_full = data[weights].to_numpy(dtype=np.float64).copy()
        if (w_full < 0).any():
            raise ValueError(f"weights column {weights!r} contains negative values")
        if not np.isfinite(w_full).all():
            raise ValueError(f"weights column {weights!r} contains non-finite values")
    else:
        w_full = None

    # Cluster column
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

    # ---------- FE residualisation path ----------
    if fe_terms:
        fe_df = data[fe_terms]
        if w_full is None:
            # Unweighted: use the Rust-backed demean kernel (Phase 1).
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
            # Weighted: the FE projection is W-weighted, so the
            # arithmetic-mean Rust kernel is wrong here. Route through
            # the same weighted-AP loop ``fepois`` uses internally.
            from .fepois import _weighted_ap_demean
            from .demean import _detect_singletons as _ds_helper
            # Factorise FEs explicitly so we control the singleton path
            fe_codes_raw: List[np.ndarray] = []
            for col in fe_terms:
                codes, uniq = pd.factorize(
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
            # Re-densify codes on kept rows
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

    # Apply mask to weights / cluster (demean already returned masked X̃, ỹ)
    if w_full is not None:
        w = w_full[keep_mask]
    else:
        w = np.ones(n_kept, dtype=np.float64)
    if cluster_arr_full is not None:
        cluster_arr_kept = cluster_arr_full[keep_mask]
    else:
        cluster_arr_kept = None

    n, p = X_dem.shape

    # ---------- WLS solve ----------
    Xw = X_dem * w[:, None]                     # diag(w) X̃
    XtWX = X_dem.T @ Xw                         # X̃' diag(w) X̃
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Normal equations are singular: {exc}. Likely cause: "
            "perfect collinearity in regressors after FE residualisation."
        ) from exc
    XtWy = X_dem.T @ (w * y_dem)
    beta = XtWX_inv @ XtWy
    resid = y_dem - X_dem @ beta

    # Within R² (matches fixest's `r2(..., 'wr2')` convention)
    if w_full is not None:
        # Weighted within R²: 1 - (Σ w u²) / (Σ w (y - ȳ_w)²)
        wsum = float(w.sum())
        ybar_w = float((w * y_dem).sum() / wsum) if wsum > 0 else 0.0
        rss = float((w * resid * resid).sum())
        tss = float((w * (y_dem - ybar_w) ** 2).sum())
    else:
        rss = float(resid @ resid)
        ybar = float(y_dem.mean())
        tss = float(((y_dem - ybar) ** 2).sum())
    r_squared_within = 1.0 - rss / max(tss, 1e-30)

    df_resid = n - p - fe_dof

    # ---------- Variance-covariance ----------
    if vcov == "iid":
        sigma2 = rss / max(df_resid, 1)
        vcov_mat = sigma2 * XtWX_inv
    elif vcov == "hc1":
        # HC1: V = bread @ Σ_i (w_i u_i)² x̃_i x̃_i' @ bread,
        # then scaled by n/(n - p - fe_dof). MacKinnon-White (1985) /
        # fixest convention.
        # Score row: s_i = w_i u_i x̃_i. meat = Σ_i s_i s_i'.
        u = (resid * w)[:, None] * X_dem        # (n, p)
        meat = u.T @ u
        vcov_mat = XtWX_inv @ meat @ XtWX_inv
        if df_resid > 0:
            vcov_mat = vcov_mat * (n / df_resid)
    else:  # cr1
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
    )


__all__ = ["feols", "FeolsResult"]
