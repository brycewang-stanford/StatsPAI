"""Native StatsPAI OLS HDFE estimator (``sp.fast.feols``).

Closes the ``sp.fast.*`` symmetry: ``fepois`` (Phase 2) ships native
Poisson HDFE; ``feols`` ships the linear analogue. Pure-Python
orchestration on top of the Phase 1 demean kernel + Phase 4 inference
primitives — independent of pyfixest.

What this is
------------
- ``y ~ x1 + x2 | fe1 + fe2`` minimal formula DSL (same as ``fepois``).
- Closed-form WLS solve on FE-residualised regressors / outcome.
- DOF accounting follows ``ssc``: the default reproduces R ``fixest``
  (pinned in ``tests/reference_parity/
  test_fast_feols_weights_fixest_parity.py``); ``ssc='statspai'`` keeps
  the pre-1.31 ``fe_dof = sum(G_k - 1)``.
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
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from ._result_protocol import jsonable as _jsonable
from ._result_protocol import tidy_records as _tidy_records
from ._validation import nonempty_sample as _nonempty_sample
from ._validation import nonnegative_finite_float as _nonnegative_finite_float
from ._validation import positive_int as _positive_int
from ._validation import positive_weight_mass as _positive_weight_mass
from ._validation import require_identified_regressors as _require_identified_regressors

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class FeolsResult(ResultProtocolMixin):
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
    rss: float
    tss: float
    r_squared_within: float
    fe_names: List[str]
    fe_cardinality: List[int]
    df_resid: int
    vcov_type: str
    ssc: str = "fixest"
    cluster_var: Optional[str] = None
    backend: str = "statspai-native"
    #: Kish effective sizes of a weighted fit (``None`` without weights)
    weight_info: Optional[Dict[str, Any]] = None
    #: Kernel that residualised on the fixed effects: ``"rust"``,
    #: ``"numpy"`` or ``"none"`` (no fixed effects)
    fe_kernel: str = "none"

    def coef(self) -> pd.Series:
        return pd.Series(self.coef_vec, index=self.coef_names, name="Estimate")

    def se(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self.vcov_matrix)),
            index=self.coef_names,
            name="Std. Error",
        )

    def vcov(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.vcov_matrix,
            index=self.coef_names,
            columns=self.coef_names,
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
        p = 2.0 * _t.sf(np.abs(t), df=df)
        return pd.DataFrame(
            {
                "Estimate": b,
                "Std. Error": s,
                "t value": t,
                "Pr(>|t|)": p,
            },
            index=self.coef_names,
        )

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
        lines.append(str(self.tidy().to_string(float_format=lambda x: f"{x:.6f}")))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Lossless JSON-safe payload for HDFE OLS results."""
        payload = {
            "kind": "fast_feols_result",
            "model": "ols_hdfe",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_dropped_singletons": self.n_dropped_singletons,
            "rss": self.rss,
            "tss": self.tss,
            "r_squared_within": self.r_squared_within,
            "df_resid": self.df_resid,
            "vcov_type": self.vcov_type,
            "ssc": self.ssc,
            "cluster_var": self.cluster_var,
            "backend": self.backend,
            "fe_kernel": self.fe_kernel,
            "fixed_effects": [
                {"name": name, "cardinality": cardinality}
                for name, cardinality in zip(
                    self.fe_names,
                    self.fe_cardinality,
                )
            ],
            "coefficients": _tidy_records(self.tidy()),
            "vcov": {
                "terms": list(self.coef_names),
                "matrix": self.vcov_matrix,
            },
        }
        return cast(dict[str, Any], _jsonable(payload))

    def to_agent_summary(self, *, max_terms: int = 10) -> dict[str, Any]:
        """Bounded agent-facing summary for fast HDFE OLS results."""
        n_terms = len(self.coef_names)
        limit = max(int(max_terms), 0)
        rows = _tidy_records(self.tidy().head(limit))
        payload = {
            "kind": "fast_feols_agent_summary",
            "model": "ols_hdfe",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_dropped_singletons": self.n_dropped_singletons,
            "df_resid": self.df_resid,
            "r_squared_within": self.r_squared_within,
            "vcov_type": self.vcov_type,
            "ssc": self.ssc,
            "cluster_var": self.cluster_var,
            "fixed_effects": [
                {"name": name, "cardinality": cardinality}
                for name, cardinality in zip(
                    self.fe_names,
                    self.fe_cardinality,
                )
            ],
            "coefficients": rows,
            "n_terms": n_terms,
            "truncated_terms": max(n_terms - limit, 0),
            "backend": self.backend,
            "fe_kernel": self.fe_kernel,
        }
        return cast(dict[str, Any], _jsonable(payload))

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
    ssc: str = "fixest",
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    weights_as: Optional[str] = None,
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

        Degrees of freedom and the CR1 factor follow ``ssc`` (below); the
        default reproduces ``fixest::feols`` and ``reghdfe``. Use the
        bootstrap (``sp.fast.boottest``) for few-cluster inference.
    cluster : str, optional
        Column name for cluster identifiers. Required when
        ``vcov="cr1"``; rejected otherwise. NaN cluster values raise.
    weights : str, optional
        Column name of observation weights. Each obs's contribution to
        the objective is scaled by ``w_i``. With ``vcov='iid'`` they are
        analytic weights (Stata ``[aw=]``, R ``fixest`` ``weights=``).
    weights_as : {'analytic', 'sampling'}, optional
        What the weights are. ``'analytic'`` is what ``weights=`` alone
        already means. ``'sampling'`` (Stata ``[pw=]``) gives the same
        coefficients with ``vcov='hc1'``, unless ``vcov=`` or
        ``cluster=`` asks for another. Saying which also silences the
        warning about dispersed weights under the classical variance.
        ``'frequency'`` is refused.
    ssc : {"fixest", "statspai"}, default "fixest"
        Small-sample correction convention for residual degrees of
        freedom and CR1 scaling.

        - ``"fixest"`` (default since 1.31.0) mirrors R ``fixest`` /
          Stata ``reghdfe`` defaults, the setting the parity harness
          pins: the absorbed effects cost ``ΣG_k - (K - 1)`` degrees of
          freedom for ``K`` dimensions (``G`` for one), and one-way
          clustered CR1 excludes FE dimensions nested in the cluster
          variable, keeping the intercept they span.
        - ``"statspai"`` is the pre-1.31 default, kept to reproduce old
          numbers: ``fe_dof = Σ(G_k - 1)`` (one degree of freedom too few
          for two-way FE) and every absorbed FE charged in CR1, which
          inflates clustered SEs when the effects are nested in the
          clusters (SE/SD 1.08, coverage 0.979 on the Track B panel).
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> df = pd.DataFrame({
    ...     "firm": rng.integers(0, 25, n),
    ...     "year": rng.integers(2010, 2016, n),
    ...     "x1": rng.normal(size=n),
    ...     "x2": rng.normal(size=n),
    ... })
    >>> df["y"] = 1 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=n)
    >>> fit = sp.fast.feols("y ~ x1 + x2 | firm + year", data=df)
    >>> text = fit.summary()                      # str
    >>> list(fit.coef().index)                    # pd.Series
    ['x1', 'x2']
    >>> type(fit.se()).__name__
    'Series'
    """
    return _feols_fit(
        formula,
        data,
        vcov=vcov,
        cluster=cluster,
        weights=weights,
        ssc=ssc,
        drop_singletons=drop_singletons,
        fe_tol=fe_tol,
        fe_maxiter=fe_maxiter,
        weights_as=weights_as,
    )[0]


def _feols_fit(
    formula: str,
    data: pd.DataFrame,
    *,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    ssc: str = "fixest",
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    weights_as: Optional[str] = None,
    note_weights: bool = True,
) -> Tuple[FeolsResult, Dict[str, Any]]:
    """Body of :func:`feols`.

    Also returns the arrays ``sp.feols`` reports on its native path: the
    estimation-sample mask, the residualised outcome and design, the
    residuals, the weights and the parameter count behind each small-sample
    factor.
    """
    if vcov not in ("iid", "hc1", "cr1"):
        raise MethodIncompatibility(
            f"feols: vcov={vcov!r}; supported: 'iid', 'hc1', or 'cr1'"
        )
    if ssc not in ("statspai", "fixest"):
        raise MethodIncompatibility(
            f"feols: ssc={ssc!r}; supported: 'statspai' or 'fixest'"
        )
    # weights_as=: what the weights are; sampling weights take hc1 unless
    # another variance was asked for.
    from ..core._agent_summary import resolve_weights_as

    weights_as = resolve_weights_as(
        weights_as,
        has_weights=weights is not None,
        native="analytic",
        entry="sp.fast.feols",
    )
    if weights_as == "sampling" and cluster is None and vcov == "iid":
        vcov = "hc1"
    if vcov == "cr1" and cluster is None:
        raise MethodIncompatibility("feols: vcov='cr1' requires cluster=<column name>")
    if cluster is not None and vcov in ("iid", "hc1"):
        raise MethodIncompatibility(
            f"feols: cluster={cluster!r} provided but vcov={vcov!r}; "
            "set vcov='cr1' to compute cluster-robust SE"
        )
    fe_maxiter = _positive_int(fe_maxiter, name="fe_maxiter", context="feols")
    fe_tol = _nonnegative_finite_float(fe_tol, name="fe_tol", context="feols")

    # Lazy imports to keep top-level `sp.fast` cheap.
    from .fepois import _parse_fepois_formula
    from .inference import _cr1_from_codes

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
        raise MethodIncompatibility(f"feols: columns missing from data: {missing}")

    n_obs = len(data)
    _nonempty_sample(n_obs, context="feols")
    # One column-major block [y | X]: the demean kernel works on columns,
    # and filling it column by column copies each input once.
    n_x = len(rhs_terms) + int(add_intercept)
    A = np.empty((n_obs, 1 + n_x), dtype=np.float64, order="F")
    A[:, 0] = data[lhs].to_numpy(dtype=np.float64)
    if add_intercept:
        A[:, 1] = 1.0
    for j, term in enumerate(rhs_terms):
        A[:, 1 + int(add_intercept) + j] = data[term].to_numpy(dtype=np.float64)
    y = A[:, 0]
    X = A[:, 1:]
    if not np.isfinite(y).all():
        raise MethodIncompatibility(
            f"feols: outcome column {lhs!r} has non-finite values"
        )
    if not np.isfinite(X).all():
        raise MethodIncompatibility(
            "feols: regressor columns contain non-finite values"
        )
    coef_names_full: List[str] = (["(Intercept)"] if add_intercept else []) + list(
        rhs_terms
    )
    if X.shape[1] == 0:
        raise ValueError(
            "No regressors after parsing — formula must include at least "
            "one RHS term (or '1' for an intercept)."
        )

    # Observation weights
    if weights is not None:
        w_full = data[weights].to_numpy(dtype=np.float64).copy()
        if (w_full < 0).any():
            raise MethodIncompatibility(
                f"feols: weights column {weights!r} contains negative values"
            )
        if not np.isfinite(w_full).all():
            raise MethodIncompatibility(
                f"feols: weights column {weights!r} contains non-finite values"
            )
        _positive_weight_mass(
            w_full,
            context=f"feols weights column {weights!r}",
        )
    else:
        w_full = None

    # Cluster column
    if cluster is not None:
        cluster_arr_full = data[cluster].to_numpy()
        cluster_codes_check, _ = pd.factorize(
            cluster_arr_full,
            sort=False,
            use_na_sentinel=True,
        )
        if (cluster_codes_check < 0).any():
            raise MethodIncompatibility(
                f"feols: cluster column {cluster!r} contains NaN; "
                "drop or impute upstream"
            )
    else:
        cluster_arr_full = None

    # ---------- FE residualisation path ----------
    if fe_terms:
        if w_full is None:
            # Unweighted: the demean kernel (Rust when installed), run in
            # place on an owned column-major copy of the kept rows of
            # [y | X]; the block itself stays raw for the checks below.
            from .demean import (
                _demean_core,
                _densify_kept,
                _detect_singletons,
                _factorize_columns,
            )

            codes_raw, n_fe_raw = _factorize_columns(
                [data[c].to_numpy() for c in fe_terms]
            )
            keep_mask = (
                _detect_singletons(codes_raw, n_obs)
                if drop_singletons
                else np.ones(n_obs, dtype=bool)
            )
            n_kept = int(keep_mask.sum())
            n_dropped_singletons = n_obs - n_kept
            if n_kept < 1:
                raise DataInsufficient(
                    "fast.demean: no rows remain after singleton pruning"
                )
            fe_codes_list, counts_fe, fe_card = _densify_kept(
                codes_raw, n_fe_raw, keep_mask, n_dropped_singletons
            )
            if n_dropped_singletons:
                A = np.asfortranarray(A[keep_mask])
                y, X = A[:, 0], A[:, 1:]
            work, _, _, _, fe_kernel = _demean_core(
                A.copy(order="F"),
                fe_codes_list,
                counts_fe,
                max_iter=fe_maxiter,
                tol=1e-12,
                tol_abs=fe_tol,
                accelerate=True,
                accel_period=5,
                backend="auto",
            )
            y_dem = work[:, 0]
            X_dem = work[:, 1:]
            y_raw_kept, X_raw_kept = y, X
        else:
            # Weighted: the FE projection is W-weighted, so the
            # arithmetic-mean Rust kernel is wrong here. Route through
            # the same weighted-AP loop ``fepois`` uses internally.
            from .._rust_backend import HAS_RUST as _has_rust
            from .demean import _detect_singletons as _ds_helper
            from .fepois import _weighted_ap_demean

            # Factorise FEs explicitly so we control the singleton path
            fe_codes_raw: List[np.ndarray] = []
            for col in fe_terms:
                codes, uniq = pd.factorize(
                    data[col],
                    sort=False,
                    use_na_sentinel=True,
                )
                if (codes < 0).any():
                    raise MethodIncompatibility(
                        f"feols: NaN in fixed effect column {col!r}"
                    )
                fe_codes_raw.append(codes.astype(np.int64))
            keep_mask = (
                _ds_helper(fe_codes_raw, n_obs)
                if drop_singletons
                else np.ones(n_obs, dtype=bool)
            )
            n_kept = int(keep_mask.sum())
            n_dropped_singletons = n_obs - n_kept
            if n_kept == 0:
                raise DataInsufficient(
                    "feols: every row was dropped as a singleton of the fixed "
                    "effects, so nothing is left to estimate on",
                    recovery_hint=(
                        "Use coarser fixed effects, or pass drop_singletons=False "
                        "to keep the singleton rows."
                    ),
                )
            # Re-densify codes on kept rows
            fe_codes_kept: List[np.ndarray] = []
            counts_list: List[np.ndarray] = []
            weighted_fe_card: List[int] = []
            for codes_k in fe_codes_raw:
                ck = codes_k[keep_mask]
                dense, uniq = pd.factorize(ck, sort=False)
                dense = dense.astype(np.int64)
                G = len(uniq)
                fe_codes_kept.append(dense)
                counts_list.append(np.bincount(dense, minlength=G).astype(np.float64))
                weighted_fe_card.append(G)
            y_kept = y[keep_mask]
            X_kept = X[keep_mask]
            w_kept = w_full[keep_mask]
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
            y_raw_kept, X_raw_kept = y_kept, X_kept
            fe_card = weighted_fe_card
            fe_codes_list = fe_codes_kept
            fe_kernel = "rust" if _has_rust and len(fe_terms) >= 2 else "numpy"
        fe_dof_statspai = _statspai_fe_dof(fe_card)
    else:
        keep_mask = np.ones(n_obs, dtype=bool)
        n_kept = n_obs
        n_dropped_singletons = 0
        y_dem = y.copy()
        X_dem = X.copy()
        y_raw_kept, X_raw_kept = y, X
        fe_card = []
        fe_codes_list = []
        fe_dof_statspai = 0
        fe_kernel = "none"

    # Apply mask to weights / cluster (demean already returned masked X̃, ỹ)
    if w_full is not None:
        w = w_full[keep_mask]
        _positive_weight_mass(
            w,
            context=f"feols kept sample weights column {weights!r}",
        )
    else:
        w = np.ones(n_kept, dtype=np.float64)
    if cluster_arr_full is not None:
        cluster_arr_kept = cluster_arr_full[keep_mask]
        # Dense codes in first-appearance order on the kept rows, as
        # factorising the kept labels would give.
        if n_kept < n_obs:
            cl_codes, cl_levels = pd.factorize(
                cluster_codes_check[keep_mask], sort=False
            )
            n_clusters = len(cl_levels)
        else:
            cl_codes = cluster_codes_check
            n_clusters = int(cl_codes.max()) + 1 if cl_codes.size else 0
        if n_clusters < 2:
            raise DataInsufficient(f"crve: need at least 2 clusters, got {n_clusters}")
    else:
        cluster_arr_kept = None

    n, p = X_dem.shape
    if ssc == "fixest":
        fe_dof = _fixest_fe_dof(fe_card)
        if vcov == "cr1":
            cr1_extra_df = _fixest_cluster_fe_dof_codes(
                fe_codes_list, fe_card, cl_codes
            )
        else:
            cr1_extra_df = fe_dof
    else:
        fe_dof = fe_dof_statspai
        cr1_extra_df = fe_dof_statspai

    # ---------- identification ----------
    # Inverting the normal equations only fails on an exactly singular
    # matrix; an absorbed regressor comes back from the residualisation as
    # rounding noise, so check explicitly against the demean's threshold.
    if not fe_terms:
        tol_abs_id, tol_rel_id = 0.0, 0.0
    elif w_full is None:
        tol_abs_id, tol_rel_id = fe_tol, 1e-12
    else:
        tol_abs_id, tol_rel_id = 0.0, fe_tol
    _require_identified_regressors(
        X_raw_kept,
        X_dem,
        coef_names_full,
        context="feols",
        weights=w if w_full is not None else None,
        tol_abs=tol_abs_id,
        tol_rel=tol_rel_id,
    )

    # ---------- WLS solve ----------
    # X̃' diag(w) X̃ (no weighted copy of X̃ when there are no weights)
    XtWX = X_dem.T @ (X_dem * w[:, None] if w_full is not None else X_dem)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            f"Normal equations are singular: {exc}. Likely cause: "
            "perfect collinearity in regressors after FE residualisation."
        ) from exc
    XtWy = X_dem.T @ (w * y_dem if w_full is not None else y_dem)
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
        u = (resid * w)[:, None] * X_dem  # (n, p)
        meat = u.T @ u
        vcov_mat = XtWX_inv @ meat @ XtWX_inv
        if df_resid > 0:
            vcov_mat = vcov_mat * (n / df_resid)
    else:  # cr1
        vcov_mat = _cr1_from_codes(
            X_dem,
            resid,
            cl_codes,
            n_clusters,
            weights=w,
            bread=XtWX_inv,
            extra_df=cr1_extra_df,
        )

    weight_info: Optional[Dict[str, Any]] = None
    if weights is not None and note_weights:
        from ..core._agent_summary import note_weight_diagnostics

        weight_info = {}
        note_weight_diagnostics(
            weight_info,
            w,
            {"iid": "classical", "hc1": "hc1", "cr1": "cluster"}[vcov],
            entry="sp.fast.feols",
            robust_option="vcov='hc1' or vcov='cr1' with cluster=",
            classical_reading="analytic weights (Stata [aw=])",
            cluster_keys=cluster_arr_kept if vcov == "cr1" else None,
            cluster=cluster,
            weights_as=weights_as,
        )

    result = FeolsResult(
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
        ssc=ssc,
        cluster_var=cluster,
        weight_info=weight_info,
        fe_kernel=fe_kernel,
    )
    arrays = {
        "keep_mask": keep_mask,
        "y": y_raw_kept,
        "y_dem": y_dem,
        "X_dem": X_dem,
        "resid": resid,
        "w": w,
        "cluster": cluster_arr_kept,
        "n_clusters": n_clusters if cluster_arr_full is not None else 0,
        "df_k": int(n - df_resid),
        "df_k_cluster": int(p + cr1_extra_df),
    }
    return result, arrays


def _statspai_fe_dof(fe_card: List[int]) -> int:
    return int(sum(int(g) - 1 for g in fe_card))


def _fixest_fe_dof(fe_card: List[int]) -> int:
    """Parameters the absorbed effects use up, as fixest counts them.

    The effects span the intercept, so one dimension of ``G`` levels costs
    ``G`` and each further dimension one less than its levels: the rank of
    the dummy design is ``sum(G_k) - (K - 1)``. Until 1.36.0 the
    one-dimension case returned ``G - 1``, one too few, which made the
    IID / HC1 standard errors ``sqrt((n - p - G) / (n - p - G + 1))`` of
    fixest's. Two or more dimensions were exact.
    """
    if not fe_card:
        return 0
    return int(sum(int(g) for g in fe_card) - (len(fe_card) - 1))


def _fixest_cluster_fe_dof(
    data: pd.DataFrame,
    fe_terms: List[str],
    fe_card: List[int],
    keep_mask: np.ndarray,
    cluster: Optional[str],
) -> int:
    if cluster is None or not fe_terms:
        return _fixest_fe_dof(fe_card)

    effective_cards: List[int] = []
    cluster_values = data.loc[keep_mask, cluster].to_numpy()
    for fe, card in zip(fe_terms, fe_card):
        fe_values = data.loc[keep_mask, fe].to_numpy()
        if _is_nested_in_cluster(fe_values, cluster_values):
            continue
        effective_cards.append(int(card))
    if not effective_cards:
        # Every absorbed dimension is nested in the clusters: fixest (and
        # reghdfe) drop them from K but keep the intercept they span.
        return 1
    return int(sum(effective_cards) - (len(effective_cards) - 1))


def _fixest_cluster_fe_dof_codes(
    fe_codes: List[np.ndarray],
    fe_card: List[int],
    cluster_codes: np.ndarray,
) -> int:
    """:func:`_fixest_cluster_fe_dof` on dense codes of the kept rows."""
    if not fe_codes:
        return _fixest_fe_dof(fe_card)
    free = [
        int(card)
        for codes, card in zip(fe_codes, fe_card)
        if not _codes_nested(codes, int(card), cluster_codes)
    ]
    if not free:
        return 1  # every effect nested in the clusters: keep the intercept
    return int(sum(free) - (len(free) - 1))


def _codes_nested(codes: np.ndarray, n_levels: int, cluster_codes: np.ndarray) -> bool:
    """Every level of dense ``codes`` lies in a single cluster."""
    owner = np.empty(n_levels, dtype=np.int64)
    owner[codes] = cluster_codes
    return bool((owner[codes] == cluster_codes).all())


def _is_nested_in_cluster(
    fe_values: np.ndarray,
    cluster_values: np.ndarray,
) -> bool:
    # Nested when every level of the effect sits in a single cluster:
    # record one cluster per level, then check every row agrees with it.
    f_codes, f_levels = pd.factorize(fe_values, sort=False)
    c_codes, _ = pd.factorize(cluster_values, sort=False)
    owner = np.empty(len(f_levels), dtype=np.int64)
    owner[f_codes] = c_codes
    return bool((owner[f_codes] == c_codes).all())


__all__ = ["feols", "FeolsResult"]
