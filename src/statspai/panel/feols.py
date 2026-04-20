"""
Unified high-dimensional fixed-effects OLS estimator — ``sp.feols()``.

Mirrors R's ``fixest::feols`` and Stata's ``reghdfe`` top-level API:

    sp.feols("y ~ x1 + x2 | firm + year", data=df, cluster="firm")

Formula grammar
---------------
The pipe ``|`` separates regressors from absorbed fixed effects.

    "y ~ x1 + x2"                   → no FE, pure OLS (constant included)
    "y ~ x1 | firm"                 → firm FE absorbed
    "y ~ x1 + x2 | firm + year"     → two-way FE (firm, year)
    "y ~ x1 | firm + year + state"  → three-way FE

Variables on both sides are bare column names (no Patsy syntax).

Inference
---------
- Default: classical OLS SE (appropriate only when errors i.i.d.).
- ``cluster='firm'``: one-way cluster-robust (Liang-Zeger CR1).
- ``cluster=['firm', 'year']``: N-way CGM via inclusion-exclusion.
- ``cluster='firm', wild=True``: wild cluster bootstrap on top of CR1.

Returns a :class:`FEOLSResult` with coef / se / vcov / R²-within, plus
reference to the reusable ``Absorber`` (enables re-running on subsamples
or for event-study path estimation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .hdfe import Absorber, absorb_ols


@dataclass
class FEOLSResult:
    """Result of ``sp.feols()``.

    Attributes
    ----------
    params : pd.Series
        Coefficient estimates indexed by regressor name.
    std_errors : pd.Series
        Standard errors indexed by regressor name.
    vcov : np.ndarray
        Variance-covariance matrix of the coefficients.
    tvalues, pvalues : pd.Series
    conf_int_lower, conf_int_upper : pd.Series
    residuals : np.ndarray
        In-sample residuals (after FE absorption).
    fitted_within : np.ndarray
        Predicted values from X β (excludes FE contribution).
    n_obs : int
    n_singletons_dropped : int
    n_fe : List[int]
        Number of groups per absorbed FE dimension.
    dof_fe : int
        Degrees of freedom consumed by the FEs.
    df_resid : int
    r2_within : float
    se_type : str
        'iid' | 'cluster' | 'multiway_cluster' | 'wild_cluster'
    cluster_info : dict
        Metadata (cluster names, counts).
    formula : str
    absorber : Absorber
        Reusable absorber (includes ``keep_mask`` to subset rows).
    """

    params: pd.Series
    std_errors: pd.Series
    vcov: np.ndarray
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series
    residuals: np.ndarray
    fitted_within: np.ndarray
    n_obs: int
    n_singletons_dropped: int
    n_fe: List[int]
    dof_fe: int
    df_resid: int
    r2_within: float
    se_type: str
    cluster_info: Dict[str, Any]
    formula: str
    absorber: Absorber
    converged: bool
    iters: int

    def summary(self) -> str:
        lines: List[str] = []
        lines.append(f"FEOLS (reghdfe-style)  |  {self.formula}")
        lines.append("=" * max(60, len(self.formula) + 25))
        lines.append(f"Obs: {self.n_obs:,d}   Singletons dropped: {self.n_singletons_dropped:,d}")
        lines.append(f"Absorbed FE: groups={self.n_fe}   dof_fe={self.dof_fe}   df_resid={self.df_resid}")
        lines.append(f"R² (within) = {self.r2_within:.4f}")
        lines.append(f"SE type: {self.se_type}")
        if self.cluster_info:
            lines.append(f"Cluster info: {self.cluster_info}")
        lines.append("-" * 60)
        lines.append(f"{'Variable':<20}{'Estimate':>12}{'Std.Err':>12}{'t':>8}{'p':>10}")
        lines.append("-" * 60)
        for name in self.params.index:
            b = self.params[name]
            se = self.std_errors[name]
            t = self.tvalues[name]
            p = self.pvalues[name]
            stars = (
                "***" if p < 0.01 else
                "**" if p < 0.05 else
                "*" if p < 0.10 else ""
            )
            lines.append(f"{name:<20}{b:>12.4f}{se:>12.4f}{t:>8.2f}{p:>10.4f}  {stars}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # Convenience: pandas-friendly attribute names
    @property
    def coef(self) -> pd.Series:
        return self.params

    @property
    def se(self) -> pd.Series:
        return self.std_errors


# ======================================================================
# Formula parsing
# ======================================================================

_FORMULA_RE = re.compile(
    r"""
    ^\s*(?P<lhs>[A-Za-z_][A-Za-z_0-9]*)\s*~\s*      # y ~
    (?P<rhs>[^|]*?)                                  # regressors
    (?:\|\s*(?P<fe>.*))?                             # optional | fe1 + fe2
    \s*$
    """,
    re.VERBOSE,
)


def _parse_formula(formula: str) -> tuple[str, List[str], List[str]]:
    m = _FORMULA_RE.match(formula)
    if not m:
        raise ValueError(f"Could not parse formula: {formula!r}")
    lhs = m.group("lhs").strip()
    rhs_str = (m.group("rhs") or "").strip()
    fe_str = (m.group("fe") or "").strip()

    def _split(s: str) -> List[str]:
        if not s:
            return []
        tokens = [t.strip() for t in s.split("+")]
        tokens = [t for t in tokens if t and t != "1"]
        if any(not re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", t) for t in tokens):
            raise ValueError(
                f"Only bare column names are supported in feols formulas; got: {s!r}"
            )
        return tokens

    x_vars = _split(rhs_str)
    fe_vars = _split(fe_str)
    return lhs, x_vars, fe_vars


# ======================================================================
# feols
# ======================================================================


def feols(
    formula: str,
    data: pd.DataFrame,
    *,
    weights: Optional[Union[str, np.ndarray]] = None,
    cluster: Optional[Union[str, List[str]]] = None,
    se_type: Optional[str] = None,
    wild: bool = False,
    wild_n_boot: int = 999,
    wild_weight_type: str = "webb",
    wild_seed: Optional[int] = None,
    alpha: float = 0.05,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
) -> FEOLSResult:
    """reghdfe-style OLS with high-dimensional fixed effects.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 | fe1 + fe2 + fe3"``. The ``| fe...`` part is
        optional.
    data : DataFrame
    weights : str or ndarray, optional
        Observation weights. Column name or raw array.
    cluster : str or list, optional
        One-way or multi-way cluster column(s).
    se_type : {'iid', 'cluster', 'multiway_cluster', 'wild_cluster'}
        Override automatic inference of SE type. Usually inferred from
        ``cluster`` / ``wild``.
    wild : bool, default False
        If True (and ``cluster`` is given), return wild-cluster-bootstrap
        p-values / CIs alongside classical cluster SE. Applied variable-
        by-variable. Only supported with a single cluster column.
    wild_n_boot : int
        Bootstrap replications.
    wild_weight_type : {'rademacher', 'webb', 'mammen'}
    wild_seed : int, optional
    alpha : float
    drop_singletons : bool
    tol, maxiter : convergence controls for the absorber.

    Returns
    -------
    FEOLSResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.feols("lwage ~ educ + exper | firm + year", data=df,
    ...                cluster='firm')
    >>> print(res.summary())
    """
    lhs, x_vars, fe_vars = _parse_formula(formula)

    # Collect all columns (y, x's, fe's, cluster, weight)
    cols = [lhs] + list(x_vars) + list(fe_vars)
    if cluster is not None:
        cluster_names = [cluster] if isinstance(cluster, str) else list(cluster)
        cols += cluster_names
    else:
        cluster_names = []
    w_col = None
    if isinstance(weights, str):
        cols.append(weights)
        w_col = weights

    df = data[list(dict.fromkeys(cols))].dropna().copy()
    if len(df) == 0:
        raise ValueError("No non-missing rows remaining after dropna.")

    y_arr = df[lhs].to_numpy(dtype=np.float64)
    if not x_vars:
        # Pure absorption (predict y from FE only) — trivial. Fit a constant.
        X_arr = np.ones((len(df), 1))
        x_names = ["_const"]
    else:
        X_arr = df[x_vars].to_numpy(dtype=np.float64)
        x_names = list(x_vars)

    if fe_vars:
        fe_mat = df[fe_vars].to_numpy()
    else:
        # No FE → fall back to plain OLS with intercept
        if not x_vars:
            raise ValueError("Need at least one regressor or one FE.")
        return _ols_no_fe(df, lhs, x_vars, weights, cluster_names, alpha, formula)

    w_arr = None
    if w_col is not None:
        w_arr = df[w_col].to_numpy(dtype=np.float64)
    elif isinstance(weights, np.ndarray):
        w_arr = np.asarray(weights, dtype=np.float64)

    cluster_arr = None
    if cluster_names:
        cluster_arr = [df[c].to_numpy() for c in cluster_names]
        if len(cluster_arr) == 1:
            cluster_arr = cluster_arr[0]

    result = absorb_ols(
        y=y_arr,
        X=X_arr,
        fe=fe_mat,
        weights=w_arr,
        cluster=cluster_arr,
        drop_singletons=drop_singletons,
        tol=tol,
        maxiter=maxiter,
        return_absorber=True,
    )

    coef = pd.Series(result["coef"], index=x_names, name="coef")
    se = pd.Series(result["se"], index=x_names, name="std_err")
    vcov = result["vcov"]

    df_resid = result["df_resid"]
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    t_stats = coef / se.replace(0, np.nan)
    pvals = pd.Series(
        2 * (1 - stats.t.cdf(np.abs(t_stats.fillna(0)), df_resid)),
        index=x_names,
    )
    ci_lo = coef - t_crit * se
    ci_hi = coef + t_crit * se

    # Inferred SE type
    if se_type is None:
        if cluster_names:
            se_type = "cluster" if len(cluster_names) == 1 else "multiway_cluster"
        else:
            se_type = "iid"
    cluster_info = {}
    if cluster_names:
        cluster_info = {
            "cluster": cluster_names,
            "n_clusters": [
                int(pd.Series(df[c]).nunique())
                for c in cluster_names
            ],
        }

    # Optional wild bootstrap
    if wild and cluster_names:
        if len(cluster_names) > 1:
            raise NotImplementedError(
                "Wild cluster bootstrap with multi-way clustering is not yet supported."
            )
        from ..inference.wild_bootstrap import wild_cluster_bootstrap
        # Run per x_var on the absorbed model: we regress y_tilde on X_tilde.
        # Build a temporary DataFrame of within-residualized variables.
        ab = result["absorber"]
        mask = ab.keep_mask
        df_sub = df.iloc[mask].reset_index(drop=True)
        yw = ab.demean(df_sub[lhs].to_numpy(dtype=np.float64), copy=True, already_masked=True)
        Xw = ab.demean(df_sub[x_vars].to_numpy(dtype=np.float64), copy=True, already_masked=True)
        cl_w = df_sub[cluster_names[0]].to_numpy()

        wild_data = pd.DataFrame(
            {"_y": yw, **{f"_x{i}": Xw[:, i] for i in range(Xw.shape[1])}, "_cl": cl_w}
        )
        p_wild: Dict[str, float] = {}
        ci_wild: Dict[str, tuple] = {}
        for i, name in enumerate(x_names):
            res_w = wild_cluster_bootstrap(
                wild_data,
                y="_y",
                x=[f"_x{j}" for j in range(Xw.shape[1])],
                cluster="_cl",
                test_var=f"_x{i}",
                h0=0.0,
                n_boot=wild_n_boot,
                weight_type=wild_weight_type,
                seed=wild_seed,
                alpha=alpha,
            )
            p_wild[name] = res_w["p_boot"]
            ci_wild[name] = res_w["ci_boot"]
        cluster_info["wild_p"] = p_wild
        cluster_info["wild_ci"] = ci_wild
        se_type = "wild_cluster"

    return FEOLSResult(
        params=coef,
        std_errors=se,
        vcov=vcov,
        tvalues=t_stats.fillna(0.0),
        pvalues=pvals,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        residuals=result["resid"],
        fitted_within=result["fitted_within"],
        n_obs=result["n"],
        n_singletons_dropped=result["n_singletons_dropped"],
        n_fe=result["n_fe"],
        dof_fe=result["dof_fe"],
        df_resid=df_resid,
        r2_within=result["r2_within"],
        se_type=se_type,
        cluster_info=cluster_info,
        formula=formula,
        absorber=result["absorber"],
        converged=result["converged"],
        iters=result["iters"],
    )


# ======================================================================
# Fallback: no-FE path
# ======================================================================


def _ols_no_fe(
    df: pd.DataFrame,
    lhs: str,
    x_vars: List[str],
    weights,
    cluster_names: List[str],
    alpha: float,
    formula: str,
) -> FEOLSResult:
    """Plain OLS with intercept when no FE is absorbed."""
    y = df[lhs].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(df)), df[x_vars].to_numpy(dtype=np.float64)])
    names = ["_const"] + list(x_vars)
    n, k = X.shape

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    df_resid = n - k
    sigma2 = float((resid ** 2).sum()) / df_resid

    if cluster_names:
        from ..inference.multiway_cluster import multiway_cluster_vcov
        cl = (
            df[cluster_names[0]].to_numpy()
            if len(cluster_names) == 1
            else [df[c].to_numpy() for c in cluster_names]
        )
        vcov = multiway_cluster_vcov(X, resid, cl, df_adjust=True, n_params=k)
        se_type = "cluster" if len(cluster_names) == 1 else "multiway_cluster"
    else:
        vcov = sigma2 * XtX_inv
        se_type = "iid"

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    t_stats = coef / np.where(se > 0, se, np.nan)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    pvals = 2 * (1 - stats.t.cdf(np.abs(np.nan_to_num(t_stats)), df_resid))

    y_bar = y.mean()
    ss_res = float(((y - X @ coef) ** 2).sum())
    ss_tot = float(((y - y_bar) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Minimal Absorber stub (identity) — returned in the ``absorber``
    # field for API symmetry when the user asks for a no-FE regression.
    class _Identity:
        keep_mask = np.ones(n, dtype=bool)
        n_kept = n
        n_dropped = 0
        n_fe: list = []
    ab = _Identity()

    return FEOLSResult(
        params=pd.Series(coef, index=names),
        std_errors=pd.Series(se, index=names),
        vcov=vcov,
        tvalues=pd.Series(t_stats, index=names).fillna(0.0),
        pvalues=pd.Series(pvals, index=names),
        conf_int_lower=pd.Series(coef - t_crit * se, index=names),
        conf_int_upper=pd.Series(coef + t_crit * se, index=names),
        residuals=resid,
        fitted_within=X @ coef,
        n_obs=n,
        n_singletons_dropped=0,
        n_fe=[],
        dof_fe=0,
        df_resid=df_resid,
        r2_within=r2,
        se_type=se_type,
        cluster_info={"cluster": cluster_names} if cluster_names else {},
        formula=formula,
        absorber=ab,  # type: ignore[arg-type]
        converged=True,
        iters=0,
    )


hdfe_ols = feols  # alias for top-level namespace export (avoids collision
# with the pyfixest-backed ``sp.feols`` wrapper).


__all__ = ["feols", "hdfe_ols", "FEOLSResult"]
