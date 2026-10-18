"""StatsPAI's own engine behind ``sp.feols`` / ``sp.fepois`` (``backend='auto'``).

``sp.feols`` and ``sp.fepois`` reached R ``fixest``'s numbers through
pyfixest, an optional dependency, and paid its fixed cost on every call.
The native fit (``sp.fast.feols`` / ``sp.fast.fepois`` on the optional
Rust kernels or their NumPy fallback) computes the same estimator with
fixest's small-sample conventions. This module decides, per call, whether
the native fit can answer it exactly as the delegate would, runs it, and
reports it in the delegate's result layout.

A call is native when every part of it is something the native fit
implements: ``lhs ~ x1 + ... | fe1 + ...`` with numeric regressors and
plain fixed-effect columns, ``vcov`` in iid / hetero / one-way CRV1, no
``ssc=``, no extra pyfixest keywords. Rows with a missing value in a used
column are dropped first, as fixest does. Anything else, and a design
whose regressors are absorbed or collinear (fixest drops those columns;
the native fit refuses them), goes to pyfixest; :func:`plan` says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.results import EconometricResults

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_HETERO = ("hetero", "hc1", "robust")


@dataclass
class Plan:
    """A call the native fit can answer, in its own terms."""

    lhs: str
    rhs: List[str]
    fe: List[str]
    vcov: str  # 'iid' | 'hc1' | 'cr1'
    cluster: Optional[str]
    weights: Optional[str]
    drop_singletons: bool
    data: pd.DataFrame
    n_missing: int

    @property
    def formula(self) -> str:
        rhs = " + ".join(self.rhs) if self.rhs else "1"
        if self.fe:
            return f"{self.lhs} ~ {rhs} | {' + '.join(self.fe)}"
        return f"{self.lhs} ~ 1 + {rhs}" if self.rhs else f"{self.lhs} ~ 1"

    @property
    def vcov_label(self) -> str:
        return {"iid": "iid", "hc1": "hetero", "cr1": "CRV"}[self.vcov]

    @property
    def vcov_detail(self) -> str:
        return {"iid": "iid", "hc1": "hetero", "cr1": "CRV1"}[self.vcov]


def _split_formula(fml: Any) -> Optional[Tuple[str, List[str], List[str]]]:
    if not isinstance(fml, str) or fml.count("~") != 1 or fml.count("|") > 1:
        return None
    lhs, rest = (s.strip() for s in fml.split("~"))
    rhs_s, _, fe_s = rest.partition("|")
    rhs = [t.strip() for t in rhs_s.split("+")]
    fe = [t.strip() for t in fe_s.split("+")] if fe_s.strip() else []
    if not _NAME.match(lhs) or any(not t for t in rhs + fe):
        return None
    if fe_s and not fe_s.strip():
        return None
    rhs = [t for t in rhs if t != "1"]
    if not rhs or any(not _NAME.match(t) for t in rhs + fe):
        return None
    names = [lhs] + rhs + fe
    if len(set(names)) != len(names):
        return None
    return lhs, rhs, fe


def _resolve_vcov(
    vcov: Any, cluster: Optional[str]
) -> Optional[Tuple[str, Optional[str]]]:
    if vcov is None:
        return ("cr1", cluster) if cluster is not None else ("iid", None)
    if cluster is not None:
        # sp.feols hands ``cluster=`` to pyfixest only when vcov is unset.
        return None
    if isinstance(vcov, str):
        v = vcov.strip().lower()
        if v == "iid":
            return "iid", None
        if v in _HETERO:
            return "hc1", None
        return None
    if isinstance(vcov, dict) and len(vcov) == 1:
        ((kind, col),) = vcov.items()
        if str(kind).upper() == "CRV1" and isinstance(col, str):
            col = col.strip()
            if _NAME.match(col):
                return "cr1", col
    return None


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def plan(
    fml: Any,
    data: Any,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    extra: Dict[str, Any],
    poisson: bool = False,
    binomial: bool = False,
) -> Union[Plan, str]:
    """A :class:`Plan` for this call, or the reason it goes to pyfixest."""
    if not isinstance(data, pd.DataFrame):
        return "data is not a pandas DataFrame"
    if ssc is not None:
        return "ssc= is a pyfixest object"
    if extra:
        return f"pyfixest keywords {sorted(extra)}"
    if fixef_rm not in ("none", "singleton"):
        return f"fixef_rm={fixef_rm!r}"
    parts = _split_formula(fml)
    if parts is None:
        return "formula uses syntax beyond 'y ~ x1 + x2 | fe1 + fe2'"
    lhs, rhs, fe = parts
    resolved = _resolve_vcov(vcov, cluster)
    if resolved is None:
        return f"vcov={vcov!r} (cluster={cluster!r})"
    kind, cl = resolved
    used = [lhs] + rhs + fe + ([weights] if weights is not None else [])
    if cl is not None and cl not in used:
        used.append(cl)
    if any(c not in data.columns for c in used):
        return "a column is missing"  # pyfixest reports it by name
    if data.columns.duplicated().any():
        return "duplicated column names"
    for c in [lhs] + rhs + ([weights] if weights is not None else []):
        if not _numeric(data[c]):
            return f"column {c!r} is not numeric"
    # Rows fixest drops: a missing value in any used column. A missing
    # cluster id is left to pyfixest, which reports it.
    missing = np.zeros(len(data), dtype=bool)
    for c in used:
        col = data[c]
        arr = col.to_numpy()
        if arr.dtype.kind == "f":
            # One pass for the common case: every value finite.
            if np.isfinite(arr).all():
                continue
            if np.isinf(arr).any():
                return f"column {c!r} has infinite values"
        if col.hasnans:
            if c == cl:
                return f"cluster column {c!r} has missing values"
            missing |= col.isna().to_numpy()
    n_missing = int(missing.sum())
    work = data.loc[~missing, used] if n_missing else data
    if len(work) == 0:
        return "no complete rows"
    if weights is not None and (work[weights].to_numpy() <= 0).any():
        return "non-positive weights"
    if poisson and (work[lhs].to_numpy() < 0).any():
        return "negative Poisson outcome"
    if binomial:
        yb = work[lhs].to_numpy()
        if ((yb < 0) | (yb > 1)).any():
            return "binomial outcome outside [0, 1]"
    return Plan(
        lhs=lhs,
        rhs=rhs,
        fe=fe,
        vcov=kind,
        cluster=cl,
        weights=weights,
        drop_singletons=(fixef_rm == "singleton"),
        data=work,
        n_missing=n_missing,
    )


def _refused(exc: Exception) -> bool:
    """The native fit refused a design that fixest fits by dropping columns."""
    from ..exceptions import NumericalInstability

    diag = getattr(exc, "diagnostics", None) or {}
    if isinstance(exc, NumericalInstability):
        return bool({"absorbed", "collinear"} & set(diag)) or (
            "singular" in str(exc).lower()
        )
    return False


def _model_info(p: Plan, family: str, kernel: str, fml: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_type": f"{family} (StatsPAI native)",
        "method": "High-Dimensional Fixed Effects" if p.fe else "OLS",
        "formula": fml,
        "vcov_type": p.vcov_label,
        "vcov_type_detail": p.vcov_detail,
        "clustervar": [p.cluster] if p.cluster else [],
        "fixed_effects": " + ".join(p.fe) if p.fe else "None",
        "backend": "statspai-native",
        "implementation": "native",
        "fe_kernel": kernel,
        "weights_name": p.weights,
    }
    if p.n_missing:
        info["n_missing_dropped"] = p.n_missing
    return info


def _inference(
    params: pd.Series,
    V: np.ndarray,
    p: Plan,
    df_t: int,
    normal: bool,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    from scipy import stats

    se = pd.Series(np.sqrt(np.clip(np.diag(V), 0.0, None)), index=params.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = params / se
    if normal:
        pv = 2.0 * stats.norm.sf(np.abs(t.to_numpy()))
    else:
        pv = 2.0 * stats.t.sf(np.abs(t.to_numpy()), max(df_t, 1))
    return se, t, pd.Series(pv, index=params.index)


def fit_feols(p: Plan, fml: str, *, lean: bool = False) -> EconometricResults:
    """Fit a planned ``sp.feols`` call natively, in the delegate's layout."""
    from ..fast.feols import _feols_fit

    res, a = _feols_fit(
        p.formula,
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
        drop_singletons=p.drop_singletons,
        note_weights=False,
    )
    names = ["Intercept" if c == "(Intercept)" else c for c in res.coef_names]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    w = np.asarray(a["w"], dtype=float)
    weighted = p.weights is not None
    if p.vcov == "cr1":
        G = int(a["n_clusters"])
        df_k, df_t = a["df_k_cluster"], G - 1
    else:
        G = 0
        df_k = a["df_k"]
        df_t = N - df_k
    se, t, pv = _inference(params, V, p, df_t, normal=False)

    # R-squared family, as pyfixest computes them (weighted sums of squares).
    u = np.asarray(a["resid"], dtype=float)
    y = np.asarray(a["y"], dtype=float)
    ssu = float(w @ (u * u))
    ssy = float(w @ (y - np.average(y, weights=w)) ** 2)
    has_intercept = 1
    diagnostics: Dict[str, Any] = {}
    if p.fe:
        k_fe = int(np.sum(np.asarray(res.fe_cardinality) - 1) + 1)
        adj = (N - has_intercept) / (N - k - k_fe)
        adj_within = (N - k_fe) / (N - k - k_fe)
        y_dem = np.asarray(a["y_dem"], dtype=float)
        ssy_within = float(w @ (y_dem * y_dem))
        r2_within = 1.0 - ssu / ssy_within
        adj_r2_within = 1.0 - ssu / ssy_within * adj_within
    else:
        adj = (N - has_intercept) / (N - k)
        r2_within = adj_r2_within = float("nan")
    diagnostics["R-squared"] = 1.0 - ssu / ssy
    diagnostics["Adj. R-squared"] = 1.0 - ssu / ssy * adj
    diagnostics["R-squared (within)"] = r2_within
    diagnostics["Adj. R-squared (within)"] = adj_r2_within
    diagnostics["RMSE"] = float(np.sqrt(ssu / N))

    model_info = _model_info(p, "OLS", res.fe_kernel, fml)
    if G:
        model_info["n_clusters"] = G
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": u,
        "fitted_values": y - u,
    }
    if G:
        data_info["df_inference"] = G - 1
    if not lean:
        sw = np.sqrt(w)[:, None] if weighted else None
        X = np.asarray(a["X_dem"], dtype=float)
        yd = np.asarray(a["y_dem"], dtype=float)
        data_info["X"] = X * sw if weighted else X
        data_info["y"] = yd * sw[:, 0] if weighted else yd
        data_info["var_names"] = names
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    return out


def fit_fepois(p: Plan, fml: str, *, tol: float, maxiter: int) -> EconometricResults:
    """Fit a planned ``sp.fepois`` call natively, in the delegate's layout."""
    from ..fast.fepois import _fepois_fit

    res, a = _fepois_fit(
        p.formula,
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
        drop_singletons=p.drop_singletons,
        # A relative deviance change of 1e-8 (fixest's and pyfixest's stop)
        # can leave the fixed effects, and so the standard errors, 1e-5
        # from the optimum; one more Newton step puts them within ~1e-9.
        tol=min(tol, 1e-10),
        maxiter=maxiter,
        note_weights=False,
    )
    names = ["Intercept" if c == "(Intercept)" else c for c in res.coef_names]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    if p.vcov == "cr1":
        G = int(pd.unique(a["cluster"]).size)
        df_k = a["df_k_cluster"]
    else:
        G = 0
        df_k = a["df_k"]
    se, t, pv = _inference(params, V, p, N - df_k, normal=True)
    y = np.asarray(a["y"], dtype=float)
    mu = np.asarray(a["mu"], dtype=float)
    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
        "Deviance": float(res.deviance),
        "Log-likelihood": float(res.log_likelihood),
    }
    model_info = _model_info(p, "Poisson", res.fe_kernel, fml)
    model_info["converged"] = bool(res.converged)
    model_info["iterations"] = int(res.iterations)
    if res.n_dropped_separation:
        model_info["n_dropped_separation"] = int(res.n_dropped_separation)
        import warnings

        warnings.warn(
            f"sp.fepois: {res.n_dropped_separation} observations removed "
            "because of separation (fixed-effect groups whose outcomes are "
            "all zero).",
            UserWarning,
            stacklevel=4,
        )
    if res.n_dropped_singletons:
        model_info["n_dropped_singletons"] = int(res.n_dropped_singletons)
    if G:
        model_info["n_clusters"] = G
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": y - mu,
        "fitted_values": mu,
    }
    if G:
        data_info["df_inference"] = G - 1
    if not res.converged:
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn(
            f"sp.fepois: IRLS stopped after {res.iterations} iterations "
            f"without meeting iwls_tol={tol}.",
            ConvergenceWarning,
            stacklevel=4,
        )
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    return out


def fit_feglm(p: Plan, fml: str, *, family: str) -> EconometricResults:
    """Fit a planned ``sp.feglm`` logit / probit call natively."""
    from ..fast._feglm import feglm_fit

    res, a = feglm_fit(
        p.lhs,
        p.rhs,
        p.fe,
        p.data,
        family=family,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
    )
    names = ["Intercept" if c == "(Intercept)" else c for c in res.coef_names]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    G = int(a["n_clusters"]) if p.vcov == "cr1" else 0
    df_k = a["df_k_cluster"] if G else a["df_k"]
    se, t, pv = _inference(params, V, p, N - df_k, normal=True)
    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
        "Deviance": float(res.deviance),
        "Log-likelihood": float(res.log_likelihood),
    }
    model_info = _model_info(p, "GLM", res.fe_kernel, fml)
    model_info["family"] = family
    model_info["converged"] = bool(res.converged)
    model_info["iterations"] = int(res.iterations)
    if res.n_dropped_separation:
        model_info["n_dropped_separation"] = int(res.n_dropped_separation)
        import warnings

        warnings.warn(
            f"sp.feglm: {res.n_dropped_separation} observations removed "
            "because their fixed-effect group has only 0 or only 1 outcomes "
            "(perfect fit), as fixest does.",
            UserWarning,
            stacklevel=4,
        )
    if G:
        model_info["n_clusters"] = G
    y = np.asarray(a["y"], dtype=float)
    mu = np.asarray(a["mu"], dtype=float)
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": y - mu,
        "fitted_values": mu,
    }
    if G:
        data_info["df_inference"] = G - 1
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    return out


def refused(exc: Exception) -> bool:
    """Whether ``exc`` is a native refusal that pyfixest can resolve."""
    return _refused(exc)


__all__ = ["Plan", "plan", "fit_feols", "fit_fepois", "fit_feglm", "refused"]
