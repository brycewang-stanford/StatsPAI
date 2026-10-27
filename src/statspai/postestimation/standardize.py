"""Standardised (beta) coefficients of a fitted linear model.

``sp.standardize_coefs(result)`` rescales a least-squares or 2SLS fit to the
coefficients of the same regression run on standardised variables. Stata's
``regress ..., beta`` column is the slope part of it; the AER convention of
z-scoring the variables before ``ivreg2`` / ``reghdfe`` is all of it. The
two agree on every slope.

The map is a fixed linear reparametrisation, so it is exact for every
variance the fit carries (classical, HC, cluster, Conley, HAC): with
``D = diag(c)`` plus the intercept row the covariance becomes ``J V J'``,
the standard errors and interval ends of a slope are multiplied by its
``c > 0`` and its t statistic and p-value do not change.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import (
    ColumnNotFound,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)

__all__ = ["standardize_coefs"]

_INTERCEPT_NAMES = ("Intercept", "(Intercept)", "const", "_cons", "Const")

#: data_info entries that fix the reference distribution of t / p / CI
_INFERENCE_KEYS = (
    "nobs",
    "df_resid",
    "df_model",
    "df_inference",
    "df_by_coefficient",
    "inference",
    "dependent_var",
    "hotelling_df",
)


def _moments(values: np.ndarray, w: Optional[np.ndarray]) -> Tuple[float, float]:
    """Mean and standard deviation (``n - 1`` divisor; Stata ``[aw]`` with
    weights, i.e. weights rescaled to sum to ``n``)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        raise DataInsufficient(
            "standardize_coefs needs at least two observations.",
            recovery_hint="Fit the model on more than one row.",
        )
    if w is None:
        return float(v.mean()), float(v.std(ddof=1))
    ww = np.asarray(w, dtype=float) * (n / float(np.sum(w)))
    m = float(np.sum(ww * v) / n)
    return m, float(np.sqrt(np.sum(ww * (v - m) ** 2) / (n - 1)))


def _params_and_vcov(result: Any) -> Tuple[pd.Series, pd.Series, Optional[np.ndarray]]:
    params = pd.Series(result.params, dtype=float).copy()
    se = pd.Series(result.std_errors, dtype=float).reindex(params.index)
    V: Optional[np.ndarray] = None
    getter = getattr(result, "cov_params", None)
    if callable(getter):
        try:
            cov = getter()
        except MethodIncompatibility:
            # the fit replaced its SEs (Conley, bootstrap) without storing
            # the matrix: slopes still scale by c * se; the intercept is
            # reported as unavailable by the caller
            cov = None
        if cov is not None:
            V = np.asarray(
                pd.DataFrame(cov).loc[params.index, params.index], dtype=float
            )
            if not np.allclose(
                np.sqrt(np.clip(np.diag(V), 0, None)),
                se.to_numpy(dtype=float),
                rtol=1e-8,
                atol=0.0,
                equal_nan=True,
            ):
                V = None  # stored matrix is not the one behind the SEs
    return params, se, V


def _is_absorbed(result: Any) -> bool:
    if type(result).__name__ == "FEOLSResult":
        return True
    mi = getattr(result, "model_info", None) or {}
    if mi.get("fixed_effects") or mi.get("absorb"):
        return True
    iv = (getattr(result, "data_info", None) or {}).get("iv") or {}
    return bool(iv.get("fe_codes"))


def _stored_design(
    result: Any,
) -> Optional[Tuple[np.ndarray, List[str], np.ndarray, Optional[np.ndarray], bool]]:
    """Raw design, names, outcome and weights of a fit that stored them.

    Returns ``(X, names, y, w, weighted_unknown)``. ``None`` when the fit
    stores no raw design (absorbed fixed effects, foreign result types).
    """
    di = getattr(result, "data_info", None) or {}
    iv = di.get("iv")
    if isinstance(iv, dict) and {"X", "y", "var_names"} <= set(iv):
        weighted = bool(iv.get("weighted"))
        return (
            np.asarray(iv["X"], dtype=float),
            [str(c) for c in iv["var_names"]],
            np.asarray(iv["y"], dtype=float).ravel(),
            None,
            weighted,
        )
    if {"X", "y", "var_names"} <= set(di):
        w = di.get("analytic_weights")
        mi = getattr(result, "model_info", None) or {}
        weighted_unknown = w is None and bool(mi.get("weights_name"))
        return (
            np.asarray(di["X"], dtype=float),
            [str(c) for c in di["var_names"]],
            np.asarray(di["y"], dtype=float).ravel(),
            None if w is None else np.asarray(w, dtype=float).ravel(),
            weighted_unknown,
        )
    return None


def _dependent_var(result: Any) -> Optional[str]:
    di = getattr(result, "data_info", None) or {}
    mi = getattr(result, "model_info", None) or {}
    dep = di.get("dependent_var") or mi.get("dependent_var")
    if isinstance(dep, str):
        return dep
    formula = getattr(result, "formula", None) or mi.get("formula")
    if isinstance(formula, str) and "~" in formula:
        return formula.split("~", 1)[0].strip()
    return None


def _resolve_weights(
    weights: Union[None, str, Sequence[float], np.ndarray],
    frame: Optional[pd.DataFrame],
    n: int,
) -> Optional[np.ndarray]:
    if weights is None:
        return None
    if isinstance(weights, str):
        if frame is None or weights not in frame.columns:
            raise ColumnNotFound(
                f"standardize_coefs: weight column {weights!r} is not in data.",
                recovery_hint="Pass data= with the weight column, or the "
                "weights themselves as an array.",
            )
        w = frame[weights].to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
    if w.size != n or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise MethodIncompatibility(
            f"standardize_coefs: weights must be {n} finite non-negative "
            f"numbers aligned with the estimation sample (got {w.size}).",
            recovery_hint="Pass the weights of the estimation sample.",
        )
    return w


def _sample_frame(
    result: Any, data: pd.DataFrame, columns: List[str], n: int
) -> pd.DataFrame:
    """The estimation-sample rows of ``data`` for ``columns``."""
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"standardize_coefs: column(s) {missing} are not in data. With "
            "absorbed fixed effects the standard deviations are those of the "
            "raw variables, so every reported coefficient must be a column "
            "of data (interactions and factor terms included).",
            recovery_hint="Create the term as a column, refit on it, and pass "
            "that frame.",
            diagnostics={"missing": missing},
        )
    frame = data.loc[:, list(dict.fromkeys(columns))]
    if len(frame) != n:
        frame = frame.dropna()
    if len(frame) != n:
        keep = getattr(getattr(result, "absorber", None), "keep_mask", None)
        if keep is not None and len(keep) == len(frame):
            frame = frame.loc[np.asarray(keep, dtype=bool)]
    if len(frame) != n:
        raise MethodIncompatibility(
            f"standardize_coefs: data has {len(frame)} usable rows but the fit "
            f"used {n}; the standard deviations must be taken on the "
            "estimation sample.",
            recovery_hint="Pass the frame the model was fitted on, restricted "
            "to the estimation sample (rows dropped for missing values or as "
            "singleton fixed-effect groups removed).",
            diagnostics={"rows": len(frame), "nobs": n},
        )
    return frame


def standardize_coefs(
    result: Any,
    data: Optional[pd.DataFrame] = None,
    *,
    variables: Optional[Sequence[str]] = None,
    scale_y: bool = True,
    weights: Union[None, str, Sequence[float], np.ndarray] = None,
) -> Any:
    """Standardised (beta) coefficients of a least-squares or 2SLS fit.

    Returns the fit re-expressed as the regression on standardised
    variables: each slope becomes ``b_k * sd(x_k) / sd(y)`` (Stata
    ``regress ..., beta``), and with an intercept the variables are also
    centred, so the result equals running the same estimator (``regress``,
    ``ivreg2``, ``reghdfe``) on z-scored variables, which is how AER papers
    produce "standardized coefficients" for IV and fixed-effects models.

    What is scaled: the coefficient, its standard error and both interval
    ends of slope ``k`` are multiplied by the same ``c_k > 0``
    (``sd(x_k) / sd(y)``); the covariance becomes ``J V J'``. The t
    statistic and p-value of every slope are unchanged, for any variance
    the fit used (classical, robust, cluster, Conley, HAC): the map is a
    fixed reparametrisation of the design. The intercept is the intercept of
    the centred regression; Stata's ``beta`` column leaves it blank.

    Parameters
    ----------
    result : fitted result
        ``sp.regress``, ``sp.iv`` / ``sp.ivreg`` (any method), ``sp.feols``
        or ``sp.hdfe_ols``. Results with absorbed fixed effects need
        ``data``.
    data : DataFrame, optional
        Needed when fixed effects were absorbed (the stored design is then
        demeaned): standard deviations are those of the raw columns on the
        estimation sample, as when the variables are z-scored before
        ``reghdfe``. Every reported coefficient must then be a column.
    variables : sequence of str, optional
        Regressors to standardise (default all except the intercept). The
        others are scaled by ``1 / sd(y)`` only, as when only the outcome
        and those variables are z-scored.
    scale_y : bool, default True
        Divide by ``sd(y)`` as well. ``False`` gives "effect of a one-sd
        change in x in units of y".
    weights : str or array, optional
        Weights of the moments when the fit is weighted but does not carry
        its weights row by row (2SLS, ``sp.feols``, ``sp.hdfe_ols``).
        ``sp.regress(weights=)`` fits carry theirs. Weighted means and
        standard deviations follow Stata ``summarize [aw=]``.

    Returns
    -------
    EconometricResults
        A new result whose ``params`` / ``std_errors`` / intervals are the
        standardised ones and whose t / p values equal the fit's. The
        scale factors, standard deviations and means are in
        ``model_info['standardized']``. The returned object reports; it
        does not predict (its design is not stored).

    Raises
    ------
    MethodIncompatibility
        A weighted fit without retrievable weights; data not aligned with
        the estimation sample; a constant regressor or outcome.
    ColumnNotFound
        With absorbed fixed effects, a coefficient that is not a column.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200), "w": rng.normal(size=200)})
    >>> df["y"] = 1 + 2 * df.x - df.w + rng.normal(size=200)
    >>> fit = sp.regress("y ~ x + w", data=df, vce="robust")
    >>> beta = sp.standardize_coefs(fit)
    >>> b = fit.params["x"] * df.x.std() / df.y.std()
    >>> bool(np.isclose(beta.params["x"], b))
    True
    >>> bool(np.isclose(beta.tvalues["x"], fit.tvalues["x"]))
    True

    References
    ----------
    [@lee2022valid] for the IV reporting context; the transformation itself
    is an exact reparametrisation and needs no citation.
    """
    params, se, V = _params_and_vcov(result)
    names = [str(c) for c in params.index]
    di = dict(getattr(result, "data_info", None) or {})
    n = int(di.get("nobs") or getattr(result, "nobs", 0) or 0)
    absorbed = _is_absorbed(result)
    intercept = next((c for c in names if c in _INTERCEPT_NAMES), None)
    if absorbed:
        intercept = None

    stored = None if absorbed else _stored_design(result)
    cols: Dict[str, np.ndarray] = {}
    w: Optional[np.ndarray] = None
    if stored is not None:
        X, xnames, y, w_stored, weighted_unknown = stored
        n = X.shape[0]
        for j, c in enumerate(xnames):
            cols[c] = X[:, j]
        missing = [c for c in names if c not in cols]
        if missing:
            raise MethodIncompatibility(
                f"standardize_coefs: the stored design has no column for "
                f"{missing}.",
                recovery_hint="Refit with sp.regress / sp.iv and pass that "
                "result unchanged.",
            )
        if w_stored is not None:
            w = w_stored
        elif weighted_unknown or weights is not None:
            if weights is None:
                raise MethodIncompatibility(
                    "standardize_coefs: the fit is weighted but does not carry "
                    "its weights row by row.",
                    recovery_hint="Pass weights= (a column of data or the "
                    "estimation-sample weights).",
                )
            frame = None
            if isinstance(weights, str):
                if data is None:
                    raise MethodIncompatibility(
                        "standardize_coefs: weights given by name need data=.",
                        recovery_hint="Pass data= as well.",
                    )
                frame = _sample_frame(result, data, [weights], n)
            w = _resolve_weights(weights, frame, n)
    else:
        if data is None:
            raise MethodIncompatibility(
                "standardize_coefs: this fit absorbed fixed effects (or stores "
                "no raw design), so the standard deviations of the raw "
                "variables must come from data=.",
                recovery_hint="Call sp.standardize_coefs(result, data=df) "
                "with the frame the model was fitted on.",
            )
        dep = _dependent_var(result)
        if dep is None:
            raise MethodIncompatibility(
                "standardize_coefs: cannot tell the outcome column of this fit.",
                recovery_hint="Fit with a formula so the outcome is recorded.",
            )
        mi = getattr(result, "model_info", None) or {}
        fe_cols: List[str] = []
        fe = mi.get("fixed_effects") or mi.get("absorb")
        if isinstance(fe, str):
            fe_cols = [f for f in fe.replace("+", " ").split() if f in data.columns]
        elif isinstance(fe, (list, tuple)):
            fe_cols = [str(f) for f in fe if str(f) in data.columns]
        wname = mi.get("weights_name") if weights is None else weights
        need = [dep] + names + fe_cols
        if isinstance(wname, str):
            need.append(wname)
        frame = _sample_frame(result, data, need, n)
        y = frame[dep].to_numpy(dtype=float)
        for c in names:
            cols[c] = frame[c].to_numpy(dtype=float)
        if isinstance(wname, str):
            w = _resolve_weights(wname, frame, n)
        elif wname is not None:
            w = _resolve_weights(wname, None, n)
        elif getattr(result, "weight_info", None):
            raise MethodIncompatibility(
                "standardize_coefs: the fit is weighted; pass weights=.",
                recovery_hint="weights='<column>' as given to the estimator.",
            )

    my, sy = _moments(y, w)
    rel = max(abs(my), 1.0)
    if not np.isfinite(sy) or sy <= 1e-12 * rel:
        raise MethodIncompatibility(
            "standardize_coefs: the outcome is constant on the estimation "
            "sample, so it has no standard deviation to scale by.",
            recovery_hint="Use scale_y=False.",
        )
    if variables is None:
        chosen = [c for c in names if c != intercept]
    else:
        chosen = [str(v) for v in variables]
        unknown = [v for v in chosen if v not in names or v == intercept]
        if unknown:
            raise MethodIncompatibility(
                f"standardize_coefs: {unknown} are not slope coefficients of "
                "this fit.",
                recovery_hint=f"Choose from {[c for c in names if c != intercept]}.",
            )
    d = 1.0 / sy if scale_y else 1.0
    omitted = {
        str(o.get("variable")) if isinstance(o, dict) else str(o)
        for o in ((getattr(result, "model_info", None) or {}).get("omitted") or ())
    }
    k = len(names)
    J = np.zeros((k, k))
    sds: Dict[str, float] = {}
    means: Dict[str, float] = {}
    for j, c in enumerate(names):
        if c == intercept:
            continue
        if c in chosen:
            m, s = _moments(cols[c], w)
            sds[c], means[c] = s, m
            if not np.isfinite(s) or s <= 1e-12 * max(abs(m), 1.0):
                if c in omitted:
                    J[j, j] = np.nan
                    continue
                raise MethodIncompatibility(
                    f"standardize_coefs: regressor {c!r} is constant on the "
                    "estimation sample.",
                    recovery_hint="Leave it out of variables= (then it is "
                    "scaled by 1 / sd(y) only).",
                )
            J[j, j] = d * s
        else:
            J[j, j] = d
    if intercept is not None:
        i0 = names.index(intercept)
        J[i0, i0] = d
        for c in chosen:
            J[i0, names.index(c)] = d * means[c]
    shift = np.zeros(k)
    if intercept is not None and scale_y:
        shift[names.index(intercept)] = -d * my

    intercept_unavailable = False
    b = params.to_numpy(dtype=float)
    b_new = J @ np.nan_to_num(b) + shift
    scale = np.diag(J).copy()
    se_arr = se.to_numpy(dtype=float)
    if V is not None and np.all(np.isfinite(V)):
        V_new = J @ V @ J.T
        se_new = np.sqrt(np.clip(np.diag(V_new), 0.0, None))
        # slopes: the same number as c * se (no rounding through V)
        slope = np.array([c != intercept for c in names])
        se_new[slope] = scale[slope] * se_arr[slope]
    else:
        V_new = None
        se_new = scale * se_arr
        if intercept is not None:
            se_new[names.index(intercept)] = np.nan
            intercept_unavailable = True
    for j, c in enumerate(names):
        if not np.isfinite(b[j]):
            b_new[j] = np.nan

    from ..core.results import EconometricResults

    data_info = {key: di[key] for key in _INFERENCE_KEYS if key in di}
    data_info.setdefault("nobs", n)
    if "df_inference" not in data_info:
        dfi = getattr(result, "df_inference", None)
        if dfi is not None:
            data_info["df_inference"] = dfi
    if "df_resid" not in data_info and getattr(result, "df_resid", None) is not None:
        data_info["df_resid"] = result.df_resid
    if V_new is not None:
        data_info["var_cov"] = V_new
    model_info = dict(getattr(result, "model_info", None) or {})
    if not model_info.get("model_type"):
        model_info["model_type"] = type(result).__name__
    model_info["standardized"] = {
        "variables": chosen,
        "scale_y": bool(scale_y),
        "scale": dict(zip(names, scale.tolist())),
        "sd": sds,
        "mean": means,
        "sd_y": sy,
        "mean_y": my,
        "weighted": w is not None,
        "intercept": intercept,
        "unstandardized_params": params.to_dict(),
        "note": (
            "Slope k is the coefficient of the regression on standardised "
            "variables: b_k * c_k with c_k = sd(x_k)/sd(y); its SE and "
            "interval ends are scaled by the same c_k and its t / p are "
            "unchanged."
        ),
    }
    alpha = getattr(result, "alpha", None)
    if alpha is not None:
        model_info.setdefault("alpha", alpha)
    if intercept_unavailable:
        model_info["standardized"]["intercept_se"] = (
            "unavailable: the fit stores no covariance matrix behind its "
            "standard errors, and the centred intercept needs the covariances"
        )
    with warnings.catch_warnings():
        # the generic non-finite-SE warning would blame a singular variance;
        # the specific reason is issued below
        if intercept_unavailable:
            warnings.simplefilter("ignore", category=ConvergenceWarning)
        out = EconometricResults(
            params=pd.Series(b_new, index=params.index),
            std_errors=pd.Series(se_new, index=params.index),
            model_info=model_info,
            data_info=data_info,
            diagnostics=dict(getattr(result, "diagnostics", None) or {}),
        )
    if intercept_unavailable:
        warnings.warn(
            f"standardize_coefs: no standard error for the centred intercept "
            f"{intercept!r}: the fit stores no covariance matrix behind its "
            "standard errors (e.g. Conley or bootstrap SEs set after the "
            "fit). The slopes are unaffected.",
            UserWarning,
            stacklevel=2,
        )
    # A slope's t / p is the fit's own (the scale cancels); copying keeps
    # them bit-identical rather than equal up to the division's rounding.
    t_old = getattr(result, "tvalues", None)
    p_old = getattr(result, "pvalues", None)
    lo_old = getattr(result, "conf_int_lower", None)
    hi_old = getattr(result, "conf_int_upper", None)
    if t_old is not None and p_old is not None:
        t_old = pd.Series(np.asarray(t_old, dtype=float).ravel(), index=params.index)
        p_old = pd.Series(np.asarray(p_old, dtype=float).ravel(), index=params.index)
        for c in names:
            if c != intercept:
                out.tvalues[c] = t_old[c]
                out.pvalues[c] = p_old[c]
    if lo_old is not None and hi_old is not None:
        lo_old = pd.Series(np.asarray(lo_old, dtype=float).ravel(), index=params.index)
        hi_old = pd.Series(np.asarray(hi_old, dtype=float).ravel(), index=params.index)
        for j, c in enumerate(names):
            if c != intercept:
                out.conf_int_lower[c] = scale[j] * lo_old[c]
                out.conf_int_upper[c] = scale[j] * hi_old[c]
    return out
