"""Predictions, comparisons, slopes and hypotheses in the marginaleffects style.

``sp.predictions`` / ``sp.comparisons`` / ``sp.slopes`` (and their ``avg_``
versions) and ``sp.hypotheses`` follow R ``marginaleffects``'s vocabulary and
conventions, so a user or agent arriving from R gets the numbers they expect:

* unit-level quantities by default, averaged by the ``avg_`` functions,
  optionally ``by=`` groups;
* averages are **unweighted** unless ``wts=`` asks for weights, and inference
  uses the **normal** distribution unless ``df=`` says otherwise --
  marginaleffects' defaults. ``sp.margins`` follows Stata instead (the fit's
  weights, t after ``regress``); the two agree when neither applies;
* a numeric variable whose values are all 0 / 1 is contrasted 1 - 0, other
  numeric variables by +1, and a factor's levels against its first level.

Everything is built on the machinery ``sp.margins`` uses (``_design.py``:
the design matrix rebuilt from the fitted formula, so interactions and
transformations are differentiated through), and standard errors are delta
method with the full coefficient covariance and an analytic Jacobian.
Each returned table carries the Jacobian of its estimates in
``.attrs['jacobian']``, which is how ``sp.hypotheses`` tests functions of
several rows (``"b1 - b2 = 0"``) with their joint covariance.

R ``marginaleffects`` (GPL-3) is used as a black-box reference only.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility
from ._covariance import inference_df, require_covariance
from ._design import design_for
from .margins import (
    _estimation_sample,
    _IndexScaleView,
    _link_derivatives,
    _mu_and_slope,
    _offset,
    _response_link,
)

__all__ = [
    "predictions",
    "avg_predictions",
    "comparisons",
    "avg_comparisons",
    "slopes",
    "avg_slopes",
    "hypotheses",
]

_COLUMNS = ["estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]


# ---------------------------------------------------------------------------
# Model context
# ---------------------------------------------------------------------------


def _fitted_frame(result: Any) -> Optional[pd.DataFrame]:
    """The DataFrame the model was fitted on, if the result still reaches it."""
    from ..core._call_record import get_call_record

    record = get_call_record(result)
    if record is None:
        return None
    for ref in record.frames().values():
        frame = ref.get()
        if frame is not None:
            return frame
    return None


class _Model:
    def __init__(self, result: Any, newdata: Optional[pd.DataFrame], fn: str) -> None:
        self.result = result
        self.fn = fn
        self.link = _response_link(result)
        params = getattr(result, "params", None)
        if not isinstance(params, pd.Series):
            raise MethodIncompatibility(
                f"{fn}: needs a regression-type result with a params Series.",
                recovery_hint="Use it after sp.regress / sp.logit / sp.probit / "
                "sp.poisson / sp.glm / sp.nbreg.",
            )
        self.beta = params.to_numpy(dtype=float)
        self.cov_source = result
        if (getattr(result, "model_info", None) or {}).get("irr"):
            self.beta = np.log(self.beta)
            self.cov_source = _IndexScaleView(result)
        if newdata is None:
            frame = _fitted_frame(result)
            if frame is None:
                raise MethodIncompatibility(
                    f"{fn}: the data the model was fitted on is no longer "
                    "available; pass newdata=.",
                    recovery_hint="Pass newdata=<the estimation data>.",
                )
            self.design = design_for(result, frame)
            frame, _ = _estimation_sample(result, frame, self.design, restrict=True)
        else:
            frame = newdata
            self.design = design_for(result, frame)
            needed = [v for v in self.design.variables if v in frame.columns]
            frame = frame.loc[frame[needed].notna().all(axis=1)] if needed else frame
        if len(frame) == 0:
            raise MethodIncompatibility(f"{fn}: no complete rows to evaluate.")
        self.frame = frame.reset_index(drop=True)
        self.V = require_covariance(self.cov_source, np.ones((1, self.beta.size)), fn)

    # -- predictions on the response or link scale -----------------------
    def predict(self, frame: pd.DataFrame, type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Unit predictions and their Jacobian wrt the coefficients (n x k)."""
        X = self.design.build(frame)
        eta = X @ self.beta + _offset(self.result, frame)
        if type == "link":
            return eta, X
        mu, dmu = _mu_and_slope(self.link, eta)
        return mu, dmu[:, None] * X

    def dydx(
        self, frame: pd.DataFrame, var: str, eps: float, type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unit derivative d prediction / d var and its Jacobian."""
        X = self.design.build(frame)
        D = self.design.derivative(frame, var, eps)
        eta = X @ self.beta + _offset(self.result, frame)
        deta = D @ self.beta
        if type == "link":
            return deta, D
        f1, f2 = _link_derivatives(self.link, eta)
        value = f1 * deta
        jac = (f2 * deta)[:, None] * X + f1[:, None] * D
        return value, jac


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _resolve_df(df: Any, result: Any) -> float:
    if df is None:
        return np.inf
    if isinstance(df, str):
        if df != "model":
            raise MethodIncompatibility("df must be None, 'model' or a number.")
        return inference_df(result)
    return float(df)


def _table(
    model: _Model,
    est: np.ndarray,
    jac: np.ndarray,
    meta: pd.DataFrame,
    *,
    df: float,
    conf_level: float,
    fn: str,
) -> pd.DataFrame:
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", jac, model.V, jac), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = est / se
    dist = stats.norm if not np.isfinite(df) else stats.t(df)
    p = 2 * dist.sf(np.abs(stat))
    q = dist.ppf(0.5 + conf_level / 2)
    out = meta.reset_index(drop=True).copy()
    out["estimate"] = est
    out["std_error"] = se
    out["statistic"] = stat
    out["p_value"] = p
    out["conf_low"] = est - q * se
    out["conf_high"] = est + q * se
    out.attrs.update(
        {
            "function": fn,
            "jacobian": jac,
            "vcov_coefficients": model.V,
            "df": df,
            "conf_level": conf_level,
            "prediction_scale": model.link,
        }
    )
    return out


def _weights(model: _Model, wts: Any) -> Optional[np.ndarray]:
    if wts is None or wts is False:
        return None
    if wts is True:
        from ._design import model_setting

        spec = model_setting(model.result, "weights")
        if not isinstance(spec, str) or spec not in model.frame.columns:
            raise MethodIncompatibility(
                f"{model.fn}: wts=True asks for the fit's weights, but the fit "
                "had no weight column in this data.",
                recovery_hint="Pass wts='<column>' instead.",
            )
        wts = spec
    if isinstance(wts, str):
        if wts not in model.frame.columns:
            raise MethodIncompatibility(f"{model.fn}: weight column {wts!r} not found.")
        return np.asarray(model.frame[wts], dtype=float)
    w = np.asarray(wts, dtype=float).ravel()
    if w.size != len(model.frame):
        raise MethodIncompatibility(f"{model.fn}: wts has the wrong length.")
    return w


def _average(
    est: np.ndarray,
    jac: np.ndarray,
    meta: pd.DataFrame,
    by: Optional[Sequence[str]],
    w: Optional[np.ndarray],
    keys: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Average unit quantities within ``keys`` (+ ``by``) groups."""
    groups = list(keys) + list(by or [])
    frame = meta[groups].copy() if groups else pd.DataFrame(index=meta.index)
    weights = np.ones(len(meta)) if w is None else w
    if not groups:
        ww = weights / weights.sum()
        return np.array([ww @ est]), (ww @ jac)[None, :], pd.DataFrame(index=[0])
    rows_e, rows_j, rows_m = [], [], []
    for key, idx in frame.groupby(groups, sort=True, dropna=False).groups.items():
        idx = np.asarray(list(idx))
        ww = weights[idx] / weights[idx].sum()
        rows_e.append(ww @ est[idx])
        rows_j.append(ww @ jac[idx])
        key = key if isinstance(key, tuple) else (key,)
        rows_m.append(dict(zip(groups, key)))
    return np.array(rows_e), np.vstack(rows_j), pd.DataFrame(rows_m)


def _check_by(model: _Model, by: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if by is None:
        return []
    by = [by] if isinstance(by, str) else list(by)
    missing = [b for b in by if b not in model.frame.columns]
    if missing:
        raise MethodIncompatibility(
            f"{model.fn}: by= column(s) {missing} not in the data."
        )
    return by


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------


def predictions(
    result: Any,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Unit-level predictions with delta-method standard errors.

    ``marginaleffects::predictions``: one row per row of ``newdata`` (default:
    the estimation sample), with the data's columns and ``estimate``,
    ``std_error``, ``statistic``, ``p_value``, ``conf_low``, ``conf_high``.
    With ``by=`` the predictions are averaged within groups (what
    :func:`avg_predictions` does).

    Parameters
    ----------
    result : fitted result
        From ``sp.regress``, ``sp.logit``, ``sp.probit``, ``sp.cloglog``,
        ``sp.poisson``, ``sp.nbreg`` or ``sp.glm`` (identity, logit, probit,
        cloglog or log link).
    newdata : DataFrame, optional
        Rows to predict for. Default: the estimation sample of the fitted data.
    by : str or list of str, optional
        Average within groups of these columns.
    type : {'response', 'link'}, default 'response'
        Prediction scale.
    wts : None, True, str or array, optional
        Weights for averages: None (unweighted, the marginaleffects default),
        True (the fit's weight column), a column name or an array.
    df : None, 'model' or float, optional
        Reference distribution: None is normal (marginaleffects), 'model'
        the fit's own (t after ``regress``), or a number.
    conf_level : float, default 0.95

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"x": rng.normal(size=300), "g": rng.integers(0, 2, 300)})
    >>> d["y"] = (rng.uniform(size=300) < 1 / (1 + np.exp(-d.x))).astype(int)
    >>> fit = sp.logit("y ~ x + g", data=d)
    >>> p = sp.predictions(fit)
    >>> bool(((p.estimate > 0) & (p.estimate < 1)).all())
    True
    """
    model = _Model(result, newdata, "predictions")
    by = _check_by(model, by)
    est, jac = model.predict(model.frame, type)
    dfv = _resolve_df(df, result)
    if by:
        e, j, meta = _average(est, jac, model.frame, by, _weights(model, wts), [])
        return _table(
            model, e, j, meta, df=dfv, conf_level=conf_level, fn="predictions"
        )
    return _table(
        model, est, jac, model.frame, df=dfv, conf_level=conf_level, fn="predictions"
    )


def avg_predictions(
    result: Any,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Average prediction, overall or ``by=`` group (``marginaleffects::avg_predictions``).

    Parameters are those of :func:`predictions`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> d = pd.DataFrame({"x": rng.normal(size=200), "g": rng.integers(0, 3, 200)})
    >>> d["y"] = 1 + d.x + d.g + rng.normal(size=200)
    >>> fit = sp.regress("y ~ x + C(g)", data=d)
    >>> sp.avg_predictions(fit, by="g").shape[0]
    3
    """
    model = _Model(result, newdata, "avg_predictions")
    by = _check_by(model, by)
    est, jac = model.predict(model.frame, type)
    e, j, meta = _average(est, jac, model.frame, by, _weights(model, wts), [])
    return _table(
        model,
        e,
        j,
        meta,
        df=_resolve_df(df, result),
        conf_level=conf_level,
        fn="avg_predictions",
    )


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------

_COMPARISONS = (
    "difference",
    "ratio",
    "lnratio",
    "ratioavg",
    "lnratioavg",
    "differenceavg",
)


def _variable_kind(model: _Model, var: str) -> str:
    if model.design.is_factor(var):
        return "factor"
    col = model.frame[var]
    if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return "factor"
    values = set(pd.unique(col.dropna()))
    return "binary" if values <= {0, 1} and len(values) == 2 else "numeric"


def _counterfactuals(
    model: _Model, var: str, spec: Any
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """``(label, lo_frame, hi_frame)`` for one variable."""
    fr = model.frame
    kind = _variable_kind(model, var)
    out = []
    if kind == "factor":
        levels = (
            list(model.design.factors.get(var, (None,))[0])
            if getattr(model.design, "factors", None) and var in model.design.factors
            else sorted(pd.unique(fr[var].dropna()), key=lambda v: (str(type(v)), v))
        )
        ref = levels[0]
        for lev in levels[1:]:
            lo, hi = fr.copy(), fr.copy()
            lo[var], hi[var] = ref, lev
            out.append((f"{lev} - {ref}", lo, hi))
        return out
    if kind == "binary" and spec is None:
        lo, hi = fr.copy(), fr.copy()
        lo[var], hi[var] = 0, 1
        return [("1 - 0", lo, hi)]
    x = fr[var].astype(float)
    lo, hi = fr.copy(), fr.copy()
    if spec is None or (isinstance(spec, (int, float)) and not isinstance(spec, bool)):
        step = 1.0 if spec is None else float(spec)
        hi[var] = x + step
        return [(f"+{step:g}", lo, hi)]
    if isinstance(spec, str) and spec in ("sd", "2sd"):
        # Centred on the sample mean for every row (marginaleffects' rule,
        # whatever its "x + sd/2" label suggests).
        half = float(x.std(ddof=1)) * (0.5 if spec == "sd" else 1.0)
        lo[var], hi[var] = float(x.mean()) - half, float(x.mean()) + half
        tag = "sd/2" if spec == "sd" else "sd"
        return [(f"(x + {tag}) - (x - {tag})", lo, hi)]
    if isinstance(spec, str) and spec in ("iqr", "minmax"):
        a, b = np.quantile(x, [0.25, 0.75]) if spec == "iqr" else (x.min(), x.max())
        lo[var], hi[var] = float(a), float(b)
        return [("Q3 - Q1" if spec == "iqr" else "Max - Min", lo, hi)]
    if (
        isinstance(spec, (list, tuple))
        and len(spec) == 2
        and all(isinstance(v, (int, float)) for v in spec)
    ):
        lo[var], hi[var] = float(spec[0]), float(spec[1])
        return [(f"{spec[1]:g} - {spec[0]:g}", lo, hi)]
    raise MethodIncompatibility(
        f"comparisons: the change for {var!r} must be a number, a pair "
        f"(low, high), or one of 'sd', '2sd', 'iqr', 'minmax'; got {spec!r}.",
        recovery_hint="E.g. variables={'x': 5}, {'x': 'sd'} or {'x': (0, 10)}.",
    )


def _label(label: str, comparison: str) -> str:
    """marginaleffects' contrast label for ``comparison``."""
    if comparison in ("difference", "differenceavg"):
        return label
    pair = re.fullmatch(r"(.+) - (.+)", label) if not label.startswith("(") else None
    if pair is None:  # "+1", centred and fixed-value numeric changes
        return f"mean({label})" if comparison.endswith("avg") else label
    hi, lo = pair.groups()
    return {
        "ratio": f"{hi} / {lo}",
        "lnratio": f"ln({hi} / {lo})",
        "ratioavg": f"mean({hi}) / mean({lo})",
        "lnratioavg": f"ln(mean({hi}) / mean({lo}))",
    }[comparison]


def _contrast(
    model: _Model, lo: pd.DataFrame, hi: pd.DataFrame, comparison: str, type: str
) -> Tuple[np.ndarray, np.ndarray]:
    m0, j0 = model.predict(lo, type)
    m1, j1 = model.predict(hi, type)
    if comparison in ("difference", "differenceavg"):
        return m1 - m0, j1 - j0
    if comparison == "ratio":
        return m1 / m0, (j1 * m0[:, None] - j0 * m1[:, None]) / (m0**2)[:, None]
    if comparison == "lnratio":
        return np.log(m1 / m0), j1 / m1[:, None] - j0 / m0[:, None]
    raise AssertionError(comparison)  # pragma: no cover - handled by caller


def _comparisons(
    result: Any,
    variables: Any,
    newdata: Optional[pd.DataFrame],
    by: Any,
    comparison: str,
    type: str,
    wts: Any,
    df: Any,
    conf_level: float,
    average: bool,
    fn: str,
) -> pd.DataFrame:
    if comparison not in _COMPARISONS:
        raise MethodIncompatibility(
            f"{fn}: comparison must be one of {list(_COMPARISONS)}, got {comparison!r}."
        )
    model = _Model(result, newdata, fn)
    by = _check_by(model, by)
    specs = _variable_specs(model, variables, fn)
    w = _weights(model, wts)
    if average or by:
        # Averaged ratios are ratios of average predictions, as in
        # marginaleffects (the mean of unit ratios is rarely the target).
        comparison = {"ratio": "ratioavg", "lnratio": "lnratioavg"}.get(
            comparison, comparison
        )
    pieces: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]] = []
    for var, spec in specs.items():
        for label, lo, hi in _counterfactuals(model, var, spec):
            meta = model.frame.copy()
            meta.insert(0, "contrast", _label(label, comparison))
            meta.insert(0, "term", var)
            if comparison in ("ratioavg", "lnratioavg"):
                m0, j0 = model.predict(lo, type)
                m1, j1 = model.predict(hi, type)
                a0, ja0, gm = _average(m0, j0, meta, by, w, ["term", "contrast"])
                a1, ja1, _ = _average(m1, j1, meta, by, w, ["term", "contrast"])
                if comparison == "ratioavg":
                    e = a1 / a0
                    j = (ja1 * a0[:, None] - ja0 * a1[:, None]) / (a0**2)[:, None]
                else:
                    e = np.log(a1 / a0)
                    j = ja1 / a1[:, None] - ja0 / a0[:, None]
                pieces.append((e, j, gm))
                continue
            est, jac = _contrast(model, lo, hi, comparison, type)
            if average or by:
                pieces.append(_average(est, jac, meta, by, w, ["term", "contrast"]))
            else:
                pieces.append((est, jac, meta))
    est = np.concatenate([p[0] for p in pieces])
    jac = np.vstack([p[1] for p in pieces])
    meta = pd.concat([p[2] for p in pieces], ignore_index=True)
    return _table(
        model, est, jac, meta, df=_resolve_df(df, result), conf_level=conf_level, fn=fn
    )


def _variable_specs(model: _Model, variables: Any, fn: str) -> Dict[str, Any]:
    if variables is None:
        names = list(model.design.variables)
        return {v: None for v in names if v in model.frame.columns}
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(variables, dict):
        specs = dict(variables)
    else:
        specs = {v: None for v in variables}
    unknown = [v for v in specs if v not in model.design.variables]
    if unknown:
        raise MethodIncompatibility(
            f"{fn}: {unknown} are not variables of the model.",
            recovery_hint=f"Model variables: {list(model.design.variables)}.",
        )
    return specs


def comparisons(
    result: Any,
    variables: Any = None,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    comparison: str = "difference",
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Unit-level contrasts of predictions (``marginaleffects::comparisons``).

    For each variable, predictions at a "high" and a "low" value of it, every
    other column held at the row's own value, contrasted per row:

    * a factor (``C(g)``, strings, categories): each level against the first;
    * a numeric 0/1 variable: 1 against 0;
    * another numeric variable: an increase of 1 from the row's value, or as
      given in ``variables=``: a number (that increase), ``"sd"`` /
      ``"2sd"`` (mean +/- sd/2, mean +/- sd), ``"iqr"`` (third against
      first quartile), ``"minmax"`` (maximum against minimum) or a pair
      ``(low, high)`` of fixed values.

    Parameters
    ----------
    result : fitted result
        As in :func:`predictions`.
    variables : str, list or dict, optional
        Variables to contrast (default: every model variable); a dict maps a
        numeric variable to its change.
    newdata : DataFrame, optional
        Rows to evaluate (default: the estimation sample).
    by : str or list of str, optional
        Average within groups.
    comparison : str, default 'difference'
        'difference', 'ratio', 'lnratio' (per row); 'ratioavg',
        'lnratioavg' (ratio of the average predictions). When the contrasts
        are averaged (``avg_comparisons`` or ``by=``), 'ratio' and 'lnratio'
        mean 'ratioavg' and 'lnratioavg', as in marginaleffects.
    type, wts, df, conf_level
        As in :func:`predictions`.

    Returns
    -------
    DataFrame
        ``term``, ``contrast``, the data's columns (unit level) or the
        ``by`` columns, and the estimate columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> d = pd.DataFrame({"x": rng.normal(size=300), "d": rng.integers(0, 2, 300)})
    >>> d["y"] = 1 + 2 * d.d + d.x + rng.normal(size=300)
    >>> fit = sp.regress("y ~ d + x", data=d)
    >>> sp.comparisons(fit, "d").shape[0] == len(d)
    True
    """
    return _comparisons(
        result,
        variables,
        newdata,
        by,
        comparison,
        type,
        wts,
        df,
        conf_level,
        average=False,
        fn="comparisons",
    )


def avg_comparisons(
    result: Any,
    variables: Any = None,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    comparison: str = "difference",
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Average contrasts (``marginaleffects::avg_comparisons``).

    The average over rows of the unit-level contrasts of :func:`comparisons`
    (``comparison='ratioavg'`` gives the ratio of average predictions
    instead). For a binary treatment on the response scale this is the
    average treatment effect implied by the model (the g-formula).

    Parameters are those of :func:`comparisons`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> d = pd.DataFrame({"x": rng.normal(size=400), "d": rng.integers(0, 2, 400)})
    >>> d["y"] = (rng.uniform(size=400) < 1 / (1 + np.exp(-(d.x + d.d)))).astype(int)
    >>> fit = sp.logit("y ~ d + x", data=d)
    >>> ate = sp.avg_comparisons(fit, "d")
    >>> bool(0 < ate.estimate.iloc[0] < 1)
    True
    """
    return _comparisons(
        result,
        variables,
        newdata,
        by,
        comparison,
        type,
        wts,
        df,
        conf_level,
        average=True,
        fn="avg_comparisons",
    )


# ---------------------------------------------------------------------------
# slopes
# ---------------------------------------------------------------------------

_SLOPES = ("dydx", "eyex", "eydx", "dyex")


def _slopes(
    result: Any,
    variables: Any,
    newdata: Optional[pd.DataFrame],
    by: Any,
    slope: str,
    type: str,
    wts: Any,
    df: Any,
    conf_level: float,
    eps: float,
    average: bool,
    fn: str,
) -> pd.DataFrame:
    if slope not in _SLOPES:
        raise MethodIncompatibility(f"{fn}: slope must be one of {list(_SLOPES)}.")
    model = _Model(result, newdata, fn)
    by = _check_by(model, by)
    specs = _variable_specs(model, variables, fn)
    w = _weights(model, wts)
    pieces = []
    for var in specs:
        if _variable_kind(model, var) in ("factor", "binary"):
            # Factors and 0/1 variables are reported as contrasts, as
            # marginaleffects does.
            for label, lo, hi in _counterfactuals(model, var, None):
                est, jac = _contrast(model, lo, hi, "difference", type)
                meta = model.frame.copy()
                meta.insert(0, "contrast", label)
                meta.insert(0, "term", var)
                pieces.append(
                    _average(est, jac, meta, by, w, ["term", "contrast"])
                    if (average or by)
                    else (est, jac, meta)
                )
            continue
        s, js = model.dydx(model.frame, var, eps, type)
        if slope != "dydx":
            mu, jmu = model.predict(model.frame, type)
            x = model.frame[var].to_numpy(dtype=float)
            if slope == "eyex":
                s, js = (
                    s * x / mu,
                    (js * x[:, None]) / mu[:, None] - (s * x / mu**2)[:, None] * jmu,
                )
            elif slope == "eydx":
                s, js = s / mu, js / mu[:, None] - (s / mu**2)[:, None] * jmu
            else:  # dyex
                s, js = s * x, js * x[:, None]
        meta = model.frame.copy()
        meta.insert(
            0,
            "contrast",
            {"dydx": "dY/dX", "eyex": "eY/eX", "eydx": "eY/dX", "dyex": "dY/eX"}[slope],
        )
        meta.insert(0, "term", var)
        pieces.append(
            _average(s, js, meta, by, w, ["term", "contrast"])
            if (average or by)
            else (s, js, meta)
        )
    est = np.concatenate([p[0] for p in pieces])
    jac = np.vstack([p[1] for p in pieces])
    meta = pd.concat([p[2] for p in pieces], ignore_index=True)
    return _table(
        model, est, jac, meta, df=_resolve_df(df, result), conf_level=conf_level, fn=fn
    )


def slopes(
    result: Any,
    variables: Any = None,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    slope: str = "dydx",
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
    eps: float = 1e-5,
) -> pd.DataFrame:
    """Unit-level partial derivatives (``marginaleffects::slopes``).

    ``dY/dX`` of the prediction with respect to each numeric variable,
    differentiating through interactions and formula transformations;
    ``slope='eyex'`` / ``'eydx'`` / ``'dyex'`` give elasticities and
    semi-elasticities. Factor and 0/1 variables are reported as contrasts,
    as marginaleffects does.

    Parameters
    ----------
    result, variables, newdata, by, type, wts, df, conf_level
        As in :func:`comparisons`.
    slope : {'dydx', 'eyex', 'eydx', 'dyex'}, default 'dydx'
    eps : float, default 1e-5
        Relative step of the central difference used on transformed terms.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(4)
    >>> d = pd.DataFrame({"x": rng.normal(size=300)})
    >>> d["y"] = 1 + 2 * d.x + 0.5 * d.x**2 + rng.normal(size=300)
    >>> fit = sp.regress("y ~ x + I(x**2)", data=d)
    >>> s = sp.slopes(fit, "x")
    >>> bool(np.allclose(s.estimate, fit.params.iloc[1] + 2 * fit.params.iloc[2] * d.x))
    True
    """
    return _slopes(
        result,
        variables,
        newdata,
        by,
        slope,
        type,
        wts,
        df,
        conf_level,
        eps,
        average=False,
        fn="slopes",
    )


def avg_slopes(
    result: Any,
    variables: Any = None,
    newdata: Optional[pd.DataFrame] = None,
    *,
    by: Optional[Union[str, Sequence[str]]] = None,
    slope: str = "dydx",
    type: str = "response",
    wts: Any = None,
    df: Any = None,
    conf_level: float = 0.95,
    eps: float = 1e-5,
) -> pd.DataFrame:
    """Average partial derivatives (``marginaleffects::avg_slopes``).

    The average marginal effect; with ``wts=True`` and ``df='model'`` it is
    Stata's ``margins, dydx()`` after a weighted fit.

    Parameters are those of :func:`slopes`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(5)
    >>> d = pd.DataFrame({"x": rng.normal(size=300)})
    >>> d["y"] = (rng.uniform(size=300) < 1 / (1 + np.exp(-d.x))).astype(int)
    >>> fit = sp.logit("y ~ x", data=d)
    >>> ame = sp.avg_slopes(fit, "x")
    >>> bool(0 < ame.estimate.iloc[0] < fit.params["x"])
    True
    """
    return _slopes(
        result,
        variables,
        newdata,
        by,
        slope,
        type,
        wts,
        df,
        conf_level,
        eps,
        average=True,
        fn="avg_slopes",
    )


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------

_B_INDEX = re.compile(r"(?<![A-Za-z0-9_])b(\d+)(?![A-Za-z0-9_])")


def hypotheses(
    result: Any,
    hypothesis: str,
    *,
    df: Any = None,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Delta-method test of a function of estimates (``marginaleffects::hypotheses``).

    ``result`` is a fitted model (the estimates are its coefficients) or a
    table returned by :func:`predictions`, :func:`comparisons` or
    :func:`slopes` (the estimates are its rows, with their joint covariance
    from the table's Jacobian). ``b1``, ``b2`` ... refer to the estimates by
    position; a model's coefficients can also be named. ``"lhs = rhs"`` tests
    ``lhs - rhs = 0``; an expression without ``=`` is tested against 0.

    Parameters
    ----------
    result : fitted result or marginaleffects-style table
    hypothesis : str
        E.g. ``"b1 - b2 = 0"``, ``"b2 / b3 = 1"``, ``"x1 = 2 * x2"``. Only
        arithmetic and the functions ``sp.nlcom`` accepts.
    df : None, 'model' or float, optional
        Reference distribution (None: normal).
    conf_level : float, default 0.95

    Returns
    -------
    DataFrame
        One row: ``term`` (the hypothesis), ``estimate`` (lhs - rhs) and the
        inference columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(6)
    >>> d = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300)})
    >>> d["y"] = 1 + d.x1 + d.x2 + rng.normal(size=300)
    >>> fit = sp.regress("y ~ x1 + x2", data=d)
    >>> h = sp.hypotheses(fit, "b2 = b3")
    >>> bool(h.p_value.iloc[0] > 0.01)
    True
    """
    from .nlcom import _evaluate

    if isinstance(result, pd.DataFrame):
        jac = result.attrs.get("jacobian")
        V = result.attrs.get("vcov_coefficients")
        if jac is None or V is None:
            raise MethodIncompatibility(
                "hypotheses: this table does not carry its Jacobian; pass a "
                "table from sp.predictions / comparisons / slopes or a fit.",
            )
        beta = result["estimate"].to_numpy(dtype=float)
        cov = jac @ V @ jac.T
        names: List[str] = []
        dfv = _resolve_df(df, None) if df != "model" else result.attrs.get("df", np.inf)
    else:
        params = result.params
        beta = params.to_numpy(dtype=float)
        cov = require_covariance(result, np.ones((1, beta.size)), "hypotheses")
        names = [str(n) for n in params.index]
        dfv = _resolve_df(df, result)
    text = hypothesis.strip()
    if text.count("=") > 1 or "==" in text:
        raise MethodIncompatibility("hypotheses: use a single '='.")
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    lookup: Dict[str, int] = {}
    for i in range(beta.size):
        lookup[f"b{i + 1}"] = i
    for i, n in enumerate(names):
        if n.isidentifier():
            lookup.setdefault(n, i)
    for m in _B_INDEX.finditer(text):
        if int(m.group(1)) > beta.size or int(m.group(1)) < 1:
            raise MethodIncompatibility(
                f"hypotheses: b{m.group(1)} is out of range (1..{beta.size})."
            )
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise MethodIncompatibility(
            f"hypotheses: cannot parse {hypothesis!r}."
        ) from exc
    value, grad = _evaluate(tree, beta, lookup, hypothesis)
    se = float(np.sqrt(max(float(grad @ cov @ grad), 0.0)))
    dist = stats.norm if not np.isfinite(dfv) else stats.t(dfv)
    stat = value / se if se > 0 else np.nan
    q = dist.ppf(0.5 + conf_level / 2)
    return pd.DataFrame(
        [
            {
                "term": hypothesis,
                "estimate": value,
                "std_error": se,
                "statistic": stat,
                "p_value": float(2 * dist.sf(abs(stat))) if se > 0 else np.nan,
                "conf_low": value - q * se,
                "conf_high": value + q * se,
            }
        ]
    )
