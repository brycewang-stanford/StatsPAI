"""Autoregressive distributed lag regressions for forecasting.

``y[t]`` is regressed by OLS on its own lags and on lags of other series:

    y[t] = b0 + b1 y[t-1] + ... + bp y[t-p]
              + d1 x[t-1] + ... + dq x[t-q] + u[t]

With no ``x`` this is an AR(p). The regressors are dated ``t - 1`` or
earlier, so the fitted equation gives a forecast of the next period from
data in hand; ``contemporaneous=True`` adds ``x[t]`` for a model that is to
be read as a distributed-lag relation and not used to forecast.

Around the fit sit the tools a forecasting regression is judged by: lag
selection by BIC or AIC on a common sample, a Granger-causality F test, the
one-step forecast with its interval, and pseudo out-of-sample forecasting,
where the model is re-estimated at each date using only data up to it and
the root mean squared forecast error is measured on forecasts the model did
not see.

Estimation is :func:`statspai.regress`, so the standard errors (``vce="hc1"``
by default, ``"hac"`` when the errors are serially correlated) and every
post-estimation tool work on ``result.model``.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["ardl", "ARDLResult"]


def _lag_name(var: str, k: int) -> str:
    return var if k == 0 else f"{var}_L{k}"


def _design(
    frame: pd.DataFrame,
    y: str,
    x: Sequence[str],
    p: int,
    q: Dict[str, int],
    contemporaneous: bool,
    trend: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Lagged regressors as columns, in the order they enter the model."""
    out = pd.DataFrame({y: frame[y].to_numpy(dtype=float)}, index=frame.index)
    names: List[str] = []
    for k in range(1, p + 1):
        out[_lag_name(y, k)] = out[y].shift(k)
        names.append(_lag_name(y, k))
    for var in x:
        col = frame[var].astype(float)
        out[var] = col  # the level is kept for forecasting from the last rows
        for k in range(0 if contemporaneous else 1, q[var] + 1):
            name = _lag_name(var, k)
            if k:
                out[name] = col.shift(k)
            names.append(name)
    if trend == "ct":
        out["_trend"] = np.arange(1.0, len(out) + 1.0)
        names.append("_trend")
    return out, names


class ARDLResult(ResultProtocolMixin):
    """A fitted AR(p) or ADL(p, q) forecasting regression.

    Attributes
    ----------
    model : EconometricResults
        The OLS fit. Pass it to ``sp.test``, ``sp.lincom`` and so on; lagged
        regressors are named ``<var>_L<k>``.
    params, std_errors : pandas.Series
        Coefficients and standard errors of ``model``.
    lags : int
        Autoregressive order ``p``.
    x_lags : dict
        Lag order of each additional regressor.
    nobs : int
        Observations in the estimation sample.
    ser : float
        Standard error of the regression, ``sqrt(SSR / (T - K))``.
    bic, aic : float
        ``ln(SSR / T) + K ln(T) / T`` and ``ln(SSR / T) + 2 K / T``, with
        ``K`` the number of coefficients including the intercept.
    ic_table : pandas.DataFrame or None
        The criteria for every candidate order, when ``lags`` was selected.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(200)
    >>> for t in range(1, 200):
    ...     y[t] = 1.0 + 0.5 * y[t - 1] + rng.normal()
    >>> res = sp.ardl(pd.DataFrame({"y": y}), "y", lags=1)
    >>> type(res).__name__
    'ARDLResult'
    >>> list(res.params.index)
    ['Intercept', 'y_L1']
    >>> res.forecast().shape
    (1, 4)
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model: Any,
        y: str,
        x: List[str],
        lags: int,
        x_lags: Dict[str, int],
        contemporaneous: bool,
        trend: str,
        regressors: List[str],
        design: pd.DataFrame,
        sample_index: pd.Index,
        fit_kwargs: Dict[str, Any],
        ic_table: Optional[pd.DataFrame],
        alpha: float,
    ) -> None:
        self.model = model
        self.y = y
        self.x = x
        self.lags = lags
        self.x_lags = x_lags
        self.contemporaneous = contemporaneous
        self.trend = trend
        self.regressors = regressors
        self.ic_table = ic_table
        self.alpha = alpha
        self._design = design
        self._sample_index = sample_index
        self._fit_kwargs = fit_kwargs

        self.params = model.params
        self.std_errors = model.std_errors
        self.nobs = int(len(sample_index))
        n_coef = len(self.params)
        resid = np.asarray(_residuals(model), dtype=float)
        self.rss = float(resid @ resid)
        self.ser = float(np.sqrt(self.rss / max(self.nobs - n_coef, 1)))
        self.r2 = float(model.r2)
        self.r2_adj = float(model.r2_adj)
        base = np.log(self.rss / self.nobs)
        self.bic = float(base + n_coef * np.log(self.nobs) / self.nobs)
        self.aic = float(base + n_coef * 2.0 / self.nobs)
        if x:
            order = ", ".join(str(x_lags[v]) for v in x)
            self.method = f"ADL({lags}, {order})"
        else:
            self.method = f"AR({lags})"
        self.estimand = "one-step-ahead forecasting regression"

    # ------------------------------------------------------------ display
    def summary(self) -> str:
        table = pd.DataFrame({"coef": self.params, "std err": self.std_errors})
        table["t"] = table["coef"] / table["std err"]
        head = f"{self.method} for {self.y}"
        lines = [
            head,
            "=" * len(head),
            table.to_string(float_format=lambda v: f"{v:.6g}"),
            "",
            f"Observations: {self.nobs}    SER: {self.ser:.6g}    "
            f"adj. R2: {self.r2_adj:.4f}",
            f"BIC: {self.bic:.4f}    AIC: {self.aic:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ARDLResult({self.method}, y={self.y!r}, nobs={self.nobs})"

    # ----------------------------------------------------------- forecast
    def _row_for(self, position: int) -> np.ndarray:
        """Regressor values for the period after ``position`` (0-based row)."""
        frame = self._design
        values: List[float] = []
        for name in self.regressors:
            if name == "_trend":
                values.append(float(position + 2))
                continue
            var, k = _split_lag(name, [self.y] + self.x)
            source = position + 1 - k
            values.append(float(frame[var].iloc[source]))
        return np.asarray(values)

    def forecast(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Forecast of the period after the last estimation observation.

        The interval is ``forecast +/- z * RMSFE`` with the RMSFE estimated
        by the standard error of the regression. It ignores the estimation
        error in the coefficients and assumes normal errors; a pseudo
        out-of-sample RMSFE from :meth:`poos` is the more honest width.

        Returns
        -------
        pandas.DataFrame
            One row with ``forecast``, ``rmsfe``, ``lower``, ``upper``.
        """
        if self.contemporaneous:
            raise MethodIncompatibility(
                "ardl.forecast: the model uses x at date t, which is not "
                "known when the forecast is made.",
                recovery_hint="Refit with contemporaneous=False.",
            )
        level = self.alpha if alpha is None else alpha
        origin = int(self._design.index.get_loc(self._sample_index[-1]))
        point = self._predict(origin, self.params)
        z = float(stats.norm.ppf(1 - level / 2))
        return pd.DataFrame(
            {
                "forecast": [point],
                "rmsfe": [self.ser],
                "lower": [point - z * self.ser],
                "upper": [point + z * self.ser],
            },
            index=pd.Index(["T+1"], name="period"),
        )

    def _predict(self, position: int, params: pd.Series) -> float:
        row = self._row_for(position)
        if np.isnan(row).any():
            raise DataInsufficient(
                "ardl: a lag needed for the forecast is missing.",
                recovery_hint="The series must be observed through the "
                "forecast origin.",
            )
        const = float(params["Intercept"]) if "Intercept" in params.index else 0.0
        return const + float(row @ params[self.regressors].to_numpy())

    # ------------------------------------------------------------ granger
    def granger(self, x: Optional[str] = None) -> Dict[str, Any]:
        """F test that every lag of ``x`` has a zero coefficient.

        The Granger-causality statistic: whether lags of ``x`` help predict
        ``y`` given the lags of ``y``. It uses the model's own covariance
        estimator, so it is heteroskedasticity- or autocorrelation-robust
        when the fit is. "Causality" here means predictive content.
        """
        from ..postestimation.hypothesis import test as _test

        if not self.x:
            raise MethodIncompatibility(
                "ardl.granger: the model has no additional regressor.",
                recovery_hint="Fit with x=[...].",
            )
        var = self.x[0] if x is None else x
        if var not in self.x:
            raise MethodIncompatibility(
                f"ardl.granger: {var!r} is not a regressor of this model.",
                recovery_hint=f"Choose one of {self.x}.",
            )
        terms = [n for n in self.regressors if _split_lag(n, self.x)[0] == var]
        return _test(self.model, " ".join(terms))

    # --------------------------------------------------------------- poos
    def poos(self, start: Any, window: str = "expanding") -> pd.DataFrame:
        """Pseudo out-of-sample one-step forecasts.

        For each date from ``start`` to the end of the data the model is
        re-estimated on observations before that date and used to forecast
        it, as a forecaster standing at the previous date would have done.

        Parameters
        ----------
        start : label or int
            First date to forecast: a label of the data's index (or of the
            ``time`` column), or a row position.
        window : {"expanding", "rolling"}, default "expanding"
            ``"expanding"`` re-estimates on every observation from the start
            of the estimation sample; ``"rolling"`` keeps the window length
            fixed at its initial size.

        Returns
        -------
        pandas.DataFrame
            ``actual``, ``forecast`` and ``error`` (actual minus forecast)
            per forecast date. ``attrs`` holds ``rmsfe`` (root mean squared
            forecast error), ``bias`` (mean error), ``bias_se`` and
            ``n_forecasts``. A bias several standard errors from zero says
            the forecasts were systematically off, which a break in the
            relationship would produce.
        """
        from ..regression.ols import regress

        if self.contemporaneous:
            raise MethodIncompatibility(
                "ardl.poos: the model uses x at date t.",
                recovery_hint="Refit with contemporaneous=False.",
            )
        if window not in ("expanding", "rolling"):
            raise MethodIncompatibility(
                f"ardl.poos: unknown window {window!r}.",
                recovery_hint="Use 'expanding' or 'rolling'.",
            )
        frame = self._design
        first = _position(frame.index, start)
        est_start = int(frame.index.get_loc(self._sample_index[0]))
        if first - est_start < len(self.params) + 2:
            raise DataInsufficient(
                "ardl.poos: too few observations before the first forecast "
                "date to estimate the model.",
                recovery_hint="Choose a later start.",
                diagnostics={"n_before": int(first - est_start)},
            )
        formula = f"{self.y} ~ " + (" + ".join(self.regressors) or "1")
        if self.trend == "n":
            formula += " - 1"
        initial = first - est_start
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for target in range(first, len(frame)):
                lo = est_start if window == "expanding" else target - initial
                train = frame.iloc[lo:target].dropna(subset=[self.y] + self.regressors)
                fit = regress(formula, data=train)
                actual = float(frame[self.y].iloc[target])
                if np.isnan(actual):
                    continue
                point = self._predict(target - 1, fit.params)
                rows.append((frame.index[target], actual, point, actual - point))
        out = pd.DataFrame(rows, columns=["date", "actual", "forecast", "error"])
        out = out.set_index("date")
        err = out["error"].to_numpy()
        out.attrs["rmsfe"] = float(np.sqrt(np.mean(err**2)))
        out.attrs["bias"] = float(err.mean())
        out.attrs["bias_se"] = (
            float(err.std(ddof=1) / np.sqrt(err.size)) if err.size > 1 else np.nan
        )
        out.attrs["n_forecasts"] = int(err.size)
        return out


def _residuals(model: Any) -> Any:
    resid = model.residuals
    return resid() if callable(resid) else resid


def _split_lag(name: str, variables: Sequence[str]) -> Tuple[str, int]:
    """``'gdp_L2'`` -> ``('gdp', 2)``; a bare variable name is lag 0."""
    for var in sorted(variables, key=len, reverse=True):
        if name == var:
            return var, 0
        prefix = var + "_L"
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            return var, int(name[len(prefix) :])
    return name, 0


def _position(index: pd.Index, label: Any) -> int:
    """Row position of a date label; an integer not in the index is a position."""
    if label in index:
        loc = index.get_loc(label)
        if not isinstance(loc, (int, np.integer)):
            raise MethodIncompatibility(
                f"ardl: date {label!r} appears more than once.",
                recovery_hint="The time index must be unique.",
            )
        return int(loc)
    if isinstance(label, (int, np.integer)) and 0 <= int(label) < len(index):
        return int(label)
    raise MethodIncompatibility(
        f"ardl: {label!r} is not a date of the data.",
        recovery_hint="Pass a label of the index (or of the time column), or "
        "an integer row position.",
    )


def ardl(
    data: pd.DataFrame,
    y: str,
    x: Optional[Union[str, Sequence[str]]] = None,
    *,
    lags: Union[int, str] = 1,
    x_lags: Union[int, str, Dict[str, int]] = 1,
    max_lags: Optional[int] = None,
    contemporaneous: bool = False,
    trend: str = "c",
    time: Optional[str] = None,
    sample: Optional[Tuple[Any, Any]] = None,
    vce: str = "hc1",
    hac_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> ARDLResult:
    """Autoregression or autoregressive distributed lag model, by OLS.

    Equivalent to R's ``dynlm`` / ``ARDL``, ``statsmodels.tsa.ardl.ARDL``
    and a Stata ``regress`` on ``L(1/p).y L(1/q).x``.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per period, in time order (or pass ``time=``), with no gaps.
    y : str
        The series to forecast.
    x : str or list of str, optional
        Other predictors. Omit for an AR(p).
    lags : int or {"bic", "aic"}, default 1
        Autoregressive order ``p``. A criterion searches ``0 .. max_lags``
        with every candidate estimated on the same observations, so the
        criteria are comparable. BIC is consistent for the true order; AIC
        tends to pick more lags.
    x_lags : int, dict or "same", default 1
        Lags of each ``x``: one order for all, a ``{name: order}`` dict, or
        ``"same"`` to tie it to ``p`` (the ADL(p, p) family a criterion then
        searches over).
    max_lags : int, optional
        Largest order searched. Default ``floor(12 * (T / 100) ** 0.25)``
        capped at 8.
    contemporaneous : bool, default False
        Include ``x[t]``. The model can then not forecast.
    trend : {"c", "ct", "n"}, default "c"
        Constant, constant and linear trend, or neither.
    time : str, optional
        Column to sort by; it becomes the index, so ``sample`` and ``poos``
        take its values.
    sample : (start, end), optional
        Labels (or row positions) of the first and last observation of the
        estimation sample. Lags are still taken from earlier rows, which is
        how a fixed estimation window is held while the lag order varies.
    vce : str, default "hc1"
        Covariance estimator, as :func:`statspai.regress`: ``"hc1"``,
        ``"hc0"``, ``"nonrobust"``, or ``"hac"`` (with ``hac_lags``) when
        the errors are serially correlated, as in a multi-period-ahead or
        distributed-lag regression.
    hac_lags : int, optional
        Newey-West lag length for ``vce="hac"``.
    alpha : float, default 0.05
        Level of the forecast interval.

    Returns
    -------
    ARDLResult

    Raises
    ------
    MethodIncompatibility
        A missing value inside the series, an unknown option, a lagged
        column name that already exists.
    DataInsufficient
        Too few observations for the requested lags.

    Notes
    -----
    The coefficients are predictive, not causal: a lag of ``x`` that helps
    forecast ``y`` may do so because both respond to something else. The
    regression also assumes the relationship is stable; check with
    :func:`statspai.structural_break` and with ``result.poos``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 240
    >>> x = rng.normal(size=n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.5 + 0.4 * y[t - 1] + 0.8 * x[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.ardl(df, "y", "x", lags=1, x_lags=1)
    >>> res.method
    'ADL(1, 1)'
    >>> bool(res.granger()["pvalue"] < 0.01)
    True
    >>> oos = res.poos(start=180)
    >>> oos.attrs["n_forecasts"]
    60
    """
    from ..regression.ols import regress

    if trend not in ("c", "ct", "n"):
        raise MethodIncompatibility(
            f"ardl: unknown trend {trend!r}.",
            recovery_hint="Use 'c', 'ct' or 'n'.",
        )
    xs = [x] if isinstance(x, str) else list(x or [])
    missing_cols = [c for c in [y] + xs if c not in data.columns]
    if missing_cols:
        raise MethodIncompatibility(
            f"ardl: column(s) {missing_cols} are not in data.",
            recovery_hint="Check the column names.",
        )
    frame = data if time is None else data.sort_values(time, kind="stable")
    if time is not None:
        frame = frame.set_index(time)
    frame = frame[[y] + xs]
    if not frame.index.is_unique:
        raise MethodIncompatibility(
            "ardl: the time index has repeated values.",
            recovery_hint="Pass one row per period; for a panel, fit each "
            "unit separately.",
        )
    complete = np.flatnonzero(frame.notna().all(axis=1).to_numpy())
    if complete.size == 0:
        raise DataInsufficient(
            "ardl: no row has all of the series observed.",
            recovery_hint="Check the columns.",
        )
    first_complete, last_complete = int(complete[0]), int(complete[-1])
    T = last_complete - first_complete + 1

    def order_of(var: str, p: int) -> int:
        if isinstance(x_lags, dict):
            if var not in x_lags:
                raise MethodIncompatibility(
                    f"ardl: x_lags has no entry for {var!r}.",
                    recovery_hint="Give every x a lag order.",
                )
            return int(x_lags[var])
        if isinstance(x_lags, str):
            if x_lags != "same":
                raise MethodIncompatibility(
                    f"ardl: unknown x_lags {x_lags!r}.",
                    recovery_hint="Use an integer, a dict or 'same'.",
                )
            return p
        return int(x_lags)

    def build(p: int) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
        q = {var: order_of(var, p) for var in xs}
        if p < 0 or any(v < 0 for v in q.values()):
            raise MethodIncompatibility(
                "ardl: lag orders must be non-negative.",
                recovery_hint="Use lags >= 0.",
            )
        design, names = _design(frame, y, xs, p, q, contemporaneous, trend)
        generated = {_lag_name(y, k) for k in range(1, p + 1)}
        for var in xs:
            generated |= {_lag_name(var, k) for k in range(1, q[var] + 1)}
        clash = sorted(generated & set([y] + xs))
        if clash or len(set(names)) != len(names):
            raise MethodIncompatibility(
                f"ardl: the lag names {clash or names} collide with the "
                "variables passed in.",
                recovery_hint="Rename the column; ardl names lags <var>_L<k>.",
            )
        return design, names, q

    def fit(p: int, rows: pd.Index) -> Tuple[Any, pd.DataFrame, List[str], Dict]:
        design, names, q = build(p)
        formula = f"{y} ~ " + (" + ".join(names) or "1")
        if trend == "n":
            if not names:
                raise MethodIncompatibility(
                    "ardl: trend='n' with no lags leaves nothing to estimate.",
                    recovery_hint="Use trend='c' or lags >= 1.",
                )
            formula += " - 1"
        sub = design.loc[rows]
        holes = int(sub[[y] + names].isna().any(axis=1).sum())
        if holes:
            raise MethodIncompatibility(
                f"ardl: {holes} row(s) of the estimation sample have a "
                "missing value or a missing lag. A lag is not defined across "
                "a gap in the series.",
                recovery_hint="Fill or drop the gap deliberately, or start "
                "the sample after it.",
                diagnostics={"n_incomplete_rows": holes},
            )
        if len(sub) <= len(names) + 2:
            raise DataInsufficient(
                f"ardl: {len(sub)} observations for {len(names) + 1} " "coefficients.",
                recovery_hint="Use fewer lags or a longer sample.",
            )
        kwargs: Dict[str, Any] = {"robust": vce}
        if hac_lags is not None:
            kwargs["hac_lags"] = hac_lags
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return regress(formula, data=sub, **kwargs), design, names, q

    # the rows a given maximum lag leaves usable, within `sample`
    def rows_for(depth: int) -> pd.Index:
        lo = first_complete + depth
        hi = last_complete
        if sample is not None:
            lo = max(lo, _position(frame.index, sample[0]))
            hi = _position(frame.index, sample[1])
        if hi < lo:
            raise DataInsufficient(
                "ardl: the sample is empty once the lags are taken.",
                recovery_hint="Widen the sample or use fewer lags.",
            )
        return frame.index[lo : hi + 1]

    ic_table: Optional[pd.DataFrame] = None
    if isinstance(lags, str):
        criterion = lags.lower()
        if criterion not in ("bic", "aic"):
            raise MethodIncompatibility(
                f"ardl: unknown lag rule {lags!r}.",
                recovery_hint="Pass an integer, 'bic' or 'aic'.",
            )
        if max_lags is None:
            max_lags = min(8, int(np.floor(12.0 * (T / 100.0) ** 0.25)))
        deepest = max([max_lags] + [order_of(v, max_lags) for v in xs])
        rows = rows_for(deepest)
        records = []
        for p in range(int(max_lags) + 1):
            if trend == "n" and p == 0 and not xs:
                continue
            model, _, names, _ = fit(p, rows)
            resid = np.asarray(_residuals(model), dtype=float)
            n, k = len(rows), len(model.params)
            base = np.log(float(resid @ resid) / n)
            records.append(
                {
                    "lags": p,
                    "bic": base + k * np.log(n) / n,
                    "aic": base + k * 2.0 / n,
                    "r2": float(model.r2),
                    "nobs": n,
                }
            )
        ic_table = pd.DataFrame(records).set_index("lags")
        p_final = int(ic_table[criterion].idxmin())
    else:
        p_final = int(lags)
        deepest = max([p_final] + [order_of(v, p_final) for v in xs])
        rows = rows_for(deepest)

    model, design, names, q = fit(p_final, rows)
    return ARDLResult(
        model=model,
        y=y,
        x=xs,
        lags=p_final,
        x_lags=q,
        contemporaneous=contemporaneous,
        trend=trend,
        regressors=names,
        design=design,
        sample_index=rows,
        fit_kwargs={"robust": vce, "hac_lags": hac_lags},
        ic_table=ic_table,
        alpha=alpha,
    )
