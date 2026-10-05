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

import itertools
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

    def forecast(
        self,
        alpha: Optional[float] = None,
        *,
        steps: int = 1,
        exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Forecasts of the periods after the last estimation observation.

        Beyond one step the forecast is dynamic: each forecast of ``y``
        stands in for the lag the next one needs.

        Parameters
        ----------
        alpha : float, optional
            Level of the interval. Default: the model's.
        steps : int, default 1
            Number of periods ahead.
        exog : pandas.DataFrame, optional
            Future values of the ``x`` series, one row per period after the
            sample in order, with the columns named as in the fit. Needed
            whenever a forecast uses an ``x`` dated after the sample: from
            the first step with ``contemporaneous=True``, from the second
            otherwise.

        Returns
        -------
        pandas.DataFrame
            One row per period with ``forecast``, ``rmsfe``, ``lower``,
            ``upper``.

        Notes
        -----
        The interval is ``forecast +/- z * RMSFE``. At one step the RMSFE
        is the standard error of the regression. At step ``h`` it is that
        times ``sqrt(1 + psi_1**2 + ... + psi_{h-1}**2)``, where ``psi``
        are the moving-average weights of the autoregressive part. It
        ignores the estimation error in the coefficients, treats the
        future ``x`` as known and assumes normal errors; a pseudo
        out-of-sample RMSFE from :meth:`poos` is the more honest width at
        one step.
        """
        steps = int(steps)
        if steps < 1:
            raise MethodIncompatibility("ardl.forecast: steps must be at least 1.")
        level = self.alpha if alpha is None else alpha
        frame = self._design
        origin = int(frame.index.get_loc(self._sample_index[-1]))
        if origin != len(frame) - 1:
            # rows after the estimation sample are data the fit did not use;
            # forecasting past them would silently ignore them
            frame = frame.iloc[: origin + 1]

        first_x_lag = 0 if self.contemporaneous else 1
        used_x = [v for v in self.x if self.x_lags[v] >= first_x_lag]
        need = steps - first_x_lag  # future rows of x the recursion reads
        future = None
        if used_x and need > 0:
            if exog is None:
                why = (
                    "the model uses x at date t"
                    if self.contemporaneous
                    else f"a {steps}-step forecast uses x after the sample"
                )
                raise MethodIncompatibility(
                    f"ardl.forecast: {why}, which is not known when the "
                    "forecast is made.",
                    recovery_hint=(
                        "Pass exog= with the assumed future values of "
                        f"{used_x} ({need} row(s))"
                        + (
                            ", or refit with contemporaneous=False."
                            if self.contemporaneous and steps == 1
                            else "."
                        )
                    ),
                )
            absent = [v for v in used_x if v not in exog.columns]
            if absent or len(exog) < need:
                raise MethodIncompatibility(
                    f"ardl.forecast: exog must have columns {used_x} and at "
                    f"least {need} row(s).",
                    diagnostics={"missing_columns": absent, "rows": len(exog)},
                )
            future = exog[used_x].to_numpy(dtype=float)[:need]
            if np.isnan(future).any():
                raise MethodIncompatibility(
                    "ardl.forecast: exog has missing values in the rows used."
                )

        series = {self.y: list(frame[self.y].to_numpy(dtype=float))}
        for var in self.x:
            values = list(frame[var].to_numpy(dtype=float))
            if future is not None and var in used_x:
                values += list(future[:, used_x.index(var)])
            series[var] = values
        const = (
            float(self.params["Intercept"]) if "Intercept" in self.params.index else 0.0
        )
        coef = self.params[self.regressors].to_numpy(dtype=float)
        terms = [
            ("_trend", 0) if name == "_trend" else _split_lag(name, [self.y] + self.x)
            for name in self.regressors
        ]
        points = []
        for h in range(1, steps + 1):
            t = origin + h  # 0-based position of the period being forecast
            row = np.array(
                [
                    float(t + 1) if var == "_trend" else series[var][t - k]
                    for var, k in terms
                ]
            )
            if np.isnan(row).any():
                raise DataInsufficient(
                    "ardl: a lag needed for the forecast is missing.",
                    recovery_hint="The series must be observed through the "
                    "forecast origin.",
                )
            point = const + float(row @ coef)
            points.append(point)
            series[self.y].append(point)

        # psi weights of the AR polynomial: psi_0 = 1, psi_j = sum phi_i psi_{j-i}
        phi = np.array(
            [
                float(self.params.get(_lag_name(self.y, i), 0.0))
                for i in range(1, self.lags + 1)
            ]
        )
        psi = [1.0]
        for j in range(1, steps):
            psi.append(
                sum(phi[i - 1] * psi[j - i] for i in range(1, min(j, len(phi)) + 1))
            )
        rmsfe = self.ser * np.sqrt(np.cumsum(np.square(psi)))
        z = float(stats.norm.ppf(1 - level / 2))
        fc = np.asarray(points)
        return pd.DataFrame(
            {
                "forecast": fc,
                "rmsfe": rmsfe,
                "lower": fc - z * rmsfe,
                "upper": fc + z * rmsfe,
            },
            index=pd.Index([f"T+{h}" for h in range(1, steps + 1)], name="period"),
        )

    # ----------------------------------------------------------- long run
    def long_run(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Long-run effect of each ``x`` on ``y``.

        A permanent unit rise in ``x`` moves ``y`` in the end by the sum of
        the coefficients on ``x`` and its lags divided by one minus the sum
        of the autoregressive coefficients. In the geometric-lag model
        ``y[t] = a + b x[t] + lam y[t-1]`` this is ``b / (1 - lam)``.

        Returns
        -------
        pandas.DataFrame
            One row per ``x`` in the model, and ``Intercept`` for the
            long-run mean of ``y`` at ``x = 0`` when the model has a
            constant, with ``coef``, ``std err``, ``z``, ``pvalue`` and the
            confidence limits. The standard errors come from the delta
            method on the model's covariance matrix, so they are robust
            when the fit is. ``attrs['ar_sum']`` is the sum of the
            autoregressive coefficients and ``attrs['adjustment']`` that
            sum minus one, the error-correction coefficient.

        Notes
        -----
        The ratio has no finite-sample moments and its normal
        approximation is poor when the sum of the autoregressive
        coefficients is near one. With that sum at or above one there is
        no long-run level to converge to and the method refuses.
        """
        level = self.alpha if alpha is None else alpha
        names = list(self.params.index)
        beta = self.params.to_numpy(dtype=float)
        cov = np.asarray(self.model.data_info["var_cov"], dtype=float)
        ar_idx = [
            names.index(_lag_name(self.y, i))
            for i in range(1, self.lags + 1)
            if _lag_name(self.y, i) in names
        ]
        ar_sum = float(beta[ar_idx].sum()) if ar_idx else 0.0
        gap = 1.0 - ar_sum
        if not gap > 0:
            raise MethodIncompatibility(
                f"ardl.long_run: the autoregressive coefficients sum to "
                f"{ar_sum:.4f}; the model has no long-run level.",
                recovery_hint="Difference the series, or test for "
                "cointegration with sp.engle_granger.",
                diagnostics={"ar_sum": ar_sum},
            )
        targets: Dict[str, List[int]] = {}
        if "Intercept" in names:
            targets["Intercept"] = [names.index("Intercept")]
        for var in self.x:
            idx = [
                names.index(n)
                for n in self.regressors
                if n != "_trend" and _split_lag(n, [self.y] + self.x)[0] == var
            ]
            if idx:
                targets[var] = idx
        if not targets:
            raise MethodIncompatibility(
                "ardl.long_run: the model has neither a constant nor an x.",
                recovery_hint="Fit with x=[...].",
            )
        rows = {}
        z_crit = float(stats.norm.ppf(1 - level / 2))
        for label, idx in targets.items():
            num = float(beta[idx].sum())
            theta = num / gap
            grad = np.zeros(len(beta))
            grad[idx] = 1.0 / gap
            grad[ar_idx] = num / gap**2
            se = float(np.sqrt(grad @ cov @ grad))
            zval = theta / se if se > 0 else np.nan
            rows[label] = {
                "coef": theta,
                "std err": se,
                "z": zval,
                "pvalue": float(2 * stats.norm.sf(abs(zval))),
                "ci_lower": theta - z_crit * se,
                "ci_upper": theta + z_crit * se,
            }
        out = pd.DataFrame(rows).T
        out.attrs["ar_sum"] = ar_sum
        out.attrs["adjustment"] = -gap
        return out

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
    x_lags : int, dict, "same", "bic" or "aic", default 1
        Lags of each ``x``: one order for all, a ``{name: order}`` dict, or
        ``"same"`` to tie it to ``p`` (the ADL(p, p) family a criterion then
        searches over). A criterion searches the order of every ``x``
        separately, from leaving it out to ``max_lags``, jointly with ``p``
        when ``lags`` names the same criterion and for the given ``p``
        otherwise: the search of statsmodels' ``ardl_select_order``. Every
        candidate is fitted on the rows the deepest one leaves. An ``x``
        the search leaves out has order ``-1`` in ``x_lags`` with
        ``contemporaneous=True`` and ``0`` without, one below its first lag.
        The search fits ``(max_lags + 1) * (max_lags + 2) ** len(x)``
        regressions with ``contemporaneous=True``.
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

    # An x is out of the model when its order is one below its first lag.
    q_floor = -1 if contemporaneous else 0
    x_criterion: Optional[str] = None
    if isinstance(x_lags, str) and x_lags.lower() in ("bic", "aic"):
        x_criterion = x_lags.lower()
        if not xs:
            raise MethodIncompatibility(
                f"ardl: x_lags={x_lags!r} selects the lags of x, but no x was given.",
                recovery_hint="Pass x=[...], or drop x_lags.",
            )
        if isinstance(lags, str) and lags.lower() != x_criterion:
            raise MethodIncompatibility(
                f"ardl: lags={lags!r} and x_lags={x_lags!r} name different "
                "criteria; one search ranks every candidate.",
                recovery_hint="Use the same criterion for both.",
            )

    def build(
        p: int, orders: Optional[Dict[str, int]] = None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
        q = dict(orders) if orders is not None else {v: order_of(v, p) for v in xs}
        searched = orders is not None
        if p < 0 or any(v < (q_floor if searched else 0) for v in q.values()):
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

    def fit(
        p: int, rows: pd.Index, orders: Optional[Dict[str, int]] = None
    ) -> Tuple[Any, pd.DataFrame, List[str], Dict]:
        design, names, q = build(p, orders)
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
    q_final: Optional[Dict[str, int]] = None
    if x_criterion is not None:
        # Every combination of orders, each fitted on the rows the deepest
        # candidate leaves, so the criteria rank like with like.
        if isinstance(lags, str):
            if max_lags is None:
                max_lags = min(8, int(np.floor(12.0 * (T / 100.0) ** 0.25)))
            p_grid = list(range(int(max_lags) + 1))
        else:
            if max_lags is None:
                max_lags = max(
                    int(lags), min(8, int(np.floor(12.0 * (T / 100.0) ** 0.25)))
                )
            p_grid = [int(lags)]
        q_grid = list(range(q_floor, int(max_lags) + 1))
        n_candidates = len(p_grid) * len(q_grid) ** len(xs)
        if n_candidates > 20000:
            raise MethodIncompatibility(
                f"ardl: {n_candidates} candidate models to search.",
                recovery_hint="Lower max_lags, or fix some of the orders.",
                diagnostics={"n_candidates": n_candidates},
            )
        rows = rows_for(max(max(p_grid), int(max_lags)))
        records = []
        for p in p_grid:
            for combo in itertools.product(q_grid, repeat=len(xs)):
                orders = dict(zip(xs, combo))
                if trend == "n" and p == 0 and all(c == q_floor for c in combo):
                    continue
                model, _, names, _ = fit(p, rows, orders)
                resid = np.asarray(_residuals(model), dtype=float)
                n, k = len(rows), len(model.params)
                base = np.log(float(resid @ resid) / n)
                records.append(
                    {
                        "lags": p,
                        **orders,
                        "bic": base + k * np.log(n) / n,
                        "aic": base + k * 2.0 / n,
                        "nobs": n,
                    }
                )
        ic_table = pd.DataFrame(records).set_index(["lags"] + xs)
        best = ic_table[x_criterion].idxmin()
        p_final = int(best[0])
        q_final = {v: int(o) for v, o in zip(xs, best[1:])}
    elif isinstance(lags, str):
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

    model, design, names, q = fit(p_final, rows, q_final)
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
