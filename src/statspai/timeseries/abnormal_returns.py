"""
Event studies on security returns.

``sp.abnormal_returns`` measures how prices moved around an event: for each
event a model of normal returns is fitted on an estimation window that ends
before the event, the abnormal return is the realised return minus the
model's prediction, and the cumulative abnormal return (CAR) adds them over
the event window. Across events the mean CAR is tested against zero.

This is the event study of the finance and accounting literature
[@brown1985using], not the event-study plot of a difference-in-differences
design (``sp.event_study``).

Tests of the mean CAR

* ``cross_sectional``: the t test on the CARs themselves. Valid under
  event-induced variance, weak when the securities differ much in
  volatility.
* ``patell`` [@patell1976corporate]: each CAR is divided by its own
  standard deviation before averaging, which weights quiet securities up.
  Assumes the event does not change the variance.
* ``bmp`` [@boehmer1991event]: the t test on the standardised CARs.
  Robust to event-induced variance.
* ``adj_patell`` and ``kp`` [@kolari2010event]: the two above corrected for
  cross-sectional correlation of abnormal returns, which matters when
  event dates cluster in calendar time.

Reference implementation: Stata ``estudy`` [@pacicco2018event].
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

_MODELS = ("market", "market_adjusted", "mean_adjusted", "factor")


class AbnormalReturnsResult(ResultProtocolMixin):
    """Outcome of :func:`abnormal_returns`.

    Attributes
    ----------
    events : pd.DataFrame
        One row per event: ``id``, ``event_date`` as given, ``event_day``
        (the trading day used as day 0), ``n_est`` (estimation
        observations), ``car``, its standard error ``se``, ``t``,
        ``pvalue``, the standardised CAR ``scar``, and the residual
        variance ``resid_var`` with its degrees of freedom ``dof``.
    ar : pd.DataFrame
        Abnormal returns in the event window; rows are relative days,
        columns are events (the index of ``events``).
    aar : pd.DataFrame
        By relative day: the average abnormal return ``aar``, the number
        of events ``n`` and the running sum ``caar``.
    tests : pd.DataFrame
        One row per test of the mean CAR, with ``statistic`` and
        ``pvalue``.
    caar, se, pvalue : float
        The mean CAR, its cross-sectional standard error and the p-value
        of the cross-sectional t test.
    n_events : int
    mean_correlation : float
        Average pairwise correlation of the estimation-window residuals,
        the quantity behind the Kolari-Pynnonen adjustments.
    skipped : list
        Events left out, each with the reason.

    Examples
    --------
    >>> import statspai as sp
    >>> isinstance(sp.AbnormalReturnsResult._citation_keys, tuple)
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "brown1985using",
        "boehmer1991event",
        "kolari2010event",
    )

    def __init__(
        self,
        *,
        events: pd.DataFrame,
        ar: pd.DataFrame,
        aar: pd.DataFrame,
        tests: pd.DataFrame,
        model: str,
        event_window: Tuple[int, int],
        estimation_window: Tuple[int, int],
        mean_correlation: float,
        correlation: str,
        skipped: List[Dict[str, Any]],
        alpha: float,
    ) -> None:
        self.method = f"Event study ({model} model)"
        self.estimand = "mean cumulative abnormal return"
        self.events = events
        self.ar = ar
        self.aar = aar
        self.tests = tests
        self.model = model
        self.event_window = event_window
        self.estimation_window = estimation_window
        self.mean_correlation = mean_correlation
        self.correlation = correlation
        self.skipped = skipped
        self.alpha = alpha
        self.n_events = int(len(events))
        self.n_obs = self.n_events
        self.caar = float(events["car"].mean())
        self.estimate = self.caar
        row = tests.loc["cross_sectional"]
        self.se = float(self.caar / row["statistic"]) if row["statistic"] else np.nan
        self.pvalue = float(row["pvalue"])

    def summary(self) -> str:
        lo, hi = self.event_window
        lines = [
            self.method,
            "=" * len(self.method),
            f"Events: {self.n_events}   event window: [{lo}, {hi}]   "
            f"estimation window: [{self.estimation_window[0]}, "
            f"{self.estimation_window[1]}]",
            f"Mean CAR: {self.caar:.6f}   (cross-sectional se {self.se:.6f})",
            "",
            f"{'test':<18}{'statistic':>12}{'p-value':>12}",
        ]
        for name, row in self.tests.iterrows():
            lines.append(f"{name:<18}{row['statistic']:>12.4f}{row['pvalue']:>12.4f}")
        lines.append("")
        lines.append(
            f"Mean correlation of estimation residuals ({self.correlation} "
            f"time): {self.mean_correlation:.4f}"
        )
        if self.skipped:
            lines.append(f"Events left out: {len(self.skipped)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AbnormalReturnsResult(n_events={self.n_events}, "
            f"caar={self.caar:.5g}, p={self.pvalue:.4g})"
        )


def _car_tests(
    scar: np.ndarray, dof: np.ndarray, car: np.ndarray, r_bar: float
) -> pd.DataFrame:
    """Tests of the mean CAR from the standardised CARs.

    ``scar`` holds each event's CAR over its standard deviation, ``dof``
    the degrees of freedom of the variance estimate behind that standard
    deviation, ``r_bar`` the mean cross-correlation of abnormal returns.
    """
    from scipy import stats

    n = len(scar)
    rows: Dict[str, Tuple[float, float]] = {}
    cs = car.mean() / (car.std(ddof=1) / np.sqrt(n))
    rows["cross_sectional"] = (cs, 2 * stats.t.sf(abs(cs), n - 1))
    # a standardised CAR is t(dof) under the null, with variance
    # dof / (dof - 2): (M - 2) / (M - 4) for the market model
    patell = scar.sum() / np.sqrt(np.sum(dof / (dof - 2.0)))
    rows["patell"] = (patell, 2 * stats.norm.sf(abs(patell)))
    bmp = scar.mean() / (scar.std(ddof=1) / np.sqrt(n))
    rows["bmp"] = (bmp, 2 * stats.norm.sf(abs(bmp)))
    inflate = 1.0 + (n - 1.0) * r_bar
    if inflate > 0:
        adj = patell * np.sqrt(1.0 / inflate)
        kp = bmp * np.sqrt((1.0 - r_bar) / inflate)
    else:
        adj = kp = np.nan
    rows["adj_patell"] = (adj, 2 * stats.norm.sf(abs(adj)))
    rows["kp"] = (kp, 2 * stats.norm.sf(abs(kp)))
    out = pd.DataFrame(rows, index=["statistic", "pvalue"]).T
    out.index.name = "test"
    return out


def _mean_correlation(residuals: Dict[Any, pd.Series]) -> float:
    """Average pairwise correlation of the residual series, over the
    index values each pair shares."""
    frame = pd.DataFrame(residuals)
    if frame.shape[1] < 2:
        return 0.0
    corr = frame.corr(min_periods=3).to_numpy()
    off = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[off]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def abnormal_returns(
    data: pd.DataFrame,
    events: pd.DataFrame,
    *,
    id: str = "id",
    date: str = "date",
    ret: str = "ret",
    event_date: str = "event_date",
    model: str = "market",
    market: Optional[str] = None,
    factors: Optional[Sequence[str]] = None,
    event_window: Tuple[int, int] = (-1, 1),
    estimation_window: Tuple[int, int] = (-250, -11),
    min_obs: int = 30,
    correlation: str = "calendar",
    alpha: float = 0.05,
) -> AbnormalReturnsResult:
    """Event study: abnormal returns around events, and tests of their mean.

    Parameters
    ----------
    data : pd.DataFrame
        Returns in long format: one row per security and trading day, with
        the security's return and, for the market and factor models, the
        market or factor returns of that day as columns.
    events : pd.DataFrame
        One row per event, with the security and the event date. A
        security may have several events.
    id, date, ret : str
        Columns of ``data``: security, trading date, return.
    event_date : str
        Column of ``events`` with the event date. ``events`` names the
        security in the same ``id`` column.
    model : {'market', 'market_adjusted', 'mean_adjusted', 'factor'}
        The model of normal returns. ``'market'``: a regression of the
        return on the market return, fitted on the estimation window.
        ``'market_adjusted'``: the return minus the market return, nothing
        estimated. ``'mean_adjusted'``: the return minus its
        estimation-window mean. ``'factor'``: a regression on the
        ``factors`` columns.
    market : str, optional
        Market return column, for ``'market'`` and ``'market_adjusted'``.
    factors : sequence of str, optional
        Factor return columns, for ``'factor'``.
    event_window : (int, int), default (-1, 1)
        First and last trading day of the window, relative to the event
        day (day 0).
    estimation_window : (int, int), default (-250, -11)
        Trading days, relative to the event day, on which the model is
        fitted. It must end before the event window begins. Days before
        the security's first observation are simply absent.
    min_obs : int, default 30
        An event with fewer estimation-window returns is left out, with a
        warning.
    correlation : {'calendar', 'event'}, default 'calendar'
        How the estimation residuals of two events are paired to measure
        cross-correlation for ``adj_patell`` and ``kp``. ``'calendar'``
        pairs returns of the same date, which is the correlation that
        clustering of event dates induces. ``'event'`` pairs returns the
        same number of days before each event, as Stata's ``estudy`` does;
        with different event dates that correlation is zero by
        construction.
    alpha : float, default 0.05

    Returns
    -------
    AbnormalReturnsResult

    Notes
    -----
    *Day 0.* The event day is the security's first trading day on or after
    the event date, so an announcement on a Saturday is assigned to Monday.
    Relative days count the security's own trading days.

    *Variance of a CAR.* For the regression models it is the variance of
    the forecast error, ``s^2 [L + 1'X*(X'X)^{-1}X*'1]`` with ``s^2`` the
    residual variance on ``n - k`` degrees of freedom, ``L`` the window
    length and ``X*`` the regressors in the event window: the second term
    is the error in the estimated coefficients, which is common to the
    days of the window [@brown1985using]. Each CAR is referred to a t
    distribution with ``n - k`` degrees of freedom.

    ``estudy`` uses the residual variance on ``n - 1`` degrees of freedom
    and a slightly different market term, so its standard deviations are
    about ``1 / (2n)`` smaller; given the same standardised CARs the test
    statistics here and there are equal. For the Patell statistic it
    takes ``(M - 2) / (M - 4)`` as the variance of a standardised CAR
    under the market-adjusted and mean-adjusted models too, where the
    variance estimate has ``M - 1`` degrees of freedom and this function
    uses ``(M - 1) / (M - 3)``; the statistics differ by 3e-5.

    *Overlapping windows.* When several securities share an event date
    their abnormal returns are correlated and the unadjusted tests
    over-reject. ``adj_patell`` and ``kp`` correct for the average
    correlation; with one common event date a calendar-time portfolio is
    the cleaner design.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> days = pd.bdate_range("2020-01-01", periods=300)
    >>> mkt = rng.normal(0, 0.01, 300)
    >>> rows = []
    >>> for j in range(20):
    ...     r = 1.1 * mkt + rng.normal(0, 0.01, 300)
    ...     r[280] += 0.03                      # the event
    ...     rows.append(pd.DataFrame({"id": j, "date": days, "ret": r, "mkt": mkt}))
    >>> data = pd.concat(rows)
    >>> events = pd.DataFrame({"id": range(20), "event_date": days[280]})
    >>> res = sp.abnormal_returns(data, events, market="mkt", event_window=(0, 0))
    >>> res.n_events
    20
    >>> bool(0.02 < res.caar < 0.04) and bool(res.pvalue < 0.001)
    True

    References
    ----------
    brown1985using, patell1976corporate, boehmer1991event, kolari2010event,
    pacicco2018event
    """
    from scipy import stats

    model_key = str(model).lower()
    if model_key not in _MODELS:
        raise MethodIncompatibility(
            f"abnormal_returns: model must be one of {_MODELS}, got {model!r}."
        )
    if str(correlation).lower() not in ("calendar", "event"):
        raise MethodIncompatibility(
            "abnormal_returns: correlation must be 'calendar' or 'event', "
            f"got {correlation!r}."
        )
    corr_key = str(correlation).lower()
    regressors: List[str] = []
    if model_key in ("market", "market_adjusted"):
        if not market:
            raise MethodIncompatibility(
                f"abnormal_returns: model={model_key!r} needs market=, the "
                "column with the market return."
            )
        regressors = [market]
    elif model_key == "factor":
        if not factors:
            raise MethodIncompatibility(
                "abnormal_returns: model='factor' needs factors=, the "
                "columns with the factor returns."
            )
        regressors = list(factors)
    for name, frame, cols in (
        ("data", data, [id, date, ret] + regressors),
        ("events", events, [id, event_date]),
    ):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise MethodIncompatibility(
                f"abnormal_returns: {name} has no column(s) {missing}."
            )
    ev_lo, ev_hi = (int(event_window[0]), int(event_window[1]))
    es_lo, es_hi = (int(estimation_window[0]), int(estimation_window[1]))
    if ev_lo > ev_hi or es_lo > es_hi:
        raise MethodIncompatibility(
            "abnormal_returns: a window is given as (first, last) with "
            "first <= last."
        )
    if es_hi >= ev_lo:
        raise MethodIncompatibility(
            "abnormal_returns: the estimation window must end before the "
            f"event window begins; got estimation_window={estimation_window} "
            f"and event_window={event_window}.",
            recovery_hint="An overlap lets the event move the model of "
            "normal returns.",
        )
    length = ev_hi - ev_lo + 1
    k = {"market_adjusted": 0, "mean_adjusted": 1}.get(model_key, 1 + len(regressors))

    panel = data[[id, date, ret] + regressors].copy()
    panel[date] = pd.to_datetime(panel[date])
    if panel.duplicated([id, date]).any():
        raise MethodIncompatibility(
            f"abnormal_returns: {id!r} and {date!r} do not identify the rows "
            "of data."
        )
    panel = panel.sort_values([id, date])
    by_id = {key: grp.reset_index(drop=True) for key, grp in panel.groupby(id)}
    ev = events[[id, event_date]].copy()
    ev[event_date] = pd.to_datetime(ev[event_date])

    records: List[Dict[str, Any]] = []
    ar_columns: Dict[Any, pd.Series] = {}
    residuals: Dict[Any, pd.Series] = {}
    skipped: List[Dict[str, Any]] = []
    rel_index = np.arange(ev_lo, ev_hi + 1)
    for label, row in ev.iterrows():
        unit, when = row[id], row[event_date]

        def skip(reason: str) -> None:
            skipped.append({"event": label, id: unit, "reason": reason})

        sec = by_id.get(unit)
        if sec is None or pd.isna(when):
            skip("no returns for this security" if sec is None else "no event date")
            continue
        pos = int(np.searchsorted(sec[date].to_numpy(), np.datetime64(when)))
        if pos >= len(sec):
            skip("event date after the last return")
            continue
        if pos + ev_lo < 0 or pos + ev_hi >= len(sec):
            skip("event window runs outside the return series")
            continue
        window = sec.iloc[pos + ev_lo : pos + ev_hi + 1]
        if window[[ret] + regressors].isna().any().any():
            skip("missing return in the event window")
            continue
        # an event near the start of the series has a short or empty
        # estimation window; a negative stop would slice from the end
        start = max(pos + es_lo, 0)
        stop = max(pos + es_hi + 1, start)
        est = sec.iloc[start:stop]
        rel_est = np.arange(start, stop) - pos
        ok = est[[ret] + regressors].notna().all(axis=1).to_numpy()
        est, rel_est = est[ok], rel_est[ok]
        n = len(est)
        if n < max(int(min_obs), k + 3):
            skip(f"{n} estimation-window returns")
            continue

        y_est = est[ret].to_numpy(dtype=float)
        y_win = window[ret].to_numpy(dtype=float)
        if model_key == "market_adjusted":
            res_est = y_est - est[market].to_numpy(dtype=float)
            ar = y_win - window[market].to_numpy(dtype=float)
            s2 = float(np.var(res_est, ddof=1))
            var_car = s2 * length
            dof = n - 1
        else:
            X = np.column_stack(
                [np.ones(n)] + [est[c].to_numpy(dtype=float) for c in regressors]
            )
            Xw = np.column_stack(
                [np.ones(length)]
                + [window[c].to_numpy(dtype=float) for c in regressors]
            )
            if np.linalg.matrix_rank(X) < X.shape[1]:
                skip("collinear regressors in the estimation window")
                continue
            beta, *_ = np.linalg.lstsq(X, y_est, rcond=None)
            res_est = y_est - X @ beta
            ar = y_win - Xw @ beta
            dof = n - k
            s2 = float(res_est @ res_est) / dof
            ones = Xw.sum(axis=0)
            var_car = s2 * (length + float(ones @ np.linalg.solve(X.T @ X, ones)))
        se = float(np.sqrt(var_car))
        car = float(ar.sum())
        t = car / se
        records.append(
            {
                "event": label,
                id: unit,
                "event_date": when,
                "event_day": sec[date].iloc[pos],
                "n_est": n,
                "car": car,
                "se": se,
                "t": t,
                "pvalue": float(2 * stats.t.sf(abs(t), dof)),
                "scar": t,
                "dof": dof,
                "resid_var": s2,
            }
        )
        ar_columns[label] = pd.Series(ar, index=rel_index)
        key = est[date].to_numpy() if corr_key == "calendar" else rel_est
        residuals[label] = pd.Series(res_est, index=key)

    if skipped:
        warnings.warn(
            f"abnormal_returns: {len(skipped)} event(s) left out "
            f"(first: {skipped[0]['reason']}); see result.skipped.",
            UserWarning,
            stacklevel=2,
        )
    if len(records) < 2:
        raise DataInsufficient(
            f"abnormal_returns: {len(records)} usable event(s); the tests "
            "compare events and need at least two.",
            recovery_hint="Check the dates, the windows and min_obs.",
            diagnostics={"skipped": skipped},
        )
    table = pd.DataFrame(records).set_index("event")
    ar_frame = pd.DataFrame(ar_columns)
    ar_frame.index.name = "day"
    aar = pd.DataFrame(
        {"aar": ar_frame.mean(axis=1), "n": ar_frame.notna().sum(axis=1)}
    )
    aar["caar"] = aar["aar"].cumsum()
    r_bar = _mean_correlation(residuals)
    tests = _car_tests(
        table["scar"].to_numpy(dtype=float),
        table["dof"].to_numpy(dtype=float),
        table["car"].to_numpy(dtype=float),
        r_bar,
    )
    return AbnormalReturnsResult(
        events=table,
        ar=ar_frame,
        aar=aar,
        tests=tests,
        model=model_key,
        event_window=(ev_lo, ev_hi),
        estimation_window=(es_lo, es_hi),
        mean_correlation=r_bar,
        correlation=corr_key,
        skipped=skipped,
        alpha=alpha,
    )
