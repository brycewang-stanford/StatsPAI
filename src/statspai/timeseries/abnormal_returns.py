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
* ``grank`` [@kolari2011nonparametric]: the generalized rank test. Each
  standardised CAR is ranked among that security's own standardised
  estimation-window abnormal returns; no distribution is assumed, and the
  test stays valid under event-induced variance and clustered dates.
* ``sign_rank``: Wilcoxon's signed-rank test on the CARs, for a median of
  zero.

Reference implementation: Stata ``estudy`` [@pacicco2018event].
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._frame_input import reads_part
from .._input_validation import missing_named_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

_MODELS = ("market", "market_adjusted", "mean_adjusted", "factor")


class AbnormalReturnsResult(ResultProtocolMixin):
    """Outcome of :func:`abnormal_returns`.

    Attributes
    ----------
    events : pd.DataFrame
        One row per event: ``id``, ``event_date`` as given, ``event_day``
        (the trading day used as day 0), ``n_est`` (estimation
        observations), ``window_days`` (trading days in the event
        window), ``n_days`` (those of them with a return), ``raw`` (the
        sum of the returns themselves over the event window), ``car``,
        its standard error ``se``, ``t``,
        ``pvalue``, the standardised CAR ``scar``, and the residual
        variance ``resid_var`` with its degrees of freedom ``dof``.
    ar : pd.DataFrame
        Abnormal returns in the event window; rows are relative days,
        columns are events (the index of ``events``).
    returns : pd.DataFrame
        The returns themselves on the same days, in the same layout.
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
    inference : bool
        False for a result of ``inference=False``: ``events`` then holds
        the CARs only, ``tests`` is empty, and ``se``, ``pvalue`` and
        ``mean_correlation`` are NaN.

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
        inference: bool = True,
        returns: Optional[pd.DataFrame] = None,
    ) -> None:
        self.method = f"Event study ({model} model)"
        self.estimand = "mean cumulative abnormal return"
        self.events = events
        self.ar = ar
        self.returns = returns
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
        self.inference = bool(inference)
        if not self.inference:
            self.se = float("nan")
            self.pvalue = float("nan")
            return
        row = tests.loc["cross_sectional"]
        self.se = float(self.caar / row["statistic"]) if row["statistic"] else np.nan
        self.pvalue = float(row["pvalue"])

    def summary(self) -> str:
        lo, hi = self.event_window
        if not self.inference:
            lines = [
                self.method,
                "=" * len(self.method),
                f"Events: {self.n_events}   event window: [{lo}, {hi}]",
                f"Mean CAR: {self.caar:.6f}",
                "",
                "Returns only (inference=False): no standard errors or tests.",
            ]
            if self.skipped:
                lines.append(f"Events left out: {len(self.skipped)}")
            return "\n".join(lines)
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
        if not self.inference:
            return (
                f"AbnormalReturnsResult(n_events={self.n_events}, "
                f"caar={self.caar:.5g}, inference=False)"
            )
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


def _grank(standardised: Dict[Any, pd.Series], scar: np.ndarray) -> Tuple[float, float]:
    """Generalized rank test of Kolari and Pynnonen (2011).

    ``standardised`` holds each event's estimation-window abnormal returns
    over their standard deviation, indexed by relative day. The event
    window is squeezed into one observation per event: its standardised
    CAR, rescaled by the cross-sectional standard deviation of the
    standardised CARs so that event-induced variance does not move the
    ranks. With ``T`` observations per event (estimation days plus that
    one), ``U = rank / (T + 1) - 1/2`` has mean zero; the statistic is the
    mean of ``U`` on the event observation over the standard deviation of
    the daily means, referred to ``t(T - 2)`` after the usual
    transformation.
    """
    from scipy import stats

    frame = pd.DataFrame(standardised)
    cross_sd = float(np.std(scar, ddof=1))
    if not cross_sd > 0:
        return np.nan, np.nan
    event_row = pd.DataFrame([scar / cross_sd], columns=frame.columns, index=["event"])
    full = pd.concat([frame, event_row])
    counts = full.notna().sum(axis=0)
    u = full.rank(axis=0) / (counts + 1.0) - 0.5
    daily = u.mean(axis=1)
    share = u.notna().sum(axis=1) / u.shape[1]
    t_len = len(full)
    s2 = float(np.sum(share * daily**2)) / t_len
    if not s2 > 0:
        return np.nan, np.nan
    z = float(daily.loc["event"]) / np.sqrt(s2)
    inside = t_len - 1 - z * z
    if inside <= 0:
        return np.nan, np.nan
    t = z * np.sqrt((t_len - 2) / inside)
    return float(t), float(2 * stats.t.sf(abs(t), t_len - 2))


def _sign_rank(car: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank z and p for a median CAR of zero (the
    statistic of ``sp.signrank``, Stata's ``signrank``)."""
    from ..inference.rank_tests import signrank

    res = signrank(pd.DataFrame({"car": car}), "car")
    return float(res.statistic), float(res.pvalue)


def _mean_correlation(
    residuals: Dict[Any, pd.Series], units: Optional[Sequence[Any]] = None
) -> float:
    """Average pairwise correlation of the residual series, over the
    index values each pair shares.

    ``units`` (one per series, calendar pairing only) leaves out the pairs
    that belong to one security. Two events of the same security share
    their estimation days, and on those days the two "series" are one
    return series fitted twice: a correlation of one that measures no
    dependence between securities. NaN when no pair of different
    securities is left.
    """
    frame = pd.DataFrame(residuals)
    if frame.shape[1] < 2:
        return 0.0
    corr = frame.corr(min_periods=3).to_numpy()
    off = ~np.eye(corr.shape[0], dtype=bool)
    if units is not None:
        codes = pd.factorize(pd.Series(list(units), dtype=object))[0]
        off &= codes[:, None] != codes[None, :]
        if not off.any():
            return float("nan")
    values = corr[off]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def _part_read(arguments: Dict[str, Any]) -> Any:
    """Columns of ``data`` a call reads, and the securities it reads them
    for: those of ``events`` (see ``_frame_input.reads_part``)."""
    names = [arguments.get(k) for k in ("id", "date", "ret")]
    key = str(arguments.get("model")).lower()
    if key in ("market", "market_adjusted"):
        names.append(arguments.get("market"))
    elif key == "factor":
        factors = arguments.get("factors")
        if not isinstance(factors, (list, tuple)):
            return None
        names += list(factors)
    if not all(isinstance(c, str) for c in names):
        return None
    unit = arguments["id"]
    rows: Tuple[Any, ...] = (unit, arguments.get("events"), unit)
    calendar = arguments.get("calendar", "market")
    if isinstance(calendar, str) and calendar.lower() == "market":
        # the trading days are the dates of the whole file: one row of
        # each date is read along with the rows of the event securities
        rows += (arguments["date"],)
    return list(dict.fromkeys(names)), rows


_INCOMPLETE = ("skip", "sum", "zero")


def _trading_days(calendar: Any, data: pd.DataFrame, date: str) -> Optional[Any]:
    """The trading calendar as a sorted ``DatetimeIndex``; None for
    ``calendar='security'`` (each security's own rows)."""
    if isinstance(calendar, str):
        key = calendar.lower()
        if key == "security":
            return None
        if key != "market":
            raise MethodIncompatibility(
                "abnormal_returns: calendar must be 'market', 'security' or "
                f"a sequence of trading dates, got {calendar!r}.",
                recovery_hint="calendar='market' counts the dates of data; "
                "a sequence such as the date column of an index file gives "
                "the trading days directly.",
            )
        values: Any = pd.unique(data[date])
    else:
        values = calendar.to_numpy() if hasattr(calendar, "to_numpy") else calendar
        values = np.asarray(list(values) if not hasattr(values, "ndim") else values)
        if values.ndim != 1 or values.size == 0:
            raise MethodIncompatibility(
                "abnormal_returns: calendar must be 'market', 'security' or "
                "a one-dimensional sequence of trading dates.",
                recovery_hint="Pass one column of dates, for example the "
                "date column of the market index file.",
            )
    days = pd.DatetimeIndex(pd.to_datetime(values))
    return days[~days.isna()].unique().sort_values()


@reads_part(_part_read)
def abnormal_returns(
    data: pd.DataFrame,
    events: pd.DataFrame,
    *,
    id: str = "id",
    date: str = "date",
    ret: str = "ret",
    event_date: str = "event_date",
    event_end: Optional[str] = None,
    model: str = "market",
    market: Optional[str] = None,
    factors: Optional[Sequence[str]] = None,
    event_window: Tuple[int, int] = (-1, 1),
    estimation_window: Tuple[int, int] = (-250, -11),
    min_obs: int = 30,
    correlation: str = "calendar",
    alpha: float = 0.05,
    inference: bool = True,
    calendar: Any = "market",
    incomplete: str = "skip",
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
    event_end : str, optional
        Column of ``events`` with the last date of an event that runs over
        several days (a conference, a trial). The event window then runs
        from ``event_window[0]`` days relative to the event day to
        ``event_window[1]`` days after the end day, so events differ in
        length. The end day is found as day 0 is: the first trading day on
        or after the date. A missing end date means a one-day event.
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
        day (day 0). With ``event_end`` the last day is relative to the
        end day of each event.
    estimation_window : (int, int), default (-250, -11)
        Trading days, relative to the event day, on which the model is
        fitted. It must end before the event window begins. Days before
        the security's first observation, and days without a row for it,
        are simply absent.
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
    inference : bool, default True
        ``False`` returns the abnormal returns and the CARs with no
        standard errors and no tests: ``result.ar``, ``result.aar`` and
        the ``car`` column of ``result.events``. The variance of a CAR is
        the only thing ``model='market_adjusted'`` takes from the
        estimation window, so that model then reads the event window
        alone: ``estimation_window`` and ``min_obs`` are ignored and an
        event with no earlier returns is kept. The other models still fit
        on the estimation window. One usable event is enough.
    calendar : {'market', 'security'} or sequence of dates, default 'market'
        The trading days that relative days count. ``'market'``: the
        distinct dates of ``data``, over all its securities, so a day on
        which the market was open and a security has no row is still a
        day. A sequence of dates (the date column of an index file such
        as CRSP's ``dsi``) gives the trading days directly; every date of
        an event security must be among them. ``'security'``: each
        security's own rows, the behaviour of 1.46.5 and earlier. The three agree
        when every event security has a row on every trading day.
    incomplete : {'skip', 'sum', 'zero'}, default 'skip'
        What to do with an event whose window has a day without a return
        (a missing value, no row for that day, or a window that runs past
        either end of the calendar). ``'skip'`` leaves the event out, with
        a warning. ``'sum'`` adds up the days that have a return.
        ``'zero'`` counts a missing return as zero where the benchmark of
        that day is known and the security has returns on both sides of
        it, which is right when the next return covers the gap; other
        days are left out as under ``'sum'``. ``result.events`` gives
        ``window_days`` and ``n_days`` (days with a return), so incomplete
        events can be told apart and filtered afterwards.

    Returns
    -------
    AbnormalReturnsResult

    Notes
    -----
    *Events of several days.* With ``event_end`` the rows of ``result.ar``
    count days from the event day and a shorter event has missing values
    in the later rows. ``result.aar`` averages the events that have the
    day and gives their number in ``n``, so its running sum ``caar`` is
    the mean CAR only up to the shortest window. ``result.caar`` and the
    tests use each event's own CAR and its own window length.

    *Day 0.* The event day is the first trading day on or after the event
    date, so an announcement on a Saturday is assigned to Monday. Relative
    days count trading days of ``calendar``: by default the days on which
    the market was open, as the dates of ``data`` show them, not the days
    on which the security itself traded. A security without a row on a
    day of its window has an incomplete window (``incomplete``).

    *Variance of a CAR.* For the regression models it is the variance of
    the forecast error, ``s^2 [L + 1'X*(X'X)^{-1}X*'1]`` with ``s^2`` the
    residual variance on ``n - k`` degrees of freedom, ``L`` the window
    length and ``X*`` the regressors in the event window: the second term
    is the error in the estimated coefficients, which is common to the
    days of the window [@brown1985using]. Each CAR is referred to a t
    distribution with ``n - k`` degrees of freedom.

    ``estudy`` uses the residual variance on ``n - 1`` degrees of freedom
    and a slightly different market term, so its market-model standard
    deviations are about ``1 / (2n)`` smaller; for a factor model it uses
    ``L * RSS / (n - 1)`` with no coefficient-error term, 1 to 6 percent
    smaller on eleven-day windows; given the same standardised CARs the
    parametric test statistics here and there are equal. Its group row is
    the sum over the window of ``log(mean(exp(AR)))``, a log-return
    aggregation that exceeds the mean CAR reported here by half the
    cross-sectional variance of the abnormal returns. Its generalized
    rank statistic scales ranks by ``L1 + 1`` and counts ``L1``
    observations where there are ``T = L1 + 1``; this function follows
    [@kolari2011nonparametric] and is 1 to 2 percent smaller. Its
    ``Wilcoxon`` option ranks each CAR within its own security's series
    and is not the signed-rank test reported here, which is Stata's
    ``signrank`` on the CARs. For the Patell statistic it
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

    Market-adjusted returns from the event window alone:

    >>> short = data[data["date"] >= days[279]]
    >>> only = sp.abnormal_returns(short, events, model="market_adjusted",
    ...                            market="mkt", inference=False)
    >>> only.n_events, list(only.events.columns)[3:]
    (20, ['n_est', 'window_days', 'n_days', 'raw', 'car'])

    References
    ----------
    brown1985using, patell1976corporate, boehmer1991event, kolari2010event,
    kolari2011nonparametric, pacicco2018event
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
    if isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            "abnormal_returns: alpha must lie strictly between 0 and 1, "
            f"got {alpha!r}.",
            recovery_hint="alpha=0.05 gives 95% intervals.",
        )
    if not isinstance(inference, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"abnormal_returns: inference must be True or False, got {inference!r}."
        )
    infer = bool(inference)
    gaps = str(incomplete).lower()
    if gaps not in _INCOMPLETE:
        raise MethodIncompatibility(
            f"abnormal_returns: incomplete must be one of {_INCOMPLETE}, "
            f"got {incomplete!r}.",
            recovery_hint="incomplete='skip' leaves out an event with a day "
            "without a return; 'sum' adds up the days it has.",
        )
    # market-adjusted returns estimate nothing; without inference the
    # estimation window has no use at all
    fitted = infer or model_key != "market_adjusted"
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
    data_cols: Dict[str, Any] = {"id": id, "date": date, "ret": ret}
    if regressors:
        if model_key == "factor":
            data_cols["factors"] = regressors
        else:
            data_cols["market"] = market
    for name, frame, cols in (
        ("data", data, data_cols),
        (
            "events",
            events,
            {"id": id, "event_date": event_date}
            | ({"event_end": event_end} if event_end is not None else {}),
        ),
    ):
        named = missing_named_columns(cols, frame.columns)
        if named:
            missing = list(dict.fromkeys(named.values()))
            raise ColumnNotFound.from_missing(
                named,
                list(frame.columns),
                message=f"abnormal_returns: {name} has no column(s) {missing}.",
                extra_diagnostics={"frame": name},
            )
    ev_lo, ev_hi = (int(event_window[0]), int(event_window[1]))
    es_lo, es_hi = (int(estimation_window[0]), int(estimation_window[1]))
    if ev_lo > ev_hi or es_lo > es_hi:
        raise MethodIncompatibility(
            "abnormal_returns: a window is given as (first, last) with "
            "first <= last."
        )
    if fitted and es_hi >= ev_lo:
        raise MethodIncompatibility(
            "abnormal_returns: the estimation window must end before the "
            f"event window begins; got estimation_window={estimation_window} "
            f"and event_window={event_window}.",
            recovery_hint="An overlap lets the event move the model of "
            "normal returns.",
        )
    k = {"market_adjusted": 0, "mean_adjusted": 1}.get(model_key, 1 + len(regressors))

    # only securities with an event are read: on a full CRSP file that is
    # a small share of the rows, and the copy, the date parse and the sort
    # below are the cost of the call
    days = _trading_days(calendar, data, date)
    wanted = data[id].isin(pd.unique(events[id]))
    panel = data.loc[wanted.to_numpy(), [id, date, ret] + regressors].copy()
    panel[date] = pd.to_datetime(panel[date])
    if panel.duplicated([id, date]).any():
        raise MethodIncompatibility(
            f"abnormal_returns: {id!r} and {date!r} do not identify the rows "
            "of data."
        )
    panel = panel.sort_values([id, date])
    by_id = {key: grp.reset_index(drop=True) for key, grp in panel.groupby(id)}
    ev = events[[id, event_date] + ([event_end] if event_end else [])].copy()
    ev[event_date] = pd.to_datetime(ev[event_date])
    if event_end:
        ev[event_end] = pd.to_datetime(ev[event_end])
    if not ev.index.is_unique:
        # two event lists stacked with pd.concat repeat their labels; an
        # event is then known by its position in the list
        ev = ev.reset_index(drop=True)

    # each security's rows placed on the trading calendar: ``cpos`` is
    # the calendar position of a row, so a day the security has no row
    # for is a gap in ``cpos``
    placed: Dict[Any, Tuple[Any, np.ndarray]] = {}

    def place(unit: Any, sec: pd.DataFrame) -> Tuple[Any, np.ndarray]:
        if unit not in placed:
            own = pd.DatetimeIndex(sec[date])
            if days is None:
                placed[unit] = (own, np.arange(len(sec)))
            else:
                cpos = days.get_indexer(own)
                if (cpos < 0).any():
                    stray = own[cpos < 0]
                    raise MethodIncompatibility(
                        f"abnormal_returns: {len(stray)} date(s) of security "
                        f"{unit!r} are not in calendar (first: "
                        f"{stray[0].date()}).",
                        recovery_hint="calendar must hold every trading "
                        "date of data; calendar='market' takes them from "
                        "data itself.",
                    )
                placed[unit] = (days, cpos)
        return placed[unit]

    records: List[Dict[str, Any]] = []
    ar_columns: Dict[Any, pd.Series] = {}
    raw_columns: Dict[Any, pd.Series] = {}
    residuals: Dict[Any, pd.Series] = {}
    standardised: Dict[Any, pd.Series] = {}
    skipped: List[Dict[str, Any]] = []
    for label, row in ev.iterrows():
        unit, when = row[id], row[event_date]

        def skip(reason: str) -> None:
            skipped.append({"event": label, id: unit, "reason": reason})

        sec = by_id.get(unit)
        if sec is None or pd.isna(when):
            skip("no returns for this security" if sec is None else "no event date")
            continue
        cal, cpos = place(unit, sec)
        pos = int(cal.searchsorted(when))
        if pos >= len(cal):
            skip("event date after the last return")
            continue
        last = pos
        if event_end and not pd.isna(row[event_end]):
            if row[event_end] < when:
                skip("event end before the event date")
                continue
            last = int(cal.searchsorted(row[event_end]))
        first_day, last_day = pos + ev_lo, last + ev_hi
        length = last_day - first_day + 1
        rel_index = np.arange(ev_lo, ev_lo + length)
        on_calendar = (first_day + np.arange(length) >= 0) & (
            first_day + np.arange(length) < len(cal)
        )
        lo_row = int(np.searchsorted(cpos, first_day))
        hi_row = int(np.searchsorted(cpos, last_day, side="right"))
        window = sec.iloc[lo_row:hi_row]
        slot = cpos[lo_row:hi_row] - first_day
        y_win = np.full(length, np.nan)
        y_win[slot] = window[ret].to_numpy(dtype=float)
        x_win = np.full((length, len(regressors)), np.nan)
        if regressors:
            x_win[slot] = window[regressors].to_numpy(dtype=float)
        # a day counts when its return and its benchmark are both there
        benchmark = on_calendar & np.isfinite(x_win).all(axis=1)
        if regressors:
            has_row = np.zeros(length, dtype=bool)
            has_row[slot] = True
            benchmark &= has_row
        observed = benchmark & np.isfinite(y_win)
        use = observed
        if not observed.all():
            if gaps == "skip":
                skip(
                    "event window runs outside the return series"
                    if not on_calendar.all()
                    else "missing return in the event window"
                )
                continue
            if gaps == "zero":
                day = first_day + np.arange(length)
                fill = (
                    benchmark & ~np.isfinite(y_win) & (day > cpos[0]) & (day < cpos[-1])
                )
                y_win[fill] = 0.0
                use = observed | fill
            if not use.any():
                skip("no return in the event window")
                continue
        whole = bool(use.all())
        n_used = int(use.sum())

        def spread(values: np.ndarray) -> pd.Series:
            """Values of the days in use, on the days of the window."""
            if whole:
                return pd.Series(values, index=rel_index)
            out = np.full(length, np.nan)
            out[use] = values
            return pd.Series(out, index=rel_index)

        if not whole:
            y_win, x_win = y_win[use], x_win[use]
        head = {
            "event": label,
            id: unit,
            "event_date": when,
            "event_day": cal[pos],
        }
        if event_end:
            head["event_end"] = row[event_end]
            head["end_day"] = cal[min(last, len(cal) - 1)]
        counts = {"window_days": int(length), "n_days": int(observed.sum())}
        if not fitted:
            ar = y_win - x_win[:, 0]
            records.append(
                {
                    **head,
                    "n_est": 0,
                    **counts,
                    "raw": float(y_win.sum()),
                    "car": float(ar.sum()),
                }
            )
            ar_columns[label] = spread(ar)
            raw_columns[label] = spread(y_win)
            continue
        # the estimation days the security has a row for; an event near
        # the start of the series has a short or empty estimation window
        start = int(np.searchsorted(cpos, pos + es_lo))
        stop = max(int(np.searchsorted(cpos, pos + es_hi, side="right")), start)
        est = sec.iloc[start:stop]
        rel_est = cpos[start:stop] - pos
        ok = est[[ret] + regressors].notna().all(axis=1).to_numpy()
        est, rel_est = est[ok], rel_est[ok]
        n = len(est)
        if n < max(int(min_obs), k + 3):
            skip(f"{n} estimation-window returns")
            continue

        y_est = est[ret].to_numpy(dtype=float)
        if model_key == "market_adjusted":
            res_est = y_est - est[market].to_numpy(dtype=float)
            ar = y_win - x_win[:, 0]
            s2 = float(np.var(res_est, ddof=1))
            var_car = s2 * n_used
            dof = n - 1
        else:
            X = np.column_stack(
                [np.ones(n)] + [est[c].to_numpy(dtype=float) for c in regressors]
            )
            Xw = np.column_stack([np.ones(n_used), x_win])
            if np.linalg.matrix_rank(X) < X.shape[1]:
                skip("collinear regressors in the estimation window")
                continue
            beta, *_ = np.linalg.lstsq(X, y_est, rcond=None)
            res_est = y_est - X @ beta
            ar = y_win - Xw @ beta
            dof = n - k
            s2 = float(res_est @ res_est) / dof
            ones = Xw.sum(axis=0)
            var_car = s2 * (n_used + float(ones @ np.linalg.solve(X.T @ X, ones)))
        # A return series the model reproduces exactly (a return equal to
        # the market's, a suspended security with zero returns) leaves a
        # residual variance that is rounding noise: the standardised CAR
        # would be noise over noise.
        car = float(ar.sum())
        if not infer:
            records.append(
                {**head, "n_est": n, **counts, "raw": float(y_win.sum()), "car": car}
            )
            ar_columns[label] = spread(ar)
            raw_columns[label] = spread(y_win)
            continue
        if not s2 > (1e-10) ** 2 * float(np.mean(y_est**2)):
            skip("no residual variance in the estimation window")
            continue
        se = float(np.sqrt(var_car))
        t = car / se
        records.append(
            {
                **head,
                "n_est": n,
                **counts,
                "raw": float(y_win.sum()),
                "car": car,
                "se": se,
                "t": t,
                "pvalue": float(2 * stats.t.sf(abs(t), dof)),
                "scar": t,
                "dof": dof,
                "resid_var": s2,
            }
        )
        ar_columns[label] = spread(ar)
        raw_columns[label] = spread(y_win)
        key = est[date].to_numpy() if corr_key == "calendar" else rel_est
        residuals[label] = pd.Series(res_est, index=key)
        standardised[label] = pd.Series(res_est / np.sqrt(s2), index=rel_est)

    if skipped:
        warnings.warn(
            f"abnormal_returns: {len(skipped)} event(s) left out "
            f"(first: {skipped[0]['reason']}); see result.skipped. "
            "incomplete='sum' keeps an event with days missing from its "
            "window and reports the days it has in n_days.",
            UserWarning,
            stacklevel=2,
        )
    if not infer and not records:
        raise DataInsufficient(
            "abnormal_returns: no usable event.",
            recovery_hint="Check the dates and the event window against "
            "the return series; result.skipped reasons are in diagnostics.",
            diagnostics={"skipped": skipped},
        )
    if infer and len(records) < 2:
        raise DataInsufficient(
            f"abnormal_returns: {len(records)} usable event(s); the tests "
            "compare events and need at least two.",
            recovery_hint="Check the dates, the windows and min_obs. "
            "inference=False returns the abnormal returns and CARs alone, "
            "for one event or more, and with model='market_adjusted' "
            "needs no estimation window.",
            diagnostics={"skipped": skipped},
        )
    table = pd.DataFrame(records).set_index("event")
    ar_frame = pd.DataFrame(ar_columns)
    ar_frame.index.name = "day"
    raw_frame = pd.DataFrame(raw_columns)
    raw_frame.index.name = "day"
    aar = pd.DataFrame(
        {"aar": ar_frame.mean(axis=1), "n": ar_frame.notna().sum(axis=1)}
    )
    aar["caar"] = aar["aar"].cumsum()
    if not infer:
        empty = pd.DataFrame({"statistic": [], "pvalue": []}, dtype=float)
        empty.index.name = "test"
        return AbnormalReturnsResult(
            events=table,
            ar=ar_frame,
            aar=aar,
            tests=empty,
            model=model_key,
            event_window=(ev_lo, ev_hi),
            estimation_window=(es_lo, es_hi),
            mean_correlation=float("nan"),
            correlation=corr_key,
            skipped=skipped,
            alpha=alpha,
            inference=False,
            returns=raw_frame,
        )
    r_bar = _mean_correlation(
        residuals, table[id].tolist() if corr_key == "calendar" else None
    )
    if not np.isfinite(r_bar):
        # every usable event belongs to one security
        warnings.warn(
            "abnormal_returns: all events belong to one security, so there "
            "is no cross-correlation between securities to adjust for; "
            "adj_patell and kp equal patell and bmp (mean_correlation = 0).",
            UserWarning,
            stacklevel=2,
        )
        r_bar = 0.0
    tests = _car_tests(
        table["scar"].to_numpy(dtype=float),
        table["dof"].to_numpy(dtype=float),
        table["car"].to_numpy(dtype=float),
        r_bar,
    )
    scar = table["scar"].to_numpy(dtype=float)
    tests.loc["grank"] = _grank(standardised, scar)
    tests.loc["sign_rank"] = _sign_rank(table["car"].to_numpy(dtype=float))
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
        returns=raw_frame,
    )
