"""Competing-risks survival analysis.

In a competing-risks setting a subject can fail from one of several mutually
exclusive causes (e.g. death from the disease of interest vs. death from
other causes). Ordinary Kaplan-Meier applied to one cause over-states that
cause's risk because it treats competing events as (non-informative)
censoring. The right descriptive quantity is the **cumulative incidence
function** (CIF) and the right regression tool for the effect of covariates
on a cause-specific cumulative incidence is the **Fine-Gray subdistribution
hazards model**.

Implemented
-----------
- :func:`cuminc` — Aalen-Johansen cumulative incidence functions for every
  cause, optionally by group, with delta-method variances/CIs and Gray's
  (1988) K-sample test.
- :func:`finegray` — Fine & Gray (1999) proportional subdistribution
  hazards regression, returning subdistribution hazard ratios.

Event coding
------------
``event`` is an integer column: ``0`` = right-censored, and ``1, 2, ...`` =
the competing causes. This matches R's ``cmprsk`` and ``survival`` packages.

References
----------
Aalen, O. (1978). "Nonparametric estimation of partial transition
probabilities in multiple decrement models." *Annals of Statistics*, 6(3),
534-545.

Gray, R.J. (1988). "A class of K-sample tests for comparing the cumulative
incidence of a competing risk." *Annals of Statistics*, 16(3), 1141-1154.
[@gray1988class]

Fine, J.P. & Gray, R.J. (1999). "A proportional hazards model for the
subdistribution of a competing risk." *Journal of the American Statistical
Association*, 94(446), 496-509. [@fine1999proportional]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "CumIncResult",
    "FineGrayResult",
    "cuminc",
    "finegray",
]


# --------------------------------------------------------------------------- #
#  Result objects
# --------------------------------------------------------------------------- #
@dataclass
class CumIncResult:
    """Cumulative incidence functions for competing risks.

    Attributes
    ----------
    cif_table : pd.DataFrame
        Long table with columns ``group`` (``"all"`` when ungrouped),
        ``cause``, ``time``, ``cif``, ``se``, ``ci_lower``, ``ci_upper``.
    causes : list
        The competing-cause labels (excluding the ``0`` censoring code).
    gray_test : dict or None
        ``cause -> {"statistic", "df", "p_value"}`` from Gray's K-sample
        test, or ``None`` when ``group`` was not supplied.
    alpha : float
        Significance level used for the confidence bands.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> t1 = rng.exponential(scale=1.0, size=n)
    >>> t2 = rng.exponential(scale=1.5, size=n)
    >>> cens = rng.exponential(scale=2.0, size=n)
    >>> time = np.minimum(np.minimum(t1, t2), cens)
    >>> status = np.where((t1 <= t2) & (t1 <= cens), 1,
    ...                   np.where((t2 < t1) & (t2 <= cens), 2, 0))
    >>> df = pd.DataFrame({"time": time, "status": status})
    >>> ci = sp.cuminc(df, duration="time", event="status")
    >>> isinstance(ci, sp.CumIncResult)
    True
    >>> ci.causes
    [1, 2]
    >>> at1 = ci.cif_at(1.0, cause=1)  # CIF for cause 1 at time t=1
    >>> bool((at1["cif"] >= 0).all())
    True
    """

    cif_table: pd.DataFrame
    causes: List[int]
    gray_test: Optional[Dict[int, Dict[str, float]]] = None
    alpha: float = 0.05

    def cif_at(self, time: float, cause: Optional[int] = None) -> pd.DataFrame:
        """Cumulative incidence (last value at or before ``time``)."""
        rows = []
        sub = self.cif_table
        causes = [cause] if cause is not None else self.causes
        for g, gdf in sub.groupby("group", sort=False):
            for c in causes:
                cdf = gdf[(gdf["cause"] == c) & (gdf["time"] <= time)]
                if len(cdf):
                    last = cdf.iloc[-1]
                    rows.append(
                        {
                            "group": g,
                            "cause": c,
                            "time": time,
                            "cif": last["cif"],
                            "se": last["se"],
                            "ci_lower": last["ci_lower"],
                            "ci_upper": last["ci_upper"],
                        }
                    )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        out = ["=" * 72, "Cumulative Incidence (Aalen-Johansen)", "=" * 72]
        for g, gdf in self.cif_table.groupby("group", sort=False):
            label = "Overall" if g == "all" else f"Group: {g}"
            out.append(f"\n{label}")
            for c in self.causes:
                cdf = gdf[gdf["cause"] == c]
                if not len(cdf):
                    continue
                last = cdf.iloc[-1]
                out.append(
                    f"  Cause {c}: CIF({last['time']:.4g}) = "
                    f"{last['cif']:.4f}  "
                    f"[{last['ci_lower']:.4f}, {last['ci_upper']:.4f}]"
                )
        if self.gray_test is not None:
            out.append("\nGray's K-sample test (equality of CIFs):")
            for c, res in self.gray_test.items():
                out.append(
                    f"  Cause {c}: chi2 = {res['statistic']:.4f}, "
                    f"df = {int(res['df'])}, p = {res['p_value']:.4g}"
                )
        out.append("\n" + "=" * 72)
        return "\n".join(out)

    def plot(self, cause: Optional[int] = None, ax: Any = None, **kwargs: Any) -> Any:
        """Step-plot the cumulative incidence function(s)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))
        causes = [cause] if cause is not None else self.causes
        for g, gdf in self.cif_table.groupby("group", sort=False):
            for c in causes:
                cdf = gdf[gdf["cause"] == c]
                if not len(cdf):
                    continue
                lbl = f"cause {c}" if g == "all" else f"{g}, cause {c}"
                ax.step(cdf["time"], cdf["cif"], where="post", label=lbl)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative incidence")
        ax.set_ylim(0, 1.0)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        n_groups = self.cif_table["group"].nunique()
        return f"<CumIncResult: {len(self.causes)} causes, " f"{n_groups} group(s)>"


@dataclass
class FineGrayResult:
    """Fine-Gray proportional subdistribution hazards model result.

    Attributes
    ----------
    params : np.ndarray
        Estimated coefficients (log subdistribution hazard ratios).
    bse : np.ndarray
        Standard errors (model-based, from the inverse information).
    covariates : list of str
        Covariate names aligned with ``params``.
    cause : int
        The cause of interest whose subdistribution was modelled.
    n_obs, n_events : int
        Sample size and number of cause-of-interest events.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> t1 = rng.exponential(scale=np.exp(-0.5 * x))
    >>> t2 = rng.exponential(scale=1.5)
    >>> cens = rng.exponential(scale=2.0)
    >>> time = np.minimum(np.minimum(t1, t2), cens)
    >>> status = np.where((t1 <= t2) & (t1 <= cens), 1,
    ...                   np.where((t2 < t1) & (t2 <= cens), 2, 0))
    >>> df = pd.DataFrame({"time": time, "status": status, "x": x})
    >>> res = sp.finegray(df, duration="time", event="status", x=["x"])
    >>> isinstance(res, sp.FineGrayResult)
    True
    >>> res.covariates
    ['x']
    >>> res.shr.shape   # one subdistribution hazard ratio per covariate
    (1,)
    """

    params: np.ndarray
    bse: np.ndarray
    covariates: List[str]
    cause: int
    n_obs: int
    n_events: int
    loglik: float
    alpha: float = 0.05

    @property
    def shr(self) -> np.ndarray:
        """Subdistribution hazard ratios, exp(coef)."""
        out: np.ndarray = np.exp(self.params)
        return out

    @property
    def zvalues(self) -> np.ndarray:
        out: np.ndarray = self.params / self.bse
        return out

    @property
    def pvalues(self) -> np.ndarray:
        out: np.ndarray = 2 * stats.norm.sf(np.abs(self.zvalues))
        return out

    @property
    def conf_int(self) -> np.ndarray:
        z = stats.norm.ppf(1 - self.alpha / 2)
        lo = self.params - z * self.bse
        hi = self.params + z * self.bse
        out: np.ndarray = np.column_stack([lo, hi])
        return out

    def tidy(self) -> pd.DataFrame:
        ci = self.conf_int
        return pd.DataFrame(
            {
                "term": self.covariates,
                "coef": self.params,
                "shr": self.shr,
                "std_err": self.bse,
                "z": self.zvalues,
                "p_value": self.pvalues,
                "shr_lower": np.exp(ci[:, 0]),
                "shr_upper": np.exp(ci[:, 1]),
            }
        )

    def summary(self) -> str:
        out = ["=" * 72, "Fine-Gray Subdistribution Hazards Model", "=" * 72]
        out.append(f"Cause of interest : {self.cause}")
        out.append(f"N                 : {self.n_obs}")
        out.append(f"Cause-{self.cause} events     : {self.n_events}")
        out.append("-" * 72)
        out.append(
            f"{'term':<16}{'sHR':>10}{'coef':>10}{'se':>10}" f"{'z':>9}{'p':>10}"
        )
        td = self.tidy()
        for _, r in td.iterrows():
            out.append(
                f"{r['term']:<16}{r['shr']:>10.4f}{r['coef']:>10.4f}"
                f"{r['std_err']:>10.4f}{r['z']:>9.3f}{r['p_value']:>10.4g}"
            )
        out.append("=" * 72)
        return "\n".join(out)

    def __repr__(self) -> str:
        return (
            f"<FineGrayResult: cause={self.cause}, "
            f"{len(self.covariates)} covariate(s)>"
        )


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _km_survival_steps(
    time: np.ndarray, indicator: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-continuous KM survival of ``indicator`` over unique event times.

    Returns ``(event_times, S_left)`` where ``S_left[i]`` is S(t_i^-), the
    survival just *before* event time ``t_i``. Used both for the overall
    survival in the Aalen-Johansen estimator and for the censoring survival
    Ĝ in Fine-Gray weights.
    """
    times = np.sort(np.unique(time[indicator == 1]))
    s_left = np.empty(len(times), dtype=float)
    surv = 1.0
    for i, t in enumerate(times):
        n_risk = np.sum(time >= t)
        d = np.sum((time == t) & (indicator == 1))
        s_left[i] = surv
        if n_risk > 0:
            surv *= 1.0 - d / n_risk
    return times, s_left


def _aalen_johansen(
    time: np.ndarray,
    event: np.ndarray,
    causes: Sequence[int],
    alpha: float,
) -> pd.DataFrame:
    """Aalen-Johansen CIF for each cause with delta-method variance.

    Variance follows the Marubini-Valsecchi / Klein-Moeschberger (2003)
    estimator (their eq. 4.7.2): three accumulating terms capturing the
    overall-survival variation, the direct cause-specific variation, and
    their covariance.
    """
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    event = event[order]
    z = stats.norm.ppf(1 - alpha / 2)

    event_times = np.sort(np.unique(time[event != 0]))
    n_times = len(event_times)

    # Per-event-time risk-set quantities.
    n_risk = np.array([np.sum(time >= t) for t in event_times], dtype=float)
    d_all = np.array(
        [np.sum((time == t) & (event != 0)) for t in event_times], dtype=float
    )
    # Overall KM survival just before each event time, S(t_i^-).
    s_left = np.empty(n_times, dtype=float)
    surv = 1.0
    for i in range(n_times):
        s_left[i] = surv
        if n_risk[i] > 0:
            surv *= 1.0 - d_all[i] / n_risk[i]

    rows = []
    for cause in causes:
        d_k = np.array(
            [np.sum((time == t) & (event == cause)) for t in event_times],
            dtype=float,
        )
        # Increment dF_k(t_i) = S(t_{i-1}) * d_{ki} / n_i.
        with np.errstate(divide="ignore", invalid="ignore"):
            inc = np.where(n_risk > 0, s_left * d_k / n_risk, 0.0)
        cif = np.cumsum(inc)

        # Delta-method variance, accumulated at each time point.
        var = np.zeros(n_times, dtype=float)
        for j in range(n_times):
            ii = np.arange(j + 1)
            ni = n_risk[ii]
            di = d_all[ii]
            dki = d_k[ii]
            sl = s_left[ii]
            f_diff = cif[j] - np.concatenate([[0.0], cif[ii[:-1]]])
            # term I: variation propagated through overall survival
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(
                    (ni > di) & (ni > 0),
                    f_diff**2 * di / (ni * (ni - di)),
                    0.0,
                )
                # term II: direct cause-k contribution
                t2 = np.where(
                    ni > 0,
                    sl**2 * ((ni - dki) / ni) * dki / ni**2,
                    0.0,
                )
                # term III: covariance (subtracted)
                t3 = np.where(
                    ni > 0,
                    f_diff * sl * dki / ni**2,
                    0.0,
                )
            var[j] = np.sum(t1) + np.sum(t2) - 2.0 * np.sum(t3)

        se = np.sqrt(np.clip(var, 0.0, None))
        ci_lo = np.clip(cif - z * se, 0.0, 1.0)
        ci_hi = np.clip(cif + z * se, 0.0, 1.0)
        for j in range(n_times):
            rows.append(
                {
                    "cause": int(cause),
                    "time": float(event_times[j]),
                    "cif": float(cif[j]),
                    "se": float(se[j]),
                    "ci_lower": float(ci_lo[j]),
                    "ci_upper": float(ci_hi[j]),
                }
            )
    return pd.DataFrame(rows)


def _gray_test(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    cause: int,
) -> Dict[str, float]:
    """Gray's (1988) K-sample test for one cause.

    Uses the subdistribution risk set (subjects who failed from a competing
    cause remain at risk) and the rho=0 weight. Returns a chi-square statistic
    on ``K-1`` degrees of freedom.
    """
    groups = np.unique(group)
    k = len(groups)
    all_times = np.sort(np.unique(time[event == cause]))
    if len(all_times) == 0 or k < 2:
        return {
            "statistic": float("nan"),
            "df": k - 1,
            "p_value": float("nan"),
        }

    # Subdistribution "at risk": never experienced cause-of-interest yet, and
    # either still under observation OR already failed from a competing cause.
    # i.e. a subject leaves the subdistribution risk set only at its own
    # cause-of-interest event time or (for censored / not-yet-failed subjects)
    # at its censoring time.
    scores = np.zeros(k - 1, dtype=float)
    vcov = np.zeros((k - 1, k - 1), dtype=float)

    for t in all_times:
        # subdistribution risk indicator per subject at time t
        not_yet_primary = ~((event == cause) & (time < t))
        competing_before = (event != 0) & (event != cause) & (time < t)
        at_risk = not_yet_primary & ((time >= t) | competing_before)
        n_at = np.array([np.sum(at_risk & (group == g)) for g in groups], dtype=float)
        n_tot = n_at.sum()
        if n_tot <= 1:
            continue
        d_at = np.array(
            [np.sum((time == t) & (event == cause) & (group == g)) for g in groups],
            dtype=float,
        )
        d_tot = d_at.sum()
        if d_tot == 0:
            continue
        # Expected events per group under H0 and hypergeometric variance.
        exp = n_at * d_tot / n_tot
        obs_minus_exp = d_at - exp
        scores += obs_minus_exp[:-1]
        if n_tot > 1:
            var_factor = d_tot * (n_tot - d_tot) / (n_tot - 1)
        else:
            var_factor = 0.0
        for a in range(k - 1):
            for b in range(k - 1):
                delta = 1.0 if a == b else 0.0
                vcov[a, b] += var_factor * (n_at[a] / n_tot * (delta - n_at[b] / n_tot))

    try:
        stat = float(scores @ np.linalg.solve(vcov, scores))
    except np.linalg.LinAlgError:
        stat = float(scores @ np.linalg.pinv(vcov) @ scores)
    df = k - 1
    p = float(stats.chi2.sf(stat, df))
    return {"statistic": stat, "df": df, "p_value": p}


# --------------------------------------------------------------------------- #
#  Public: cuminc
# --------------------------------------------------------------------------- #
def cuminc(
    data: pd.DataFrame,
    duration: str,
    event: str,
    group: Optional[str] = None,
    alpha: float = 0.05,
) -> CumIncResult:
    """Cumulative incidence functions for competing risks (Aalen-Johansen).

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    duration : str
        Column name for the follow-up time.
    event : str
        Column name for the event indicator. ``0`` = censored;
        ``1, 2, ...`` = competing causes.
    group : str, optional
        Column name for a grouping variable. When supplied, CIFs are
        estimated per group and Gray's K-sample test is reported per cause.
    alpha : float
        Significance level for the confidence bands.

    Returns
    -------
    CumIncResult
        With ``.cif_table``, ``.gray_test``, ``.summary()``, ``.plot()``.

    Notes
    -----
    The cumulative incidence for a single cause is *not* ``1 - KM`` applied to
    that cause; treating competing events as censoring over-states the risk.
    The Aalen-Johansen estimator weights each cause-specific increment by the
    overall (all-cause) survival probability, so the CIFs of all causes plus
    the overall survival sum to one at every time.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> arm = rng.integers(0, 2, n)
    >>> t1 = rng.exponential(scale=1.0 + 0.5 * arm, size=n)   # cause 1
    >>> t2 = rng.exponential(scale=1.5, size=n)               # cause 2
    >>> cens = rng.exponential(scale=2.0, size=n)
    >>> time = np.minimum(np.minimum(t1, t2), cens)
    >>> status = np.where((t1 <= t2) & (t1 <= cens), 1,
    ...                   np.where((t2 < t1) & (t2 <= cens), 2, 0))
    >>> df = pd.DataFrame({"time": time, "status": status, "arm": arm})
    >>> ci = sp.cuminc(df, duration="time", event="status", group="arm")
    >>> isinstance(ci, sp.CumIncResult)
    True
    >>> ci.causes
    [1, 2]
    >>> ci.summary()        # doctest: +SKIP
    >>> ci.plot(cause=1)    # doctest: +SKIP
    """
    cols = [duration, event] + ([group] if group else [])
    data = data.dropna(subset=cols)
    time = np.asarray(data[duration], dtype=float)
    ev = np.asarray(data[event], dtype=int)
    causes = sorted(int(c) for c in np.unique(ev) if c != 0)
    if not causes:
        raise ValueError("No events found (all observations censored).")

    tables = []
    gray = None
    if group is None:
        tab = _aalen_johansen(time, ev, causes, alpha)
        tab.insert(0, "group", "all")
        tables.append(tab)
    else:
        gvals = np.asarray(data[group])
        for g in pd.unique(gvals):
            mask = gvals == g
            tab = _aalen_johansen(time[mask], ev[mask], causes, alpha)
            tab.insert(0, "group", g)
            tables.append(tab)
        gray = {c: _gray_test(time, ev, gvals, c) for c in causes}

    cif_table = pd.concat(tables, ignore_index=True)
    return CumIncResult(cif_table=cif_table, causes=causes, gray_test=gray, alpha=alpha)


# --------------------------------------------------------------------------- #
#  Public: finegray
# --------------------------------------------------------------------------- #
def _finegray_weights(
    time: np.ndarray, event: np.ndarray, cause: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Censoring-survival Ĝ steps for Fine-Gray IPCW weights.

    Returns the KM estimate of the censoring distribution evaluated as a
    right-continuous step function helper: ``(g_times, g_vals)`` where
    ``g_vals[i]`` = Ĝ(g_times[i]).
    """
    censor_indicator = (event == 0).astype(int)
    times = np.sort(np.unique(time))
    g_vals = np.empty(len(times), dtype=float)
    surv = 1.0
    for i, t in enumerate(times):
        n_risk = np.sum(time >= t)
        d_c = np.sum((time == t) & (censor_indicator == 1))
        if n_risk > 0:
            surv *= 1.0 - d_c / n_risk
        g_vals[i] = surv
    return times, g_vals


def _g_at(g_times: np.ndarray, g_vals: np.ndarray, t: float) -> float:
    """Right-continuous Ĝ(t): last step value at or before ``t``."""
    idx = np.searchsorted(g_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(g_vals[idx])


def finegray(
    data: pd.DataFrame,
    duration: str,
    event: str,
    x: Sequence[str],
    cause: int = 1,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> FineGrayResult:
    """Fine & Gray (1999) proportional subdistribution hazards model.

    Models the effect of covariates on the cumulative incidence of ``cause``
    through its subdistribution hazard, so coefficients exponentiate to
    **subdistribution hazard ratios** that map monotonically to the CIF
    (unlike cause-specific Cox coefficients).

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    duration : str
        Follow-up-time column.
    event : str
        Event indicator: ``0`` = censored, ``1, 2, ...`` = causes.
    x : sequence of str
        Covariate column names.
    cause : int
        Cause of interest (default ``1``).
    alpha : float
        Significance level for confidence intervals.
    max_iter, tol : int, float
        Newton-Raphson controls.

    Returns
    -------
    FineGrayResult
        With ``.shr``, ``.tidy()``, ``.summary()``.

    Notes
    -----
    Subjects who fail from a competing cause are retained in the risk set with
    time-decaying inverse-probability-of-censoring weights
    ``w_i(t) = Ĝ(t) / Ĝ(T_i)`` (Ĝ = KM estimate of the censoring survival).
    The weighted partial likelihood is maximised by Newton-Raphson with the
    Breslow tie approximation. Standard errors are model-based (inverse
    information); a fully robust sandwich variance that accounts for
    estimating Ĝ is not yet implemented.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> t1 = rng.exponential(scale=np.exp(-0.5 * x))   # cause of interest
    >>> t2 = rng.exponential(scale=1.5)                # competing cause
    >>> cens = rng.exponential(scale=2.0)
    >>> time = np.minimum(np.minimum(t1, t2), cens)
    >>> status = np.where((t1 <= t2) & (t1 <= cens), 1,
    ...                   np.where((t2 < t1) & (t2 <= cens), 2, 0))
    >>> df = pd.DataFrame({"time": time, "status": status, "x": x})
    >>> res = sp.finegray(
    ...     df, duration="time", event="status", x=["x"], cause=1
    ... )
    >>> res.cause
    1
    >>> res.tidy()["term"].tolist()
    ['x']
    >>> bool(res.shr[0] > 0)  # subdistribution hazard ratio
    True

    References
    ----------
    fine1999proportional
    """
    cols = [duration, event] + list(x)
    data = data.dropna(subset=cols)
    time = np.asarray(data[duration], dtype=float)
    ev = np.asarray(data[event], dtype=int)
    X = np.asarray(data[list(x)], dtype=float)
    n, p = X.shape
    if (ev == cause).sum() == 0:
        raise ValueError(f"No events for cause={cause}.")

    g_times, g_vals = _finegray_weights(time, ev, cause)
    g_self = np.array([_g_at(g_times, g_vals, t) for t in time])

    # Cause-of-interest event times (Breslow ties).
    event_times = np.sort(np.unique(time[ev == cause]))

    # Precompute, for each event time, the subdistribution risk weights.
    # w_i(t) = 1 if still at risk (T_i >= t); = Ĝ(t)/Ĝ(T_i) if i had a
    # competing event before t; = 0 otherwise (censored / primary-failed).
    competing = (ev != 0) & (ev != cause)
    weight_rows = []
    risk_index = []
    for t in event_times:
        g_t = _g_at(g_times, g_vals, t)
        w = np.zeros(n, dtype=float)
        at_risk = time >= t
        w[at_risk] = 1.0
        comp_before = competing & (time < t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w[comp_before] = np.where(
                g_self[comp_before] > 0,
                g_t / g_self[comp_before],
                0.0,
            )
        idx = np.where(w > 0)[0]
        weight_rows.append(w[idx])
        risk_index.append(idx)

    # Events at each event time (Breslow): sum of covariates of primary events.
    d_counts = []
    event_x_sum = []
    for t in event_times:
        ev_mask = (time == t) & (ev == cause)
        d_counts.append(int(ev_mask.sum()))
        event_x_sum.append(X[ev_mask].sum(axis=0))
    d_counts_arr = np.array(d_counts)
    event_x_sum_arr = np.array(event_x_sum)

    beta = np.zeros(p, dtype=float)

    def _ll_grad_hess(
        b: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        ll = 0.0
        grad = np.zeros(p, dtype=float)
        hess = np.zeros((p, p), dtype=float)
        for m, t in enumerate(event_times):
            idx = risk_index[m]
            w = weight_rows[m]
            xr = X[idx]
            eta = xr @ b
            ew = w * np.exp(eta)
            s0 = ew.sum()
            if s0 <= 0:
                continue
            s1 = ew @ xr
            s2 = (ew[:, None] * xr).T @ xr
            d = d_counts_arr[m]
            ll += event_x_sum_arr[m] @ b - d * np.log(s0)
            grad += event_x_sum_arr[m] - d * s1 / s0
            hess -= d * (s2 / s0 - np.outer(s1, s1) / s0**2)
        return ll, grad, hess

    ll = -np.inf
    for _ in range(max_iter):
        ll, grad, hess = _ll_grad_hess(beta)
        try:
            step = np.asarray(np.linalg.solve(hess, grad), dtype=float)
        except np.linalg.LinAlgError:
            step = np.asarray(np.linalg.pinv(hess) @ grad, dtype=float)
        step = step.reshape(p)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    _, _, hess = _ll_grad_hess(beta)
    try:
        cov = np.linalg.inv(-hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-hess)
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return FineGrayResult(
        params=beta,
        bse=bse,
        covariates=list(x),
        cause=int(cause),
        n_obs=n,
        n_events=int((ev == cause).sum()),
        loglik=float(ll),
        alpha=alpha,
    )
