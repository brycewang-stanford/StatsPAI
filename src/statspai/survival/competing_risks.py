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

from .._result_serialize import ResultProtocolMixin

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
class CumIncResult(ResultProtocolMixin):
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
class FineGrayResult(ResultProtocolMixin):
    """Fine-Gray proportional subdistribution hazards model result.

    Attributes
    ----------
    params : np.ndarray
        Estimated coefficients (log subdistribution hazard ratios).
    bse : np.ndarray
        Standard errors (Fine-Gray sandwich by default; see ``vce``).
    covariates : list of str
        Covariate names aligned with ``params``.
    cause : int
        The cause of interest whose subdistribution was modelled.
    n_obs, n_events : int
        Sample size and number of cause-of-interest events.
    vcov : np.ndarray
        Covariance matrix behind ``bse``.
    vce : str
        ``"robust"`` (Fine-Gray sandwich), ``"robust+n/(n-1)"`` (Stata
        ``stcrreg`` scaling) or ``"model"`` (inverse information).

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
    vcov: Optional[np.ndarray] = None
    vce: str = "robust"

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
    variance: str = "delta",
    conf_type: str = "linear",
) -> pd.DataFrame:
    """Aalen-Johansen CIF for each cause with its pointwise variance.

    ``variance="delta"`` is the Marubini-Valsecchi (1995) delta-method
    variance -- the estimator Stata's ``stcompet`` computes::

        Var F_k(t) = sum_{t_j<=t} (F_k(t) - F_k(t_j))^2 d_j / (n_j (n_j - d_j))
                   + sum_{t_j<=t} S(t_{j-1})^2 d_kj (n_j - d_kj) / n_j^3
                   - 2 sum_{t_j<=t} (F_k(t) - F_k(t_j)) S(t_{j-1}) d_kj / n_j^2

    ``variance="gray"`` is the asymptotic variance R ``cmprsk::cuminc``
    reports (Gray's estimator, with the hypergeometric tie factor
    ``1 - (d - 1)/(n - 1)`` on every increment).
    """
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    event = event[order]
    z = stats.norm.ppf(1 - alpha / 2)

    event_times = np.sort(np.unique(time[event != 0]))
    n_times = len(event_times)

    # Per-event-time risk-set quantities (time is sorted ascending).
    n_risk = (len(time) - np.searchsorted(time, event_times, side="left")).astype(float)
    ev_idx = np.searchsorted(event_times, time[event != 0])
    d_all = np.bincount(ev_idx, minlength=n_times).astype(float)
    # Overall KM survival just before (s_left) and at (s_right) each time.
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_risk > 0, 1.0 - d_all / n_risk, 1.0)
    s_right = np.cumprod(factors)
    s_left = np.concatenate([[1.0], s_right[:-1]])

    rows = []
    for cause in causes:
        k_idx = np.searchsorted(event_times, time[event == cause])
        d_k = np.bincount(k_idx, minlength=n_times).astype(float)
        # Increment dF_k(t_i) = S(t_{i-1}) * d_{ki} / n_i.
        with np.errstate(divide="ignore", invalid="ignore"):
            inc = np.where(n_risk > 0, s_left * d_k / n_risk, 0.0)
        cif = np.cumsum(inc)

        if variance == "delta":
            var = _cif_var_delta(cif, s_left, n_risk, d_all, d_k)
        elif variance == "gray":
            var = _cif_var_gray(cif, s_left, s_right, n_risk, d_all, d_k)
        else:  # pragma: no cover - validated by the caller
            raise ValueError(f"unknown variance {variance!r}")

        se = np.sqrt(np.clip(var, 0.0, None))
        if conf_type == "linear":
            ci_lo = np.clip(cif - z * se, 0.0, 1.0)
            ci_hi = np.clip(cif + z * se, 0.0, 1.0)
        elif conf_type == "log-log":
            # F^exp(+-z se / (F log F)); stcompet's default bounds.
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = z * se / (cif * np.log(cif))
                ci_lo = np.where((cif > 0) & (cif < 1), cif ** np.exp(-expo), np.nan)
                ci_hi = np.where((cif > 0) & (cif < 1), cif ** np.exp(expo), np.nan)
        else:  # pragma: no cover - validated by the caller
            raise ValueError(f"unknown conf_type {conf_type!r}")
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


def _cif_var_delta(
    cif: np.ndarray,
    s_left: np.ndarray,
    n_risk: np.ndarray,
    d_all: np.ndarray,
    d_k: np.ndarray,
) -> np.ndarray:
    """Marubini-Valsecchi delta-method variance at every event time."""
    n_times = len(cif)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.where(
            (n_risk > d_all) & (n_risk > 0), d_all / (n_risk * (n_risk - d_all)), 0.0
        )
        a2 = np.where(n_risk > 0, s_left**2 * d_k * (n_risk - d_k) / n_risk**3, 0.0)
        a3 = np.where(n_risk > 0, s_left * d_k / n_risk**2, 0.0)
    var = np.empty(n_times, dtype=float)
    c2 = np.cumsum(a2)
    for j in range(n_times):
        diff = cif[j] - cif[: j + 1]  # F(t) - F(t_j), zero at j itself
        var[j] = (
            np.sum(diff**2 * a1[: j + 1]) + c2[j] - 2.0 * np.sum(diff * a3[: j + 1])
        )
    return var


def _cif_var_gray(
    cif: np.ndarray,
    s_left: np.ndarray,
    s_right: np.ndarray,
    n_risk: np.ndarray,
    d_all: np.ndarray,
    d_k: np.ndarray,
) -> np.ndarray:
    """Gray's asymptotic CIF variance (R ``cmprsk::cuminc``).

    With ``S`` the all-cause KM, the variance at ``t`` is
    ``v1(t) + F(t)^2 v3(t) - 2 F(t) v2(t)``, where each event time ``s <= t``
    adds ``g^2 c``, ``g h c`` and ``h^2 c`` to ``v1, v2, v3``, with
    ``c = S(s-)^2 (1 - (d-1)/(n-1)) d / n^2``, ``h = 1/S(s)`` and
    ``g = 1 + F(s) h`` for failures from the cause of interest and
    ``g = F(s) h`` for failures from the other causes.
    """
    n_times = len(cif)
    d_o = d_all - d_k
    v1 = np.zeros(n_times)
    v2 = np.zeros(n_times)
    v3 = np.zeros(n_times)
    acc1 = acc2 = acc3 = 0.0
    for j in range(n_times):
        n = n_risk[j]
        h = 1.0 / s_right[j] if s_right[j] > 0 else 0.0
        if d_o[j] > 0 and s_right[j] > 0:
            tie = 1.0 - (d_o[j] - 1.0) / (n - 1.0) if d_o[j] > 1 else 1.0
            c = s_left[j] ** 2 * tie * d_o[j] / n**2
            g = cif[j] * h
            acc1 += g * g * c
            acc2 += h * g * c
            acc3 += h * h * c
        if d_k[j] > 0:
            tie = 1.0 - (d_k[j] - 1.0) / (n - 1.0) if d_k[j] > 1 else 1.0
            c = s_left[j] ** 2 * tie * d_k[j] / n**2
            g = 1.0 + h * cif[j]
            acc1 += g * g * c
            acc2 += h * g * c
            acc3 += h * h * c
        v1[j], v2[j], v3[j] = acc1, acc2, acc3
    return v1 + cif**2 * v3 - 2.0 * cif * v2


def _gray_test(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
    cause: int,
    rho: float = 0.0,
) -> Dict[str, float]:
    """Gray's (1988) K-sample test for equality of one cause's CIF.

    The score for group ``k`` integrates ``(1 - F(t-))^rho`` against the
    difference between the group's subdistribution hazard and the pooled
    one, with the subdistribution risk set estimated by
    ``R_k(t) = n_k(t) (1 - F_k(t-)) / S_k(t-)`` (``S_k`` the group's
    all-cause KM, ``F_k`` its cumulative incidence). The covariance is
    Gray's asymptotic estimator, which propagates the variability of the
    group-wise Aalen-Johansen estimates (so it is not the hypergeometric
    log-rank variance). Numerically identical to R ``cmprsk::cuminc``'s
    ``Tests`` (unstratified). Returns a chi-square on ``K - 1`` d.o.f.
    """
    groups = np.unique(group)
    k = len(groups)
    if k < 2 or not np.any(event == cause):
        return {
            "statistic": float("nan"),
            "df": k - 1,
            "p_value": float("nan"),
        }
    gidx = np.searchsorted(groups, group)
    code = np.where(event == cause, 1, np.where(event == 0, 0, 2))
    utimes = np.sort(np.unique(time))
    tpos = np.searchsorted(utimes, time)
    m = len(utimes)
    # d[c, t, g]: number censored (c=0), failing from the cause (1), or
    # from a competing cause (2) at each unique time, by group.
    d = np.zeros((3, m, k))
    np.add.at(d, (code, tpos, gidx), 1.0)
    # Risk set at each time by group: n minus everyone who left earlier.
    left_before = np.cumsum(d.sum(axis=0), axis=0) - d.sum(axis=0)
    rs_all = np.bincount(gidx, minlength=k).astype(float)[None, :] - left_before

    k1 = k - 1
    score = np.zeros(k1)
    vmat = np.zeros((k1, k1))
    c = np.zeros((k, k))
    v2 = np.zeros((k1, k))
    v3 = np.zeros(k)
    f1m = np.zeros(k)  # group CIF, left limit
    skmm = np.ones(k)  # group all-cause KM, left limit
    fm = 0.0  # pooled CIF, left limit
    for t in range(m):
        d1 = d[1, t]
        d2 = d[2, t]
        nd1 = d1.sum()
        nd2 = d2.sum()
        if nd1 == 0 and nd2 == 0:
            continue
        rs = rs_all[t]
        live = rs > 0
        skm = skmm.copy()
        f1 = f1m.copy()
        skm[live] = skmm[live] * (rs[live] - d1[live] - d2[live]) / rs[live]
        f1[live] = f1m[live] + skmm[live] * d1[live] / rs[live]
        hk = np.where(live, rs / np.where(live, skmm, 1.0), 0.0)
        rk = np.where(live, rs * (1.0 - f1m) / np.where(live, skmm, 1.0), 0.0)
        tr = hk.sum()
        tq = rk.sum()
        f = fm + nd1 / tr
        fb = (1.0 - fm) ** rho
        a = -fb * np.outer(hk, hk) / tr
        a[np.diag_indices(k)] = fb * hk * (1.0 - hk / tr)
        a[~live, :] = 0.0
        a[:, ~live] = 0.0
        c += a * nd1 / (tr * (1.0 - fm))
        score += np.where(live, fb * (d1 - nd1 * rk / tq), 0.0)[:k1]
        if nd1 > 0:
            for g in np.where(live)[0]:
                t4 = 1.0 - (1.0 - f) / skm[g] if skm[g] > 0 else 1.0
                t5 = 1.0 - (nd1 - 1.0) / (tr * skmm[g] - 1.0) if nd1 > 1 else 1.0
                t3 = t5 * skmm[g] * nd1 / (tr * rs[g])
                v3[g] += t4 * t4 * t3
                col = a[:k1, g] - t4 * c[:k1, g]
                v2[:, g] += col * t4 * t3
                vmat += np.outer(col, col) * t3
        if nd2 > 0:
            for g in np.where(live & (d2 > 0) & (skm > 0))[0]:
                t4 = (1.0 - f) / skm[g]
                t5 = 1.0 - (d2[g] - 1.0) / (rs[g] - 1.0) if d2[g] > 1 else 1.0
                t3 = t5 * skmm[g] ** 2 * d2[g] / rs[g] ** 2
                v3[g] += t4 * t4 * t3
                col = t4 * c[:k1, g]
                v2[:, g] -= col * t4 * t3
                vmat += np.outer(col, col) * t3
        fm = f
        f1m = f1
        skmm = skm
    vmat += (c[:k1] * v3[None, :]) @ c[:k1].T + c[:k1] @ v2.T + v2 @ c[:k1].T

    try:
        stat = float(score @ np.linalg.solve(vmat, score))
    except np.linalg.LinAlgError:
        stat = float(score @ np.linalg.pinv(vmat) @ score)
    df = k1
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
    variance: str = "delta",
    conf_type: str = "linear",
    rho: float = 0.0,
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
    variance : {"delta", "gray"}, default "delta"
        Pointwise variance of the CIF. ``"delta"`` is the Marubini-Valsecchi
        delta-method estimator (the one Stata's ``stcompet`` reports);
        ``"gray"`` is Gray's asymptotic variance, the ``var`` component of
        R ``cmprsk::cuminc``.
    conf_type : {"linear", "log-log"}, default "linear"
        ``"linear"`` is ``F +/- z se`` clipped to [0, 1]; ``"log-log"`` is
        ``F^exp(+/- z se / (F log F))``, stcompet's bounds.
    rho : float, default 0
        Power of the ``(1 - F(t-))^rho`` weight in Gray's test (cmprsk's
        ``rho``).

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
    if variance not in ("delta", "gray"):
        raise ValueError("variance must be 'delta' or 'gray'")
    if conf_type not in ("linear", "log-log"):
        raise ValueError("conf_type must be 'linear' or 'log-log'")
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
        tab = _aalen_johansen(time, ev, causes, alpha, variance, conf_type)
        tab.insert(0, "group", "all")
        tables.append(tab)
    else:
        gvals = np.asarray(data[group])
        for g in pd.unique(gvals):
            mask = gvals == g
            tab = _aalen_johansen(
                time[mask], ev[mask], causes, alpha, variance, conf_type
            )
            tab.insert(0, "group", g)
            tables.append(tab)
        gray = {c: _gray_test(time, ev, gvals, c, rho) for c in causes}

    cif_table = pd.concat(tables, ignore_index=True)
    return CumIncResult(cif_table=cif_table, causes=causes, gray_test=gray, alpha=alpha)


# --------------------------------------------------------------------------- #
#  Public: finegray
# --------------------------------------------------------------------------- #
def _censoring_km_left(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """KM of the censoring distribution evaluated just before each ``time``.

    Returns ``Ĝ(T_i-)`` for every subject: the product over censoring times
    strictly earlier than ``T_i`` of ``1 - c_u / n_u``. Left limits are what
    Fine & Gray's weights use (and what R ``cmprsk::crr`` / Stata
    ``stcrreg`` evaluate), so a censoring tied with an event does not
    down-weight that event's own risk set.
    """
    utimes = np.sort(np.unique(time))
    n_risk = len(time) - np.searchsorted(np.sort(time), utimes, side="left")
    c_cnt = np.bincount(
        np.searchsorted(utimes, time[event == 0]), minlength=len(utimes)
    ).astype(float)
    g_right = np.cumprod(1.0 - c_cnt / n_risk)
    g_left = np.concatenate([[1.0], g_right[:-1]])
    out: np.ndarray = g_left[np.searchsorted(utimes, time)]
    return out


def finegray(
    data: pd.DataFrame,
    duration: str,
    event: str,
    x: Sequence[str],
    cause: int = 1,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-10,
    vce: str = "robust",
    small_sample: bool = False,
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
        Newton-Raphson controls (``tol`` is on the largest coefficient step).
    vce : {"robust", "model"}, default "robust"
        ``"robust"`` is Fine & Gray's sandwich variance, whose score
        residuals include the term for estimating the censoring
        distribution: the ``var`` of R ``cmprsk::crr``. ``"model"`` is the
        inverse information of the weighted partial likelihood
        (``crr``'s ``invinf``), which ignores both the weighting and the
        estimation of Ĝ and is not a valid variance for this model.
    small_sample : bool, default False
        Multiply the robust variance by ``n / (n - 1)``, as Stata's
        ``stcrreg`` does.

    Returns
    -------
    FineGrayResult
        With ``.shr``, ``.tidy()``, ``.summary()``.

    Notes
    -----
    Subjects who fail from a competing cause are retained in the risk set with
    time-decaying inverse-probability-of-censoring weights
    ``w_i(t) = Ĝ(t-) / Ĝ(T_i-)`` (Ĝ = KM estimate of the censoring survival,
    evaluated at left limits). The weighted partial likelihood is maximised
    by Newton-Raphson with the Breslow tie approximation. Coefficients,
    both variances and the log pseudo-likelihood match R ``cmprsk::crr``;
    with ``small_sample=True`` the standard errors match Stata ``stcrreg``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> t1 = rng.exponential(scale=np.exp(-0.5 * x))   # cause of interest
    >>> t2 = rng.exponential(scale=1.5, size=n)        # competing cause
    >>> cens = rng.exponential(scale=2.0, size=n)
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
    if vce not in ("robust", "model"):
        raise ValueError("vce must be 'robust' or 'model'")
    cols = [duration, event] + list(x)
    data = data.dropna(subset=cols)
    time = np.asarray(data[duration], dtype=float)
    ev = np.asarray(data[event], dtype=int)
    X = np.asarray(data[list(x)], dtype=float)
    n, p = X.shape
    if (ev == cause).sum() == 0:
        raise ValueError(f"No events for cause={cause}.")

    g_self = _censoring_km_left(time, ev)  # Ĝ(T_i-)
    competing = (ev != 0) & (ev != cause)
    is_event = ev == cause

    # Distinct cause-of-interest event times, their multiplicities, and the
    # weight matrix W[m, i] = w_i(t_m) of every subject in each risk set.
    event_times = np.sort(np.unique(time[is_event]))
    m_of_event = np.searchsorted(event_times, time[is_event])
    d_counts = np.bincount(m_of_event, minlength=len(event_times)).astype(float)
    event_x_sum = np.zeros((len(event_times), p))
    np.add.at(event_x_sum, m_of_event, X[is_event])
    # Ĝ(t_m-) is Ĝ(T_j-) of any subject j failing at t_m.
    g_evt = np.empty(len(event_times))
    g_evt[m_of_event] = g_self[is_event]
    at_risk = time[None, :] >= event_times[:, None]
    comp_before = competing[None, :] & ~at_risk
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(g_self[None, :] > 0, g_evt[:, None] / g_self[None, :], 0.0)
    W = np.where(at_risk, 1.0, np.where(comp_before, ratio, 0.0))

    def _risk_sums(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ew = W * np.exp(X @ b)[None, :]
        s0 = ew.sum(axis=1)
        s1 = ew @ X
        return ew, s0, s1

    def _ll_grad_hess(
        b: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        ew, s0, s1 = _risk_sums(b)
        xbar = s1 / s0[:, None]
        ll = float(np.sum(event_x_sum @ b - d_counts * np.log(s0)))
        grad = (event_x_sum - d_counts[:, None] * xbar).sum(axis=0)
        s2 = np.einsum("mi,ij,ik->jk", ew * (d_counts / s0)[:, None], X, X)
        hess = -(s2 - (xbar * d_counts[:, None]).T @ xbar)
        return ll, grad, hess

    beta = np.zeros(p, dtype=float)
    converged = False
    for _ in range(max_iter):
        ll, grad, hess = _ll_grad_hess(beta)
        try:
            step = np.asarray(np.linalg.solve(hess, grad), dtype=float)
        except np.linalg.LinAlgError:
            step = np.asarray(np.linalg.pinv(hess) @ grad, dtype=float)
        beta = beta - step.reshape(p)
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        import warnings

        warnings.warn(
            f"finegray: Newton-Raphson did not converge in {max_iter} "
            "iterations; estimates may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    ll, _, hess = _ll_grad_hess(beta)
    info = -hess
    try:
        inv_info = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        inv_info = np.linalg.pinv(info)

    if vce == "model":
        cov = inv_info
    else:
        resid = _finegray_score_residuals(
            beta,
            X,
            time,
            ev,
            is_event,
            competing,
            g_self,
            event_times,
            d_counts,
            W,
        )
        cov = inv_info @ (resid.T @ resid) @ inv_info
        if small_sample:
            cov = cov * n / (n - 1.0)
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return FineGrayResult(
        params=beta,
        bse=bse,
        covariates=list(x),
        cause=int(cause),
        n_obs=n,
        n_events=int(is_event.sum()),
        loglik=float(ll),
        alpha=alpha,
        vcov=cov,
        vce=vce + ("+n/(n-1)" if (vce == "robust" and small_sample) else ""),
    )


def _finegray_score_residuals(
    beta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    ev: np.ndarray,
    is_event: np.ndarray,
    competing: np.ndarray,
    g_self: np.ndarray,
    event_times: np.ndarray,
    d_counts: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Per-subject influence terms ``eta_i + psi_i`` of Fine & Gray (1999).

    ``eta_i`` is the weighted-martingale score residual
    ``int (X_i - xbar(t)) w_i(t) dM_i(t)`` with the Breslow increment
    ``dLambda(t_m) = d_m / S0(t_m)``. ``psi_i = int q(u)/pi(u) dM^c_i(u)``
    carries the variability from estimating Ĝ, where ``pi(u)`` is the number
    at risk at ``u`` and
    ``q(u) = sum_{t_m >= u} d_m / S0(t_m) sum_{j competing, T_j < u}
    w_j(t_m) exp(X_j b) (X_j - xbar(t_m))``.
    """
    n, p = X.shape
    risk = np.exp(X @ beta)
    ew = W * risk[None, :]
    s0 = ew.sum(axis=1)
    xbar = (ew @ X) / s0[:, None]
    dlam = d_counts / s0

    # eta: dN part minus the compensator.
    eta = np.zeros((n, p))
    m_of_event = np.searchsorted(event_times, time[is_event])
    eta[is_event] = X[is_event] - xbar[m_of_event]
    comp_w = ew * dlam[:, None]  # (M, n)
    eta -= X * comp_w.sum(axis=0)[:, None] - comp_w.T @ xbar

    # psi: censoring-martingale part, evaluated at distinct censoring times.
    cens = ev == 0
    if not np.any(cens):
        return eta
    c_times = np.sort(np.unique(time[cens]))
    c_cnt = np.bincount(
        np.searchsorted(c_times, time[cens]), minlength=len(c_times)
    ).astype(float)
    pi_u = (n - np.searchsorted(np.sort(time), c_times, side="left")).astype(float)
    q = np.zeros((len(c_times), p))
    comp_idx = np.where(competing)[0]
    if len(comp_idx):
        comp_idx = comp_idx[np.argsort(time[comp_idx], kind="mergesort")]
        # a[m, j]: weight of competing subject j in event time m's
        # compensator (non-zero only when T_j < t_m).
        a = ew[:, comp_idx] * dlam[:, None]  # (M, J)
        xj = X[comp_idx]
        tj = time[comp_idx]
        # Sweep u upwards: keep column sums over the event times t_m >= u.
        col = a.sum(axis=0)  # (J,)
        colx = a.T @ xbar  # (J, p)
        m0 = 0
        for r, u in enumerate(c_times):
            while m0 < len(event_times) and event_times[m0] < u:
                col -= a[m0]
                colx -= np.outer(a[m0], xbar[m0])
                m0 += 1
            j0 = int(np.searchsorted(tj, u, side="left"))  # T_j < u
            if j0 == 0 or m0 == len(event_times):
                continue
            q[r] = col[:j0] @ xj[:j0] - colx[:j0].sum(axis=0)
    step = q * (c_cnt / pi_u**2)[:, None]
    cum = np.cumsum(step, axis=0)
    # sum over censoring times u <= T_i
    pos = np.searchsorted(c_times, time, side="right") - 1
    psi = np.where(pos[:, None] >= 0, -cum[np.clip(pos, 0, None)], 0.0)
    ci = np.searchsorted(c_times, time[cens])
    psi[cens] += q[ci] / pi_u[ci][:, None]
    out: np.ndarray = eta + psi
    return out
