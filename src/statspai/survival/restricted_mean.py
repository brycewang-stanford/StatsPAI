"""Restricted mean survival time (RMST) by group, with differences and
ratios: R ``survRM2::rmst2`` without weights, and the IPT-weighted
(confounder-adjusted) version with the infinitesimal-jackknife variance of
R ``survival`` (``residuals(survfit, type = "rmst")``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._km import _codes, km_ij_rmst_influence, km_weighted_curve, resolve_km_se


class RMSTResult(ResultProtocolMixin):
    """Restricted mean survival time per group and contrasts.

    Attributes
    ----------
    table : pd.DataFrame
        One row per group: ``group``, ``rmst``, ``se``, ``ci_lower``,
        ``ci_upper``, ``rmtl`` (restricted mean time lost, ``tau - rmst``).
    contrasts : pd.DataFrame
        For two groups: the difference (second minus first level, unless
        ``reference=`` says otherwise) with a Wald interval and p-value, and
        the RMST and RMTL ratios with intervals on the log scale (the three
        rows of ``survRM2::rmst2``'s ``unadjusted.result``).
    tau : float
        Truncation time.
    se_method : str
        ``'greenwood'`` or ``'ij'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> arm = rng.integers(0, 2, n)
    >>> t = rng.exponential(np.where(arm == 1, 8.0, 5.0))
    >>> c = rng.uniform(0, 15, n)
    >>> df = pd.DataFrame({"time": np.minimum(t, c), "event": (t <= c).astype(int),
    ...                    "arm": arm})
    >>> res = sp.rmst(df, duration="time", event="event", group="arm", tau=10)
    >>> list(res.table["group"])
    ['0', '1']
    >>> bool(res.contrasts.loc[0, "estimate"] > 0)
    True
    """

    def __init__(
        self,
        table: pd.DataFrame,
        contrasts: pd.DataFrame,
        tau: float,
        se_method: str,
        alpha: float,
        weighted: bool,
    ):
        self.table = table
        self.contrasts = contrasts
        self.tau = float(tau)
        self.se_method = se_method
        self.alpha = alpha
        self.weighted = weighted

    @property
    def estimate(self) -> float:
        """The RMST difference (two groups) or the single RMST."""
        if len(self.contrasts):
            return float(self.contrasts.loc[0, "estimate"])
        return float(self.table.loc[0, "rmst"])

    @property
    def se(self) -> float:
        if len(self.contrasts):
            return float(self.contrasts.loc[0, "se"])
        return float(self.table.loc[0, "se"])

    def summary(self) -> str:
        lines = [
            f"Restricted mean survival time, tau = {self.tau:g} "
            f"({'weighted, ' if self.weighted else ''}{self.se_method} SE)",
            self.table.to_string(index=False),
        ]
        if len(self.contrasts):
            lines += ["", self.contrasts.to_string(index=False)]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _greenwood_rmst_var(
    times: np.ndarray, n_risk: np.ndarray, d: np.ndarray, surv: np.ndarray, tau: float
) -> float:
    """``sum_{t_j <= tau} A_j^2 d_j / (Y_j (Y_j - d_j))`` with ``A_j`` the
    area under ``S`` from ``t_j`` to ``tau`` (survRM2, survfit rmean)."""
    keep = times <= tau
    t = times[keep]
    s = surv[keep]
    edges = np.append(t, tau)
    seg = s * np.diff(edges)  # area of each step after t_j
    A = np.cumsum(seg[::-1])[::-1]
    Y = n_risk[keep]
    dj = d[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = np.where(Y > dj, dj / (Y * (Y - dj)), 0.0)
    return float(np.sum(A**2 * inc))


def _rmst_value(times: np.ndarray, surv: np.ndarray, tau: float) -> float:
    keep = times <= tau
    edges = np.concatenate([[0.0], times[keep], [tau]])
    heights = np.concatenate([[1.0], surv[keep]])
    return float(np.sum(heights * np.diff(edges)))


def rmst(
    data: pd.DataFrame,
    duration: str,
    event: str,
    group: Optional[str] = None,
    tau: Optional[float] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    se_method: Optional[str] = None,
    reference: Any = None,
    alpha: float = 0.05,
) -> RMSTResult:
    """Restricted mean survival time per group, with difference and ratios.

    ``RMST(tau) = int_0^tau S(t) dt`` of the Kaplan-Meier curve of each
    group. With ``weights=`` (inverse-probability-of-treatment weights) the
    curves are the IPTW-adjusted ones of ``sp.kaplan_meier(weights=)`` and
    the contrast is the confounder-adjusted RMST difference.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    duration : str
        Follow-up time column.
    event : str
        Event indicator column (1 = event, 0 = censored).
    group : str, optional
        Group / treatment column. With two levels the contrasts are reported.
    tau : float, optional
        Truncation time. Default: the smallest of the groups' largest
        observed times (survRM2's default), so that every curve is defined
        up to ``tau``. A ``tau`` beyond a group's last observed time is
        refused.
    weights : str, optional
        Column of non-negative case weights.
    cluster : str, optional
        Column of independent units for the robust variance.
    se_method : {'greenwood', 'ij'}, optional
        ``'greenwood'``: the variance of survRM2 / ``survfit``'s ``rmean``
        on the (weighted) counts. ``'ij'``: the infinitesimal jackknife,
        ``sum`` over units of the squared weighted influence (R
        ``residuals(survfit, times = tau, type = "rmst")``), which treats
        the weights as sampling weights. Not given: ``'ij'`` with a cluster
        or non-integer weights (R survfit's rule), ``'greenwood'``
        otherwise. With IPT weights estimated from a propensity model the
        IJ variance holds them fixed; for the ATE that is usually
        conservative.
    reference : optional
        The level subtracted in the difference (default: the first level).
    alpha : float, default 0.05
        Significance level of the intervals.

    Returns
    -------
    RMSTResult
        ``.table`` (per group), ``.contrasts`` (difference, RMST ratio,
        RMTL ratio), ``.estimate`` / ``.se`` (the difference).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> t = rng.exponential(np.exp(0.5 * a - 0.5 * x) * 5)
    >>> c = rng.uniform(0, 20, n)
    >>> ps = 1 / (1 + np.exp(-x))
    >>> df = pd.DataFrame({"time": np.minimum(t, c), "event": (t <= c).astype(int),
    ...                    "a": a, "w": np.where(a == 1, 1 / ps, 1 / (1 - ps))})
    >>> res = sp.rmst(df, "time", "event", group="a", tau=10, weights="w")
    >>> res.se_method
    'ij'
    """
    cols = [duration, event] + [c for c in (group, weights, cluster) if c]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise DataInsufficient(
            f"rmst: columns not found: {missing}",
            recovery_hint="Check the column names.",
        )
    from ._validate import check_duration_event, warn_missing_weights

    warn_missing_weights(data, weights, "rmst")
    df = data.dropna(subset=cols)
    check_duration_event(df[duration], df[event], "rmst")
    T_all = df[duration].to_numpy(dtype=float)
    E_all = df[event].to_numpy()
    w_all = df[weights].to_numpy(dtype=float) if weights else None
    if w_all is not None and (not np.all(np.isfinite(w_all)) or (w_all < 0).any()):
        raise MethodIncompatibility(
            "rmst: weights must be finite and non-negative.",
            recovery_hint="Check the weight column.",
        )
    method = resolve_km_se(se_method, w_all, cluster)
    if method == "xie_liu":
        raise MethodIncompatibility(
            "rmst: se_method='xie_liu' is a variance of the curve, not of its " "area.",
            recovery_hint="Use se_method='ij' or 'greenwood'.",
        )
    g_all = (
        df[group].astype(str).to_numpy()
        if group
        else np.full(len(df), "all", dtype=object)
    )
    levels = [str(v) for v in pd.unique(df[group].sort_values())] if group else ["all"]
    if reference is not None:
        ref = str(reference)
        if ref not in levels:
            raise MethodIncompatibility(
                f"rmst: reference={reference!r} is not a level of {group!r}.",
                recovery_hint=f"Use one of {levels}.",
            )
        levels = [ref] + [lv for lv in levels if lv != ref]
    max_t = {lv: float(T_all[g_all == lv].max()) for lv in levels}
    if tau is None:
        tau = min(max_t.values())
    tau = float(tau)
    too_far = [lv for lv in levels if tau > max_t[lv]]
    if too_far:
        raise DataInsufficient(
            f"rmst: tau={tau:g} is beyond the last observed time of group(s) "
            f"{too_far}; the curve is not identified there.",
            recovery_hint=f"Use tau <= {min(max_t.values()):g}.",
            diagnostics={"max_time": max_t},
        )
    clus = df[cluster].to_numpy() if cluster else None
    codes, G = _codes(clus, len(df))

    z = stats.norm.ppf(1 - alpha / 2)
    rows: List[Dict[str, Any]] = []
    infl: Dict[str, np.ndarray] = {}
    for lv in levels:
        sel = g_all == lv
        T = T_all[sel]
        E = E_all[sel]
        w = np.ones(int(sel.sum())) if w_all is None else w_all[sel]
        times, n_risk, d, _, surv = km_weighted_curve(T, E, w)
        est = _rmst_value(times, surv, tau)
        if method == "ij":
            infl[lv] = km_ij_rmst_influence(
                T, E, w, times, n_risk, d, surv, tau, codes[sel], G
            )
            var = float(infl[lv] @ infl[lv])
        else:
            var = _greenwood_rmst_var(times, n_risk, d, surv, tau)
        se = float(np.sqrt(var))
        rows.append(
            {
                "group": lv,
                "rmst": est,
                "se": se,
                "ci_lower": est - z * se,
                "ci_upper": est + z * se,
                "rmtl": tau - est,
            }
        )
    table = pd.DataFrame(rows)

    contrasts = pd.DataFrame()
    if len(levels) == 2:
        r0, r1 = table.loc[0], table.loc[1]
        diff = r1["rmst"] - r0["rmst"]
        if method == "ij":
            # clusters may span both groups: difference of the totals
            se_d = float(np.linalg.norm(infl[levels[1]] - infl[levels[0]]))
        else:
            se_d = float(np.hypot(r1["se"], r0["se"]))
        out = [
            {
                "contrast": f"RMST ({levels[1]})-({levels[0]})",
                "estimate": diff,
                "se": se_d,
                "ci_lower": diff - z * se_d,
                "ci_upper": diff + z * se_d,
                "p_value": float(2 * stats.norm.sf(abs(diff) / se_d)),
            }
        ]
        for name, a1, a0 in (
            ("RMST", r1["rmst"], r0["rmst"]),
            ("RMTL", r1["rmtl"], r0["rmtl"]),
        ):
            if method == "ij":
                sign = 1.0 if name == "RMST" else -1.0
                lr = infl[levels[1]] * sign / a1 - infl[levels[0]] * sign / a0
                se_l = float(np.linalg.norm(lr))
            else:
                se_l = float(np.sqrt((r1["se"] / a1) ** 2 + (r0["se"] / a0) ** 2))
            lr_est = float(np.log(a1 / a0))
            out.append(
                {
                    "contrast": f"{name} ({levels[1]})/({levels[0]})",
                    "estimate": float(np.exp(lr_est)),
                    "se": se_l,
                    "ci_lower": float(np.exp(lr_est - z * se_l)),
                    "ci_upper": float(np.exp(lr_est + z * se_l)),
                    "p_value": float(2 * stats.norm.sf(abs(lr_est) / se_l)),
                }
            )
        contrasts = pd.DataFrame(out)
    return RMSTResult(table, contrasts, tau, method, alpha, weighted=w_all is not None)
