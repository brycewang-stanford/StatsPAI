"""Trend adjustment and proxy-IV estimation for ``sp.xtevent``.

Both follow Stata ``xtevent`` 3.1.0 (Freyaldenhoven, Hansen, Perez Perez
and Shapiro), checked against its output on the fixtures in
``tests/reference_parity/_fixtures`` (``xtevent_extensions_Stata.json``).

Trend (``trend=``, Stata ``trend(#1)``): a linear pre-event trend through
the normalised coefficient at -1 is fitted to the coefficients at
``trend .. -2`` and removed from every event-time coefficient.
``method='gmm'`` (Stata's default) fits the slope by minimum distance with
the inverse covariance of those coefficients and adjusts the estimates and
their covariance; ``method='ols'`` drops those event-time regressors and
adds the event time itself as a regressor (binary single-adoption policies
only).

Proxy IV (``proxy=``, Stata ``proxy()`` / ``proxyiv()``; Freyaldenhoven,
Hansen and Shapiro 2019): a variable affected by the confound but not by
the policy enters as an endogenous regressor, instrumented by leads of the
policy change (whose event-time coefficients are then normalised to zero)
or by external variables; fitted by two-stage least squares with unit
effects (Stata ``xtivreg, fe``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["trend_gmm_adjust", "ols_trend_column", "proxy_normalisations", "iv_within"]


def trend_gmm_adjust(
    b: pd.Series,
    V: pd.DataFrame,
    event_names: Dict[int, str],
    trend: int,
) -> Tuple[pd.Series, pd.DataFrame, float]:
    """Remove a linear trend fitted to the coefficients at ``trend .. -2``.

    ``event_names`` maps event time to coefficient name for every estimated
    event-time coefficient (the normalised -1 is absent). Returns the
    adjusted coefficients, their covariance and the slope.
    """
    ks = sorted(event_names)
    names = [event_names[k] for k in ks]
    fit_ks = [k for k in ks if trend <= k <= -2]
    fit_names = [event_names[k] for k in fit_ks]
    d_L = b[fit_names].to_numpy(dtype=float)
    V_L = V.loc[fit_names, fit_names].to_numpy(dtype=float)
    h_L = np.array([k + 1.0 for k in fit_ks])
    W = np.linalg.inv(V_L)
    v_phi = 1.0 / float(h_L @ W @ h_L)
    lam_L = v_phi * (h_L @ W)  # phi_hat = lam_L @ d_L
    phi = float(lam_L @ d_L)
    all_names = list(b.index)
    pos = {nm: i for i, nm in enumerate(all_names)}
    H = np.zeros(len(all_names))
    for k, nm in zip(ks, names):
        H[pos[nm]] = k + 1.0
    lam = np.zeros(len(all_names))
    for nm, w in zip(fit_names, lam_L):
        lam[pos[nm]] = w
    A = np.eye(len(all_names)) - np.outer(H, lam)
    b_adj = pd.Series(A @ b.to_numpy(dtype=float), index=all_names)
    Vm = A @ V.to_numpy(dtype=float) @ A.T
    V_adj = pd.DataFrame(0.5 * (Vm + Vm.T), index=all_names, columns=all_names)
    return b_adj, V_adj, phi


def ols_trend_column(
    df: pd.DataFrame, policy: str, panel: str, time: str, lo: int, hi: int
) -> np.ndarray:
    """Event time inside the window, 0 outside it and for never-treated
    units (Stata ``_ttrend``). Requires a 0/1 policy adopted once."""
    z = pd.to_numeric(df[policy], errors="coerce")
    if not z.dropna().isin([0, 1]).all():
        raise MethodIncompatibility(
            "xtevent: trend_method='ols' needs a 0/1 policy adopted once.",
            recovery_hint="Use trend_method='gmm' (Stata's default).",
        )
    tt = df[time].to_numpy()
    first_obs = df.groupby(panel)[time].transform("min").to_numpy()
    first_on = (
        pd.Series(np.where(z.to_numpy() == 1, tt, np.nan), index=df.index)
        .groupby(df[panel])
        .transform("min")
        .to_numpy()
    )
    later_off = (
        pd.Series(
            np.where(
                (z.to_numpy() == 0) & (tt > np.nan_to_num(first_on, nan=np.inf)), 1, 0
            ),
            index=df.index,
        )
        .groupby(df[panel])
        .transform("max")
        .to_numpy()
    )
    if later_off.any():
        raise MethodIncompatibility(
            "xtevent: trend_method='ols' needs a policy that, once on, stays on.",
            recovery_hint="Use trend_method='gmm' for policies that switch off.",
        )
    if np.any(first_on == first_obs):
        raise MethodIncompatibility(
            "xtevent: trend_method='ols' cannot time the event of units treated "
            "in their first observed period.",
            recovery_hint="Drop always-treated units or use trend_method='gmm'.",
        )
    k = tt - first_on
    out = np.where((k >= lo) & (k <= hi), k, 0.0)
    return np.where(np.isnan(out), 0.0, out)


def proxy_normalisations(
    proxies: List[str],
    proxy_iv: Union[None, str, Sequence[Union[int, str]]],
    norm: int,
    lo: int,
    lead_columns: Dict[int, np.ndarray],
    data: pd.DataFrame,
) -> Tuple[List[int], List[int], List[str], Optional[int]]:
    """Leads used as instruments, extra normalised event times, external
    instruments, and the lead picked by ``'select'``."""
    selected = None
    if proxy_iv is None:
        proxy_iv = "select" if len(proxies) == 1 else list(range(1, len(proxies) + 1))
    if isinstance(proxy_iv, str) and proxy_iv == "select":
        if len(proxies) != 1:
            raise MethodIncompatibility(
                "xtevent: proxy_iv='select' needs exactly one proxy.",
                recovery_hint="Pass the leads, e.g. proxy_iv=[1, 2].",
            )
        best = -np.inf
        x = pd.to_numeric(data[proxies[0]], errors="coerce").to_numpy(dtype=float)
        for v in range(1, -lo + 1):
            f = lead_columns[-v]
            ok = np.isfinite(x) & np.isfinite(f)
            if ok.sum() < 3 or np.var(f[ok]) == 0:
                continue
            r = np.corrcoef(x[ok], f[ok])[0, 1]
            F = r**2 / (1 - r**2) * (ok.sum() - 2)
            if F > best:
                best, selected = F, v
        if selected is None:
            raise DataInsufficient("xtevent: no lead of the policy change varies.")
        proxy_iv = [selected]
    items = [proxy_iv] if isinstance(proxy_iv, (int, str)) else list(proxy_iv)
    leads = [int(v) for v in items if isinstance(v, (int, np.integer))]
    external = [str(v) for v in items if not isinstance(v, (int, np.integer))]
    ivnorm: List[int] = []
    for v in leads:
        if v < 1 or v > -lo:
            raise MethodIncompatibility(
                f"xtevent: proxy_iv lead {v} is outside the window (1..{-lo}).",
            )
        ivnorm.append(-v - 1 if -v == norm else -v)
        if ivnorm[-1] < lo - 1:
            raise MethodIncompatibility(
                f"xtevent: lead {v + 1} needed for the extra normalisation is "
                "outside the window.",
            )
    available = [k for k in range(-1, lo - 2, -1) if k != norm]
    for _ in external:
        available = [k for k in available if k not in ivnorm]
        if len(available) <= 1:
            raise MethodIncompatibility(
                "xtevent: too many instruments for the pre-event periods.",
                recovery_hint="Widen the window or use fewer instruments.",
            )
        ivnorm.append(max(available))
    if leads and len(proxies) == len(set([norm] + ivnorm)):
        ivnorm.append(min(ivnorm) - 1)
    return leads, ivnorm, external, selected


def iv_within(
    y: np.ndarray,
    endog: np.ndarray,
    exog: np.ndarray,
    instruments: np.ndarray,
    unit: np.ndarray,
    cluster: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """2SLS with unit effects swept out (Stata ``xtivreg, fe``).

    Columns are ``[endog, exog]``. The iid variance divides by
    ``N - units - K``; the clustered one is multiplied by
    ``G/(G-1) (N-1)/(N-K-1)`` (``K`` regressors, constant counted).
    """

    def within(M):
        M = np.asarray(M, dtype=float)
        frame = pd.DataFrame(M)
        return M - frame.groupby(unit).transform("mean").to_numpy()

    yw = within(y[:, None])[:, 0]
    X = within(np.column_stack([endog, exog]))
    Z = within(np.column_stack([instruments, exog]))
    n, k = X.shape
    if np.linalg.matrix_rank(Z) < Z.shape[1] or Z.shape[1] < k:
        raise DataInsufficient(
            "xtevent: the instruments are collinear with the included regressors "
            "or fewer than the proxies.",
            recovery_hint="Choose other leads (proxy_iv=) or an external instrument.",
        )
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.solve(Xh.T @ X, Xh.T @ yw)
    resid = yw - X @ beta
    bread = np.linalg.inv(Xh.T @ Xh)
    if cluster is None:
        n_units = len(np.unique(unit))
        V = (resid @ resid / (n - n_units - k)) * bread
    else:
        codes = pd.factorize(cluster)[0]
        g = int(codes.max()) + 1
        S = np.zeros((g, k))
        np.add.at(S, codes, Xh * resid[:, None])
        V = g / (g - 1) * (n - 1) / (n - k - 1) * bread @ (S.T @ S) @ bread
    return beta, V
