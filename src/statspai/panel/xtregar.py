"""Fixed- and random-effects linear models with an AR(1) disturbance
(Stata ``xtregar``).

``y_it = a + x_it'b + nu_i + e_it`` with ``e_it = rho e_{i,t-1} + eta_it``,
on panels that may be unbalanced and unequally spaced in time.

``rho`` is estimated after removing the panel means: iterated Prais-Winsten
on the demeaned data without a constant (a period after a gap restarts the
series), with one of Stata's ``rhotype`` formulas, or ``onestep`` (Baltagi
and Wu's one-step estimator), or fixed by the user. Given ``rho`` the data
are transformed by Baltagi and Wu's ``C_i(rho)``, which handles a gap of
``d`` periods through ``rho^d``. The fixed-effects estimator then drops the
first period of each panel and runs the within regression on the
transformed data; the random-effects estimator is Baltagi and Wu's
feasible GLS, with variance components from the residuals of OLS on the
transformed data.

The conventions follow Stata 18 ``xtregar`` (its Methods and formulas, and
its output on the fixtures of ``tests/test_xtregar_stata.py``), with one
departure: the standard error of the fixed-effects constant (see
:func:`xtregar`).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core._collinear import independent_columns
from ..core.results import EconometricResults
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility

__all__ = ["xtregar"]

_RHOTYPES = ("dw", "regress", "freg", "tscorr", "theil", "nagar", "onestep")


class _Layout:
    """Sort order, panel codes and spacing of an ``(id, time)`` panel."""

    def __init__(self, ids: np.ndarray, times: np.ndarray) -> None:
        self.codes, uniq = pd.factorize(ids, sort=True)
        self.G = len(uniq)
        self.n = len(ids)
        prev = np.r_[-1, self.codes[:-1]]
        self.first = self.codes != prev
        step = np.r_[0.0, np.diff(times)]
        self.step = np.where(self.first, 1.0, step)
        # consecutive pairs (t, t - 1) inside a panel
        self.pair = ~self.first & (self.step == 1)
        self.sizes = np.bincount(self.codes, minlength=self.G)

    def mean(self, v: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Panel means of ``v`` (rows x cols), expanded back to the rows."""
        v2 = v.reshape(self.n, -1)
        w = np.ones(self.n) if rows is None else rows.astype(float)
        cnt = np.bincount(self.codes, weights=w, minlength=self.G)
        out = np.empty_like(v2, dtype=float)
        for j in range(v2.shape[1]):
            s = np.bincount(self.codes, weights=v2[:, j] * w, minlength=self.G)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, j] = (s / cnt)[self.codes]
        return out.reshape(v.shape)


def _rho_from(e: np.ndarray, lay: _Layout, kind: str, k: int) -> float:
    """``rho`` from residuals ``e`` (Stata's ``rhotype`` formulas over the
    consecutive pairs; ``k`` counts the regressors and the constant)."""
    p = lay.pair
    cur, lag = e[p], e[np.flatnonzero(p) - 1]
    num = float(cur @ lag)
    see = float(e @ e)
    n, mc = lay.n, int(p.sum())
    if kind == "regress":
        return num / float(lag @ lag)
    if kind == "freg":
        return num / float(cur @ cur)
    if kind == "tscorr":
        return num / see
    if kind == "theil":
        return num / see * (n - k) / mc
    if kind == "onestep":
        return n / mc * num / see
    rho_dw = 1.0 - float(np.sum((cur - lag) ** 2)) / see / 2.0
    if kind == "dw":
        return rho_dw
    return (rho_dw * n**2 + k**2) / (n**2 - k**2)  # nagar


def _prais_restart(z: np.ndarray, rho: float, lay: _Layout) -> np.ndarray:
    """Prais-Winsten transform; a period without its predecessor (first of
    a panel or after a gap) is scaled by ``sqrt(1 - rho^2)``."""
    lag = np.zeros_like(z)
    lag[1:] = z[:-1]
    head = (~lay.pair).reshape((-1,) + (1,) * (z.ndim - 1))
    return np.where(head, np.sqrt(1.0 - rho**2) * z, z - rho * lag)


def _baltagi_wu(z: np.ndarray, rho: float, lay: _Layout) -> np.ndarray:
    """Baltagi and Wu's ``C_i(rho) z``: a gap of ``d`` periods uses
    ``rho^d``, scaled so that the transformed disturbance has variance
    ``sigma_eta^2`` throughout."""
    lag = np.zeros_like(z)
    lag[1:] = z[:-1]
    sh = (-1,) + (1,) * (z.ndim - 1)
    d = lay.step.reshape(sh)
    c = np.sqrt(1.0 - rho**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rest = c * (z - rho**d * lag) / np.sqrt(1.0 - rho ** (2 * d))
    return np.where(lay.first.reshape(sh), c * z, rest)


def _estimate_rho(
    yd: np.ndarray,
    Xd: np.ndarray,
    lay: _Layout,
    kind: str,
    k: int,
    twostep: bool,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> Dict[str, Any]:
    """Iterated (or two-step) Prais-Winsten on the demeaned data, no
    constant, as ``prais ..., rhotype() noconstant``; ``onestep`` stops at
    the OLS residuals."""
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    if kind == "onestep":
        return {"rho": _rho_from(yd - Xd @ beta, lay, kind, k), "iterations": 0}
    rho, path = 0.0, []
    converged = False
    for _ in range(1 if twostep else maxiter):
        new = _rho_from(yd - Xd @ beta, lay, kind, k)
        if not np.isfinite(new) or abs(new) >= 1:
            raise MethodIncompatibility(
                f"xtregar: the {kind} estimate of rho is {new:.4f}; the "
                "transformation needs |rho| < 1.",
                recovery_hint="Use rhotype='tscorr', which is bounded by one, "
                "or fix rho with rhof=.",
                diagnostics={"rho": float(new), "rhotype": kind},
            )
        Xs, ys = _prais_restart(Xd, new, lay), _prais_restart(yd, new, lay)
        beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        path.append(float(new))
        done = abs(new - rho) < tol
        rho = float(new)
        if done:
            converged = True
            break
    if not (converged or twostep):
        warnings.warn(
            f"xtregar: rho did not converge in {maxiter} iterations.",
            ConvergenceWarning,
            stacklevel=4,
        )
    return {"rho": rho, "iterations": len(path), "rho_path": path}


def _lbi(e: np.ndarray, lay: _Layout) -> Dict[str, float]:
    """Bhargava-Franzini-Narendranathan Durbin-Watson as modified by
    Baltagi and Wu, and their LBI statistic, from within residuals."""
    see = float(e @ e)
    lag = np.zeros_like(e)
    lag[1:] = e[:-1]
    diff = (e - np.where(lay.pair, lag, 0.0))[~lay.first]
    d1 = float(diff @ diff) / see
    nxt_pair = np.r_[lay.pair[1:], False]
    last = np.r_[lay.first[1:], True]
    d2 = float(np.sum(np.where(~last & ~nxt_pair, e**2, 0.0))) / see
    d3 = float(np.sum(e[lay.first] ** 2)) / see
    d4 = float(np.sum(e[last] ** 2)) / see
    return {"d1": d1, "LBI": d1 + d2 + d3 + d4}


def _stata_pctile(v: np.ndarray, p: float) -> float:
    """Stata ``_pctile`` (definition 1 averaging at exact ranks)."""
    v = np.sort(np.asarray(v, dtype=float))
    n = v.size
    pos = n * p / 100.0
    i = int(np.floor(pos + 1e-12))
    if abs(pos - round(pos)) < 1e-12:
        i = int(round(pos))
        return float((v[i - 1] + v[min(i, n - 1)]) / 2.0) if i > 0 else float(v[0])
    return float(v[i])


def _harmonic(sizes: np.ndarray) -> float:
    sizes = sizes[sizes > 0]
    return float(len(sizes) / np.sum(1.0 / sizes))


def _fe(
    lay: _Layout,
    y: np.ndarray,
    X: np.ndarray,
    rho: float,
    names: List[str],
) -> Dict[str, Any]:
    ys, Xs = _baltagi_wu(y, rho, lay), _baltagi_wu(X, rho, lay)
    keep = ~lay.first
    n = int(keep.sum())
    groups = int(np.count_nonzero(np.bincount(lay.codes[keep], minlength=lay.G)))
    Z = np.column_stack([ys, Xs])
    Zw = Z - lay.mean(Z, keep) + Z[keep].mean(axis=0)
    yw, Xw = Zw[keep, 0], Zw[keep, 1:]
    p = Xw.shape[1]
    D = np.column_stack([Xw, np.ones(n)])
    bread = np.linalg.inv(D.T @ D)
    coef = bread @ D.T @ yw
    resid = yw - D @ coef
    rss = float(resid @ resid)
    tss = float(np.sum((yw - yw.mean()) ** 2))
    df_r = n - p - groups
    if df_r <= 0:
        raise DataInsufficient(
            "xtregar: too few observations for the within estimator after "
            "dropping the first period of each panel.",
            recovery_hint="Use method='re' or more periods per panel.",
        )
    s2 = rss / df_r
    V = s2 * bread
    # Stata reports the constant on the scale of the untransformed model
    # (divided by 1 - rho) and that scaling carries over to its variance.
    J = np.eye(p + 1)
    J[p, p] = 1.0 / (1.0 - rho)
    coef = J @ coef
    V = J @ V @ J.T
    b = coef[:p]
    # fit statistics on the untransformed estimation sample
    xb = X[keep] @ b
    yk = y[keep]
    codes_k = lay.codes[keep]
    cnt = np.bincount(codes_k, minlength=lay.G)
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.bincount(codes_k, yk, minlength=lay.G) / cnt
        xbbar = np.bincount(codes_k, xb, minlength=lay.G) / cnt
    present = cnt > 0
    u = (ybar - xbbar)[present]
    sigma_u = float(np.std(u, ddof=1))
    sigma_e = float(np.sqrt(s2))
    r2_b = float(np.corrcoef(xbbar[codes_k], ybar[codes_k])[0, 1] ** 2)
    r2_o = float(np.corrcoef(xb, yk)[0, 1] ** 2)
    corr = float(np.corrcoef((ybar - xbbar)[codes_k], xb)[0, 1])
    wald = float(b @ np.linalg.solve(V[:p, :p], b)) / p
    # F test that all u_i = 0: pooled OLS on the transformed data, the
    # constant transformed with them
    ones = _baltagi_wu(np.ones(lay.n), rho, lay)
    P = np.column_stack([Xs[keep], ones[keep]])
    rp = ys[keep] - P @ np.linalg.lstsq(P, ys[keep], rcond=None)[0]
    rss_pooled = float(rp @ rp)
    f_u = ((rss_pooled - rss) / (groups - 1)) / s2
    ll = -0.5 * n * (1.0 + np.log(2 * np.pi) + np.log(rss / n))
    ll_0 = -0.5 * n * (1.0 + np.log(2 * np.pi) + np.log(tss / n))
    return {
        "coef": coef,
        "V": V,
        "n": n,
        "groups": groups,
        "df_resid": df_r,
        "sigma_u": sigma_u,
        "sigma_e": sigma_e,
        "rho_fov": sigma_u**2 / (sigma_u**2 + sigma_e**2),
        "r2_within": 1.0 - rss / tss,
        "r2_between": r2_b,
        "r2_overall": r2_o,
        "r2_adj": 1.0 - (rss / tss) * (n - 1) / df_r,
        "corr_u_xb": corr,
        "F": wald,
        "F_df": (p, df_r),
        "F_pvalue": float(stats.f.sf(wald, p, df_r)),
        "F_u0": f_u,
        "F_u0_df": (groups - 1, df_r),
        "F_u0_pvalue": float(stats.f.sf(f_u, groups - 1, df_r)),
        "rss": rss,
        "mss": tss - rss,
        "rmse": float(np.sqrt(rss / (n - p - 1))),
        "log_likelihood": float(ll),
        "log_likelihood_0": float(ll_0),
        "Tbar": _harmonic(cnt),
        "g_min": int(cnt[present].min()),
        "g_max": int(cnt[present].max()),
    }


def _re(
    lay: _Layout,
    y: np.ndarray,
    X: np.ndarray,
    rho: float,
) -> Dict[str, Any]:
    n = lay.n
    D = np.column_stack([X, np.ones(n)])
    ys, Ds = _baltagi_wu(y, rho, lay), _baltagi_wu(D, rho, lay)
    mu = ys - Ds @ np.linalg.lstsq(Ds, ys, rcond=None)[0]
    g = _baltagi_wu(np.ones(n), rho, lay) / np.sqrt(1.0 - rho**2)
    c = lay.codes
    mug = np.bincount(c, mu * g, minlength=lay.G)
    gg = np.bincount(c, g * g, minlength=lay.G)
    q = float(np.sum(mug**2 / gg))
    s2e = (float(mu @ mu) - q) / float(np.sum(lay.sizes - 1))
    s2u = (q - lay.G * s2e) / float(gg.sum())
    if s2u < 0:
        s2u = 0.0
    omega = np.sqrt(gg * s2u + s2e)
    theta = 1.0 - np.sqrt(s2e) / omega
    Z = np.column_stack([ys, Ds])
    Z2 = Z.copy()
    for j in range(Z.shape[1]):
        sgz = np.bincount(c, g * Z[:, j], minlength=lay.G)
        Z2[:, j] = Z[:, j] - theta[c] * g * (sgz / gg)[c]
    y2, D2 = Z2[:, 0], Z2[:, 1:]
    k = D2.shape[1]
    bread = np.linalg.inv(D2.T @ D2)
    coef = bread @ D2.T @ y2
    resid = y2 - D2 @ coef
    V = float(resid @ resid) / (n - k) * bread
    p = k - 1
    b = coef[:p]
    xb = X @ b
    r2_o = float(np.corrcoef(xb, y)[0, 1] ** 2)
    xbm, ym = lay.mean(xb), lay.mean(y)
    r2_w = float(np.corrcoef(xb - xbm, y - ym)[0, 1] ** 2)
    first = lay.first
    r2_b = float(np.corrcoef(xbm[first], ym[first])[0, 1] ** 2)
    # Wald test of the slopes (Stata labels it chi2(k) with k counting the
    # constant, e(df_m); the statistic has p degrees of freedom)
    chi2 = float(b @ np.linalg.solve(V[:p, :p], b))
    # sigma_u on the scale of the untransformed panel effect
    sigma_u = float(np.sqrt(s2u / (1.0 - rho**2)))
    sigma_e = float(np.sqrt(s2e))
    th_obs = theta[c]
    return {
        "coef": coef,
        "V": V,
        "n": n,
        "groups": lay.G,
        "sigma_u": sigma_u,
        "sigma_e": sigma_e,
        "rho_fov": sigma_u**2 / (sigma_u**2 + sigma_e**2),
        "r2_within": r2_w,
        "r2_between": r2_b,
        "r2_overall": r2_o,
        "chi2": chi2,
        "chi2_df": p,
        "chi2_pvalue": float(stats.chi2.sf(chi2, p)),
        "theta": {
            "min": float(theta.min()),
            "p5": _stata_pctile(th_obs, 5),
            "p50": _stata_pctile(th_obs, 50),
            "p95": _stata_pctile(th_obs, 95),
            "max": float(theta.max()),
        },
        "Tbar": _harmonic(lay.sizes),
        "g_min": int(lay.sizes.min()),
        "g_max": int(lay.sizes.max()),
    }


@records_call("xtregar")
@accepts_aliases(entity="id", covariates="x")
def xtregar(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str]],
    id: str,
    time: str,
    method: str = "re",
    rhotype: str = "dw",
    rhof: Optional[float] = None,
    twostep: bool = False,
    lbi: bool = False,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Fixed- or random-effects linear regression with an AR(1) disturbance.

    Equivalent to Stata's ``xtregar y x, fe|re rhotype() rhof() twostep
    lbi``: ``y_it = a + x_it'b + nu_i + e_it`` with
    ``e_it = rho e_{i,t-1} + eta_it``, on unbalanced panels whose periods
    may be unequally spaced (a gap of ``d`` periods enters through
    ``rho^d``).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome.
    x : str or list of str
        Regressors (a constant is added).
    id, time : str
        Panel identifier and integer time index (one unit = one period;
        rescale a monthly index of quarterly data first).
    method : {'re', 'fe'}, default 're'
        ``'re'``: Baltagi and Wu's feasible GLS for a random panel effect
        (Stata's default). ``'fe'``: the within estimator, which uses the
        first period of each panel only to quasi-difference the second.
    rhotype : {'dw', 'regress', 'freg', 'tscorr', 'theil', 'nagar', \
'onestep'}, default 'dw'
        Estimator of ``rho`` from the residuals of the demeaned regression:
        ``1 - d/2`` from the Durbin-Watson ``d``, the slope of ``e_t`` on
        ``e_{t-1}`` or on ``e_{t+1}``, the first autocorrelation, Theil's or
        Nagar's adjustment, or Baltagi and Wu's one-step estimator. All but
        ``onestep`` are iterated with Prais-Winsten to convergence (Stata's
        tolerance, 1e-6) unless ``twostep``.
    rhof : float, optional
        Use this ``rho`` instead of estimating it (``|rhof| < 1``).
    twostep : bool, default False
        Stop at the first estimate of ``rho``.
    lbi : bool, default False
        Also report Baltagi and Wu's modified Bhargava-Franzini-
        Narendranathan Durbin-Watson statistic and their locally best
        invariant (LBI) statistic for ``rho = 0`` (no p-values: their null
        distributions are not tabulated).
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients and standard errors (t for ``'fe'``, z for ``'re'``);
        ``model_info`` holds ``rho_ar``, ``sigma_u``, ``sigma_e``,
        ``rho_fov``, the within, between and overall R-squared, and for
        ``'fe'`` the F tests of the slopes and of ``u_i = 0``,
        ``corr(u_i, Xb)`` and the log-likelihood; for ``'re'`` the Wald
        chi-squared of the slopes and the quantiles of the GLS weights
        ``theta_i``.

    Raises
    ------
    MethodIncompatibility
        Duplicate ``(id, time)`` rows, a non-integer time index, an unknown
        option, or ``|rho| >= 1``.
    DataInsufficient
        Too few observations for the estimator.

    Notes
    -----
    The fixed-effects constant is the intercept of the within regression on
    the transformed data divided by ``1 - rho``, as Stata reports it. Its
    standard error is divided by the same factor; Stata leaves the standard
    error on the transformed scale, so its ``_cons`` standard error is
    ``1 - rho`` times this one. The slopes and their standard errors are
    Stata's.

    The random-effects Wald statistic is Stata's; Stata labels it
    ``chi2(p + 1)`` (``e(df_m)`` counts the constant) although it tests the
    ``p`` slopes, and its p-value uses ``p + 1`` degrees of freedom. Here
    ``chi2_df`` is ``p``.

    ``sigma_e`` is the standard deviation of the innovation ``eta``;
    ``sigma_u`` of the random-effects model is on the scale of the panel
    effect in the untransformed model.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(60):
    ...     e, nu = rng.normal(), rng.normal()
    ...     for t in range(8):
    ...         e = 0.5 * e + rng.normal()
    ...         x = rng.normal()
    ...         rows.append((i, t, 1 + 2 * x + nu + e, x))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "y", "x"])
    >>> fit = sp.xtregar(df, y="y", x="x", id="id", time="t", method="fe")
    >>> round(float(fit.params["x"]), 1)  # doctest: +SKIP
    2.0

    References
    ----------
    baltagi1999unequally
    """
    xs = [x] if isinstance(x, str) else list(x)
    check_columns(data, {"y": y, "x": xs, "id": id, "time": time}, function="xtregar")
    method = str(method).lower()
    if method not in ("re", "fe"):
        raise MethodIncompatibility(
            f"xtregar: method={method!r} is not 're' or 'fe'.",
            recovery_hint="method='re' (random effects, Stata's default) or 'fe'.",
        )
    kind = str(rhotype).lower()
    if kind not in _RHOTYPES:
        raise MethodIncompatibility(
            f"xtregar: rhotype={rhotype!r} is not one of {', '.join(_RHOTYPES)}.",
            recovery_hint="rhotype='dw' is Stata's default; 'tscorr' is bounded "
            "by one.",
        )
    if twostep and kind == "onestep":
        raise MethodIncompatibility(
            "xtregar: twostep does not apply to rhotype='onestep'.",
            recovery_hint="Drop twostep=True; onestep is not iterated.",
        )
    if rhof is not None and not (np.isfinite(rhof) and abs(float(rhof)) < 1):
        raise MethodIncompatibility(
            f"xtregar: rhof={rhof!r} must lie strictly between -1 and 1.",
            recovery_hint="Pass |rhof| < 1, or omit it to estimate rho.",
        )
    df = data.loc[data[[y] + xs + [id, time]].notna().all(axis=1)]
    tvals = df[time].to_numpy(dtype=float)
    if not np.all(np.isfinite(tvals)) or np.any(tvals != np.round(tvals)):
        raise MethodIncompatibility(
            f"xtregar: the time index {time!r} must be integer-valued.",
            recovery_hint="Map the dates to consecutive integers (one per "
            "period) first; a gap of d periods is then d steps.",
        )
    df = df.assign(_t=tvals).sort_values([id, "_t"], kind="stable")
    if df.duplicated([id, "_t"]).any():
        raise MethodIncompatibility(
            f"xtregar: repeated ({id}, {time}) pairs.",
            recovery_hint="Each panel needs at most one row per period.",
            diagnostics={"n_duplicates": int(df.duplicated([id, "_t"]).sum())},
        )
    lay = _Layout(df[id].to_numpy(), df["_t"].to_numpy())
    if lay.G < 2:
        raise DataInsufficient("xtregar: needs at least two panels.")
    yv = df[y].to_numpy(dtype=float)
    X = np.column_stack([df[v].to_numpy(dtype=float) for v in xs])
    names = list(xs)

    # demeaned data (rho estimation, LBI); time-invariant regressors are not
    # identified in the within estimator, as in Stata
    Xd = X - lay.mean(X)
    yd = yv - lay.mean(yv)
    omitted: List[Dict[str, str]] = []
    if method == "fe":
        # a regressor without within-panel variation is absorbed by the
        # panel effects (its demeaned column is rounding noise)
        within = np.sqrt(np.einsum("ij,ij->j", Xd, Xd))
        scale = np.sqrt(np.einsum("ij,ij->j", X, X))
        varies = within > 1e-9 * np.maximum(scale, 1e-300)
        omitted = [
            {"variable": names[j], "reason": "no within-panel variation"}
            for j in np.flatnonzero(~varies)
        ]
        kept_v = list(np.flatnonzero(varies))
        if kept_v:
            sub, more = independent_columns(Xd[:, kept_v], [names[j] for j in kept_v])
            omitted += more
            kept = [kept_v[j] for j in sub]
        else:
            kept = []
        if not kept:
            raise MethodIncompatibility(
                "xtregar: no regressor varies within panels, so the within "
                "estimator has nothing to estimate.",
                recovery_hint="Use method='re' for time-invariant regressors.",
            )
        if omitted:
            warnings.warn(
                "xtregar: "
                + "; ".join(f"{o['variable']} omitted ({o['reason']})" for o in omitted)
                + " in the within estimator.",
                UserWarning,
                stacklevel=3,
            )
            X, Xd = X[:, kept], Xd[:, kept]
            names = [names[j] for j in kept]
    else:
        kept, omitted = independent_columns(
            np.column_stack([np.ones(lay.n), X]), ["_cons"] + names
        )
        if omitted:
            raise MethodIncompatibility(
                "xtregar: collinear regressors: "
                + ", ".join(o["variable"] for o in omitted)
                + ".",
                recovery_hint="Drop the dependent regressor(s).",
                diagnostics={"omitted": omitted},
            )
    k = X.shape[1] + 1
    if lay.n <= k + 1 or int(lay.pair.sum()) < 2:
        raise DataInsufficient(
            "xtregar: too few observations, or fewer than two pairs of "
            "consecutive periods to estimate rho.",
            recovery_hint="Check the time index: consecutive periods must "
            "differ by one.",
        )
    if rhof is not None:
        rho_info: Dict[str, Any] = {"rho": float(rhof), "iterations": 0}
    else:
        rho_info = _estimate_rho(yd, Xd, lay, kind, k, twostep)
    rho = float(rho_info["rho"])
    fit = _fe(lay, yv, X, rho, names) if method == "fe" else _re(lay, yv, X, rho)
    out_names = names + ["_cons"]
    se = np.sqrt(np.diag(fit["V"]))
    info: Dict[str, Any] = {
        "model_type": (
            "FE (within) regression with AR(1) disturbances"
            if method == "fe"
            else "RE GLS regression with AR(1) disturbances"
        ),
        "method": method,
        "rhotype": "fixed" if rhof is not None else kind,
        "twostep": bool(twostep),
        "rho_ar": rho,
        "rho_iterations": rho_info["iterations"],
        "sigma_u": fit["sigma_u"],
        "sigma_e": fit["sigma_e"],
        "rho_fov": fit["rho_fov"],
        "r2_within": fit["r2_within"],
        "r2_between": fit["r2_between"],
        "r2_overall": fit["r2_overall"],
        "n_groups": fit["groups"],
        "g_min": fit["g_min"],
        "g_max": fit["g_max"],
        "Tbar": fit["Tbar"],
        "omitted": omitted,
        "id": id,
        "time": time,
        "alpha": alpha,
    }
    for key in (
        "r2_adj",
        "corr_u_xb",
        "F",
        "F_df",
        "F_pvalue",
        "F_u0",
        "F_u0_df",
        "F_u0_pvalue",
        "rss",
        "mss",
        "rmse",
        "log_likelihood",
        "log_likelihood_0",
        "chi2",
        "chi2_df",
        "chi2_pvalue",
        "theta",
    ):
        if key in fit:
            info[key] = fit[key]
    if lbi:
        e = yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0]
        stat = _lbi(e, lay)
        info["bfn_dw"] = stat["d1"]
        info["lbi"] = stat["LBI"]
        info["n_lbi"] = lay.n
    data_info: Dict[str, Any] = {
        "nobs": fit["n"],
        "n_obs": fit["n"],
        "dependent_var": y,
        "var_cov": fit["V"],
        "var_names": out_names,
        "inference": "t" if method == "fe" else "z",
    }
    if method == "fe":
        data_info["df_resid"] = fit["df_resid"]
    return EconometricResults(
        params=pd.Series(fit["coef"], index=out_names),
        std_errors=pd.Series(se, index=out_names),
        model_info=info,
        data_info=data_info,
    )
