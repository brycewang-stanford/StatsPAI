"""Panel-corrected standard errors (Stata ``xtpcse``; Beck and Katz 1995).

Coefficients are OLS, or Prais-Winsten with a common (``ar1``) or
panel-specific (``psar1``) AR(1) coefficient. Their covariance is

    V = (X'X)^{-1} [ sum_t X_t' S X_t ] (X'X)^{-1},

the sum over periods of the panels observed in that period, with ``S`` the
panel-by-panel covariance of the disturbances estimated from the residuals:
``S_ij = e_i'e_j / T_ij`` over the periods the two panels share (all panels
at once, ``casewise``, by default; each pair, ``pairwise``), a diagonal
``S`` with ``hetonly``, and ``s^2 I`` with ``independent``. ``nmk`` divides
by ``N - k`` instead of ``N``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xtpcse"]


def _rho_panel(e: np.ndarray, gap_free: np.ndarray, how: str) -> float:
    """AR(1) coefficient of one panel's residual series (Stata ``rhotype``).

    ``gap_free[t]`` is True when observation ``t`` follows ``t-1`` in time.
    """
    lag = np.r_[np.nan, e[:-1]]
    ok = gap_free & np.isfinite(lag)
    if how == "regress":  # e_t on e_{t-1}
        den = float(lag[ok] @ lag[ok])
        return float(e[ok] @ lag[ok]) / den if den > 0 else 0.0
    if how == "freg":  # e_{t-1} on e_t (leads)
        den = float(e[ok] @ e[ok])
        return float(e[ok] @ lag[ok]) / den if den > 0 else 0.0
    if how == "tscorr":
        den = float(e @ e)
        return float(e[ok] @ lag[ok]) / den if den > 0 else 0.0
    # dw: 1 - DW / 2
    d = e[ok] - lag[ok]
    den = float(e @ e)
    return 1.0 - 0.5 * float(d @ d) / den if den > 0 else 0.0


@records_call("xtpcse")
@accepts_aliases(entity="id", covariates="x")
def xtpcse(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str]],
    id: str,
    time: str,
    correlation: str = "independent",
    rhotype: str = "regress",
    np1: bool = False,
    hetonly: bool = False,
    independent: bool = False,
    pairwise: bool = False,
    nmk: bool = False,
    noconstant: bool = False,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Linear regression with panel-corrected standard errors.

    Equivalent to Stata's ``xtpcse y x, correlation() rhotype() np1
    hetonly independent pairwise nmk``: OLS (or Prais-Winsten under AR(1)
    disturbances) with standard errors that allow the disturbances to be
    heteroskedastic across panels and contemporaneously correlated between
    them (Beck and Katz 1995).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Dependent variable.
    x : str or list of str
        Regressors (a constant is added unless ``noconstant``).
    id, time : str
        Panel and time identifiers (integer-spaced time; a gap breaks the
        AR(1) chain, as in Stata).
    correlation : {'independent', 'ar1', 'psar1'}, default 'independent'
        Within-panel autocorrelation: none (OLS), a common AR(1)
        coefficient, or panel-specific ones (Prais-Winsten).
    rhotype : {'regress', 'freg', 'tscorr', 'dw'}, default 'regress'
        How each panel's AR(1) coefficient is computed from the OLS
        residuals.
    np1 : bool, default False
        With ``correlation='ar1'``, weight the panel coefficients by
        ``T_i`` instead of ``T_i - 1`` in the common coefficient.
    hetonly : bool, default False
        Panel heteroskedasticity only (no contemporaneous correlation).
    independent : bool, default False
        One disturbance variance for all observations.
    pairwise : bool, default False
        Estimate each covariance from the periods two panels share, instead
        of only the periods all panels share (``casewise``).
    nmk : bool, default False
        Normalise by ``N - k`` instead of ``N``.
    noconstant : bool, default False
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients with z inference. ``model_info`` holds the AR(1)
        coefficient(s) (``rho``), the panel covariance ``sigma`` (an array),
        the number of panels and the periods used for it.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> m, T = 8, 25
    >>> common = rng.normal(size=T)
    >>> df = pd.DataFrame({"id": np.repeat(np.arange(m), T),
    ...                    "t": np.tile(np.arange(T), m)})
    >>> df["x"] = rng.normal(size=m * T)
    >>> df["y"] = 1 + 0.5 * df["x"] + common[df["t"]] + rng.normal(size=m * T)
    >>> fit = sp.xtpcse(df, y="y", x="x", id="id", time="t")
    >>> list(fit.params.index)
    ['x', '_cons']

    References
    ----------
    beck1995what
    """
    corr = str(correlation).lower()
    if corr not in ("independent", "ar1", "psar1"):
        raise MethodIncompatibility(
            f"xtpcse: correlation must be 'independent', 'ar1' or 'psar1'; "
            f"got {correlation!r}.",
        )
    how = str(rhotype).lower()
    if how not in ("regress", "freg", "tscorr", "dw"):
        raise MethodIncompatibility(
            f"xtpcse: rhotype must be 'regress', 'freg', 'tscorr' or 'dw'; "
            f"got {rhotype!r}.",
        )
    if hetonly and independent:
        raise MethodIncompatibility(
            "xtpcse: hetonly and independent are alternatives; pass one.",
        )
    xs = [x] if isinstance(x, str) else list(x)
    check_columns(data, {"y": y, "x": xs, "id": id, "time": time}, function="xtpcse")
    df = data.loc[data[[y, id, time] + xs].notna().all(axis=1)]
    df = df.sort_values([id, time], kind="mergesort")
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "xtpcse: repeated time values within a panel.",
            recovery_hint="Each (id, time) pair must appear once.",
        )
    pcodes, panels = pd.factorize(df[id], sort=True)
    tcodes, periods = pd.factorize(df[time], sort=True)
    m, nT = len(panels), len(periods)
    if m < 2:
        raise DataInsufficient("xtpcse: needs at least two panels.")
    N = len(df)
    yv = df[y].to_numpy(dtype=float)
    cols = [df[v].to_numpy(dtype=float) for v in xs]
    names = list(xs)
    if not noconstant:
        cols.append(np.ones(N))
        names.append("_cons")
    X = np.column_stack(cols)
    k = X.shape[1]
    tval = df[time].to_numpy()
    prev_same = np.r_[False, pcodes[1:] == pcodes[:-1]]
    try:
        step = np.r_[np.nan, np.diff(tval.astype(float))]
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "xtpcse: time must be numeric (integer-spaced periods).",
        ) from exc
    gap_free = prev_same & (step == 1)

    info: Dict[str, Any] = {"correlation": corr, "n_panels": m, "n_periods": nT}
    if corr == "independent":
        Xs, ys = X, yv
    else:
        b_ols = np.linalg.lstsq(X, yv, rcond=None)[0]
        e = yv - X @ b_ols
        rho_i = np.zeros(m)
        pairs = np.zeros(m)
        for i in range(m):
            sel = pcodes == i
            rho_i[i] = _rho_panel(e[sel], gap_free[sel], how)
            pairs[i] = gap_free[sel].sum()
        if corr == "ar1":
            # Stata weights each panel by its number of consecutive pairs
            # (T_i - 1 on a panel without gaps; one more with np1), not by
            # T_i - 1 as its manual says for unbalanced panels
            w = pairs + 1.0 if np1 else pairs
            rho_used = np.full(m, float(np.sum(w * rho_i) / np.sum(w)))
            info["rho"] = float(rho_used[0])
        else:
            rho_used = rho_i
            info["rho"] = dict(zip([str(p) for p in panels], rho_i.tolist()))
        r = rho_used[pcodes]
        # Prais-Winsten: the first observation of a run is scaled by
        # sqrt(1 - rho^2), the others quasi-differenced
        lagy = np.r_[np.nan, yv[:-1]]
        lagX = np.vstack([np.full((1, k), np.nan), X[:-1]])
        ys = np.where(gap_free, yv - r * lagy, np.sqrt(1 - r * r) * yv)
        Xs = np.where(
            gap_free[:, None], X - r[:, None] * lagX, np.sqrt(1 - r * r)[:, None] * X
        )
    XtX_inv = np.linalg.inv(Xs.T @ Xs)
    beta = XtX_inv @ (Xs.T @ ys)
    e = ys - Xs @ beta
    denom_adj = (N - k) if nmk else N

    if independent:
        s2 = float(e @ e) / denom_adj
        V = s2 * XtX_inv
        info["sigma2"] = s2
    else:
        E = np.full((m, nT), np.nan)
        E[pcodes, tcodes] = e
        if hetonly:
            Ti = np.bincount(pcodes, minlength=m).astype(float)
            s2_i = np.nansum(E * E, axis=1) / Ti
            if nmk:
                s2_i = s2_i * N / (N - k)
            S = np.diag(s2_i)
        else:
            if pairwise:
                obs = np.isfinite(E).astype(float)
                E0 = np.where(np.isfinite(E), E, 0.0)
                Tij = obs @ obs.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    S = (E0 @ E0.T) / Tij
                if np.any(Tij == 0):
                    raise DataInsufficient(
                        "xtpcse: two panels share no period, so their "
                        "covariance cannot be estimated (pairwise).",
                        recovery_hint="Use hetonly=True or drop the panels "
                        "that do not overlap.",
                    )
                info["periods_used"] = None
            else:
                common = np.all(np.isfinite(E), axis=0)
                Tc = int(common.sum())
                if Tc == 0:
                    raise DataInsufficient(
                        "xtpcse: no period is observed for every panel, so "
                        "the casewise covariance has no data.",
                        recovery_hint="Pass pairwise=True or hetonly=True.",
                    )
                Ec = E[:, common]
                S = Ec @ Ec.T / Tc
                info["periods_used"] = Tc
            if nmk:
                S = S * N / (N - k)
        meat = np.zeros((k, k))
        Xt = np.zeros((m, k))
        for t in range(nT):
            sel = tcodes == t
            Xt[:] = 0.0
            Xt[pcodes[sel]] = Xs[sel]
            meat += Xt.T @ S @ Xt
        V = XtX_inv @ meat @ XtX_inv
        info["sigma"] = S
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    resid_raw = yv - X @ beta
    r2 = 1.0 - float(resid_raw @ resid_raw) / float(np.sum((yv - yv.mean()) ** 2))
    info.update(
        model_type="Panel-corrected SE (xtpcse)",
        hetonly=hetonly,
        independent=independent,
        pairwise=pairwise,
        nmk=nmk,
        r2=r2,
        alpha=alpha,
        id=id,
        time=time,
    )
    return EconometricResults(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": N,
            "n_obs": N,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "df_resid": N - k,
            "inference": "z",
        },
        diagnostics={"z": dict(zip(names, z))},
    )
