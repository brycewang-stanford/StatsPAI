"""Goodman-Bacon decomposition with time-varying covariates.

With covariates the TWFE coefficient is (Goodman-Bacon 2021, Section 5)

    beta_DD|X = Omega * beta_w + (1 - Omega) * sum_kl s_kl * beta_kl,

where ``beta_w`` uses only the variation of the covariate-residualised
treatment *within* timing groups, ``Omega`` is that variation's share of
the total, and each between-group term ``beta_kl`` (timing groups ``k``
earlier than ``l``) combines the 2x2 with the covariates' timing-group
means (``beta_D``) and the part of treatment the covariate means predict
(``beta_b``). This follows R ``bacondecomp`` 0.1.1 (MIT) term by term; the
identity above holds exactly and is asserted in the tests.

Every regression with unit and period effects here runs on a balanced
panel (or a balanced subset of its units), where two-way demeaning is
exact: ``x - xbar_i - xbar_t + xbar``.

With a weight that is constant within unit (``weights=``) every mean,
variance and least-squares fit below is the ``w``-weighted one; the unit
mean stays the plain time mean (the unit's weight is the same in every
period). An integer weight is the same as repeating the unit, so this is
the unweighted decomposition of the replicated panel, and the identity
above holds for the weighted TWFE coefficient. Without weights every
helper runs the original unweighted expression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["bacon_with_controls"]


def _wmeans(M: np.ndarray, keys: list, w: np.ndarray) -> np.ndarray:
    """``w``-weighted group means of the columns of ``M``, broadcast back."""
    df = pd.DataFrame(M * w[:, None])
    num = df.groupby(keys).transform("sum").to_numpy()
    den = pd.Series(w).groupby(keys).transform("sum").to_numpy()
    return num / den[:, None]


def _dd(
    M: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Two-way demeaning on a balanced panel (columns of ``M``)."""
    M = np.asarray(M, dtype=float)
    two_d = M.ndim == 1
    if two_d:
        M = M[:, None]
    df = pd.DataFrame(M)
    if w is None:
        out = (
            M
            - df.groupby(unit).transform("mean").to_numpy()
            - df.groupby(time).transform("mean").to_numpy()
            + M.mean(axis=0, keepdims=True)
        )
    else:
        out = (
            M
            - df.groupby(unit).transform("mean").to_numpy()
            - _wmeans(M, [time], w)
            + np.average(M, axis=0, weights=w)[None, :]
        )
    _ret: np.ndarray = out[:, 0] if two_d else out
    return _ret


def _cell_mean(
    v: np.ndarray, *keys: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    _ret: np.ndarray
    if w is None:
        _ret = pd.Series(v).groupby(list(keys)).transform("mean").to_numpy()
    else:
        _ret = _wmeans(np.asarray(v, dtype=float)[:, None], list(keys), w)[:, 0]
    return _ret


def _mean(v: np.ndarray, w: Optional[np.ndarray]) -> float:
    return float(v.mean() if w is None else np.average(v, weights=w))


def _var(v: np.ndarray, w: Optional[np.ndarray]) -> float:
    if w is None:
        return float(np.var(v))
    return float(np.average((v - np.average(v, weights=w)) ** 2, weights=w))


def _sumsq(v: np.ndarray, w: Optional[np.ndarray]) -> float:
    return float(np.sum(v**2) if w is None else np.sum(w * v**2))


def _lstsq(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if w is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.sqrt(w)
    X = X * (r[:, None] if X.ndim == 2 else r)
    return np.linalg.lstsq(X, y * r, rcond=None)[0]


def _slope(y: np.ndarray, x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        return float(
            np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        )
    xc = x - np.average(x, weights=w)
    yc = y - np.average(y, weights=w)
    return float(np.sum(w * xc * yc) / np.sum(w * xc**2))


def bacon_with_controls(
    df: pd.DataFrame,
    *,
    y: str,
    treat: str,
    time: str,
    id: str,
    covariates: List[str],
    treat_time: str,
    weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Within component, Omega and the between-group 2x2 terms."""
    nonnum = [
        c
        for c in covariates
        if not (
            pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        )
    ]
    if nonnum:
        raise MethodIncompatibility(
            f"bacon_decomposition: covariates {nonnum} are not numeric.",
            recovery_hint="Add dummy columns for a categorical covariate "
            "(pd.get_dummies(df[col], drop_first=True)) and list those.",
        )
    X = df[covariates].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise DataInsufficient(
            "bacon_decomposition: covariates have missing values; the "
            "decomposition needs a balanced panel with complete data.",
            recovery_hint="Drop or impute the missing covariate values first.",
        )
    unit = pd.factorize(df[id])[0]
    per = pd.factorize(df[time])[0]
    g = df[treat_time].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    N = len(df)
    w = None if weights is None else df[weights].to_numpy(dtype=float)

    # treatment net of covariates and both sets of effects (FWL)
    Xd = _dd(X, unit, per, w)
    Dd = _dd(D, unit, per, w)
    scale = np.maximum(np.abs(X).max(axis=0), 1.0)
    flat = [
        c
        for j, c in enumerate(covariates)
        if np.abs(Xd[:, j]).max() <= 1e-10 * scale[j]
    ]
    if flat:
        # used to be reported as "the treatment is collinear" (2026-10-20)
        raise MethodIncompatibility(
            f"bacon_decomposition: covariates {flat} do not vary once unit "
            "and period effects are removed (time-invariant, or common to "
            "all units in each period); the two-way effects absorb them.",
            recovery_hint="Drop them; they do not change the TWFE estimate.",
            diagnostics={"absorbed": flat},
        )
    if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        raise MethodIncompatibility(
            "bacon_decomposition: the covariates are collinear with each "
            "other once unit and period effects are removed.",
            recovery_hint="Drop the redundant covariate(s).",
        )
    if np.linalg.matrix_rank(np.column_stack([Xd, Dd])) < Xd.shape[1] + 1:
        raise MethodIncompatibility(
            "bacon_decomposition: the treatment is collinear with the "
            "covariates once unit and period effects are removed.",
            recovery_hint="Drop covariates that are functions of treatment timing.",
        )
    b = _lstsq(Xd, Dd, w)
    d = Dd - Xd @ b  # residual of treat ~ X + unit + period effects
    p = D - d  # its fitted value

    d_i = _cell_mean(d, unit)  # one weight per unit: the plain time mean
    d_t = _cell_mean(d, per, w=w)
    d_it_til = d - d_i - d_t + _mean(d, w)
    d_kt = _cell_mean(d, g, per, w=w)
    d_k = _cell_mean(d, g, w=w)
    d_ikt_til = d - d_i - (d_kt - d_k)
    omega = float(_var(d_ikt_til, w) / _var(d_it_til, w))
    if w is None:
        beta_w = float(
            np.mean((Y - Y.mean()) * (d_ikt_til - d_ikt_til.mean())) / np.var(d_ikt_til)
        )
    else:
        beta_w = _slope(Y, d_ikt_til, w)

    gX = np.column_stack([_cell_mean(X[:, j], g, per, w=w) for j in range(X.shape[1])])
    g_p = _cell_mean(p, g, per, w=w)

    groups = sorted(np.unique(g[np.isfinite(g)]))
    all_groups = groups + ([np.inf] if np.isinf(g).any() else [])
    t_min = float(df[time].min())
    rows = []
    for k in groups:
        for l_ in all_groups:
            if not l_ > k:
                continue
            m = np.isin(g, [k, l_])
            u, t = unit[m], per[m]
            wm = None if w is None else w[m]
            Dt = _dd(D[m], u, t, wm)
            VD = _var(Dt, wm)
            # A pair whose treatment does not vary within units over the
            # window (an always-treated group against the never treated)
            # has no 2x2 variation: VD is rounding noise, the D term drops
            # out and the pair is its covariate-mean term alone.
            no_2x2 = VD < 1e-12
            pg = _dd(gX[m], u, t, wm)
            Z = np.column_stack([np.ones(m.sum()), pg])
            coef = _lstsq(Z, Dt, wm)
            fitted = Z @ coef
            if no_2x2:
                fitted = np.zeros_like(Dt)
                d_term, beta_d = 0.0, float("nan")
            else:
                tss = _sumsq(Dt - _mean(Dt, wm), wm)
                rsq = 1.0 - _sumsq(Dt - fitted, wm) / tss
                d_term = (1 - rsq) * VD
            dp = fitted - _dd(g_p[m], u, t, wm)
            Vdp = _var(dp, wm)
            Yd = _dd(Y[m], u, t, wm)
            if not no_2x2:
                W = np.column_stack([Dt, pg])
                bd = _lstsq(W, Yd, wm)
                beta_d = float(bd[0])
            dpd = _dd(dp, u, t, wm)
            has_b = _sumsq(dpd, wm) > 1e-14 * max(1.0, _sumsq(Yd, wm))
            beta_b = _slope(Yd, dpd, wm) if has_b else 0.0
            n = float(m.sum()) if wm is None else float(wm.sum())
            s_kl = n**2 * (d_term + Vdp)
            if d_term + Vdp <= 0:
                continue
            est = ((d_term * beta_d if d_term else 0.0) + Vdp * beta_b) / (d_term + Vdp)
            if np.isinf(l_):
                kind, control = "Treated vs Untreated", "Never"
            elif k == t_min or l_ == t_min:
                kind, control = "Later vs Always Treated", l_
            else:
                kind, control = "Both Treated", l_
            rows.append(
                {
                    "type": kind,
                    "treated": k,
                    "control": control,
                    "estimate": float(est),
                    "weight": float(s_kl),
                    "beta_2x2": beta_d,
                    "beta_between": beta_b,
                }
            )
    decomp = pd.DataFrame(rows)
    if len(decomp):
        decomp["weight"] = decomp["weight"] / decomp["weight"].sum()
    between = (
        float((decomp["weight"] * decomp["estimate"]).sum()) if len(decomp) else 0.0
    )
    return {
        "decomposition": decomp,
        "omega": omega,
        "beta_within": beta_w,
        "beta_between": between,
        "weighted_sum": omega * beta_w + (1 - omega) * between,
        "n_obs": N,
    }


def twfe_with_controls(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    covariates: List[str],
    weights: Optional[str] = None,
) -> float:
    """TWFE coefficient on the treatment with covariates (balanced panel)."""
    unit = pd.factorize(df[id])[0]
    per = pd.factorize(df[time])[0]
    w = None if weights is None else df[weights].to_numpy(dtype=float)
    M = _dd(df[[y, treat, *covariates]].to_numpy(dtype=float), unit, per, w)
    beta = _lstsq(M[:, 1:], M[:, 0], w)
    return float(beta[0])
