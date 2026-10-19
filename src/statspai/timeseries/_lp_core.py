"""Shared primitives for the local-projection estimators.

One home for the horizon regressions used by the IV, state-dependent and
panel local projections (``_lp_linear.py``, ``_lp_state.py``,
``_lp_panel.py``): OLS and two-stage least squares with ordinary,
heteroskedasticity-robust or Newey-West variance, the lag block, and the
regime weight of a state-dependent projection.

Newey-West: Bartlett weights ``1 - l / (L + 1)``, no small-sample factor;
for two-stage least squares the score ``e_t (P_Z X)_t`` sandwiched by
``(X' P_Z X)^{-1}``. These reproduce the outputs of ``lpirfs`` (GPL, used
only as a black box for numerical comparison) to 1e-14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..exceptions import MethodIncompatibility

_VCE = ("hac", "robust", "ols")


def resolve_vce(vce: Optional[str], default: str = "hac") -> str:
    """Normalise ``vce=`` and refuse unknown values."""
    if vce is None:
        return default
    key = str(vce).lower().replace("-", "_")
    aliases = {"nw": "hac", "newey_west": "hac", "hc0": "robust", "iid": "ols"}
    key = aliases.get(key, key)
    if key not in _VCE:
        raise MethodIncompatibility(
            f"sp.local_projections: vce={vce!r} is not one of {list(_VCE)}.",
            recovery_hint="Use vce='hac' (Newey-West), 'robust' (HC0) or "
            "'ols' (homoskedastic).",
        )
    return key


def bartlett_meat(scores: np.ndarray, lags: int) -> np.ndarray:
    """``sum_l w_l (G_l + G_l')`` with Bartlett weights ``1 - l/(L+1)``."""
    meat = scores.T @ scores
    for lag in range(1, int(lags) + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = scores[lag:].T @ scores[:-lag]
        meat += w * (g + g.T)
    return meat


@dataclass
class HorizonFit:
    """Coefficients and covariance of one horizon regression."""

    beta: np.ndarray
    vcov: np.ndarray
    nobs: int
    first_stage_f: Optional[float] = None


def _dfk_factors(n: int, k: int, dfk: Optional[bool]):
    """Divisor of ``e'e`` for ``vce='ols'`` and the factor on the sandwich.

    ``None`` is the ``lpirfs`` convention (``n - k`` for OLS, no factor on
    the sandwich); ``True`` / ``False`` are Stata ``lpirf`` with and without
    ``dfk``, which scales every variance type by ``n / (n - k)``.
    """
    if dfk is None:
        return n - k, 1.0
    if dfk:
        return n - k, n / (n - k)
    return n, 1.0


def ols_fit(
    y: np.ndarray, X: np.ndarray, vce: str, hac_lags: int, dfk: Optional[bool] = None
) -> HorizonFit:
    """OLS of ``y`` on ``X`` (``X`` already holds the constant)."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    divisor, factor = _dfk_factors(n, k, dfk)
    if vce == "ols":
        vcov = float(e @ e) / divisor * xtx_inv
    else:
        lags = 0 if vce == "robust" else hac_lags
        vcov = factor * (xtx_inv @ bartlett_meat(X * e[:, None], lags) @ xtx_inv)
    return HorizonFit(beta=beta, vcov=vcov, nobs=n)


def tsls_fit(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    vce: str,
    hac_lags: int,
    dfk: Optional[bool] = None,
    endog_col: int = 1,
) -> HorizonFit:
    """Two-stage least squares; ``X`` and ``Z`` both hold the constant.

    ``vce='ols'`` is ``s^2 (Xhat'Xhat)^{-1}`` with ``s^2 = e'e/(n-k)``,
    the ``lpirfs`` ``use_nw = FALSE`` branch; the other two are the
    ``lpirfs`` Newey-West sandwich (``vce='robust'`` is its lag 0).
    The first-stage F is the Wald F of the excluded instruments in the
    regression of column ``endog_col`` of ``X`` on ``Z`` with the same
    variance type.
    """
    n, k = X.shape
    ztz_inv = np.linalg.inv(Z.T @ Z)
    pz_x = Z @ (ztz_inv @ (Z.T @ X))
    xhx_inv = np.linalg.inv(pz_x.T @ pz_x)
    beta = xhx_inv @ (pz_x.T @ y)
    e = y - X @ beta
    divisor, factor = _dfk_factors(n, k, dfk)
    if vce == "ols":
        vcov = float(e @ e) / divisor * xhx_inv
    else:
        lags = 0 if vce == "robust" else hac_lags
        vcov = factor * (xhx_inv @ bartlett_meat(pz_x * e[:, None], lags) @ xhx_inv)
    return HorizonFit(
        beta=beta,
        vcov=vcov,
        nobs=n,
        first_stage_f=first_stage_f(X[:, endog_col], X, Z, vce, hac_lags, dfk),
    )


def first_stage_f(
    d: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    vce: str,
    hac_lags: int,
    dfk: Optional[bool],
) -> float:
    """Wald F of the excluded instruments in the first stage of ``d``.

    The excluded instruments are the columns of ``Z`` that are not
    columns of ``X``; they are identified by position: ``Z`` is laid out
    as ``[constant, instruments, exogenous regressors]`` and ``X`` as
    ``[constant, endogenous, exogenous regressors]``.
    """
    m = Z.shape[1] - (X.shape[1] - 1)
    fit = ols_fit(d, Z, vce, hac_lags, dfk)
    b = fit.beta[1 : 1 + m]
    V = fit.vcov[1 : 1 + m, 1 : 1 + m]
    try:
        wald = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return float("nan")
    return wald / m


def lag_columns(
    frame: pd.DataFrame, variables: Sequence[str], lags: Sequence[int]
) -> Dict[str, np.ndarray]:
    """``{<var>_L<k>: var_{t-k}}`` for every variable and lag, in order.

    The names follow the ``sp.ardl`` / ``sp.stata`` convention.
    """
    out: Dict[str, np.ndarray] = {}
    for lag in lags:
        for v in variables:
            col = frame[v].to_numpy(dtype=float)
            shifted = np.full_like(col, np.nan)
            if lag < len(col):
                shifted[lag:] = col[: len(col) - lag]
            out[f"{v}_L{lag}"] = shifted
    return out


def lead(x: np.ndarray, h: int) -> np.ndarray:
    """``x_{t+h}`` with NaN past the end."""
    out = np.full_like(x, np.nan, dtype=float)
    if h < len(x):
        out[: len(x) - h] = x[h:]
    return out


def resolve_lags(lags) -> List[int]:
    """``lags=p`` means lags 1..p; a list is taken as the lag set."""
    if lags is None:
        return []
    if isinstance(lags, (int, np.integer)):
        if lags < 0:
            raise MethodIncompatibility(
                f"sp.local_projections: lags={lags!r} is negative.",
                recovery_hint="Pass lags=0 for no lags or a positive count.",
            )
        return list(range(1, int(lags) + 1))
    out = sorted({int(v) for v in lags})
    if not out or out[0] < 1:
        raise MethodIncompatibility(
            f"sp.local_projections: lags={lags!r} must be positive integers.",
            recovery_hint="Pass lags=[1, 2] (or lags=2) for lags 1 and 2.",
        )
    return out


def hp_cycle(x: np.ndarray, smooth: float) -> np.ndarray:
    """Two-sided Hodrick-Prescott cycle ``x - (I + lambda D'D)^{-1} x``."""
    n = len(x)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    trend = spsolve((sparse.identity(n) + smooth * (d.T @ d)).tocsc(), x)
    return np.asarray(x - trend, dtype=float)


def regime_weight(
    state: np.ndarray,
    transform: str,
    gamma: Optional[float],
    hp_lambda: Optional[float],
    lag_state: bool,
) -> np.ndarray:
    """Weight ``F_t`` of regime 1 in a state-dependent projection.

    ``transform='logistic'``: ``F = exp(-gamma z) / (1 + exp(-gamma z))``
    with ``z`` the state itself, or, when ``hp_lambda`` is given, the
    standardised Hodrick-Prescott cycle of the state (``lpirfs``
    ``use_hp``). ``F`` is near 1 when ``z`` is low. ``transform='none'``:
    ``F`` is the state itself and must lie in [0, 1] (an indicator).
    ``lag_state`` uses ``F_{t-1}``, so the regime is set before the shock.
    """
    z = np.asarray(state, dtype=float)
    if transform == "logistic":
        if gamma is None or not gamma > 0:
            raise MethodIncompatibility(
                f"sp.local_projections: gamma={gamma!r} for the logistic "
                "state transform.",
                recovery_hint="Pass gamma > 0, e.g. gamma=3 (Auerbach and "
                "Gorodnichenko's choice for a standardised cycle).",
            )
        if hp_lambda is not None:
            if np.isnan(z).any():
                raise MethodIncompatibility(
                    "sp.local_projections: the state has missing values, so "
                    "its Hodrick-Prescott cycle is not defined.",
                    recovery_hint="Fill or drop the missing periods first.",
                )
            c = hp_cycle(z, float(hp_lambda))
            z = (c - c.mean()) / c.std(ddof=1)
        # exp(-g z) / (1 + exp(-g z)) == 1 / (1 + exp(g z)), stable form
        F = 0.5 * (1.0 - np.tanh(0.5 * gamma * z))
    elif transform == "none":
        finite = z[~np.isnan(z)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise MethodIncompatibility(
                "sp.local_projections: with state_transform='none' the state "
                "is the regime weight and must lie in [0, 1].",
                recovery_hint="Pass an indicator, or use "
                "state_transform='logistic' with gamma=.",
            )
        F = z.copy()
    else:
        raise MethodIncompatibility(
            f"sp.local_projections: state_transform={transform!r} is not "
            "'logistic' or 'none'.",
            recovery_hint="Use 'logistic' (smooth transition) or 'none' "
            "(the state is already a 0/1 or [0, 1] weight).",
        )
    if lag_state:
        F = np.concatenate([[np.nan], F[:-1]])
    return F
