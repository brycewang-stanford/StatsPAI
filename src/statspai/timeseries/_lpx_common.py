"""Shared pieces of the explicit local-projection designs.

The single-series (OLS and two-stage least squares), state-dependent and
panel projections of :func:`statspai.local_projections` all reduce to one
regression per horizon followed by a sandwich variance. This module holds
those two steps and the argument checks the three designs share.

Every horizon regression is ``lhs = X b + e``. For OLS the score regressors
are ``X``; for two-stage least squares they are the first-stage fitted
values ``Xhat = P_W X`` and the residual is the structural one,
``e = lhs - X b``. The variance is ``B M B`` with ``B = (S'S)^{-1}`` for the
score regressors ``S`` and a meat ``M`` chosen by ``vce``:

* ``'ols'``     ``M = s2 S'S`` with ``s2 = e'e / d``;
* ``'robust'``  ``M = sum_t e_t^2 s_t s_t'`` (HC0);
* ``'hac'``     Bartlett-weighted autocovariances of ``s_t e_t`` in row
  order up to lag ``L`` [@newey1987simple];
* ``'cluster'`` ``M = sum_g u_g u_g'``, ``u_g`` the sum of ``s_t e_t``
  within cluster ``g``;
* ``'driscoll_kraay'`` the Bartlett estimator applied to the period sums
  of ``s_t e_t``, robust to cross-sectional dependence of any form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from ..core._collinear import independent_columns
from ..exceptions import DataInsufficient, MethodIncompatibility

FN = "sp.local_projections"


def refuse(message: str, hint: str, **diagnostics: Any) -> MethodIncompatibility:
    """A :class:`MethodIncompatibility` naming ``sp.local_projections``."""
    return MethodIncompatibility(
        f"{FN}: {message}", recovery_hint=hint, diagnostics=diagnostics
    )


# --------------------------------------------------------------------- #
#  Argument normalisation
# --------------------------------------------------------------------- #


def lag_list(lags: Any) -> List[int]:
    """``lags`` as a sorted list of positive lag orders.

    An integer ``p`` means ``1..p`` (``0`` means none); a sequence lists the
    orders themselves, as Stata ``lags(1 3)`` does.
    """
    if isinstance(lags, (bool, np.bool_)):
        raise refuse(
            f"lags={lags!r} is a boolean.",
            "Pass the number of lags p (lags 1..p) or a list of lag orders.",
        )
    if isinstance(lags, (int, np.integer)):
        if lags < 0:
            raise refuse(
                f"lags={lags} is negative.",
                "Pass lags=0 for no lags, p for lags 1..p, or a list such "
                "as [1, 3].",
            )
        return list(range(1, int(lags) + 1))
    try:
        orders = [o for o in lags]
    except TypeError:
        raise refuse(
            f"lags={lags!r} is neither an integer nor a list of integers.",
            "Pass the number of lags p (lags 1..p) or a list of lag orders.",
        ) from None
    ok = all(
        isinstance(o, (int, np.integer)) and not isinstance(o, (bool, np.bool_))
        for o in orders
    )
    if not ok or any(o < 1 for o in orders):
        raise refuse(
            f"lags={list(orders)!r}: a list of lags must hold positive "
            "integers (the lag orders).",
            "e.g. lags=[1, 3] for the first and third lags, or lags=2 for "
            "lags 1 and 2.",
        )
    return sorted({int(o) for o in orders})


def check_choice(name: str, value: Any, allowed: Sequence[Any], hint: str) -> None:
    """Refuse ``value`` unless it is one of ``allowed``."""
    if value not in allowed:
        raise refuse(f"{name}={value!r} is not one of {list(allowed)}.", hint)


def check_nw_lags(nw_lags: Optional[int], default_rule: str) -> None:
    """Refuse a negative or non-integer Newey-West truncation lag."""
    if nw_lags is None:
        return
    if (
        isinstance(nw_lags, (bool, np.bool_))
        or not isinstance(nw_lags, (int, np.integer))
        or nw_lags < 0
    ):
        raise MethodIncompatibility(
            f"nw_lags must be a non-negative integer, got {nw_lags!r}.",
            recovery_hint=f"Leave it unset for the default ({default_rule}); "
            "nw_lags=0 is the heteroskedasticity-robust (White) variance.",
        )


def trend_columns(n_rows: int, trend: int) -> List[np.ndarray]:
    """Deterministic trend regressors over the row positions ``1..n``."""
    t = np.arange(1, n_rows + 1, dtype=float)
    return [t, t**2][:trend]


# --------------------------------------------------------------------- #
#  One regression
# --------------------------------------------------------------------- #


@dataclass
class HorizonFit:
    """Estimates of one horizon regression."""

    coef: np.ndarray
    resid: np.ndarray
    score: np.ndarray  # regressors entering the meat (X, or Xhat for 2SLS)
    bread: np.ndarray  # (score' score)^{-1}

    @property
    def n(self) -> int:
        return int(self.score.shape[0])

    @property
    def k(self) -> int:
        return int(self.score.shape[1])


def require_full_rank(
    X: np.ndarray, names: Sequence[str], horizon: int, what: str = "regressors"
) -> None:
    """Refuse a design whose columns are linearly dependent."""
    _, dependent = independent_columns(X, list(names))
    if dependent:
        raise refuse(
            f"at horizon {horizon} the {what} are perfectly collinear ("
            + "; ".join(f"{d['variable']}: {d['reason']}" for d in dependent)
            + "), so the coefficient on the shock is not identified.",
            "Drop the redundant control, lag variable or instrument.",
            horizon=horizon,
            omitted=dependent,
        )


def fit_ols(X: np.ndarray, y: np.ndarray) -> HorizonFit:
    """Least squares of ``y`` on ``X`` (full column rank assumed)."""
    q, r = np.linalg.qr(X, mode="reduced")
    coef = solve_triangular(r, q.T @ y, lower=False)
    r_inv = solve_triangular(r, np.eye(r.shape[0]), lower=False)
    return HorizonFit(coef, y - X @ coef, X, r_inv @ r_inv.T)


def fit_2sls(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> HorizonFit:
    """Two-stage least squares of ``y`` on ``X`` with instruments ``W``.

    ``W`` holds every instrument, the included exogenous regressors among
    them. The score regressors are the projections ``Xhat = P_W X``.
    """
    qw, _ = np.linalg.qr(W, mode="reduced")
    xhat = qw @ (qw.T @ X)
    q, r = np.linalg.qr(xhat, mode="reduced")
    coef = solve_triangular(r, q.T @ y, lower=False)
    r_inv = solve_triangular(r, np.eye(r.shape[0]), lower=False)
    return HorizonFit(coef, y - X @ coef, xhat, r_inv @ r_inv.T)


def prune_instruments(
    W: np.ndarray, names: Sequence[str], excluded: Sequence[int], horizon: int
) -> tuple:
    """Drop excluded instruments that add nothing to the included ones.

    The included regressors are scanned first, so an excluded instrument
    that duplicates (or is a combination of) them and the instruments
    before it is the column omitted, as Stata's ``ivregress`` does. Returns
    the kept matrix, the positions of the surviving excluded instruments in
    it, and the names dropped. No surviving excluded instrument means the
    shock is not identified.
    """
    excluded = list(excluded)
    order = [j for j in range(W.shape[1]) if j not in excluded] + excluded
    keep, dropped = independent_columns(W, list(names), order=order)
    gone = [j for j in range(W.shape[1]) if j not in keep]
    if any(j not in excluded for j in gone):
        require_full_rank(W, names, horizon, what="instruments")
    lost = [names[j] for j in gone]
    survivors = [j for j in excluded if j in keep]
    if not survivors:
        raise refuse(
            f"at horizon {horizon} no excluded instrument varies apart from "
            "the included regressors ("
            + "; ".join(f"{d['variable']}: {d['reason']}" for d in dropped)
            + "), so the shock is not identified.",
            "Use an instrument that is not a control, a lag or a combination "
            "of them.",
            horizon=horizon,
            omitted=dropped,
        )
    return W[:, keep], [keep.index(j) for j in survivors], lost


def variance_settings(dfk: Optional[bool], n: int, k: int) -> tuple:
    """``(divisor of e'e for vce='ols', factor on the sandwiches)``.

    ``dfk=None`` is the ``lpirfs`` convention (``n - k``, no factor),
    ``True`` Stata ``lpirf, dfk`` (``n - k``, factor ``n / (n - k)``) and
    ``False`` Stata ``lpirf`` (``n``, no factor).
    """
    if dfk is None:
        return n - k, 1.0
    if dfk:
        return n - k, n / (n - k)
    return n, 1.0


def first_stage_f(
    endog: np.ndarray,
    W: np.ndarray,
    excluded: Sequence[int],
    vce: str,
    *,
    dfk: Optional[bool] = None,
    absorbed: int = 0,
    **cov_kw: Any,
) -> float:
    """Wald F of the excluded instruments in the first stage.

    The first-stage covariance is of the kind the projection uses (``vce``,
    its lag, clusters and ``dfk``), so under ``vce='ols'`` this is the
    classical F and under ``'robust'`` / ``'hac'`` / ``'cluster'`` /
    ``'driscoll_kraay'`` the robust Wald statistic divided by the number of
    excluded instruments.
    """
    fit = fit_ols(W, endog)
    divisor, factor = variance_settings(dfk, fit.n, fit.k)
    V = covariance(
        fit, vce, resid_dof=divisor - absorbed, sandwich_factor=factor, **cov_kw
    )
    ex = list(excluded)
    b = fit.coef[ex]
    try:
        stat = float(b @ np.linalg.solve(V[np.ix_(ex, ex)], b))
    except np.linalg.LinAlgError:
        return float("nan")
    return stat / len(ex)


def check_exact_fit(fit: HorizonFit, lhs: np.ndarray, horizon: int) -> None:
    """Refuse a horizon regression that reproduces its outcome exactly.

    Its residuals are rounding noise, so the standard errors would be a
    "zero" whose size depends on the platform's BLAS.
    """
    scale = float(np.linalg.norm(lhs - lhs.mean()))
    if float(np.linalg.norm(fit.resid)) <= 1e-10 * max(
        scale, float(np.abs(lhs).max(initial=0.0))
    ):
        raise refuse(
            f"at horizon {horizon} the regressors reproduce the outcome "
            "exactly, so its standard errors are rounding noise.",
            "Check that the outcome (or an identity of it) is not among the "
            "controls or lag variables at this horizon.",
            horizon=horizon,
        )


# --------------------------------------------------------------------- #
#  Variance estimators
# --------------------------------------------------------------------- #


def _bartlett(scores: np.ndarray, lag: int) -> np.ndarray:
    """``sum_{l=-L}^{L} (1 - |l|/(L+1)) Gamma_l`` of the rows of ``scores``."""
    meat = np.asarray(scores.T @ scores, dtype=float)
    for ell in range(1, min(int(lag), scores.shape[0] - 1) + 1):
        gamma = scores[ell:].T @ scores[:-ell]
        meat += (1.0 - ell / (lag + 1.0)) * (gamma + gamma.T)
    return meat


def covariance(
    fit: HorizonFit,
    vce: str,
    *,
    resid_dof: Optional[int] = None,
    hac_lag: int = 0,
    sandwich_factor: float = 1.0,
    groups: Optional[np.ndarray] = None,
    periods: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Covariance matrix of ``fit.coef`` under ``vce``.

    ``resid_dof`` is the divisor of ``e'e`` for ``vce='ols'``;
    ``sandwich_factor`` multiplies every other estimator (``n / (n - k)``
    for Stata's ``dfk``, 1 otherwise). ``groups`` are the cluster labels
    and ``periods`` the integer period codes (Driscoll-Kraay).
    """
    s, e, bread = fit.score, fit.resid, fit.bread
    if vce == "ols":
        dof = fit.n - fit.k if resid_dof is None else resid_dof
        return np.asarray(float(e @ e) / dof * bread, dtype=float)
    se = s * e[:, None]
    if vce == "robust":
        meat = se.T @ se
    elif vce == "hac":
        meat = _bartlett(se, hac_lag)
    elif vce == "cluster":
        assert groups is not None
        codes, _ = pd.factorize(groups)
        sums = np.zeros((codes.max() + 1, se.shape[1]))
        np.add.at(sums, codes, se)
        meat = sums.T @ sums
    elif vce == "driscoll_kraay":
        assert periods is not None
        order = np.unique(periods)
        pos = np.searchsorted(order, periods)
        sums = np.zeros((len(order), se.shape[1]))
        np.add.at(sums, pos, se)
        meat = _bartlett(sums, hac_lag)
    else:  # pragma: no cover - callers validate vce
        raise ValueError(vce)
    return np.asarray(sandwich_factor * (bread @ meat @ bread), dtype=float)


def check_dof(n: int, k: int, horizon: int, absorbed: int = 0) -> None:
    """Refuse a horizon regression with no residual degrees of freedom."""
    if n - k - absorbed <= 0:
        raise DataInsufficient(
            f"{FN}: {n} observations at horizon {horizon} leave no degrees "
            f"of freedom for {k + absorbed} parameters.",
            recovery_hint="Reduce horizons=, the lags or the controls, or "
            "use a longer sample.",
            diagnostics={"horizon": horizon, "n_obs": n, "n_params": k + absorbed},
        )


def as_float_columns(data: pd.DataFrame, cols: Sequence[str]) -> Dict[str, np.ndarray]:
    """The named columns as float arrays (duplicates looked up once)."""
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        if c in out:
            continue
        try:
            out[c] = data[c].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise refuse(
                f"column {c!r} is not numeric.",
                "Convert it to numbers first (e.g. pd.to_numeric).",
                column=c,
            ) from None
    return out
