"""
Robust regression: M, S and MM estimators.

Least squares gives every observation the same say, so a handful of gross
errors can decide a coefficient. The estimators here bound that influence.

* **M** [@huber1964robust]: minimise ``sum rho(r_i / s)`` for a ``rho`` that
  grows more slowly than the square. Protects against outliers in the
  outcome, not against high-leverage points.
* **S** [@rousseeuw1984robust]: the coefficients that make a robust scale of
  the residuals as small as possible. Breakdown point 50%, but only 28.7%
  as efficient as least squares when the errors are normal.
* **MM** [@yohai1987high]: an S fit for the scale and the starting values,
  then a redescending M step at that scale. Keeps the 50% breakdown point
  and reaches a chosen efficiency (85% by default).

``sp.robreg`` is checked against R ``robustbase::lmrob`` / ``MASS::rlm`` and
Stata ``robreg`` (Jann). The S search follows the fast-S algorithm
[@salibian2006fast]; nothing in this module is ported from those packages.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

_MADN = 0.6744897501960817  # Phi^{-1}(0.75): MAD / _MADN estimates sigma
_MAD_ROUNDED = 0.6745  # the same constant as rlm and Stata rreg write it


# ---------------------------------------------------------------------------
# rho functions
# ---------------------------------------------------------------------------


class _Psi:
    """A rho function with tuning constant ``c``: rho, psi = rho', psi'
    and the weight psi(u) / u."""

    def __init__(self, kind: str, c: float) -> None:
        self.kind = kind
        self.c = float(c)

    def rho(self, u: np.ndarray) -> np.ndarray:
        c = self.c
        if self.kind == "huber":
            a = np.abs(u)
            return np.where(a <= c, 0.5 * u * u, c * a - 0.5 * c * c)
        t = np.clip(u / c, -1.0, 1.0)
        return (c * c / 6.0) * (1.0 - (1.0 - t * t) ** 3)

    def psi(self, u: np.ndarray) -> np.ndarray:
        c = self.c
        if self.kind == "huber":
            return np.clip(u, -c, c)
        t = u / c
        return np.where(np.abs(t) <= 1.0, u * (1.0 - t * t) ** 2, 0.0)

    def dpsi(self, u: np.ndarray) -> np.ndarray:
        c = self.c
        if self.kind == "huber":
            return np.asarray(np.abs(u) <= c, dtype=float)
        t2 = (u / c) ** 2
        return np.where(t2 <= 1.0, (1.0 - t2) * (1.0 - 5.0 * t2), 0.0)

    def weight(self, u: np.ndarray) -> np.ndarray:
        c = self.c
        if self.kind == "huber":
            a = np.abs(u)
            return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))
        t2 = (u / c) ** 2
        return np.where(t2 <= 1.0, (1.0 - t2) ** 2, 0.0)

    @property
    def rho_max(self) -> float:
        return self.c * self.c / 6.0  # bisquare only


@lru_cache(maxsize=1)
def _legendre_nodes() -> Tuple[np.ndarray, np.ndarray]:
    nodes, wts = np.polynomial.legendre.leggauss(200)
    return nodes, wts


def _gauss_expect_core(fn: Callable[[np.ndarray], np.ndarray], c: float) -> float:
    """Integral of fn(z) phi(z) over [-c, c] by Gauss-Legendre. The bisquare
    is a polynomial there, so the rule is exact to rounding."""
    nodes, wts = _legendre_nodes()
    z = c * nodes
    dens = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return float(c * np.sum(wts * fn(z) * dens))


def _efficiency(kind: str, c: float) -> float:
    """Asymptotic efficiency at the normal: (E psi')^2 / E psi^2."""
    from scipy.stats import norm

    if kind == "huber":
        inside = 2.0 * norm.cdf(c) - 1.0
        e_psi2 = inside - 2.0 * c * norm.pdf(c) + 2.0 * c * c * norm.sf(c)
        return float(inside**2 / e_psi2)
    p = _Psi(kind, c)
    return _gauss_expect_core(p.dpsi, c) ** 2 / _gauss_expect_core(
        lambda z: p.psi(z) ** 2, c
    )


@lru_cache(maxsize=64)
def _tuning_for_efficiency(kind: str, eff: float) -> float:
    from scipy.optimize import brentq

    return float(brentq(lambda c: _efficiency(kind, c) - eff, 0.05, 30.0, xtol=1e-13))


@lru_cache(maxsize=64)
def _tuning_for_breakdown(bdp: float) -> float:
    """Bisquare constant whose S estimator has breakdown point ``bdp``."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def gap(c: float) -> float:
        p = _Psi("bisquare", c)
        inner = _gauss_expect_core(p.rho, c) / p.rho_max
        return float(inner + 2.0 * norm.sf(c) - bdp)

    return float(brentq(gap, 0.3, 30.0, xtol=1e-13))


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return np.asarray(beta, dtype=float)


def _m_scale(
    r: np.ndarray, chi: _Psi, b: float, s0: Optional[float] = None, p: int = 0
) -> float:
    """The scale ``s`` solving sum(rho(r / s)) / (n - p) = b * rho_max."""
    if s0 is None or not np.isfinite(s0) or s0 <= 0:
        s0 = float(np.median(np.abs(r)) / _MADN)
    if s0 <= 0:
        return 0.0
    s = s0
    target = b * chi.rho_max * (len(r) - p) / len(r)
    for _ in range(500):
        ratio = float(np.mean(chi.rho(r / s))) / target
        s_new = s * np.sqrt(ratio)
        if abs(s_new - s) <= 1e-15 * s:
            s = s_new
            break
        s = s_new
    # polish: the fixed point converges linearly, Newton finishes it
    for _ in range(50):
        u = r / s
        f = float(np.mean(chi.rho(u))) - target
        df = -float(np.mean(chi.psi(u) * u)) / s
        if df == 0:
            break
        step = f / df
        s_new = s - step
        if s_new <= 0:
            break
        done = abs(step) <= 1e-15 * s
        s = s_new
        if done:
            break
    return float(s)


def _independent_rows(
    X: np.ndarray, rng: np.random.Generator, p: int
) -> Optional[np.ndarray]:
    """``p`` linearly independent rows of ``X``, met in random order.

    A plain random ``p``-subset of a design with indicator columns is
    almost always singular (some category is absent). Rows are taken in
    random order and kept when they add a direction, which leaves the draw
    uniform when every ``p``-subset is in general position.
    """
    n = X.shape[0]
    order = rng.permutation(n)
    basis = np.zeros((p, p))
    chosen: List[int] = []
    for idx in order:
        x = X[idx]
        norm = np.linalg.norm(x)
        if norm == 0:
            continue
        resid = x - basis[: len(chosen)].T @ (basis[: len(chosen)] @ x)
        rn = np.linalg.norm(resid)
        if rn > 1e-8 * norm:
            basis[len(chosen)] = resid / rn
            chosen.append(int(idx))
            if len(chosen) == p:
                return np.asarray(chosen)
    return None


def _irls_fixed_scale(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    scale: float,
    psi: _Psi,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, int, bool]:
    for it in range(1, maxiter + 1):
        r = y - X @ beta
        beta_new = _wls(X, y, psi.weight(r / scale))
        change = np.max(np.abs(beta_new - beta)) / max(np.max(np.abs(beta_new)), 1e-12)
        beta = beta_new
        if change < tol:
            return beta, it, True
    return beta, maxiter, False


def _s_estimate(
    X: np.ndarray,
    y: np.ndarray,
    chi: _Psi,
    b: float,
    n_resample: int,
    n_keep: int,
    k_steps: int,
    rng: np.random.Generator,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, float]:
    """Fast-S: random elemental starts, a few refining steps each, the best
    few iterated to convergence."""
    n, p = X.shape

    target = b * chi.rho_max * (n - p)

    def fast_wls(w: np.ndarray) -> np.ndarray:
        # normal equations: several times faster than an SVD, and the
        # screening steps do not need the last digits
        Xw = X * w[:, None]
        try:
            return np.asarray(np.linalg.solve(Xw.T @ X, Xw.T @ y), dtype=float)
        except np.linalg.LinAlgError:
            return _wls(X, y, w)

    def refine(
        beta: np.ndarray, steps: int, to_convergence: bool
    ) -> Tuple[np.ndarray, float]:
        """I-steps of fast-S: one step of the scale iteration, then one
        reweighted least-squares step. Iterated, the pair converges to a
        local minimum of the S scale, at which the scale solves its
        equation exactly."""
        r = y - X @ beta
        s = float(np.median(np.abs(r)) / _MADN)
        for _ in range(steps):
            if s <= 0:
                break
            s = s * np.sqrt(float(np.sum(chi.rho(r / s))) / target)
            w = chi.weight(r / s)
            if np.count_nonzero(w) < p:
                break
            beta_new = fast_wls(w)
            delta = np.max(np.abs(beta_new - beta)) / max(
                np.max(np.abs(beta_new)), 1e-12
            )
            beta = beta_new
            r = y - X @ beta
            if to_convergence and delta < 1e-7:
                break
        return beta, float(s)

    def polish(beta: np.ndarray) -> Tuple[np.ndarray, float]:
        """The same iteration with the scale solved exactly and the
        least-squares step by an orthogonal decomposition."""
        r = y - X @ beta
        s = _m_scale(r, chi, b, None, p)
        for _ in range(maxiter):
            if s <= 0:
                break
            w = chi.weight(r / s)
            if np.count_nonzero(w) < p:
                break
            beta_new = _wls(X, y, w)
            delta = np.max(np.abs(beta_new - beta)) / max(
                np.max(np.abs(beta_new)), 1e-12
            )
            beta = beta_new
            r = y - X @ beta
            s = _m_scale(r, chi, b, s, p)
            if delta < tol:
                break
        return beta, float(s)

    candidates: List[Tuple[float, np.ndarray]] = []
    for _ in range(n_resample):
        rows = _independent_rows(X, rng, p)
        if rows is None:
            raise MethodIncompatibility(
                "robreg: the design matrix is rank deficient; the S estimator "
                "needs linearly independent regressors.",
                recovery_hint="Drop the collinear regressor(s).",
            )
        try:
            beta0 = np.linalg.solve(X[rows], y[rows])
        except np.linalg.LinAlgError:
            continue
        beta1, s1 = refine(beta0, k_steps, False)
        if np.isfinite(s1) and s1 > 0:
            candidates.append((float(s1), beta1))
    if not candidates:
        raise ConvergenceFailure(
            "robreg: no elemental start produced an S candidate.",
            recovery_hint="Increase n_resample or check the data.",
        )
    candidates.sort(key=lambda c: c[0])
    # the most promising starts are taken to their local minimum; only the
    # best of those is worth the exact iteration
    best_s, best_beta = np.inf, candidates[0][1]
    for _, beta_c in candidates[: max(1, n_keep)]:
        beta_f, s_f = refine(beta_c, maxiter, True)
        if 0 < s_f < best_s:
            best_s, best_beta = s_f, beta_f
    best_beta, best_s = polish(best_beta)
    if best_s <= 0:
        raise DataInsufficient(
            "robreg: more than half of the observations lie exactly on a "
            "hyperplane; the S scale is zero.",
            recovery_hint="This is an exact fit; least squares on those rows "
            "describes it.",
        )
    return best_beta, float(best_s)


def _lad(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least absolute deviations by linear programming."""
    from scipy.optimize import linprog

    n, p = X.shape
    cost = np.concatenate([np.zeros(p), np.ones(2 * n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise ConvergenceFailure(f"robreg: the LAD start failed ({res.message}).")
    return np.asarray(res.x[:p], dtype=float)


# ---------------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------------


def _huber_cov(X: np.ndarray, r: np.ndarray, scale: float, psi: _Psi) -> np.ndarray:
    """Huber's (1981, sec. 7.6) covariance of an M estimate:
    ``K^2 * [sum psi^2 / (n - p)] / [mean psi']^2 * s^2 (X'X)^{-1}`` with
    ``K = 1 + (p / n) var(psi') / mean(psi')^2``."""
    n, p = X.shape
    u = r / scale
    d = psi.dpsi(u)
    m = float(np.mean(d))
    K = 1.0 + p * float(np.var(d, ddof=1)) / (n * m * m)
    s2 = scale**2 * float(np.sum(psi.psi(u) ** 2)) / (n - p)
    return np.asarray((K / m) ** 2 * s2 * np.linalg.inv(X.T @ X), dtype=float)


def _stacked_cov(
    G: np.ndarray, scores: np.ndarray, n: int, k: int, small: bool
) -> np.ndarray:
    """Sandwich of a stacked system: G^{-1} (sum g g') G^{-T}, times
    ``n / (n - k)`` when ``small``; ``G`` is the summed Jacobian."""
    Ginv = np.linalg.inv(G)
    factor = n / (n - k) if small else 1.0
    return np.asarray(factor * Ginv @ (scores.T @ scores) @ Ginv.T, dtype=float)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def robreg(
    formula: str,
    data: pd.DataFrame,
    method: str = "mm",
    *,
    psi: Optional[str] = None,
    efficiency: Optional[float] = None,
    tuning: Optional[float] = None,
    init: str = "ls",
    scale: Any = "mad",
    vce: str = "robust",
    small: bool = True,
    breakdown: float = 0.5,
    tuning_s: Optional[float] = None,
    n_resample: int = 500,
    n_keep: int = 25,
    random_state: Optional[int] = 0,
    maxiter: int = 500,
    tol: float = 1e-10,
    alpha: float = 0.05,
) -> EconometricResults:
    """Robust regression by M, S or MM estimation.

    Parameters
    ----------
    formula : str
        Regression formula, e.g. ``"y ~ x1 + x2"``.
    data : pd.DataFrame
    method : {'mm', 'm', 's'}, default 'mm'
        ``'mm'``: S estimate for the scale and start, then a bisquare M
        step at that scale (R ``lmrob``, Stata ``robreg mm``). ``'m'``:
        Huber or bisquare M estimate (R ``MASS::rlm``, Stata ``robreg m``).
        ``'s'``: the S estimate alone.
    psi : {'huber', 'bisquare'}, optional
        The M step's rho function. ``method='m'`` defaults to ``'huber'``;
        MM and S use the bisquare, which is what makes them resistant to
        leverage points.
    efficiency : float, optional
        Efficiency of the M step relative to least squares under normal
        errors, in (0, 1). Default 0.95 for ``'m'`` and 0.85 for ``'mm'``
        (Stata ``robreg``'s defaults; ``lmrob`` and ``rlm`` default to
        0.95 throughout).
    tuning : float, optional
        Set the tuning constant directly instead of through
        ``efficiency``: 1.345 (Huber, 95%), 4.685 (bisquare, 95%), 3.4437
        (bisquare, 85%).
    init : {'ls', 'lad'}, default 'ls'
        ``method='m'`` only: the starting fit. ``'ls'`` is ``rlm``'s,
        ``'lad'`` is ``robreg m``'s.
    scale : {'mad', 'fixed'} or float, default 'mad'
        ``method='m'`` only. ``'mad'`` re-estimates the scale from the
        residuals at every iteration, as ``median(|r|) / 0.6745``
        (``rlm``). ``'fixed'`` keeps the normalised median absolute
        deviation of the starting residuals (``robreg m`` with
        ``init='lad'``). A number fixes the scale at that value.
    vce : {'robust', 'huber'}, default 'robust'
        ``'robust'``: the sandwich of the stacked estimating equations
        (regression, S step and scale together for MM), valid under
        heteroskedastic and asymmetric errors (Stata ``robreg``; ``lmrob``
        with ``small=False``). ``'huber'``: Huber's formula with his finite-sample
        correction, which assumes errors that are symmetric and
        independent of the regressors (``summary.rlm``).
    small : bool, default True
        With ``vce='robust'``: multiply the sandwich by ``n / (n - k)``,
        as ``robreg`` does. ``lmrob`` does not.
    breakdown : float, default 0.5
        Breakdown point of the S step.
    tuning_s : float, optional
        Bisquare constant of the S step. Default: the constant that gives
        ``breakdown`` exactly (1.547645 for 0.5). ``lmrob`` uses the
        rounded 1.54764, which moves its scale in the sixth digit.
    n_resample : int, default 500
        Elemental starts tried by the S search.
    n_keep : int, default 25
        Best candidates iterated to convergence. ``lmrob`` refines 2 and
        ``robreg`` 5; on data with extreme leverage the scale has several
        local minima close to the global one and a few candidates miss it.
    random_state : int or None, default 0
        Seed of the S search. The search is a global optimisation with
        random starts; on data with a clear majority pattern any seed finds
        the same minimum, and ``model_info['s_scale']`` lets you compare.
    maxiter, tol : int, float
        Iteration limit and relative coefficient tolerance.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``model_info['weights']`` holds the robustness weights
        ``psi(r / s) / (r / s)`` in [0, 1], indexed like the rows used: 1
        for an observation treated as in least squares, 0 for one the fit
        ignores. ``model_info['scale']`` is the residual scale; for MM,
        ``model_info['s_coefficients']`` is the S start.

    Notes
    -----
    Use ``method='mm'`` unless there is a reason not to. A monotone M
    estimate has breakdown point zero in the regressors: one observation
    far out in ``x`` can still carry the fit.

    A weight of zero is a statement that the model does not describe that
    observation, not that the observation is wrong. Look at the rows with
    small weights before reporting the fit.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"x": rng.normal(size=500)})
    >>> df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=500)
    >>> df.loc[:24, "y"] += 30          # 5% gross errors
    >>> res = sp.robreg("y ~ x", df)
    >>> bool(abs(res.params["x"] - 0.5) < 0.15)
    True
    >>> bool(res.model_info["weights"].iloc[:25].max() < 1e-6)
    True

    References
    ----------
    huber1964robust, rousseeuw1984robust, yohai1987high, salibian2006fast,
    leone2019influential
    """
    method_key = str(method).lower()
    if method_key not in ("m", "s", "mm"):
        raise MethodIncompatibility(
            f"robreg: method must be 'm', 's' or 'mm', got {method!r}."
        )
    vce_key = str(vce).lower()
    if vce_key not in ("robust", "huber"):
        raise MethodIncompatibility(
            f"robreg: vce must be 'robust' or 'huber', got {vce!r}."
        )
    psi_key = (psi or ("huber" if method_key == "m" else "bisquare")).lower()
    if psi_key in ("biweight", "tukey"):
        psi_key = "bisquare"
    if psi_key not in ("huber", "bisquare"):
        raise MethodIncompatibility(
            f"robreg: psi must be 'huber' or 'bisquare', got {psi!r}."
        )
    if method_key in ("s", "mm") and psi_key != "bisquare":
        raise MethodIncompatibility(
            f"robreg: method={method_key!r} uses the bisquare; a monotone psi "
            "has no breakdown-point guarantee.",
            recovery_hint="Drop psi=, or use method='m' for a Huber fit.",
        )
    if efficiency is not None and tuning is not None:
        raise MethodIncompatibility("robreg: give efficiency= or tuning=, not both.")
    if efficiency is not None and not 0.0 < efficiency < 1.0:
        raise MethodIncompatibility(
            f"robreg: efficiency must lie in (0, 1), got {efficiency!r}."
        )
    if not 0.0 < breakdown <= 0.5:
        raise MethodIncompatibility(
            f"robreg: breakdown must lie in (0, 0.5], got {breakdown!r}."
        )
    if method_key != "m" and (init != "ls" or scale != "mad"):
        raise MethodIncompatibility(
            "robreg: init= and scale= belong to method='m'; MM and S take "
            "their start and scale from the S search."
        )
    init_key = str(init).lower()
    if init_key not in ("ls", "lad"):
        raise MethodIncompatibility(
            f"robreg: init must be 'ls' or 'lad', got {init!r}."
        )

    y_df, X_df = create_design_matrices(formula, data)
    names = [str(c) for c in X_df.columns]
    index = X_df.index
    X = np.asarray(X_df, dtype=float)
    y = np.asarray(y_df, dtype=float).ravel()
    n, p = X.shape
    if n <= p:
        raise DataInsufficient(
            f"robreg: {n} observations for {p} coefficients.",
            recovery_hint="Robust regression needs clearly more rows than "
            "regressors.",
        )
    if np.linalg.matrix_rank(X) < p:
        raise MethodIncompatibility(
            "robreg: the regressors are collinear.",
            recovery_hint="Drop the redundant regressor(s).",
        )

    info: Dict[str, Any] = {}
    if method_key == "m":
        eff = 0.95 if (efficiency is None and tuning is None) else efficiency
        c = (
            float(tuning)
            if tuning is not None
            else _tuning_for_efficiency(psi_key, float(eff))  # type: ignore[arg-type]
        )
        psi_fn = _Psi(psi_key, c)
        beta = (
            _lad(X, y)
            if init_key == "lad"
            else np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)
        )
        fixed: Optional[float]
        if isinstance(scale, str):
            scale_key = scale.lower()
            if scale_key not in ("mad", "fixed"):
                raise MethodIncompatibility(
                    f"robreg: scale must be 'mad', 'fixed' or a number, got "
                    f"{scale!r}."
                )
            r0 = y - X @ beta
            if scale_key == "fixed":
                # an L1 fit interpolates p points; their zero residuals say
                # nothing about the spread
                a0 = np.abs(r0)
                if init_key == "lad":
                    a0 = np.sort(a0)[p:]
                fixed = float(np.median(a0) / _MADN)
            else:
                fixed = None
        else:
            fixed = float(scale)
            if not np.isfinite(fixed) or fixed <= 0:
                raise MethodIncompatibility(
                    f"robreg: scale must be positive, got {scale!r}."
                )
        converged = False
        s = fixed if fixed is not None else 0.0
        it = 0
        for it in range(1, maxiter + 1):
            r = y - X @ beta
            if fixed is None:
                s = float(np.median(np.abs(r)) / _MAD_ROUNDED)
            if s <= 0:
                raise DataInsufficient(
                    "robreg: the residual scale is zero; at least half of "
                    "the observations are fitted exactly."
                )
            beta_new = _wls(X, y, psi_fn.weight(r / s))
            change = np.max(np.abs(beta_new - beta)) / max(
                np.max(np.abs(beta_new)), 1e-12
            )
            beta = beta_new
            if change < tol:
                converged = True
                break
        r = y - X @ beta
        if fixed is None:
            s = float(np.median(np.abs(r)) / _MAD_ROUNDED)
        sigma = float(s)
        info.update(iterations=it, converged=converged, init=init_key)
        u = r / sigma
        if vce_key == "huber":
            cov = _huber_cov(X, r, sigma, psi_fn)
        else:
            G = (X * psi_fn.dpsi(u)[:, None]).T @ X / sigma
            cov = _stacked_cov(G, X * psi_fn.psi(u)[:, None], n, p, small)
        eff_out = _efficiency(psi_key, c)
    else:
        c_s = (
            float(tuning_s)
            if tuning_s is not None
            else _tuning_for_breakdown(breakdown)
        )
        chi = _Psi("bisquare", c_s)
        rng = np.random.default_rng(random_state)
        beta_s, sigma = _s_estimate(
            X, y, chi, breakdown, n_resample, n_keep, 2, rng, maxiter, tol
        )
        info.update(
            s_coefficients=pd.Series(beta_s, index=names),
            s_scale=sigma,
            tuning_s=c_s,
            breakdown=breakdown,
            n_resample=n_resample,
        )
        r_s = y - X @ beta_s
        u0 = r_s / sigma
        # stacked equations of the S step: rho0'(u0) x = 0, rho0(u0) = b rho_max
        g_s = X * chi.psi(u0)[:, None]
        g_sig = chi.rho(u0) - breakdown * chi.rho_max
        d0 = chi.dpsi(u0)
        G_ss = (X * d0[:, None]).T @ X / sigma
        G_s_sig = X.T @ (d0 * u0) / sigma
        G_sig_s = X.T @ chi.psi(u0) / sigma
        G_sig_sig = float(np.sum(chi.psi(u0) * u0)) / sigma
        if method_key == "s":
            beta, r = beta_s, r_s
            psi_fn, c = chi, c_s
            converged, it = True, 0
            G = np.zeros((p + 1, p + 1))
            G[:p, :p], G[:p, p] = G_ss, G_s_sig
            G[p, :p], G[p, p] = G_sig_s, G_sig_sig
            scores = np.column_stack([g_s, g_sig])
            full = _stacked_cov(G, scores, n, p, small)
            cov = full[:p, :p]
            info["scale_se"] = float(np.sqrt(full[p, p]))
            if vce_key == "huber":
                cov = _huber_cov(X, r, sigma, psi_fn)
        else:
            eff = 0.85 if (efficiency is None and tuning is None) else efficiency
            c = (
                float(tuning)
                if tuning is not None
                else _tuning_for_efficiency(
                    "bisquare", float(eff)  # type: ignore[arg-type]
                )
            )
            psi_fn = _Psi("bisquare", c)
            beta, it, converged = _irls_fixed_scale(
                X, y, beta_s.copy(), sigma, psi_fn, maxiter, tol
            )
            r = y - X @ beta
            u = r / sigma
            if vce_key == "huber":
                cov = _huber_cov(X, r, sigma, psi_fn)
            else:
                # order: MM coefficients, S coefficients, scale
                q = 2 * p + 1
                G = np.zeros((q, q))
                d = psi_fn.dpsi(u)
                G[:p, :p] = (X * d[:, None]).T @ X / sigma
                G[:p, 2 * p] = X.T @ (d * u) / sigma
                G[p : 2 * p, p : 2 * p] = G_ss
                G[p : 2 * p, 2 * p] = G_s_sig
                G[2 * p, p : 2 * p] = G_sig_s
                G[2 * p, 2 * p] = G_sig_sig
                scores = np.column_stack([X * psi_fn.psi(u)[:, None], g_s, g_sig])
                full = _stacked_cov(G, scores, n, p, small)
                cov = full[:p, :p]
                info["scale_se"] = float(np.sqrt(full[2 * p, 2 * p]))
        info.update(iterations=it, converged=converged)
        eff_out = _efficiency("bisquare", c)

    if not converged:
        warnings.warn(
            f"robreg: the M step did not converge in {maxiter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )
    weights = pd.Series(psi_fn.weight(r / sigma), index=index, name="weight")
    se = np.sqrt(np.diag(cov))
    label = {"m": "M", "s": "S", "mm": "MM"}[method_key]
    model_info: Dict[str, Any] = {
        "model_type": f"Robust regression ({label})",
        "method": f"{label} estimator ({psi_fn.kind}, "
        f"{100 * eff_out:.1f}% efficiency)",
        "estimator": method_key,
        "psi": psi_fn.kind,
        "tuning": float(c),
        "efficiency": float(eff_out),
        "scale": sigma,
        "vce": vce_key,
        "robust": vce_key,
        "weights": weights,
        "n_downweighted": int((weights < 0.5).sum()),
        "n_zero_weight": int((weights == 0).sum()),
        "vcov": pd.DataFrame(cov, index=names, columns=names),
        "formula": formula,
        "alpha": alpha,
        **info,
    }
    fitted = X @ beta
    data_info: Dict[str, Any] = {
        "nobs": int(n),
        "dependent_var": str(y_df.columns[-1]) if hasattr(y_df, "columns") else "y",
        "df_resid": int(n - p),
        "df_model": p - 1 if "Intercept" in names else p,
        "var_cov": cov,
        "fitted_values": fitted,
        "residuals": r,
    }
    diagnostics: Dict[str, Any] = {
        "Scale": sigma,
        "Tuning constant": float(c),
        "Efficiency": float(eff_out),
        "Weights below 0.5": model_info["n_downweighted"],
        "Weights equal to 0": model_info["n_zero_weight"],
    }
    return EconometricResults(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
