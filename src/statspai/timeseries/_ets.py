"""Exponential smoothing in its innovations state space form (ETS).

``sp.ets`` fits one model of the error / trend / seasonal taxonomy by
maximum likelihood, or selects among them by AICc, and forecasts with
prediction intervals. The likelihood, the admissible parameter region,
the model search and the forecast variances follow Hyndman, Koehler,
Snyder and Grose (2002) and Hyndman, Koehler, Ord and Snyder (2008), the
framework behind R's ``forecast::ets`` and ``fable::ETS``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from . import _ets_core as core
from ._forecast_common import (
    Levels,
    check_period,
    classical_decomposition,
    forecast_frame,
    future_index,
    normalise_levels,
    path_quantiles,
    read_series,
)

_CODE = {"N": 0, "A": 1, "M": 2}
_LOWER = np.array([1e-4, 1e-4, 1e-4, 0.80])
_UPPER = np.array([0.9999, 0.9999, 0.9999, 0.98])
_SMOOTH = ("alpha", "beta", "gamma", "phi")


def _model_name(error: str, trend: str, season: str, damped: bool) -> str:
    tr = trend + ("d" if damped and trend != "N" else "")
    return f"ETS({error},{tr},{season})"


@dataclass
class ETSResult(ResultProtocolMixin):
    """Fitted ETS model returned by :func:`statspai.ets`.

    Attributes
    ----------
    model : str
        ``"ETS(M,Ad,M)"``-style name: error, trend (``d`` when damped) and
        seasonal component, each ``N`` none, ``A`` additive or ``M``
        multiplicative.
    params : pd.Series
        Smoothing parameters ``alpha``, ``beta``, ``gamma``, ``phi`` (those
        the model has). ``beta`` and ``gamma`` are on the state space
        scale: the trend equation is ``b_t = phi b_{t-1} + beta e_t``.
    initial_state : pd.Series
        ``l`` (level), ``b`` (growth) and the seasonal terms ``s0, s-1,
        ...`` at time zero.
    states : pd.DataFrame
        One row per period from time zero: ``level``, ``slope`` and
        ``season`` (the seasonal term that enters the *next* forecast is
        ``season_lag<m-1>``).
    fitted_values, residuals : np.ndarray
        One-step forecasts and innovation residuals. Under a
        multiplicative error the innovation is the relative error
        ``(y - fitted) / fitted``; ``response_residuals`` is ``y - fitted``.
    sigma2 : float
        Innovation variance, ``sum(e^2) / (n - k)`` with ``k`` the number
        of estimated smoothing parameters and initial states.
    log_likelihood, aic, aicc, bic : float
        The likelihood drops the same constant as Hyndman et al. (2008),
        so the criteria are comparable across ETS models and with R's
        ``ets``, not with an ARIMA likelihood.
    candidates : pd.DataFrame or None
        Every model tried by the automatic search, with its criteria.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = 50 + np.cumsum(rng.normal(0.5, 1.0, 80))
    >>> res = sp.ets(y, model="AAN")
    >>> res.model
    'ETS(A,A,N)'
    >>> list(res.forecast(3).columns)
    ['forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "hyndman2002state",
        "hyndman2008forecasting",
    )

    model: str
    error: str
    trend: str
    season: str
    damped: bool
    period: int
    params: pd.Series
    initial_state: pd.Series
    states: pd.DataFrame
    fitted_values: np.ndarray
    residuals: np.ndarray
    sigma2: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    mse: float
    n: int
    n_params: int
    converged: bool = True
    candidates: Optional[pd.DataFrame] = None
    fixed: Dict[str, float] = field(default_factory=dict)
    _y: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    _index: Optional[pd.Index] = field(default=None, repr=False)
    _name: str = "y"

    # ------------------------------------------------------------------
    @property
    def response_residuals(self) -> np.ndarray:
        """``y - fitted_values``, on the scale of the data."""
        return np.asarray(self._y - self.fitted_values, dtype=float)

    @property
    def n_arma_params(self) -> int:
        """Degrees of freedom a portmanteau test should give up (none:
        the convention of Hyndman and Athanasopoulos for ETS residuals)."""
        return 0

    def _codes(self) -> Tuple[int, int, int]:
        return int(self.error == "M"), _CODE[self.trend], _CODE[self.season]

    def _smoothing(self) -> Tuple[float, float, float, float]:
        p = self.params
        return (
            float(p["alpha"]),
            float(p.get("beta", 0.0)),
            float(p.get("gamma", 0.0)),
            float(p.get("phi", 1.0)),
        )

    def _last_state(self) -> np.ndarray:
        row = self.states.iloc[-1].to_numpy(dtype=float)
        m = self.period if self.season != "N" else 0
        out = np.zeros(2 + max(m, 0))
        out[0] = row[0]
        k = 1
        if self.trend != "N":
            out[1] = row[1]
            k = 2
        if m:
            out[2 : 2 + m] = row[k : k + m]
        return out

    def forecast(
        self,
        horizon: int = 10,
        level: Levels = (80, 95),
        *,
        simulate: bool = False,
        bootstrap: bool = False,
        n_paths: int = 5000,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Point forecasts and prediction intervals.

        Parameters
        ----------
        horizon : int, default 10
            Number of periods ahead.
        level : float or sequence of float, default (80, 95)
            Coverage of the intervals, in percent.
        simulate : bool, default False
            Take the intervals from simulated future paths instead of the
            analytic variance. Models with a multiplicative trend, or an
            additive error with a multiplicative season, have no analytic
            variance and are always simulated.
        bootstrap : bool, default False
            Draw the future innovations from the residuals instead of a
            normal distribution (implies ``simulate``).
        n_paths : int, default 5000
        seed : int or None, default 0
            Seed of the simulation; the default makes the intervals
            reproducible.

        Returns
        -------
        pd.DataFrame
            ``forecast`` plus ``lower_<level>`` / ``upper_<level>`` for
            every level, indexed by the forecast periods.

        Notes
        -----
        The intervals treat the estimated parameters as known. For a
        multiplicative error with a multiplicative season the point
        forecast is the forecast mean of Hyndman et al. (2008, section
        6.4.4), which is slightly above the forecast with future
        innovations at zero.
        """
        h = int(horizon)
        if h < 1:
            raise MethodIncompatibility(
                f"forecast: horizon must be at least 1, got {horizon!r}.",
                recovery_hint="Pass a positive number of periods.",
            )
        levels = normalise_levels(level)
        err, tr, se = self._codes()
        alpha, beta, gamma, phi = self._smoothing()
        m = self.period if se else 1
        last = self._last_state()
        mean = core.point_forecast(h, m, tr, se, phi, last)
        idx = future_index(self._index, self.n, h)
        var: Optional[np.ndarray] = None
        if not (simulate or bootstrap):
            if err == 0 and tr < 2 and se < 2:
                var = core.class1_variance(
                    h, m, tr, se, alpha, beta, gamma, phi, self.sigma2
                )
            elif err == 1 and tr < 2 and se < 2:
                var = core.class2_variance(
                    mean, m, tr, se, alpha, beta, gamma, phi, self.sigma2
                )
            elif err == 1 and tr < 2 and se == 2:
                mean, var = core.class3_moments(
                    h, m, tr, alpha, beta, gamma, phi, self.sigma2, last
                )
        if var is not None:
            return forecast_frame(mean, levels, sd=np.sqrt(var), index=idx)
        rng = np.random.default_rng(seed)
        if bootstrap:
            pool = self.residuals - self.residuals.mean()
            innov = rng.choice(pool, size=(int(n_paths), h), replace=True)
        else:
            innov = rng.normal(0.0, np.sqrt(self.sigma2), size=(int(n_paths), h))
        paths = core.simulate_paths(
            h, m, err, tr, se, alpha, beta, gamma, phi, last, innov
        )
        return forecast_frame(
            mean, levels, bounds=path_quantiles(paths, levels), index=idx
        )

    def simulate(
        self, horizon: int = 10, n_paths: int = 5, seed: Optional[int] = 0
    ) -> pd.DataFrame:
        """Future sample paths (one column each) with normal innovations."""
        h = int(horizon)
        err, tr, se = self._codes()
        alpha, beta, gamma, phi = self._smoothing()
        m = self.period if se else 1
        rng = np.random.default_rng(seed)
        innov = rng.normal(0.0, np.sqrt(self.sigma2), size=(int(n_paths), h))
        paths = core.simulate_paths(
            h, m, err, tr, se, alpha, beta, gamma, phi, self._last_state(), innov
        )
        return pd.DataFrame(
            paths.T,
            index=future_index(self._index, self.n, h),
            columns=[f"path_{i}" for i in range(int(n_paths))],
        )

    def components(self) -> pd.DataFrame:
        """The series with its level, slope, season and remainder at each
        period (the decomposition a fitted ETS model implies)."""
        st = self.states.iloc[1:]
        out = pd.DataFrame({self._name: self._y})
        out["level"] = st["level"].to_numpy()
        if "slope" in st:
            out["slope"] = st["slope"].to_numpy()
        if "season" in st:
            out["season"] = st["season"].to_numpy()
        out["remainder"] = self.residuals
        if self._index is not None:
            out.index = self._index
        return out

    def summary(self) -> str:
        lines = [
            self.model + (f"  (period {self.period})" if self.season != "N" else ""),
            "-" * 46,
            f"n          : {self.n}",
            f"sigma^2    : {self.sigma2:.6g}",
            f"Log-Lik    : {self.log_likelihood:.4f}",
            f"AIC        : {self.aic:.4f}",
            f"AICc       : {self.aicc:.4f}",
            f"BIC        : {self.bic:.4f}",
            "",
            "Smoothing parameters:",
        ]
        for nm, val in self.params.items():
            tag = "  (fixed)" if nm in self.fixed else ""
            lines.append(f"  {nm:<8s} {val:>12.4f}{tag}")
        lines.append("")
        lines.append("Initial states:")
        init = self.initial_state
        head = [k for k in init.index if not k.startswith("s")]
        for nm in head:
            lines.append(f"  {nm:<8s} {init[nm]:>12.4f}")
        seas = [k for k in init.index if k.startswith("s")]
        if seas:
            vals = "  ".join(f"{init[k]:.4f}" for k in seas)
            lines.append(f"  s        {vals}")
        if not self.converged:
            lines.append("")
            lines.append("Warning: the optimiser did not report convergence.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def plot(
        self,
        horizon: int = 10,
        level: Levels = (80, 95),
        ax: Optional[Any] = None,
    ) -> Any:
        """Observed series, fitted values and forecasts with intervals."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        levels = normalise_levels(level)
        fc = self.forecast(horizon, levels)
        x = np.arange(self.n) if self._index is None else self._index
        xf = fc.index
        if self._index is not None and isinstance(self._index, pd.PeriodIndex):
            x = self._index.to_timestamp()
            xf = fc.index.to_timestamp()
        ax.plot(x, self._y, color="black", linewidth=1.0, label="observed")
        ax.plot(x, self.fitted_values, color="C0", linewidth=0.8, label="fitted")
        ax.plot(xf, fc["forecast"], color="C3", label="forecast")
        for pct in levels[::-1]:
            lab = f"{pct:g}"
            ax.fill_between(
                xf, fc[f"lower_{lab}"], fc[f"upper_{lab}"], color="C3", alpha=0.15
            )
        ax.set_title(self.model)
        ax.legend()
        return ax


# ----------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------
def _initial_states(
    y: np.ndarray, m: int, trend: str, season: str
) -> Tuple[float, Optional[float], Optional[np.ndarray]]:
    """Starting values of the initial states: a classical decomposition
    for the seasonal terms, a straight line through the first seasonally
    adjusted observations for level and growth."""
    n = y.shape[0]
    seas: Optional[np.ndarray] = None
    y_sa = y
    if season != "N":
        mult = season == "M"
        if n < 3 * m:
            t = np.arange(1, n + 1, dtype=float)
            X = np.column_stack(
                [
                    np.ones(n),
                    t,
                    np.sin(2 * np.pi * t / m),
                    np.cos(2 * np.pi * t / m),
                ]
            )
            if m == 2:
                X = X[:, :3]
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            line = coef[0] + coef[1] * t
            with np.errstate(divide="ignore", invalid="ignore"):
                seasonal = y / line if mult else y - line
        else:
            seasonal = classical_decomposition(y, m, multiplicative=mult)[1]
        seas = seasonal[1:m][::-1].copy()
        if mult:
            seas = np.maximum(seas, 1e-2)
            if seas.sum() > m:
                seas = seas / np.sum(seas + 1e-2)
            y_sa = y / np.maximum(seasonal, 1e-2)
        else:
            y_sa = y - seasonal
    maxn = min(max(10, 2 * m), n)
    if trend == "N":
        return float(np.mean(y_sa[:maxn])), None, seas
    t = np.arange(1, maxn + 1, dtype=float)
    slope, intercept = np.polyfit(t, y_sa[:maxn], 1)
    if trend == "A":
        l0, b0 = float(intercept), float(slope)
        if abs(l0 + b0) < 1e-8:
            l0 *= 1 + 1e-3
            b0 *= 1 - 1e-3
        return l0, b0, seas
    l0 = float(intercept + slope)
    if abs(l0) < 1e-8:
        l0 = 1e-7
    b0 = float((intercept + 2 * slope) / l0)
    l0 = l0 / b0
    if abs(b0) > 1e10:
        b0 = float(np.sign(b0) * 1e10)
    if l0 < 1e-8 or b0 < 1e-8:
        l0 = max(float(y_sa[0]), 1e-3)
        b0 = max(float(y_sa[1] / y_sa[0]), 1e-3)
    return l0, b0, seas


class _Spec:
    """One model of the taxonomy on one series: the map from the free
    parameter vector to the likelihood."""

    def __init__(
        self,
        y: np.ndarray,
        m: int,
        error: str,
        trend: str,
        season: str,
        damped: bool,
        fixed: Dict[str, float],
        bounds: str,
    ) -> None:
        self.y = y
        self.n = y.shape[0]
        self.error, self.trend, self.season = error, trend, season
        self.damped = damped and trend != "N"
        self.m = m if season != "N" else 1
        self.bounds = bounds
        self.fixed = dict(fixed)
        self.has = {
            "alpha": True,
            "beta": trend != "N",
            "gamma": season != "N",
            "phi": self.damped,
        }
        for nm in fixed:
            if not self.has.get(nm, False):
                label = _model_name(error, trend, season, damped)
                raise MethodIncompatibility(
                    f"ets: {nm}= was given but {label} has no {nm}.",
                    recovery_hint="Drop the argument or change the model.",
                )
        self.free = [nm for nm in _SMOOTH if self.has[nm] and nm not in fixed]
        self.n_seas = self.m - 1 if season != "N" else 0
        self.n_state = 1 + int(trend != "N") + self.n_seas
        self.k = len(self.free) + self.n_state
        ncol = 2 + (self.m if season != "N" else 0)
        self._states = np.zeros((self.n + 1, ncol))
        self._fit = np.zeros(self.n)
        self._res = np.zeros(self.n)
        self.codes = (int(error == "M"), _CODE[trend], _CODE[season])
        self._pos = np.array(
            [self.free.index(nm) if nm in self.free else -1 for nm in _SMOOTH],
            dtype=np.int64,
        )
        self._fixed_arr = np.array(
            [float(self.fixed.get(nm, 1.0 if nm == "phi" else 0.0)) for nm in _SMOOTH]
        )
        self._has_arr = np.array([self.has[nm] for nm in _SMOOTH], dtype=np.bool_)
        self._bounds_code = {"both": 0, "usual": 1, "admissible": 2}[bounds]

    # -- parameter vector <-> named pieces
    def unpack(
        self, x: np.ndarray
    ) -> Tuple[Dict[str, Optional[float]], float, float, np.ndarray]:
        par: Dict[str, Optional[float]] = {nm: None for nm in _SMOOTH}
        i = 0
        for nm in self.free:
            par[nm] = float(x[i])
            i += 1
        for nm, val in self.fixed.items():
            par[nm] = float(val)
        l0 = float(x[i])
        i += 1
        b0 = 0.0
        if self.trend != "N":
            b0 = float(x[i])
            i += 1
        if self.season != "N":
            head = np.asarray(x[i : i + self.n_seas], dtype=float)
            last = (self.m - head.sum()) if self.season == "M" else -head.sum()
            s0 = np.append(head, last)
        else:
            s0 = np.zeros(1)
        return par, l0, b0, s0

    def run(self, x: np.ndarray) -> Tuple[float, float, int]:
        par, l0, b0, s0 = self.unpack(x)
        return self._run(par, l0, b0, s0)

    def _run(
        self,
        par: Dict[str, Optional[float]],
        l0: float,
        b0: float,
        s0: np.ndarray,
    ) -> Tuple[float, float, int]:
        err, tr, se = self.codes
        sse, sumlog, bad = core.ets_filter(
            self.y,
            self.m,
            err,
            tr,
            se,
            float(par["alpha"] or 0.0),
            float(par["beta"] or 0.0),
            float(par["gamma"] or 0.0),
            float(par["phi"]) if par["phi"] is not None else 1.0,
            l0,
            b0,
            s0,
            self._states,
            self._fit,
            self._res,
        )
        return float(sse), float(sumlog), int(bad)

    def objective(self, x: np.ndarray) -> float:
        return float(
            core.ets_objective(
                np.ascontiguousarray(x, dtype=float),
                self.y,
                self.m,
                self.codes[0],
                self.codes[1],
                self.codes[2],
                self._pos,
                self._fixed_arr,
                self._has_arr,
                self._bounds_code,
                _LOWER,
                _UPPER,
                self._states,
                self._fit,
                self._res,
            )
        )

    def start(self) -> np.ndarray:
        lo, up = _LOWER.copy(), _UPPER.copy()
        if self.bounds == "admissible":
            lo[:3] = 0.0
            up[:3] = 1e-3
        vals: Dict[str, float] = {}
        alpha = self.fixed.get("alpha")
        if alpha is None:
            alpha = lo[0] + 0.2 * (up[0] - lo[0]) / self.m
            if alpha > 1 or alpha < 0:
                alpha = lo[0] + 2e-3
        vals["alpha"] = float(alpha)
        if self.has["beta"]:
            beta = self.fixed.get("beta")
            if beta is None:
                ub = min(up[1], alpha)
                beta = lo[1] + 0.1 * (ub - lo[1])
                if beta < 0 or beta > alpha:
                    beta = alpha - 1e-3
            vals["beta"] = float(beta)
        if self.has["gamma"]:
            gamma = self.fixed.get("gamma")
            if gamma is None:
                ug = min(up[2], 1 - alpha)
                gamma = lo[2] + 0.05 * (ug - lo[2])
                if gamma < 0 or gamma > 1 - alpha:
                    gamma = 1 - alpha - 1e-3
            vals["gamma"] = float(gamma)
        if self.has["phi"]:
            phi = self.fixed.get("phi")
            if phi is None:
                phi = lo[3] + 0.99 * (up[3] - lo[3])
            vals["phi"] = float(phi)
        l0, b0, seas = _initial_states(self.y, self.m, self.trend, self.season)
        x = [vals[nm] for nm in self.free] + [l0]
        if b0 is not None:
            x.append(b0)
        if seas is not None:
            x.extend(seas.tolist())
        return np.asarray(x, dtype=float)


def _simplex(spec: _Spec, x: np.ndarray, shrink: float) -> np.ndarray:
    d = x.shape[0]
    scale = max(float(np.std(spec.y)), 1e-8)
    n_smooth = len(spec.free)
    simplex = np.tile(x, (d + 1, 1))
    for i in range(d):
        if i < n_smooth:
            step = 0.05 * shrink
            if x[i] + step > 0.97:
                step = -step
        else:
            step = shrink * max(0.1 * abs(x[i]), 0.02 * scale)
            if spec.season == "M" and i >= d - spec.n_seas:
                step = shrink * 0.05
            if spec.trend == "M" and i == n_smooth + 1:
                step = shrink * 0.02
        simplex[i + 1, i] += step
    return simplex


def _polish(spec: _Spec, x: np.ndarray, fx: float) -> Tuple[np.ndarray, float]:
    """Quasi-Newton refinement inside the 'usual' box.

    The smoothing parameters are mapped to a box (``beta`` as a share of
    ``alpha``, ``gamma`` as a share of ``1 - alpha``) so that bound
    constraints replace the inequalities between them; a simplex search
    alone stalls when a seasonal model has a dozen initial states.
    """
    if spec.bounds == "admissible":
        return x, fx
    k = len(spec.free)
    lo, up = _LOWER, _UPPER
    fixed_alpha = spec.fixed.get("alpha")

    def to_box(v: np.ndarray) -> np.ndarray:
        u = v.copy()
        a = fixed_alpha if fixed_alpha is not None else v[spec.free.index("alpha")]
        for i, nm in enumerate(spec.free):
            if nm == "beta":
                top = min(up[1], a)
                u[i] = (v[i] - lo[1]) / max(top - lo[1], 1e-12)
            elif nm == "gamma":
                top = min(up[2], 1.0 - a)
                u[i] = (v[i] - lo[2]) / max(top - lo[2], 1e-12)
        return u

    def from_box(u: np.ndarray) -> np.ndarray:
        v = u.copy()
        a = fixed_alpha if fixed_alpha is not None else u[spec.free.index("alpha")]
        for i, nm in enumerate(spec.free):
            if nm == "beta":
                top = min(up[1], a)
                v[i] = lo[1] + u[i] * (top - lo[1])
            elif nm == "gamma":
                top = min(up[2], 1.0 - a)
                v[i] = lo[2] + u[i] * (top - lo[2])
        return v

    box: List[Tuple[Optional[float], Optional[float]]] = []
    for nm in spec.free:
        if nm == "alpha":
            box.append((lo[0], up[0]))
        elif nm == "phi":
            box.append((lo[3], up[3]))
        else:
            box.append((0.0, 1.0))
    box += [(None, None)] * (x.shape[0] - k)
    u0 = to_box(x)
    box_lo = np.array([float(b[0] or 0.0) for b in box[:k]])
    box_hi = np.array([float(b[1] or 0.0) for b in box[:k]])
    u0[:k] = np.clip(u0[:k], box_lo, box_hi)
    spread = max(float(np.std(spec.y)), 1e-8)
    scale = np.maximum(np.abs(u0), 0.1 * spread)
    scale[:k] = 1.0
    if spec.season == "M":
        scale[x.shape[0] - spec.n_seas :] = 1.0
    if spec.trend == "M":
        scale[k + 1] = 1.0

    def fun(z: np.ndarray) -> float:
        return spec.objective(from_box(z * scale))

    zbox = [
        (None if b[0] is None else b[0] / sc, None if b[1] is None else b[1] / sc)
        for b, sc in zip(box, scale)
    ]
    try:
        res = optimize.minimize(
            fun,
            u0 / scale,
            method="L-BFGS-B",
            bounds=zbox,
            options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9, "eps": 1e-7},
        )
    except (ValueError, FloatingPointError):
        return x, fx
    if np.isfinite(res.fun) and res.fun < fx:
        return from_box(np.asarray(res.x) * scale), float(res.fun)
    return x, fx


def _optimise(
    spec: _Spec, x0: np.ndarray, maxiter: int
) -> Tuple[np.ndarray, float, bool]:
    """Quasi-Newton search inside the box, then a simplex search from its
    optimum, repeated until the criterion stops improving.

    The likelihood of a seasonal model is nearly flat along directions
    that trade the initial seasonal states against each other, so the
    stopping rule is on the criterion (``1e-4`` in minus twice the log
    likelihood), not on the parameters.
    """
    best_x = np.asarray(x0, dtype=float)
    best_f = spec.objective(best_x)
    converged = False
    d = best_x.shape[0]
    budget = int(min(maxiter, 250 * d))
    for attempt in range(8):
        before = best_f
        best_x, best_f = _polish(spec, best_x, best_f)
        res = optimize.minimize(
            spec.objective,
            best_x,
            method="Nelder-Mead",
            options={
                "initial_simplex": _simplex(spec, best_x, 0.5**attempt),
                "maxiter": budget,
                "maxfev": budget,
                "xatol": 1e-8,
                "fatol": 1e-9,
                "adaptive": d > 8,
            },
        )
        if res.fun < best_f:
            best_x, best_f = np.asarray(res.x, dtype=float), float(res.fun)
        if attempt > 0 and before - best_f < 1e-4:
            converged = True
            break
    return best_x, best_f, converged


def _fit_one(
    y: np.ndarray,
    m: int,
    error: str,
    trend: str,
    season: str,
    damped: bool,
    fixed: Dict[str, float],
    bounds: str,
    maxiter: int,
) -> Optional[Dict[str, Any]]:
    spec = _Spec(y, m, error, trend, season, damped, fixed, bounds)
    n = spec.n
    if n <= spec.k + 1:
        return None
    x0 = spec.start()
    if spec.objective(x0) >= core.BAD:
        # the textbook starting values can sit outside the region; pull
        # the smoothing parameters to its interior before giving up
        x0[: len(spec.free)] = np.clip(x0[: len(spec.free)], 0.01, 0.2)
        for i, nm in enumerate(spec.free):
            if nm == "phi":
                x0[i] = 0.9
            elif nm == "beta":
                x0[i] = 0.01
            elif nm == "gamma":
                x0[i] = 0.01
        if spec.objective(x0) >= core.BAD:
            return None
    x, fval, conv = _optimise(spec, x0, maxiter)
    if fval >= core.BAD:
        return None
    par, l0, b0, s0 = spec.unpack(x)
    sse, sumlog, _ = spec._run(par, l0, b0, s0)
    np_ = spec.k + 1
    aic = fval + 2 * np_
    denom = n - np_ - 1
    aicc = aic + 2 * np_ * (np_ + 1) / denom if denom > 0 else np.inf
    bic = fval + np.log(n) * np_
    return {
        "spec": spec,
        "x": x,
        "par": par,
        "l0": l0,
        "b0": b0,
        "s0": s0,
        "states": spec._states.copy(),
        "fitted": spec._fit.copy(),
        "resid": spec._res.copy(),
        "sse": sse,
        "neg2ll": fval,
        "aic": float(aic),
        "aicc": float(aicc),
        "bic": float(bic),
        "converged": conv,
    }


def _parse_model(model: str) -> Tuple[str, str, str, Optional[bool]]:
    txt = str(model).strip().upper().replace(" ", "")
    if txt.startswith("ETS(") and txt.endswith(")"):
        txt = txt[4:-1].replace(",", "")
    damped: Optional[bool] = None
    if len(txt) == 4 and txt[2] == "D":
        damped = True
        txt = txt[:2] + txt[3]
    if len(txt) != 3 or any(c not in ok for c, ok in zip(txt, ("AMZ", "NAMZ", "NAMZ"))):
        raise MethodIncompatibility(
            f"ets: model={model!r} is not an ETS specification.",
            recovery_hint=(
                "Three letters, error / trend / season: e.g. 'ANN' (simple "
                "exponential smoothing), 'AAN' (Holt), 'AAdN' (damped), "
                "'MAM' (Holt-Winters multiplicative), 'ZZZ' (choose by AICc)."
            ),
        )
    return txt[0], txt[1], txt[2], damped


def ets(
    y: Any,
    model: str = "ZZZ",
    *,
    period: int = 1,
    damped: Optional[bool] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    phi: Optional[float] = None,
    ic: str = "aicc",
    additive_only: bool = False,
    allow_multiplicative_trend: bool = False,
    restrict: bool = True,
    bounds: str = "both",
    maxiter: int = 2000,
    data: Optional[pd.DataFrame] = None,
) -> ETSResult:
    """Exponential smoothing state space model (ETS), fitted by maximum
    likelihood or chosen by information criterion.

    Covers simple exponential smoothing, Holt's linear and damped trend
    methods and the Holt-Winters seasonal methods, each with an additive
    or a multiplicative error, as models with prediction intervals.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series, in time order; a column name when ``data`` is given.
        It must have no missing values between its first and last
        observation.
    model : str, default "ZZZ"
        Error, trend and seasonal component: ``N`` none, ``A`` additive,
        ``M`` multiplicative, ``Z`` chosen automatically. ``"ANN"`` is
        simple exponential smoothing, ``"AAN"`` Holt's method, ``"AAA"`` /
        ``"MAM"`` the additive / multiplicative Holt-Winters methods. A
        ``d`` after the trend letter (``"AAdN"``) damps the trend.
    period : int, default 1
        Length of the seasonal cycle (4 quarterly, 12 monthly). With
        ``period=1`` no seasonal model is considered.
    damped : bool, optional
        Damp the trend. ``None`` tries both when the trend is chosen
        automatically, and means no damping for a named trend.
    alpha, beta, gamma, phi : float, optional
        Fix a smoothing parameter instead of estimating it. A fixed
        parameter is not counted in the information criteria.
    ic : {"aicc", "aic", "bic"}, default "aicc"
        Criterion of the automatic choice.
    additive_only : bool, default False
        Restrict the choice to fully additive models.
    allow_multiplicative_trend : bool, default False
        Let the automatic choice consider a multiplicative trend. Those
        models forecast poorly at long horizons and are left out unless
        asked for.
    restrict : bool, default True
        Leave out the combinations that can divide by a state near zero:
        an additive error with a multiplicative trend or season, and a
        multiplicative trend with an additive season.
    bounds : {"both", "usual", "admissible"}, default "both"
        Parameter region. ``"usual"``: ``0 < beta < alpha < 1``,
        ``0 < gamma < 1 - alpha``, ``0.8 <= phi <= 0.98``.
        ``"admissible"``: the forecastability region of the model.
        ``"both"``: their intersection.
    maxiter : int, default 2000
        Iteration limit of each Nelder-Mead run.
    data : pd.DataFrame, optional

    Returns
    -------
    ETSResult
        ``params``, ``initial_state``, ``states``, ``fitted_values``,
        ``residuals``, ``sigma2``, ``aic`` / ``aicc`` / ``bic``,
        ``forecast(horizon, level)``, ``components()``, ``summary()``,
        ``plot()``; ``candidates`` lists the models an automatic choice
        compared.

    Notes
    -----
    The initial states are estimated with the smoothing parameters, as in
    R's ``forecast::ets`` (statsmodels' ``ExponentialSmoothing`` sets them
    heuristically by default, which gives different estimates). The
    likelihood is evaluated exactly as there: at ``ets``'s parameters this
    function returns its log likelihood, fitted values and forecast
    intervals to rounding error. The optimiser is restarted from its own
    optimum, so the reported likelihood is at least as high as a single
    Nelder-Mead run reaches.

    Multiplicative models need a strictly positive series. Seasonal
    periods above 24 are refused: the seasonal initial states alone would
    take that many parameters; use :func:`statspai.stl` and model the
    seasonally adjusted series, or Fourier terms in :func:`statspai.arima`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> t = np.arange(96)
    >>> season = np.tile([0.9, 1.1, 1.2, 0.8], 24)
    >>> y = pd.Series((100 + 0.8 * t) * season * (1 + rng.normal(0, 0.02, 96)))
    >>> fit = sp.ets(y, period=4)          # model chosen by AICc
    >>> fit.season
    'M'
    >>> hw = sp.ets(y, model="MAM", period=4)
    >>> fc = hw.forecast(8, level=95)
    >>> fc.shape
    (8, 3)

    References
    ----------
    hyndman2002state, hyndman2008forecasting, hyndman2008admissible,
    hyndman2026fpppy
    """
    values, index, name = read_series(y, data, fn="ets")
    m = check_period(period, fn="ets")
    err_l, tr_l, se_l, damped_in_name = _parse_model(model)
    if damped_in_name:
        damped = True
    if ic not in ("aicc", "aic", "bic"):
        raise MethodIncompatibility(
            f"ets: ic={ic!r} is not one of 'aicc', 'aic', 'bic'.",
            recovery_hint="Use ic='aicc'.",
        )
    if bounds not in ("both", "usual", "admissible"):
        raise MethodIncompatibility(
            f"ets: bounds={bounds!r} is not one of 'both', 'usual', 'admissible'.",
            recovery_hint="Use bounds='both'.",
        )
    n = values.shape[0]
    if n < 4:
        raise DataInsufficient(
            f"ets: {n} observations are too few to fit a model.",
            recovery_hint="At least 4 observations are needed.",
        )
    if np.ptp(values) == 0:
        raise DataInsufficient(
            "ets: the series is constant.",
            recovery_hint="There is nothing to smooth; the forecast is that value.",
        )
    if se_l in ("A", "M") and m == 1:
        raise MethodIncompatibility(
            f"ets: model={model!r} has a seasonal component but period=1.",
            recovery_hint="Pass period= (4 for quarterly, 12 for monthly data).",
        )
    if m > 24 and se_l != "N":
        if se_l == "Z":
            warnings.warn(
                f"ets: period={m} is above 24; no seasonal model is considered. "
                "Use sp.stl for the seasonal pattern.",
                UserWarning,
                stacklevel=2,
            )
            se_l = "N"
        else:
            raise MethodIncompatibility(
                f"ets: a seasonal model with period={m} would estimate {m - 1} "
                "seasonal starting values.",
                recovery_hint=(
                    "Decompose with sp.stl and model the seasonally adjusted "
                    "series, or use Fourier terms (sp.fourier_terms) in sp.arima."
                ),
            )
    positive = bool(values.min() > 0)
    for letter, what in ((err_l, "error"), (tr_l, "trend"), (se_l, "season")):
        if letter == "M" and not positive:
            raise MethodIncompatibility(
                f"ets: a multiplicative {what} needs a strictly positive "
                f"series; the minimum is {values.min():g}.",
                recovery_hint="Use an additive model.",
            )
    fixed = {
        nm: float(v)
        for nm, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("phi", phi))
        if v is not None
    }
    if "phi" in fixed and damped is None:
        damped = True

    errors = [err_l] if err_l != "Z" else ["A", "M"]
    if tr_l != "Z":
        trends = [tr_l]
    else:
        trends = ["N", "A"] + (["M"] if allow_multiplicative_trend else [])
    if se_l != "Z":
        seasons = [se_l]
    else:
        seasons = ["N"] if (m == 1 or n <= m) else ["N", "A", "M"]
    auto = "Z" in (err_l, tr_l, se_l) or (damped is None and tr_l == "Z")

    tried: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_key: Tuple[str, str, str, bool] = ("A", "N", "N", False)
    for e in errors:
        for t in trends:
            if t == "N":
                damp_opts = [False]
            elif damped is None:
                damp_opts = [False, True] if tr_l == "Z" else [False]
            else:
                damp_opts = [bool(damped)]
            for s in seasons:
                for d in damp_opts:
                    if auto:
                        if restrict and (
                            (e == "A" and (t == "M" or s == "M"))
                            or (t == "M" and s == "A")
                        ):
                            continue
                        if additive_only and "M" in (e, t, s):
                            continue
                        if not positive and "M" in (e, t, s):
                            continue
                    use_fixed = {
                        k: v
                        for k, v in fixed.items()
                        if not (auto and k == "beta" and t == "N")
                        and not (auto and k == "gamma" and s == "N")
                        and not (auto and k == "phi" and not d)
                    }
                    fit = _fit_one(
                        values, m, e, t, s, d, use_fixed, bounds, int(maxiter)
                    )
                    label = _model_name(e, t, s, d)
                    if fit is None:
                        tried.append(
                            {
                                "model": label,
                                "aic": np.nan,
                                "aicc": np.nan,
                                "bic": np.nan,
                                "log_likelihood": np.nan,
                            }
                        )
                        continue
                    tried.append(
                        {
                            "model": label,
                            "aic": fit["aic"],
                            "aicc": fit["aicc"],
                            "bic": fit["bic"],
                            "log_likelihood": -0.5 * fit["neg2ll"],
                        }
                    )
                    if np.isfinite(fit[ic]) and (best is None or fit[ic] < best[ic]):
                        best, best_key = fit, (e, t, s, d)
    if best is None:
        raise DataInsufficient(
            "ets: no model could be fitted.",
            recovery_hint=(
                "The series is too short for the requested model, or the "
                "fixed parameters are outside the parameter region "
                f"(bounds={bounds!r})."
            ),
            diagnostics={"n": n, "tried": [r["model"] for r in tried]},
        )
    e, t, s, d = best_key
    spec: _Spec = best["spec"]
    par = best["par"]
    params = pd.Series(
        {nm: float(par[nm]) for nm in _SMOOTH if spec.has[nm]}, dtype=float
    )
    init: Dict[str, float] = {"l": best["l0"]}
    if t != "N":
        init["b"] = best["b0"]
    if s != "N":
        for j, v in enumerate(best["s0"]):
            init["s0" if j == 0 else f"s-{j}"] = float(v)
    st = best["states"]
    cols: Dict[str, np.ndarray] = {"level": st[:, 0]}
    if t != "N":
        cols["slope"] = st[:, 1]
    if s != "N":
        cols["season"] = st[:, 2]
        for j in range(1, spec.m):
            cols[f"season_lag{j}"] = st[:, 2 + j]
    np_ = spec.k + 1
    if not best["converged"]:
        warnings.warn(
            f"ets: the optimiser stopped at the iteration limit for "
            f"{_model_name(e, t, s, d)}; raise maxiter.",
            ConvergenceWarning,
            stacklevel=2,
        )
    result = ETSResult(
        model=_model_name(e, t, s, d),
        error=e,
        trend=t,
        season=s,
        damped=bool(d),
        period=spec.m,
        params=params,
        initial_state=pd.Series(init, dtype=float),
        states=pd.DataFrame(cols),
        fitted_values=best["fitted"],
        residuals=best["resid"],
        sigma2=float(best["sse"] / (n - np_ + 1)),
        log_likelihood=float(-0.5 * best["neg2ll"]),
        aic=best["aic"],
        aicc=best["aicc"],
        bic=best["bic"],
        mse=float(np.mean((values - best["fitted"]) ** 2)),
        n=n,
        n_params=np_,
        converged=bool(best["converged"]),
        candidates=(
            pd.DataFrame(tried)
            .sort_values(ic, na_position="last")
            .reset_index(drop=True)
            if len(tried) > 1
            else None
        ),
        fixed={k: v for k, v in fixed.items() if k in params.index},
        _y=values,
        _index=index,
        _name=name,
    )
    from ..output._lineage import attach_provenance as _attach_prov

    _attach_prov(
        result,
        function="sp.timeseries.ets",
        params={
            "model": model,
            "period": m,
            "damped": damped,
            "ic": ic,
            "bounds": bounds,
            "fixed": fixed,
        },
        data=None,
        overwrite=False,
    )
    return result
