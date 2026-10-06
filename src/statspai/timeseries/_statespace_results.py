"""Result containers of :mod:`statspai.timeseries.statespace`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from . import _statespace_core as core

__all__ = ["KalmanResult", "StateSpaceResult"]

_SYSTEM = ("A", "G", "F", "Q", "R")


@dataclass
class KalmanResult(ResultProtocolMixin):
    """Output of :func:`kalman_filter`.

    Attributes
    ----------
    predicted_state, predicted_cov : ndarray
        ``E[X_t | Y_1..Y_{t-1}]`` and its mean squared error, shapes
        ``(T, m)`` and ``(T, m, m)``.
    filtered_state, filtered_cov : ndarray
        The same given ``Y_1..Y_t``.
    smoothed_state, smoothed_cov : ndarray or None
        The same given the whole sample.
    innovations : ndarray
        One-step prediction errors ``Y_t - A_t - G_t X_{t|t-1}``, ``(T, n)``,
        NaN where ``Y`` is missing.
    innovations_cov : ndarray
        Their covariance, ``(T, n, n)``, NaN in the rows and columns of
        missing elements.
    std_innovations : ndarray
        ``L_t^{-1}`` times the prediction errors, ``L_t`` the lower
        Cholesky factor of their covariance among the observed elements:
        uncorrelated with unit variance under the model.
    loglik : float
        Gaussian log-likelihood, the sum of ``loglik_obs`` from date
        ``burn + 1`` on.
    loglik_obs : ndarray
        Contribution of each date; zero when the whole row is missing.
    n_obs : int
        Scalar observations that enter ``loglik``.
    n_dates : int
        Dates with at least one of them.
    x0, P0 : ndarray
        Moments of the initial state that were used.
    init : str
        ``'user'``, ``'stationary'`` or ``'diffuse'``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.normal(size=60)) + rng.normal(size=60)
    >>> out = sp.kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=1.0)
    >>> isinstance(out, sp.KalmanResult)
    True
    >>> out.states("filtered").shape
    (60, 2)
    """

    predicted_state: np.ndarray
    predicted_cov: np.ndarray
    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    smoothed_state: Optional[np.ndarray]
    smoothed_cov: Optional[np.ndarray]
    innovations: np.ndarray
    innovations_cov: np.ndarray
    std_innovations: np.ndarray
    loglik: float
    loglik_obs: np.ndarray
    n_obs: int
    n_dates: int
    x0: np.ndarray
    P0: np.ndarray
    init: str
    obs_names: List[str] = field(default_factory=list)
    state_names: List[str] = field(default_factory=list)
    index: Optional[pd.Index] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("kalman1960new",)

    def states(self, which: str = "smoothed") -> pd.DataFrame:
        """State estimates with their standard errors as a DataFrame.

        ``which`` is ``'predicted'``, ``'filtered'`` or ``'smoothed'``; the
        columns are the state names and ``<name>_se``.
        """
        if which not in ("predicted", "filtered", "smoothed"):
            raise MethodIncompatibility(
                f"which={which!r} is not 'predicted', 'filtered' or 'smoothed'."
            )
        mean = getattr(self, f"{which}_state")
        cov = getattr(self, f"{which}_cov")
        if mean is None:
            raise MethodIncompatibility(
                "The smoother was not run.", recovery_hint="Call with smooth=True."
            )
        sd = np.sqrt(np.clip(np.einsum("tii->ti", cov), 0.0, None))
        out = pd.DataFrame(mean, columns=self.state_names, index=self.index)
        for j, nm in enumerate(self.state_names):
            out[f"{nm}_se"] = sd[:, j]
        return out

    def forecast(self, steps: int = 1, **future: Any) -> Dict[str, Any]:
        """Forecast states and observations beyond the sample.

        Parameters
        ----------
        steps : int
            Horizon.
        **future
            Values of ``F``, ``G``, ``Q``, ``R`` or ``A`` over the forecast
            dates, with a leading axis of length ``steps`` (or constant). A
            matrix that varied in the sample must be given here.

        Returns
        -------
        dict
            ``state`` and ``obs`` (DataFrames of forecasts), ``state_se``
            and ``obs_se`` (root mean squared errors), ``state_cov`` and
            ``obs_cov`` (arrays ``(steps, ., .)``). The mean squared error
            of an observation includes ``R``; the system matrices are taken
            as known.
        """
        if int(steps) < 1:
            raise MethodIncompatibility("steps must be at least 1.")
        steps = int(steps)
        unknown = sorted(set(future) - set(_SYSTEM))
        if unknown:
            raise MethodIncompatibility(f"Unknown system matrices {unknown}.")
        sysm = self._state["sys"]
        n, m = len(self.obs_names), len(self.state_names)
        shapes = {"A": (n,), "G": (n, m), "F": (m, m), "Q": (m, m), "R": (n, n)}
        use = {}
        for key in _SYSTEM:
            if future.get(key) is not None:
                use[key] = core.as_stack(future[key], shapes[key], steps, key)
            elif sysm[key].shape[0] > 1:
                raise MethodIncompatibility(
                    f"{key} varies over the sample; its values over the "
                    "forecast dates are needed.",
                    recovery_hint=f"Pass {key}= with a leading axis of {steps}.",
                )
            else:
                use[key] = sysm[key]
        xs, Ps, ys, Ss = core.forecast_moments(
            steps,
            self.filtered_state[-1],
            self.filtered_cov[-1],
            use["A"],
            use["G"],
            use["F"],
            use["Q"],
            use["R"],
        )
        idx = pd.RangeIndex(1, steps + 1, name="step")
        return {
            "state": pd.DataFrame(xs, columns=self.state_names, index=idx),
            "state_se": pd.DataFrame(
                np.sqrt(np.einsum("tii->ti", Ps)), columns=self.state_names, index=idx
            ),
            "state_cov": Ps,
            "obs": pd.DataFrame(ys, columns=self.obs_names, index=idx),
            "obs_se": pd.DataFrame(
                np.sqrt(np.einsum("tii->ti", Ss)), columns=self.obs_names, index=idx
            ),
            "obs_cov": Ss,
        }

    def summary(self) -> str:
        T = self.predicted_state.shape[0]
        lines = [
            "Kalman filter"
            + (" and smoother" if self.smoothed_state is not None else ""),
            f"Dates: {T}    Observables: {len(self.obs_names)}    "
            f"States: {len(self.state_names)}",
            f"Observations in the likelihood: {self.n_obs} on {self.n_dates} dates",
            f"Log likelihood: {self.loglik:.6f}    Initial state: {self.init}",
        ]
        which = "smoothed" if self.smoothed_state is not None else "filtered"
        frame = self.states(which)
        rows = sorted({0, T // 2, T - 1})
        lines += ["", f"States, {which} (first, middle and last date)"]
        lines.append(frame.iloc[rows].to_string(float_format=lambda v: f"{v:.5g}"))
        for note in self.model_info.get("notes", []):
            lines += ["", f"Note: {note}"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "loglik": float(self.loglik),
            "n_obs": int(self.n_obs),
            "n_dates": int(self.n_dates),
            "init": self.init,
            "obs_names": list(self.obs_names),
            "state_names": list(self.state_names),
            "last_filtered_state": [float(v) for v in self.filtered_state[-1]],
            "model_info": dict(self.model_info),
        }

    def plot(self, which: str = "smoothed", alpha: float = 0.05) -> Any:
        """State paths with pointwise ``1 - alpha`` bands."""
        import matplotlib.pyplot as plt

        frame = self.states(which)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        names = self.state_names
        fig, axes = plt.subplots(
            len(names), 1, figsize=(7.0, 2.4 * len(names)), squeeze=False
        )
        t = np.arange(len(frame))
        for ax, nm in zip(axes[:, 0], names):
            mid = frame[nm].to_numpy()
            half = z * frame[f"{nm}_se"].to_numpy()
            ax.plot(t, mid, lw=1.2)
            ax.fill_between(t, mid - half, mid + half, alpha=0.25)
            ax.set_title(f"{nm} ({which})")
        fig.tight_layout()
        return fig


def package(
    res: Dict[str, Any],
    names: List[str],
    index: pd.Index,
    burn: int,
    state_names: Optional[Sequence[str]],
    kappa: float,
) -> KalmanResult:
    mask = res["mask"]
    m = res["m"]
    T = mask.shape[0]
    if not 0 <= int(burn) < T:
        raise MethodIncompatibility(f"burn must be between 0 and {T - 1}.")
    snames = (
        [f"x{j + 1}" for j in range(m)] if state_names is None else list(state_names)
    )
    if len(snames) != m:
        raise MethodIncompatibility(f"state_names must have {m} entries.")
    v = np.where(mask, res["v"], np.nan)
    e = np.where(mask, res["e"], np.nan)
    both = mask[:, :, None] & mask[:, None, :]
    S = np.where(both, res["S"], np.nan)
    notes: List[str] = []
    if res["rule"] == "diffuse":
        notes.append(
            f"Approximate diffuse initial state: P0 = {kappa:g} I. The first "
            "prediction errors carry that variance into the likelihood; "
            "burn= leaves them out."
        )
    return KalmanResult(
        predicted_state=res["xp"],
        predicted_cov=res["Pp"],
        filtered_state=res["xf"],
        filtered_cov=res["Pf"],
        smoothed_state=res.get("xs"),
        smoothed_cov=res.get("Ps"),
        innovations=v,
        innovations_cov=S,
        std_innovations=e,
        loglik=float(res["ll"][burn:].sum()),
        loglik_obs=res["ll"],
        n_obs=int(mask[burn:].sum()),
        n_dates=int(mask[burn:].any(axis=1).sum()),
        x0=res["x0"],
        P0=res["P0"],
        init=res["rule"],
        obs_names=names,
        state_names=snames,
        index=index,
        model_info={"burn": int(burn), "notes": notes},
        _state={"sys": res["sys"]},
    )


@dataclass
class StateSpaceResult(ResultProtocolMixin):
    """Maximum-likelihood fit returned by :func:`statespace`.

    Attributes
    ----------
    params : pd.Series
        Estimates of the parameter vector passed to ``build``.
    se : pd.Series
        Standard errors.
    table : pd.DataFrame
        ``estimate``, ``se``, ``z``, ``pvalue``, ``lower``, ``upper``.
    cov : pd.DataFrame
        Covariance of the estimates.
    transformed : pd.DataFrame or None
        The same table for ``transform(params)``, by the delta method.
    loglik, aic, bic : float
    n_obs, n_dates : int
        Scalar observations in the likelihood and dates carrying them;
        ``bic`` uses ``n_dates``.
    converged : bool
        Scaled gradient near zero and a positive definite Hessian.
    filter : KalmanResult
        Filter and smoother output at the estimates.
    model_info : dict
        ``gradient``, ``scaled_gradient``, ``hessian_pd``, ``steps`` (what
        each optimiser returned), ``vce``.

    Examples
    --------
    A local level model with both variances estimated:

    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> level = np.cumsum(0.3 * rng.normal(size=150))
    >>> y = level + rng.normal(size=150)
    >>> def build(th):
    ...     return {"F": 1.0, "G": 1.0, "Q": np.exp(th[0]), "R": np.exp(th[1])}
    >>> fit = sp.statespace(y, build, [0.0, 0.0], init="diffuse", burn=1,
    ...                     param_names=["log_q", "log_r"])
    >>> type(fit).__name__
    'StateSpaceResult'
    >>> list(fit.params.index)
    ['log_q', 'log_r']
    """

    params: pd.Series
    se: pd.Series
    table: pd.DataFrame
    cov: pd.DataFrame
    transformed: Optional[pd.DataFrame]
    loglik: float
    aic: float
    bic: float
    n_obs: int
    n_dates: int
    converged: bool
    filter: KalmanResult
    alpha: float = 0.05
    model_info: Dict[str, Any] = field(default_factory=dict)

    _citation_keys = ("kalman1960new",)

    @property
    def coef(self) -> pd.Series:
        return self.params

    def forecast(self, steps: int = 1, **future: Any) -> Dict[str, Any]:
        """Forecast at the estimates; see :meth:`KalmanResult.forecast`.

        Parameter uncertainty is not included in the mean squared errors.
        """
        return self.filter.forecast(steps, **future)

    def plot(self, which: str = "smoothed") -> Any:
        """State paths with pointwise bands; see :meth:`KalmanResult.plot`."""
        return self.filter.plot(which, alpha=self.alpha)

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "State space model, maximum likelihood",
            f"Observations: {self.n_obs} on {self.n_dates} dates    "
            f"Parameters: {len(self.params)}",
            f"Log likelihood: {self.loglik:.6f}    AIC: {self.aic:.4f}    "
            f"BIC: {self.bic:.4f}",
            f"Converged: {self.converged}    max scaled gradient: "
            f"{info.get('scaled_gradient', float('nan')):.3g}    "
            f"vce: {info.get('vce')}",
            "",
            self.table.to_string(float_format=lambda v: f"{v:.6g}"),
        ]
        if self.transformed is not None:
            lines += ["", "Transformed parameters (delta method)"]
            lines.append(self.transformed.to_string(float_format=lambda v: f"{v:.6g}"))
        for note in info.get("notes", []):
            lines += ["", f"Note: {note}"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "params": {str(k): float(v) for k, v in self.params.items()},
            "se": {str(k): float(v) for k, v in self.se.items()},
            "loglik": float(self.loglik),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "n_obs": int(self.n_obs),
            "n_dates": int(self.n_dates),
            "converged": bool(self.converged),
            "model_info": {
                k: v
                for k, v in self.model_info.items()
                if not isinstance(v, np.ndarray)
            },
        }


def inference(
    est: np.ndarray, cov: np.ndarray, names: List[str], alpha: float
) -> pd.DataFrame:
    se = np.sqrt(np.where(np.diag(cov) >= 0, np.diag(cov), np.nan))
    z = est / se
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return pd.DataFrame(
        {
            "estimate": est,
            "se": se,
            "z": z,
            "pvalue": 2.0 * stats.norm.sf(np.abs(z)),
            "lower": est - crit * se,
            "upper": est + crit * se,
        },
        index=names,
    )
