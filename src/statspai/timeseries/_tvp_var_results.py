"""Result object of :func:`statspai.timeseries.tvp_var.tvp_var`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from ._tvp_var_core import ma_coefficients, max_roots

__all__ = ["TVPVARResult"]


@dataclass
class TVPVARResult(ResultProtocolMixin):
    """Result of :func:`tvp_var`.

    Attributes
    ----------
    coef_filtered, se_filtered : ndarray, shape (T, K, K p + 1)
        Coefficients given the data up to each date, and their standard
        errors. Axis 1 is the equation, axis 2 the regressor in the order
        of ``terms`` (lags first, ``_cons`` last, as in
        :func:`statspai.var`). ``T`` counts the dates after the first
        ``lags``.
    coef_smoothed, se_smoothed : ndarray or None
        The same given the whole sample (``method='kalman'`` only).
    sigma : pd.DataFrame
        Error covariance. ``'kalman'``: constant, the covariance of the
        one-step forecast errors (see the notes of :func:`tvp_var`).
        ``'forgetting'``: the value at the last date.
    sigma_t : ndarray (T, K, K) or None
        The path of the error covariance (``'forgetting'``).
    variances : pd.DataFrame or None
        ``'kalman'``: one row per equation, the error variance ``obs`` and
        the innovation variance of every coefficient.
    loglik : float
        ``'kalman'``: the sum of the equations' Kalman-filter log
        likelihoods (the equations are estimated separately).
        ``'forgetting'``: the sum of the log one-step predictive
        densities.
    index : pd.Index
        Dates of the ``T`` rows.
    var_names, terms : list of str
    lags, n_obs, method, alpha

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     a = 0.2 + 0.6 * t / 160
    ...     y[t, 0] = a * y[t - 1, 0] + rng.normal()
    ...     y[t, 1] = 0.3 * y[t - 1, 1] + rng.normal()
    >>> df = pd.DataFrame(y, columns=["x", "z"])
    >>> fit = sp.tvp_var(df, lags=1, method="forgetting", lam=0.98)
    >>> fit.coef_filtered.shape
    (159, 2, 3)
    >>> fit.terms
    ['L1.x', 'L1.z', '_cons']
    >>> list(fit.irf(at=-1, periods=4).columns)
    ['date', 'shock', 'response', 'period', 'irf']
    """

    coef_filtered: np.ndarray
    se_filtered: np.ndarray
    coef_smoothed: Optional[np.ndarray]
    se_smoothed: Optional[np.ndarray]
    sigma: pd.DataFrame
    sigma_t: Optional[np.ndarray]
    variances: Optional[pd.DataFrame]
    loglik: float
    index: pd.Index
    var_names: List[str]
    terms: List[str]
    lags: int
    n_obs: int
    method: str
    alpha: float = 0.05
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("petris2009dynamic", "kalman1960new", "lutkepohl2005new")

    # ------------------------------------------------------------------ #
    def _kind(self, kind: Optional[str]) -> str:
        if kind is None:
            return "smoothed" if self.coef_smoothed is not None else "filtered"
        kind = str(kind).lower()
        if kind not in ("filtered", "smoothed"):
            raise MethodIncompatibility(
                f"tvp_var: kind={kind!r} is not 'filtered' or 'smoothed'."
            )
        if kind == "smoothed" and self.coef_smoothed is None:
            raise MethodIncompatibility(
                "tvp_var: method='forgetting' has no smoothed coefficients.",
                recovery_hint="Use kind='filtered', or fit with method='kalman'.",
            )
        return kind

    def _arrays(self, kind: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        if self._kind(kind) == "smoothed":
            assert self.coef_smoothed is not None and self.se_smoothed is not None
            return self.coef_smoothed, self.se_smoothed
        return self.coef_filtered, self.se_filtered

    def _position(self, at: Any) -> int:
        T = len(self.index)
        if isinstance(at, (int, np.integer)) and not isinstance(at, bool):
            pos = int(at) + T if at < 0 else int(at)
            if not 0 <= pos < T:
                raise MethodIncompatibility(
                    f"tvp_var: position {at} is outside 0 .. {T - 1}.",
                    recovery_hint="Integers count the estimation dates, "
                    "which start after the first `lags` rows.",
                )
            return pos
        key = pd.Timestamp(at) if isinstance(self.index, pd.DatetimeIndex) else at
        if isinstance(self.index, pd.PeriodIndex) and not isinstance(key, pd.Period):
            key = pd.Period(key, freq=self.index.freq)
        if key not in self.index:
            raise MethodIncompatibility(
                f"tvp_var: {at!r} is not among the estimation dates "
                f"({self.index[0]} .. {self.index[-1]}).",
                recovery_hint="Pass a label of the index or an integer position.",
            )
        loc = self.index.get_loc(key)
        if not isinstance(loc, (int, np.integer)):
            raise MethodIncompatibility(f"tvp_var: date {at!r} is not unique.")
        return int(loc)

    def _sigma_at(self, pos: int) -> np.ndarray:
        if self.sigma_t is not None:
            return np.asarray(self.sigma_t[pos], dtype=float)
        out: np.ndarray = self.sigma.to_numpy(dtype=float)
        return out

    # ------------------------------------------------------------------ #
    def coefficients(
        self, kind: Optional[str] = None, alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """Coefficient paths, one row per date, equation and regressor.

        Parameters
        ----------
        kind : {'smoothed', 'filtered'}, optional
            Default: smoothed when the method has them, else filtered.
        alpha : float, optional
            ``1 - alpha`` is the pointwise coverage of ``lower`` /
            ``upper`` (normal quantiles); default the ``alpha`` of the fit.

        Returns
        -------
        pd.DataFrame
            ``date``, ``equation``, ``term``, ``coef``, ``se``, ``lower``,
            ``upper``.
        """
        a = self.alpha if alpha is None else float(alpha)
        if not 0.0 < a < 1.0:
            raise MethodIncompatibility(f"tvp_var: alpha={a} is not in (0, 1).")
        coef, se = self._arrays(kind)
        T, K, k = coef.shape
        z = float(stats.norm.ppf(1.0 - a / 2.0))
        c, s = coef.reshape(-1), se.reshape(-1)
        return pd.DataFrame(
            {
                "date": np.repeat(np.asarray(self.index), K * k),
                "equation": np.tile(np.repeat(self.var_names, k), T),
                "term": np.tile(self.terms, T * K),
                "coef": c,
                "se": s,
                "lower": c - z * s,
                "upper": c + z * s,
            }
        )

    def irf(
        self,
        at: Any = -1,
        periods: int = 10,
        orthogonal: bool = True,
        cumulative: bool = False,
        kind: Optional[str] = None,
    ) -> pd.DataFrame:
        """Impulse responses of the VAR frozen at one or several dates.

        The coefficients and the error covariance of date ``at`` are held
        fixed and the responses are those of that constant VAR. It answers
        "how would a shock propagate if the economy stayed as it was at
        that date", and ignores that coefficients keep moving afterwards.

        Parameters
        ----------
        at : label, int or list of them, default -1
            Dates: labels of ``index``, or integers, which are always
            positions among the estimation dates (negative from the end).
            The default is the last date.
        periods : int, default 10
        orthogonal : bool, default True
            Responses to one-standard-deviation shocks from the Cholesky
            factor of the error covariance in the order of the variables,
            as in :func:`statspai.irf`. ``False``: unit innovations.
        cumulative : bool, default False
        kind : {'smoothed', 'filtered'}, optional

        Returns
        -------
        pd.DataFrame
            ``date``, ``shock``, ``response``, ``period``, ``irf``.
        """
        if periods < 0:
            raise MethodIncompatibility("tvp_var: periods must be non-negative.")
        coef, _ = self._arrays(kind)
        many = isinstance(at, (list, tuple, np.ndarray, pd.Index))
        frames = []
        K = len(self.var_names)
        for one in list(at) if many else [at]:
            pos = self._position(one)
            phi = ma_coefficients(coef[pos], self.lags, periods)
            if orthogonal:
                phi = phi @ np.linalg.cholesky(self._sigma_at(pos))
            if cumulative:
                phi = np.cumsum(phi, axis=0)
            # phi[s, i, j]: response i to shock j; rows ordered shock,
            # response, period as in SVARResult.irf
            frames.append(
                pd.DataFrame(
                    {
                        "date": self.index[pos],
                        "shock": np.repeat(self.var_names, K * (periods + 1)),
                        "response": np.tile(np.repeat(self.var_names, periods + 1), K),
                        "period": np.tile(np.arange(periods + 1), K * K),
                        "irf": phi.transpose(2, 1, 0).reshape(-1),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def stability(self, kind: Optional[str] = None) -> pd.DataFrame:
        """Largest companion root of the frozen VAR at every date.

        Returns
        -------
        pd.DataFrame
            Indexed by date: ``max_root`` (modulus) and ``explosive``
            (``max_root >= 1``). ``attrs['n_explosive']`` counts the
            flagged dates.
        """
        coef, _ = self._arrays(kind)
        roots = max_roots(coef, self.lags)
        out = pd.DataFrame(
            {"max_root": roots, "explosive": roots >= 1.0}, index=self.index
        )
        out.attrs["n_explosive"] = int((roots >= 1.0).sum())
        return out

    def forecast(self, steps: int = 1, alpha: Optional[float] = None) -> pd.DataFrame:
        """Forecasts from the last filtered coefficients.

        The coefficients stay at their last filtered value (the forecast
        of a random walk) and are treated as known: the standard errors
        come from the error covariance of the last date only, as in
        ``VARResult.forecast``, and understate the uncertainty.

        Returns
        -------
        pd.DataFrame
            Indexed by horizon ``1 .. steps``; for each variable
            ``<name>``, ``<name>_se``, ``<name>_lower``, ``<name>_upper``.
        """
        if steps < 1:
            raise MethodIncompatibility("tvp_var: steps must be at least 1.")
        a = self.alpha if alpha is None else float(alpha)
        if not 0.0 < a < 1.0:
            raise MethodIncompatibility(f"tvp_var: alpha={a} is not in (0, 1).")
        B = self.coef_filtered[-1]
        K, p = len(self.var_names), self.lags
        hist = [np.asarray(v, dtype=float) for v in self._state["last_obs"]]
        means = []
        for _ in range(steps):
            x = np.concatenate([hist[-lag] for lag in range(1, p + 1)] + [np.ones(1)])
            nxt = B @ x
            means.append(nxt)
            hist.append(nxt)
        phi = ma_coefficients(B, p, steps - 1)
        sig = self._sigma_at(len(self.index) - 1)
        mse = np.cumsum(np.einsum("sij,jk,slk->sil", phi, sig, phi), axis=0)
        z = float(stats.norm.ppf(1.0 - a / 2.0))
        cols: Dict[str, np.ndarray] = {}
        mean = np.asarray(means)
        for i, nm in enumerate(self.var_names):
            sd = np.sqrt(mse[:, i, i])
            cols[nm] = mean[:, i]
            cols[f"{nm}_se"] = sd
            cols[f"{nm}_lower"] = mean[:, i] - z * sd
            cols[f"{nm}_upper"] = mean[:, i] + z * sd
        _ = K
        return pd.DataFrame(cols, index=pd.RangeIndex(1, steps + 1, name="horizon"))

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        K, k = len(self.var_names), len(self.terms)
        kind = self._kind(None)
        coef, _ = self._arrays(None)
        stab = self.stability()
        lines = [
            f"Time-varying-parameter VAR ({self.method})",
            "=" * 66,
            f"Variables: {', '.join(self.var_names)}    Lags: {self.lags}",
            f"Observations: {self.n_obs}    Coefficients per equation: {k}"
            f"    Log likelihood: {self.loglik:.4f}",
        ]
        if self.method == "forgetting":
            lines.append(
                f"Forgetting factor lam = {self.model_info['lam']:g}, "
                f"covariance decay kappa = {self.model_info['kappa']:g}"
            )
        lines += [
            f"Largest companion root: min {stab['max_root'].min():.4f}, "
            f"max {stab['max_root'].max():.4f}; explosive at "
            f"{stab.attrs['n_explosive']} of {len(stab)} dates",
            "",
            f"Range of the {kind} coefficients over time (min .. max)",
        ]
        lo, hi = coef.min(axis=0), coef.max(axis=0)
        table = pd.DataFrame(
            [[f"{lo[i, j]:.4g} .. {hi[i, j]:.4g}" for j in range(k)] for i in range(K)],
            index=self.var_names,
            columns=self.terms,
        )
        lines.append(table.T.to_string())
        if self.variances is not None:
            lines += [
                "",
                "Error variance (obs) and coefficient innovation variances",
                self.variances.T.to_string(float_format=lambda v: f"{v:.4g}"),
            ]
        sig_label = (
            "Error covariance (last date)"
            if self.sigma_t is not None
            else "Error covariance"
        )
        lines += [
            "",
            sig_label,
            self.sigma.to_string(float_format=lambda v: f"{v:.5g}"),
        ]
        for note in self.model_info.get("notes", []):
            lines += ["", f"Note: {note}"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        stab = self.stability()
        out: Dict[str, Any] = {
            "method": self.method,
            "var_names": list(self.var_names),
            "terms": list(self.terms),
            "lags": int(self.lags),
            "n_obs": int(self.n_obs),
            "loglik": float(self.loglik),
            "sigma": self.sigma.to_numpy(dtype=float).tolist(),
            "last_filtered_coefficients": self.coef_filtered[-1].tolist(),
            "max_root_range": [
                float(stab["max_root"].min()),
                float(stab["max_root"].max()),
            ],
            "n_explosive_dates": int(stab.attrs["n_explosive"]),
            "constant_coefficients": list(
                self.model_info.get("constant_coefficients", [])
            ),
            "notes": list(self.model_info.get("notes", [])),
        }
        if self.variances is not None:
            out["variances"] = {
                str(i): {str(c): float(v) for c, v in row.items()}
                for i, row in self.variances.iterrows()
            }
        return out

    def plot(
        self,
        equation: Optional[str] = None,
        terms: Optional[Sequence[str]] = None,
        kind: Optional[str] = None,
    ) -> Any:
        """Coefficient paths with pointwise ``1 - alpha`` bands.

        Parameters
        ----------
        equation : str, optional
            One equation; default all of them (one column each).
        terms : list of str, optional
            Regressors to show; default all.
        kind : {'smoothed', 'filtered'}, optional
        """
        import matplotlib.pyplot as plt

        coef, se = self._arrays(kind)
        eqs = [equation] if equation is not None else list(self.var_names)
        shown = list(terms) if terms is not None else list(self.terms)
        unknown = [e for e in eqs if e not in self.var_names] + [
            t for t in shown if t not in self.terms
        ]
        if unknown:
            raise MethodIncompatibility(
                f"tvp_var: {unknown} are not equations {self.var_names} or "
                f"terms {self.terms}."
            )
        z = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        fig, axes = plt.subplots(
            len(shown),
            len(eqs),
            figsize=(3.6 * len(eqs), 2.0 * len(shown)),
            squeeze=False,
            sharex=True,
        )
        t = np.asarray(self.index)
        for c, eq in enumerate(eqs):
            i = self.var_names.index(eq)
            for r, term in enumerate(shown):
                j = self.terms.index(term)
                ax = axes[r, c]
                ax.plot(t, coef[:, i, j], lw=1.2)
                ax.fill_between(
                    t,
                    coef[:, i, j] - z * se[:, i, j],
                    coef[:, i, j] + z * se[:, i, j],
                    alpha=0.25,
                )
                ax.set_title(f"{eq}: {term}", fontsize=9)
        fig.tight_layout()
        return fig
