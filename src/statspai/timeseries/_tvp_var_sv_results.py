"""Result object of :func:`statspai.timeseries.tvp_var_sv.tvp_var_sv`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = ["TVPVARSVResult"]


def _bands(x: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Median, mean and equal-tailed band over the first axis (draws)."""
    lo, med, hi = np.quantile(x, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=0)
    return {
        "median": med.reshape(-1),
        "mean": x.mean(axis=0).reshape(-1),
        "lower": lo.reshape(-1),
        "upper": hi.reshape(-1),
    }


@dataclass
class TVPVARSVResult(ResultProtocolMixin):
    """Posterior draws of a TVP-VAR with stochastic volatility.

    Attributes
    ----------
    coef_draws : ndarray, shape (draws, T, K, K p + 1)
        Coefficients: axis 2 is the equation, axis 3 the regressor in the
        order of ``terms`` (lags first, ``_cons`` last). ``T`` counts the
        estimation dates (after the training sample).
    a_draws : ndarray, shape (draws, T, K (K - 1) / 2)
        Free elements of the unit lower-triangular ``A_t``, row by row.
    logsig_draws : ndarray, shape (draws, T, K)
        Log standard deviations of the orthogonal (structural) shocks.
    q_diag_draws, s_draws, w_draws : ndarray
        Innovation covariances of the three sets of states (for ``Q`` the
        diagonal only).
    index : pd.Index
        Dates of the ``T`` rows.
    var_names, terms : list of str
    lags, n_obs, alpha
    model_info : dict
        Prior, sampler settings and notes.

    Examples
    --------
    >>> import numpy as np, pandas as pd, warnings
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((140, 2))
    >>> for t in range(1, 140):
    ...     sd = 1.0 if t < 90 else 2.0
    ...     y[t, 0] = 0.5 * y[t - 1, 0] + sd * rng.normal()
    ...     y[t, 1] = 0.3 * y[t - 1, 1] + 0.4 * y[t, 0] + rng.normal()
    >>> df = pd.DataFrame(y, columns=["x", "z"])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # a chain this short mixes badly
    ...     fit = sp.tvp_var_sv(df, lags=1, training=40, draws=200,
    ...                         burnin=100, seed=1)
    >>> type(fit).__name__
    'TVPVARSVResult'
    >>> sorted(fit.volatility()["variable"].unique())
    ['x', 'z']
    """

    coef_draws: np.ndarray
    a_draws: np.ndarray
    logsig_draws: np.ndarray
    q_diag_draws: np.ndarray
    s_draws: np.ndarray
    w_draws: np.ndarray
    index: pd.Index
    var_names: List[str]
    terms: List[str]
    lags: int
    n_obs: int
    alpha: float = 0.05
    model_info: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("primiceri2005time", "carter1994gibbs")

    # ------------------------------------------------------------------ #
    def _alpha(self, alpha: Optional[float]) -> float:
        a = self.alpha if alpha is None else float(alpha)
        if not 0.0 < a < 1.0:
            raise MethodIncompatibility(f"tvp_var_sv: alpha={a} is not in (0, 1).")
        return a

    def _position(self, at: Any) -> int:
        T = len(self.index)
        if isinstance(at, (int, np.integer)) and not isinstance(at, bool):
            pos = int(at) + T if at < 0 else int(at)
            if not 0 <= pos < T:
                raise MethodIncompatibility(
                    f"tvp_var_sv: position {at} is outside 0 .. {T - 1}.",
                    recovery_hint="Integers count the estimation dates, "
                    "which start after the training sample.",
                )
            return pos
        key = pd.Timestamp(at) if isinstance(self.index, pd.DatetimeIndex) else at
        if isinstance(self.index, pd.PeriodIndex) and not isinstance(key, pd.Period):
            key = pd.Period(key, freq=self.index.freq)
        if key not in self.index:
            raise MethodIncompatibility(
                f"tvp_var_sv: {at!r} is not among the estimation dates "
                f"({self.index[0]} .. {self.index[-1]}).",
                recovery_hint="Pass a label of the index or an integer position.",
            )
        loc = self.index.get_loc(key)
        if not isinstance(loc, (int, np.integer)):
            raise MethodIncompatibility(f"tvp_var_sv: date {at!r} is not unique.")
        return int(loc)

    def _impact(self, sel: Any = slice(None)) -> np.ndarray:
        """``A_t^{-1} diag(sigma_t)`` for the dates ``sel``: (draws, ., K, K)."""
        K = len(self.var_names)
        a = np.asarray(self.a_draws[:, sel], dtype=float)
        A = np.zeros(a.shape[:-1] + (K, K))
        A[..., np.arange(K), np.arange(K)] = 1.0
        rows, cols = np.tril_indices(K, -1)
        A[..., rows, cols] = a
        sd = np.exp(np.asarray(self.logsig_draws[:, sel], dtype=float))
        out: np.ndarray = np.linalg.inv(A) * sd[..., None, :]
        return out

    # ------------------------------------------------------------------ #
    def coefficients(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Posterior summary of the coefficient paths.

        Returns
        -------
        pd.DataFrame
            ``date``, ``equation``, ``term``, ``median``, ``mean``,
            ``lower``, ``upper``; one row per date, equation and regressor.
        """
        a = self._alpha(alpha)
        _, T, K, k = self.coef_draws.shape
        cols: Dict[str, Any] = {
            "date": np.repeat(np.asarray(self.index), K * k),
            "equation": np.tile(np.repeat(self.var_names, k), T),
            "term": np.tile(self.terms, T * K),
        }
        cols.update(_bands(self.coef_draws, a))
        return pd.DataFrame(cols)

    def volatility(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Standard deviations of the structural shocks, ``sigma_{i,t}``.

        Shock ``i`` is the innovation of variable ``i`` that is orthogonal
        to the innovations of the variables ordered before it.

        Returns
        -------
        pd.DataFrame
            ``date``, ``variable``, ``median``, ``mean``, ``lower``,
            ``upper``.
        """
        a = self._alpha(alpha)
        _, T, K = self.logsig_draws.shape
        cols: Dict[str, Any] = {
            "date": np.repeat(np.asarray(self.index), K),
            "variable": np.tile(self.var_names, T),
        }
        cols.update(_bands(np.exp(self.logsig_draws.astype(float)), a))
        return pd.DataFrame(cols)

    def covariance(
        self, kind: str = "cov", alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """Reduced-form error covariance ``Omega_t`` at every date.

        Parameters
        ----------
        kind : {'cov', 'corr', 'sd'}
            Covariances, correlations, or the standard deviations of the
            reduced-form errors (the square roots of the diagonal).

        Returns
        -------
        pd.DataFrame
            ``date``, ``row``, ``col`` (``variable`` for ``'sd'``),
            ``median``, ``mean``, ``lower``, ``upper``.
        """
        a = self._alpha(alpha)
        kind = str(kind).lower()
        if kind not in ("cov", "corr", "sd"):
            raise MethodIncompatibility(
                f"tvp_var_sv: kind={kind!r} is not 'cov', 'corr' or 'sd'."
            )
        imp = self._impact()
        om = imp @ np.swapaxes(imp, -1, -2)
        T, K = om.shape[1], om.shape[2]
        sd = np.sqrt(np.einsum("ntii->nti", om))
        if kind == "sd":
            cols: Dict[str, Any] = {
                "date": np.repeat(np.asarray(self.index), K),
                "variable": np.tile(self.var_names, T),
            }
            cols.update(_bands(sd, a))
            return pd.DataFrame(cols)
        if kind == "corr":
            om = om / (sd[..., :, None] * sd[..., None, :])
        cols = {
            "date": np.repeat(np.asarray(self.index), K * K),
            "row": np.tile(np.repeat(self.var_names, K), T),
            "col": np.tile(self.var_names, T * K),
        }
        cols.update(_bands(om, a))
        return pd.DataFrame(cols)

    def irf_draws(
        self,
        at: Any = -1,
        periods: int = 10,
        shock_size: str = "sd",
        cumulative: bool = False,
    ) -> np.ndarray:
        """Posterior draws of the impulse responses at one date.

        Returns
        -------
        ndarray, shape (draws, periods + 1, K, K)
            ``[d, s, i, j]``: response of variable ``i`` after ``s``
            periods to structural shock ``j`` in draw ``d``. Differences
            of two such arrays (two dates) give the posterior of the
            change in a response.
        """
        if periods < 0:
            raise MethodIncompatibility("tvp_var_sv: periods must be non-negative.")
        size = str(shock_size).lower()
        if size not in ("sd", "unit"):
            raise MethodIncompatibility(
                f"tvp_var_sv: shock_size={shock_size!r} is not 'sd' or 'unit'."
            )
        pos = self._position(at)
        K, p = len(self.var_names), self.lags
        coef = np.asarray(self.coef_draws[:, pos], dtype=float)
        n = coef.shape[0]
        phi = np.zeros((periods + 1, n, K, K))
        phi[0] = np.eye(K)
        for s in range(1, periods + 1):
            for j in range(min(s, p)):
                phi[s] += phi[s - j - 1] @ coef[:, :, j * K : (j + 1) * K]
        imp = self._impact(pos)
        if size == "unit":
            imp = imp / np.exp(np.asarray(self.logsig_draws[:, pos], float))[:, None, :]
        out = np.swapaxes(phi @ imp, 0, 1)
        if cumulative:
            out = np.cumsum(out, axis=1)
        res: np.ndarray = out
        return res

    def irf(
        self,
        at: Any = -1,
        periods: int = 10,
        shock_size: str = "sd",
        cumulative: bool = False,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Posterior impulse responses of the VAR frozen at chosen dates.

        Identification is recursive in the order of the variables: the
        impact matrix at date ``t`` is ``A_t^{-1} diag(sigma_t)``. In each
        draw the coefficients and the impact matrix of the date are held
        fixed, so the responses answer "how would a shock propagate if the
        economy stayed as it was at that date".

        Parameters
        ----------
        at : label, int or list of them, default -1
            Dates: labels of ``index``, or integer positions among the
            estimation dates (negative from the end).
        periods : int, default 10
        shock_size : {'sd', 'unit'}
            ``'sd'``: a shock of one standard deviation of that date, so
            responses differ across dates also because the shocks differ
            in size. ``'unit'``: a shock that moves its own variable by
            one unit on impact at every date, which isolates changes in
            the transmission.
        cumulative : bool, default False
        alpha : float, optional

        Returns
        -------
        pd.DataFrame
            ``date``, ``shock``, ``response``, ``period``, ``median``,
            ``mean``, ``lower``, ``upper``.
        """
        a = self._alpha(alpha)
        many = isinstance(at, (list, tuple, np.ndarray, pd.Index))
        K = len(self.var_names)
        frames = []
        for one in list(at) if many else [at]:
            d = self.irf_draws(one, periods, shock_size, cumulative)
            # rows ordered shock, response, period as in TVPVARResult.irf
            cols: Dict[str, Any] = {
                "date": self.index[self._position(one)],
                "shock": np.repeat(self.var_names, K * (periods + 1)),
                "response": np.tile(np.repeat(self.var_names, periods + 1), K),
                "period": np.tile(np.arange(periods + 1), K * K),
            }
            cols.update(_bands(d.transpose(0, 3, 2, 1), a))
            frames.append(pd.DataFrame(cols))
        return pd.concat(frames, ignore_index=True)

    def stability(self, max_draws: int = 1000) -> pd.DataFrame:
        """Share of posterior draws with an explosive VAR at every date.

        Parameters
        ----------
        max_draws : int, default 1000
            Evenly spaced draws used (roots are computed for each draw and
            date).

        Returns
        -------
        pd.DataFrame
            Indexed by date: ``share_explosive`` (largest companion root
            of modulus 1 or more) and ``median_max_root``.
            ``attrs['share_any']`` is the share of draws that are
            explosive at one date or more.
        """
        if max_draws < 1:
            raise MethodIncompatibility("tvp_var_sv: max_draws must be positive.")
        key = f"stab{max_draws}"
        if key not in self._cache:
            n, T, K, _ = self.coef_draws.shape
            sel = np.unique(np.linspace(0, n - 1, min(n, max_draws)).astype(int))
            kp = K * self.lags
            comp = np.zeros((sel.size, T, kp, kp))
            comp[:, :, :K, :] = self.coef_draws[sel][:, :, :, :kp]
            if self.lags > 1:
                comp[:, :, K:, : kp - K] = np.eye(kp - K)
            roots = np.abs(np.linalg.eigvals(comp)).max(axis=-1)
            out = pd.DataFrame(
                {
                    "share_explosive": (roots >= 1.0).mean(axis=0),
                    "median_max_root": np.median(roots, axis=0),
                },
                index=self.index,
            )
            out.attrs["share_any"] = float((roots >= 1.0).any(axis=1).mean())
            out.attrs["n_draws"] = int(sel.size)
            self._cache[key] = out
        res: pd.DataFrame = self._cache[key]
        return res

    def _functionals(self) -> pd.DataFrame:
        T, K = self.n_obs, len(self.var_names)
        cols: Dict[str, np.ndarray] = {
            "tr(Q)": self.q_diag_draws.sum(axis=1),
            "tr(W)": np.einsum("nii->n", self.w_draws),
        }
        if self.s_draws.shape[1]:
            cols["tr(S)"] = np.einsum("nii->n", self.s_draws)
        for label, t in (("first", 0), ("mid", T // 2), ("last", T - 1)):
            for i, v in enumerate(self.var_names):
                cols[f"log sigma[{v}] {label}"] = self.logsig_draws[:, t, i]
        t = T // 2
        for i, v in enumerate(self.var_names):
            cols[f"{v}: {self.terms[i]} mid"] = self.coef_draws[:, t, i, i]
        for j in range(self.a_draws.shape[2]):
            cols[f"a[{j + 1}] mid"] = self.a_draws[:, t, j]
        del K
        return pd.DataFrame({c: np.asarray(v, dtype=float) for c, v in cols.items()})

    def diagnostics(self) -> pd.DataFrame:
        """Convergence statistics of the chain for summary functionals.

        The traces of ``Q``, ``S`` and ``W`` (the slowest-moving parts of
        the chain), the log volatilities at the first, middle and last
        date, and the own first lag coefficients and the covariance states
        at the middle date.

        Returns
        -------
        pd.DataFrame
            Indexed by functional: posterior ``mean``, ``ess`` (effective
            sample size, :func:`statspai.mcmc_ess`), ``inefficiency``
            (draws per effectively independent draw), ``geweke_z`` and
            ``geweke_p`` (:func:`statspai.geweke_diag`: equality of the
            means of the first 10% and the last 50% of the chain).
        """
        if "diag" not in self._cache:
            from ..mcmc.diagnostics import geweke_diag, mcmc_ess

            f = self._functionals()
            # a functional that never moves (a fixed state) has no ESS
            f = f.loc[:, f.std(axis=0) > 0.0]
            ess = mcmc_ess(f)
            gw = geweke_diag(f).table
            self._cache["diag"] = pd.DataFrame(
                {
                    "mean": f.mean(axis=0),
                    "ess": ess,
                    "inefficiency": f.shape[0] / ess,
                    "geweke_z": gw["z"],
                    "geweke_p": gw["p_value"],
                }
            )
        res: pd.DataFrame = self._cache["diag"]
        return res

    # ------------------------------------------------------------------ #
    def _medians(self) -> Tuple[np.ndarray, np.ndarray]:
        if "med" not in self._cache:
            self._cache["med"] = (
                np.median(self.coef_draws, axis=0),
                np.median(np.exp(self.logsig_draws.astype(float)), axis=0),
            )
        res: Tuple[np.ndarray, np.ndarray] = self._cache["med"]
        return res

    def summary(self) -> str:
        K, k = len(self.var_names), len(self.terms)
        info = self.model_info
        coef, vol = self._medians()
        stab = self.stability()
        diag = self.diagnostics()
        tr = info.get("training")
        lines = [
            "TVP-VAR with stochastic volatility (Gibbs sampler)",
            "=" * 66,
            f"Variables (recursive order): {', '.join(self.var_names)}"
            f"    Lags: {self.lags}",
            f"Estimation dates: {self.n_obs} ({self.index[0]} .. "
            f"{self.index[-1]})    Prior: "
            + (f"training sample of {tr} rows" if tr else "no training sample"),
            f"Draws kept: {self.coef_draws.shape[0]} (burn-in "
            f"{info.get('burnin')}, thin {info.get('thin')})",
            "",
            "Standard deviation of the structural shocks (posterior median)",
        ]
        rows = sorted({0, self.n_obs // 2, self.n_obs - 1})
        lines.append(
            pd.DataFrame(
                vol[rows], index=[self.index[r] for r in rows], columns=self.var_names
            ).to_string(float_format=lambda v: f"{v:.4g}")
        )
        ratio = vol.max(axis=0) / vol.min(axis=0)
        lines.append(
            "max / min over time: "
            + ", ".join(f"{v} {r:.2f}" for v, r in zip(self.var_names, ratio))
        )
        lo, hi = coef.min(axis=0), coef.max(axis=0)
        lines += ["", "Range over time of the posterior median coefficients"]
        lines.append(
            pd.DataFrame(
                [
                    [f"{lo[i, j]:.4g} .. {hi[i, j]:.4g}" for j in range(k)]
                    for i in range(K)
                ],
                index=self.var_names,
                columns=self.terms,
            ).T.to_string()
        )
        lines += [
            "",
            f"Explosive draws: {stab.attrs['share_any']:.1%} at some date; "
            f"largest share at one date {stab['share_explosive'].max():.1%}",
            "",
            "Convergence (ESS, inefficiency factor, Geweke z)",
            diag[["ess", "inefficiency", "geweke_z"]].to_string(
                float_format=lambda v: f"{v:.2f}"
            ),
        ]
        for note in info.get("notes", []):
            lines += ["", f"Note: {note}"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        coef, vol = self._medians()
        stab = self.stability()
        diag = self.diagnostics()
        return {
            "method": "tvp_var_sv",
            "var_names": list(self.var_names),
            "terms": list(self.terms),
            "lags": int(self.lags),
            "n_obs": int(self.n_obs),
            "n_draws": int(self.coef_draws.shape[0]),
            "training": self.model_info.get("training"),
            "volatility_median_first": vol[0].tolist(),
            "volatility_median_last": vol[-1].tolist(),
            "coefficients_median_last": coef[-1].tolist(),
            "share_explosive_any_date": float(stab.attrs["share_any"]),
            "min_ess": float(diag["ess"].min()),
            "max_abs_geweke_z": float(diag["geweke_z"].abs().max()),
            "notes": list(self.model_info.get("notes", [])),
        }

    def plot(self, what: str = "volatility", alpha: Optional[float] = None) -> Any:
        """Posterior median paths with equal-tailed bands.

        Parameters
        ----------
        what : {'volatility', 'coefficients', 'correlation'}
            Standard deviations of the structural shocks; coefficient
            paths (one column per equation); or the correlations of the
            reduced-form errors.
        """
        import matplotlib.pyplot as plt

        what = str(what).lower()
        t = np.asarray(self.index)
        K = len(self.var_names)
        if what == "volatility":
            tab = self.volatility(alpha)
            panels = [(v, tab[tab["variable"] == v]) for v in self.var_names]
            shape = (K, 1)
        elif what == "coefficients":
            tab = self.coefficients(alpha)
            panels = [
                (f"{e}: {tm}", tab[(tab["equation"] == e) & (tab["term"] == tm)])
                for tm in self.terms
                for e in self.var_names
            ]
            shape = (len(self.terms), K)
        elif what == "correlation":
            tab = self.covariance("corr", alpha)
            panels = [
                (f"corr({r}, {c})", tab[(tab["row"] == r) & (tab["col"] == c)])
                for i, r in enumerate(self.var_names)
                for c in self.var_names[:i]
            ]
            shape = (len(panels), 1)
        else:
            raise MethodIncompatibility(
                f"tvp_var_sv: what={what!r} is not 'volatility', "
                "'coefficients' or 'correlation'."
            )
        fig, axes = plt.subplots(
            shape[0],
            shape[1],
            figsize=(3.8 * shape[1] + 1.5, 2.0 * shape[0] + 0.5),
            squeeze=False,
            sharex=True,
        )
        for ax, (title, part) in zip(axes.reshape(-1), panels):
            ax.plot(t, part["median"].to_numpy(), lw=1.2)
            ax.fill_between(
                t, part["lower"].to_numpy(), part["upper"].to_numpy(), alpha=0.25
            )
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        return fig
