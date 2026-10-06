"""
Gaussian process regression: ``sp.gp_regress``.

    y_i = m + f(x_i) + e_i,   f ~ GP(0, s_f^2 k(x, x')),   e_i ~ N(0, s_n^2)

The regression function is given a prior directly, so no functional form
is chosen. Given the kernel's hyperparameters the posterior of ``f`` is
Gaussian in closed form. The hyperparameters are estimated by maximising
the marginal likelihood, or fixed by the user.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import design_for

_KERNELS = ("rbf", "matern32", "matern52")


def _kernel(
    kind: str, A: np.ndarray, B: np.ndarray, length: np.ndarray, signal_var: float
) -> np.ndarray:
    """``signal_var * k(|a - b| / length)`` for the rows of ``A`` and ``B``."""
    a = A / length
    b = B / length
    d2 = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T
    np.maximum(d2, 0.0, out=d2)
    if kind == "rbf":
        return np.asarray(signal_var * np.exp(-0.5 * d2))
    d = np.sqrt(d2)
    if kind == "matern32":
        s = np.sqrt(3.0) * d
        return np.asarray(signal_var * (1.0 + s) * np.exp(-s))
    s = np.sqrt(5.0) * d
    return np.asarray(signal_var * (1.0 + s + s * s / 3.0) * np.exp(-s))


def _fit_pieces(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    length: np.ndarray,
    signal_var: float,
    noise_var: float,
    mean: Optional[float],
    reml: bool = True,
) -> Dict[str, Any]:
    """Cholesky factor, mean, weights and log marginal likelihood."""
    n = y.size
    K = _kernel(kind, X, X, length, signal_var)
    K[np.diag_indices(n)] += noise_var
    chol = linalg.cholesky(K, lower=True)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    if mean is None:
        ki1 = linalg.cho_solve((chol, True), np.ones(n))
        s11 = float(ki1.sum())
        m = float(ki1 @ y) / s11
        resid = y - m
        w = linalg.cho_solve((chol, True), resid)
        if reml:
            # restricted likelihood: the constant has a flat prior
            lml = (
                -0.5 * float(resid @ w)
                - 0.5 * logdet
                - 0.5 * np.log(s11)
                - 0.5 * (n - 1) * np.log(2.0 * np.pi)
            )
        else:
            # profile likelihood: the constant is replaced by its estimate
            lml = -0.5 * float(resid @ w) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
    else:
        m, ki1, s11 = float(mean), None, None
        resid = y - m
        w = linalg.cho_solve((chol, True), resid)
        lml = -0.5 * float(resid @ w) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
    return {"chol": chol, "mean": m, "w": w, "lml": float(lml), "ki1": ki1, "s11": s11}


@dataclass
class GPResult(ResultProtocolMixin):
    """Result of :func:`gp_regress`.

    Attributes
    ----------
    params : pd.Series
        The hyperparameters: one ``length_scale[<regressor>]`` per
        regressor (or a single ``length_scale``), ``signal_var``,
        ``noise_var`` and the constant ``mean``.
    fitted : pd.Series
        Posterior mean of the regression function at the sample points.
    loglik : float
        Log marginal likelihood at the hyperparameters. When the constant
        is estimated this is the restricted likelihood, which is not
        comparable with a fit that fixes the mean.
    kernel, formula, n_obs, level

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.sort(rng.uniform(0, 6, 80))
    >>> df = pd.DataFrame({"x": x, "y": np.sin(x) + 0.2 * rng.normal(size=80)})
    >>> fit = sp.gp_regress("y ~ x", df)
    >>> isinstance(fit, sp.GPResult)
    True
    >>> fit.predict(pd.DataFrame({"x": [1.0, 2.0]})).shape
    (2, 4)
    """

    params: pd.Series
    fitted: pd.Series
    loglik: float
    kernel: str
    formula: str
    n_obs: int
    level: float = 0.95
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("rasmussen2005gaussian",)

    @property
    def coef(self) -> pd.Series:
        return self.params

    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood at the hyperparameters."""
        return float(self.loglik)

    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        noise: bool = False,
        level: Optional[float] = None,
    ) -> pd.DataFrame:
        """Posterior mean and interval of the regression function.

        Parameters
        ----------
        newdata : DataFrame, optional
            Points at which to predict. Default: the sample.
        noise : bool, default False
            ``False`` gives the interval of the regression function,
            ``True`` that of a new observation (adds the noise variance).
        level : float, optional
            Mass of the interval. Default: the level of the fit.

        Returns
        -------
        DataFrame
            Columns ``mean``, ``sd``, ``lower``, ``upper``.
        """
        st = self._state
        if newdata is None:
            Xs, index = st["X"], st["index"]
        else:
            Xs = design_for(st["design_info"], st["all_names"], newdata)[:, st["keep"]]
            index = newdata.index
        lv = self.level if level is None else float(level)
        if not 0.0 < lv < 1.0:
            raise MethodIncompatibility(f"level must be in (0, 1); got {lv}.")
        ks = _kernel(self.kernel, Xs, st["X"], st["length"], st["signal_var"])
        mean = st["mean"] + ks @ st["w"]
        v = linalg.solve_triangular(st["chol"], ks.T, lower=True)
        var = st["signal_var"] - (v * v).sum(axis=0)
        if st["ki1"] is not None:
            # the estimated constant adds its own uncertainty
            var = var + (1.0 - ks @ st["ki1"]) ** 2 / st["s11"]
        var = np.maximum(var, 0.0) + (st["noise_var"] if noise else 0.0)
        sd = np.sqrt(var)
        z = stats.norm.ppf(0.5 + lv / 2.0)
        return pd.DataFrame(
            {"mean": mean, "sd": sd, "lower": mean - z * sd, "upper": mean + z * sd},
            index=index,
        )

    def expected_improvement(
        self,
        newdata: pd.DataFrame,
        minimize: bool = True,
        best: Optional[float] = None,
    ) -> pd.Series:
        """Expected improvement over the best value seen so far.

        The criterion of efficient global optimisation (Jones, Schonlau
        and Welch 1998): large where the predicted value is good, or
        where it is uncertain. The next point to evaluate is the one
        that maximises it.

        Parameters
        ----------
        newdata : DataFrame
            Candidate points.
        minimize : bool, default True
            Whether small or large outcomes are wanted.
        best : float, optional
            The value to improve on. Default: the smallest (largest)
            outcome in the sample.

        Returns
        -------
        Series
            ``E[max(best - Y(x), 0)]`` for each candidate (``Y(x) -
            best`` when maximising), from the posterior of the
            regression function.
        """
        pred = self.predict(newdata)
        info = self.model_info
        if best is None:
            best = info["y_min"] if minimize else info["y_max"]
        sd = np.maximum(pred["sd"].to_numpy(), 1e-12)
        gap = (best - pred["mean"].to_numpy()) * (1.0 if minimize else -1.0)
        u = gap / sd
        ei = sd * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        return pd.Series(ei, index=pred.index, name="expected_improvement")

    def summary(self) -> str:
        lines = [
            f"Gaussian process regression ({self.kernel} kernel)    {self.formula}",
            f"Observations: {self.n_obs}    "
            f"Log marginal likelihood: {self.loglik:.4f}",
            f"Hyperparameters: {self.model_info.get('estimation', '')}",
            "",
            self.params.to_string(float_format=lambda v: f"{v:.5g}"),
        ]
        for note in self.model_info.get("notes", []):
            lines.append("")
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "formula": self.formula,
            "n_obs": int(self.n_obs),
            "loglik": float(self.loglik),
            "params": {str(k): float(v) for k, v in self.params.items()},
            "model_info": dict(self.model_info),
        }


def gp_regress(
    formula: str,
    data: pd.DataFrame,
    kernel: str = "rbf",
    ard: bool = True,
    length_scale: Any = None,
    signal_var: Optional[float] = None,
    noise_var: Optional[float] = None,
    mean: Optional[float] = None,
    optimize_hyper: bool = True,
    restarts: int = 4,
    seed: Optional[int] = None,
    level: float = 0.95,
    interpolate: bool = False,
    likelihood: str = "reml",
) -> GPResult:
    """Gaussian process regression.

    A nonparametric regression with honest uncertainty bands: the
    regression function has a Gaussian process prior, and its posterior
    at any point is normal with a closed-form mean and variance.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"``. The regressors enter the kernel; the intercept
        of the formula is the constant mean of the process.
    data : DataFrame
    kernel : {'rbf', 'matern52', 'matern32'}, default 'rbf'
        ``'rbf'`` (squared exponential) gives infinitely smooth functions,
        the Matern kernels rougher ones.
    ard : bool, default True
        One length scale per regressor. With ``False`` a single length
        scale is shared, which only makes sense for regressors on a
        common scale.
    length_scale, signal_var, noise_var : optional
        Starting values, or the values used when ``optimize_hyper=False``.
        Defaults: the standard deviation of each regressor, and half the
        variance of the outcome for each variance.
    mean : float, optional
        A known constant mean. By default the constant is estimated with
        a flat prior and its uncertainty enters the bands.
    optimize_hyper : bool, default True
        Maximise the marginal likelihood over the hyperparameters.
    restarts : int, default 4
        Additional random starting points; the marginal likelihood can
        have several local maxima.
    seed : int, optional
        For the random starting points.
    level : float, default 0.95
    interpolate : bool, default False
        The outcome has no noise (the output of a deterministic computer
        model or simulator with fixed random numbers): the fit passes
        through the data and the bands shrink to zero at the sample
        points. The noise variance is fixed at a negligible value that
        keeps the kernel matrix invertible instead of being estimated.
    likelihood : {'reml', 'ml'}, default 'reml'
        How the constant mean is treated when the hyperparameters are
        estimated. ``'reml'`` integrates it out under a flat prior;
        ``'ml'`` replaces it by its estimate (the profile likelihood, the
        convention of kriging software such as R's ``rkriging`` and
        ``DiceKriging``). They differ in small samples; REML gives
        somewhat larger variances.

    Returns
    -------
    GPResult
        ``params`` holds the hyperparameters, ``predict()`` the posterior
        mean and bands of the regression function at any points,
        ``expected_improvement()`` the criterion for choosing the next
        point when the function is being minimised or maximised.

    Notes
    -----
    The cost grows with the cube of the sample size; a few thousand
    observations are the practical limit.

    The bands condition on the estimated hyperparameters and do not carry
    their uncertainty, so they are somewhat too narrow in small samples.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.sort(rng.uniform(0, 6, 80))
    >>> df = pd.DataFrame({"x": x, "y": np.sin(x) + 0.2 * rng.normal(size=80)})
    >>> fit = sp.gp_regress("y ~ x", df)
    >>> list(fit.params.index)
    ['length_scale[x]', 'signal_var', 'noise_var', 'mean']
    >>> band = fit.predict(pd.DataFrame({"x": [1.5]}))
    >>> bool(band["lower"].iloc[0] < np.sin(1.5) < band["upper"].iloc[0])
    True

    References
    ----------
    rasmussen2005gaussian
    """
    kind = str(kernel).lower().replace("-", "").replace("_", "")
    kind = {"se": "rbf", "squaredexponential": "rbf", "matern": "matern52"}.get(
        kind, kind
    )
    if kind not in _KERNELS:
        raise MethodIncompatibility(
            f"kernel must be one of {', '.join(_KERNELS)}; got {kernel!r}."
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    lik = str(likelihood).lower()
    if lik not in ("reml", "ml"):
        raise MethodIncompatibility(
            f"likelihood must be 'reml' or 'ml'; got {likelihood!r}."
        )
    reml = lik == "reml"
    if interpolate and noise_var is not None:
        raise MethodIncompatibility(
            "interpolate=True fixes the noise variance; do not pass noise_var."
        )
    y_df, X_df = create_design_matrices(formula, data)
    y = np.asarray(y_df, dtype=float).reshape(-1)
    Xall = np.asarray(X_df, dtype=float)
    all_names = [str(c) for c in X_df.columns]
    keep = [j for j in range(Xall.shape[1]) if np.ptp(Xall[:, j]) > 0]
    if not keep:
        raise MethodIncompatibility("The formula has no regressor that varies.")
    X = Xall[:, keep]
    names = [all_names[j] for j in keep]
    n, k = X.shape
    if n < 5:
        raise DataInsufficient(f"{n} observations are too few.")
    vy = float(y.var())
    if vy <= 0:
        raise MethodIncompatibility("The outcome does not vary.")
    notes: List[str] = []
    if n > 3000:
        notes.append(
            f"{n} observations: the cost grows with n cubed. Consider a "
            "random subsample for the hyperparameters."
        )
    n_len = k if ard else 1
    if length_scale is None:
        len0 = X.std(axis=0) if ard else np.array([float(np.mean(X.std(axis=0)))])
    else:
        len0 = np.broadcast_to(np.asarray(length_scale, dtype=float), (n_len,)).copy()
    sf0 = 0.5 * vy if signal_var is None else float(signal_var)
    sn0 = 0.5 * vy if noise_var is None else float(noise_var)
    jitter = 1e-8 * vy
    if interpolate:
        sn0 = jitter
        if signal_var is None:
            sf0 = vy
    if np.any(len0 <= 0) or sf0 <= 0 or sn0 < 0:
        raise MethodIncompatibility(
            "length_scale and signal_var must be positive, noise_var non-negative."
        )

    def expand(lv: np.ndarray) -> np.ndarray:
        return lv if ard else np.repeat(lv, k)

    def neg(theta: np.ndarray) -> float:
        try:
            out = _fit_pieces(
                kind,
                X,
                y,
                expand(np.exp(theta[:n_len])),
                float(np.exp(theta[n_len])),
                jitter if interpolate else float(np.exp(theta[n_len + 1])),
                mean,
                reml,
            )
        except linalg.LinAlgError:
            return 1e25
        return -out["lml"] if np.isfinite(out["lml"]) else 1e25

    estimation = "fixed by the user"
    length, sf, sn = len0, sf0, sn0
    if optimize_hyper:
        rng = np.random.default_rng(seed)
        start = np.log(np.r_[len0, sf0, max(sn0, 1e-8 * vy)])
        starts = [start] + [
            start + rng.normal(0.0, 1.0, start.size)
            for _ in range(max(int(restarts), 0))
        ]
        # The likelihood is flat where the length scale is far below the
        # spacing of the data (every point is then its own island), and an
        # optimiser started there does not leave. So the surface is first
        # scanned over a coarse grid of length scales and noise shares,
        # and the best two points of the scan are added as starts.
        if length_scale is None:
            scan = []
            for mult in (3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
                for share in (0.5, 0.05, 0.005):
                    cand = np.log(
                        np.r_[
                            mult * len0,
                            vy * (1.0 if interpolate else 1.0 - share),
                            max(share * vy, 1e-8 * vy),
                        ]
                    )
                    scan.append((neg(cand), cand))
                    if interpolate:
                        break
            scan.sort(key=lambda item: item[0])
            starts += [cand for val, cand in scan[:2] if val < 1e24]
        lo = np.log(np.r_[np.full(n_len, 1e-3) * len0, 1e-6 * vy, 1e-10 * vy])
        hi = np.log(np.r_[np.full(n_len, 1e4) * len0, 1e6 * vy, 1e3 * vy])
        best = None
        for s0 in starts:
            sol = optimize.minimize(
                neg, np.clip(s0, lo, hi), method="L-BFGS-B", bounds=list(zip(lo, hi))
            )
            if best is None or sol.fun < best.fun:
                best = sol
        assert best is not None
        if not np.isfinite(best.fun) or best.fun >= 1e24:
            raise MethodIncompatibility(
                "The marginal likelihood could not be evaluated; the kernel "
                "matrix is numerically singular. Check for duplicated rows "
                "with different outcomes and no noise."
            )
        th = best.x
        length, sf, sn = (
            np.exp(th[:n_len]),
            float(np.exp(th[n_len])),
            jitter if interpolate else float(np.exp(th[n_len + 1])),
        )
        estimation = "maximum marginal likelihood"
        at_bound = np.isclose(th, lo, atol=1e-6) | np.isclose(th, hi, atol=1e-6)
        if at_bound[:n_len].any():
            flat = [
                nm
                for nm, hit in zip(names if ard else ["all"], at_bound[:n_len])
                if hit
            ]
            notes.append(
                "Length scale at its bound for "
                + ", ".join(flat)
                + ": a very large value means the regressor does not matter, "
                "a very small one that the fit interpolates noise."
            )
    gaps = []
    for j in range(k):
        u = np.unique(X[:, j])
        gaps.append(float(np.diff(u).min()) if u.size > 1 else np.inf)
    short = [nm for nm, lj, g in zip(names, expand(length), gaps) if lj < g / 4.0]
    if short and k == 1:
        notes.append(
            "The length scale is below a quarter of the smallest gap between "
            "the observed values of "
            + ", ".join(short)
            + ": the fit says nothing between the data points and reverts to "
            "the mean there. With replicated or widely spaced designs the "
            "likelihood can prefer this; fix length_scale if a smooth "
            "function is expected."
        )
    pieces = _fit_pieces(kind, X, y, expand(length), sf, sn, mean, reml)
    labels = [f"length_scale[{nm}]" for nm in names] if ard else ["length_scale"]
    params = pd.Series(
        np.r_[length, sf, sn, pieces["mean"]],
        index=labels + ["signal_var", "noise_var", "mean"],
    )
    state = {
        "X": X,
        "index": X_df.index,
        "design_info": getattr(X_df, "design_info", None),
        "all_names": all_names,
        "keep": keep,
        "length": expand(length),
        "signal_var": sf,
        "noise_var": sn,
        "mean": pieces["mean"],
        "w": pieces["w"],
        "chol": pieces["chol"],
        "ki1": pieces["ki1"],
        "s11": pieces["s11"],
    }
    fitted = pieces["mean"] + (_kernel(kind, X, X, expand(length), sf) @ pieces["w"])
    res = GPResult(
        params=params,
        fitted=pd.Series(fitted, index=X_df.index, name="fitted"),
        loglik=pieces["lml"],
        kernel=kind,
        formula=formula,
        n_obs=n,
        level=level,
        model_info={
            "estimation": estimation,
            "ard": bool(ard),
            "mean": "estimated (flat prior)" if mean is None else "fixed",
            "interpolate": bool(interpolate),
            "likelihood": lik,
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "notes": notes,
            "regressors": names,
        },
        _state=state,
    )
    for note in notes:
        warnings.warn(note, ConvergenceWarning, stacklevel=2)
    return res
