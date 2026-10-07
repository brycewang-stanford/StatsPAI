"""
Checking a fitted Bayesian model against the data: posterior predictive
checks, Bayesian R-squared and its leave-one-out version.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility, StatsPAIWarning
from . import _workflow as W
from .crossval import LOOResult, loo_predict

_STATS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda a: a.mean(axis=-1),
    "sd": lambda a: a.std(axis=-1, ddof=1),
    "var": lambda a: a.var(axis=-1, ddof=1),
    "min": lambda a: a.min(axis=-1),
    "max": lambda a: a.max(axis=-1),
    "median": lambda a: np.median(a, axis=-1),
    "prop_zero": lambda a: (a == 0).mean(axis=-1),
    "skew": lambda a: (
        ((a - a.mean(axis=-1, keepdims=True)) ** 3).mean(axis=-1) / a.std(axis=-1) ** 3
    ),
}


def _need_model(fit: Any, what: str) -> None:
    if not callable(getattr(fit, "posterior_predict", None)):
        raise MethodIncompatibility(
            f"{what} needs a fitted Bayesian regression "
            "(the result of sp.bayes_regress)."
        )


def _no_grouped(fit: Any, what: str) -> None:
    if getattr(fit, "_trials", None) is not None:
        raise MethodIncompatibility(
            f"{what} is defined for one outcome per row; this fit is on "
            "grouped binomial counts. Compare models with sp.loo."
        )


def _observed(fit: Any) -> np.ndarray:
    mdl = fit._model
    y = mdl.yi if fit.model in ("oprobit", "ologit", "mlogit") else mdl.y
    return np.asarray(y, dtype=float)


@dataclass
class PPCResult(ResultProtocolMixin):
    """Result of :func:`ppc`.

    Attributes
    ----------
    stat : str
        Name of the test statistic.
    observed : float
        Its value on the data.
    replicated : ndarray
        Its value on each replicated data set.
    p_value : float
        Share of replications with a statistic at least as large as the
        observed one. Values near 0 or 1 mean the model does not
        reproduce this feature of the data.
    y, y_rep : ndarray
        The observed outcome and the replications (one row each).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=80)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=80)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=500, burnin=200, seed=1)
    >>> out = sp.ppc(fit, stat="max", seed=1)
    >>> bool(0.02 < out.p_value < 0.98)
    True
    """

    stat: str
    observed: float
    replicated: np.ndarray
    p_value: float
    y: np.ndarray = field(repr=False)
    y_rep: np.ndarray = field(repr=False)

    _citation_keys = ("gabry2019visualization", "gelman2020regression")

    def to_dict(self) -> Dict[str, Any]:
        q = np.quantile(self.replicated, [0.025, 0.5, 0.975])
        return {
            "stat": self.stat,
            "observed": self.observed,
            "p_value": self.p_value,
            "replicated_median": float(q[1]),
            "replicated_interval": [float(q[0]), float(q[2])],
            "n_replications": int(self.replicated.size),
        }

    def summary(self) -> str:
        q = np.quantile(self.replicated, [0.025, 0.5, 0.975])
        return "\n".join(
            [
                f"Posterior predictive check: {self.stat}",
                f"Observed: {self.observed:.5g}",
                f"Replicated: median {q[1]:.5g}, 95% interval "
                f"[{q[0]:.5g}, {q[2]:.5g}] ({self.replicated.size} data sets)",
                f"P(T(y_rep) >= T(y)) = {self.p_value:.3f}",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, kind: str = "stat", n_overlay: int = 50) -> Any:
        """Draw the check.

        ``kind='stat'``: histogram of the replicated statistic with the
        observed value marked. ``kind='density'``: the distribution of
        the observed outcome over those of ``n_overlay`` replications.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.8))
        if kind == "stat":
            ax.hist(self.replicated, bins=30, alpha=0.6)
            ax.axvline(self.observed, linewidth=2, color="black")
            ax.set_xlabel(f"{self.stat} of replicated data (line: observed)")
        elif kind == "density":
            lo = min(self.y.min(), self.y_rep.min())
            hi = max(self.y.max(), self.y_rep.max())
            bins = np.linspace(lo, hi, 41)
            mids = 0.5 * (bins[1:] + bins[:-1])
            for row in self.y_rep[:n_overlay]:
                ax.plot(mids, np.histogram(row, bins, density=True)[0], alpha=0.25)
            ax.plot(
                mids,
                np.histogram(self.y, bins, density=True)[0],
                color="black",
                linewidth=2,
            )
            ax.set_xlabel("outcome (black: observed)")
        else:
            raise MethodIncompatibility("kind must be 'stat' or 'density'.")
        fig.tight_layout()
        return fig


def ppc(
    fit: Any,
    stat: Union[str, Callable[[np.ndarray], float]] = "mean",
    draws: Optional[int] = 1000,
    seed: Optional[int] = None,
) -> PPCResult:
    """Posterior predictive check of a fitted Bayesian regression.

    Simulates replicated data sets from the fitted model, one per
    posterior draw, and compares a statistic of each with the same
    statistic of the observed data. A model that cannot produce data
    looking like the data it was fitted to is wrong in that respect,
    whatever its coefficients say.

    Parameters
    ----------
    fit : fitted model
        A result of ``sp.bayes_regress``.
    stat : str or callable, default 'mean'
        ``'mean'``, ``'sd'``, ``'var'``, ``'min'``, ``'max'``,
        ``'median'``, ``'skew'``, ``'prop_zero'`` (share of zeros, the
        usual check of a count model), or a function of one data vector
        returning a number.
    draws : int, default 1000
        Number of replicated data sets (posterior draws used, evenly
        spaced). ``None`` uses every draw.
    seed : int, optional

    Returns
    -------
    PPCResult

    Notes
    -----
    Choose statistics the model does not fit directly. The mean of a
    Gaussian regression with an intercept is reproduced by construction
    and its check says nothing; the minimum, the maximum, the share of
    zeros or the skewness can fail. The p-value is a description of the
    discrepancy, not a calibrated test: it uses the data twice and is
    conservative.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = rng.negative_binomial(1, 0.2, size=200)
    >>> fit = sp.bayes_regress("y ~ x", df, model="poisson", draws=1000,
    ...                        burnin=300, seed=1)
    >>> bool(sp.ppc(fit, stat="prop_zero", seed=1).p_value < 0.05)
    True

    References
    ----------
    gabry2019visualization, gelman2020regression
    """
    _need_model(fit, "ppc")
    y = _observed(fit)
    total = len(fit.draws)
    use = None if draws is None else min(int(draws), total)
    y_rep = fit.posterior_predict(draws=use, seed=seed)
    if callable(stat):
        name = getattr(stat, "__name__", "statistic")
        observed = float(stat(y))
        replicated = np.array([float(stat(row)) for row in y_rep])
    else:
        name = str(stat).lower()
        if name not in _STATS:
            raise MethodIncompatibility(
                f"Unknown statistic {stat!r}. Available: "
                f"{', '.join(_STATS)}, or pass a function."
            )
        observed = float(_STATS[name](y))
        with np.errstate(divide="ignore", invalid="ignore"):
            replicated = np.asarray(_STATS[name](y_rep), dtype=float)
    return PPCResult(
        stat=name,
        observed=observed,
        replicated=replicated,
        p_value=float(np.mean(replicated >= observed)),
        y=y,
        y_rep=y_rep,
    )


@dataclass
class BayesR2Result(ResultProtocolMixin):
    """Result of :func:`bayes_r2` and :func:`loo_r2`.

    Attributes
    ----------
    kind : str
        ``'model'``, ``'residual'`` or ``'loo'``.
    draws : ndarray
        Posterior draws of R-squared (``kind='loo'``: Bayesian bootstrap
        draws around the point estimate).
    estimate : float
        Median of the draws; for ``kind='loo'`` the plug-in value
        ``1 - var(y - yhat_loo) / var(y)``.
    mean, sd, lower, upper : float
        Mean, standard deviation and central interval of the draws.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=100)})
    >>> df["y"] = df["x"] + rng.normal(size=100)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=1000, burnin=300, seed=1)
    >>> bool(0.3 < sp.bayes_r2(fit).estimate < 0.7)
    True
    """

    kind: str
    draws: np.ndarray = field(repr=False)
    estimate: float
    mean: float
    sd: float
    lower: float
    upper: float
    level: float = 0.95

    _citation_keys = ("gelman2019rsquared",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "estimate": self.estimate,
            "mean": self.mean,
            "sd": self.sd,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "n_draws": int(self.draws.size),
        }

    def summary(self) -> str:
        label = {
            "model": "Bayesian R-squared (model-based residual variance)",
            "residual": "Bayesian R-squared (variance of the residual draws)",
            "loo": "Leave-one-out R-squared",
        }[self.kind]
        pct = f"{100 * self.level:g}%"
        return (
            f"{label}\n"
            f"Estimate: {self.estimate:.4f}    sd: {self.sd:.4f}    "
            f"{pct} interval: [{self.lower:.4f}, {self.upper:.4f}]"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _r2_result(
    kind: str, draws: np.ndarray, estimate: float, level: float
) -> "BayesR2Result":
    lo = (1.0 - level) / 2.0
    return BayesR2Result(
        kind=kind,
        draws=draws,
        estimate=float(estimate),
        mean=float(draws.mean()),
        sd=float(draws.std(ddof=1)),
        lower=float(np.quantile(draws, lo)),
        upper=float(np.quantile(draws, 1.0 - lo)),
        level=level,
    )


def bayes_r2(fit: Any, kind: str = "model", level: float = 0.95) -> BayesR2Result:
    """Bayesian R-squared: the share of predictive variance explained.

    For each posterior draw, ``Var(fit) / (Var(fit) + Var(res))`` with
    ``Var(fit)`` the variance over observations of the expected outcome.
    Unlike the classical ratio it cannot exceed one when the prior or a
    small sample pulls the fit away from least squares, and it comes
    with a posterior distribution.

    Parameters
    ----------
    fit : fitted model
        A result of ``sp.bayes_regress``.
    kind : {'model', 'residual'}, default 'model'
        ``'model'``: ``Var(res)`` is the residual variance the model
        expects: ``sigma^2`` for the Gaussian model, the mean of
        ``mu (1 - mu)`` for binary outcomes, of ``mu`` for Poisson and of
        ``mu + alpha mu^2`` for the negative binomial. For Gaussian and
        binary models this is what R ``rstanarm::bayes_R2`` reports.
        ``'residual'``: ``Var(res)`` is the variance of ``y`` minus the
        expected outcome of that draw, the first definition in Gelman et
        al. (2019); it depends on the realised outcomes.
    level : float, default 0.95
        Mass of the reported interval.

    Returns
    -------
    BayesR2Result

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=100)})
    >>> df["y"] = df["x"] + rng.normal(size=100)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=1000, burnin=300, seed=1)
    >>> out = sp.bayes_r2(fit)
    >>> out.draws.shape
    (1000,)

    References
    ----------
    gelman2019rsquared
    """
    _need_model(fit, "bayes_r2")
    _no_grouped(fit, "bayes_r2")
    if fit.model in ("oprobit", "ologit", "mlogit", "quantile", "tobit"):
        raise MethodIncompatibility(
            f"R-squared is not defined for model='{fit.model}': the outcome "
            "scale has no variance for the model to explain."
        )
    mu = fit.posterior_epred()
    var_fit = mu.var(axis=1, ddof=1)
    if kind == "model":
        var_res = W.residual_variance(fit._model, fit.draws.to_numpy(), mu)
        if var_res is None:
            raise MethodIncompatibility(
                "The model has no finite residual variance; use kind='residual'."
            )
    elif kind == "residual":
        var_res = (_observed(fit)[None, :] - mu).var(axis=1, ddof=1)
    else:
        raise MethodIncompatibility("kind must be 'model' or 'residual'.")
    r2 = var_fit / (var_fit + var_res)
    return _r2_result(kind, r2, float(np.median(r2)), level)


def loo_r2(
    fit: Any,
    loo_result: Optional[LOOResult] = None,
    n_boot: int = 4000,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesR2Result:
    """Leave-one-out R-squared.

    ``1 - Var(y - yhat_loo) / Var(y)``, with ``yhat_loo`` the prediction
    of each observation from a posterior that did not see it
    (:func:`loo_predict`). It estimates the share of variance the model
    would explain in new data, so, unlike :func:`bayes_r2`, it goes down
    when a regressor adds noise and can be negative.

    Parameters
    ----------
    fit : fitted model
        A result of ``sp.bayes_regress``.
    loo_result : LOOResult, optional
        The output of ``sp.loo(fit)``; computed when omitted.
    n_boot : int, default 4000
        Bayesian bootstrap draws (Dirichlet weights on the observations)
        behind the interval. They describe sampling uncertainty in the
        two variances, not posterior uncertainty.
    seed : int, optional
    level : float, default 0.95

    Returns
    -------
    BayesR2Result
        ``estimate`` is the plug-in value and does not depend on the
        seed.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=100)})
    >>> df["y"] = df["x"] + rng.normal(size=100)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=1000, burnin=300, seed=1)
    >>> bool(sp.loo_r2(fit, seed=1).estimate < sp.bayes_r2(fit).estimate + 0.05)
    True

    References
    ----------
    gelman2019rsquared, vehtari2017practical
    """
    _need_model(fit, "loo_r2")
    _no_grouped(fit, "loo_r2")
    if fit.model in ("oprobit", "ologit", "mlogit", "quantile", "tobit"):
        raise MethodIncompatibility(
            f"R-squared is not defined for model='{fit.model}'."
        )
    y = _observed(fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatsPAIWarning)
        y_loo = loo_predict(fit, loo_result)
    err = y - y_loo
    n = y.size
    estimate = 1.0 - err.var(ddof=1) / y.var(ddof=1)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n), size=int(n_boot))
    scale = n / (n - 1.0)
    var_y = (w @ y**2 - (w @ y) ** 2) * scale
    var_e = (w @ err**2 - (w @ err) ** 2) * scale
    draws = np.clip(1.0 - var_e / var_y, -1.0, 1.0)
    return _r2_result("loo", draws, estimate, level)


def mad_sd(x: Any, axis: Optional[int] = 0) -> Any:
    """Median absolute deviation scaled to a standard deviation.

    ``1.4826 * median(|x - median(x)|)``: equal to the standard deviation
    for a normal sample and insensitive to the tails. Next to the median
    it is the robust summary of a set of simulation draws used throughout
    Gelman, Hill and Vehtari (2020).

    Parameters
    ----------
    x : array-like or DataFrame
    axis : int or None, default 0
        Axis to reduce; ``None`` flattens.

    Returns
    -------
    float, ndarray or Series

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> round(float(sp.mad_sd(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))), 4)
    1.4826

    References
    ----------
    gelman2020regression
    """
    if isinstance(x, pd.DataFrame):
        med = x.median(axis=0)
        return 1.4826 * (x - med).abs().median(axis=0)
    arr = np.asarray(x, dtype=float)
    med = np.median(arr, axis=axis, keepdims=True)
    out = 1.4826 * np.median(np.abs(arr - med), axis=axis)
    return float(out) if np.ndim(out) == 0 else out


__all__ = ["PPCResult", "BayesR2Result", "ppc", "bayes_r2", "loo_r2", "mad_sd"]
