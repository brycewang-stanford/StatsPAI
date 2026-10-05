"""The delete-one jackknife for an arbitrary statistic.

Recompute the statistic with each observation (or each cluster) left out;
the spread of the ``n`` leave-one-out values, scaled by ``(n - 1) / n``,
estimates the variance of the statistic. Unlike the bootstrap it involves
no random draws, so two programs given the same data agree to rounding.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError
from .bootstrap import _as_statistic_vector

__all__ = ["jackknife", "JackknifeResult"]


@dataclass
class JackknifeResult(ResultProtocolMixin):
    """Outcome of :func:`jackknife` for a scalar statistic.

    Attributes
    ----------
    estimate : float
        The statistic on the full sample.
    se : float
        Jackknife standard error.
    ci_lower, ci_upper : float
        ``estimate -/+ t(n_reps - 1) * se``.
    pvalue : float
        Two-sided p-value of ``estimate = 0`` against ``t(n_reps - 1)``.
    bias : float
        Jackknife estimate of the bias, ``(n - 1) * (mean of the
        leave-one-out values - estimate)``.
    n_reps : int
        Leave-one-out values that could be computed.
    replicates : numpy.ndarray
        The leave-one-out values.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": np.arange(10.0)})
    >>> res = sp.jackknife(df, lambda d: d["y"].mean())
    >>> res.n_reps
    10
    >>> bool(abs(res.se - df["y"].std() / 10 ** 0.5) < 1e-12)
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    bias: float
    alpha: float
    n_reps: int
    replicates: np.ndarray
    cluster: Optional[str] = None

    def summary(self) -> str:
        unit = f"clusters of {self.cluster}" if self.cluster else "observations"
        return "\n".join(
            [
                "Jackknife inference",
                f"  Estimate:   {self.estimate:.6f}",
                f"  Std. error: {self.se:.6f}",
                f"  CI ({1 - self.alpha:.0%}):    [{self.ci_lower:.6f}, "
                f"{self.ci_upper:.6f}]  (t, {self.n_reps - 1} df)",
                f"  p-value:    {self.pvalue:.4f}",
                f"  Bias:       {self.bias:.6f}",
                f"  Replications: {self.n_reps} ({unit})",
            ]
        )

    def __repr__(self) -> str:
        return self.summary()


def jackknife(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], Any],
    *,
    cluster: Optional[str] = None,
    mse: bool = False,
    alpha: float = 0.05,
) -> Union[JackknifeResult, EconometricResults]:
    """Delete-one jackknife standard errors for any statistic.

    Parameters
    ----------
    data : pandas.DataFrame
        The estimation sample: every row is left out once, so rows the
        statistic would not use (missing values, another subsample) must
        be removed first.
    statistic : callable
        A function of a DataFrame returning the statistic: a float, a
        vector (array / Series) or a fitted result with ``.params``.
        ``lambda d: sp.regress("y ~ x", data=d)`` is Stata's
        ``regress y x, vce(jackknife)``; a function of the coefficients,
        such as a ratio or a predicted level, is Stata's
        ``jackknife (exp): command``.
    cluster : str, optional
        Leave out one cluster at a time instead of one row.
    mse : bool, default False
        Centre the leave-one-out values on the full-sample statistic
        instead of on their own mean (Stata's ``mse`` option). The two
        differ by ``n - 1`` times the squared distance between the two
        centres.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    JackknifeResult or EconometricResults
        For a scalar statistic, a :class:`JackknifeResult`. For a vector,
        an ``EconometricResults`` with the full-sample values as
        coefficients, jackknife standard errors and covariance, and t
        inference with ``n - 1`` degrees of freedom; the leave-one-out
        values are in ``model_info['replicates']``.

    Notes
    -----
    The variance is ``(n - 1) / n * sum_i (theta_(i) - theta_bar)^2`` with
    ``n`` the number of rows (or clusters) whose leave-one-out value could
    be computed. For the coefficients of a linear regression it is close to
    the HC3 covariance. It is consistent for smooth functions of sample
    moments; it is **not** consistent for the median or other quantiles,
    for which the bootstrap should be used
    [@hansen2022econometrics, chapter 10].

    The cost is one evaluation of ``statistic`` per row or cluster.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=60)})
    >>> df["y"] = 1 + 2 * df.x + rng.normal(size=60)
    >>> fit = sp.jackknife(df, lambda d: sp.regress("y ~ x", data=d))
    >>> fit.model_info["n_reps"]
    60
    >>> ratio = sp.jackknife(
    ...     df, lambda d: (lambda b: b["Intercept"] / b["x"])(
    ...         sp.regress("y ~ x", data=d).params))
    >>> bool(ratio.se > 0)
    True

    References
    ----------
    hansen2022econometrics
    """
    if not callable(statistic):
        raise MethodIncompatibility(
            "sp.jackknife: statistic must be a function of a DataFrame.",
            recovery_hint="Pass e.g. lambda d: sp.regress('y ~ x', data=d).",
        )
    if cluster is not None and cluster not in data.columns:
        raise MethodIncompatibility(
            f"sp.jackknife: cluster={cluster!r} is not a column.",
            recovery_hint="Pass the name of the cluster variable.",
        )
    names, theta = _as_statistic_vector(statistic(data))
    scalar = names is None
    full = np.atleast_1d(np.asarray(theta, dtype=float))
    labels: List[str] = ["_jk_1"] if scalar else list(names or [])

    if cluster is None:
        groups = [np.array([i]) for i in range(len(data))]
    else:
        codes = pd.factorize(data[cluster])[0]
        if (codes < 0).any():
            raise MethodIncompatibility(
                "sp.jackknife: the cluster variable has missing values.",
                recovery_hint="Drop those rows first.",
            )
        groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    if len(groups) < 3:
        raise DataInsufficient(
            f"sp.jackknife: only {len(groups)} "
            + ("clusters" if cluster else "observations")
            + " to leave out.",
            recovery_hint="The jackknife needs at least three.",
        )

    keep = np.ones(len(data), dtype=bool)
    replicates = np.full((len(groups), full.size), np.nan)
    for g, rows in enumerate(groups):
        keep[rows] = False
        try:
            got_names, got = _as_statistic_vector(statistic(data.iloc[keep]))
        except (
            StatsPAIError,
            ArithmeticError,
            ValueError,
            KeyError,
            IndexError,
            np.linalg.LinAlgError,
        ):
            # a subsample the statistic cannot handle: counted and reported
            # below, as Stata reports a failed replication
            got_names, got = None, np.nan
        keep[rows] = True
        got = np.atleast_1d(np.asarray(got, dtype=float))
        if got.size == full.size and (scalar or got_names == labels):
            replicates[g] = got
    ok = np.isfinite(replicates).all(axis=1)
    failed = int((~ok).sum())
    if failed:
        warnings.warn(
            f"sp.jackknife: {failed} of {len(groups)} leave-one-out "
            "replications failed or changed the set of coefficients and "
            "were excluded.",
            RuntimeWarning,
            stacklevel=2,
        )
    replicates = replicates[ok]
    n = len(replicates)
    if n < 3:
        raise DataInsufficient(
            "sp.jackknife: fewer than three replications succeeded.",
            recovery_hint="Check that the statistic runs on a sample with "
            "one row removed.",
        )
    centre = full if mse else replicates.mean(axis=0)
    dev = replicates - centre
    cov = (n - 1) / n * dev.T @ dev
    se = np.sqrt(np.diag(cov))
    bias = (n - 1) * (replicates.mean(axis=0) - full)
    df = n - 1
    crit = float(stats.t.ppf(1 - alpha / 2, df))

    if scalar:
        t = full[0] / se[0] if se[0] > 0 else np.nan
        return JackknifeResult(
            estimate=float(full[0]),
            se=float(se[0]),
            ci_lower=float(full[0] - crit * se[0]),
            ci_upper=float(full[0] + crit * se[0]),
            pvalue=float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan,
            bias=float(bias[0]),
            alpha=alpha,
            n_reps=n,
            replicates=replicates[:, 0],
            cluster=cluster,
        )
    res = EconometricResults(
        params=pd.Series(full, index=labels),
        std_errors=pd.Series(se, index=labels),
        model_info={
            "model_type": "Jackknife",
            "method": "jackknife" + (f", cluster({cluster})" if cluster else ""),
            "n_reps": n,
            "n_failed": failed,
            "cluster": cluster,
            "mse": bool(mse),
            "bias": pd.Series(bias, index=labels),
            "replicates": pd.DataFrame(replicates, columns=labels),
            "alpha": alpha,
        },
        data_info={
            "nobs": int(len(data)),
            "var_cov": cov,
            "var_names": labels,
            "df_resid": df,
        },
        diagnostics={},
    )
    res.alpha = alpha
    res._compute_statistics()
    return res
