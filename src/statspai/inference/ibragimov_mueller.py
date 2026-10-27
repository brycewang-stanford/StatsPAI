"""Ibragimov-Mueller t-statistic based inference with few groups.

Estimate the coefficient of interest separately in each of ``q`` groups
(clusters, spatial blocks, regions) and run the one-sample t-test on the
``q`` estimates with ``q - 1`` degrees of freedom. When the group
estimators are approximately independent and normal, with possibly
different variances, the test is conservative at the usual levels
(two-sided at most 0.083), however heterogeneous the groups are, and the
dependence inside each group is left completely unrestricted
[@ibragimov2010tstatistic; @ibragimov2016inference].
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

__all__ = ["ibragimov_mueller", "IbragimovMuellerResult"]

#: Two-sided level below which the group t-test is shown to be
#: conservative under heterogeneous group variances (Ibragimov and Mueller
#: 2010, building on Bakirov and Szekely 2006).
_SIZE_BOUND = 0.083


class IbragimovMuellerResult(ResultProtocolMixin):
    """Result of :func:`ibragimov_mueller`.

    Attributes
    ----------
    estimate : float
        Mean of the group estimates.
    se : float
        ``s / sqrt(q)``, with ``s`` the sample standard deviation of the
        group estimates.
    tstat, pvalue : float
        t statistic of ``H0: beta = h0`` and its two-sided p-value from
        ``t(q - 1)``.
    df : int
        ``q - 1``.
    ci : tuple of float
        ``estimate -/+ t_{q-1}(1 - alpha/2) * se``.
    group_estimates : DataFrame
        One row per group: ``group``, ``estimate``, ``n``.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.IbragimovMuellerResult.__name__
    'IbragimovMuellerResult'
    """

    def __init__(
        self,
        coef: str,
        group_estimates: pd.DataFrame,
        h0: float,
        alpha: float,
        estimator: str,
        cluster: str,
    ) -> None:
        b = group_estimates["estimate"].to_numpy(dtype=float)
        q = b.size
        self.coef = coef
        self.cluster = cluster
        self.estimator = estimator
        self.alpha = float(alpha)
        self.h0 = float(h0)
        self.q = int(q)
        self.df = int(q - 1)
        self.group_estimates = group_estimates
        self.estimate = float(b.mean())
        sd = float(b.std(ddof=1))
        self.sd_between = sd
        self.se = sd / np.sqrt(q)
        spread = max(float(np.max(np.abs(b))), 1.0)
        if not np.isfinite(sd) or sd <= 1e-12 * spread:
            raise DataInsufficient(
                f"ibragimov_mueller: the {q} group estimates of {coef!r} are "
                "identical, so the between-group t statistic is undefined.",
                recovery_hint="Use groups whose estimates vary (a coefficient "
                "fixed by design has no sampling variation to test).",
                diagnostics={"estimates": b.tolist()},
            )
        self.tstat = (self.estimate - self.h0) / self.se
        self.pvalue = float(2 * stats.t.sf(abs(self.tstat), self.df))
        crit = float(stats.t.ppf(1 - self.alpha / 2, self.df))
        self.critical_value = crit
        self.ci = (self.estimate - crit * self.se, self.estimate + crit * self.se)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coef": self.coef,
            "estimate": self.estimate,
            "se": self.se,
            "tstat": self.tstat,
            "df": self.df,
            "pvalue": self.pvalue,
            "ci_lower": self.ci[0],
            "ci_upper": self.ci[1],
            "alpha": self.alpha,
            "h0": self.h0,
            "q": self.q,
            "cluster": self.cluster,
            "estimator": self.estimator,
            "group_estimates": self.group_estimates.to_dict("records"),
        }

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "term": self.coef,
                    "estimate": self.estimate,
                    "std_error": self.se,
                    "statistic": self.tstat,
                    "df": self.df,
                    "p_value": self.pvalue,
                    "conf_low": self.ci[0],
                    "conf_high": self.ci[1],
                }
            ]
        )

    def summary(self) -> str:
        lines = [
            "Ibragimov-Mueller group t-test",
            "=" * 60,
            f"Coefficient : {self.coef}   estimator: {self.estimator}",
            f"Groups      : q = {self.q} ({self.cluster}), t({self.df}) reference",
            f"Estimate    : {self.estimate:.6g}   SE = {self.se:.6g}",
            f"H0: b = {self.h0:g}   t = {self.tstat:.4f}   p = {self.pvalue:.4g}",
            f"{100 * (1 - self.alpha):g}% CI   : [{self.ci[0]:.6g}, {self.ci[1]:.6g}]",
            "-" * 60,
            self.group_estimates.to_string(index=False),
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _fit_one(
    estimator: Union[str, Callable[[pd.DataFrame], Any]],
    formula: Optional[str],
    sub: pd.DataFrame,
    coef: Optional[str],
    fit_kwargs: Dict[str, Any],
    key: Any = None,
) -> float:
    if callable(estimator):
        out = estimator(sub)
    else:
        import statspai as sp

        fn = sp.regress if estimator == "ols" else sp.iv
        with warnings.catch_warnings():
            # group fits re-announce the same listwise deletion each time;
            # the totals are reported once on the result
            warnings.simplefilter("ignore", category=UserWarning)
            out = fn(formula, data=sub, **fit_kwargs)
    if isinstance(out, (int, float, np.floating, np.integer)):
        return float(out)
    params = getattr(out, "params", None)
    if params is None or coef is None:
        raise MethodIncompatibility(
            "ibragimov_mueller: the estimator returned neither a number nor a "
            "result with params[coef].",
            recovery_hint="Return a float from the callable, or a fitted result "
            "and pass coef=.",
        )
    if coef not in params.index:
        raise MethodIncompatibility(
            f"ibragimov_mueller: {coef!r} is not a coefficient of the fit in "
            f"group {key!r} (has {list(params.index)}); it may have been "
            "omitted as collinear there.",
            recovery_hint="Pass coef= as it appears in result.params, and use "
            "groups in which the regressor varies.",
        )
    return float(params[coef])


def ibragimov_mueller(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    *,
    cluster: str,
    coef: Optional[str] = None,
    estimator: Union[str, Callable[[pd.DataFrame], Any]] = "ols",
    h0: float = 0.0,
    alpha: float = 0.05,
    min_group_size: int = 2,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> IbragimovMuellerResult:
    """Ibragimov-Mueller group t-test for one coefficient.

    Splits the data into the ``q`` groups of ``cluster``, estimates the
    coefficient separately in each, and tests ``beta = h0`` with the
    ordinary one-sample t statistic of the ``q`` estimates against
    ``t(q - 1)``. Valid with very few groups (``q`` as small as 2) and
    arbitrary dependence and heterogeneity within groups, as long as the
    group estimates are approximately independent and normal; it is
    conservative at two-sided levels up to 0.083 even when the group
    estimates have different variances.

    Parameters
    ----------
    data : DataFrame
    formula : str, optional
        Model formula for ``estimator='ols'`` (``sp.regress``) or
        ``estimator='iv'`` (``sp.iv``, e.g. ``"y ~ (x ~ z) + w"``).
        Ignored, and must be left out, when ``estimator`` is a callable.
    cluster : str
        Column defining the groups (clusters, spatial blocks, regions).
        Rows with a missing group are refused.
    coef : str, optional
        Coefficient to test. Required unless the callable returns a float.
    estimator : {'ols', 'iv'} or callable, default 'ols'
        A callable receives each group's DataFrame and returns either the
        estimate (float) or a fitted result with ``params[coef]``.
    h0 : float, default 0.0
        Null value.
    alpha : float, default 0.05
        Level of the interval. A warning is issued above 0.083, where the
        size guarantee under heterogeneous groups does not hold.
    min_group_size : int, default 2
        Smallest group allowed; smaller groups raise.
    fit_kwargs : dict, optional
        Passed to ``sp.regress`` / ``sp.iv`` in each group (e.g.
        ``{"method": "liml"}``). Standard-error options are irrelevant:
        only the point estimate of each group is used.

    Returns
    -------
    IbragimovMuellerResult
        ``estimate``, ``se``, ``tstat``, ``df``, ``pvalue``, ``ci`` and
        ``group_estimates``; ``summary()``, ``tidy()``, ``to_dict()``.

    Raises
    ------
    DataInsufficient
        Fewer than two groups, a group below ``min_group_size``, or
        identical group estimates.
    MethodIncompatibility
        A group fit fails (the group is named), or bad arguments.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(8), 50)
    >>> x = rng.normal(size=400)
    >>> df = pd.DataFrame({"g": g, "x": x,
    ...                    "y": 0.5 * x + rng.normal(size=400) * (1 + g)})
    >>> res = sp.ibragimov_mueller(df, "y ~ x", cluster="g", coef="x")
    >>> res.df
    7
    >>> bool(res.ci[0] < res.estimate < res.ci[1])
    True

    References
    ----------
    [@ibragimov2010tstatistic], [@ibragimov2016inference]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "ibragimov_mueller: data must be a DataFrame.",
            recovery_hint="Pass a pandas DataFrame.",
        )
    if cluster not in data.columns:
        raise ColumnNotFound(
            f"ibragimov_mueller: cluster column {cluster!r} is not in data.",
            recovery_hint="Pass the name of the group column.",
        )
    if not (0.0 < float(alpha) < 1.0):
        raise MethodIncompatibility(
            f"ibragimov_mueller: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="Use alpha=0.05.",
        )
    if callable(estimator):
        if formula is not None:
            raise MethodIncompatibility(
                "ibragimov_mueller: formula= is not used with a callable " "estimator.",
                recovery_hint="Drop formula=; the callable fits each group.",
            )
        label = getattr(estimator, "__name__", "callable")
    else:
        if estimator not in ("ols", "iv"):
            raise MethodIncompatibility(
                f"ibragimov_mueller: estimator must be 'ols', 'iv' or a "
                f"callable, got {estimator!r}.",
                recovery_hint="estimator='ols' fits sp.regress, 'iv' sp.iv.",
            )
        if not isinstance(formula, str) or coef is None:
            raise MethodIncompatibility(
                "ibragimov_mueller: estimator='ols' / 'iv' needs formula= and "
                "coef=.",
                recovery_hint='e.g. formula="y ~ (x ~ z) + w", coef="x".',
            )
        label = "sp.regress" if estimator == "ols" else "sp.iv"
    if data[cluster].isna().any():
        raise MethodIncompatibility(
            f"ibragimov_mueller: {int(data[cluster].isna().sum())} row(s) have "
            f"no {cluster!r}; they belong to no group.",
            recovery_hint="Drop or assign those rows first.",
        )
    if float(alpha) > _SIZE_BOUND:
        warnings.warn(
            f"ibragimov_mueller: alpha={alpha} exceeds 0.083; the t(q-1) test "
            "is shown to be conservative under heterogeneous group variances "
            "only at two-sided levels up to about 0.083.",
            UserWarning,
            stacklevel=2,
        )
    kw = dict(fit_kwargs or {})
    rows = []
    for key, sub in data.groupby(cluster, sort=True):
        n_g = len(sub)
        if n_g < int(min_group_size):
            raise DataInsufficient(
                f"ibragimov_mueller: group {key!r} has {n_g} row(s) "
                f"(min_group_size={min_group_size}).",
                recovery_hint="Merge small groups into neighbours; each group "
                "must identify the coefficient on its own.",
                diagnostics={"group": key, "n": n_g},
            )
        try:
            est = _fit_one(estimator, formula, sub, coef, kw, key)
        except (MethodIncompatibility, DataInsufficient):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with the group
            raise MethodIncompatibility(
                f"ibragimov_mueller: the fit in group {key!r} failed: "
                f"{type(exc).__name__}: {exc}",
                recovery_hint="Every group must identify the coefficient "
                "(variation in the regressor and, for IV, a first stage); "
                "merge or drop groups that do not.",
                diagnostics={"group": key, "n": n_g},
            ) from exc
        if not np.isfinite(est):
            raise MethodIncompatibility(
                f"ibragimov_mueller: the estimate in group {key!r} is not " "finite.",
                recovery_hint="Check that the coefficient is identified in "
                "every group.",
                diagnostics={"group": key, "n": n_g},
            )
        rows.append({"group": key, "estimate": est, "n": n_g})
    if len(rows) < 2:
        raise DataInsufficient(
            f"ibragimov_mueller: {len(rows)} group(s) in {cluster!r}; the test "
            "needs at least two.",
            recovery_hint="Use a grouping with two or more groups.",
        )
    table = pd.DataFrame(rows)
    return IbragimovMuellerResult(
        coef=str(coef) if coef is not None else label,
        group_estimates=table,
        h0=h0,
        alpha=alpha,
        estimator=label,
        cluster=cluster,
    )
