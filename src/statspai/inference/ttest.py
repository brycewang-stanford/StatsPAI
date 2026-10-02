"""Student / Welch t tests on means: one sample, two samples, paired.

The estimand is a mean or a difference of two means. Stata's ``ttest`` and
R's ``t.test`` are the references; the conventions that differ between them
are parameters here rather than defaults to guess:

* The two-sample test pools the variances unless ``unequal=True`` (Stata's
  default; R's is the reverse).
* With unequal variances the degrees of freedom are Satterthwaite's, which
  is also what R's ``t.test`` reports and calls Welch's. Stata's ``welch``
  option is a different approximation (Welch 1947), selected by
  ``welch=True``.
* A ``by=`` difference is the mean of the group with the smaller value
  minus the mean of the group with the larger one, as in Stata.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = ["ttest", "TTestResult"]


class TTestResult(ResultProtocolMixin):
    """Outcome of :func:`ttest`.

    Attributes
    ----------
    estimate : float
        The sample mean (one sample), the mean of the differences (paired),
        or the difference between the two group means.
    se, statistic, df : float
        Standard error of ``estimate``, the t statistic for
        ``H0: estimate == null`` and its degrees of freedom.
    pvalue : float
        Two-sided p-value. ``pvalue_less`` and ``pvalue_greater`` are the
        one-sided ones (``Ha: diff < null`` and ``Ha: diff > null``).
    ci : tuple of float
        ``1 - alpha`` confidence interval for ``estimate``.
    groups : pandas.DataFrame
        One row per sample (and ``combined`` for two unpaired samples) with
        ``n``, ``mean``, ``se``, ``sd``, ``ci_lower`` and ``ci_upper``.
    method : str
        Which test was run.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"g": np.repeat([0, 1], 50)})
    >>> df["y"] = 1.0 + 0.5 * df.g + rng.normal(size=100)
    >>> res = sp.ttest(df, "y", by="g", unequal=True)
    >>> type(res).__name__
    'TTestResult'
    >>> list(res.groups.index)
    ['0', '1', 'combined']
    >>> bool(res.ci[0] < res.estimate < res.ci[1])
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        method: str,
        estimate: float,
        se: float,
        statistic: float,
        df: float,
        pvalue: float,
        pvalue_less: float,
        pvalue_greater: float,
        ci: Tuple[float, float],
        null: float,
        alpha: float,
        n_obs: int,
        groups: pd.DataFrame,
    ) -> None:
        self.method = method
        self.estimand = "difference in means" if len(groups) > 1 else "mean"
        self.estimate = estimate
        self.se = se
        self.statistic = statistic
        self.df = df
        self.pvalue = pvalue
        self.pvalue_less = pvalue_less
        self.pvalue_greater = pvalue_greater
        self.ci = ci
        self.null = null
        self.alpha = alpha
        self.n_obs = n_obs
        self.groups = groups

    def summary(self) -> str:
        level = 100 * (1 - self.alpha)
        table = self.groups.rename(
            columns={
                "n": "Obs",
                "mean": "Mean",
                "se": "Std. err.",
                "sd": "Std. dev.",
                "ci_lower": f"[{level:g}% conf.",
                "ci_upper": "interval]",
            }
        )
        lines = [
            self.method,
            "=" * len(self.method),
            table.to_string(float_format=lambda v: f"{v:.6g}"),
            "",
            f"{self.estimand} = {self.estimate:.6g}   (se {self.se:.6g})",
            f"t = {self.statistic:.4f},  degrees of freedom = {self.df:.6g}",
            f"H0: {self.estimand} = {self.null:g}",
            f"  Ha: <  p = {self.pvalue_less:.4f}",
            f"  Ha: != p = {self.pvalue:.4f}",
            f"  Ha: >  p = {self.pvalue_greater:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TTestResult({self.method}: estimate={self.estimate:.6g}, "
            f"t={self.statistic:.4f}, df={self.df:.6g}, p={self.pvalue:.4g})"
        )


def _row(x: np.ndarray, alpha: float) -> Dict[str, float]:
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    half = stats.t.ppf(1 - alpha / 2, n - 1) * se if n > 1 else np.nan
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "sd": sd,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
    }


def _column(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        raise MethodIncompatibility(
            f"ttest: column {name!r} is not in data.",
            recovery_hint="Check the column name.",
            diagnostics={"columns": [str(c) for c in data.columns][:20]},
        )
    col = data[name]
    if not pd.api.types.is_numeric_dtype(col):
        raise MethodIncompatibility(
            f"ttest: column {name!r} is not numeric.",
            recovery_hint="Encode it as a number first, or use sp.tab for "
            "categorical variables.",
        )
    return col.astype(float)


def ttest(
    data: pd.DataFrame,
    y: str,
    by: Optional[str] = None,
    *,
    other: Optional[str] = None,
    mu: float = 0.0,
    paired: bool = True,
    unequal: bool = False,
    welch: bool = False,
    alpha: float = 0.05,
) -> TTestResult:
    """t test on a mean, a paired difference, or a difference of two means.

    Equivalent to Stata's ``ttest`` and R's ``t.test``.

    ======================================  ===============================
    Call                                    Stata
    ======================================  ===============================
    ``sp.ttest(df, "y", mu=5)``             ``ttest y == 5``
    ``sp.ttest(df, "y", by="g")``           ``ttest y, by(g)``
    ``sp.ttest(df, "y", other="x")``        ``ttest y == x``
    ``sp.ttest(df, "y", other="x",          ``ttest y == x, unpaired``
    paired=False)``
    ======================================  ===============================

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        The variable whose mean is tested.
    by : str, optional
        Grouping column with exactly two distinct non-missing values. The
        estimate is the mean of ``y`` in the group with the smaller value
        minus the mean in the group with the larger one.
    other : str, optional
        A second variable. With ``paired=True`` (default) the test is on the
        mean of ``y - other`` over rows where both are observed; with
        ``paired=False`` the two columns are independent samples, each using
        its own non-missing rows.
    mu : float, default 0.0
        The null value: of the mean for one sample, of the difference
        otherwise.
    paired : bool, default True
        Only used with ``other=``.
    unequal : bool, default False
        Two unpaired samples: do not pool the variances. The degrees of
        freedom are Satterthwaite's (R's ``t.test`` default).
    welch : bool, default False
        Two unpaired samples: unequal variances with Welch's (1947) degrees
        of freedom instead of Satterthwaite's, as Stata's ``welch`` option.
        Implies ``unequal=True``.
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level of the intervals.

    Returns
    -------
    TTestResult

    Raises
    ------
    MethodIncompatibility
        When ``by`` does not have exactly two groups, when ``by`` and
        ``other`` are both given, or when a sample has fewer than two
        observations.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"g": np.repeat([0, 1], 50)})
    >>> df["y"] = 1.0 + 0.5 * df.g + rng.normal(size=100)
    >>> res = sp.ttest(df, "y", by="g", unequal=True)
    >>> bool(res.pvalue < 0.05)
    True
    >>> one = sp.ttest(df, "y", mu=1.0)
    >>> one.method
    'One-sample t test'
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"ttest: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is 1 minus the confidence level, e.g. 0.05.",
        )
    if by is not None and other is not None:
        raise MethodIncompatibility(
            "ttest: pass by= (one variable, two groups) or other= (two "
            "variables), not both.",
            recovery_hint="Drop one of the two arguments.",
        )
    unequal = bool(unequal or welch)
    yv = _column(data, y)

    def too_small(label: str) -> MethodIncompatibility:
        return MethodIncompatibility(
            f"ttest: {label} has fewer than two non-missing observations, so "
            "its variance is not defined.",
            recovery_hint="Check the sample and the grouping column.",
        )

    # ---------------------------------------------- one sample / paired
    if by is None and (other is None or paired):
        if other is None:
            x = yv.dropna().to_numpy()
            method, labels = "One-sample t test", {y: x}
        else:
            ov = _column(data, other)
            keep = yv.notna() & ov.notna()
            x = (yv[keep] - ov[keep]).to_numpy()
            method = "Paired t test"
            labels = {y: yv[keep].to_numpy(), other: ov[keep].to_numpy(), "diff": x}
        if x.size < 2:
            raise too_small(y)
        rows = {k: _row(v, alpha) for k, v in labels.items()}
        base = rows["diff"] if other is not None else rows[y]
        est, se, dof = base["mean"], base["se"], float(x.size - 1)
        n_obs = int(x.size)
        groups = pd.DataFrame.from_dict(rows, orient="index")
    # ---------------------------------------------- two unpaired samples
    else:
        if by is not None:
            gv = data[by]
            keep = yv.notna() & gv.notna()
            levels = sorted(pd.unique(gv[keep]))
            if len(levels) != 2:
                raise MethodIncompatibility(
                    f"ttest: by={by!r} must have exactly two groups; found "
                    f"{len(levels)}.",
                    recovery_hint="Restrict the data to two groups, or use "
                    "sp.regress with C(group) for several.",
                    diagnostics={"n_groups": len(levels)},
                )
            a = yv[keep & (gv == levels[0])].to_numpy()
            b = yv[keep & (gv == levels[1])].to_numpy()
            names = [str(levels[0]), str(levels[1])]
        else:
            a = yv.dropna().to_numpy()
            b = _column(data, str(other)).dropna().to_numpy()
            names = [y, str(other)]
        if a.size < 2:
            raise too_small(names[0])
        if b.size < 2:
            raise too_small(names[1])
        ra, rb = _row(a, alpha), _row(b, alpha)
        rc = _row(np.concatenate([a, b]), alpha)
        na, nb = a.size, b.size
        va, vb = ra["sd"] ** 2, rb["sd"] ** 2
        est = ra["mean"] - rb["mean"]
        if unequal:
            qa, qb = va / na, vb / nb
            se = float(np.sqrt(qa + qb))
            if welch:
                dof = -2.0 + (qa + qb) ** 2 / (qa**2 / (na + 1) + qb**2 / (nb + 1))
                kind = "unequal variances (Welch's degrees of freedom)"
            else:
                dof = (qa + qb) ** 2 / (qa**2 / (na - 1) + qb**2 / (nb - 1))
                kind = "unequal variances (Satterthwaite's degrees of freedom)"
        else:
            dof = float(na + nb - 2)
            pooled = ((na - 1) * va + (nb - 1) * vb) / dof
            se = float(np.sqrt(pooled * (1 / na + 1 / nb)))
            kind = "equal variances"
        method = f"Two-sample t test with {kind}"
        n_obs = int(na + nb)
        groups = pd.DataFrame.from_dict(
            {names[0]: ra, names[1]: rb, "combined": rc}, orient="index"
        )

    groups["n"] = groups["n"].astype(int)
    if not se > 0:
        raise MethodIncompatibility(
            "ttest: the standard error is zero (no variation in the sample), "
            "so the t statistic is not defined.",
            recovery_hint="Check that the variable is not constant.",
        )
    statistic = (est - mu) / se
    less = float(stats.t.cdf(statistic, dof))
    greater = float(stats.t.sf(statistic, dof))
    half = float(stats.t.ppf(1 - alpha / 2, dof)) * se
    return TTestResult(
        method=method,
        estimate=float(est),
        se=float(se),
        statistic=float(statistic),
        df=float(dof),
        pvalue=float(2 * min(less, greater)),
        pvalue_less=less,
        pvalue_greater=greater,
        ci=(float(est - half), float(est + half)),
        null=float(mu),
        alpha=float(alpha),
        n_obs=n_obs,
        groups=groups,
    )
