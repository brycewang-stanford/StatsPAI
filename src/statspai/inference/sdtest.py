"""Tests on the variance of a normal population, and z tests on its mean.

``sdtest`` is the chi-squared test that a standard deviation equals a given
value and the F test that two standard deviations are equal; ``ztest`` is the
test on a mean, or on a difference of two means, when the standard deviation
is known rather than estimated. Stata's ``sdtest`` / ``sdtesti`` and
``ztest`` / ``ztesti`` are the references, R's ``var.test`` for the ratio.

Both accept summary statistics in place of data (``n=``, ``mean=``, ``sd=``),
which is how a textbook states such a problem.

* The two-sided p-value of a variance test is twice the smaller tail, as in
  Stata and in R's ``var.test``: the chi-squared and F distributions are not
  symmetric, so there is no "as extreme on the other side" to add up.
* A ``by=`` ratio is the standard deviation of the group with the smaller
  value over that of the group with the larger one, as in Stata.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .ttest import TTestResult, _column, _row

__all__ = ["sdtest", "ztest", "SDTestResult"]


class SDTestResult(ResultProtocolMixin):
    """Outcome of :func:`sdtest`.

    Attributes
    ----------
    estimate : float
        The sample standard deviation (one sample) or the ratio of the two
        sample standard deviations.
    statistic : float
        ``(n - 1) s^2 / sd0^2`` (chi-squared) for one sample, ``s1^2 / s2^2``
        (F) for two.
    df : float or tuple of float
        Degrees of freedom: ``n - 1``, or ``(n1 - 1, n2 - 1)``.
    pvalue, pvalue_less, pvalue_greater : float
        Two-sided p-value (twice the smaller tail) and the one-sided ones
        (``Ha: sd < sd0`` / ``Ha: sd > sd0``, or the same for the ratio
        against one).
    ci : tuple of float
        ``1 - alpha`` confidence interval for the standard deviation (one
        sample) or for the ratio of standard deviations (two samples).
    groups : pandas.DataFrame
        One row per sample with ``n``, ``mean``, ``se``, ``sd``,
        ``ci_lower`` and ``ci_upper`` (the interval is for the mean).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.sdtest(n=10, sd=1.3 ** 0.5, sd0=2)
    >>> round(res.statistic, 3), res.df
    (2.925, 9.0)
    >>> bool(res.pvalue_less < 0.05)
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        method: str,
        estimate: float,
        statistic: float,
        df: Union[float, Tuple[float, float]],
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
        two = isinstance(df, tuple)
        self.estimand = "ratio of standard deviations" if two else "standard deviation"
        self.estimate = estimate
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
        if isinstance(self.df, tuple):
            stat = f"F = {self.statistic:.4f},  degrees of freedom = " + ", ".join(
                f"{d:g}" for d in self.df
            )
        else:
            stat = f"chi2 = {self.statistic:.4f},  degrees of freedom = {self.df:g}"
        lines = [
            self.method,
            "=" * len(self.method),
            table.to_string(float_format=lambda v: f"{v:.6g}"),
            "",
            f"{self.estimand} = {self.estimate:.6g}   "
            f"({level:g}% CI {self.ci[0]:.6g}, {self.ci[1]:.6g})",
            stat,
            f"H0: {self.estimand} = {self.null:g}",
            f"  Ha: <  p = {self.pvalue_less:.4f}",
            f"  Ha: != p = {self.pvalue:.4f}",
            f"  Ha: >  p = {self.pvalue_greater:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SDTestResult({self.method}: estimate={self.estimate:.6g}, "
            f"statistic={self.statistic:.4f}, p={self.pvalue:.4g})"
        )


# ------------------------------------------------------------------ samples
def _check_alpha(alpha: float, who: str) -> None:
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"{who}: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is 1 minus the confidence level, e.g. 0.05.",
        )


def _pair(value: Any) -> List[Any]:
    """A scalar or a two-item sequence as a list of one or two items."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _summary_row(n: float, mean: float, sd: float, alpha: float) -> Dict[str, float]:
    se = sd / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * se if n > 1 else np.nan
    return {
        "n": int(n),
        "mean": float(mean),
        "se": float(se),
        "sd": float(sd),
        "ci_lower": float(mean - half),
        "ci_upper": float(mean + half),
    }


def _samples(
    who: str,
    data: Optional[pd.DataFrame],
    y: Optional[str],
    by: Optional[str],
    other: Optional[str],
    n: Any,
    mean: Any,
    sd: Any,
    alpha: float,
    need_sd: bool,
) -> Tuple[Dict[str, Dict[str, float]], Optional[np.ndarray]]:
    """The one or two samples as rows of summary statistics.

    Returns the rows keyed by sample label, and for two samples of data the
    pooled observations (for the ``combined`` row), else ``None``.
    """
    if data is None:
        if n is None:
            raise MethodIncompatibility(
                f"{who}: give data and y, or the summary statistics n= "
                "(with mean= and/or sd=).",
                recovery_hint=f"sp.{who}(df, 'y', ...) or sp.{who}(n=10, ...).",
            )
        if y is not None or by is not None or other is not None:
            raise MethodIncompatibility(
                f"{who}: y / by / other name columns of data; with summary "
                "statistics there is no data.",
                recovery_hint="Drop them, or pass data.",
            )
        ns = _pair(n)
        means = _pair(mean) if mean is not None else [np.nan] * len(ns)
        sds = _pair(sd) if sd is not None else [np.nan] * len(ns)
        if len(ns) not in (1, 2) or len(means) != len(ns) or len(sds) != len(ns):
            raise MethodIncompatibility(
                f"{who}: n, mean and sd must each describe the same one or "
                "two samples.",
                recovery_hint="Pass scalars for one sample, pairs for two.",
            )
        rows = {}
        for i, (ni, mi, si) in enumerate(zip(ns, means, sds), start=1):
            if not float(ni) >= 2 or float(ni) != int(ni):
                raise MethodIncompatibility(
                    f"{who}: n must be an integer of at least 2, got {ni!r}.",
                    recovery_hint="n is the number of observations.",
                )
            if need_sd and not (si is not None and float(si) > 0):
                raise MethodIncompatibility(
                    f"{who}: sd must be a positive number, got {si!r}.",
                    recovery_hint="sd is the sample standard deviation.",
                )
            label = "x" if len(ns) == 1 else f"x{i}"
            rows[label] = _summary_row(float(ni), float(mi), float(si), alpha)
        return rows, None

    if y is None:
        raise MethodIncompatibility(
            f"{who}: y= names the variable to test.",
            recovery_hint="Pass the column name.",
        )
    if n is not None or mean is not None:
        raise MethodIncompatibility(
            f"{who}: n= and mean= are summary statistics, used in place of data.",
            recovery_hint="Drop them, or drop data.",
        )
    if by is not None and other is not None:
        raise MethodIncompatibility(
            f"{who}: pass by= (one variable, two groups) or other= (two "
            "variables), not both.",
            recovery_hint="Drop one of the two arguments.",
        )
    yv = _column(data, y)
    if by is not None:
        gv = data[by]
        keep = yv.notna() & gv.notna()
        levels = sorted(pd.unique(gv[keep]))
        if len(levels) != 2:
            raise MethodIncompatibility(
                f"{who}: by={by!r} must have exactly two groups; found "
                f"{len(levels)}.",
                recovery_hint="Restrict the data to two groups.",
                diagnostics={"n_groups": len(levels)},
            )
        arrays = {
            str(levels[0]): yv[keep & (gv == levels[0])].to_numpy(),
            str(levels[1]): yv[keep & (gv == levels[1])].to_numpy(),
        }
    elif other is not None:
        arrays = {
            y: yv.dropna().to_numpy(),
            other: _column(data, other).dropna().to_numpy(),
        }
    else:
        arrays = {y: yv.dropna().to_numpy()}
    for label, arr in arrays.items():
        if arr.size < 2:
            raise MethodIncompatibility(
                f"{who}: {label} has fewer than two non-missing observations.",
                recovery_hint="Check the sample and the grouping column.",
            )
    rows = {label: _row(arr, alpha) for label, arr in arrays.items()}
    pooled = np.concatenate(list(arrays.values())) if len(arrays) == 2 else None
    return rows, pooled


def _frame(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    groups = pd.DataFrame.from_dict(rows, orient="index")
    groups["n"] = groups["n"].astype(int)
    return groups


# ------------------------------------------------------------------- sdtest
def sdtest(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    by: Optional[str] = None,
    *,
    other: Optional[str] = None,
    sd0: Optional[float] = None,
    n: Union[None, int, Sequence[int]] = None,
    mean: Union[None, float, Sequence[float]] = None,
    sd: Union[None, float, Sequence[float]] = None,
    alpha: float = 0.05,
) -> SDTestResult:
    """Test that a standard deviation equals a value, or that two are equal.

    One sample: under normality ``(n - 1) s^2 / sd0^2`` is chi-squared with
    ``n - 1`` degrees of freedom. Two samples: ``s1^2 / s2^2`` is F with
    ``(n1 - 1, n2 - 1)``. Equivalent to Stata's ``sdtest`` / ``sdtesti``; the
    two-sample test is R's ``var.test``.

    ==========================================  ============================
    Call                                        Stata
    ==========================================  ============================
    ``sp.sdtest(df, "y", sd0=5)``               ``sdtest y == 5``
    ``sp.sdtest(df, "y", by="g")``              ``sdtest y, by(g)``
    ``sp.sdtest(df, "y", other="x")``           ``sdtest y == x``
    ``sp.sdtest(n=10, sd=1.14, sd0=2)``         ``sdtesti 10 . 1.14 2``
    ``sp.sdtest(n=(52, 22), sd=(4.7, 6.6))``    ``sdtesti 52 . 4.7 22 . 6.6``
    ==========================================  ============================

    Parameters
    ----------
    data : pandas.DataFrame, optional
    y : str, optional
        The variable whose standard deviation is tested.
    by : str, optional
        Grouping column with exactly two distinct non-missing values. The
        ratio is the standard deviation in the group with the smaller value
        over that in the group with the larger one.
    other : str, optional
        A second variable; each column uses its own non-missing rows.
    sd0 : float, optional
        One sample: the standard deviation under the null. Required there,
        not used with two samples (whose null is a ratio of one).
    n, mean, sd : number or pair of numbers, optional
        Summary statistics in place of ``data``: the number of observations
        and the sample standard deviation (divisor ``n - 1``) of one sample,
        or pairs for two. ``mean`` is only reported.
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level of the intervals.

    Returns
    -------
    SDTestResult

    Raises
    ------
    MethodIncompatibility
        When neither data nor summary statistics are given, when ``by``
        does not have exactly two groups, when a one-sample call has no
        ``sd0``, or when a sample has no variation.

    Notes
    -----
    Both tests are exact under normality and are not robust to departures
    from it: with heavy tails they reject a true null far more often than
    ``alpha``.

    Examples
    --------
    A student's scores had standard deviation 2. After tutoring, ten tests
    have sample variance 1.3. Did the scores become more stable?

    >>> import statspai as sp
    >>> res = sp.sdtest(n=10, sd=1.3 ** 0.5, sd0=2)
    >>> round(res.statistic, 3)
    2.925
    >>> round(res.pvalue_less, 4)   # Ha: sd < 2
    0.0329

    Two groups of a data set:

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"g": np.repeat([0, 1], 40)})
    >>> df["y"] = rng.normal(size=80) * np.where(df.g == 1, 2.0, 1.0)
    >>> two = sp.sdtest(df, "y", by="g")
    >>> two.method
    'Variance ratio test'
    >>> bool(two.pvalue < 0.05)
    True
    """
    _check_alpha(alpha, "sdtest")
    rows, pooled = _samples(
        "sdtest", data, y, by, other, n, mean, sd, alpha, need_sd=True
    )
    labels = list(rows)
    for label in labels:
        if not rows[label]["sd"] > 0:
            raise MethodIncompatibility(
                f"sdtest: {label} has no variation, so the test is not defined.",
                recovery_hint="Check that the variable is not constant.",
            )

    if len(labels) == 1:
        if sd0 is None or not float(sd0) > 0:
            raise MethodIncompatibility(
                "sdtest: a one-sample test needs sd0=, the positive standard "
                "deviation under the null.",
                recovery_hint="sp.sdtest(df, 'y', sd0=5); for two samples "
                "pass by= or other=.",
            )
        row = rows[labels[0]]
        dof = float(row["n"] - 1)
        s = row["sd"]
        statistic = dof * s**2 / float(sd0) ** 2
        less = float(stats.chi2.cdf(statistic, dof))
        greater = float(stats.chi2.sf(statistic, dof))
        ci = (
            float(s * np.sqrt(dof / stats.chi2.ppf(1 - alpha / 2, dof))),
            float(s * np.sqrt(dof / stats.chi2.ppf(alpha / 2, dof))),
        )
        return SDTestResult(
            method="One-sample test of variance",
            estimate=float(s),
            statistic=float(statistic),
            df=dof,
            pvalue=float(min(1.0, 2 * min(less, greater))),
            pvalue_less=less,
            pvalue_greater=greater,
            ci=ci,
            null=float(sd0),
            alpha=float(alpha),
            n_obs=int(row["n"]),
            groups=_frame(rows),
        )

    if sd0 is not None:
        raise MethodIncompatibility(
            "sdtest: sd0= is the null of a one-sample test; two samples are "
            "tested against a ratio of one.",
            recovery_hint="Drop sd0=, or drop by= / other=.",
        )
    ra, rb = rows[labels[0]], rows[labels[1]]
    d1, d2 = float(ra["n"] - 1), float(rb["n"] - 1)
    statistic = ra["sd"] ** 2 / rb["sd"] ** 2
    less = float(stats.f.cdf(statistic, d1, d2))
    greater = float(stats.f.sf(statistic, d1, d2))
    ratio = ra["sd"] / rb["sd"]
    ci = (
        float(ratio / np.sqrt(stats.f.ppf(1 - alpha / 2, d1, d2))),
        float(ratio / np.sqrt(stats.f.ppf(alpha / 2, d1, d2))),
    )
    if pooled is not None:
        rows["combined"] = _row(pooled, alpha)
    return SDTestResult(
        method="Variance ratio test",
        estimate=float(ratio),
        statistic=float(statistic),
        df=(d1, d2),
        pvalue=float(min(1.0, 2 * min(less, greater))),
        pvalue_less=less,
        pvalue_greater=greater,
        ci=ci,
        null=1.0,
        alpha=float(alpha),
        n_obs=int(ra["n"] + rb["n"]),
        groups=_frame(rows),
    )


# -------------------------------------------------------------------- ztest
def ztest(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    by: Optional[str] = None,
    *,
    other: Optional[str] = None,
    mu: float = 0.0,
    sd: Union[float, Sequence[float]] = 1.0,
    n: Union[None, int, Sequence[int]] = None,
    mean: Union[None, float, Sequence[float]] = None,
    alpha: float = 0.05,
) -> TTestResult:
    """z test on a mean, or on a difference of two means, with known ``sd``.

    The population standard deviation is taken as given, so the statistic is
    standard normal whatever the sample size. Equivalent to Stata's
    ``ztest`` / ``ztesti``. When the standard deviation is estimated from
    the same data, use :func:`ttest`.

    ============================================  =========================
    Call                                          Stata
    ============================================  =========================
    ``sp.ztest(df, "y", mu=20, sd=6)``            ``ztest y == 20, sd(6)``
    ``sp.ztest(df, "y", by="g", sd=6)``           ``ztest y, by(g) sd(6)``
    ``sp.ztest(df, "y", by="g", sd=(5, 7))``      ``ztest y, by(g) sd1(5) sd2(7)``
    ``sp.ztest(n=10, mean=88, sd=0.71, mu=85)``   ``ztesti 10 88 0.71 85``
    ============================================  =========================

    Parameters
    ----------
    data : pandas.DataFrame, optional
    y : str, optional
        The variable whose mean is tested.
    by : str, optional
        Grouping column with exactly two distinct non-missing values; the
        estimate is the mean in the group with the smaller value minus the
        mean in the group with the larger one.
    other : str, optional
        A second variable, treated as an independent sample (each column
        uses its own non-missing rows). A paired z test needs the known
        correlation of the two and is not offered.
    mu : float, default 0.0
        The null value of the mean, or of the difference.
    sd : float or pair of floats, default 1.0
        The known population standard deviation; a pair gives one per
        sample.
    n, mean : number or pair of numbers, optional
        Summary statistics in place of ``data``: observations and sample
        mean of one sample, or pairs for two.
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level of the intervals.

    Returns
    -------
    TTestResult
        With ``df`` infinite and ``statistic`` the z statistic.

    Raises
    ------
    MethodIncompatibility
        When neither data nor summary statistics are given, when ``by``
        does not have exactly two groups, or when ``sd`` is not positive.

    Examples
    --------
    Scores were N(85, 0.5). After tutoring, ten tests average 88. Did the
    mean rise?

    >>> import statspai as sp
    >>> res = sp.ztest(n=10, mean=88, sd=0.5 ** 0.5, mu=85)
    >>> round(res.statistic, 2)
    13.42
    >>> bool(res.pvalue_greater < 0.05)   # Ha: mean > 85
    True
    """
    _check_alpha(alpha, "ztest")
    rows, _ = _samples("ztest", data, y, by, other, n, mean, None, alpha, need_sd=False)
    labels = list(rows)
    sds = [float(s) for s in _pair(sd)]
    if len(sds) == 1:
        sds = sds * len(labels)
    if len(sds) != len(labels) or not all(s > 0 for s in sds):
        raise MethodIncompatibility(
            "ztest: sd must be a positive number, or one per sample.",
            recovery_hint="sd is the known population standard deviation.",
        )
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    for label, s in zip(labels, sds):
        row = rows[label]
        if not np.isfinite(row["mean"]):
            raise MethodIncompatibility(
                "ztest: mean= is required with summary statistics.",
                recovery_hint="sp.ztest(n=10, mean=88, sd=0.7, mu=85).",
            )
        se = s / np.sqrt(row["n"])
        row.update(
            se=float(se),
            sd=s,
            ci_lower=float(row["mean"] - zcrit * se),
            ci_upper=float(row["mean"] + zcrit * se),
        )
    if len(labels) == 1:
        est, se = rows[labels[0]]["mean"], rows[labels[0]]["se"]
        method = "One-sample z test"
        n_obs = int(rows[labels[0]]["n"])
    else:
        ra, rb = rows[labels[0]], rows[labels[1]]
        est = ra["mean"] - rb["mean"]
        se = float(np.sqrt(ra["se"] ** 2 + rb["se"] ** 2))
        method = "Two-sample z test"
        n_obs = int(ra["n"] + rb["n"])
    statistic = (est - mu) / se
    less = float(stats.norm.cdf(statistic))
    greater = float(stats.norm.sf(statistic))
    result = TTestResult(
        method=method,
        estimate=float(est),
        se=float(se),
        statistic=float(statistic),
        df=float("inf"),
        pvalue=float(2 * min(less, greater)),
        pvalue_less=less,
        pvalue_greater=greater,
        ci=(float(est - zcrit * se), float(est + zcrit * se)),
        null=float(mu),
        alpha=float(alpha),
        n_obs=n_obs,
        groups=_frame(rows),
    )
    result.statistic_name = "z"
    return result
