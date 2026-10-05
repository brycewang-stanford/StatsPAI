"""Tests and intervals of a first statistics chapter: proportions, normality,
and confidence intervals for a mean, a variance or a proportion.

* :func:`prtest` -- z test that a proportion equals a value, or that two are
  equal (Stata ``prtest`` / ``prtesti``).
* :func:`sktest` -- skewness and kurtosis tests for normality with the joint
  test (Stata ``sktest``).
* :func:`swilk` -- the Shapiro-Wilk W test (Stata ``swilk``, R
  ``shapiro.test``).
* :func:`ci` -- confidence intervals for means, variances, standard
  deviations and proportions (Stata ``ci means`` / ``ci variances`` /
  ``ci proportions``).

The conventions that differ between packages are stated where they apply:
the two-sample proportion test uses the pooled standard error under the
null and the unpooled one for the interval, and the default interval for a
proportion is the exact (Clopper-Pearson) one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from .ttest import TTestResult

__all__ = ["prtest", "sktest", "swilk", "ci"]

_Vars = Union[None, str, Sequence[str]]


def _check_alpha(alpha: float, who: str) -> None:
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"{who}: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is 1 minus the confidence level, e.g. 0.05.",
        )


def _col(data: pd.DataFrame, name: Any, who: str) -> pd.Series:
    """Column ``name`` as floats, with errors that name the caller."""
    if name not in data.columns:
        raise MethodIncompatibility(
            f"{who}: column {name!r} is not in data.",
            recovery_hint="Check the column name.",
            diagnostics={"columns": [str(c) for c in data.columns][:20]},
        )
    col = data[name]
    if not pd.api.types.is_numeric_dtype(col):
        raise MethodIncompatibility(
            f"{who}: column {name!r} is not numeric.",
            recovery_hint="Encode it as a number first.",
        )
    return col.astype(float)


def _as_frame(data: Any) -> Any:
    """A Series or a one-dimensional array is one variable: the residuals
    of a fit, say. Anything else is returned as it came."""
    if isinstance(data, pd.Series):
        return data.to_frame(name="x" if data.name is None else data.name)
    if isinstance(data, (np.ndarray, list, tuple)):
        values = np.asarray(data)
        if values.ndim == 1:
            return pd.DataFrame({"x": values})
    return data


def _numeric_columns(data: pd.DataFrame, variables: _Vars, who: str) -> List[Any]:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{who}: data must be a DataFrame.",
            recovery_hint=f"sp.{who}(df, ['x', 'y'])",
        )
    if variables is None:
        names = list(data.select_dtypes("number").columns)
    else:
        names = [variables] if isinstance(variables, str) else list(variables)
    if not names:
        raise MethodIncompatibility(
            f"{who}: no numeric variable to work on.",
            recovery_hint="Name the columns.",
        )
    for name in names:
        _col(data, name, who)  # raises on a missing or non-numeric column
    return names


# ------------------------------------------------------------------ prtest
def _binary(data: pd.DataFrame, name: str) -> pd.Series:
    col = _col(data, name, "prtest").dropna()
    if not col.isin([0.0, 1.0]).all():
        raise MethodIncompatibility(
            f"prtest: {name!r} must be coded 0 / 1.",
            recovery_hint="Recode the variable, e.g. (df[col] == value).astype(int).",
        )
    return col


def _prop_row(n: float, p: float, z: float) -> Dict[str, float]:
    se = float(np.sqrt(p * (1 - p) / n))
    return {
        "n": int(n),
        "mean": float(p),
        "se": se,
        "sd": float(np.sqrt(p * (1 - p))),
        "ci_lower": float(p - z * se),
        "ci_upper": float(p + z * se),
    }


def prtest(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    by: Optional[str] = None,
    *,
    other: Optional[str] = None,
    p: float = 0.5,
    n: Union[None, int, Sequence[int]] = None,
    proportion: Union[None, float, Sequence[float]] = None,
    alpha: float = 0.05,
) -> TTestResult:
    """Large-sample z test on a proportion, or on the difference of two.

    Equivalent to Stata's ``prtest`` / ``prtesti`` and (without continuity
    correction) R's ``prop.test``. For a small sample use the exact
    :func:`bitest`.

    ============================================  ===========================
    Call                                          Stata
    ============================================  ===========================
    ``sp.prtest(df, "d", p=0.4)``                 ``prtest d == 0.4``
    ``sp.prtest(df, "d", by="g")``                ``prtest d, by(g)``
    ``sp.prtest(df, "d", other="e")``             ``prtest d == e``
    ``sp.prtest(n=50, proportion=0.52, p=0.4)``   ``prtesti 50 0.52 0.4``
    ``sp.prtest(n=(30, 45),                       ``prtesti 30 0.4 45 0.67``
    proportion=(0.4, 0.67))``
    ============================================  ===========================

    Parameters
    ----------
    data : pandas.DataFrame, optional
    y : str, optional
        A 0 / 1 variable.
    by : str, optional
        Grouping column with exactly two distinct non-missing values. The
        estimate is the proportion in the group with the smaller value
        minus that in the group with the larger one.
    other : str, optional
        A second 0 / 1 variable, treated as an independent sample.
    p : float, default 0.5
        One sample: the proportion under the null. Not used with two
        samples, whose null is equal proportions.
    n, proportion : number or pair of numbers, optional
        Summary statistics in place of ``data``: observations and sample
        proportion of one sample, or pairs for two.
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level of the intervals.

    Returns
    -------
    TTestResult
        ``statistic`` is z (``df`` is infinite). With two samples the
        statistic uses the standard error pooled under the null
        (``result.se_null``); ``se`` and ``ci`` use the unpooled one.

    Raises
    ------
    MethodIncompatibility
        A variable that is not 0 / 1, ``by`` without exactly two groups,
        or a null or sample proportion outside (0, 1).

    Notes
    -----
    The normal approximation is poor when ``n p`` or ``n (1 - p)`` is
    small; the intervals here are Wald intervals and can leave [0, 1].
    :func:`ci` with ``stat='proportions'`` gives exact and Wilson
    intervals.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.prtest(n=50, proportion=0.52, p=0.4)
    >>> round(res.statistic, 4)
    1.7321
    >>> round(res.pvalue, 4)
    0.0833
    """
    _check_alpha(alpha, "prtest")
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    if data is None:
        if y is not None or by is not None or other is not None:
            raise MethodIncompatibility(
                "prtest: y / by / other name columns of data; pass data, or "
                "the summary statistics n= and proportion=.",
                recovery_hint="sp.prtest(n=50, proportion=0.52, p=0.4)",
            )
        if n is None or proportion is None:
            raise MethodIncompatibility(
                "prtest: give data and y, or both n= and proportion=.",
                recovery_hint="sp.prtest(n=50, proportion=0.52, p=0.4)",
            )
        ns: List[Any] = list(n) if isinstance(n, (list, tuple, np.ndarray)) else [n]
        ps: List[Any] = (
            list(proportion)
            if isinstance(proportion, (list, tuple, np.ndarray))
            else [proportion]
        )
        if len(ns) != len(ps) or len(ns) not in (1, 2):
            raise MethodIncompatibility(
                "prtest: n and proportion must describe the same one or two "
                "samples.",
                recovery_hint="Pass scalars for one sample, pairs for two.",
            )
        samples = {
            ("x" if len(ns) == 1 else f"x{i}"): (float(ni), float(pi))
            for i, (ni, pi) in enumerate(zip(ns, ps), start=1)
        }
    else:
        if n is not None or proportion is not None:
            raise MethodIncompatibility(
                "prtest: n= and proportion= are summary statistics, used in "
                "place of data.",
                recovery_hint="Drop them, or drop data.",
            )
        if y is None:
            raise MethodIncompatibility(
                "prtest: y= names the 0 / 1 variable to test.",
                recovery_hint="Pass the column name.",
            )
        if by is not None and other is not None:
            raise MethodIncompatibility(
                "prtest: pass by= or other=, not both.",
                recovery_hint="Drop one of the two arguments.",
            )
        if by is not None:
            keep = data[y].notna() & data[by].notna()
            levels = sorted(pd.unique(data.loc[keep, by]))
            if len(levels) != 2:
                raise MethodIncompatibility(
                    f"prtest: by={by!r} must have exactly two groups; found "
                    f"{len(levels)}.",
                    recovery_hint="Restrict the data to two groups.",
                    diagnostics={"n_groups": len(levels)},
                )
            cols = {
                str(lv): _binary(data.loc[keep & (data[by] == lv)], y) for lv in levels
            }
        elif other is not None:
            cols = {y: _binary(data, y), other: _binary(data, other)}
        else:
            cols = {y: _binary(data, y)}
        samples = {k: (float(v.size), float(v.mean())) for k, v in cols.items()}
    for label, (ni, pi) in samples.items():
        if ni < 1 or ni != int(ni) or not 0 <= pi <= 1:
            raise MethodIncompatibility(
                f"prtest: sample {label!r} needs a positive integer n and a "
                f"proportion in [0, 1]; got n={ni!r}, proportion={pi!r}.",
                recovery_hint="Check the summary statistics.",
            )
    rows = {k: _prop_row(ni, pi, zcrit) for k, (ni, pi) in samples.items()}
    labels = list(rows)
    if len(labels) == 1:
        if not 0 < p < 1:
            raise MethodIncompatibility(
                f"prtest: the null proportion p must be in (0, 1), got {p!r}.",
                recovery_hint="p is the proportion under the null hypothesis.",
            )
        row = rows[labels[0]]
        est, se, null = row["mean"], row["se"], float(p)
        se_null = float(np.sqrt(p * (1 - p) / row["n"]))
        method = "One-sample test of proportion"
        n_obs = int(row["n"])
    else:
        ra, rb = rows[labels[0]], rows[labels[1]]
        est = ra["mean"] - rb["mean"]
        se = float(np.sqrt(ra["se"] ** 2 + rb["se"] ** 2))
        pooled = (ra["n"] * ra["mean"] + rb["n"] * rb["mean"]) / (ra["n"] + rb["n"])
        se_null = float(np.sqrt(pooled * (1 - pooled) * (1 / ra["n"] + 1 / rb["n"])))
        null = 0.0
        method = "Two-sample test of proportions"
        n_obs = int(ra["n"] + rb["n"])
    if not se_null > 0:
        raise MethodIncompatibility(
            "prtest: the standard error under the null is zero (every "
            "observation is the same), so the statistic is not defined.",
            recovery_hint="Check the variable.",
        )
    statistic = (est - null) / se_null
    less = float(stats.norm.cdf(statistic))
    greater = float(stats.norm.sf(statistic))
    groups = pd.DataFrame.from_dict(rows, orient="index")
    groups["n"] = groups["n"].astype(int)
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
        null=null,
        alpha=float(alpha),
        n_obs=n_obs,
        groups=groups,
    )
    result.statistic_name = "z"
    result.estimand = "proportion" if len(labels) == 1 else "difference in proportions"
    result.se_null = se_null
    return result


# ------------------------------------------------------------------ sktest
def _sk_statistics(x: np.ndarray, adjust: bool) -> Dict[str, float]:
    """D'Agostino's skewness test, the Anscombe-Glynn kurtosis test and
    their joint chi-squared, with Royston's small-sample adjustment."""
    n = float(x.size)
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    skew = float(np.mean(dev**3) / m2**1.5)
    kurt = float(np.mean(dev**4) / m2**2)
    # skewness
    y = skew * np.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
    beta2 = (
        3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)
        / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    )  # fmt: skip
    w2 = -1 + np.sqrt(2 * (beta2 - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
    a = np.sqrt(2 / (w2 - 1))
    z1 = delta * np.log(y / a + np.sqrt((y / a) ** 2 + 1))
    # kurtosis
    e_b2 = 3 * (n - 1) / (n + 1)
    v_b2 = 24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    xk = (kurt - e_b2) / np.sqrt(v_b2)
    rb1 = (6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9))) * np.sqrt(
        6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))
    )
    big_a = 6 + (8 / rb1) * (2 / rb1 + np.sqrt(1 + 4 / rb1**2))
    z2 = (
        (1 - 2 / (9 * big_a))
        - np.cbrt((1 - 2 / big_a) / (1 + xk * np.sqrt(2 / (big_a - 4))))
    ) / np.sqrt(2 / (9 * big_a))
    k2 = float(z1**2 + z2**2)
    if adjust:
        # Royston's adjustment: map the chi-squared to a normal deviate, correct
        # it, and map back, so that the test has its nominal size in small
        # samples
        zc = -stats.norm.ppf(np.exp(-0.5 * k2))
        logn = np.log(n)
        cut = 0.55 * n**0.2 - 0.21
        a1 = (-5 + 3.46 * logn) * np.exp(-1.37 * logn)
        b1 = 1 + (0.854 - 0.148 * logn) * np.exp(-0.55 * logn)
        b2mb1 = 2.13 / (1 - 2.37 * logn)
        a2 = a1 - b2mb1 * cut
        b2 = b2mb1 + b1
        if zc < -1:
            z = zc
        elif zc < cut:
            z = a1 + b1 * zc
        else:
            z = a2 + b2 * zc
        p_joint = float(stats.norm.sf(z))
        k2 = float(-2 * np.log(p_joint))
    else:
        p_joint = float(stats.chi2.sf(k2, 2))
    return {
        "n": int(n),
        "skewness": skew,
        "kurtosis": kurt,
        "p_skew": float(2 * stats.norm.sf(abs(z1))),
        "p_kurt": float(2 * stats.norm.sf(abs(z2))),
        "chi2": k2,
        "p_chi2": p_joint,
    }


def sktest(
    data: pd.DataFrame, variables: _Vars = None, *, adjust: bool = True
) -> pd.DataFrame:
    """Skewness and kurtosis tests for normality, and their joint test.

    One row per variable: the test that the skewness is that of a normal
    (zero), the test that the kurtosis is (three), and the joint
    chi-squared with two degrees of freedom. Equivalent to Stata's
    ``sktest``.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : str or list of str, optional
        Columns to test; every numeric column when omitted. Missing values
        are dropped variable by variable.
    adjust : bool, default True
        Apply Royston's adjustment to the joint statistic (the one
        Stata's ``sktest`` documents), which keeps its size in small
        samples. ``False`` is the D'Agostino,
        Belanger and D'Agostino (1990) statistic as it stands (Stata's
        ``noadjust``).

    Returns
    -------
    pandas.DataFrame
        Indexed by variable, with ``n``, ``skewness``, ``kurtosis`` (the
        moment coefficients, 3 for a normal), ``p_skew``, ``p_kurt``,
        ``chi2`` and ``p_chi2``.

    Raises
    ------
    DataInsufficient
        Fewer than 8 observations of a variable (the approximations are
        not defined below that), or no variation.

    Notes
    -----
    Rejecting says the data are unlikely under normality. With a large
    sample the test rejects for departures too small to matter, and with a
    small one it has little power; a plot of the distribution is the
    better guide to whether a departure matters.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"z": rng.normal(size=300),
    ...                    "e": rng.exponential(size=300)})
    >>> out = sp.sktest(df)
    >>> list(out.columns)
    ['n', 'skewness', 'kurtosis', 'p_skew', 'p_kurt', 'chi2', 'p_chi2']
    >>> bool(out.loc["e", "p_chi2"] < 0.01 < out.loc["z", "p_chi2"])
    True

    References
    ----------
    [@dagostino1990suggestion]
    """
    data = _as_frame(data)
    rows = {}
    for name in _numeric_columns(data, variables, "sktest"):
        x = _col(data, name, "sktest").dropna().to_numpy()
        if x.size < 8 or np.ptp(x) == 0:
            raise DataInsufficient(
                f"sktest: {name!r} needs at least 8 observations that are "
                "not all equal.",
                recovery_hint="Check the variable.",
                diagnostics={"n_obs": int(x.size)},
            )
        rows[name] = _sk_statistics(x, adjust)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["n"] = out["n"].astype(int)
    return out


# ------------------------------------------------------------------- swilk
def swilk(data: pd.DataFrame, variables: _Vars = None) -> pd.DataFrame:
    """Shapiro-Wilk W test for normality.

    ``data`` may also be a Series or a one-dimensional array, which is
    tested as a single variable.

    Equivalent to Stata's ``swilk`` and R's ``shapiro.test``: ``W`` is the
    squared correlation of the ordered sample with the expected normal
    order statistics, and the p-value comes from Royston's (1992) normal
    approximation.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : str or list of str, optional
        Columns to test; every numeric column when omitted. Missing values
        are dropped variable by variable.

    Returns
    -------
    pandas.DataFrame
        Indexed by variable, with ``n``, ``W``, ``V`` (Royston's index of
        departure from normality, 1 at the median of its null
        distribution; reported for ``n >= 12``), ``z`` and ``pvalue``.

    Raises
    ------
    DataInsufficient
        Fewer than 4 or more than 5,000 observations (the range the
        approximation covers), or no variation.

    Notes
    -----
    The same caution as for :func:`sktest` applies: in large samples a
    rejection can reflect a departure of no practical importance.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"e": rng.exponential(size=200)})
    >>> out = sp.swilk(df, "e")
    >>> bool(out.loc["e", "pvalue"] < 0.001)
    True

    References
    ----------
    [@royston1992approximating]
    """
    data = _as_frame(data)
    rows = {}
    for name in _numeric_columns(data, variables, "swilk"):
        x = _col(data, name, "swilk").dropna().to_numpy()
        n = x.size
        if n < 4 or n > 5000 or np.ptp(x) == 0:
            raise DataInsufficient(
                f"swilk: {name!r} has {n} usable observations; the test "
                "covers 4 to 5,000 observations that are not all equal.",
                recovery_hint="For a larger sample use sp.sktest.",
                diagnostics={"n_obs": int(n)},
            )
        w = float(stats.shapiro(x).statistic)
        if n >= 12:
            logn = np.log(n)
            mu = 0.0038915 * logn**3 - 0.083751 * logn**2 - 0.31082 * logn - 1.5861
            sigma = np.exp(0.0030302 * logn**2 - 0.082676 * logn - 0.4803)
            z = (np.log(1 - w) - mu) / sigma
            v = (1 - w) / np.exp(mu)
        else:
            gamma = -2.273 + 0.459 * n
            mu = 0.5440 - 0.39978 * n + 0.025054 * n**2 - 0.0006714 * n**3
            sigma = np.exp(1.3822 - 0.77857 * n + 0.062767 * n**2 - 0.0020322 * n**3)
            z = (-np.log(gamma - np.log(1 - w)) - mu) / sigma
            v = np.nan
        rows[name] = {
            "n": int(n),
            "W": w,
            "V": float(v),
            "z": float(z),
            "pvalue": float(stats.norm.sf(z)),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["n"] = out["n"].astype(int)
    return out


# ---------------------------------------------------------------------- ci
_CI_STATS = {
    "mean": "means", "means": "means",
    "variance": "variances", "variances": "variances",
    "sd": "sd",
    "proportion": "proportions", "proportions": "proportions",
}  # fmt: skip
_PROPORTION_METHODS = ("exact", "wald", "wilson", "agresti", "jeffreys")


def _proportion_interval(k: float, n: float, alpha: float, method: str) -> Any:
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    if method == "exact":
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    elif method == "wald":
        half = z * np.sqrt(p * (1 - p) / n)
        lo, hi = p - half, p + half
    elif method == "wilson":
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lo, hi = centre - half, centre + half
    elif method == "agresti":
        nt = n + z * z
        pt = (k + z * z / 2) / nt
        half = z * np.sqrt(pt * (1 - pt) / nt)
        lo, hi = pt - half, pt + half
    else:  # jeffreys
        lo = stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5) if k > 0 else 0.0
        hi = stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5) if k < n else 1.0
    return float(lo), float(hi)


def ci(
    data: pd.DataFrame,
    variables: _Vars = None,
    *,
    stat: str = "means",
    method: str = "exact",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Confidence intervals for means, variances or proportions.

    Equivalent to Stata's ``ci means``, ``ci variances`` (``sd`` for the
    standard deviation) and ``ci proportions``.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : str or list of str, optional
        Columns; every numeric column when omitted. Missing values are
        dropped variable by variable.
    stat : {'means', 'variances', 'sd', 'proportions'}, default 'means'
        ``'means'``: ``mean +/- t(n - 1) se``. ``'variances'`` / ``'sd'``:
        the chi-squared interval ``(n - 1) s^2 / chi2``, exact for a normal
        population and sensitive to departures from it. ``'proportions'``:
        for 0 / 1 variables, by ``method``.
    method : {'exact', 'wald', 'wilson', 'agresti', 'jeffreys'}
        ``stat='proportions'`` only. ``'exact'`` (the default, as in Stata)
        is the Clopper-Pearson interval, which never covers less than
        ``1 - alpha`` and is conservative; ``'wilson'`` is the score
        interval, usually the better choice for reporting.
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level.

    Returns
    -------
    pandas.DataFrame
        Indexed by variable, with ``n``, the estimate (``mean``,
        ``variance``, ``sd`` or ``proportion``), ``se`` (means and
        proportions), ``ci_lower`` and ``ci_upper``.

    Raises
    ------
    MethodIncompatibility
        An unknown ``stat`` or ``method``, or a proportion asked of a
        variable that is not 0 / 1.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.normal(2, 1, 100),
    ...                    "d": rng.integers(0, 2, 100)})
    >>> means = sp.ci(df, "y")
    >>> bool(means.loc["y", "ci_lower"] < 2 < means.loc["y", "ci_upper"])
    True
    >>> list(sp.ci(df, "d", stat="proportions", method="wilson").columns)
    ['n', 'proportion', 'se', 'ci_lower', 'ci_upper']
    """
    _check_alpha(alpha, "ci")
    kind = _CI_STATS.get(str(stat).lower())
    if kind is None:
        raise MethodIncompatibility(
            f"ci: stat={stat!r} is not one of 'means', 'variances', 'sd', "
            "'proportions'.",
            recovery_hint="sp.ci(df, 'y', stat='means')",
        )
    method = str(method).lower()
    if method not in _PROPORTION_METHODS:
        raise MethodIncompatibility(
            f"ci: method={method!r} is not one of {list(_PROPORTION_METHODS)}.",
            recovery_hint="method applies to stat='proportions'.",
        )
    rows: Dict[str, Dict[str, float]] = {}
    for name in _numeric_columns(data, variables, "ci"):
        x = _col(data, name, "ci").dropna().to_numpy()
        n = x.size
        if n < (1 if kind == "proportions" else 2):
            raise DataInsufficient(
                f"ci: {name!r} has too few observations.",
                recovery_hint="Check the variable.",
                diagnostics={"n_obs": int(n)},
            )
        if kind == "means":
            se = float(x.std(ddof=1) / np.sqrt(n))
            half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
            mean = float(x.mean())
            rows[name] = {"n": n, "mean": mean, "se": se,
                          "ci_lower": mean - half, "ci_upper": mean + half}  # fmt: skip
        elif kind in ("variances", "sd"):
            var = float(x.var(ddof=1))
            lo = (n - 1) * var / stats.chi2.ppf(1 - alpha / 2, n - 1)
            hi = (n - 1) * var / stats.chi2.ppf(alpha / 2, n - 1)
            if kind == "sd":
                rows[name] = {"n": n, "sd": float(np.sqrt(var)),
                              "ci_lower": float(np.sqrt(lo)),
                              "ci_upper": float(np.sqrt(hi))}  # fmt: skip
            else:
                rows[name] = {"n": n, "variance": var,
                              "ci_lower": float(lo), "ci_upper": float(hi)}  # fmt: skip
        else:
            if not np.isin(x, (0.0, 1.0)).all():
                raise MethodIncompatibility(
                    f"ci: {name!r} must be coded 0 / 1 for a proportion.",
                    recovery_hint="Recode the variable first.",
                )
            k = float(x.sum())
            lo, hi = _proportion_interval(k, float(n), alpha, method)
            prop = k / n
            rows[name] = {"n": n, "proportion": prop,
                          "se": float(np.sqrt(prop * (1 - prop) / n)),
                          "ci_lower": lo, "ci_upper": hi}  # fmt: skip
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["n"] = out["n"].astype(int)
    out.attrs["stat"] = kind
    out.attrs["alpha"] = float(alpha)
    if kind == "proportions":
        out.attrs["method"] = method
    return out
