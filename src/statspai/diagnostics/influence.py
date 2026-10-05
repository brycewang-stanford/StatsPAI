"""Influence statistics after a linear or a logistic regression, and the
goodness-of-fit tests of a logistic regression.

* :func:`influence_measures` -- leverage, standardized and studentized
  residuals, Cook's distance, DFFITS, Welsch distance, COVRATIO, the
  standard errors of the prediction and DFBETAs after ``sp.regress``
  (Stata ``predict, leverage | rstandard | rstudent | cooksd | dfits |
  welsch | covratio | stdp | stdf | stdr`` and ``dfbeta``; R
  ``influence.measures``).
* :func:`logit_influence` -- Pregibon's diagnostics by covariate pattern
  after ``sp.logit`` (Stata ``predict, residuals | rstandard | deviance |
  hat | dx2 | ddeviance | dbeta`` after ``logit`` / ``logistic``).
* :func:`logit_gof` -- Pearson's goodness-of-fit test over covariate
  patterns and the Hosmer-Lemeshow test (Stata ``estat gof``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["influence_measures", "logit_influence", "logit_gof", "roc_area"]


def _linear_fit(result: Any, who: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    info = getattr(result, "data_info", None) or {}
    model = getattr(result, "model_info", None) or {}
    if info.get("X") is None or info.get("residuals") is None:
        raise MethodIncompatibility(
            f"sp.{who}: the result does not store its design matrix and "
            "residuals; it needs a linear fit from sp.regress.",
            recovery_hint="Fit the model with sp.regress and pass that result.",
        )
    if model.get("family") not in (None, "gaussian"):
        raise MethodIncompatibility(
            f"sp.{who}: these statistics are those of a linear regression.",
            recovery_hint="After sp.logit use sp.logit_influence.",
        )
    if info.get("weights") is not None or model.get("weights") is not None:
        raise MethodIncompatibility(
            f"sp.{who}: a weighted fit is not supported.",
            recovery_hint="Refit without weights, or compute the statistics "
            "from the weighted hat matrix directly.",
        )
    x = np.asarray(info["X"], dtype=float)
    e = np.asarray(info["residuals"], dtype=float)
    names = [str(n) for n in (info.get("var_names") or result.params.index)]
    return x, e, names


def influence_measures(result: Any) -> pd.DataFrame:
    """Influence statistics of every observation of a linear regression.

    With ``h`` the diagonal of the hat matrix, ``e`` the residual, ``s``
    the root mean squared error on ``n - k`` degrees of freedom and
    ``s(i)`` the one of the fit without observation i:

    ==============  ====================================================
    ``leverage``    ``h``
    ``rstandard``   ``e / (s sqrt(1 - h))``
    ``rstudent``    ``e / (s(i) sqrt(1 - h))``
    ``cooksd``      ``rstandard^2 h / (k (1 - h))``
    ``dfits``       ``rstudent sqrt(h / (1 - h))``
    ``welsch``      ``dfits sqrt((n - 1) / (1 - h))``
    ``covratio``    ``1 / ((1 - h) ((n - k - 1 + rstudent^2) / (n - k))^k)``
    ``stdp``        ``s sqrt(h)``, the standard error of the prediction
    ``stdf``        ``s sqrt(1 + h)``, of the forecast
    ``stdr``        ``s sqrt(1 - h)``, of the residual
    ``dfbeta_<x>``  the change of the coefficient of x when the
                    observation is left out, in units of its standard
                    error computed with ``s(i)``
    ==============  ====================================================

    These are the definitions of Belsley, Kuh and Welsch (1980) that
    Stata's ``predict`` after ``regress`` and R's ``influence.measures``
    follow.

    Parameters
    ----------
    result : EconometricResults
        An unweighted fit of ``sp.regress``.

    Returns
    -------
    pandas.DataFrame
        One row per observation of the estimation sample, in its order,
        with the columns above and ``fitted`` and ``residual``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=50)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=50)
    >>> out = sp.influence_measures(sp.regress("y ~ x", data=df))
    >>> bool(np.isclose(out["leverage"].sum(), 2.0))
    True
    """
    x, e, names = _linear_fit(result, "influence_measures")
    n, k = x.shape
    if n <= k + 1:
        raise DataInsufficient(
            "sp.influence_measures: too few observations to leave one out.",
            recovery_hint="The fit needs more than k + 1 observations.",
        )
    q, r = np.linalg.qr(x)
    h = np.clip(np.sum(q * q, axis=1), 0.0, 1.0)
    s2 = float(e @ e) / (n - k)
    one_less = np.where(1 - h > 1e-12, 1 - h, np.nan)
    rstandard = e / np.sqrt(s2 * one_less)
    s2_i = ((n - k) * s2 - e**2 / one_less) / (n - k - 1)
    rstudent = e / np.sqrt(s2_i * one_less)
    dfits = rstudent * np.sqrt(h / one_less)
    out = pd.DataFrame(
        {
            "fitted": np.asarray(getattr(result, "data_info")["y"], dtype=float) - e,
            "residual": e,
            "leverage": h,
            "rstandard": rstandard,
            "rstudent": rstudent,
            "cooksd": rstandard**2 * h / (k * one_less),
            "dfits": dfits,
            "welsch": dfits * np.sqrt((n - 1) / one_less),
            "covratio": 1.0 / (one_less * ((n - k - 1 + rstudent**2) / (n - k)) ** k),
            "stdp": np.sqrt(s2 * h),
            "stdf": np.sqrt(s2 * (1 + h)),
            "stdr": np.sqrt(s2 * one_less),
        }
    )
    # (X'X)^-1 x_i e_i / (1 - h_i), scaled by s(i) sqrt((X'X)^-1_jj)
    r_inv = np.linalg.inv(r)
    change = (q @ r_inv.T) * (e / one_less)[:, None]
    scale = np.sqrt(np.sum(r_inv**2, axis=1))
    dfbeta = change / (np.sqrt(s2_i)[:, None] * scale[None, :])
    for j, name in enumerate(names):
        out[f"dfbeta_{name}"] = dfbeta[:, j]
    index = getattr(getattr(result, "data_info", {}).get("X"), "index", None)
    if index is not None and len(index) == n:
        out.index = index
    return out


# ------------------------------------------------------------- logistic
def _binary_fit(result: Any, who: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = getattr(result, "data_info", None) or {}
    model = getattr(result, "model_info", None) or {}
    x, y, p = info.get("X"), info.get("y"), info.get("fitted_values")
    link = str(model.get("link") or model.get("model_type") or "").lower()
    if x is None or y is None or p is None or model.get("family") != "binomial":
        raise MethodIncompatibility(
            f"sp.{who} needs a binary-outcome fit that stores its design "
            "matrix and fitted probabilities.",
            recovery_hint="Fit the model with sp.logit.",
        )
    if who == "logit_influence" and "logit" not in link:
        raise MethodIncompatibility(
            "sp.logit_influence: Pregibon's diagnostics are those of the "
            "logistic model.",
            recovery_hint="Fit the model with sp.logit.",
        )
    return (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(p, dtype=float),
    )


def _patterns(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """The covariate pattern of each row (0 .. J-1) and J."""
    _, inverse = np.unique(np.round(x, 12), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    return inverse, int(inverse.max()) + 1


def logit_influence(result: Any) -> pd.DataFrame:
    """Pregibon's (1981) diagnostics after a logistic regression.

    The statistics are computed for covariate patterns: the ``m`` rows
    that share the values of every regressor are one binomial observation
    with ``y`` successes and fitted probability ``p``, and every row of a
    pattern gets the pattern's value. With ``V`` the model-based covariance
    of the coefficients:

    ==============  ====================================================
    ``residual``    Pearson, ``(y - m p) / sqrt(m p (1 - p))``
    ``hat``         ``m p (1 - p) x V x'``, the leverage
    ``rstandard``   ``residual / sqrt(1 - hat)``
    ``deviance``    the signed square root of the pattern's deviance
    ``dx2``         ``rstandard^2``: the fall in Pearson's chi-squared
                    when the pattern is left out (Hosmer-Lemeshow)
    ``ddeviance``   ``deviance^2 / (1 - hat)``: the fall in the deviance
    ``dbeta``       ``residual^2 hat / (1 - hat)^2``: Pregibon's
                    influence on the coefficients
    ==============  ====================================================

    These are the statistics of Stata's ``predict`` after ``logit`` /
    ``logistic``.

    Parameters
    ----------
    result : EconometricResults
        A fit of ``sp.logit``.

    Returns
    -------
    pandas.DataFrame
        One row per observation of the estimation sample, with the
        columns above and ``pattern``, ``m``, ``successes``, ``p``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.integers(0, 4, size=200)})
    >>> df["d"] = (rng.random(200) < 0.2 + 0.15 * df["x"]).astype(int)
    >>> out = sp.logit_influence(sp.logit("d ~ x", data=df))
    >>> int(out["pattern"].nunique())
    4
    """
    x, y, p = _binary_fit(result, "logit_influence")
    pattern, n_patterns = _patterns(x)
    m = np.bincount(pattern, minlength=n_patterns).astype(float)
    successes = np.bincount(pattern, weights=y, minlength=n_patterns)
    first = np.unique(pattern, return_index=True)[1]
    pj, xj = p[first], x[first]
    w = p * (1 - p)
    v = np.linalg.inv((x * w[:, None]).T @ x)
    hat = m * pj * (1 - pj) * np.einsum("ij,jk,ik->i", xj, v, xj)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (successes - m * pj) / np.sqrt(m * pj * (1 - pj))
        up = np.where(successes > 0, successes * np.log(successes / (m * pj)), 0.0)
        failures = m - successes
        down = np.where(failures > 0, failures * np.log(failures / (m * (1 - pj))), 0.0)
        deviance = np.sign(successes - m * pj) * np.sqrt(np.maximum(2 * (up + down), 0))
        free = np.where(1 - hat > 1e-12, 1 - hat, np.nan)
        rstandard = resid / np.sqrt(free)
    by_pattern = {
        "m": m,
        "successes": successes,
        "p": pj,
        "residual": resid,
        "hat": hat,
        "rstandard": rstandard,
        "deviance": deviance,
        "dx2": rstandard**2,
        "ddeviance": deviance**2 / free,
        "dbeta": resid**2 * hat / free**2,
    }
    out = pd.DataFrame({"pattern": pattern + 1})
    for name, values in by_pattern.items():
        out[name] = values[pattern]
    return out


def logit_gof(result: Any, groups: Optional[int] = None) -> Dict[str, Any]:
    """Goodness of fit of a binary-outcome model.

    Without ``groups``, Pearson's chi-squared over the ``J`` covariate
    patterns, ``sum (y - m p)^2 / (m p (1 - p))``, on ``J - k`` degrees of
    freedom. With ``groups = g``, the Hosmer-Lemeshow test: the
    observations are sorted into ``g`` groups by the quantiles of the
    fitted probability (rows with equal probabilities stay together), and
    ``sum (O - E)^2 / (E (1 - E / n))`` over the groups is referred to
    chi-squared with ``g - 2`` degrees of freedom (Stata ``estat gof`` and
    ``estat gof, group(g)``).

    Parameters
    ----------
    result : EconometricResults
        A fit of ``sp.logit`` or ``sp.probit``.
    groups : int, optional

    Returns
    -------
    dict
        ``statistic``, ``df``, ``pvalue``, ``n``, ``n_patterns`` or
        ``n_groups``, and for the grouped test the ``table`` of observed
        and expected counts.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=300)})
    >>> df["d"] = (rng.random(300) < 1 / (1 + np.exp(-df["x"]))).astype(int)
    >>> out = sp.logit_gof(sp.logit("d ~ x", data=df), groups=10)
    >>> int(out["df"])
    8
    """
    x, y, p = _binary_fit(result, "logit_gof")
    n, k = x.shape
    if groups is None:
        pattern, n_patterns = _patterns(x)
        m = np.bincount(pattern, minlength=n_patterns).astype(float)
        successes = np.bincount(pattern, weights=y, minlength=n_patterns)
        pj = p[np.unique(pattern, return_index=True)[1]]
        stat = float(np.sum((successes - m * pj) ** 2 / (m * pj * (1 - pj))))
        df = n_patterns - k
        return {
            "test": "Pearson goodness-of-fit test",
            "statistic": stat,
            "df": df,
            "pvalue": float(stats.chi2.sf(stat, df)) if df > 0 else float("nan"),
            "n": n,
            "n_patterns": n_patterns,
        }
    if groups < 3:
        raise MethodIncompatibility(
            "sp.logit_gof: the Hosmer-Lemeshow test needs at least 3 groups.",
            recovery_hint="groups=10 is the usual choice.",
        )
    ordered = np.sort(p)
    cuts = []
    for q in range(1, groups):
        h = n * q / groups
        i = int(np.floor(h + 1e-12))
        if abs(h - i) < 1e-12:
            cuts.append((ordered[max(i, 1) - 1] + ordered[min(i + 1, n) - 1]) / 2.0)
        else:
            cuts.append(ordered[min(i + 1, n) - 1])
    group = np.searchsorted(np.asarray(cuts), p, side="left")
    rows = []
    for g in np.unique(group):
        sel = group == g
        rows.append(
            {
                "group": int(g) + 1,
                "prob": float(p[sel].max()),
                "obs_1": float(y[sel].sum()),
                "exp_1": float(p[sel].sum()),
                "obs_0": float(sel.sum() - y[sel].sum()),
                "exp_0": float(sel.sum() - p[sel].sum()),
                "total": int(sel.sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("group")
    stat = float(
        np.sum(
            (table["obs_1"] - table["exp_1"]) ** 2
            / (table["exp_1"] * (1 - table["exp_1"] / table["total"]))
        )
    )
    df = len(table) - 2
    return {
        "test": "Hosmer-Lemeshow goodness-of-fit test",
        "statistic": stat,
        "df": df,
        "pvalue": float(stats.chi2.sf(stat, df)) if df > 0 else float("nan"),
        "n": n,
        "n_groups": len(table),
        "table": table,
    }


def roc_area(y: Any, score: Any) -> Dict[str, float]:
    """Area under the ROC curve of ``score`` for the outcome ``y``.

    The Mann-Whitney estimate, ties counted one half, with the standard
    error of DeLong, DeLong and Clarke-Pearson (1988) -- the numbers of
    Stata's ``lroc`` and ``roctab``.
    """
    y = np.asarray(y, dtype=float) != 0
    score = np.asarray(score, dtype=float)
    pos, neg = score[y], score[~y]
    if pos.size == 0 or neg.size == 0:
        raise DataInsufficient(
            "roc_area: the outcome does not vary.",
            recovery_hint="Both outcomes must occur.",
        )
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    auc = (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (
        pos.size * neg.size
    )
    # placement values: for each case, the share of the other class below it
    neg_sorted, pos_sorted = np.sort(neg), np.sort(pos)
    v10 = (
        np.searchsorted(neg_sorted, pos, side="left")
        + 0.5 * (np.searchsorted(neg_sorted, pos, side="right")
                 - np.searchsorted(neg_sorted, pos, side="left"))
    ) / neg.size  # fmt: skip
    v01 = (
        pos.size
        - np.searchsorted(pos_sorted, neg, side="right")
        + 0.5 * (np.searchsorted(pos_sorted, neg, side="right")
                 - np.searchsorted(pos_sorted, neg, side="left"))
    ) / pos.size  # fmt: skip
    var = v10.var(ddof=1) / pos.size + v01.var(ddof=1) / neg.size
    return {"auc": float(auc), "se": float(np.sqrt(var)), "n": float(y.size)}
