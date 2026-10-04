"""Specification tests after a linear regression, behind :func:`statspai.estat`.

Each function takes a fitted linear result (anything with ``data_info``
holding ``X``, ``y``, ``residuals`` and ``fitted_values``) and returns a dict.
The options are the ones the textbook versions of the tests differ by, so
that both Stata's ``estat`` and R's ``lmtest`` defaults can be asked for by
name:

==============  ==========================  ===============================
test            Stata default               R (``lmtest``) default
==============  ==========================  ===============================
``hettest``     ``variables='fitted'``,     ``variables='rhs'``,
                ``version='normal'``        ``version='iid'``
``reset``       ``powers=4``                ``powers=3``
``bgodfrey``    ``fill='zero'``             ``fill='zero'``
==============  ==========================  ===============================

:func:`statspai.estat` uses the right-hand column.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility

__all__ = [
    "hettest",
    "white",
    "imtest",
    "reset",
    "bgodfrey",
    "vif",
    "information_criteria",
    "classification",
]


# ------------------------------------------------------------------ helpers
def _arrays(result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    info = getattr(result, "data_info", None) or {}
    missing = [k for k in ("X", "y", "residuals") if info.get(k) is None]
    if missing:
        raise MethodIncompatibility(
            f"sp.estat: the result does not store {', '.join(missing)}; this "
            "test needs a linear fit from sp.regress.",
            recovery_hint="Fit the model with sp.regress and pass that result.",
        )
    X = np.asarray(info["X"], dtype=float)
    y = np.asarray(info["y"], dtype=float)
    e = np.asarray(info["residuals"], dtype=float)
    names = [str(n) for n in (info.get("var_names") or result.params.index)]
    return X, y, e, names


def _constant_column(X: np.ndarray) -> Optional[int]:
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) == 0 and X[0, j] != 0:
            return j
    return None


def _independent(columns: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The columns of ``columns`` that add to the span of ``base``, in order.

    A power of a dummy variable is the dummy itself, and the square of one
    regressor can be another regressor already in the model; such terms
    carry no restriction and are left out, as Stata and R do. A column
    counts as redundant when what is left of it after projecting on the
    columns kept so far is below 1e-6 of its own length -- loose enough to
    catch a square that was stored in single precision.
    """
    kept = base / np.where(np.abs(base).max(axis=0) > 0, np.abs(base).max(axis=0), 1.0)
    out: List[int] = []
    for j in range(columns.shape[1]):
        col = columns[:, j]
        scale = float(np.abs(col).max())
        if scale == 0 or not np.isfinite(scale):
            continue
        col = col / scale
        beta = np.linalg.lstsq(kept, col, rcond=None)[0]
        left = col - kept @ beta
        if np.linalg.norm(left) > 1e-6 * np.linalg.norm(col):
            kept = np.column_stack([kept, col])
            out.append(j)
    return columns[:, out]


def _fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return resid, float(resid @ resid)


def _decision(pvalue: float, alpha: float, reject: str, keep: str) -> str:
    if pvalue < alpha:
        return f"REJECT H0 at {alpha:.0%}: {reject}"
    return f"Cannot reject H0 at {alpha:.0%}: {keep}"


# ------------------------------------------------------- heteroskedasticity
def hettest(
    result: Any,
    *,
    variables: Union[None, str, Sequence[str], np.ndarray] = None,
    version: str = "iid",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Breusch-Pagan / Cook-Weisberg test.

    ``variables`` chooses what the error variance may depend on: ``'rhs'``
    (the default; every regressor), ``'fitted'`` (the fitted values, one
    degree of freedom), a list of regressor names, or an array with one row
    per estimation observation. ``version`` is ``'iid'`` (Koenker's N R^2,
    which does not assume normal errors), ``'normal'`` (the original score
    test, half the explained sum of squares) or ``'fstat'``.
    """
    X, y, e, names = _arrays(result)
    n = X.shape[0]
    const = _constant_column(X)
    others = [j for j in range(X.shape[1]) if j != const]
    if variables is None or (isinstance(variables, str) and variables == "rhs"):
        Z, label = X[:, others], "all regressors"
    elif isinstance(variables, str) and variables == "fitted":
        fitted = (getattr(result, "data_info", None) or {}).get("fitted_values")
        fitted = y - e if fitted is None else np.asarray(fitted, dtype=float)
        Z, label = fitted.reshape(-1, 1), "fitted values"
    elif isinstance(variables, np.ndarray):
        Z = variables.reshape(n, -1).astype(float)
        label = f"{Z.shape[1]} supplied variable(s)"
    else:
        wanted = [variables] if isinstance(variables, str) else list(variables)
        unknown = [v for v in wanted if v not in names]
        if unknown:
            raise MethodIncompatibility(
                f"sp.estat hettest: {unknown} are not regressors of the model "
                f"({', '.join(names)}).",
                recovery_hint="Name regressors of the model, or pass the "
                "variables as an array aligned with the estimation sample.",
            )
        Z, label = X[:, [names.index(v) for v in wanted]], ", ".join(wanted)
    Z = _independent(Z, np.ones((n, 1)))
    df = Z.shape[1]
    if df == 0:
        raise MethodIncompatibility(
            "sp.estat hettest: the variables have no variation.",
            recovery_hint="Choose variables that vary in the sample.",
        )
    design = np.column_stack([np.ones(n), Z])
    version = version.lower()
    if version not in ("iid", "normal", "fstat"):
        raise MethodIncompatibility(
            f"sp.estat hettest: version={version!r} is not one of 'iid', "
            "'normal', 'fstat'.",
            recovery_hint="Use version='iid' (the default).",
        )
    e2 = e**2
    out: Dict[str, Any] = {
        "test": "Breusch-Pagan test for heteroskedasticity",
        "H0": "Constant variance (homoskedasticity)",
        "H1": f"Variance depends on {label}",
        "variables": label,
        "version": version,
    }
    if version == "normal":
        g = e2 / (e @ e / n)
        _, rss = _fit(design, g)
        stat = 0.5 * (float(((g - g.mean()) ** 2).sum()) - rss)
        pval = float(sp_stats.chi2.sf(stat, df))
        out.update(statistic_label=f"chi2({df})", df=df)
    else:
        _, rss = _fit(design, e2)
        tss = float(((e2 - e2.mean()) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        if version == "iid":
            stat = n * r2
            pval = float(sp_stats.chi2.sf(stat, df))
            out.update(statistic_label=f"chi2({df})", df=df)
        else:
            df2 = n - df - 1
            stat = (r2 / df) / ((1.0 - r2) / df2)
            pval = float(sp_stats.f.sf(stat, df, df2))
            out.update(statistic_label=f"F({df}, {df2})", df1=df, df2=df2)
    out.update(
        statistic=float(stat),
        pvalue=pval,
        interpretation=_decision(
            pval,
            alpha,
            "evidence of heteroskedasticity. Consider robust standard errors.",
            "no evidence of heteroskedasticity.",
        ),
    )
    return out


def white(
    result: Any,
    *,
    variables: Union[None, str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """White's general test: N R^2 from the regression of the squared
    residuals on the regressors, their squares and their cross-products.
    Redundant terms (the square of a dummy) do not count as restrictions.

    ``variables='fitted'`` is the special case that regresses the squared
    residuals on the fitted values and their squares: two restrictions
    whatever the number of regressors, so it keeps its power in a model
    with many of them.
    """
    X, y, e, _ = _arrays(result)
    n = X.shape[0]
    if variables is not None and variables not in ("rhs", "fitted"):
        raise MethodIncompatibility(
            f"sp.estat white: variables={variables!r} is not 'rhs' or 'fitted'.",
            recovery_hint="Use variables='fitted' for the special form of the "
            "test, or leave it out for the general one.",
        )
    special = variables == "fitted"
    if special:
        fitted = (getattr(result, "data_info", None) or {}).get("fitted_values")
        fitted = y - e if fitted is None else np.asarray(fitted, dtype=float)
        terms = [fitted, fitted**2]
    else:
        const = _constant_column(X)
        cols = [j for j in range(X.shape[1]) if j != const]
        terms = [X[:, j] for j in cols]
        terms += [X[:, j] ** 2 for j in cols]
        terms += [X[:, a] * X[:, b] for a, b in combinations(cols, 2)]
    Z = _independent(np.column_stack(terms), np.ones((n, 1)))
    df = Z.shape[1]
    e2 = e**2
    _, rss = _fit(np.column_stack([np.ones(n), Z]), e2)
    tss = float(((e2 - e2.mean()) ** 2).sum())
    stat = n * (1.0 - rss / tss) if tss > 0 else 0.0
    pval = float(sp_stats.chi2.sf(stat, df))
    return {
        "test": "White's test for heteroskedasticity"
        + (" (fitted values and their squares)" if special else ""),
        "H0": "Homoskedasticity",
        "H1": (
            "Variance depends on the fitted values and their squares"
            if special
            else "Unrestricted heteroskedasticity"
        ),
        "variables": "fitted" if special else "rhs",
        "statistic": float(stat),
        "statistic_label": f"chi2({df})",
        "df": df,
        "pvalue": pval,
        "interpretation": _decision(
            pval,
            alpha,
            "evidence of heteroskedasticity (general form). Consider robust "
            "standard errors.",
            "no evidence of heteroskedasticity.",
        ),
    }


def imtest(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Information-matrix test of a linear regression, in Cameron and
    Trivedi's decomposition (Stata ``estat imtest``).

    Three orthogonal parts: heteroskedasticity (White's test), skewness
    (``N`` times the uncentred R-squared of ``e^3 - 3 s^2 e`` on the
    regressors) and kurtosis (the same for ``e^4 - 6 s^2 e^2 + 3 s^4`` on a
    constant), with ``s^2 = RSS / N``. The total is their sum. The main
    ``statistic`` is the heteroskedasticity part.
    """
    X, _, e, _ = _arrays(result)
    n = X.shape[0]
    s2 = float(e @ e) / n
    out = white(result, alpha=alpha)

    def uncentred(target: np.ndarray, design: np.ndarray) -> float:
        _, rss = _fit(design, target)
        total = float(target @ target)
        return n * (1.0 - rss / total) if total > 0 else 0.0

    const = _constant_column(X)
    df_skew = X.shape[1] - (1 if const is not None else 0)
    skew = uncentred(e**3 - 3.0 * s2 * e, X)
    kurt = uncentred(e**4 - 6.0 * s2 * e**2 + 3.0 * s2**2, np.ones((n, 1)))
    total = out["statistic"] + skew + kurt
    df_total = out["df"] + df_skew + 1
    table = pd.DataFrame(
        {
            "chi2": [out["statistic"], skew, kurt, total],
            "df": [out["df"], df_skew, 1, df_total],
        },
        index=["heteroskedasticity", "skewness", "kurtosis", "total"],
    )
    table["p"] = sp_stats.chi2.sf(table["chi2"], table["df"])
    out.update(
        test="Information-matrix test (Cameron-Trivedi decomposition)",
        table=table,
        skewness=float(skew),
        kurtosis=float(kurt),
        total=float(total),
        total_df=int(df_total),
        total_pvalue=float(table.loc["total", "p"]),
    )
    return out


# ----------------------------------------------------------- functional form
def reset(
    result: Any, *, powers: int = 3, rhs: bool = False, alpha: float = 0.05
) -> Dict[str, Any]:
    """Ramsey's RESET: an F test of the added powers 2 .. ``powers`` of the
    fitted values, or with ``rhs=True`` of every regressor."""
    X, y, e, _ = _arrays(result)
    n, k = X.shape
    if powers < 2:
        raise MethodIncompatibility(
            "sp.estat reset: powers must be at least 2.",
            recovery_hint="powers=3 adds the square and the cube.",
        )
    if rhs:
        const = _constant_column(X)
        base = [X[:, j] for j in range(k) if j != const]
    else:
        fitted = y - e
        # the scale of the fitted values does not change the test; unit
        # scale keeps the fourth power well conditioned
        spread = float(np.ptp(fitted)) or 1.0
        base = [(fitted - fitted.min()) / spread]
    added = np.column_stack([b**p for p in range(2, powers + 1) for b in base])
    added = _independent(added, X)
    df1 = added.shape[1]
    df2 = n - k - df1
    if df1 == 0 or df2 <= 0:
        raise MethodIncompatibility(
            "sp.estat reset: no power adds to the model (or no residual "
            "degrees of freedom are left).",
            recovery_hint="Use fewer powers, or check that the regressors vary.",
        )
    rss_r = float(e @ e)
    _, rss_u = _fit(np.column_stack([X, added]), y)
    stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    pval = float(sp_stats.f.sf(stat, df1, df2))
    what = "regressors" if rhs else "fitted values"
    return {
        "test": "Ramsey RESET test",
        "H0": "Model has no omitted nonlinearities",
        "H1": f"Powers of the {what} are significant",
        "statistic": float(stat),
        "statistic_label": f"F({df1}, {df2})",
        "df1": df1,
        "df2": df2,
        "pvalue": pval,
        "powers": powers,
        "rhs": rhs,
        "interpretation": _decision(
            pval,
            alpha,
            "functional form may be misspecified. Consider adding nonlinear "
            "terms or transformations.",
            "no evidence of misspecification.",
        ),
    }


# -------------------------------------------------------- serial correlation
def bgodfrey(
    result: Any,
    *,
    lags: int = 1,
    fill: str = "zero",
    version: str = "iid",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Breusch-Godfrey LM test for serial correlation up to ``lags``.

    ``version='fstat'`` reports the F test that the lagged residuals have
    zero coefficients in the auxiliary regression instead of N R^2 (the F
    form of R's ``lmtest::bgtest`` and of statsmodels; Stata prints this
    statistic as ``estat durbinalt, small``, and its ``estat bgodfrey,
    small`` is the chi-squared statistic divided by ``lags``).

    The residuals are regressed on the regressors and their own lags. The
    lagged residual does not exist for the first ``lags`` observations:
    ``fill='zero'`` sets it to zero and keeps every observation (the default
    of Stata's ``estat bgodfrey`` and of R's ``lmtest::bgtest``);
    ``fill='drop'`` leaves those observations out (Stata's ``nomiss0``).
    The rows must be in time order.
    """
    X, _, e, _ = _arrays(result)
    n = X.shape[0]
    if fill not in ("zero", "drop"):
        raise MethodIncompatibility(
            f"sp.estat bgodfrey: fill={fill!r} is not 'zero' or 'drop'.",
            recovery_hint="Use fill='zero' (the default).",
        )
    if lags < 1 or lags >= n:
        raise MethodIncompatibility(
            f"sp.estat bgodfrey: lags={lags} is outside 1 .. {n - 1}.",
            recovery_hint="Use a small number of lags.",
        )
    lagged = np.zeros((n, lags))
    for p in range(1, lags + 1):
        lagged[p:, p - 1] = e[:-p]
    rows = slice(lags, None) if fill == "drop" else slice(None)
    design = np.column_stack([X, lagged])[rows]
    target = e[rows]
    _, rss = _fit(design, target)
    tss = float(((target - target.mean()) ** 2).sum())
    version = str(version).lower()
    if version not in ("iid", "fstat"):
        raise MethodIncompatibility(
            f"sp.estat bgodfrey: version={version!r} is not 'iid' or 'fstat'.",
            recovery_hint="Use version='iid' (N R-squared, the default) or "
            "version='fstat'.",
        )
    word = "lag" if lags == 1 else "lags"
    shape: Dict[str, Any]
    if version == "fstat":
        _, rss_restricted = _fit(X[rows], target)
        df2 = len(target) - design.shape[1]
        stat = ((rss_restricted - rss) / lags) / (rss / df2)
        pval = float(sp_stats.f.sf(stat, lags, df2))
        shape = {"statistic_label": f"F({lags}, {df2})", "df1": lags, "df2": df2}
    else:
        stat = len(target) * (1.0 - rss / tss) if tss > 0 else 0.0
        pval = float(sp_stats.chi2.sf(stat, lags))
        shape = {"statistic_label": f"chi2({lags})", "df": lags}
    return {
        "test": f"Breusch-Godfrey LM test ({lags} {word})",
        "H0": "No serial correlation",
        "H1": f"Serial correlation up to order {lags}",
        "statistic": float(stat),
        **shape,
        "version": version,
        "pvalue": pval,
        "lags": lags,
        "fill": fill,
        "interpretation": _decision(
            pval,
            alpha,
            f"evidence of serial correlation up to {lags} {word}. Consider "
            "Newey-West standard errors.",
            f"no evidence of serial correlation up to {lags} {word}.",
        ),
    }


def durbinalt(
    result: Any, *, lags: int = 1, version: str = "iid", alpha: float = 0.05
) -> Dict[str, Any]:
    """Durbin's alternative test for serial correlation up to ``lags``.

    The residuals are regressed on the regressors and their own lags, the
    missing lags set to zero, and the lagged residuals are tested jointly:
    ``version='fstat'`` is that F statistic, the default is ``lags`` times
    it, referred to a chi-squared with ``lags`` degrees of freedom (Stata
    ``estat durbinalt`` with and without ``small``). Unlike the
    Durbin-Watson statistic it stays valid when the regressors are not
    strictly exogenous, for example with a lagged dependent variable.
    """
    f_form = bgodfrey(result, lags=lags, fill="zero", version="fstat", alpha=alpha)
    version = str(version).lower()
    if version not in ("iid", "fstat"):
        raise MethodIncompatibility(
            f"sp.estat durbinalt: version={version!r} is not 'iid' or 'fstat'.",
            recovery_hint="Use version='iid' (the default) or version='fstat'.",
        )
    out = dict(f_form)
    out["test"] = f"Durbin's alternative test ({lags} {'lag' if lags == 1 else 'lags'})"
    out["version"] = version
    if version == "iid":
        stat = lags * float(f_form["statistic"])
        pval = float(sp_stats.chi2.sf(stat, lags))
        for key in ("df1", "df2"):
            out.pop(key, None)
        word = "lag" if lags == 1 else "lags"
        out.update(
            statistic=stat,
            statistic_label=f"chi2({lags})",
            df=lags,
            pvalue=pval,
            interpretation=_decision(
                pval,
                alpha,
                f"evidence of serial correlation up to {lags} {word}. Consider "
                "Newey-West standard errors.",
                f"no evidence of serial correlation up to {lags} {word}.",
            ),
        )
    return out


def archlm(
    result: Any, *, lags: int = 1, version: str = "iid", alpha: float = 0.05
) -> Dict[str, Any]:
    """Engle's LM test for autoregressive conditional heteroskedasticity.

    The squared residuals are regressed on a constant and ``lags`` of their
    own; the statistic is ``(T - lags) R^2``, chi-squared with ``lags``
    degrees of freedom under the null of no ARCH (Stata ``estat archlm``),
    or with ``version='fstat'`` the F statistic of that regression. The
    rows must be in time order.
    """
    _, _, e, _ = _arrays(result)
    n = e.shape[0]
    if lags < 1 or lags >= n - 2:
        raise MethodIncompatibility(
            f"sp.estat archlm: lags={lags} is outside 1 .. {n - 3}.",
            recovery_hint="Use a small number of lags.",
        )
    version = str(version).lower()
    if version not in ("iid", "fstat"):
        raise MethodIncompatibility(
            f"sp.estat archlm: version={version!r} is not 'iid' or 'fstat'.",
            recovery_hint="Use version='iid' (the default) or version='fstat'.",
        )
    e2 = e**2
    target = e2[lags:]
    design = np.column_stack(
        [np.ones(n - lags)] + [e2[lags - p : n - p] for p in range(1, lags + 1)]
    )
    _, rss = _fit(design, target)
    tss = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    shape: Dict[str, Any]
    if version == "fstat":
        df2 = len(target) - lags - 1
        stat = (r2 / lags) / ((1.0 - r2) / df2)
        pval = float(sp_stats.f.sf(stat, lags, df2))
        shape = {"statistic_label": f"F({lags}, {df2})", "df1": lags, "df2": df2}
    else:
        stat = len(target) * r2
        pval = float(sp_stats.chi2.sf(stat, lags))
        shape = {"statistic_label": f"chi2({lags})", "df": lags}
    word = "lag" if lags == 1 else "lags"
    return {
        "test": f"LM test for ARCH effects ({lags} {word})",
        "H0": "No ARCH effects",
        "H1": f"ARCH({lags}) disturbance",
        "statistic": float(stat),
        **shape,
        "version": version,
        "pvalue": pval,
        "lags": lags,
        "interpretation": _decision(
            pval,
            alpha,
            "the error variance depends on past squared errors. The usual "
            "standard errors remain valid under the other assumptions, but "
            "see sp.garch for the variance dynamics.",
            f"no evidence of ARCH effects up to {lags} {word}.",
        ),
    }


# ------------------------------------------------------- multicollinearity
def vif(result: Any, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Variance inflation factor of every regressor: ``1 / (1 - R_j^2)`` from
    the regression of regressor ``j`` on the others and a constant."""
    X, _, _, names = _arrays(result)
    n, k = X.shape
    const = _constant_column(X)
    cols = [j for j in range(k) if j != const]
    rows = []
    for j in cols:
        target = X[:, j]
        design = np.column_stack([np.ones(n)] + [X[:, c] for c in cols if c != j])
        _, rss = _fit(design, target)
        tss = float(((target - target.mean()) ** 2).sum())
        tolerance = rss / tss if tss > 0 else 0.0
        rows.append(
            {
                "variable": names[j] if j < len(names) else f"x{j}",
                "VIF": 1.0 / tolerance if tolerance > 0 else np.inf,
                "1/VIF": tolerance,
            }
        )
    table = pd.DataFrame(rows, columns=["variable", "VIF", "1/VIF"])
    top = float(table["VIF"].max()) if len(table) else 0.0
    if top > 10:
        interp = (
            f"Max VIF = {top:.2f}: serious multicollinearity. Consider dropping "
            "or combining variables."
        )
    elif top > 5:
        interp = f"Max VIF = {top:.2f}: moderate multicollinearity."
    else:
        interp = f"Max VIF = {top:.2f}: no multicollinearity concern."
    return {
        "test": "Variance Inflation Factors",
        "vif_table": table,
        "mean_vif": float(table["VIF"].mean()) if len(table) else 0.0,
        "max_vif": top,
        "interpretation": interp,
    }


# ---------------------------------------------------- information criteria
def information_criteria(result: Any) -> Dict[str, Any]:
    """Log likelihood, AIC, BIC and HQIC.

    ``AIC = -2 log L + 2k`` and ``BIC = -2 log L + k log N`` with ``k`` the
    number of estimated coefficients, as Stata's ``estat ic`` counts them
    (R's ``AIC()`` adds one for the error variance of a linear model). For a
    linear fit the Gaussian log likelihood is evaluated at the maximum
    likelihood variance ``RSS / N``.
    """
    info = getattr(result, "data_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}
    model = getattr(result, "model_info", None) or {}
    k = int(len(result.params))
    resid = info.get("residuals")
    n = int(info.get("nobs") or getattr(result, "nobs", 0) or 0)
    # a likelihood the model reports itself comes first: after logit the
    # residuals are not Gaussian errors
    ll = model.get("ll", diag.get("Log-Likelihood", diag.get("log_likelihood")))
    if ll is None or not np.isfinite(ll):
        if resid is None or info.get("X") is None:
            raise MethodIncompatibility(
                "sp.estat ic: the result stores neither a log likelihood nor "
                "residuals.",
                recovery_hint="Pass a likelihood-based or linear fit.",
            )
        resid = np.asarray(resid, dtype=float)
        n = int(resid.size)
        rss = float(resid @ resid)
        ll = -0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0) if rss > 0 else np.inf
    if n <= 0:
        raise MethodIncompatibility(
            "sp.estat ic: the number of observations is not stored.",
            recovery_hint="Pass a result fitted by sp.regress / sp.logit / ...",
        )
    ll = float(ll)
    return {
        "test": "Information Criteria",
        "ll": ll,
        "AIC": -2.0 * ll + 2.0 * k,
        "BIC": -2.0 * ll + k * float(np.log(n)),
        "HQIC": -2.0 * ll + 2.0 * k * float(np.log(np.log(n))) if n > 2 else np.nan,
        "n": n,
        "k": k,
        "interpretation": (
            "Lower values indicate a better fit-complexity trade-off; compare "
            "models fitted on the same observations. BIC penalises "
            "complexity more than AIC."
        ),
    }


# ------------------------------------------------------- classification
def classification(result: Any, *, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Classification table after a binary-outcome model.

    An observation is classified as positive when its fitted probability is
    at least ``threshold`` (0.5 unless given), as Stata's ``estat
    classification`` does.
    """
    info = getattr(result, "data_info", None) or {}
    model = getattr(result, "model_info", None) or {}
    y, prob = info.get("y"), info.get("fitted_values")
    if y is None or prob is None or model.get("family") != "binomial":
        raise MethodIncompatibility(
            "sp.estat classification needs a binary-outcome fit (sp.logit, "
            "sp.probit) that stores its fitted probabilities.",
            recovery_hint="Fit the model with sp.logit or sp.probit.",
        )
    cut = 0.5 if threshold is None else float(threshold)
    y = np.asarray(y, dtype=float) != 0
    positive = np.asarray(prob, dtype=float) >= cut
    tp = int(np.sum(positive & y))
    fp = int(np.sum(positive & ~y))
    fn = int(np.sum(~positive & y))
    tn = int(np.sum(~positive & ~y))

    def share(a: int, b: int) -> float:
        return a / b if b else float("nan")

    table = pd.DataFrame(
        {"D": [tp, fn, tp + fn], "~D": [fp, tn, fp + tn]},
        index=["classified +", "classified -", "Total"],
    )
    table["Total"] = table["D"] + table["~D"]
    return {
        "test": "Classification table",
        "threshold": cut,
        "table": table,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": share(tp, tp + fn),
        "specificity": share(tn, tn + fp),
        "ppv": share(tp, tp + fp),
        "npv": share(tn, tn + fn),
        "correctly_classified": share(tp + tn, tp + fp + fn + tn),
        "interpretation": (
            f"{100 * share(tp + tn, tp + fp + fn + tn):.2f}% of the observations "
            f"are classified correctly at the threshold {cut:g}."
        ),
    }
