"""
CATE diagnostics and visualisation for Meta-Learners.

Provides tools for analysing and presenting heterogeneous treatment
effect estimates:
- ``cate_summary``: descriptive statistics of CATE distribution
- ``cate_plot``: histogram / density of individual-level effects
- ``cate_by_group``: group-level ATE with CIs (e.g., by quartile)
- ``cate_importance``: variable importance for CATE heterogeneity
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def cate_summary(result: CausalResult) -> pd.DataFrame:
    """
    Descriptive statistics of the CATE distribution.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()`` containing ``model_info['cate']``.

    Returns
    -------
    pd.DataFrame
        Summary statistics of the fitted conditional effects: mean, sd,
        min, q25, median, q75, max, fraction positive, IQR, N. The
        ``"Mean"`` row is the plain average of the fitted CATEs; it is not
        the estimator's doubly-robust average effect (use
        ``average_treatment_effect()`` on a forest), and for a continuous
        treatment it averages partial effects. It was labelled
        ``"Mean (ATE)"`` before 1.32.0.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> tab = sp.cate_summary(res)
    >>> int(tab.loc["N", "CATE"])
    400
    >>> round(float(tab.loc["Frac. Positive", "CATE"]), 2)
    0.78
    """
    cate = _extract_cate(result)

    summary = {
        "Mean": np.mean(cate),
        "Std. Dev.": np.std(cate, ddof=1),
        "Min": np.min(cate),
        "Q25": np.percentile(cate, 25),
        "Median": np.median(cate),
        "Q75": np.percentile(cate, 75),
        "Max": np.max(cate),
        "Frac. Positive": np.mean(cate > 0),
        "IQR": np.percentile(cate, 75) - np.percentile(cate, 25),
        "N": len(cate),
    }
    return pd.DataFrame(summary, index=["CATE"]).T


def cate_by_group(
    result: Any,
    data: pd.DataFrame,
    by: str,
    n_groups: int = 4,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Group-level average treatment effects.

    Splits the CATE distribution by a covariate (or by CATE quartiles
    if ``by='cate'``) and reports group means with confidence intervals.

    Parameters
    ----------
    result : CausalResult, CausalForest, or array-like
        Result from ``metalearner()`` / ``tarnet()``, a fitted
        ``causal_forest()`` model, or a raw array of per-unit CATE
        estimates.
    data : pd.DataFrame
        Original data (same rows as the estimation sample).
    by : str
        Column name to group by, or 'cate' to group by CATE quartiles.
    n_groups : int, default 4
        Number of quantile groups when ``by='cate'`` or when the
        grouping variable is continuous.
    alpha : float, default 0.05
        Significance level for CIs.

    Returns
    -------
    pd.DataFrame
        Columns: group, n, mean_cate, se, ci_lower, ci_upper.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> groups = sp.cate_by_group(res, df, by="cate", n_groups=4)
    >>> groups["n"].tolist()
    [100, 100, 100, 100]
    >>> round(float(groups["mean_cate"].iloc[-1]), 2)
    2.7
    """
    cate = _extract_cate(result)

    if by == "cate":
        labels = pd.qcut(cate, q=n_groups, labels=False, duplicates="drop")
        group_name = "CATE Quartile"
    else:
        if by not in data.columns:
            raise ValueError(f"Column '{by}' not found in data")
        col = data[by].values[: len(cate)]
        if pd.api.types.is_numeric_dtype(col) and len(np.unique(col)) > n_groups:
            labels = pd.qcut(col, q=n_groups, labels=False, duplicates="drop")
        else:
            labels = col
        group_name = by

    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for g in sorted(np.unique(labels)):
        mask = labels == g
        c = cate[mask]
        n = len(c)
        m = float(np.mean(c))
        se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {
                "group": g,
                "n": n,
                "mean_cate": m,
                "se": se,
                "ci_lower": m - z * se,
                "ci_upper": m + z * se,
            }
        )

    df = pd.DataFrame(rows)
    df.index.name = group_name
    return df


def cate_plot(
    result: CausalResult,
    kind: str = "hist",
    ax: Any = None,
    figsize: tuple = (8, 5),
    color: str = "#2C3E50",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Plot the CATE distribution.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()``.
    kind : str, default 'hist'
        'hist' for histogram, 'kde' for kernel density, 'both'.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> fig, ax = sp.cate_plot(res, kind="hist")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    cate = _extract_cate(result)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if kind in ("hist", "both"):
        ax.hist(
            cate,
            bins=kwargs.get("bins", 40),
            density=True,
            alpha=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    if kind in ("kde", "both"):
        from scipy.stats import gaussian_kde

        xs = np.linspace(cate.min(), cate.max(), 300)
        kde = gaussian_kde(cate)
        ax.plot(xs, kde(xs), color=color, linewidth=2)

    # Mark ATE
    ate = np.mean(cate)
    ax.axvline(
        ate, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"ATE = {ate:.3f}"
    )
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    learner_name = result.model_info.get("learner", "Meta-Learner")
    ax.set_xlabel("Conditional Average Treatment Effect (CATE)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title or f"CATE Distribution ({learner_name})", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def cate_group_plot(
    group_df: pd.DataFrame,
    ax: Any = None,
    figsize: tuple = (8, 5),
    color: str = "#2C3E50",
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot group-level CATEs with confidence intervals.

    Parameters
    ----------
    group_df : pd.DataFrame
        Output from ``cate_by_group()``.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> groups = sp.cate_by_group(res, df, by="cate", n_groups=4)
    >>> fig, ax = sp.cate_group_plot(groups)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    groups = group_df["group"].values
    means = group_df["mean_cate"].values
    ci_lo = group_df["ci_lower"].values
    ci_hi = group_df["ci_upper"].values

    x = np.arange(len(groups))
    ax.bar(x, means, color=color, alpha=0.7, edgecolor="white")
    ax.errorbar(
        x,
        means,
        yerr=[means - ci_lo, ci_hi - means],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.2,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_ylabel("Mean CATE", fontsize=11)
    ax.set_title(title or "Group-Level Treatment Effects", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def _cddf_inputs(
    result: Any,
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_folds: int,
    propensity: Any,
    baseline: Any,
    proxy: Any,
    seed: int,
    context: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, str, str
]:
    """Outcome, treatment, propensity, baseline, proxy and their provenance."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    from . import _cddf

    in_sample = _extract_cate(result)
    n = len(in_sample)
    for col in [y, treat] + list(covariates):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"Column '{col}' not found in data",
                recovery_hint="Pass the same data frame used to fit the CATE model.",
            )
    Y = data[y].to_numpy(dtype=float)[:n]
    D = data[treat].to_numpy(dtype=float)[:n]
    X = data[list(covariates)].to_numpy(dtype=float)[:n]
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"{context} requires a binary 0/1 treatment.",
            recovery_hint="Recode the treatment as 0/1.",
        )

    if propensity is None:
        from sklearn.ensemble import GradientBoostingClassifier

        prop_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        p: np.ndarray = np.zeros(n, dtype=float)
        for train_idx, test_idx in kf.split(X):
            m = clone(prop_model)
            m.fit(X[train_idx], D[train_idx])
            p[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        p = np.clip(p, 0.01, 0.99)
        p_source = "cross_fit_gbm"
    else:
        p = _cddf.resolve_vector(propensity, data, n, "propensity")
        p_source = "supplied"
    _cddf.check_propensity(p)
    B = (
        None
        if baseline is None
        else _cddf.resolve_vector(baseline, data, n, "baseline")
    )
    S, s_source = _cddf.resolve_proxy(
        result, proxy, data, X, Y, D, n_folds, seed, in_sample, context
    )
    return Y, D, p, B, S, p_source, s_source


def blp_test(
    result: CausalResult,
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_folds: int = 5,
    alpha: float = 0.05,
    *,
    propensity: Any = None,
    baseline: Any = None,
    proxy: Any = "cross_fit",
    vce: str = "HC1",
    seed: int = 42,
) -> Dict[str, Any]:
    """Best Linear Predictor (BLP) of the CATE, Chernozhukov-Demirer-Duflo-Fernandez-Val.

    Weighted least squares with weights ``w = 1 / (p(Z)(1 - p(Z)))``::

        Y = a0 + a1 B(Z) + beta_1 (D - p(Z))
            + beta_2 (D - p(Z)) (S(Z) - mean(S)) + e

    where ``S(Z)`` is the CATE proxy, ``p(Z)`` the propensity score and
    ``B(Z)`` an optional baseline proxy (e.g. a prediction of ``E[Y | Z,
    D = 0]``). ``beta_1`` is the ATE; ``beta_2`` is the slope of the true
    CATE on the proxy -- ``beta_2 = 0`` means the proxy carries no
    heterogeneity signal, ``beta_2 = 1`` a perfectly calibrated one. This
    is ``GenericML::BLP`` and, fed the same ``Y, D, p, B, S``, reproduces
    it to the floating-point floor.

    The proxy must not be fit on the outcomes it is evaluated against:
    in-sample predictions of a flexible learner correlate with the noise in
    ``Y`` and push ``beta_2`` up even when the effect is constant. By
    default (``proxy='cross_fit'``) the metalearner behind ``result`` is
    refit out of fold (``n_folds``, ``seed``) to produce ``S``.

    .. versionchanged:: 1.30.0
       Now the CDDF weighted regression with an out-of-fold proxy.
       Earlier releases ran unweighted OLS on the in-sample CATE
       predictions: with no true heterogeneity the default T-learner then
       reported ``beta_2`` near 1.4 with ``p < 1e-50``. The docstring's
       claim of equivalence to ``grf::test_calibration`` was also wrong;
       that test is :func:`sp.calibration_test`.

    Parameters
    ----------
    result : CausalResult, fitted CATE model, or array
        A ``metalearner()`` result (its learner is refit out of fold), or
        per-unit CATE predictions used as supplied.
    data : pd.DataFrame
    y, treat : str
        Outcome and binary treatment columns.
    covariates : list of str
        Covariates ``Z`` (for the propensity and the proxy refit).
    n_folds : int, default 5
        Folds for the propensity and proxy cross-fits.
    alpha : float, default 0.05
    propensity : array or column name, optional
        Known or estimated ``p(Z)``. Default: 5-fold gradient-boosting
        classifier, clipped to [0.01, 0.99]. In a randomized experiment
        pass the design probabilities.
    baseline : array or column name, optional
        Baseline proxy ``B(Z)`` added as a control.
    proxy : {'cross_fit', 'in_sample'} or array
        Source of ``S(Z)``.
    vce : {'HC1', 'const', 'HC0', 'HC2', 'HC3'}
        Covariance of the WLS fit (``sandwich::vcovHC`` types; GenericML's
        default is ``'const'``).
    seed : int, default 42
        Fold seed for the proxy refit.

    Returns
    -------
    dict
        ``beta1`` / ``beta2`` with ``_se``, ``_pvalue`` (two-sided),
        ``_pvalue_right`` (one-sided, H1: beta > 0) and ``_ci``;
        ``heterogeneity_significant`` (two-sided test of ``beta2 = 0`` at
        ``alpha``); ``proxy_source``, ``propensity_source``, ``vcov_type``.

    References
    ----------
    Chernozhukov, Demirer, Duflo and Fernandez-Val (2025), Econometrica
    93(4), 1121-1164. [@chernozhukov2025generic]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 800
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2,
    ...                    "p": np.full(n, 0.5)})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> blp = sp.blp_test(res, df, y="y", treat="d",
    ...                   covariates=["x1", "x2"], propensity="p")
    >>> blp["heterogeneity_significant"]
    True
    >>> blp["proxy_source"]
    'cross_fit'
    """
    from scipy import stats as _st

    from . import _cddf

    Y, D, p, B, S, p_source, s_source = _cddf_inputs(
        result,
        data,
        y,
        treat,
        covariates,
        n_folds,
        propensity,
        baseline,
        proxy,
        seed,
        "blp_test()",
    )
    n = len(Y)
    cols = [np.ones(n)]
    if B is not None:
        cols.append(B)
    k0 = len(cols)
    cols += [D - p, (D - p) * (S - S.mean())]
    beta, V = _cddf.wls_fit(Y, np.column_stack(cols), 1.0 / (p * (1.0 - p)), vce)
    se = np.sqrt(np.diag(V))
    z = float(_st.norm.ppf(1 - alpha / 2))
    out: Dict[str, Any] = {}
    for j, name in ((k0, "beta1"), (k0 + 1, "beta2")):
        stat = beta[j] / se[j]
        out[name] = float(beta[j])
        out[f"{name}_se"] = float(se[j])
        out[f"{name}_pvalue"] = float(2 * _st.norm.sf(abs(stat)))
        out[f"{name}_pvalue_right"] = float(_st.norm.sf(stat))
        out[f"{name}_ci"] = (float(beta[j] - z * se[j]), float(beta[j] + z * se[j]))
    out["heterogeneity_significant"] = bool(out["beta2_pvalue"] < alpha)
    out["proxy_source"] = s_source
    out["propensity_source"] = p_source
    out["vcov_type"] = vce
    out["n"] = n
    return out


def gate_test(
    result: CausalResult,
    data: pd.DataFrame,
    by: str,
    n_groups: int = 4,
    alpha: float = 0.05,
    *,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    propensity: Any = None,
    baseline: Any = None,
    proxy: Any = "cross_fit",
    vce: str = "HC1",
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sorted group average treatment effects (GATES) and heterogeneity tests.

    **With** ``y``, ``treat`` and ``covariates`` this is the
    Chernozhukov-Demirer-Duflo-Fernandez-Val GATES regression
    (``GenericML::GATES`` with ``monotonize = FALSE``): weighted least
    squares with weights ``1 / (p(1 - p))`` of ``Y`` on an intercept, the
    optional baseline proxy ``B(Z)`` and ``(D - p) 1{G_k}`` for groups
    ``G_1..G_K``. Groups are quantile groups of the CATE proxy when
    ``by='cate'`` (``GenericML::quantile_group``: type-7 cut points,
    left-closed) and of the column ``by`` otherwise (its values are used
    directly when it has at most ``n_groups`` levels). Each ``gamma_k``
    is the ATE in group ``k``, estimated from outcomes;
    ``top_vs_bottom`` is ``gamma_K - gamma_1`` with the covariance term;
    the omnibus test is the Wald test that all ``gamma_k`` are equal
    (chi-square, ``K - 1`` df). The proxy defaults to an out-of-fold refit
    of the result's learner, as in :func:`blp_test`.

    **Without** them the function falls back to the pre-1.30 descriptive
    summary: group means of the predicted CATEs with ``sd / sqrt(n)``
    SEs. Those p-values treat the model's own predictions as data -- with
    ``by='cate'`` the groups are formed from the same predictions -- so
    they are not a test of heterogeneity; a warning says so.

    .. versionchanged:: 1.30.0
       Added the GATES regression (``y`` / ``treat`` / ``covariates``).

    Parameters
    ----------
    result : CausalResult, fitted CATE model, or array
    data : pd.DataFrame
    by : str
        ``'cate'`` or a column name.
    n_groups : int, default 4
    alpha : float, default 0.05
    y, treat, covariates : optional
        Outcome, binary treatment and covariates; required for GATES.
    propensity, baseline, proxy, vce, n_folds, seed
        As in :func:`blp_test`.

    Returns
    -------
    dict
        ``gate_table`` (group, n, gate, se, ci_lower, ci_upper for GATES;
        group, n, mean_cate, se, ci_lower, ci_upper for the descriptive
        path), ``omnibus_stat`` / ``omnibus_pvalue`` (``omnibus_F`` on the
        descriptive path), ``top_vs_bottom_diff`` / ``_se`` / ``_pvalue``,
        and ``method``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 800
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2,
    ...                    "p": np.full(n, 0.5)})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> gt = sp.gate_test(res, df, by="cate", y="y", treat="d",
    ...                   covariates=["x1", "x2"], propensity="p")
    >>> gt["method"]
    'gates'
    >>> bool(gt["top_vs_bottom_pvalue"] < 0.05)
    True
    """
    if y is None or treat is None or covariates is None:
        return _gate_descriptive(result, data, by, n_groups, alpha)

    from scipy import stats as _st

    from . import _cddf

    Y, D, p, B, S, p_source, s_source = _cddf_inputs(
        result,
        data,
        y,
        treat,
        list(covariates),
        n_folds,
        propensity,
        baseline,
        proxy,
        seed,
        "gate_test()",
    )
    n = len(Y)
    if by == "cate":
        labels = _cddf.quantile_groups(S, n_groups)
        levels = list(range(n_groups))
        group_name = "CATE group"
    else:
        if by not in data.columns:
            raise MethodIncompatibility(
                f"Column '{by}' not found in data",
                recovery_hint="Pass by='cate' or an existing column name.",
            )
        col = data[by].to_numpy()[:n]
        if pd.api.types.is_numeric_dtype(col) and len(np.unique(col)) > n_groups:
            labels = _cddf.quantile_groups(col.astype(float), n_groups)
            levels = list(range(n_groups))
        else:
            levels = sorted(np.unique(col).tolist())
            labels = np.searchsorted(np.asarray(levels), col)
        group_name = by
    K = len(levels)
    G = (labels[:, None] == np.arange(K)[None, :]).astype(float)
    cols = [np.ones(n)]
    if B is not None:
        cols.append(B)
    k0 = len(cols)
    X = np.column_stack(cols + [(D - p)[:, None] * G])
    beta, V = _cddf.wls_fit(Y, X, 1.0 / (p * (1.0 - p)), vce)
    g = beta[k0:]
    Vg = V[k0:, k0:]
    se = np.sqrt(np.diag(Vg))
    z = float(_st.norm.ppf(1 - alpha / 2))
    table = pd.DataFrame(
        {
            "group": levels,
            "n": G.sum(axis=0).astype(int),
            "gate": g,
            "se": se,
            "ci_lower": g - z * se,
            "ci_upper": g + z * se,
        }
    )
    table.index.name = group_name
    diff = float(g[-1] - g[0])
    se_diff = float(np.sqrt(Vg[-1, -1] + Vg[0, 0] - 2 * Vg[-1, 0]))
    C = np.column_stack([np.ones(K - 1), -np.eye(K - 1)])  # gamma_1 - gamma_k
    cg = C @ g
    wald = float(cg @ np.linalg.solve(C @ Vg @ C.T, cg))
    return {
        "gate_table": table,
        "omnibus_stat": wald,
        "omnibus_df": K - 1,
        "omnibus_pvalue": float(_st.chi2.sf(wald, K - 1)),
        "top_vs_bottom_diff": diff,
        "top_vs_bottom_se": se_diff,
        "top_vs_bottom_pvalue": float(2 * _st.norm.sf(abs(diff / se_diff))),
        "method": "gates",
        "proxy_source": s_source,
        "propensity_source": p_source,
        "vcov_type": vce,
    }


def _gate_descriptive(
    result: Any, data: pd.DataFrame, by: str, n_groups: int, alpha: float
) -> Dict[str, Any]:
    """Pre-1.30 behaviour: summaries of the predicted CATEs by group."""
    import warnings

    warnings.warn(
        "gate_test() without y / treat / covariates summarises the model's "
        "own CATE predictions; its p-values treat those predictions as data "
        "and are not a test of treatment-effect heterogeneity. Pass y=, "
        "treat= and covariates= for the GATES regression.",
        UserWarning,
        stacklevel=3,
    )
    cate = _extract_cate(result)
    gate_df = cate_by_group(result, data, by=by, n_groups=n_groups, alpha=alpha)

    if by == "cate":
        labels = pd.qcut(cate, q=n_groups, labels=False, duplicates="drop")
    else:
        col = data[by].values[: len(cate)]
        if pd.api.types.is_numeric_dtype(col) and len(np.unique(col)) > n_groups:
            labels = pd.qcut(col, q=n_groups, labels=False, duplicates="drop")
        else:
            labels = col

    groups_list = [cate[labels == g] for g in sorted(np.unique(labels))]
    if len(groups_list) >= 2:
        f_stat, f_pvalue = stats.f_oneway(*groups_list)
    else:
        f_stat, f_pvalue = np.nan, np.nan

    sorted_gate = gate_df.sort_values("mean_cate")
    bottom = sorted_gate.iloc[0]
    top = sorted_gate.iloc[-1]
    diff = top["mean_cate"] - bottom["mean_cate"]
    se_diff = np.sqrt(top["se"] ** 2 + bottom["se"] ** 2)
    if se_diff > 0:
        tvb_pvalue = float(2 * stats.norm.sf(abs(diff / se_diff)))
    else:
        tvb_pvalue = np.nan

    return {
        "gate_table": gate_df,
        "omnibus_F": float(f_stat),
        "omnibus_pvalue": float(f_pvalue),
        "top_vs_bottom_diff": float(diff),
        "top_vs_bottom_se": float(se_diff),
        "top_vs_bottom_pvalue": float(tvb_pvalue),
        "method": "descriptive_predictions",
    }


def compare_metalearners(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learners: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Fit multiple meta-learners and compare their ATE estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate / effect modifier variables.
    learners : list of str, optional
        Which learners to compare. Default: all five ('s','t','x','r','dr').
    **kwargs
        Additional arguments passed to ``metalearner()``.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: learner, ate, se, ci_lower,
        ci_upper, pvalue, cate_std, cate_iqr.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> age = rng.normal(size=n)
    >>> edu = rng.normal(size=n)
    >>> training = rng.integers(0, 2, size=n)
    >>> wage = 0.5 * age + 0.3 * edu + (1.0 + age) * training + rng.normal(size=n)
    >>> df = pd.DataFrame({"wage": wage, "training": training,
    ...                    "age": age, "edu": edu})
    >>> comp = sp.compare_metalearners(df, y='wage', treat='training',
    ...                                 covariates=['age', 'edu'])
    >>> "ate" in comp.columns
    True
    >>> int(comp.shape[0])
    5
    """
    from .metalearners import metalearner as _metalearner

    if learners is None:
        learners = ["s", "t", "x", "r", "dr"]

    learner_names = {
        "s": "S-Learner",
        "t": "T-Learner",
        "x": "X-Learner",
        "r": "R-Learner",
        "dr": "DR-Learner",
    }

    rows = []
    for lr in learners:
        result = _metalearner(
            data,
            y=y,
            treat=treat,
            covariates=covariates,
            learner=lr,
            **kwargs,
        )
        cate = result.model_info["cate"]
        rows.append(
            {
                "learner": learner_names.get(lr, lr),
                "ate": result.estimate,
                "se": result.se,
                "ci_lower": result.ci[0],
                "ci_upper": result.ci[1],
                "pvalue": result.pvalue,
                # Default updated v1.11.4 — every learner now uses the AIPW
                # influence-function SE; the legacy 'bootstrap' fallback was
                # statistically invalid for non-DR learners.
                "se_method": result.model_info.get(
                    "se_method", "aipw_influence_function"
                ),
                "cate_std": float(np.std(cate)),
                "cate_iqr": float(np.percentile(cate, 75) - np.percentile(cate, 25)),
            }
        )

    return pd.DataFrame(rows)


def predict_cate(
    result: CausalResult,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """
    Predict CATE on new (out-of-sample) data.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()`` containing a fitted estimator.
    new_data : pd.DataFrame
        New data with the same covariate columns used in estimation.

    Returns
    -------
    np.ndarray
        Predicted CATE for each row of new_data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.metalearner(df, y="y", treat="d",
    ...                      covariates=["x1", "x2"], learner="t")
    >>> new_data = pd.DataFrame({"x1": [-1.0, 0.0, 1.0],
    ...                          "x2": [0.0, 0.0, 0.0]})
    >>> tau_hat = sp.predict_cate(res, new_data)
    >>> int(tau_hat.shape[0])
    3
    >>> bool(tau_hat[2] > tau_hat[0])
    True
    """
    model_info = getattr(result, "model_info", None)
    if not isinstance(model_info, dict) or "_estimator" not in model_info:
        raise ValueError(
            "Result does not contain a fitted estimator. predict_cate() "
            "expects a metalearner() result; got "
            f"{type(result).__name__}. Use sp.metalearner(...) (or another "
            "CATE estimator that stores '_estimator' in model_info)."
        )
    covariates = result.model_info.get("covariates")
    if covariates is None:
        raise ValueError("Result does not contain covariate names.")
    covariate_names = list(covariates)
    for c in covariate_names:
        if c not in new_data.columns:
            raise ValueError(f"Column '{c}' not found in new_data")

    X_new = new_data[covariate_names].values.astype(float)
    est = result.model_info["_estimator"]
    return np.asarray(est.effect(X_new), dtype=float)


# ======================================================================
# Internal
# ======================================================================


def _extract_cate(result: Any) -> np.ndarray:
    """Extract a CATE array from any CATE-bearing input.

    Accepts a :class:`CausalResult` with ``model_info['cate']`` (the
    metalearner / tarnet convention), a fitted model exposing
    ``.effect()`` (e.g. :class:`~statspai.forest.CausalForest`), or a
    raw array of per-unit effects.
    """
    if isinstance(result, (np.ndarray, list, tuple, pd.Series)):
        return np.asarray(result, dtype=float).ravel()
    if hasattr(result, "model_info") and "cate" in getattr(result, "model_info", {}):
        return np.asarray(result.model_info["cate"])
    # Fitted CATE model (CausalForest & friends): predict on the
    # training sample.  A CausalResult without stored effects also has
    # an .effect() method, but it raises — fall through to the shared
    # error below rather than leaking its internal exception.
    if hasattr(result, "effect") and callable(result.effect):
        from ..exceptions import MethodIncompatibility

        try:
            return np.asarray(result.effect(), dtype=float).ravel()
        except TypeError:
            stored_x = getattr(result, "_X_original", None)
            if stored_x is not None:
                return np.asarray(result.effect(stored_x), dtype=float).ravel()
        except MethodIncompatibility:
            pass
    raise ValueError(
        "Input does not contain CATE estimates. Pass a metalearner()/"
        "tarnet() result, a fitted causal_forest() model, or a raw "
        "array of per-unit effects."
    )
