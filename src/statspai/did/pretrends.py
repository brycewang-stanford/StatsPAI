"""
Pre-trends testing and sensitivity analysis for Difference-in-Differences.

Implements three current-methodology routines:

1. **pretrends_power** -- Roth (2022) power analysis for pre-trend tests.
   A non-significant pre-trend test is uninformative when power is low.

2. **sensitivity_rr** -- Rambachan & Roth (2023) honest confidence intervals
   for the ATT under bounded violations of parallel trends (C-LF method).

3. **pretrends_test** -- Joint Wald / F test of pre-treatment coefficients.

References
----------
- Roth, J. (2022). Pretest with Caution: Event-Study Estimates after
  Testing for Parallel Trends. *AER: Insights*, 4(3), 305--322.
- Rambachan, A. & Roth, J. (2023). A More Credible Approach to Parallel
  Trends. *Review of Economic Studies*, 90(5), 2555--2591. [@roth2022pretest]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


class _UnsupportedSensitivityMethod(MethodIncompatibility, NotImplementedError):
    """Compatibility bridge for unsupported sensitivity-analysis methods."""


def _require_string_option(value: Any, name: str, context: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a string option.",
            diagnostics={"context": context, name: repr(value)},
        )
    out = value.strip()
    if not out:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a non-empty string option.",
            diagnostics={"context": context, name: repr(value)},
        )
    return out


def _require_open_unit_float(value: Any, name: str, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a number in (0, 1).",
            diagnostics={"context": context, name: repr(value)},
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a number in (0, 1).",
            diagnostics={"context": context, name: repr(value)},
        ) from exc
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be in (0, 1).",
            diagnostics={"context": context, name: out},
        )
    return out


def _require_int_at_least(
    value: Any,
    name: str,
    context: str,
    minimum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be an integer >= {minimum}.",
            diagnostics={"context": context, name: repr(value), "minimum": minimum},
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be an integer >= {minimum}.",
            diagnostics={"context": context, name: repr(value), "minimum": minimum},
        ) from exc
    if out < minimum:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be >= {minimum}.",
            diagnostics={"context": context, name: out, "minimum": minimum},
        )
    return out


def _finite_vector(value: Any, name: str, context: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be numeric.",
            diagnostics={"context": context, name: repr(value)},
        ) from exc
    if out.ndim != 1 or out.size == 0:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a non-empty one-dimensional array.",
            diagnostics={"context": context, name: repr(value)},
        )
    if not np.all(np.isfinite(out)):
        raise MethodIncompatibility(
            f"{context}: `{name}` must contain only finite values.",
            diagnostics={"context": context, name: out.tolist()},
        )
    return out


def _require_nonnegative(values: np.ndarray, name: str, context: str) -> None:
    if np.any(values < 0):
        raise MethodIncompatibility(
            f"{context}: `{name}` must contain non-negative values.",
            diagnostics={"context": context, name: values.tolist()},
        )


def _extract_event_study(result: Any) -> pd.DataFrame:
    """Pull the event-study DataFrame from a CausalResult.

    Looks in ``result.model_info['event_study']`` first, then falls back
    to ``result.detail``.  Raises ``ValueError`` with a helpful message
    when no event-study data can be found.
    """
    es = None
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        es = result.model_info.get("event_study", None)
    if es is None and hasattr(result, "detail"):
        es = result.detail
    if es is None or (isinstance(es, pd.DataFrame) and es.empty):
        raise DataInsufficient(
            "Cannot extract event-study estimates from the result object. "
            "Make sure you pass a CausalResult with an 'event_study' key in "
            "model_info or a non-empty 'detail' DataFrame.",
            diagnostics={"context": "pretrends", "has_result": result is not None},
        )
    if not isinstance(es, pd.DataFrame):
        raise MethodIncompatibility(
            "Expected a DataFrame for event-study estimates.",
            diagnostics={"context": "pretrends", "type": type(es).__name__},
        )
    return es


def _resolve_columns(df: pd.DataFrame) -> tuple:
    """Return (time_col, est_col, se_col) after inspecting column names."""
    if not isinstance(df, pd.DataFrame):
        raise MethodIncompatibility(
            "event-study estimates must be a pandas DataFrame.",
            diagnostics={"context": "pretrends", "type": type(df).__name__},
        )
    # Time column
    time_candidates = ["relative_time", "rel_time", "event_time", "t", "time", "period"]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise MethodIncompatibility(
            f"Cannot find a relative-time column. Looked for {time_candidates}; "
            f"columns are {list(df.columns)}.",
            diagnostics={
                "context": "pretrends",
                "missing": "relative_time",
                "columns": list(df.columns),
            },
        )

    # Estimate column
    est_candidates = ["estimate", "att", "coef", "coefficient", "beta", "effect"]
    est_col = None
    for c in est_candidates:
        if c in df.columns:
            est_col = c
            break
    if est_col is None:
        raise MethodIncompatibility(
            f"Cannot find an estimate column. Looked for {est_candidates}; "
            f"columns are {list(df.columns)}.",
            diagnostics={
                "context": "pretrends",
                "missing": "estimate",
                "columns": list(df.columns),
            },
        )

    # SE column
    se_candidates = ["se", "std_error", "std.error", "stderr", "std_err"]
    se_col = None
    for c in se_candidates:
        if c in df.columns:
            se_col = c
            break
    if se_col is None:
        raise MethodIncompatibility(
            f"Cannot find a standard-error column. Looked for {se_candidates}; "
            f"columns are {list(df.columns)}.",
            diagnostics={
                "context": "pretrends",
                "missing": "standard_error",
                "columns": list(df.columns),
            },
        )

    return time_col, est_col, se_col


def _split_pre_post(
    df: pd.DataFrame,
    time_col: str,
    est_col: str,
    se_col: str,
) -> tuple:
    """Split event-study into pre-period (t < 0) and post-period (t >= 1)."""
    for col in (time_col, est_col, se_col):
        _finite_vector(df[col], col, "pretrends")
    pre = df[df[time_col] < 0].sort_values(time_col).copy()
    post = df[df[time_col] >= 1].sort_values(time_col).copy()
    return pre, post


def _pre_arrays(
    pre: pd.DataFrame,
    est_col: str,
    se_col: str,
    context: str,
) -> tuple:
    if len(pre) == 0:
        raise DataInsufficient(
            "No pre-treatment periods found (relative_time < 0).",
            diagnostics={"context": context},
        )
    beta_pre_all = _finite_vector(pre[est_col], est_col, context)
    se_pre_all = _finite_vector(pre[se_col], se_col, context)
    _require_nonnegative(se_pre_all, se_col, context)
    estimated = se_pre_all > 0
    if not estimated.any():
        raise DataInsufficient(
            "All pre-treatment periods have zero standard error (only the "
            "reference period is present); the pre-trend test is undefined.",
            diagnostics={"context": context, "n_pre": int(len(pre))},
        )
    return beta_pre_all, se_pre_all, estimated


def _normal_rectangle_probability(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """``P(lower < X < upper)`` for ``X ~ N(mean, cov)``.

    This is the multivariate-normal rectangle probability that R's
    ``mvtnorm::pmvnorm`` computes, and it is what Roth's ``pretrends``
    package integrates to get the rejection probability of the
    coefficient-by-coefficient pre-test.

    SciPy grew a ``lower_limit`` argument for exactly this in 1.10; on
    older SciPy the same quantity is assembled from orthant probabilities
    by inclusion-exclusion over the 2^K corners of the rectangle. K is the
    number of pre-periods, so the fallback is cheap in practice.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mvn = sp_stats.multivariate_normal
    try:
        prob = float(
            mvn.cdf(upper, mean=mean, cov=cov, lower_limit=lower, allow_singular=True)
        )
    except TypeError:  # pragma: no cover - SciPy < 1.10
        import itertools

        k = len(mean)
        prob = 0.0
        for mask in itertools.product((0, 1), repeat=k):
            corner = np.where(np.asarray(mask, dtype=bool), lower, upper)
            sign = (-1.0) ** sum(mask)
            prob += sign * float(
                mvn.cdf(corner, mean=mean, cov=cov, allow_singular=True)
            )
    return float(min(max(prob, 0.0), 1.0))


def _pre_vcv(
    result: Any,
    se_pre: np.ndarray,
    estimated: np.ndarray,
    K_all: int,
    K: int,
    context: str,
) -> np.ndarray:
    vcv = None
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        vcv = result.model_info.get("vcv_pre", None)
    if vcv is None:
        warnings.warn(
            f"{context}: no pre-period covariance matrix found in "
            "`result.model_info['vcv_pre']`, so the pre-treatment event-study "
            "coefficients are being treated as MUTUALLY INDEPENDENT "
            "(diagonal covariance). They are not independent in general: they "
            "share an omitted reference period and the same unit/time fixed "
            "effects, so the off-diagonal covariances are typically large and "
            "negative. Wald pre-trend statistics, Roth (2022) power, and "
            "Rambachan-Roth breakdown Mbar computed from this fallback can be "
            "materially misstated. Supply the full pre-period covariance via "
            "`result.model_info['vcv_pre']` — sp.event_study computes it and "
            "will write it when called with `expose_pre_vcov=True` (opt-in "
            "during the current release; it becomes the default afterwards).",
            UserWarning,
            stacklevel=3,
        )
        return np.diag(se_pre**2)
    try:
        out = np.asarray(vcv, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: `vcv_pre` must be numeric.",
            diagnostics={"context": context, "vcv_pre": repr(vcv)},
        ) from exc
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise MethodIncompatibility(
            f"{context}: `vcv_pre` must be a square matrix.",
            diagnostics={"context": context, "shape": out.shape},
        )
    if out.shape[0] == K_all:
        out = out[np.ix_(estimated, estimated)]
    elif out.shape[0] != K:
        raise MethodIncompatibility(
            f"{context}: `vcv_pre` has incompatible shape.",
            diagnostics={
                "context": context,
                "shape": out.shape,
                "expected": [(K, K), (K_all, K_all)],
            },
        )
    if not np.all(np.isfinite(out)):
        raise MethodIncompatibility(
            f"{context}: `vcv_pre` must contain only finite values.",
            diagnostics={"context": context},
        )
    return out


def _invert_vcv(vcv: np.ndarray, context: str, target: str) -> np.ndarray:
    try:
        return np.linalg.inv(vcv)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "Pre-period variance-covariance matrix is singular even after "
            "dropping the reference period: the pre-treatment coefficients are "
            f"collinear, so the {target} is undefined. Supply a full-rank "
            "'vcv_pre' via model_info or narrow the event window.",
            diagnostics={"context": context, "shape": vcv.shape},
        ) from exc


# ────────────────────────────────────────────────────────────────────
# 1. pretrends_test — Joint test of H0: all pre-treatment coefs = 0
# ────────────────────────────────────────────────────────────────────


def pretrends_test(
    result: Any,
    type: str = "wald",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Joint test of pre-treatment coefficients.

    Tests H0: beta_pre = 0 (all pre-treatment event-study coefficients
    are jointly zero).

    Parameters
    ----------
    result : CausalResult
        Event-study result containing pre-treatment estimates and SEs.
    type : ``'wald'`` or ``'f'``
        ``'wald'``: chi-squared test statistic.
        ``'f'``: scaled F-statistic (requires ``df_resid`` in model_info).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: ``statistic``, ``pvalue``, ``df``, ``type``,
        ``reject``, ``interpretation``.

    References
    ----------
    Standard Wald test; see Roth (2022) for caveats on interpretation.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     cohort = 4 if i < 40 else 0          # 0 = never treated
    ...     for t in range(8):
    ...         post = cohort > 0 and t >= cohort
    ...         y = 0.3 * t + (2.0 if post else 0.0) + (i % 5) + rng.normal()
    ...         rows.append((i, t, cohort, y))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])
    >>> es = sp.event_study(df, y="y", treat_time="cohort", time="t", unit="id")
    >>> sp.pretrends_test(es)
    """
    context = "pretrends_test"
    test_type = _require_string_option(type, "type", context).lower()
    alpha = _require_open_unit_float(alpha, "alpha", context)
    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, _ = _split_pre_post(es, time_col, est_col, se_col)

    beta_pre_all, se_pre_all, estimated = _pre_arrays(
        pre,
        est_col,
        se_col,
        context,
    )
    K_all = len(beta_pre_all)
    beta_pre = beta_pre_all[estimated]
    se_pre = se_pre_all[estimated]
    K = len(beta_pre)

    # Build variance-covariance matrix (diagonal if full VCV unavailable)
    vcv = _pre_vcv(result, se_pre, estimated, K_all, K, context)
    vcv_inv = _invert_vcv(vcv, context, "pre-trend test")
    wald_stat = float(beta_pre @ vcv_inv @ beta_pre)

    if test_type == "wald":
        pvalue = float(1.0 - sp_stats.chi2.cdf(wald_stat, df=K))
        stat_label = f"Wald chi2({K})"
        out_type = "wald"
    elif test_type == "f":
        df_resid = None
        if hasattr(result, "model_info") and isinstance(result.model_info, dict):
            df_resid = result.model_info.get("df_resid", None)
        if hasattr(result, "n_obs") and df_resid is None:
            df_resid = max(result.n_obs - K, K + 1)
        if df_resid is None:
            df_resid = 1000  # conservative fallback
        f_stat = wald_stat / K
        pvalue = float(1.0 - sp_stats.f.cdf(f_stat, dfn=K, dfd=df_resid))
        wald_stat = f_stat
        stat_label = f"F({K}, {df_resid})"
        out_type = "f"
    else:
        raise MethodIncompatibility(
            f"type must be 'wald' or 'f', got '{type}'.",
            diagnostics={
                "context": context,
                "type": type,
                "valid_types": ["wald", "f"],
            },
        )

    reject = pvalue < alpha
    if reject:
        interpretation = (
            f"Reject H0 at alpha={alpha}: evidence against parallel pre-trends."
        )
    else:
        interpretation = (
            f"Cannot reject parallel trends at alpha={alpha}. "
            "Note: non-rejection may reflect low power (see pretrends_power)."
        )

    return {
        "statistic": wald_stat,
        "pvalue": pvalue,
        "df": K,
        "type": out_type,
        "stat_label": stat_label,
        "reject": reject,
        "alpha": alpha,
        "interpretation": interpretation,
    }


# ────────────────────────────────────────────────────────────────────
# 2. pretrends_power — Roth (2022) power of the pre-test
# ────────────────────────────────────────────────────────────────────


def pretrends_power(
    result: Any,
    delta: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    test: str = "individual",
) -> Dict[str, Any]:
    """Power of the pre-trend test against a hypothesised violation.

    Implements the power calculation from Roth (2022, AER: Insights).
    A non-significant pre-trend test is uninformative when the test has
    low power against economically meaningful violations of parallel
    trends.

    Parameters
    ----------
    result : CausalResult
        Event-study result with pre-treatment estimates and SEs.
    delta : array-like, optional
        Hypothesised trend violation in the pre-period (length = number
        of pre-periods).  Default: linear trend
        ``delta[k] = (k+1) * min(|SE|)`` -- a violation equal to one SE
        at the furthest lag, declining linearly to near-zero.
    alpha : float, default 0.05
        Significance level of the pre-trend test.
    test : {"individual", "joint"}, default "individual"
        Which pre-test the power refers to.

        ``"individual"`` is the practice Roth (2022) analyses and the one
        his ``pretrends`` R package implements: the analyst eyeballs the
        event-study plot and calls the pre-trends into question if *any*
        pre-period coefficient is individually significant. Power is then
        one minus the probability that every pre-period coefficient falls
        inside its own ``+/- z_{1-alpha/2} * SE`` band, integrated over
        the joint normal with mean ``delta`` -- a multivariate-normal
        rectangle probability.

        ``"joint"`` is the power of the joint Wald test that all
        pre-period coefficients are zero, ``chi2(K)`` with
        non-centrality ``delta' Sigma^-1 delta``. Reported by
        :func:`pretrends_test`, and a strictly different quantity --
        not a tighter or looser version of the same one. The two are not
        even comparable at face value: the joint test has size exactly
        ``alpha``, while the coefficient-by-coefficient test rejects with
        probability above ``alpha`` under the null because each of the K
        coefficients gets its own ``alpha``-level look. Which comes out
        more powerful against a given trend depends on the design.
        ``power_joint`` is always reported alongside, so both are
        available from one call.

        .. versionchanged:: 1.21.0
           The default moved from ``"joint"`` to ``"individual"`` so the
           number matches Roth's ``pretrends`` package. This changes the
           returned ``power`` for existing calls -- see MIGRATION.md.
           Pass ``test="joint"`` to recover the previous behaviour.

    Returns
    -------
    dict
        Always: ``power``, ``power_under_null``, ``bayes_factor``,
        ``df``, ``delta``, ``alpha``, ``test``, ``warning``.
        ``likelihood_ratio`` is present when the pre-period point
        estimates are available. ``test="joint"`` additionally reports
        ``noncentrality`` and ``critical_value``; ``test="individual"``
        reports ``threshold_tstat``.

    Notes
    -----
    ``bayes_factor`` and ``likelihood_ratio`` follow the ``pretrends``
    package: the Bayes factor is ``(1 - power) / (1 - power_under_null)``,
    the odds that a passed pre-test moves in favour of the hypothesised
    trend, and the likelihood ratio compares the observed pre-period
    coefficients under ``delta`` versus under no violation.

    References
    ----------
    Roth, J. (2022). Pretest with Caution: Event-Study Estimates after
    Testing for Parallel Trends. *AER: Insights*, 4(3), 305--322. [@roth2022pretest]

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     cohort = 4 if i < 40 else 0          # 0 = never treated
    ...     for t in range(8):
    ...         post = cohort > 0 and t >= cohort
    ...         y = 0.3 * t + (2.0 if post else 0.0) + (i % 5) + rng.normal()
    ...         rows.append((i, t, cohort, y))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])
    >>> es = sp.event_study(df, y="y", treat_time="cohort", time="t", unit="id")
    >>> sp.pretrends_power(es)
    """
    context = "pretrends_power"
    alpha = _require_open_unit_float(alpha, "alpha", context)
    test = _require_string_option(test, "test", context).lower()
    if test not in {"individual", "joint"}:
        raise MethodIncompatibility(
            f"{context}: `test` must be 'individual' or 'joint'.",
            diagnostics={"context": context, "test": test},
        )
    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, _ = _split_pre_post(es, time_col, est_col, se_col)

    beta_pre_all, se_pre_all, estimated = _pre_arrays(pre, est_col, se_col, context)
    K_all = len(se_pre_all)
    pre = pre.loc[estimated]
    se_pre = se_pre_all[estimated]
    beta_pre = beta_pre_all[estimated]
    K = len(se_pre)

    # Build VCV (diagonal if full VCV unavailable)
    vcv = _pre_vcv(result, se_pre, estimated, K_all, K, context)
    vcv_inv = _invert_vcv(vcv, context, "pre-trend power")

    # Default delta: linear trend scaled by minimum SE
    if delta is None:
        min_se = np.min(np.abs(se_pre))
        # Pre-periods are sorted earliest to latest: t=-K, ..., t=-1
        # Linear trend: magnitude grows toward treatment
        delta = np.array([(i + 1) / K * min_se for i in range(K)])
    else:
        delta = _finite_vector(delta, "delta", context)
        if len(delta) == K_all:
            # One entry per pre-period including the reference: align it.
            delta = delta[estimated]
        elif len(delta) != K:
            raise MethodIncompatibility(
                f"delta has length {len(delta)} but there are {K} estimated "
                "pre-periods (the reference period is excluded).",
                diagnostics={
                    "context": context,
                    "delta_length": len(delta),
                    "expected_lengths": [K, K_all],
                },
            )

    # Joint Wald quantities are always reported: they are cheap, several
    # callers already read `noncentrality` / `critical_value`, and
    # pretrends_test() reports the realised statistic on the same scale.
    ncp = float(delta @ vcv_inv @ delta)
    crit = float(sp_stats.chi2.ppf(1.0 - alpha, df=K))
    joint_power = float(1.0 - sp_stats.ncx2.cdf(crit, df=K, nc=ncp))
    thresh = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))

    if test == "joint":
        power = joint_power
        power_null = alpha
    else:
        # Coefficient-by-coefficient pre-test, as in Roth's `pretrends`
        # package: reject if ANY pre-period t-stat exceeds the threshold.
        ub = np.sqrt(np.diag(vcv)) * thresh
        power = 1.0 - _normal_rectangle_probability(delta, vcv, -ub, ub)
        power_null = 1.0 - _normal_rectangle_probability(np.zeros(K), vcv, -ub, ub)
    extra = {
        "noncentrality": ncp,
        "critical_value": crit,
        "threshold_tstat": thresh,
        "power_joint": joint_power,
    }

    # Bayes factor: how a passed pre-test shifts the odds toward `delta`.
    denom = 1.0 - power_null
    bayes_factor = float((1.0 - power) / denom) if denom > 0 else float("nan")

    # Likelihood ratio of the observed pre-period coefficients under the
    # hypothesised trend versus under no violation.
    try:
        mvn = sp_stats.multivariate_normal
        lik_delta = float(mvn.pdf(beta_pre, mean=delta, cov=vcv, allow_singular=True))
        lik_zero = float(
            mvn.pdf(beta_pre, mean=np.zeros(K), cov=vcv, allow_singular=True)
        )
        likelihood_ratio = lik_delta / lik_zero if lik_zero > 0 else float("nan")
    except (ValueError, np.linalg.LinAlgError):
        likelihood_ratio = float("nan")

    warning = None
    if power < 0.50:
        warning = (
            f"LOW POWER ({power:.2f}): the pre-trend test has less than 50% "
            "power against the hypothesised violation. A non-significant "
            "pre-trend test is therefore uninformative. Consider the "
            "sensitivity analysis in sensitivity_rr()."
        )
    elif power < 0.80:
        warning = (
            f"MODERATE POWER ({power:.2f}): power is below the conventional "
            "80% threshold. Interpret a non-significant pre-trend test "
            "with caution."
        )

    out = {
        "power": power,
        "power_under_null": float(power_null),
        "bayes_factor": bayes_factor,
        "likelihood_ratio": likelihood_ratio,
        "test": test,
        "df": K,
        "delta": delta,
        "alpha": alpha,
        "warning": warning,
    }
    out.update(extra)
    return out


def pretrends_slope_for_power(
    result: Any,
    target_power: float = 0.5,
    alpha: float = 0.05,
    test: str = "individual",
) -> Dict[str, Any]:
    """Slope of a linear pre-trend the pre-test would detect ``target_power``
    of the time.

    The mirror image of :func:`pretrends_power`: instead of asking how
    much power the pre-test has against a chosen violation, it asks how
    large a violation has to be before the pre-test is even a coin flip.
    Roth's ``pretrends`` package exposes the same quantity as
    ``slope_for_power``, and it is the number to quote when a reader asks
    what a passed pre-test actually rules out.

    The hypothesised violation is linear in event time,
    ``delta_t = slope * (t - t_ref)``, with the reference period taken to
    be ``t = -1``.

    Parameters
    ----------
    result : CausalResult
        Event-study result with pre-treatment estimates and SEs. As with
        :func:`pretrends_power`, supply the full pre-period covariance via
        ``model_info['vcv_pre']`` -- the diagonal fallback overstates the
        detectable slope.
    target_power : float, default 0.5
        Power the returned slope achieves. 0.5 is the ``pretrends``
        default: the trend a pre-test catches half the time.
    alpha : float, default 0.05
        Significance level of the pre-test.
    test : {"individual", "joint"}, default "individual"
        Which pre-test to solve against; see :func:`pretrends_power`.

    Returns
    -------
    dict
        Keys: ``slope``, ``target_power``, ``achieved_power``, ``delta``,
        ``times``, ``test``, ``alpha``.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     cohort = 4 if i < 40 else 0          # 0 = never treated
    ...     for t in range(8):
    ...         post = cohort > 0 and t >= cohort
    ...         y = 0.3 * t + (2.0 if post else 0.0) + (i % 5) + rng.normal()
    ...         rows.append((i, t, cohort, y))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])
    >>> es = sp.event_study(df, y="y", treat_time="cohort", time="t", unit="id")
    >>> out = sp.pretrends_slope_for_power(es)
    >>> round(out["slope"], 3)
    0.245
    >>> out["target_power"]
    0.5

    References
    ----------
    Roth, J. (2022). Pretest with Caution: Event-Study Estimates after
    Testing for Parallel Trends. *AER: Insights*, 4(3), 305--322. [@roth2022pretest]
    """
    context = "pretrends_slope_for_power"
    target_power = _require_open_unit_float(target_power, "target_power", context)
    alpha = _require_open_unit_float(alpha, "alpha", context)

    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, _ = _split_pre_post(es, time_col, est_col, se_col)
    _, se_pre_all, estimated = _pre_arrays(pre, est_col, se_col, context)
    times = np.asarray(pre.loc[estimated, time_col], dtype=float)

    if target_power <= alpha:
        raise MethodIncompatibility(
            f"{context}: `target_power` must exceed `alpha` -- the pre-test "
            "already rejects with probability alpha when there is no "
            "violation at all, so no slope achieves less than that.",
            diagnostics={
                "context": context,
                "target_power": target_power,
                "alpha": alpha,
            },
        )

    # delta(slope) is linear in slope and power is monotone in |slope|,
    # so bracket upwards from a unit-SE trend and bisect.
    unit = float(np.min(np.abs(se_pre_all[estimated])))
    span = np.abs(times - (-1.0))
    span[span == 0] = 1.0

    def _power_at(slope: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(
                pretrends_power(
                    result,
                    delta=slope * (times - (-1.0)),
                    alpha=alpha,
                    test=test,
                )["power"]
            )

    lo, hi = 0.0, unit / float(np.max(span))
    for _ in range(60):
        if _power_at(hi) >= target_power:
            break
        lo, hi = hi, hi * 2.0
    else:  # pragma: no cover - power is monotone and unbounded in slope
        raise NumericalInstability(
            f"{context}: could not bracket a slope reaching the target power.",
            diagnostics={"context": context, "target_power": target_power},
        )

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _power_at(mid) < target_power:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * max(1.0, hi):
            break
    slope = 0.5 * (lo + hi)

    return {
        "slope": slope,
        "target_power": target_power,
        "achieved_power": _power_at(slope),
        "delta": slope * (times - (-1.0)),
        "times": times,
        "test": test,
        "alpha": alpha,
    }


# ────────────────────────────────────────────────────────────────────
# 3. sensitivity_rr — Rambachan & Roth (2023) honest CIs
# ────────────────────────────────────────────────────────────────────


@dataclass
class SensitivityResult(ResultProtocolMixin):
    """Result of Rambachan & Roth (2023) sensitivity analysis.

    Attributes
    ----------
    mbar_grid : np.ndarray
        Grid of M-bar values tested.
    ci_lower : np.ndarray
        Lower bound of the honest CI at each M-bar.
    ci_upper : np.ndarray
        Upper bound of the honest CI at each M-bar.
    breakdown_mbar : float
        Smallest M-bar for which the CI includes zero (sign reversal).
    att : float
        Point estimate of the ATT.
    att_se : float
        Standard error of the ATT.
    method : str
        Extrapolation method used (``'C-LF'``).
    alpha : float
        Significance level.

    Methods
    -------
    summary()
        Print a formatted summary table.
    plot()
        Matplotlib sensitivity plot (M-bar vs CI).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(60):
    ...     g = 5 if i < 30 else 0
    ...     for t in range(1, 9):
    ...         post = 1 if (g and t >= 5) else 0
    ...         y = 1.0 + 0.2 * t + i / 120 + 2.0 * post + rng.normal(0, 0.5)
    ...         rows.append({"unit": i, "time": t, "y": y, "g": g})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.event_study(df, y="y", treat_time="g", time="time",
    ...                         unit="unit", window=(-3, 3))
    >>> sens = sp.sensitivity_rr(result, Mbar=[0.0, 0.5, 1.0])
    >>> type(sens).__name__
    'SensitivityResult'
    >>> bool(isinstance(sens.summary(), str))
    True
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax = sens.plot(ax=ax)
    >>> fig.savefig("sensitivity.png")  # doctest: +SKIP

    References
    ----------
    Rambachan, A. & Roth, J. (2023). [@rambachan2023more]
    """

    mbar_grid: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    breakdown_mbar: float
    att: float
    att_se: float
    method: str = "C-LF"
    alpha: float = 0.05

    # ── Pretty printing ──────────────────────────────────────────── #

    def summary(self) -> str:
        """Return a formatted summary string."""
        z = sp_stats.norm.ppf(1.0 - self.alpha / 2)
        lines = []
        hbar = "\u2501" * 58
        lines.append(hbar)
        lines.append("  Rambachan & Roth (2023) Sensitivity Analysis")
        lines.append(f"  Method: {self.method}  |  Alpha: {self.alpha}")
        lines.append(hbar)
        lines.append(f"  ATT = {self.att:.4f}  (SE = {self.att_se:.4f})")
        lines.append(
            f"  Original CI: [{self.att - z * self.att_se:.4f}, "
            f"{self.att + z * self.att_se:.4f}]"
        )
        lines.append("")
        lines.append(
            f"  {'Mbar':>8s}  {'CI Lower':>12s}  "
            f"{'CI Upper':>12s}  {'Includes 0?':>12s}"
        )
        lines.append(
            f"  {'----':>8s}  {'--------':>12s}  "
            f"{'--------':>12s}  {'-----------':>12s}"
        )
        for i, m in enumerate(self.mbar_grid):
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            inc = "Yes" if lo <= 0 <= hi else "No"
            lines.append(f"  {m:8.3f}  {lo:12.4f}  {hi:12.4f}  {inc:>12s}")
        lines.append("")
        if np.isfinite(self.breakdown_mbar):
            lines.append(f"  Breakdown Mbar = {self.breakdown_mbar:.4f}")
            lines.append("  (smallest Mbar where CI includes zero)")
        else:
            lines.append("  No breakdown: CI excludes zero for all Mbar in grid.")
        lines.append(hbar)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows = ""
        for i, m in enumerate(self.mbar_grid):
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            inc = lo <= 0 <= hi
            bg = ' style="background:#fff3cd"' if inc else ""
            inc_str = "Yes" if inc else "No"
            rows += (
                f"<tr{bg}><td>{m:.3f}</td>"
                f"<td>{lo:.4f}</td><td>{hi:.4f}</td>"
                f"<td>{inc_str}</td></tr>\n"
            )
        bd = (
            f"<b>{self.breakdown_mbar:.4f}</b>"
            if np.isfinite(self.breakdown_mbar)
            else "None (robust for all tested Mbar)"
        )
        return f"""
        <div style="font-family:monospace; max-width:600px">
        <h3>Rambachan &amp; Roth (2023) Sensitivity Analysis</h3>
        <p>Method: {self.method} | ATT = {self.att:.4f}
           (SE = {self.att_se:.4f}) | Alpha = {self.alpha}</p>
        <table border="1" cellpadding="4" style="border-collapse:collapse">
        <tr><th>Mbar</th><th>CI Lower</th><th>CI Upper</th>
            <th>Includes 0?</th></tr>
        {rows}
        </table>
        <p>Breakdown Mbar: {bd}</p>
        </div>
        """

    # ── Plot ─────────────────────────────────────────────────────── #

    def plot(
        self,
        ax: Any = None,
        figsize: tuple[float, float] = (8, 5),
        **kwargs: Any,
    ) -> Any:
        """Sensitivity plot: M-bar on x-axis, honest CI band on y-axis.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        figsize : tuple, default (8, 5)
        **kwargs : passed to ``ax.fill_between``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        fill_kw = dict(alpha=0.3, color="steelblue", label="Honest CI")
        fill_kw.update(kwargs)
        ax.fill_between(self.mbar_grid, self.ci_lower, self.ci_upper, **fill_kw)
        ax.plot(self.mbar_grid, self.ci_lower, color="steelblue", linewidth=0.8)
        ax.plot(self.mbar_grid, self.ci_upper, color="steelblue", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(
            self.att,
            color="crimson",
            linestyle="-",
            linewidth=1.0,
            label=f"ATT = {self.att:.4f}",
        )

        if np.isfinite(self.breakdown_mbar):
            ax.axvline(
                self.breakdown_mbar,
                color="orange",
                linestyle=":",
                linewidth=1.2,
                label=f"Breakdown Mbar = {self.breakdown_mbar:.3f}",
            )

        ax.set_xlabel(r"$\bar{M}$ (Max. violation of parallel trends)")
        ax.set_ylabel("Treatment effect")
        ax.set_title("Rambachan & Roth (2023) Sensitivity Analysis")
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        return ax


def sensitivity_rr(
    result: Any,
    Mbar: Optional[Union[np.ndarray, List[float]]] = None,
    method: str = "C-LF",
    alpha: float = 0.05,
    n_grid: int = 20,
) -> SensitivityResult:
    """Rambachan & Roth (2023) honest confidence intervals.

    Computes confidence intervals for the ATT that are valid under
    bounded departures from parallel trends.  The *conditional
    linear-in-relative-time* (C-LF) restriction assumes the
    post-treatment violation is bounded by a linear extrapolation of the
    pre-trend plus an additional M-bar of slack.

    Parameters
    ----------
    result : CausalResult
        Event-study result with pre- and post-treatment estimates.
    Mbar : array-like, optional
        Grid of M-bar values.  Default:
        ``np.linspace(0, 3 * max_pre_slope, n_grid)``.
    method : ``'C-LF'``
        Extrapolation method.  Currently only C-LF is implemented.
    alpha : float, default 0.05
        Significance level.
    n_grid : int, default 20
        Number of grid points when ``Mbar`` is not supplied.

    Returns
    -------
    SensitivityResult

    Notes
    -----
    .. versionchanged:: next
       The pre-period trend is now fitted by **generalised** least squares
       using the full pre-period covariance from
       ``result.model_info['vcv_pre']`` when it is available
       (``sp.event_study`` supplies it). Previously the fit always used
       diagonal ``1/se**2`` weights, i.e. it assumed the pre-treatment
       event-study coefficients were mutually independent -- they are not,
       since they share the omitted reference period and the unit/time
       fixed effects. Breakdown ``Mbar`` values therefore move slightly
       relative to earlier releases. When no covariance is available the
       diagonal fallback is still used, but it now warns loudly.
        Object with ``.summary()``, ``.plot()``, ``.mbar_grid``,
        ``.ci_lower``, ``.ci_upper``, ``.breakdown_mbar``.

    References
    ----------
    Rambachan, A. & Roth, J. (2023). A More Credible Approach to
    Parallel Trends. *Review of Economic Studies*, 90(5), 2555--2591.
    [@rambachan2023more]

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     cohort = 4 if i < 40 else 0          # 0 = never treated
    ...     for t in range(8):
    ...         post = cohort > 0 and t >= cohort
    ...         y = 0.3 * t + (2.0 if post else 0.0) + (i % 5) + rng.normal()
    ...         rows.append((i, t, cohort, y))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])
    >>> es = sp.event_study(df, y="y", treat_time="cohort", time="t", unit="id")
    >>> sens = sp.sensitivity_rr(es, Mbar=[0, 0.01, 0.02, 0.05])
    >>> sens.summary()
    """
    context = "sensitivity_rr"
    method = _require_string_option(method, "method", context).upper()
    alpha = _require_open_unit_float(alpha, "alpha", context)
    n_grid = _require_int_at_least(n_grid, "n_grid", context, 1)
    if method != "C-LF":
        raise _UnsupportedSensitivityMethod(
            f"Only method='C-LF' is currently implemented, got '{method}'.",
            diagnostics={
                "context": context,
                "method": method,
                "valid_methods": ["C-LF"],
            },
        )

    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, post = _split_pre_post(es, time_col, est_col, se_col)

    if len(pre) == 0:
        raise DataInsufficient(
            "No pre-treatment periods found (relative_time < 0).",
            diagnostics={"context": context},
        )
    if len(post) == 0:
        raise DataInsufficient(
            "No post-treatment periods found (relative_time >= 1).",
            diagnostics={"context": context},
        )

    # ── Extract ATT ──────────────────────────────────────────────── #
    att = (
        float(result.estimate)
        if hasattr(result, "estimate")
        else float(post[est_col].iloc[0])
    )
    att_se = float(result.se) if hasattr(result, "se") else float(post[se_col].iloc[0])
    if not np.isfinite(att) or not np.isfinite(att_se) or att_se < 0:
        raise MethodIncompatibility(
            "sensitivity_rr: ATT and standard error must be finite, with "
            "non-negative SE.",
            diagnostics={"context": context, "att": att, "att_se": att_se},
        )

    # ── Fit linear trend through pre-period ──────────────────────── #
    pre_t = _finite_vector(pre[time_col], time_col, context)
    pre_est = _finite_vector(pre[est_col], est_col, context)

    if len(pre_t) >= 2:
        # Generalised least squares through the pre-period estimates.
        #
        # The pre-treatment event-study coefficients are NOT independent: they
        # share an omitted reference period and the same unit/time fixed
        # effects.  When the full pre-period covariance is available we use
        # its inverse as the GLS weight matrix; otherwise we fall back to the
        # diagonal 1/se^2 weights (and `_pre_vcv` warns loudly about it).
        # The reference period (se == 0, coefficient pinned to 0 by
        # construction) keeps its near-infinite diagonal weight so the fitted
        # line still passes through it, exactly as before.
        pre_se = _finite_vector(pre[se_col], se_col, context)
        _require_nonnegative(pre_se, se_col, context)
        weights = 1.0 / (pre_se**2 + 1e-16)
        W = np.diag(weights)
        estimated_s = pre_se > 0
        if estimated_s.any():
            vcv_s = _pre_vcv(
                result,
                pre_se[estimated_s],
                estimated_s,
                len(pre_se),
                int(estimated_s.sum()),
                context,
            )
            try:
                W_est = np.linalg.inv(vcv_s)
            except np.linalg.LinAlgError:
                W_est = None
            if W_est is not None:
                idx = np.where(estimated_s)[0]
                W[np.ix_(idx, idx)] = W_est
        # GLS: y = a + b*t
        X = np.column_stack([np.ones(len(pre_t)), pre_t])
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ pre_est
        try:
            coefs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                "sensitivity_rr: pre-period linear trend is not identified.",
                diagnostics={"context": context, "n_pre": int(len(pre_t))},
            ) from exc
        slope = coefs[1]
    else:
        # Single pre-period: slope = estimate / |time|
        slope = pre_est[0] / max(abs(pre_t[0]), 1.0)

    # ── Extrapolate linear trend to post-period ──────────────────── #
    post_t = _finite_vector(post[time_col], time_col, context)
    # Baseline bias for the first post-period
    baseline_bias = abs(slope) * post_t[0]

    # Sensitivity factor: how much each unit of Mbar adds to the bias.
    # Under C-LF, the sensitivity factor for relative time h is h itself.
    sensitivity_factor = float(np.max(post_t))

    # ── Build Mbar grid ──────────────────────────────────────────── #
    max_pre_slope = max(abs(slope), 1e-6)
    if Mbar is None:
        mbar_grid = np.linspace(0.0, 3.0 * max_pre_slope, n_grid)
    else:
        mbar_grid = _finite_vector(Mbar, "Mbar", context)
        if np.any(mbar_grid < 0):
            raise MethodIncompatibility(
                "sensitivity_rr: `Mbar` values must be non-negative.",
                diagnostics={"context": context, "Mbar": mbar_grid.tolist()},
            )

    z = sp_stats.norm.ppf(1.0 - alpha / 2)

    ci_lower = np.empty(len(mbar_grid))
    ci_upper = np.empty(len(mbar_grid))

    for i, m in enumerate(mbar_grid):
        max_bias = baseline_bias + m * sensitivity_factor
        ci_lower[i] = att - max_bias - z * att_se
        ci_upper[i] = att + max_bias + z * att_se

    # ── Breakdown M-bar ──────────────────────────────────────────── #
    includes_zero = (ci_lower <= 0) & (ci_upper >= 0)
    breakdown_idx = np.where(includes_zero)[0]
    if len(breakdown_idx) > 0:
        breakdown_mbar = float(mbar_grid[breakdown_idx[0]])
    else:
        breakdown_mbar = float("inf")

    return SensitivityResult(
        mbar_grid=mbar_grid,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        breakdown_mbar=breakdown_mbar,
        att=att,
        att_se=att_se,
        method=method,
        alpha=alpha,
    )


# ────────────────────────────────────────────────────────────────────
# Convenience: formatted combined report
# ────────────────────────────────────────────────────────────────────


def pretrends_summary(
    result: Any,
    delta: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> str:
    """Print a combined pre-trends diagnostic report.

    Runs ``pretrends_test`` and ``pretrends_power`` and formats the
    output in a single table.

    Parameters
    ----------
    result : CausalResult
        Event-study result.
    delta : array-like, optional
        Passed to ``pretrends_power``.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    str
        Formatted report.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(60):
    ...     g = 5 if i < 30 else 0
    ...     for t in range(1, 9):
    ...         post = 1 if (g and t >= 5) else 0
    ...         y = 1.0 + 0.2 * t + i / 120 + 2.0 * post + rng.normal(0, 0.5)
    ...         rows.append({"unit": i, "time": t, "y": y, "g": g})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.event_study(df, y="y", treat_time="g", time="time",
    ...                         unit="unit", window=(-3, 3))
    >>> report = sp.pretrends_summary(result)  # also prints the report
    >>> bool(isinstance(report, str))
    True
    """
    test = pretrends_test(result, type="wald", alpha=alpha)
    pwr = pretrends_power(result, delta=delta, alpha=alpha)

    hbar = "\u2501" * 58
    lines = [
        hbar,
        "  Pre-Trends Analysis",
        hbar,
        "  Joint pre-trend test:",
        f"    {test['stat_label']} = {test['statistic']:.2f}, "
        f"p = {test['pvalue']:.3f}",
    ]
    if test["reject"]:
        lines.append("    \u2192 Evidence against parallel pre-trends")
    else:
        lines.append("    \u2192 Cannot reject parallel trends")

    lines.append("")
    lines.append("  Power against linear violation:")
    lines.append(
        f"    Power = {pwr['power']:.2f}",
    )
    if pwr["warning"] and pwr["power"] < 0.50:
        lines.append("    \u2190 LOW POWER WARNING")
    elif pwr["warning"] and pwr["power"] < 0.80:
        lines.append("    \u2190 Moderate power")
    lines.append(hbar)

    report = "\n".join(lines)
    print(report)
    return report


__all__ = [
    "pretrends_test",
    "pretrends_power",
    "sensitivity_rr",
    "SensitivityResult",
    "pretrends_summary",
]
