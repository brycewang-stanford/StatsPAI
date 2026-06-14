"""E-value: sensitivity analysis for unmeasured confounding.

The E-value is the minimum strength of association, on the risk-ratio
scale, that an unmeasured confounder would need to have with *both* the
treatment and the outcome — above and beyond the measured covariates —
to fully explain away an observed treatment-outcome association.

    E-value = RR + sqrt(RR * (RR - 1))     (for RR >= 1; the protective
                                            case uses 1/RR)

For a confidence interval the E-value is computed for the confidence
limit closest to the null (or to a user-specified ``true`` value). If
the interval already contains that value, its E-value is exactly 1.

This module reproduces the R ``EValue`` package (VanderWeele, Ding,
Mathur, Smith) to machine precision, including:

* ratio measures ``RR`` / ``OR`` / ``HR`` with the rare / common-outcome
  conversions to the risk-ratio scale;
* the mean-difference (``MD`` / ``SMD``) and ``OLS`` approximations
  (``RR ~ exp(0.91 * d)`` with the ``1.78 * SE`` interval rule);
* the exact risk-difference E-value from a 2x2 table (``evalue_rd``,
  Ding & VanderWeele 2016 grid method);
* "non-null" E-values for a reference value ``true`` other than the
  null, via the generalised threshold function.

References
----------
VanderWeele, T. J. & Ding, P. (2017). Sensitivity Analysis in
Observational Research: Introducing the E-Value. *Annals of Internal
Medicine* 167(4), 268-274. [@vanderweele2017sensitivity]

Ding, P. & VanderWeele, T. J. (2016). Sensitivity Analysis Without
Assumptions. *Epidemiology* 27(3), 368-377. [@ding2016sensitivity]

Mathur, M. B., Ding, P., Riddell, C. A. & VanderWeele, T. J. (2018).
Web Site and R Package for Computing E-values. *Epidemiology* 29(5),
e45-e47.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy import stats as sp_stats


_RATIO_MEASURES = {"RR", "OR", "HR"}
_DIFF_MEASURES = {"MD", "SMD", "OLS", "DIFF", "RD"}
_VALID_MEASURES = _RATIO_MEASURES | _DIFF_MEASURES

# VanderWeele (2017) approximate conversion of a standardised mean
# difference d to a risk ratio: RR ~ exp(0.91 * d); the confidence
# interval uses exp(0.91 * d +/- 1.78 * SE)  (1.78 = 0.91 * 1.96).
_MD_SLOPE = 0.91
_MD_CI_FACTOR = 1.78


def evalue(
    estimate: float,
    se: Optional[float] = None,
    ci: Optional[Tuple[float, float]] = None,
    measure: str = "RR",
    rare: Optional[bool] = None,
    sd: Optional[float] = None,
    delta: float = 1.0,
    true: Optional[float] = None,
    rare_outcome: Optional[bool] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute the E-value for sensitivity to unmeasured confounding.

    Parameters
    ----------
    estimate : float
        Point estimate on the scale given by ``measure``:

        - ``'RR'`` risk ratio, ``'OR'`` odds ratio, ``'HR'`` hazard ratio
          (all must be > 0);
        - ``'MD'`` / ``'SMD'`` standardised mean difference;
        - ``'OLS'`` raw linear-regression coefficient (supply ``sd``);
        - ``'DIFF'`` / ``'RD'`` risk difference (approximate scalar path;
          prefer :func:`evalue_rd` with cell counts for the exact value).
    se : float, optional
        Standard error of ``estimate``. Required for ``MD`` / ``OLS`` to
        obtain a confidence-interval E-value; for ratio measures it is
        used (with ``alpha``) to build a CI when ``ci`` is not given.
    ci : tuple (lower, upper), optional
        Confidence interval on the ``measure`` scale. Takes precedence
        over ``se`` for ratio measures.
    measure : str, default 'RR'
        One of ``'RR'``, ``'OR'``, ``'HR'``, ``'MD'``, ``'SMD'``,
        ``'OLS'``, ``'DIFF'``, ``'RD'``.
    rare : bool, optional
        For ``OR`` / ``HR`` only: whether the rare-outcome approximation
        applies. ``rare=True`` treats OR/HR ~ RR; ``rare=False`` (the
        default) uses the exact common-outcome conversion to the RR
        scale (``sqrt(OR)`` for OR; the Ding-VanderWeele formula for HR).
    sd : float, optional
        Outcome standard deviation, required for ``measure='OLS'`` to
        standardise the coefficient.
    delta : float, default 1.0
        Contrast size for ``OLS`` (E-value for a ``delta``-unit change
        in the exposure).
    true : float, optional
        Reference value the confounding would have to move the estimate
        to. Defaults to the null: 1 for ratio measures, 0 for difference
        measures. A non-null ``true`` gives a "non-null" E-value.
    rare_outcome : bool, optional
        Deprecated alias for ``rare`` (kept for backwards compatibility).
    alpha : float, default 0.05
        Significance level used to build a CI from ``se`` when ``ci`` is
        not supplied (ratio measures).

    Returns
    -------
    dict
        ``evalue_estimate`` (point E-value), ``evalue_ci`` (E-value for
        the confidence limit closest to ``true``; ``None`` if no CI/SE),
        ``rr_estimate`` and ``rr_ci`` (the risk ratios used), ``measure``,
        ``true``, ``original_estimate``, ``ci`` (if available) and a
        plain-language ``interpretation``.

    Examples
    --------
    >>> import statspai as sp
    >>> round(sp.evalue(estimate=2.5, measure="RR")["evalue_estimate"], 4)
    4.4365
    >>> round(sp.evalue(estimate=0.5, se=0.1, sd=2.0, measure="OLS")["evalue_estimate"], 4)
    1.8218
    """
    measure = measure.upper()
    if measure not in _VALID_MEASURES:
        raise ValueError(
            f"measure must be one of {sorted(_VALID_MEASURES)}, got '{measure}'"
        )
    if rare is None and rare_outcome is not None:
        rare = rare_outcome
    if se is not None and se < 0:
        raise ValueError("Standard error cannot be negative.")

    # Default reference (null) value on the native scale.
    null_value = 0.0 if measure in _DIFF_MEASURES else 1.0
    true_native = null_value if true is None else float(true)

    # ------------------------------------------------------------------
    # Map (estimate, CI) onto the risk-ratio scale.
    # ------------------------------------------------------------------
    if measure in _RATIO_MEASURES:
        if estimate <= 0:
            raise ValueError(f"{measure} must be positive, got {estimate}.")
        if true_native < 0:
            raise ValueError("true value is impossible for a ratio measure.")
        if ci is None and se is not None:
            z = sp_stats.norm.ppf(1 - alpha / 2)
            ci = (estimate - z * se, estimate + z * se)
        rr = _ratio_to_rr(estimate, measure, rare)
        true_rr = _ratio_to_rr(true_native, measure, rare) if true_native > 0 else 0.0
        if ci is not None:
            _check_ci(estimate, ci)
            rr_lo = _ratio_to_rr(ci[0], measure, rare)
            rr_hi = _ratio_to_rr(ci[1], measure, rare)
        else:
            rr_lo = rr_hi = None

    elif measure in ("MD", "SMD", "OLS"):
        est_d, se_d = estimate, se
        if measure == "OLS":
            if sd is None or sd <= 0:
                raise ValueError("measure='OLS' requires the outcome sd > 0.")
            d = abs(delta)
            est_d = estimate * d / sd
            se_d = None if se is None else se * d / sd
            true_md = true_native * d / sd
        else:
            true_md = true_native
        rr = float(np.exp(_MD_SLOPE * est_d))
        true_rr = float(np.exp(_MD_SLOPE * true_md))
        if se_d is not None:
            rr_lo = float(np.exp(_MD_SLOPE * est_d - _MD_CI_FACTOR * se_d))
            rr_hi = float(np.exp(_MD_SLOPE * est_d + _MD_CI_FACTOR * se_d))
        elif ci is not None:
            # CI supplied on the (standardised) difference scale.
            _check_ci(estimate, ci)
            scale = (abs(delta) / sd) if measure == "OLS" else 1.0
            rr_lo = float(np.exp(_MD_SLOPE * ci[0] * scale))
            rr_hi = float(np.exp(_MD_SLOPE * ci[1] * scale))
        else:
            rr_lo = rr_hi = None

    else:  # DIFF / RD scalar (approximate; see evalue_rd for the exact path)
        rr = _diff_to_rr(estimate)
        true_rr = _diff_to_rr(true_native)
        if ci is None and se is not None:
            z = sp_stats.norm.ppf(1 - alpha / 2)
            ci = (estimate - z * se, estimate + z * se)
        if ci is not None:
            _check_ci(estimate, ci)
            rr_lo = _diff_to_rr(ci[0])
            rr_hi = _diff_to_rr(ci[1])
        else:
            rr_lo = rr_hi = None

    # ------------------------------------------------------------------
    # E-values on the risk-ratio scale (generalised threshold function).
    # ------------------------------------------------------------------
    ev_estimate = _threshold(rr, true_rr)
    rr_ci, ev_ci = _ci_evalue(rr, rr_lo, rr_hi, true_rr)

    interpretation = _interpret(ev_estimate, ev_ci, measure, true_native)

    result: Dict[str, Any] = {
        "evalue_estimate": ev_estimate,
        "evalue_ci": ev_ci,
        "rr_estimate": rr,
        "rr_ci": rr_ci,
        "measure": measure,
        "true": true_native,
        "original_estimate": estimate,
        "interpretation": interpretation,
    }
    if ci is not None:
        result["ci"] = ci
    return result


def evalue_rd(
    n11: float,
    n10: float,
    n01: float,
    n00: float,
    true: float = 0.0,
    alpha: float = 0.05,
    grid: float = 1e-4,
) -> Dict[str, Any]:
    """Exact E-value for a risk difference from a 2x2 table.

    Implements the Ding & VanderWeele (2016) risk-difference E-value,
    a faithful port of ``EValue::evalues.RD``. The exposure must be
    coded so that the risk difference is non-negative.

    Parameters
    ----------
    n11, n10 : float
        Exposed cases and exposed non-cases.
    n01, n00 : float
        Unexposed cases and unexposed non-cases.
    true : float, default 0.0
        Reference risk difference (must be <= the observed RD).
    alpha : float, default 0.05
        Significance level for the confidence-limit E-value.
    grid : float, default 1e-4
        Step size of the bias-factor grid search for the CI E-value.

    Returns
    -------
    dict
        ``evalue_estimate`` (point E-value), ``evalue_ci`` (E-value for
        the lower confidence limit; 1 if the CI already crosses
        ``true``), ``rd`` (the observed risk difference), ``bias_factor``
        and ``measure='RD'``.

    Examples
    --------
    >>> import statspai as sp
    >>> round(sp.evalue_rd(200, 150, 100, 250)["evalue_estimate"], 4)
    3.4142
    """
    cells = np.array([n11, n10, n01, n00], dtype=float)
    if np.any(cells < 0):
        raise ValueError("Negative cell counts are impossible.")
    N = cells.sum()
    N1 = n10 + n11
    N0 = n00 + n01
    if N1 <= 0 or N0 <= 0:
        raise ValueError("Each exposure group must have at least one unit.")
    f = N1 / N
    p1 = n11 / N1
    p0 = n01 / N0
    if p1 < p0:
        raise ValueError(
            "RD < 0; relabel the exposure so the risk difference is > 0."
        )
    if (p1 - p0) <= true:
        raise ValueError("true value must be <= the observed risk difference.")

    s2_f = f * (1 - f) / N
    s2_p1 = p1 * (1 - p1) / N1
    s2_p0 = p0 * (1 - p0) / N0

    diff = p0 * (1 - f) - p1 * f
    est_bf = (
        np.sqrt((true + diff) ** 2 + 4 * p1 * p0 * f * (1 - f)) - (true + diff)
    ) / (2 * p0 * f)
    ev_estimate = _threshold(est_bf)

    z = sp_stats.norm.ppf(1 - alpha / 2)
    lower_ci = (p1 - p0) - z * np.sqrt(s2_p1 + s2_p0)
    if lower_ci <= true:
        ev_ci = 1.0
    else:
        bf_search = np.arange(1.0, est_bf, grid)
        rd_search = p1 - p0 * bf_search
        f_search = f + (1 - f) / bf_search
        low_search = rd_search * f_search - z * np.sqrt(
            (s2_p1 + s2_p0 * bf_search ** 2) * f_search ** 2
            + rd_search ** 2 * (1 - 1 / bf_search) ** 2 * s2_f
        )
        hit = np.where(low_search <= true)[0]
        ev_ci = _threshold(bf_search[hit[0]]) if hit.size else 1.0

    return {
        "evalue_estimate": float(ev_estimate),
        "evalue_ci": float(ev_ci),
        "rd": float(p1 - p0),
        "bias_factor": float(est_bf),
        "measure": "RD",
        "true": float(true),
    }


def bias_factor(rr_eu: float, rr_ud: float) -> float:
    """Confounding bias factor ``B`` (Ding & VanderWeele 2016).

    Given the maximum risk ratios of an unmeasured confounder ``U`` with
    the exposure (``rr_eu``) and with the outcome (``rr_ud``), the
    factor by which they can bias a risk ratio is

        B = (rr_eu * rr_ud) / (rr_eu + rr_ud - 1).

    The E-value is the value ``e`` such that ``bias_factor(e, e)`` equals
    the observed risk ratio.

    Examples
    --------
    >>> import statspai as sp
    >>> round(sp.bias_factor(2.0, 2.0), 4)
    1.3333
    >>> round(sp.bias_factor(3.0, 3.0), 4)
    1.8
    """
    if rr_eu < 1 or rr_ud < 1:
        raise ValueError("Confounding associations must be >= 1.")
    return float((rr_eu * rr_ud) / (rr_eu + rr_ud - 1.0))


def evalue_from_result(
    result,
    measure: str = "SMD",
    rare: Optional[bool] = None,
    rare_outcome: Optional[bool] = None,
    true: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute an E-value from a StatsPAI CausalResult object.

    Parameters
    ----------
    result : CausalResult
        Result from any StatsPAI causal estimator exposing a scalar
        ``estimate`` (and ideally ``se`` / ``ci``).
    measure : str, default 'SMD'
        How to interpret ``result.estimate`` (for ATE/ATT on continuous
        outcomes ``'SMD'`` is appropriate; pass ``'RR'`` / ``'OR'`` /
        ``'HR'`` for ratio estimates).
    rare, rare_outcome, true :
        Passed through to :func:`evalue`.

    Returns
    -------
    dict
        Same structure as :func:`evalue`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for u in range(100):
    ...     treated = int(u < 50)
    ...     fe = rng.normal()
    ...     for post in (0, 1):
    ...         y = (1.0 + fe + 0.5 * post
    ...              + 0.8 * treated * post + rng.normal())
    ...         rows.append({"unit": u, "y": y,
    ...                      "treated": treated, "post": post})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.did(df, y="y", treat="treated", time="post")
    >>> ev = sp.evalue_from_result(res, measure="SMD")
    >>> ev["measure"]
    'SMD'
    >>> ev["evalue_estimate"] > 1.0
    True
    """
    if not hasattr(result, "estimate"):
        raise TypeError(
            "evalue_from_result expects a CausalResult carrying a single "
            "causal estimate (.estimate / .se / .ci), e.g. from sp.did, "
            "sp.iv, sp.dml, sp.synth or sp.metalearner. Got "
            f"{type(result).__name__}, which exposes no scalar treatment "
            "effect. For a single regression coefficient call "
            "sp.evalue(estimate=..., se=...) directly."
        )
    return evalue(
        estimate=result.estimate,
        se=getattr(result, "se", None),
        ci=getattr(result, "ci", None),
        measure=measure,
        rare=rare,
        rare_outcome=rare_outcome,
        true=true,
        alpha=getattr(result, "alpha", 0.05),
    )


# ======================================================================
# Internals
# ======================================================================

def _threshold(x: Optional[float], true: float = 1.0) -> Optional[float]:
    """Generalised E-value threshold (port of ``EValue:::threshold``).

    Returns the minimum confounding risk ratio needed to shift an
    observed risk ratio ``x`` to the reference value ``true`` (default
    the null, 1). Protective ratios (``x < 1``) are reflected.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = float(x)
    if x < 0:
        raise ValueError("The risk ratio must be non-negative.")
    if x <= 1.0:
        x = 1.0 / x if x > 0 else np.inf
        true = 1.0 / true if true > 0 else np.inf
    if true <= x:
        return float((x + np.sqrt(x * (x - true))) / true)
    rat = true / x
    return float(rat + np.sqrt(rat * (rat - 1.0)))


def _ci_evalue(rr, rr_lo, rr_hi, true_rr):
    """E-value for the confidence limit closest to ``true_rr``.

    If the interval already contains ``true_rr`` the E-value is 1.0.
    Returns ``(rr_ci_used, evalue_ci)``.
    """
    if rr_lo is None and rr_hi is None:
        return None, None
    if rr > true_rr:
        rr_ci = rr_lo
        if rr_ci is None:
            return None, None
        return rr_ci, (1.0 if rr_ci <= true_rr else _threshold(rr_ci, true_rr))
    if rr < true_rr:
        rr_ci = rr_hi
        if rr_ci is None:
            return None, None
        return rr_ci, (1.0 if rr_ci >= true_rr else _threshold(rr_ci, true_rr))
    # rr == true_rr: the estimate already sits at the reference value.
    return (rr_lo if rr_lo is not None else rr_hi), 1.0


def _ratio_to_rr(value: float, measure: str, rare: Optional[bool]) -> float:
    """Convert a ratio effect measure to the risk-ratio scale."""
    if value <= 0:
        return 1e-10
    if measure == "RR":
        return float(value)
    if measure == "HR":
        if rare:
            return float(value)
        # Ding-VanderWeele common-outcome HR -> RR conversion.
        return float((1 - 0.5 ** np.sqrt(value)) / (1 - 0.5 ** np.sqrt(1 / value)))
    if measure == "OR":
        if rare:
            return float(value)
        return float(np.sqrt(value))
    return float(value)


def _diff_to_rr(value: float) -> float:
    """Approximate risk-ratio for a scalar risk difference.

    This scalar path is an approximation kept for convenience; for the
    exact risk-difference E-value use :func:`evalue_rd` with the 2x2
    cell counts.
    """
    capped = float(np.clip(value, -1.0, 1.0))
    return float(np.exp(2.0 * capped))


def _check_ci(estimate, ci) -> None:
    """Validate a (lower, upper) confidence interval."""
    lo, hi = ci
    if lo > hi:
        raise ValueError(
            "Lower confidence limit should be <= the upper limit."
        )
    if not (lo <= estimate <= hi):
        raise ValueError("Point estimate should lie inside the CI.")


def _interpret(ev_est, ev_ci, measure, true_value) -> str:
    """Plain-language interpretation of the E-value(s)."""
    lines = []
    target = "the null" if true_value in (0.0, 1.0) else f"{true_value:g}"
    if ev_est is not None:
        lines.append(f"E-value for point estimate: {ev_est:.2f}")
        if ev_est > 3:
            strength = "very robust"
        elif ev_est > 2:
            strength = "moderately robust"
        elif ev_est > 1.5:
            strength = "somewhat robust"
        else:
            strength = "potentially sensitive"
        lines.append(
            f"The observed association is {strength} to unmeasured "
            "confounding."
        )
        lines.append(
            "An unmeasured confounder would need to be associated with "
            "both treatment and outcome by a risk ratio of at least "
            f"{ev_est:.2f} (each, beyond the measured covariates) to move "
            f"the point estimate to {target}."
        )
    if ev_ci is not None:
        lines.append(f"E-value for CI limit: {ev_ci:.2f}")
        if ev_ci <= 1.0:
            lines.append(
                "The confidence interval already includes the reference "
                "value, so its E-value is 1.0."
            )
        else:
            lines.append(
                "To move the confidence interval to include the reference "
                f"value, the confounder associations must be >= {ev_ci:.2f}."
            )
    return "\n".join(lines)
