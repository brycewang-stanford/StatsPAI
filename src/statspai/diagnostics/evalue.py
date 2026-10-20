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

The formulas are taken from the papers below. The R ``EValue`` package
(GPL-2) served only as a black-box check (its outputs on our test cases,
never its source); the results agree to machine precision. Covered:

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

``_threshold`` and ``evalue_rd`` are clean-room implementations from these
papers (spec: ``docs/dev/cleanroom/evalue.md``).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from .._dict_result import DictResult
from ..exceptions import DataInsufficient, MethodIncompatibility


class EValueResult(DictResult):
    """Result of :func:`evalue`: a ``dict`` (keys unchanged) with
    ``summary()`` / ``glance()`` / ``to_latex()`` ..."""

    _title = "E-value (sensitivity to unmeasured confounding)"
    _citation_keys = ("vanderweele2017sensitivity",)


_RATIO_MEASURES = {"RR", "OR", "HR"}
_DIFF_MEASURES = {"MD", "SMD", "OLS", "DIFF", "RD"}
_VALID_MEASURES = _RATIO_MEASURES | _DIFF_MEASURES

# VanderWeele (2017) approximate conversion of a standardised mean
# difference d to a risk ratio: RR ~ exp(0.91 * d); the confidence
# interval uses exp(0.91 * d +/- 1.78 * SE)  (1.78 = 0.91 * 1.96).
_MD_SLOPE = 0.91
_MD_CI_FACTOR = 1.78


def _require_string_option(value: Any, name: str, context: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a string option.",
            diagnostics={"context": context, name: repr(value)},
        )
    out = value.upper().strip()
    if not out:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a non-empty string option.",
            diagnostics={"context": context, name: repr(value)},
        )
    return out


def _finite_float(value: Any, name: str, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a finite number.",
            diagnostics={"context": context, name: repr(value)},
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be a finite number.",
            diagnostics={"context": context, name: repr(value)},
        ) from exc
    if not np.isfinite(out):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be finite.",
            diagnostics={"context": context, name: out},
        )
    return out


def _positive_float(value: Any, name: str, context: str) -> float:
    out = _finite_float(value, name, context)
    if out <= 0:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be positive.",
            diagnostics={"context": context, name: out},
        )
    return out


def _nonnegative_float(value: Any, name: str, context: str) -> float:
    out = _finite_float(value, name, context)
    if out < 0:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be non-negative.",
            diagnostics={"context": context, name: out},
        )
    return out


def _open_unit_float(value: Any, name: str, context: str) -> float:
    out = _finite_float(value, name, context)
    if not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"{context}: `{name}` must be in (0, 1).",
            diagnostics={"context": context, name: out},
        )
    return out


def _optional_bool(value: Optional[bool], name: str, context: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"{context}: `{name}` must be boolean when supplied.",
            diagnostics={"context": context, name: repr(value)},
        )
    return bool(value)


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
) -> "EValueResult":
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
    EValueResult
        A ``dict`` (the keys below, unchanged) that also has
        ``summary()`` / ``glance()`` / ``to_latex()`` / ``to_word()``.
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
    >>> out = sp.evalue(estimate=0.5, se=0.1, sd=2.0, measure="OLS")
    >>> round(out["evalue_estimate"], 4)
    1.8218
    """
    context = "evalue"
    estimate = _finite_float(estimate, "estimate", context)
    measure = _require_string_option(measure, "measure", context)
    if measure not in _VALID_MEASURES:
        raise MethodIncompatibility(
            f"measure must be one of {sorted(_VALID_MEASURES)}, got '{measure}'",
            diagnostics={
                "context": context,
                "measure": measure,
                "valid_measures": sorted(_VALID_MEASURES),
            },
        )
    if rare is None and rare_outcome is not None:
        rare = rare_outcome
    rare = _optional_bool(rare, "rare", context)
    if se is not None:
        se = _nonnegative_float(se, "se", context)
    alpha = _open_unit_float(alpha, "alpha", context)
    delta = _finite_float(delta, "delta", context)

    # Default reference (null) value on the native scale.
    null_value = 0.0 if measure in _DIFF_MEASURES else 1.0
    true_native = null_value if true is None else _finite_float(true, "true", context)

    # ------------------------------------------------------------------
    # Map (estimate, CI) onto the risk-ratio scale.
    # ------------------------------------------------------------------
    rr_lo: Optional[float]
    rr_hi: Optional[float]
    if measure in _RATIO_MEASURES:
        if estimate <= 0:
            raise MethodIncompatibility(
                f"{measure} must be positive, got {estimate}.",
                diagnostics={"context": context, "estimate": estimate},
            )
        if true_native < 0:
            raise MethodIncompatibility(
                "true value is impossible for a ratio measure.",
                diagnostics={"context": context, "true": true_native},
            )
        if ci is None and se is not None:
            z = sp_stats.norm.ppf(1 - alpha / 2)
            ci = (estimate - z * se, estimate + z * se)
        rr = _ratio_to_rr(estimate, measure, rare)
        true_rr = _ratio_to_rr(true_native, measure, rare) if true_native > 0 else 0.0
        if ci is not None:
            ci = _check_ci(estimate, ci)
            rr_lo = _ratio_to_rr(ci[0], measure, rare)
            rr_hi = _ratio_to_rr(ci[1], measure, rare)
        else:
            rr_lo = rr_hi = None

    elif measure in ("MD", "SMD", "OLS"):
        est_d = estimate
        se_d: Optional[float] = se
        ols_scale = 1.0
        if measure == "OLS":
            if sd is None:
                raise MethodIncompatibility(
                    "measure='OLS' requires the outcome sd > 0.",
                    diagnostics={"context": context, "sd": None},
                )
            sd = _positive_float(sd, "sd", context)
            ols_scale = abs(delta) / sd
            est_d = estimate * ols_scale
            se_d = None if se is None else se * ols_scale
            true_md = true_native * ols_scale
        else:
            true_md = true_native
        rr = float(np.exp(_MD_SLOPE * est_d))
        true_rr = float(np.exp(_MD_SLOPE * true_md))
        if se_d is not None:
            rr_lo = float(np.exp(_MD_SLOPE * est_d - _MD_CI_FACTOR * se_d))
            rr_hi = float(np.exp(_MD_SLOPE * est_d + _MD_CI_FACTOR * se_d))
        elif ci is not None:
            # CI supplied on the (standardised) difference scale.
            ci = _check_ci(estimate, ci)
            scale = ols_scale if measure == "OLS" else 1.0
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
            ci = _check_ci(estimate, ci)
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
    return EValueResult(result)


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

    Uses the bound of Ding & VanderWeele (2016, Proposition A.8) on the
    population-standardised causal risk difference under confounding with
    bounding factor ``B``,

        RD_true >= (p1 - p0 * B) * (f + (1 - f) / B),

    where ``p1`` / ``p0`` are the outcome risks in the exposed / unexposed
    and ``f`` is the exposure prevalence. The point E-value converts the
    bias factor that moves this bound to ``true`` (closed-form root of
    the quadratic in their Proposition A.11) to the E-value scale. The
    confidence-limit E-value uses the smallest bias factor on a grid
    ``1, 1 + grid, 1 + 2*grid, ...`` at which the delta-method lower
    confidence limit of the bound reaches ``true`` (their equation A.5).

    Parameters
    ----------
    n11, n10 : float
        Exposed cases and exposed non-cases.
    n01, n00 : float
        Unexposed cases and unexposed non-cases.
    true : float, default 0.0
        Reference risk difference (must be below the observed RD).
    alpha : float, default 0.05
        Significance level for the confidence-limit E-value.
    grid : float, default 1e-4
        Step size of the bias-factor grid search for the CI E-value.

    Returns
    -------
    dict
        ``evalue_estimate`` (point E-value), ``evalue_ci`` (E-value for the
        lower confidence limit; 1.0 when the interval already reaches
        ``true``), ``rd`` (observed risk difference), ``bias_factor`` (the
        bias factor that moves the point estimate to ``true``),
        ``measure`` (``'RD'``) and ``true``.

    Notes
    -----
    Clean-room implementation from the paper (spec:
    ``docs/dev/cleanroom/evalue.md``); R ``EValue::evalues.RD`` (GPL-2)
    was used only as a black box and agrees to 1e-12. When no grid point
    up to the point-estimate bias factor qualifies (a coarse grid on a very
    precise table), that bias factor itself is used as the last grid
    point, so the answer still overshoots the exact root by less than one
    grid step; R returns ``NA`` there.

    References
    ----------
    [@ding2016sensitivity]; [@vanderweele2017sensitivity].

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.evalue_rd(200, 150, 100, 250)
    >>> round(out["evalue_estimate"], 6), round(out["evalue_ci"], 6)
    (3.414214, 2.726306)
    """
    context = "evalue_rd"
    cells = [
        _finite_float(value, name, context)
        for value, name in ((n11, "n11"), (n10, "n10"), (n01, "n01"), (n00, "n00"))
    ]
    if min(cells) < 0:
        raise MethodIncompatibility(
            "Negative cell counts are impossible.",
            recovery_hint="Pass the four non-negative 2x2 cell counts.",
            diagnostics={"context": context, "cells": cells},
        )
    true = _finite_float(true, "true", context)
    alpha = _open_unit_float(alpha, "alpha", context)
    grid = _positive_float(grid, "grid", context)

    exp_case, exp_noncase, unexp_case, unexp_noncase = cells
    n_exposed = exp_case + exp_noncase
    n_unexposed = unexp_case + unexp_noncase
    if n_exposed <= 0 or n_unexposed <= 0:
        raise DataInsufficient(
            "Each exposure group must have at least one unit.",
            recovery_hint="Check the 2x2 table: both exposure rows need units.",
            diagnostics={"context": context, "N1": n_exposed, "N0": n_unexposed},
        )
    n_total = n_exposed + n_unexposed
    prev = n_exposed / n_total
    risk1 = exp_case / n_exposed
    risk0 = unexp_case / n_unexposed
    rd = risk1 - risk0
    if rd < 0:
        raise MethodIncompatibility(
            "RD < 0; relabel the exposure so the risk difference is > 0.",
            recovery_hint="Swap the exposed and unexposed rows of the table.",
            diagnostics={"context": context, "rd": rd},
        )
    if true >= rd:
        raise MethodIncompatibility(
            "true value must be strictly less than the observed risk difference.",
            recovery_hint="Choose true below the observed RD (0 by default).",
            diagnostics={"context": context, "rd": rd, "true": true},
        )
    if risk0 <= 0:
        raise MethodIncompatibility(
            "The unexposed group has no cases, so the observed risk ratio is "
            "infinite and no finite bias factor moves the risk difference to "
            "`true`: the risk-difference E-value is unbounded.",
            recovery_hint=(
                "Report the E-value as unbounded, or add a continuity "
                "correction (e.g. 0.5 to every cell) before calling evalue_rd."
            ),
            diagnostics={"context": context, "n01": unexp_case, "rd": rd},
        )

    # Point estimate: positive root of Proposition A.11's quadratic.
    shift = true + risk0 * (1.0 - prev) - risk1 * prev
    bf_point = (
        np.sqrt(shift * shift + 4.0 * risk1 * risk0 * prev * (1.0 - prev)) - shift
    ) / (2.0 * risk0 * prev)
    bf_point = float(bf_point)

    # Confidence limit: first grid bias factor whose lower limit <= true.
    var1 = risk1 * (1.0 - risk1) / n_exposed
    var0 = risk0 * (1.0 - risk0) / n_unexposed
    var_prev = prev * (1.0 - prev) / n_total
    z = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))

    def lower_limit(bf: np.ndarray) -> np.ndarray:
        weight = prev + (1.0 - prev) / bf
        reduced = risk1 - risk0 * bf
        var = (var1 + var0 * bf**2) * weight**2 + reduced**2 * (
            1.0 - 1.0 / bf
        ) ** 2 * var_prev
        return reduced * weight - z * np.sqrt(var)

    n_steps = int(np.floor((bf_point - 1.0) / grid + 1e-10))
    bf_ci = bf_point  # used when no grid point qualifies
    block = 1_000_000
    for start in range(0, n_steps + 1, block):
        steps = np.arange(start, min(start + block, n_steps + 1), dtype=float)
        candidates = 1.0 + steps * grid
        hits = np.flatnonzero(lower_limit(candidates) <= true)
        if hits.size:
            bf_ci = float(candidates[hits[0]])
            break

    return {
        "evalue_estimate": _threshold(bf_point),
        "evalue_ci": _threshold(bf_ci),
        "rd": float(rd),
        "bias_factor": bf_point,
        "measure": "RD",
        "true": true,
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
    rr_eu = _finite_float(rr_eu, "rr_eu", "bias_factor")
    rr_ud = _finite_float(rr_ud, "rr_ud", "bias_factor")
    if rr_eu < 1 or rr_ud < 1:
        raise MethodIncompatibility(
            "Confounding associations must be >= 1.",
            diagnostics={"rr_eu": rr_eu, "rr_ud": rr_ud},
        )
    return float((rr_eu * rr_ud) / (rr_eu + rr_ud - 1.0))


def evalue_from_result(
    result: Any,
    measure: str = "SMD",
    rare: Optional[bool] = None,
    rare_outcome: Optional[bool] = None,
    true: Optional[float] = None,
    *,
    sd: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute an E-value from a StatsPAI CausalResult object.

    Parameters
    ----------
    result : CausalResult
        Result from any StatsPAI causal estimator exposing a scalar
        ``estimate`` (and ideally ``se`` / ``ci``).
    measure : str, default 'SMD'
        How to interpret ``result.estimate``. ``'SMD'`` / ``'MD'`` read it
        as a difference in means on the outcome's scale and divide it
        (with its SE and CI) by ``sd`` before converting to the risk-ratio
        scale; pass ``'RR'`` / ``'OR'`` / ``'HR'`` for ratio estimates.
    rare, rare_outcome, true :
        Passed through to :func:`evalue`.
    sd : float, optional
        Standard deviation of the outcome, required for ``'SMD'`` /
        ``'MD'``: the E-value of a mean difference depends on it, and the
        raw ATT is not a standardised difference unless the outcome
        already has unit variance (then pass ``sd=1``). Usually
        ``df[y].std()``. Ignored for ratio measures.

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
    >>> ev = sp.evalue_from_result(res, measure="SMD", sd=df["y"].std())
    >>> ev["measure"]
    'SMD'
    >>> ev["evalue_estimate"] > 1.0
    True
    """
    if not hasattr(result, "estimate"):
        raise MethodIncompatibility(
            "evalue_from_result expects a CausalResult carrying a single "
            "causal estimate (.estimate / .se / .ci), e.g. from sp.did, "
            "sp.iv, sp.dml, sp.synth or sp.metalearner. Got "
            f"{type(result).__name__}, which exposes no scalar treatment "
            "effect. For a single regression coefficient call "
            "sp.evalue(estimate=..., se=...) directly.",
            diagnostics={
                "context": "evalue_from_result",
                "type": type(result).__name__,
            },
            alternative_functions=["sp.evalue"],
        )
    estimate = result.estimate
    se = getattr(result, "se", None)
    ci = getattr(result, "ci", None)
    if str(measure).upper() in ("SMD", "MD"):
        if sd is None:
            raise MethodIncompatibility(
                f"evalue_from_result(measure={measure!r}) needs the outcome's "
                "standard deviation: the estimate is a difference in means on "
                "the outcome's own scale, and its E-value changes with the "
                "units of the outcome unless it is standardised first.",
                recovery_hint=(
                    "Pass sd=df[<outcome>].std() (sd=1 if the outcome is "
                    "already standardised), or measure='RR' / 'OR' / 'HR' "
                    "when the estimate is a ratio."
                ),
                diagnostics={
                    "context": "evalue_from_result",
                    "measure": measure,
                    "estimate": estimate,
                },
                alternative_functions=["sp.evalue"],
            )
        sd = float(sd)
        if not np.isfinite(sd) or sd <= 0:
            raise MethodIncompatibility(
                f"sd must be a positive finite number, got {sd!r}.",
                recovery_hint="Pass the outcome's standard deviation, e.g. df[y].std().",
                diagnostics={"context": "evalue_from_result", "sd": sd},
            )
        estimate = float(estimate) / sd
        se = None if se is None else float(se) / sd
        if ci is not None:
            ci = (float(ci[0]) / sd, float(ci[1]) / sd)
    out = evalue(
        estimate=estimate,
        se=se,
        ci=ci,
        measure=measure,
        rare=rare,
        rare_outcome=rare_outcome,
        true=true,
        alpha=getattr(result, "alpha", 0.05),
    )
    if str(measure).upper() in ("SMD", "MD"):
        out["outcome_sd"] = sd
    return out


# ======================================================================
# Internals
# ======================================================================


def _threshold(x: Optional[float], true: float = 1.0) -> Optional[float]:
    """E-value for moving a risk ratio ``x`` to the reference ratio ``true``.

    Confounding can change a risk ratio by at most the bounding factor
    ``B(e, e) = e**2 / (2e - 1)`` when both confounder associations equal
    ``e`` (Ding & VanderWeele 2016). Moving ``x`` to ``true`` needs a bias
    factor ``r = x / true`` (``x >= true``) or ``true / x`` (``x < true``),
    and the E-value is the inverse ``e = r + sqrt(r * (r - 1))``. With
    ``true = 1`` this is the VanderWeele-Ding (2017) E-value.

    Returns ``None`` for a missing ``x``; ``inf`` when no finite
    confounding reaches ``true`` (``x = 0`` or ``true = 0``); NaN when
    both are 0 or ``true`` is NaN.
    """
    if x is None:
        return None
    x = float(x)
    if np.isnan(x):
        return None
    if x < 0:
        raise MethodIncompatibility(
            "The risk ratio must be non-negative.",
            recovery_hint="Pass a risk ratio >= 0 (convert other measures first).",
            diagnostics={"context": "evalue", "rr": x},
        )
    true = float(true)
    if np.isnan(true):
        return float("nan")
    if true == 0.0 or x == 0.0:
        if x == true:
            return float("nan")
        return float("inf")
    if x == true:
        return 1.0
    ratio = x / true if x > true else true / x
    if np.isinf(ratio):
        return float("inf")
    return float(ratio + np.sqrt(ratio * (ratio - 1.0)))


def _ci_evalue(
    rr: float,
    rr_lo: Optional[float],
    rr_hi: Optional[float],
    true_rr: float,
) -> Tuple[Optional[float], Optional[float]]:
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


def _check_ci(estimate: float, ci: Tuple[float, float]) -> Tuple[float, float]:
    """Validate a (lower, upper) confidence interval."""
    try:
        lo_raw, hi_raw = ci
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "CI must be a two-element (lower, upper) tuple.",
            diagnostics={"context": "evalue", "ci": repr(ci)},
        ) from exc
    lo = _finite_float(lo_raw, "ci_lower", "evalue")
    hi = _finite_float(hi_raw, "ci_upper", "evalue")
    if lo > hi:
        raise MethodIncompatibility(
            "Lower confidence limit should be <= the upper limit.",
            diagnostics={"context": "evalue", "ci": [lo, hi]},
        )
    if not (lo <= estimate <= hi):
        raise MethodIncompatibility(
            "Point estimate should lie inside the CI.",
            diagnostics={"context": "evalue", "estimate": estimate, "ci": [lo, hi]},
        )
    return lo, hi


def _interpret(
    ev_est: Optional[float],
    ev_ci: Optional[float],
    measure: str,
    true_value: float,
) -> str:
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
            f"The observed association is {strength} to unmeasured " "confounding."
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
