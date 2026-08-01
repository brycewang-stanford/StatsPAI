"""
Unified sensitivity dashboard for any CausalResult / EconometricResults.

A single ``.sensitivity()`` call that runs every applicable
sensitivity analysis and returns a tidy report:

  - **E-value** (VanderWeele & Ding 2017) — always applicable
  - **Oster delta** (Oster 2019) — requires R^2 estimates
  - **Rosenbaum Gamma** (Rosenbaum 2002) — requires matched-pair
    outcomes exposed as ``result.matched_pairs``
  - **Sensemakr** (Cinelli & Hazlett 2020) — requires the raw
    estimation data, passed via ``data`` / ``y`` / ``treat`` /
    ``controls`` (result objects do not carry the data)
  - **Breakdown frontier** — how much bias flips the sign

Attached as the ``sensitivity`` method of :class:`CausalResult` and
:class:`EconometricResults` via a lightweight monkey-patch in
``statspai.__init__``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..exceptions import MethodIncompatibility

__all__ = ["SensitivityDashboard", "unified_sensitivity"]


@dataclass
class SensitivityDashboard:
    """Result of a unified sensitivity analysis.

    Always contains an ``e_value`` entry; other entries are optional
    depending on what the estimator provides.
    """

    e_value_point: float
    e_value_ci: Optional[float]
    rr_observed: float
    ci_observed: tuple[float, float]
    oster: Optional[Dict[str, float]] = None
    rosenbaum: Optional[Dict[str, float]] = None
    sensemakr: Optional[Dict[str, float]] = None
    breakdown: Optional[Dict[str, float]] = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bar = "=" * 60
        lines = [
            "Unified Sensitivity Dashboard",
            bar,
            f"  Risk ratio the E-value is based on: {self.rr_observed:.4f}",
            f"  Observed 95% CI: [{self.ci_observed[0]:.4f}, "
            f"{self.ci_observed[1]:.4f}]",
            "",
            f"  E-value (point) : {self.e_value_point:.4f}",
        ]
        if self.e_value_ci is not None:
            lines.append(f"  E-value (CI)    : {self.e_value_ci:.4f}")
        if self.oster is not None:
            lines.append(
                f"  Oster delta     : "
                f"{self.oster.get('delta', float('nan')):.3f}  "
                f"(bias-adjusted beta = "
                f"{self.oster.get('beta_star', float('nan')):.4f})"
            )
        if self.rosenbaum is not None:
            lines.append(
                f"  Rosenbaum Gamma : "
                f"{self.rosenbaum.get('gamma_critical', float('nan')):.3f}"
            )
        if self.sensemakr is not None:
            lines.append(
                f"  Sensemakr RV(q=1)   : "
                f"{self.sensemakr.get('rv_q1', float('nan')):.4f}"
            )
            if "rv_qa" in self.sensemakr:
                lines.append(
                    f"  Sensemakr RV(q=1,a) : " f"{self.sensemakr['rv_qa']:.4f}"
                )
        if self.breakdown is not None:
            lines.append(
                f"  Breakdown bias     : "
                f"{self.breakdown.get('bias_to_flip', float('nan')):.4f}"
            )
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        lines.append(bar)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Extraction helpers
# --------------------------------------------------------------------------- #


def _float_or_nan(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _finite_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    return None if np.isnan(out) else out


_INTERCEPT_NAMES = frozenset({"intercept", "const", "(intercept)", "_cons"})


def _coefficient_terms(params: Any) -> list[str]:
    """Names in a params Series that are not the intercept."""
    index = getattr(params, "index", None)
    if index is None:
        return []
    return [n for n in index if str(n).strip().lower() not in _INTERCEPT_NAMES]


def _extract_estimate(
    result: Any,
    term: Optional[str] = None,
) -> tuple[Optional[float], Optional[float], Optional[tuple[float, float]]]:
    """Pull (estimate, se, ci) for one coefficient off a result object.

    All three come from the *same* term. Before 1.21.0 this took
    ``params.iloc[0]`` and ``std_errors.iloc[0]`` — for any formula
    regression that is the **intercept**, so the whole dashboard silently
    reported the sensitivity of the intercept rather than of the treatment
    effect, and paired it with an intercept SE.

    Scalar-estimate results (``CausalResult`` and friends) are unambiguous
    and are used directly. For a coefficient vector we use ``term`` when
    given, fall back to the single non-intercept coefficient when there is
    exactly one, and otherwise refuse to guess.
    """
    estimate = None
    se = None
    ci = None

    for attr in ("estimate", "coef", "coefficient", "ate"):
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, (int, float, np.floating)):
                estimate = float(v)
                break

    params = getattr(result, "params", None)
    if estimate is None and params is not None:
        terms = _coefficient_terms(params)
        if term is not None:
            if term not in getattr(params, "index", []):
                raise MethodIncompatibility(
                    f"term={term!r} is not a coefficient of this result.",
                    diagnostics={
                        "context": "unified_sensitivity",
                        "requested_term": term,
                        "available_terms": [str(t) for t in terms],
                    },
                    recovery_hint=(
                        "Pass one of the available terms, or supply "
                        "estimate=/se=/ci= directly."
                    ),
                )
            chosen = term
        elif len(terms) == 1:
            chosen = terms[0]
        else:
            raise MethodIncompatibility(
                "unified_sensitivity cannot tell which coefficient you mean: "
                f"this result has {len(terms)} non-intercept coefficients.",
                diagnostics={
                    "context": "unified_sensitivity",
                    "available_terms": [str(t) for t in terms],
                },
                recovery_hint=(
                    "Pass term='<treatment column>' to pick the coefficient "
                    "to analyse. Guessing would silently report the "
                    "sensitivity of the wrong parameter."
                ),
            )
        estimate = float(params[chosen])
        std_errors = getattr(result, "std_errors", None)
        if std_errors is not None:
            try:
                se = float(std_errors[chosen])
            except Exception:
                se = None
        conf_int = getattr(result, "conf_int", None)
        if callable(conf_int):
            try:
                row = conf_int().loc[chosen]
                ci = (float(row.iloc[0]), float(row.iloc[1]))
            except Exception:
                ci = None

    if se is None and hasattr(result, "se"):
        v = getattr(result, "se")
        if isinstance(v, (int, float, np.floating)):
            se = float(v)
    if ci is None and hasattr(result, "ci"):
        v = getattr(result, "ci")
        try:
            ci = (float(v[0]), float(v[1]))
        except Exception:
            pass
    if ci is None and estimate is not None and se is not None:
        ci = (estimate - 1.96 * se, estimate + 1.96 * se)
    return estimate, se, ci


class _SkipEValue(Exception):
    """Internal sentinel: the E-value is not defined for these inputs."""


_RATIO_ESTIMANDS = ("rr", "risk ratio", "odds ratio", "or", "hazard ratio", "hr")


def _outcome_sd(
    result: Any,
    data: Any,
    y: Optional[str],
    outcome_sd: Optional[float],
) -> Optional[float]:
    """Outcome standard deviation, needed to standardise a raw coefficient."""
    if outcome_sd is not None:
        sd = float(outcome_sd)
        return sd if sd > 0 else None
    if data is not None and y is not None:
        try:
            sd = float(np.asarray(data[y], dtype=float).std(ddof=1))
        except Exception:
            return None
        return sd if np.isfinite(sd) and sd > 0 else None
    for attr in ("outcome_sd", "y_sd", "dv_sd"):
        value = getattr(result, attr, None)
        if isinstance(value, (int, float, np.floating)) and float(value) > 0:
            return float(value)
    return None


def _looks_like_a_ratio(result: Any) -> bool:
    """Whether the result's own metadata says the estimate is a ratio."""
    for attr in ("estimand", "measure", "scale"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip().lower() in _RATIO_ESTIMANDS:
            return True
    return False


def _coerce_matched_pairs(mp: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``result.matched_pairs`` into (treated, control) outcome arrays.

    Accepts a 2-tuple/list ``(treated, control)``, an ``(n, 2)`` array
    (column 0 = treated, column 1 = control), or a DataFrame / dict with
    ``'treated'`` and ``'control'`` entries.
    """
    if isinstance(mp, dict) and "treated" in mp and "control" in mp:
        return (
            np.asarray(mp["treated"], dtype=float),
            np.asarray(mp["control"], dtype=float),
        )
    if hasattr(mp, "columns"):  # DataFrame-like
        cols = {str(c).lower(): c for c in mp.columns}
        if "treated" in cols and "control" in cols:
            return (
                np.asarray(mp[cols["treated"]], dtype=float),
                np.asarray(mp[cols["control"]], dtype=float),
            )
        if mp.shape[1] == 2:
            arr = np.asarray(mp, dtype=float)
            return arr[:, 0], arr[:, 1]
    if isinstance(mp, (tuple, list)) and len(mp) == 2:
        return (np.asarray(mp[0], dtype=float), np.asarray(mp[1], dtype=float))
    arr = np.asarray(mp)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].astype(float), arr[:, 1].astype(float)
    raise ValueError(
        "matched_pairs must be a (treated, control) pair of outcome "
        "arrays, an (n, 2) array, or a DataFrame/dict with 'treated' "
        "and 'control' entries"
    )


# --------------------------------------------------------------------------- #
#  Main entry
# --------------------------------------------------------------------------- #


def unified_sensitivity(
    result: Any,
    *,
    term: Optional[str] = None,
    measure: str = "auto",
    outcome_sd: Optional[float] = None,
    r2_short: Optional[float] = None,
    r2_long: Optional[float] = None,
    r2_treated: Optional[float] = None,
    r2_controlled: Optional[float] = None,
    beta_uncontrolled: Optional[float] = None,
    rho_max: float = 1.0,
    data: Any = None,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    controls: Optional[Sequence[str]] = None,
    include_oster: bool = True,
    include_rosenbaum: bool = True,
    include_sensemakr: bool = True,
) -> SensitivityDashboard:
    """Run all applicable sensitivity analyses in one shot.

    Parameters
    ----------
    result : CausalResult / EconometricResults / dataclass with
        ``estimate``, ``se``, ``ci`` attributes.
    r2_short, r2_long : float, optional
        R^2 of the short (treatment-only) and long (with controls)
        regression, for Oster's delta. Usually unnecessary: when ``data``,
        ``y``, ``treat`` and ``controls`` are supplied these are derived
        from the data, which is what keeps this delta equal to the one
        ``sp.oster_delta`` reports on the same specification.
    r2_treated, r2_controlled : float, optional
        Deprecated aliases for ``r2_short`` / ``r2_long``. The old names
        read like sensemakr's partial R^2 (outcome/treatment variance
        explained by an unobservable) but were consumed as the short/long
        regression R^2, so supplying sensemakr-style values silently
        produced a wrong delta*. Use the new names.
    beta_uncontrolled : float, optional
        Short-regression (no controls) treatment estimate; required for
        Oster's delta together with the two R^2 values.
    rho_max : float, default 1.0
        Oster's ``R_max`` — the R^2 of the hypothetical long regression
        that additionally includes all unobservables. The default 1.0 is
        the most conservative bound.
    term : str, optional
        Which coefficient to analyse, when ``result`` carries a vector of
        them (e.g. an OLS fit). Required whenever there is more than one
        non-intercept coefficient — the dashboard raises rather than guess,
        because guessing silently answers the sensitivity question about
        the wrong parameter. Ignored for results that expose a single
        scalar ``estimate`` / ``ate``.
    measure : {'auto', 'RR', 'OR', 'HR', 'OLS', 'SMD'}, default 'auto'
        Scale the estimate lives on, for the **E-value** component. The
        E-value is defined for risk ratios, so a raw regression coefficient
        must be standardised before it means anything:
        ``RR ~ exp(0.91 * d)`` for a standardised mean difference ``d``
        (``vanderweele2017sensitivity``).

        ``'auto'`` reads the result's own ``estimand`` / ``measure`` and
        uses ``'RR'`` when it names a ratio, otherwise treats the estimate
        as a difference on the outcome's scale and standardises it. Pass
        ``'RR'`` explicitly if the estimate really is a ratio the result
        does not advertise.
    outcome_sd : float, optional
        Standard deviation of the outcome, used to standardise the
        coefficient. Taken from ``data[y]`` when both are supplied. Without
        it the E-value is skipped with a note instead of being computed on
        an unknown scale.
    data : pd.DataFrame, optional
        Raw estimation data. Required for the **Sensemakr** component:
        the Cinelli-Hazlett robustness value is computed from the
        underlying regression, which result objects do not carry. When
        omitted, the Sensemakr component is skipped with an explanatory
        note — call ``sp.sensemakr(data, y, treat, controls)`` directly
        instead.
    y, treat : str, optional
        Outcome / treatment column names in ``data`` (Sensemakr only).
    controls : sequence of str, optional
        Control column names in ``data`` (Sensemakr only).

    Returns
    -------
    SensitivityDashboard

    Examples
    --------
    >>> import statspai as sp
    >>> from types import SimpleNamespace
    >>> # Any result exposing estimate / se / ci works. Say what scale the
    >>> # estimate is on — a risk ratio here — so the E-value is meaningful.
    >>> result = SimpleNamespace(estimate=1.35, se=0.10, ci=(1.15, 1.55))
    >>> dash = sp.unified_sensitivity(result, measure="RR")
    >>> type(dash).__name__
    'SensitivityDashboard'
    >>> bool(dash.e_value_point >= 1.0)  # E-values are >= 1 by construction
    True
    >>> dash.breakdown is not None
    True
    >>> # A mean difference needs the outcome SD to be standardised first.
    >>> diff = SimpleNamespace(estimate=0.35, se=0.10, ci=(0.15, 0.55))
    >>> dash2 = sp.unified_sensitivity(diff, outcome_sd=2.0)
    >>> bool(dash2.e_value_point >= 1.0)
    True
    >>> # Without a scale the E-value is skipped rather than invented.
    >>> import math
    >>> math.isnan(sp.unified_sensitivity(diff).e_value_point)
    True
    """
    from ..diagnostics.evalue import evalue as _evalue_fn

    # `treat` already names the treatment column for the Sensemakr
    # component. If the caller supplied it and it is a coefficient of this
    # result, it is also the answer to "which term did you mean?" — asking
    # for the same name twice would be pointless ceremony.
    resolved_term = term
    if resolved_term is None and treat is not None:
        params = getattr(result, "params", None)
        if params is not None and treat in getattr(params, "index", []):
            resolved_term = treat

    estimate, se, ci = _extract_estimate(result, term=resolved_term)
    if estimate is None or se is None or ci is None:
        raise ValueError(
            "Could not extract (estimate, se, ci) from result object. "
            "Supply them directly via "
            ".sensitivity(estimate=..., se=..., ci=...)."
        )

    notes: list[str] = []

    if r2_treated is not None or r2_controlled is not None:
        warnings.warn(
            "r2_treated / r2_controlled are deprecated aliases for "
            "r2_short / r2_long (the short- and long-regression R^2 for "
            "Oster's delta). The old names read like sensemakr's partial "
            "R^2, and passing sensemakr-style values here silently "
            "produced a delta* that disagreed with sp.oster_delta on the "
            "same specification. Prefer supplying data=, y=, treat= and "
            "controls= and letting the R^2 be derived.",
            DeprecationWarning,
            stacklevel=2,
        )
        if r2_short is None:
            r2_short = r2_treated
        if r2_long is None:
            r2_long = r2_controlled

    # 1. E-value.  The E-value is defined on the risk-ratio scale, so a raw
    # linear-regression coefficient has to be standardised first —
    # VanderWeele & Ding (2017) approximate RR ~ exp(0.91 * d) for a
    # standardised mean difference d, which sp.evalue implements as
    # measure='OLS' given the outcome SD.
    #
    # This used to force measure='RR' and pass the coefficient through
    # unchanged whenever it was positive, so a treatment effect of $1,548
    # was read as a risk ratio of 1548 and produced an E-value of 3096 —
    # arithmetically fine, meaningless as a quantity. Without a scale we now
    # skip the E-value and say why rather than emit that number.
    resolved_measure = measure
    ev_kwargs: Optional[Dict[str, Any]] = None
    if resolved_measure == "auto":
        resolved_measure = "RR" if _looks_like_a_ratio(result) else "OLS"
    if resolved_measure in {"RR", "OR", "HR"}:
        rr, rr_ci = float(estimate), (float(ci[0]), float(ci[1]))
        ev_kwargs = {"estimate": rr, "ci": rr_ci, "measure": resolved_measure}
    else:
        sd = _outcome_sd(result, data, y, outcome_sd)
        if sd is None:
            rr, rr_ci = float("nan"), (float("nan"), float("nan"))
            notes.append(
                "E-value skipped: the estimate is a difference on the "
                "outcome's own scale, and the E-value is defined for risk "
                "ratios. Standardising it needs the outcome SD — pass "
                "outcome_sd=, or data= together with y=. (Pass "
                "measure='RR' if the estimate really is a ratio.)"
            )
        else:
            # Report the RR the E-value was actually computed from, not the
            # raw coefficient — otherwise the summary line and the E-value
            # below it describe different quantities.
            _d = float(estimate) / sd
            _d_lo, _d_hi = float(ci[0]) / sd, float(ci[1]) / sd
            rr = float(np.exp(0.91 * _d))
            rr_ci = (float(np.exp(0.91 * _d_lo)), float(np.exp(0.91 * _d_hi)))
            ev_kwargs = {
                "estimate": float(estimate),
                "se": float(se),
                "sd": sd,
                "measure": "OLS",
            }

    try:
        if ev_kwargs is None:
            raise _SkipEValue
        ev = _evalue_fn(**ev_kwargs)
        if isinstance(ev, dict):
            e_point = _float_or_nan(
                ev.get(
                    "evalue_estimate",
                    ev.get("evalue", ev.get("e_point", float("nan"))),
                )
            )
            e_ci_val = ev.get("evalue_ci", ev.get("e_ci", None))
        else:
            e_point = _float_or_nan(
                getattr(
                    ev,
                    "evalue_estimate",
                    getattr(
                        ev,
                        "evalue",
                        getattr(ev, "e_point", float("nan")),
                    ),
                )
            )
            e_ci_val = getattr(ev, "evalue_ci", getattr(ev, "e_ci", None))
        e_ci = _finite_optional_float(e_ci_val)
    except _SkipEValue:
        e_point, e_ci = float("nan"), None
    except Exception as exc:
        notes.append(f"E-value computation failed: {exc}")
        e_point, e_ci = float("nan"), None

    # 2. Oster delta.  Requires both R^2 values AND the short-regression
    # estimate (``beta_uncontrolled``).  Fabricating beta_uncontrolled
    # would produce a meaningless delta, so we skip unless it is supplied.
    oster = None
    # Preferred path: derive Oster's inputs from the data, exactly as
    # sp.oster_delta does. Hand-supplied R^2 values are easy to get wrong —
    # `r2_treated` / `r2_controlled` read like sensemakr's partial R^2 but
    # are consumed as the short/long regression R^2, and feeding sensemakr
    # numbers here silently produced a different delta* than sp.oster_delta
    # reported on the same data (-12.765 vs -2.339). Deriving them means a
    # report cannot disagree with itself.
    if (
        include_oster
        and oster is None
        and r2_short is None
        and data is not None
        and y is not None
        and treat is not None
        and controls is not None
    ):
        try:
            from ..bounds.partial_id import oster_delta as _oster_delta

            _od = _oster_delta(
                data,
                y=y,
                x_base=[treat],
                x_controls=list(controls),
                r_max=0,
                n_boot=0,
            )
            _info = getattr(_od, "model_info", {}) or {}
            oster = {
                "delta": _float_or_nan(_info.get("delta_star", float("nan"))),
                "beta_star": _float_or_nan(_info.get("beta_star_delta1", float("nan"))),
            }
        except Exception as exc:
            notes.append(f"Oster delta from data skipped: {exc}")

    if (
        include_oster
        and oster is None
        and r2_short is not None
        and r2_long is not None
        and beta_uncontrolled is not None
    ):
        try:
            from ..diagnostics import oster_bounds as _oster_bounds

            # sp.oster_bounds signature uses beta_short / beta_long /
            # r2_short / r2_long + r_max.
            od = _oster_bounds(
                beta_short=float(beta_uncontrolled),
                beta_long=float(estimate),
                r2_short=float(r2_short),
                r2_long=float(r2_long),
                r_max=float(rho_max),
                delta=1.0,
            )
            if isinstance(od, dict):
                # ``delta_for_zero`` is Oster's breakdown delta (the
                # quantity of interest); the ``delta`` key in the return
                # dict merely echoes the *input* proportionality (1.0).
                oster = {
                    "delta": _float_or_nan(
                        od.get(
                            "delta_for_zero",
                            od.get("delta_breakdown", float("nan")),
                        )
                    ),
                    "beta_star": _float_or_nan(
                        od.get(
                            "beta_star",
                            od.get("beta_adjusted", float("nan")),
                        )
                    ),
                }
            else:
                oster = {
                    "delta": float(getattr(od, "delta", float("nan"))),
                    "beta_star": float(getattr(od, "beta_star", float("nan"))),
                }
        except Exception as exc:
            import warnings as _warnings

            _warnings.warn(f"Oster delta skipped: {exc}", stacklevel=2)
            notes.append(f"Oster delta skipped: {exc}")
    elif include_oster and oster is None:
        notes.append(
            "Oster delta skipped: supply data=, y=, treat= and controls= to "
            "derive it, or pass r2_short, r2_long and beta_uncontrolled "
            "(the short-regression estimate) explicitly."
        )

    # 3. Rosenbaum bounds — requires matched-pair outcomes. Runs when the
    #    result exposes ``matched_pairs`` (see :func:`_coerce_matched_pairs`
    #    for accepted shapes); skipped otherwise.
    rosenbaum = None
    if include_rosenbaum and getattr(result, "matched_pairs", None) is not None:
        try:
            from ..diagnostics import rosenbaum_bounds as _rb

            treated_y, control_y = _coerce_matched_pairs(result.matched_pairs)
            rb = _rb(treated_y.tolist(), control_y.tolist(), alternative="two-sided")
            rosenbaum = {"gamma_critical": float(rb.gamma_critical)}
        except Exception as exc:
            import warnings as _warnings

            msg = (
                f"Rosenbaum Gamma skipped: {exc}. Expose matched_pairs as "
                "(treated, control) outcome arrays, or call "
                "sp.rosenbaum_bounds(treated, control) directly."
            )
            _warnings.warn(msg, stacklevel=2)
            notes.append(msg)

    # 4. Sensemakr (Cinelli & Hazlett 2020). The robustness value is
    #    computed from the underlying regression, so it needs the raw
    #    estimation data — result objects do not carry it. Run only when
    #    (data, y, treat, controls) are supplied explicitly.
    sensemakr = None
    if include_sensemakr:
        sm_args = {"data": data, "y": y, "treat": treat, "controls": controls}
        if (
            data is not None
            and y is not None
            and treat is not None
            and controls is not None
        ):
            controls_list = list(controls)
            try:
                from ..diagnostics.sensemakr import sensemakr as _sm

                sm = _sm(data, y=y, treat=treat, controls=controls_list)
                sensemakr = {
                    "rv_q1": float(sm["rv_q"]),
                    "rv_qa": float(sm["rv_qa"]),
                }
            except Exception as exc:
                import warnings as _warnings

                msg = (
                    f"Sensemakr failed: {exc}. Check that data contains "
                    f"numeric columns {[y, treat] + controls_list}, or "
                    "call sp.sensemakr(data, y, treat, controls) directly."
                )
                _warnings.warn(msg, stacklevel=2)
                notes.append(msg)
        elif any(v is not None for v in sm_args.values()):
            missing = [k for k, v in sm_args.items() if v is None]
            notes.append(
                "Sensemakr skipped: missing "
                + ", ".join(missing)
                + " (all of data, y, treat, controls are required)."
            )
        else:
            notes.append(
                "Sensemakr skipped: requires the raw estimation data, "
                "which the result object does not carry. Pass data=, y=, "
                "treat=, controls= to unified_sensitivity() / "
                ".sensitivity(), or call "
                "sp.sensemakr(data, y, treat, controls) directly."
            )

    # 5. Breakdown frontier: the smallest additive bias that moves the
    #    CI bound closest to zero *through* zero, flipping
    #    the significance conclusion (Masten & Poirier 2021).  This is
    #    the bound on the CI side closer to null, not the point estimate.
    lo, hi = ci
    sign = np.sign(estimate)
    if sign > 0:
        # Lower bound: how close we are to losing significance.
        ci_near_null = lo
    elif sign < 0:
        ci_near_null = hi
    else:
        ci_near_null = 0.0
    if estimate != 0.0 and not np.isnan(ci_near_null):
        breakdown_bias = abs(ci_near_null)
        breakdown_frac = breakdown_bias / (abs(estimate) + 1e-12)
    else:
        breakdown_bias = float("nan")
        breakdown_frac = float("nan")
    breakdown = {
        "bias_to_flip": breakdown_bias,
        "fraction_of_estimate": breakdown_frac,
    }

    return SensitivityDashboard(
        e_value_point=e_point,
        e_value_ci=e_ci,
        rr_observed=rr,
        ci_observed=rr_ci,
        oster=oster,
        rosenbaum=rosenbaum,
        sensemakr=sensemakr,
        breakdown=breakdown,
        notes=notes,
    )
