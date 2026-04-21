"""
Unified sensitivity dashboard for any CausalResult / EconometricResults.

A single ``.sensitivity()`` call that runs every applicable
sensitivity analysis and returns a tidy report:

  - **E-value** (VanderWeele & Ding 2017) — always applicable
  - **Oster delta** (Oster 2019) — requires R^2 estimates
  - **Rosenbaum Gamma** (Rosenbaum 2002) — matched / IPW-weighted designs
  - **Sensemakr** (Cinelli & Hazlett 2020) — regression-based
  - **Breakdown frontier** — how much bias flips the sign

Attached as the ``sensitivity`` method of :class:`CausalResult` and
:class:`EconometricResults` via a lightweight monkey-patch in
``statspai.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


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
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        bar = "=" * 60
        lines = [
            "Unified Sensitivity Dashboard",
            bar,
            f"  Observed RR (or effect-as-RR proxy): {self.rr_observed:.4f}",
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


def _extract_estimate(result) -> tuple[float, float, tuple[float, float]]:
    """Pull (estimate, se, ci) from a heterogeneous result object."""
    estimate = None
    se = None
    ci = None
    for attr in ("estimate", "coef", "coefficient", "ate"):
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, (int, float, np.floating)):
                estimate = float(v)
                break
    if estimate is None and hasattr(result, "params"):
        try:
            p = result.params
            # pandas Series or array — take first entry
            estimate = float(p.iloc[0] if hasattr(p, "iloc") else p[0])
        except Exception:
            pass
    if hasattr(result, "se"):
        v = getattr(result, "se")
        if isinstance(v, (int, float, np.floating)):
            se = float(v)
    if se is None and hasattr(result, "std_errors"):
        try:
            s = result.std_errors
            se = float(s.iloc[0] if hasattr(s, "iloc") else s[0])
        except Exception:
            pass
    if hasattr(result, "ci"):
        v = getattr(result, "ci")
        try:
            ci = (float(v[0]), float(v[1]))
        except Exception:
            pass
    if ci is None and estimate is not None and se is not None:
        ci = (estimate - 1.96 * se, estimate + 1.96 * se)
    return estimate, se, ci


def _as_risk_ratio(estimate: float, se: float, ci) -> tuple[float, tuple[float, float]]:
    """Interpret estimate as an RR for E-value.

    If estimate > 1 already, treat as RR directly.  Otherwise use the
    conservative VanderWeele-Ding conversion
    RR = (1 + d) / 1, where ``d`` is a standardized effect size.
    """
    if estimate is None:
        return float("nan"), (float("nan"), float("nan"))
    # Assume estimate already on RR scale if strictly positive and CI > 0.
    if estimate > 0 and ci is not None and ci[0] > 0:
        return float(estimate), (float(ci[0]), float(ci[1]))
    # Otherwise convert mean difference to RR via ABS + 1
    rr = 1 + abs(estimate)
    ci_conv = (1 + abs(ci[0]) if ci else rr, 1 + abs(ci[1]) if ci else rr)
    return float(rr), (float(min(ci_conv)), float(max(ci_conv)))


# --------------------------------------------------------------------------- #
#  Main entry
# --------------------------------------------------------------------------- #


def unified_sensitivity(
    result,
    *,
    r2_treated: Optional[float] = None,
    r2_controlled: Optional[float] = None,
    beta_uncontrolled: Optional[float] = None,
    rho_max: float = 1.0,
    include_oster: bool = True,
    include_rosenbaum: bool = True,
    include_sensemakr: bool = True,
) -> SensitivityDashboard:
    """Run all applicable sensitivity analyses in one shot.

    Parameters
    ----------
    result : CausalResult / EconometricResults / dataclass with
        ``estimate``, ``se``, ``ci`` attributes.
    r2_treated, r2_controlled : float, optional
        Required for Oster's delta; R^2 from the short and long
        regression respectively.
    rho_max : float, default 1.0
        Oster's bound on the ratio of omitted-to-observed selection.

    Returns
    -------
    SensitivityDashboard
    """
    from ..diagnostics.evalue import evalue as _evalue_fn

    estimate, se, ci = _extract_estimate(result)
    if estimate is None or se is None or ci is None:
        raise ValueError(
            "Could not extract (estimate, se, ci) from result object. "
            "Supply them directly via .sensitivity(estimate=..., se=..., ci=...)."
        )

    rr, rr_ci = _as_risk_ratio(estimate, se, ci)

    notes = []
    # 1. E-value (always).  ``evalue`` returns a dict with keys
    # ``evalue_estimate`` (point) and ``evalue_ci``.
    try:
        ev = _evalue_fn(estimate=rr, ci=rr_ci, measure="RR")
        if isinstance(ev, dict):
            e_point = float(ev.get("evalue_estimate",
                                    ev.get("evalue",
                                           ev.get("e_point", float("nan")))))
            e_ci_val = ev.get("evalue_ci", ev.get("e_ci", None))
        else:
            e_point = float(getattr(ev, "evalue_estimate",
                                     getattr(ev, "evalue",
                                             getattr(ev, "e_point",
                                                     float("nan")))))
            e_ci_val = getattr(ev, "evalue_ci", getattr(ev, "e_ci", None))
        e_ci = float(e_ci_val) if e_ci_val is not None and not \
            (isinstance(e_ci_val, float) and np.isnan(e_ci_val)) else None
    except Exception as exc:
        notes.append(f"E-value computation failed: {exc}")
        e_point, e_ci = float("nan"), None

    # 2. Oster delta.  Requires both R^2 values AND the short-regression
    # estimate (``beta_uncontrolled``).  Fabricating beta_uncontrolled
    # would produce a meaningless delta, so we skip unless it is supplied.
    oster = None
    if (include_oster and r2_treated is not None
            and r2_controlled is not None
            and beta_uncontrolled is not None):
        try:
            from ..diagnostics import oster_bounds as _oster_bounds
            # sp.oster_bounds signature uses beta_short / beta_long /
            # r2_short / r2_long + r_max.
            od = _oster_bounds(
                beta_short=float(beta_uncontrolled),
                beta_long=float(estimate),
                r2_short=float(r2_controlled),
                r2_long=float(r2_treated),
                r_max=float(rho_max),
                delta=1.0,
            )
            if isinstance(od, dict):
                oster = {
                    "delta": float(od.get("delta",
                                            od.get("delta_breakdown",
                                                    float("nan")))),
                    "beta_star": float(od.get("beta_star",
                                                od.get("beta_adjusted",
                                                        float("nan")))),
                }
            else:
                oster = {
                    "delta": float(getattr(od, "delta", float("nan"))),
                    "beta_star": float(getattr(od, "beta_star",
                                                 float("nan"))),
                }
        except Exception as exc:
            import warnings as _warnings
            _warnings.warn(f"Oster delta skipped: {exc}", stacklevel=2)
            notes.append(f"Oster delta skipped: {exc}")
    elif (include_oster and (r2_treated is not None
                              or r2_controlled is not None)):
        notes.append(
            "Oster delta skipped: need r2_treated, r2_controlled, and "
            "beta_uncontrolled (the short-regression estimate)."
        )

    # 3. Rosenbaum bounds (requires matched/weighted data; we skip if
    #    the result doesn't expose a matching pair structure).
    rosenbaum = None
    if include_rosenbaum and hasattr(result, "matched_pairs") and \
            result.matched_pairs is not None:
        try:
            from ..diagnostics.rosenbaum_bounds import rosenbaum_bounds as _rb
            rb = _rb(result)
            rosenbaum = {
                "gamma_critical": float(getattr(rb, "gamma_critical",
                                                 float("nan"))),
            }
        except Exception as exc:
            notes.append(f"Rosenbaum Gamma skipped: {exc}")

    # 4. Sensemakr (regression) — best-effort if a regression is exposed.
    sensemakr = None
    if include_sensemakr and hasattr(result, "params") and \
            hasattr(result, "std_errors"):
        try:
            from ..diagnostics.sensemakr import sensemakr as _sm
            # pull first coefficient name from params
            name = list(result.params.index)[0]
            sm = _sm(result, treatment=name)
            sensemakr = {"rv_q1": float(getattr(sm, "rv_q1", float("nan")))}
        except Exception as exc:
            notes.append(f"Sensemakr skipped: {exc}")

    # 5. Breakdown frontier: the smallest additive bias that moves the
    #    CI bound closest to zero *through* zero, flipping
    #    the significance conclusion (Masten & Poirier 2021).  This is
    #    the bound on the CI side closer to null, not the point estimate.
    lo, hi = ci
    sign = np.sign(estimate)
    if sign > 0:
        ci_near_null = lo  # lower bound: how close we are to losing significance
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
