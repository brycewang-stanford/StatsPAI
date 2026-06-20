"""
Decision-oriented summaries for StatsPAI result objects.

This module hosts the logic behind ``CausalResult.decision_summary()``.
It answers the question a statistical-significance test cannot: *is the
estimated effect large enough to matter?*

The design is a deliberate **borrow-and-adapt** from PyMC-Labs' CausalPy,
which pairs every effect with a *Region of Practical Equivalence* (ROPE)
and returns an ``EffectSummary`` exposing both a machine-readable
``.table`` and a human-readable ``.text``.  Two adaptations make it fit
StatsPAI's frequentist-first, Stata/R-aligned world:

1. **CI-vs-ROPE, not posterior mass.**  CausalPy computes the posterior
   probability inside the ROPE.  Most StatsPAI estimators are frequentist,
   so we classify the *confidence interval* against the ROPE.  Comparing
   the whole CI to a ``(-r, +r)`` band is exactly the two-one-sided-tests
   (TOST) equivalence logic of Lakens (2018): a CI inside the ROPE
   establishes practical equivalence; a CI beyond it establishes a
   meaningful effect; a CI straddling a boundary is inconclusive.

2. **Honest abstention.**  Practical significance is domain-specific, so
   the ROPE must be supplied by the analyst (the *smallest effect size of
   interest*, SESOI).  When it is absent we report statistical
   significance only and say so explicitly — never inventing a threshold,
   and never letting "not significant" masquerade as "no effect".

References
----------
Lakens, D. (2018). Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses. *Social Psychological and Personality
Science*, 8(4), 355-362.

Kruschke, J. K. (2018). Rejecting or accepting parameter values in
Bayesian estimation. *Advances in Methods and Practices in Psychological
Science*, 1(2), 270-280.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Verdict code -> short human label.  Codes are stable identifiers an
# agent can branch on; labels are for display.
_LABELS: Dict[str, str] = {
    "meaningful_effect": "Significant & practically meaningful",
    "negligible_significant": "Significant but practically negligible",
    "equivalent": "Practically equivalent to zero",
    "significant_uncertain_magnitude": "Significant; magnitude uncertain vs ROPE",
    "inconclusive": "Inconclusive (underpowered)",
    "statistically_significant": "Statistically significant (ROPE not set)",
    "not_significant": "Not significant (ROPE not set)",
    "indeterminate_ci": "Indeterminate (no usable interval)",
    "undefined": "Undefined (non-finite estimate)",
}

# Tri-state practical-significance verdict per code.  ``None`` means
# "cannot be determined", which is itself an honest, actionable answer.
_PRACTICAL: Dict[str, Optional[bool]] = {
    "meaningful_effect": True,
    "negligible_significant": False,
    "equivalent": False,
    "significant_uncertain_magnitude": None,
    "inconclusive": None,
    "statistically_significant": None,
    "not_significant": None,
    "indeterminate_ci": None,
    "undefined": None,
}


def _fmt(x: Optional[float]) -> str:
    """Compact, sign-preserving number format for prose."""
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.4g}"


def _as_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _coerce_rope(rope: Any, sesoi: Optional[float]) -> Optional[Tuple[float, float]]:
    """Normalise ``rope`` / ``sesoi`` into an ordered ``(lo, hi)`` band.

    ``rope`` may be a positive scalar ``r`` (band ``(-r, r)``) or an
    explicit ``(lo, hi)`` pair.  ``sesoi`` is a convenience alias for the
    scalar form (the *smallest effect size of interest*).  Returns
    ``None`` when neither is supplied.
    """
    if rope is not None and sesoi is not None:
        raise ValueError(
            "Pass either rope= or sesoi=, not both (sesoi is the scalar "
            "form of rope)."
        )
    if rope is None and sesoi is None:
        return None

    if sesoi is not None:
        r = _as_float(sesoi)
        if r is None or r <= 0:
            raise ValueError(f"sesoi must be a positive number; got {sesoi!r}.")
        return (-r, r)

    # rope path
    if np.isscalar(rope):
        r = _as_float(rope)
        if r is None or r <= 0:
            raise ValueError(
                f"Scalar rope must be a positive number; got {rope!r}. "
                "Use a (lo, hi) tuple for an asymmetric region."
            )
        return (-r, r)

    try:
        lo, hi = rope  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(
            "rope must be a positive scalar r (meaning (-r, r)) or a "
            f"(lo, hi) tuple; got {rope!r}."
        )
    lo_f, hi_f = _as_float(lo), _as_float(hi)
    if lo_f is None or hi_f is None or not lo_f < hi_f:
        raise ValueError(f"rope bounds must be finite with lo < hi; got {rope!r}.")
    return (lo_f, hi_f)


def _classify(
    ci: Optional[Tuple[float, float]],
    rope: Optional[Tuple[float, float]],
    stat_sig: bool,
) -> Tuple[str, Optional[str]]:
    """Return ``(verdict_code, ci_vs_rope)``.

    ``ci_vs_rope`` is ``"outside"`` / ``"inside"`` / ``"overlap"`` or
    ``None`` when not assessable.
    """
    if rope is None:
        return (
            "statistically_significant" if stat_sig else "not_significant",
            None,
        )
    if ci is None:
        return "indeterminate_ci", None

    rlo, rhi = rope
    lo, hi = ci
    inside = (lo >= rlo) and (hi <= rhi)
    outside = (lo > rhi) or (hi < rlo)

    if outside:
        rel = "outside"
        code = "meaningful_effect"
    elif inside:
        rel = "inside"
        code = "negligible_significant" if stat_sig else "equivalent"
    else:
        rel = "overlap"
        code = "significant_uncertain_magnitude" if stat_sig else "inconclusive"
    return code, rel


def _verdict_sentence(code: str, rope: Optional[Tuple[float, float]]) -> str:
    """Prose clause explaining the ROPE verdict (no leading capital)."""
    if rope is not None:
        band = f"({_fmt(rope[0])}, {_fmt(rope[1])})"
    else:
        band = ""
    return {
        "meaningful_effect": (
            f"and lies entirely beyond the region of practical equivalence "
            f"{band}, so the effect is large enough to matter."
        ),
        "negligible_significant": (
            f"but falls entirely within the ROPE {band} — a precisely "
            "estimated yet practically negligible effect (common in large "
            "samples)."
        ),
        "equivalent": (
            f"and falls entirely within the ROPE {band}; the data are "
            "consistent with a practically negligible effect (equivalence "
            "established)."
        ),
        "significant_uncertain_magnitude": (
            f"but straddles a ROPE boundary {band}; the effect is non-zero "
            "yet could be either negligible or meaningful — more precision "
            "is needed."
        ),
        "inconclusive": (
            f"and straddles both zero and the ROPE {band}; the study cannot "
            "rule out either a null or a practically meaningful effect "
            "(likely underpowered)."
        ),
        "statistically_significant": (
            "Practical significance was not assessed — pass rope= (the "
            "smallest effect size of interest) to judge whether the "
            "magnitude matters."
        ),
        "not_significant": (
            "Practical significance was not assessed; absent a ROPE this is "
            "not evidence of no effect (the study may be underpowered) — "
            "pass rope= to run an equivalence test."
        ),
        "indeterminate_ci": (
            "No usable confidence interval is available, so the magnitude "
            "cannot be weighed against the ROPE."
        ),
        "undefined": "the estimate is not finite, so no decision can be made.",
    }.get(code, "")


class DecisionSummary:
    """Result of :meth:`CausalResult.decision_summary`.

    Mirrors CausalPy's ``EffectSummary``: a single object exposing a
    machine-readable :attr:`table` (one-row :class:`pandas.DataFrame`) and
    a human-readable :attr:`text`.  ``str(summary)`` prints the prose.
    """

    __slots__ = (
        "verdict",
        "label",
        "text",
        "table",
        "statistically_significant",
        "practically_significant",
        "ci_vs_rope",
        "estimate",
        "ci",
        "rope",
        "alpha",
    )

    def __init__(
        self,
        *,
        verdict: str,
        label: str,
        text: str,
        table: pd.DataFrame,
        statistically_significant: Optional[bool],
        practically_significant: Optional[bool],
        ci_vs_rope: Optional[str],
        estimate: Optional[float],
        ci: Optional[Tuple[float, float]],
        rope: Optional[Tuple[float, float]],
        alpha: float,
    ) -> None:
        self.verdict = verdict
        self.label = label
        self.text = text
        self.table = table
        self.statistically_significant = statistically_significant
        self.practically_significant = practically_significant
        self.ci_vs_rope = ci_vs_rope
        self.estimate = estimate
        self.ci = ci
        self.rope = rope
        self.alpha = alpha

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready payload for agent / MCP consumption."""
        return {
            "verdict": self.verdict,
            "label": self.label,
            "statistically_significant": self.statistically_significant,
            "practically_significant": self.practically_significant,
            "ci_vs_rope": self.ci_vs_rope,
            "estimate": self.estimate,
            "ci": list(self.ci) if self.ci is not None else None,
            "rope": list(self.rope) if self.rope is not None else None,
            "alpha": self.alpha,
            "text": self.text,
        }

    def __str__(self) -> str:  # pragma: no cover - thin
        return self.text

    def __repr__(self) -> str:  # pragma: no cover - thin
        return f"DecisionSummary(verdict={self.verdict!r}, {self.label!r})"


def decision_summary(
    result: Any,
    *,
    rope: Any = None,
    sesoi: Optional[float] = None,
    alpha: Optional[float] = None,
) -> DecisionSummary:
    """Judge an effect's statistical *and* practical significance.

    Parameters
    ----------
    result : CausalResult
        Any StatsPAI causal result (DID / RD / SCM / IV / matching …).
    rope : float or (float, float), optional
        Region of practical equivalence — the band of effect sizes too
        small to matter.  A positive scalar ``r`` means ``(-r, r)``; a
        tuple specifies an asymmetric band.  Should normally bracket
        zero.  When omitted, only statistical significance is reported.
    sesoi : float, optional
        Smallest effect size of interest; convenience alias for
        ``rope=sesoi`` (the symmetric band ``(-sesoi, sesoi)``).  Mutually
        exclusive with ``rope``.
    alpha : float, optional
        Significance level for the interval.  Defaults to the level the
        estimator used (``result.alpha``).  Supplying a different value
        recomputes a normal-approximation CI from the standard error.

    Returns
    -------
    DecisionSummary
        Object with ``.table`` (machine-readable) and ``.text`` (prose).

    Notes
    -----
    The CI-vs-ROPE classification is the frequentist equivalence-testing
    logic of Lakens (2018): a CI fully inside the ROPE establishes
    practical equivalence; a CI fully beyond it establishes a meaningful
    effect; a CI crossing a boundary is inconclusive.

    Examples
    --------
    >>> r = sp.did(df, y='wage', treat='treated', time='post')
    >>> print(r.decision_summary(rope=0.5).text)
    """
    rope_band = _coerce_rope(rope, sesoi)

    method = getattr(result, "method", "") or "Estimator"
    estimand = getattr(result, "estimand", None) or "effect"
    est = _as_float(getattr(result, "estimate", None))
    se = _as_float(getattr(result, "se", None))
    pval = _as_float(getattr(result, "pvalue", None))

    eff_alpha = alpha if alpha is not None else getattr(result, "alpha", None)
    if eff_alpha is None:
        eff_alpha = 0.05
    if not 0 < eff_alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {eff_alpha!r}.")

    # --- Resolve the interval -------------------------------------------
    ci: Optional[Tuple[float, float]] = None
    ci_method = "native"
    native_ci = getattr(result, "ci", None)
    native_alpha = getattr(result, "alpha", None)
    want_recompute = alpha is not None and (
        native_alpha is None or not np.isclose(alpha, native_alpha)
    )

    if not want_recompute and native_ci is not None and len(native_ci) == 2:
        lo, hi = _as_float(native_ci[0]), _as_float(native_ci[1])
        if lo is not None and hi is not None:
            ci = (lo, hi)
    if ci is None and est is not None and se is not None and se > 0:
        # Normal-approximation interval at eff_alpha.
        from scipy.stats import norm

        z = float(norm.ppf(1 - eff_alpha / 2))
        ci = (est - z * se, est + z * se)
        ci_method = "normal-approx"

    # --- Statistical significance (CI-based, pvalue fallback) -----------
    if ci is not None:
        stat_sig: Optional[bool] = (ci[0] > 0) or (ci[1] < 0)
    elif pval is not None:
        stat_sig = pval < eff_alpha
    else:
        stat_sig = None

    # --- Classify --------------------------------------------------------
    if est is None:
        code, rel = "undefined", None
    else:
        code, rel = _classify(ci, rope_band, bool(stat_sig))
    label = _LABELS.get(code, code)
    practical = _PRACTICAL.get(code)

    # --- Prose -----------------------------------------------------------
    conf = round((1 - eff_alpha) * 100, 4)
    conf_str = f"{conf:g}"
    text = _compose_text(
        result=result,
        method=method,
        estimand=estimand,
        est=est,
        se=se,
        pval=pval,
        ci=ci,
        ci_method=ci_method,
        conf_str=conf_str,
        stat_sig=stat_sig,
        code=code,
        rope_band=rope_band,
        label=label,
    )

    # --- Machine-readable one-row table ---------------------------------
    table = pd.DataFrame(
        [
            {
                "method": method,
                "estimand": estimand,
                "estimate": est,
                "se": se,
                "ci_low": ci[0] if ci else None,
                "ci_high": ci[1] if ci else None,
                "conf_level": conf,
                "pvalue": pval,
                "statistically_significant": stat_sig,
                "rope_low": rope_band[0] if rope_band else None,
                "rope_high": rope_band[1] if rope_band else None,
                "ci_vs_rope": rel,
                "practically_significant": practical,
                "verdict": code,
                "label": label,
            }
        ]
    )

    return DecisionSummary(
        verdict=code,
        label=label,
        text=text,
        table=table,
        statistically_significant=stat_sig,
        practically_significant=practical,
        ci_vs_rope=rel,
        estimate=est,
        ci=ci,
        rope=rope_band,
        alpha=eff_alpha,
    )


def _compose_text(
    *,
    result: Any,
    method: str,
    estimand: str,
    est: Optional[float],
    se: Optional[float],
    pval: Optional[float],
    ci: Optional[Tuple[float, float]],
    ci_method: str,
    conf_str: str,
    stat_sig: Optional[bool],
    code: str,
    rope_band: Optional[Tuple[float, float]],
    label: str,
) -> str:
    """Assemble the human-readable verdict paragraph."""
    if est is None:
        return (
            f"{method}: the {estimand} estimate is not finite, so no "
            "decision can be made. Inspect the data for collinearity or "
            "zero variance (see result.violations())."
        )

    # Sentence 1: the numbers.
    parts: List[str] = [f"{method} estimates the {estimand} at {_fmt(est)}"]
    qual: List[str] = []
    if se is not None:
        qual.append(f"SE {_fmt(se)}")
    if ci is not None:
        approx = " normal-approx" if ci_method == "normal-approx" else ""
        qual.append(f"{conf_str}%{approx} CI [{_fmt(ci[0])}, {_fmt(ci[1])}]")
    if pval is not None:
        qual.append(f"p = {_fmt(pval)}")
    if qual:
        parts[0] += " (" + ", ".join(qual) + ")"
    parts[0] += "."

    # Sentence 2: statistical significance + ROPE verdict, fused.
    rope_clause = _verdict_sentence(code, rope_band)
    if rope_band is None:
        if stat_sig is True:
            parts.append(
                "The interval excludes zero, so the effect is "
                "statistically distinguishable from no effect."
            )
        elif stat_sig is False:
            parts.append(
                "The interval includes zero, so the effect is not "
                "statistically distinguishable from no effect."
            )
        parts.append(rope_clause)
    elif code == "indeterminate_ci":
        parts.append(rope_clause)
    else:
        if stat_sig is True:
            lead = "The interval excludes zero"
        elif stat_sig is False:
            lead = "The interval includes zero"
        else:
            lead = "The interval"
        parts.append(f"{lead} {rope_clause}")

    text = " ".join(p for p in parts if p)

    # Tie the verdict back to assumption diagnostics (CausalPy's
    # "diagnostics-by-design" ethos): a meaningful effect resting on a
    # violated assumption deserves a flag.
    errs = [v for v in (result.violations() or []) if v.get("severity") == "error"]
    if errs:
        tests = ", ".join(sorted({v.get("test", "?") for v in errs}))
        text += (
            f" Caution: {len(errs)} assumption check(s) flagged as errors "
            f"({tests}) — see result.violations()."
        )

    return text
