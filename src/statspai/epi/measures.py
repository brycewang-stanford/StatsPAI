"""
Epidemiological measures of association.

Core 2x2-table quantities used throughout epidemiology, biostatistics, and
public health research.  These primitives are the foundation on which the
rest of the epidemiology module is built.

Implemented
-----------
- :func:`odds_ratio` — OR with Woolf / exact CIs
- :func:`relative_risk` — RR (risk ratio) with log-binomial CI
- :func:`risk_difference` — RD with Wald / Newcombe CI
- :func:`attributable_risk` — population attributable fraction (Levin)
- :func:`incidence_rate_ratio` — person-time IRR with exact Poisson CI
- :func:`number_needed_to_treat` — NNT with CI propagation
- :func:`prevalence_ratio` — cross-sectional PR (Zou 2004 log-binomial)

References
----------
Rothman, K.J., Greenland, S. & Lash, T.L. (2008). *Modern Epidemiology*,
3rd ed. Lippincott Williams & Wilkins.

Woolf, B. (1955). "On estimating the relation between blood group and
disease." *Annals of Human Genetics*, 19(4), 251-253. [@woolf1955estimating]

Zou, G. (2004). "A modified Poisson regression approach to prospective
studies with binary data." *American Journal of Epidemiology*, 159(7),
702-706. [@zou2004modified]
"""

from __future__ import annotations

import inspect as _inspect
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = [
    "OR2x2Result",
    "RR2x2Result",
    "RD2x2Result",
    "ARResult",
    "IRRResult",
    "NNTResult",
    "odds_ratio",
    "relative_risk",
    "risk_difference",
    "attributable_risk",
    "incidence_rate_ratio",
    "number_needed_to_treat",
    "prevalence_ratio",
]


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _coerce_2x2(
    a: Any,
    b: Any = None,
    c: Any = None,
    d: Any = None,
) -> tuple[float, float, float, float]:
    """Accept either ``(a, b, c, d)`` counts or a 2x2 matrix.

    Returns floats to allow continuity corrections downstream without
    integer truncation issues.
    """
    if b is None:
        try:
            arr = np.asarray(a, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "2x2 table must contain numeric counts.",
                recovery_hint=(
                    "Pass numeric cell counts as a 2x2 array or as " "a, b, c, d."
                ),
                diagnostics={"input_type": type(a).__name__},
            ) from exc
        if arr.shape != (2, 2):
            raise MethodIncompatibility(
                "2x2 table expected; got shape %s" % (arr.shape,),
                recovery_hint=(
                    "Pass rows [exposed, unexposed] and columns " "[event, non-event]."
                ),
                diagnostics={"shape": arr.shape},
            )
        (a, b), (c, d) = arr
    elif c is None or d is None:
        raise MethodIncompatibility(
            "Pass either a full 2x2 table or all four cell counts.",
            recovery_hint=(
                "Call with odds_ratio([[a, b], [c, d]]) or " "odds_ratio(a, b, c, d)."
            ),
            diagnostics={
                "missing": [
                    name for name, value in {"c": c, "d": d}.items() if value is None
                ]
            },
        )
    try:
        counts = np.asarray([a, b, c, d], dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "Counts must be numeric.",
            recovery_hint="Pass finite numeric cell counts.",
            diagnostics={"values": [repr(x) for x in (a, b, c, d)]},
        ) from exc
    if not np.all(np.isfinite(counts)):
        raise MethodIncompatibility(
            "Counts must be finite.",
            recovery_hint=(
                "Drop or impute non-finite counts before computing "
                "epidemiology measures."
            ),
            diagnostics={"values": counts.tolist()},
        )
    if np.any(counts < 0):
        raise MethodIncompatibility(
            "Counts must be non-negative.",
            recovery_hint=(
                "Check the 2x2 table construction; event and non-event "
                "counts cannot be negative."
            ),
            diagnostics={"values": counts.tolist()},
        )
    return (
        float(counts[0]),
        float(counts[1]),
        float(counts[2]),
        float(counts[3]),
    )


def _haldane_correction(
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[float, float, float, float]:
    """Haldane-Anscombe 0.5 continuity correction when any cell is zero."""
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return a, b, c, d


def _z_crit(alpha: float) -> float:
    try:
        alpha_float = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "alpha must be a finite probability in (0, 1).",
            recovery_hint="Use a significance level such as alpha=0.05.",
            diagnostics={"alpha": repr(alpha)},
        ) from exc
    if not np.isfinite(alpha_float) or not 0 < alpha_float < 1:
        raise MethodIncompatibility(
            "alpha must be a finite probability in (0, 1).",
            recovery_hint="Use a significance level such as alpha=0.05.",
            diagnostics={"alpha": alpha_float},
        )
    return float(stats.norm.ppf(1 - alpha_float / 2))


def _ci_label(level: float) -> str:
    """'95% CI' style label for a confidence level in (0, 1)."""
    return f"{100 * level:g}% CI"


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class OR2x2Result(ResultProtocolMixin):
    """Result of a 2x2 odds-ratio calculation."""

    estimate: float
    se_log: float
    ci: tuple[float, float]
    p_value: float
    a: float
    b: float
    c: float
    d: float
    method: str
    # Conditional maximum-likelihood OR (the estimate R ``fisher.test``
    # reports); set for ``method="exact"`` only.
    conditional_mle: Optional[float] = None

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.ci
        text = (
            f"Odds Ratio ({self.method})\n"
            f"  OR = {self.estimate:.4f}"
            f"   {_ci_label(self.level)} [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
        )
        if self.conditional_mle is not None:
            text += f"  conditional MLE = {self.conditional_mle:.4f}\n"
        return text + (
            f"  2x2 table: [[{self.a:.0f}, {self.b:.0f}], "
            f"[{self.c:.0f}, {self.d:.0f}]]"
        )


@dataclass
class RR2x2Result(ResultProtocolMixin):
    """Result of a 2x2 relative-risk (risk-ratio) calculation."""

    estimate: float
    se_log: float
    ci: tuple[float, float]
    p_value: float
    risk_exposed: float
    risk_unexposed: float
    method: str

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Relative Risk ({self.method})\n"
            f"  RR = {self.estimate:.4f}"
            f"   {_ci_label(self.level)} [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  Risk (exposed)   = {self.risk_exposed:.4f}\n"
            f"  Risk (unexposed) = {self.risk_unexposed:.4f}"
        )


@dataclass
class RD2x2Result(ResultProtocolMixin):
    """Result of a 2x2 risk-difference calculation."""

    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    risk_exposed: float
    risk_unexposed: float
    method: str

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Risk Difference ({self.method})\n"
            f"  RD = {self.estimate:+.4f}"
            f"   {_ci_label(self.level)} [{lo:+.4f}, {hi:+.4f}]   "
            f"p = {self.p_value:.4g}"
        )


@dataclass
class ARResult(ResultProtocolMixin):
    """Attributable-risk quantities (Levin 1953, Miettinen 1974)."""

    # AR% among the exposed (a.k.a. attributable fraction in exposed)
    ar_exposed: float
    paf: float  # Population attributable fraction (Levin)
    paf_ci: tuple[float, float]
    prevalence_exposed: float
    rr: float

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.paf_ci
        return (
            "Attributable Risk\n"
            f"  AF (exposed) = {self.ar_exposed:.4f}\n"
            f"  PAF (Levin)  = {self.paf:.4f}"
            f"   {_ci_label(self.level)} [{lo:.4f}, {hi:.4f}]\n"
            f"  Prevalence exposed = {self.prevalence_exposed:.4f}\n"
            f"  RR = {self.rr:.4f}"
        )


@dataclass
class IRRResult(ResultProtocolMixin):
    """Incidence rate ratio from person-time data."""

    estimate: float
    ci: tuple[float, float]
    p_value: float
    rate_exposed: float
    rate_unexposed: float
    events_exposed: float
    events_unexposed: float
    pt_exposed: float
    pt_unexposed: float
    method: str

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Incidence Rate Ratio ({self.method})\n"
            f"  IRR = {self.estimate:.4f}"
            f"   {_ci_label(self.level)} [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  Rate (exposed)   = {self.rate_exposed:.6f}  "
            f"({self.events_exposed:.0f}/{self.pt_exposed:.1f} person-time)\n"
            f"  Rate (unexposed) = {self.rate_unexposed:.6f}  "
            f"({self.events_unexposed:.0f}/"
            f"{self.pt_unexposed:.1f} person-time)"
        )


@dataclass
class NNTResult(ResultProtocolMixin):
    """Number needed to treat (benefit) or harm.

    ``estimate`` is ``1 / |RD|`` and ``ci`` the interval for that
    magnitude. ``ci_signed`` holds the reciprocals ``1 / RD`` of the
    risk-difference limits in ascending order (what R ``epiR`` reports).
    When the risk-difference interval contains zero, ``ci_disjoint`` is
    True: the compatible NNTs are ``1 / |L|`` or more on the benefit side
    and ``1 / U`` or more on the harm side (Altman's "NNTB x to infinity
    to NNTH y"), ``ci_signed`` holds ``(1/L, 1/U)`` and ``ci`` is
    ``(1 / max(|L|, U), inf)``.
    """

    estimate: float
    ci: tuple[float, float]
    risk_difference: float
    ci_signed: tuple[float, float] = (float("nan"), float("nan"))
    ci_disjoint: bool = False

    # Confidence level of ``ci`` (1 - alpha), used for the summary label.
    level: float = 0.95

    def summary(self) -> str:
        lo, hi = self.ci
        # NNT is reported as NNT-Benefit or NNT-Harm; RD sign drives it.
        # Negative RD (lower risk in the exposed / treated row) is benefit.
        tag = "NNT Benefit" if self.risk_difference < 0 else "NNT Harm"
        head = f"{tag}\n  NNT = {self.estimate:.2f}"
        if not self.ci_disjoint:
            return head + f"   {_ci_label(self.level)} [{lo:.2f}, {hi:.2f}]"
        s_lo, s_hi = self.ci_signed
        # s_lo = 1/L <= 0 (benefit side), s_hi = 1/U >= 0 (harm side).
        sides = [("NNTB", abs(s_lo)), ("NNTH", abs(s_hi))]
        if self.risk_difference > 0:
            sides.reverse()
        (t1, v1), (t2, v2) = sides
        return head + (
            f"   {_ci_label(self.level)} {t1} {v1:.2f} to infinity to {t2} {v2:.2f}"
            "  (risk-difference interval includes 0)"
        )


# --------------------------------------------------------------------------- #
#  Odds ratio
# --------------------------------------------------------------------------- #


def odds_ratio(
    a: Any,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    method: str = "woolf",
    alpha: float = 0.05,
    correction: Optional[str] = None,
) -> OR2x2Result:
    """Odds ratio from a 2x2 table.

    The standard epidemiology 2x2 layout is::

                        Outcome+   Outcome-
        Exposed            a          b
        Unexposed          c          d

    Parameters
    ----------
    a, b, c, d : float
        Cell counts, or pass a 2x2 array-like as ``a``.
    method : {"woolf", "exact"}, default "woolf"
        Inference method.  "woolf" uses the asymptotic log-OR standard
        error and the Wald test; "exact" uses the exact conditional
        (noncentral hypergeometric) interval and Fisher's exact test.
        The point estimate ``estimate`` is the sample OR ``ad / (bc)``
        under both methods, as Stata ``cc`` / ``cci``.
    alpha : float, default 0.05
    correction : {None, "haldane"}, default None
        Only used by ``method="woolf"``. ``None``: with a zero cell the
        estimate is the plain ratio (0 or ``inf``; NaN for 0/0) and the
        Woolf interval and Wald p-value are NaN, with a warning, as Stata
        ``cci, woolf`` reports no interval there. ``"haldane"``: add 0.5
        to every cell when any cell is zero and report that ratio and its
        finite interval.

        .. versionchanged:: 1.42
           The Haldane correction used to be applied by default, so a
           zero cell moved the point estimate (``0, 10, 5, 20`` gave
           0.1774 instead of 0).

    Returns
    -------
    OR2x2Result
        With ``method="exact"``, ``conditional_mle`` also holds the
        conditional maximum-likelihood OR, the estimate R
        ``fisher.test`` reports and the one the exact interval is built
        around.

    Notes
    -----
    The exact limits solve the two tail equations of the noncentral
    hypergeometric distribution to machine precision. R ``fisher.test``
    and Stata ``cci, exact`` solve the same equations with looser root
    finders, so their limits differ from these in the fourth or fifth
    significant digit (R up to 1.2e-3 relative, Stata up to 2.2e-4 on the
    tables in the tests); R's conditional MLE differs by about 1e-5.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.odds_ratio(50, 20, 30, 40)
    >>> round(res.estimate, 3)
    3.333
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    if method not in ("woolf", "exact"):
        raise MethodIncompatibility(
            "method must be 'woolf' or 'exact'",
            recovery_hint=(
                "Use method='woolf' for asymptotic inference or "
                "method='exact' for Fisher-style inference."
            ),
            diagnostics={"method": method},
        )

    if correction not in (None, "haldane"):
        raise MethodIncompatibility(
            f"correction must be None or 'haldane', got {correction!r}.",
            recovery_hint="Omit correction= for the plain ratio (Stata cci) or "
            "pass correction='haldane' to add 0.5 to every cell when one is "
            "zero.",
            diagnostics={"correction": correction},
        )
    has_zero = min(a, b, c, d) == 0
    if method == "woolf" and correction is None and has_zero:
        # Plain ratio, as Stata cci: log OR is infinite, so there is no
        # Woolf interval or Wald test.
        num, den = a * d, b * c
        if den > 0:
            or_plain = num / den
        else:
            or_plain = float("inf") if num > 0 else float("nan")
        warnings.warn(
            "odds_ratio: no Woolf interval or Wald p-value with a zero cell "
            f"({a:g}, {b:g}, {c:g}, {d:g}); ci and p_value are NaN. Pass "
            "correction='haldane' for the continuity-corrected ratio and "
            "interval, or method='exact'.",
            UserWarning,
            stacklevel=2,
        )
        nan = float("nan")
        return OR2x2Result(
            estimate=float(or_plain),
            se_log=nan,
            ci=(nan, nan),
            p_value=nan,
            a=a,
            b=b,
            c=c,
            d=d,
            method=method,
            level=1 - alpha,
        )

    # Haldane correction for the log-OR SE (the exact path's Woolf fallback
    # and the reported se_log need a finite value).
    a_c, b_c, c_c, d_c = _haldane_correction(a, b, c, d)
    or_point = (a_c * d_c) / (b_c * c_c)
    log_or = np.log(or_point)
    se_log = float(np.sqrt(1 / a_c + 1 / b_c + 1 / c_c + 1 / d_c))

    cmle: Optional[float] = None
    if method == "woolf":
        z = _z_crit(alpha)
        ci = (float(np.exp(log_or - z * se_log)), float(np.exp(log_or + z * se_log)))
        # Two-sided chi-square test for H0: OR = 1
        z_stat = log_or / se_log if se_log > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(z_stat)))
    else:
        # Fisher's exact returns the sample (unconditional MLE) OR and the
        # exact p-value.
        table = np.array([[a, b], [c, d]], dtype=int)
        try:
            res = stats.fisher_exact(table, alternative="two-sided")
            # scipy returns (odds_ratio, p_value)
            if hasattr(res, "statistic"):
                or_exact = float(res.statistic)
                p = float(res.pvalue)
            else:
                or_exact = float(res[0])
                p = float(res[1])
            # Use scipy.stats.contingency.odds_ratio for exact CI if available
            try:
                from scipy.stats.contingency import odds_ratio as _scipy_or

                obj = _scipy_or(table, kind="conditional")
                ci_lo, ci_hi = obj.confidence_interval(confidence_level=1 - alpha)
                ci = (float(ci_lo), float(ci_hi))
                cmle = float(obj.statistic)
                or_point = or_exact
            except Exception as exc:
                from ..core._fallback import warn_fallback

                warn_fallback(
                    "exact conditional confidence interval for the odds ratio",
                    exc,
                    "reporting the Woolf (log-normal) interval instead",
                )
                # Fall back to Woolf CI.
                z = _z_crit(alpha)
                ci = (
                    float(np.exp(log_or - z * se_log)),
                    float(np.exp(log_or + z * se_log)),
                )
        except Exception as exc:  # pragma: no cover
            raise NumericalInstability(
                f"scipy.stats.fisher_exact failed: {exc}",
                recovery_hint=(
                    "Use method='woolf' or check that the 2x2 table is " "well formed."
                ),
                diagnostics={"table": table.tolist()},
            ) from exc

    _result = OR2x2Result(
        estimate=float(or_point),
        se_log=se_log,
        ci=ci,
        p_value=p,
        a=a,
        b=b,
        c=c,
        d=d,
        method=method,
        conditional_mle=cmle,
        level=1 - alpha,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.epi.odds_ratio",
            params={"method": method, "alpha": alpha, "correction": correction},
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# --------------------------------------------------------------------------- #
#  Relative risk
# --------------------------------------------------------------------------- #


def relative_risk(
    a: Any,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    alpha: float = 0.05,
    correction: Optional[str] = None,
) -> RR2x2Result:
    """Relative risk (risk ratio) with Katz log-RR confidence interval.

    Parameters
    ----------
    a, b, c, d : float
        Cell counts (exposed cases, exposed non-cases, unexposed cases,
        unexposed non-cases), or pass a 2x2 array-like as ``a``.
    alpha : float, default 0.05
    correction : {None, "haldane"}, default None
        ``None``: the plain ratio ``(a / (a + b)) / (c / (c + d))``, as
        Stata ``cs`` / ``csi``. A zero ``a`` gives 0 and a zero ``c`` gives
        ``inf``; the Katz interval and the Wald p-value need ``a > 0`` and
        ``c > 0`` and are NaN otherwise, with a warning (Stata reports no
        interval there either). ``"haldane"``: add 0.5 to every cell when
        any cell is zero (Haldane-Anscombe) and report that ratio and its
        finite interval.

        .. versionchanged:: 1.42
           The Haldane correction used to be applied by default, so a
           zero cell moved the point estimate (``0, 80, 20, 80`` gave
           0.0301 instead of 0).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.relative_risk(40, 60, 20, 80)
    >>> round(res.estimate, 3)
    2.0
    >>> round(res.risk_exposed, 3), round(res.risk_unexposed, 3)
    (0.4, 0.2)
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        raise DataInsufficient(
            "Empty exposure row in 2x2 table.",
            recovery_hint=("Each exposure row must contain at least one observation."),
            diagnostics={"n_exposed": n1, "n_unexposed": n0},
        )

    if correction not in (None, "haldane"):
        raise MethodIncompatibility(
            f"correction must be None or 'haldane', got {correction!r}.",
            recovery_hint="Omit correction= for the plain ratio (Stata cs) or "
            "pass correction='haldane' to add 0.5 to every cell when one is "
            "zero.",
            diagnostics={"correction": correction},
        )
    z = _z_crit(alpha)
    if correction is None and (a == 0 or c == 0):
        # Plain ratio, as Stata cs: 0 when a = 0, inf when c = 0 (Stata
        # prints '.'). log RR is infinite, so there is no Katz interval.
        rr = 0.0 if a == 0 and c > 0 else (float("inf") if a > 0 else float("nan"))
        warnings.warn(
            "relative_risk: no Katz interval or Wald p-value with a zero "
            f"case count (a={a:g}, c={c:g}); ci and p_value are NaN. Pass "
            "correction='haldane' for the continuity-corrected ratio and "
            "interval.",
            UserWarning,
            stacklevel=2,
        )
        nan = float("nan")
        return RR2x2Result(
            estimate=float(rr),
            se_log=nan,
            ci=(nan, nan),
            p_value=nan,
            risk_exposed=float(a / n1),
            risk_unexposed=float(c / n0),
            method="katz-log",
            level=1 - alpha,
        )

    if correction == "haldane":
        a_c, b_c, c_c, d_c = _haldane_correction(a, b, c, d)
    else:
        # A zero b or d leaves the ratio and the Katz variance finite; Stata
        # cs reports both on the raw counts.
        a_c, b_c, c_c, d_c = a, b, c, d
    p1 = a_c / (a_c + b_c)
    p0 = c_c / (c_c + d_c)
    rr = p1 / p0
    # Katz (1978) log-RR SE in its textbook form:
    #   sqrt(1/a_c - 1/(a_c+b_c) + 1/c_c - 1/(c_c+d_c))
    # This is equivalent to sqrt((1-p1)/a_c + (1-p0)/c_c) when the
    # counts are the Haldane-corrected values, and is stable for
    # sparse cells (no ad-hoc epsilon required).
    se_log = float(
        np.sqrt(1.0 / a_c - 1.0 / (a_c + b_c) + 1.0 / c_c - 1.0 / (c_c + d_c))
    )
    log_rr = np.log(rr)
    ci = (float(np.exp(log_rr - z * se_log)), float(np.exp(log_rr + z * se_log)))
    z_stat = log_rr / se_log if se_log > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z_stat)))

    return RR2x2Result(
        estimate=float(rr),
        se_log=se_log,
        ci=ci,
        p_value=p,
        risk_exposed=float(a / n1),
        risk_unexposed=float(c / n0),
        method="katz-log",
        level=1 - alpha,
    )


def prevalence_ratio(*args: Any, **kwargs: Any) -> RR2x2Result:
    """Prevalence ratio (cross-sectional RR); mathematically identical
    to :func:`relative_risk` when called on a 2x2 prevalence table.
    Distinguished for semantic clarity in cross-sectional studies.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.prevalence_ratio(40, 60, 20, 80)
    >>> round(res.estimate, 3)
    2.0
    >>> res.method
    'prevalence-ratio'

    References
    ----------
    zou2004modified
    """
    result = relative_risk(*args, **kwargs)
    return RR2x2Result(
        estimate=result.estimate,
        se_log=result.se_log,
        ci=result.ci,
        p_value=result.p_value,
        risk_exposed=result.risk_exposed,
        risk_unexposed=result.risk_unexposed,
        method="prevalence-ratio",
        level=result.level,
    )


# Expose relative_risk's real parameter list on the thin alias so
# introspection (help(), sp.function_schema, agent tooling) is informative.
prevalence_ratio.__signature__ = _inspect.signature(  # type: ignore[attr-defined]
    relative_risk
)


# --------------------------------------------------------------------------- #
#  Risk difference
# --------------------------------------------------------------------------- #


def risk_difference(
    a: Any,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    method: str = "wald",
    alpha: float = 0.05,
) -> RD2x2Result:
    """Risk difference with Wald or Newcombe CI.

    Parameters
    ----------
    method : {"wald", "newcombe"}
        Newcombe's hybrid score CI avoids the Wald overshoot problem
        near 0 or 1.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.risk_difference(40, 60, 20, 80)
    >>> round(res.estimate, 3)
    0.2
    >>> round(res.risk_exposed, 3), round(res.risk_unexposed, 3)
    (0.4, 0.2)
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    if method not in ("wald", "newcombe"):
        raise MethodIncompatibility(
            "method must be 'wald' or 'newcombe'",
            recovery_hint="Use method='wald' or method='newcombe'.",
            diagnostics={"method": method},
        )

    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        raise DataInsufficient(
            "Empty row in 2x2 table.",
            recovery_hint=("Each 2x2 row must contain at least one observation."),
            diagnostics={"n_exposed": n1, "n_unexposed": n0},
        )
    p1 = a / n1
    p0 = c / n0
    rd = p1 - p0
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
    z = _z_crit(alpha)

    if method == "wald":
        ci = (rd - z * se, rd + z * se)
    else:
        # Newcombe hybrid score: combine Wilson score intervals per arm.
        def _wilson(k: float, n: float) -> tuple[float, float]:
            if n == 0:
                return 0.0, 1.0
            phat = k / n
            denom = 1 + z**2 / n
            centre = (phat + z**2 / (2 * n)) / denom
            half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
            return float(centre - half), float(centre + half)

        l1, u1 = _wilson(a, n1)
        l0, u0 = _wilson(c, n0)
        # Newcombe's method 10
        lower = rd - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
        upper = rd + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
        ci = (float(lower), float(upper))

    # Wald Z-test for H0: p1 == p0 (pooled variance for significance)
    p_pool = (a + c) / (n1 + n0)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z_stat = rd / se_pool if se_pool > 0 else 0.0
    p_val = float(2 * stats.norm.sf(abs(z_stat)))

    return RD2x2Result(
        estimate=float(rd),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        p_value=p_val,
        risk_exposed=float(p1),
        risk_unexposed=float(p0),
        method=method,
        level=1 - alpha,
    )


# --------------------------------------------------------------------------- #
#  Attributable risk / population attributable fraction
# --------------------------------------------------------------------------- #


def attributable_risk(
    a: Any,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    alpha: float = 0.05,
    prevalence: str = "estimated",
) -> ARResult:
    """Attributable fractions in exposed + in population (Levin 1953).

    Computes, from the raw counts:
      - AF_exposed = (RR - 1) / RR
      - PAF = P_e * (RR - 1) / [1 + P_e * (RR - 1)] = (R - R_0) / R

    where P_e is the proportion exposed, R the overall risk and R_0 the
    risk among the unexposed (Stata ``cs`` / ``csi``).

    The PAF interval is the delta method on ``log(1 - PAF)``, where
    ``1 - PAF = R_0 / R``. ``prevalence`` says whether the exposure
    prevalence ``P_e = (a + b) / N`` is an estimate or a design constant:

    * ``"estimated"`` (default): the table is one random sample (a
      cross-sectional survey, or a cohort enrolled without regard to
      exposure), so ``P_e`` is estimated from it and its sampling error
      enters the interval. The variance is the multinomial delta method
      on the four cells. This is Stata ``punaf, vce(unconditional)``
      after ``glm y x, family(binomial) link(log) vce(robust)`` except for
      the robust ``N / (N - 1)`` factor, which is left out here.
    * ``"fixed"``: the exposure groups were sampled separately with sizes
      fixed by design and ``P_e`` is known. The variance is the Katz
      variance of ``log RR`` scaled by ``d log(1 - PAF) / d log RR``, as
      Stata ``punaf`` with the default conditional ``margins`` variance.

    Under random sampling the ``"fixed"`` interval is too narrow when the
    outcome is common: with ``P_e = 0.3``, ``R_0 = 0.45``, ``RR = 2`` and
    ``N = 400`` it covered 91.4% in a simulation against 94.7% for
    ``"estimated"``.

    The interval needs events in both exposure groups. With ``a = 0`` or
    ``c = 0`` the point estimates are still the plug-in values (``PAF = 1``
    when no unexposed case occurred) but ``paf_ci`` is ``(nan, nan)`` with
    a warning; Stata ``cs`` reports no interval there either.

    .. versionchanged:: 1.42
       ``prevalence="estimated"`` is the default; the interval used to
       hold ``P_e`` fixed in every case (``"fixed"`` keeps it).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.attributable_risk(40, 60, 20, 80)
    >>> round(res.ar_exposed, 3)
    0.5
    >>> round(res.paf, 4), round(res.prevalence_exposed, 3)
    (0.3333, 0.5)
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    z = _z_crit(alpha)
    if prevalence not in ("estimated", "fixed"):
        raise MethodIncompatibility(
            f"prevalence must be 'estimated' or 'fixed', got {prevalence!r}.",
            recovery_hint="Use 'estimated' when the table is one random "
            "sample and 'fixed' when the exposure groups were sampled "
            "separately with fixed sizes.",
            diagnostics={"prevalence": prevalence},
        )
    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        raise DataInsufficient(
            "Empty exposure row in 2x2 table.",
            recovery_hint=("Each exposure row must contain at least one observation."),
            diagnostics={"n_exposed": n1, "n_unexposed": n0},
        )
    if a + c == 0:
        raise DataInsufficient(
            "No case in the 2x2 table: attributable fractions are undefined.",
            recovery_hint=(
                "Attributable fractions need at least one case (a + c > 0)."
            ),
            diagnostics={"a": a, "c": c},
        )
    n_total = a + b + c + d
    p_exposed = (a + b) / n_total
    # Plug-in values on the raw counts. Until 1.40.2 a zero anywhere in the
    # table moved the point estimates to the Haldane-corrected RR (10, 0, 5,
    # 5 gave PAF 0.3125 for the true 1/3).
    if c > 0:
        rr = (a / n1) / (c / n0)
        af_exposed = (rr - 1) / rr if rr > 0 else float("-inf")
        numer = p_exposed * (rr - 1)
        paf = numer / (1 + numer)
    else:
        # No unexposed case: RR is infinite and every case is attributable.
        rr = float("inf")
        af_exposed = 1.0
        numer = float("inf")
        paf = 1.0

    if a > 0 and c > 0:
        if prevalence == "fixed":
            # Delta method on log(1 - PAF) via the Katz var(log RR):
            # 1 - PAF = 1 / (1 + P_e (RR - 1)),
            # d log(1 - PAF) / d log RR = -P_e RR / (1 + P_e (RR - 1)).
            se_log_rr = float(np.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n0))
            d_deriv = -p_exposed * rr / (1 + numer)
            se_lnq = abs(d_deriv) * se_log_rr
        else:
            # Multinomial delta method on log(1 - PAF) = log c - log(c + d)
            # - log(a + c) + log N. The gradient g sums to zero against the
            # counts (the function is scale free), so the variance is
            # sum_i g_i^2 n_i.
            g = np.array(
                [
                    -1.0 / (a + c) + 1.0 / n_total,
                    1.0 / n_total,
                    1.0 / c - 1.0 / n0 - 1.0 / (a + c) + 1.0 / n_total,
                    -1.0 / n0 + 1.0 / n_total,
                ]
            )
            se_lnq = float(np.sqrt(np.sum(g**2 * np.array([a, b, c, d]))))
        one_minus_paf = 1 - paf
        lo = 1 - np.exp(np.log(one_minus_paf) + z * se_lnq)
        hi = 1 - np.exp(np.log(one_minus_paf) - z * se_lnq)
        paf_ci = (float(lo), float(hi))
    else:
        warnings.warn(
            "attributable_risk: no interval for the PAF without events in "
            f"both exposure groups (a={a:g}, c={c:g}); paf_ci is (nan, nan).",
            UserWarning,
            stacklevel=2,
        )
        paf_ci = (float("nan"), float("nan"))

    return ARResult(
        ar_exposed=float(af_exposed),
        paf=float(paf),
        paf_ci=paf_ci,
        prevalence_exposed=float(p_exposed),
        rr=float(rr),
        level=1 - alpha,
    )


# --------------------------------------------------------------------------- #
#  Incidence rate ratio (person-time)
# --------------------------------------------------------------------------- #


def incidence_rate_ratio(
    events_exposed: float,
    pt_exposed: float,
    events_unexposed: float,
    pt_unexposed: float,
    *,
    alpha: float = 0.05,
    method: str = "exact",
) -> IRRResult:
    """Person-time incidence rate ratio with exact Poisson CI.

    Parameters
    ----------
    events_exposed, events_unexposed : float
        Event counts.
    pt_exposed, pt_unexposed : float
        Person-time at risk in each group (any time unit, as long as
        consistent).
    method : {"exact", "wald"}
        "exact": the exact conditional (Clopper-Pearson) interval and the
        exact conditional binomial test, as R ``poisson.test(c(D1, D0),
        c(PT1, PT0))``; with non-integer counts the p-value is the Wald one.
        "wald": ``log IRR +/- z se`` with ``se = sqrt(1/D1 + 1/D0)`` and the
        Wald test, a zero count replaced by 0.5 in both.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.incidence_rate_ratio(30, 1000.0, 15, 1000.0)
    >>> round(res.estimate, 3)
    2.0
    >>> round(res.rate_exposed, 3), round(res.rate_unexposed, 3)
    (0.03, 0.015)
    """
    if method not in ("exact", "wald"):
        raise MethodIncompatibility(
            "method must be 'exact' or 'wald'",
            recovery_hint=(
                "Use method='exact' for small event counts or "
                "method='wald' for asymptotic inference."
            ),
            diagnostics={"method": method},
        )
    try:
        events_exposed = float(events_exposed)
        pt_exposed = float(pt_exposed)
        events_unexposed = float(events_unexposed)
        pt_unexposed = float(pt_unexposed)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "Events and person-time must be numeric.",
            recovery_hint=(
                "Pass finite numeric event counts and positive person-time."
            ),
            diagnostics={
                "events_exposed": repr(events_exposed),
                "pt_exposed": repr(pt_exposed),
                "events_unexposed": repr(events_unexposed),
                "pt_unexposed": repr(pt_unexposed),
            },
        ) from exc
    values = np.asarray(
        [events_exposed, pt_exposed, events_unexposed, pt_unexposed],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise MethodIncompatibility(
            "Events and person-time must be finite.",
            recovery_hint=(
                "Drop or impute non-finite rates inputs before computing " "IRR."
            ),
            diagnostics={
                "events_exposed": events_exposed,
                "pt_exposed": pt_exposed,
                "events_unexposed": events_unexposed,
                "pt_unexposed": pt_unexposed,
            },
        )
    if pt_exposed <= 0 or pt_unexposed <= 0:
        raise MethodIncompatibility(
            "Person-time must be positive.",
            recovery_hint=(
                "Use strictly positive person-time in both exposure groups."
            ),
            diagnostics={
                "pt_exposed": pt_exposed,
                "pt_unexposed": pt_unexposed,
            },
        )
    if events_exposed < 0 or events_unexposed < 0:
        raise MethodIncompatibility(
            "Event counts must be non-negative.",
            recovery_hint=("Check event aggregation; counts cannot be negative."),
            diagnostics={
                "events_exposed": events_exposed,
                "events_unexposed": events_unexposed,
            },
        )

    rate1 = events_exposed / pt_exposed
    rate0 = events_unexposed / pt_unexposed

    if events_exposed == 0 and events_unexposed == 0:
        return IRRResult(
            estimate=float("nan"),
            ci=(float("nan"), float("nan")),
            p_value=float("nan"),
            rate_exposed=rate1,
            rate_unexposed=rate0,
            events_exposed=events_exposed,
            events_unexposed=events_unexposed,
            pt_exposed=pt_exposed,
            pt_unexposed=pt_unexposed,
            method=method,
            level=1 - alpha,
        )

    irr = rate1 / rate0 if rate0 > 0 else np.inf
    # Log-IRR SE
    e1 = max(events_exposed, 0.5)
    e0 = max(events_unexposed, 0.5)
    se_log = float(np.sqrt(1 / e1 + 1 / e0))

    # The Wald log-IRR on the same 0.5-floored counts as its SE. With both
    # counts positive this is log(irr) exactly; with a zero count it stays
    # finite (until 1.40.2 a zero count gave log(inf) or log(0): p = 0 for
    # 3 vs 0 events and p = 1 for 0 vs 3).
    log_irr_w = float(np.log((e1 / pt_exposed) / (e0 / pt_unexposed)))

    if method == "wald":
        z = _z_crit(alpha)
        ci = (
            float(np.exp(log_irr_w - z * se_log)),
            float(np.exp(log_irr_w + z * se_log)),
        )
    else:
        # Conditional on total events, D1 ~ Binomial(D, pi) where
        # pi = (lambda1 * PT1) / (lambda1*PT1 + lambda0*PT0)
        # IRR is a one-to-one transform of pi.
        d = events_exposed + events_unexposed
        if d == 0:
            ci = (float("nan"), float("nan"))
        else:
            from scipy.stats import beta

            k = events_exposed
            # Clopper-Pearson exact binomial CI for pi
            if k == 0:
                lo_pi = 0.0
            else:
                lo_pi = float(beta.ppf(alpha / 2, k, d - k + 1))
            if k == d:
                hi_pi = 1.0
            else:
                hi_pi = float(beta.ppf(1 - alpha / 2, k + 1, d - k))
            # IRR = (pi / (1-pi)) * (PT0 / PT1)
            ratio = pt_unexposed / pt_exposed
            lo_irr = (lo_pi / (1 - lo_pi)) * ratio if lo_pi < 1 else np.inf
            hi_irr = (hi_pi / (1 - hi_pi)) * ratio if hi_pi < 1 else np.inf
            ci = (float(lo_irr), float(hi_irr))

    integer_counts = (
        float(events_exposed).is_integer() and float(events_unexposed).is_integer()
    )
    if method == "exact" and integer_counts:
        # Exact conditional test, the companion of the exact interval:
        # D1 ~ Binomial(D1 + D0, PT1 / (PT1 + PT0)) under IRR = 1, two-sided
        # as R poisson.test(c(D1, D0), c(PT1, PT0)). Until 1.40.2 the exact
        # method reported the Wald p-value.
        p = float(
            stats.binomtest(
                int(events_exposed),
                int(events_exposed + events_unexposed),
                pt_exposed / (pt_exposed + pt_unexposed),
            ).pvalue
        )
    else:
        # Two-sided Wald p-value on log-IRR (non-integer counts have no
        # exact binomial test)
        z_stat = log_irr_w / se_log if se_log > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(z_stat)))

    return IRRResult(
        estimate=float(irr),
        ci=ci,
        p_value=p,
        rate_exposed=float(rate1),
        rate_unexposed=float(rate0),
        events_exposed=float(events_exposed),
        events_unexposed=float(events_unexposed),
        pt_exposed=float(pt_exposed),
        pt_unexposed=float(pt_unexposed),
        method=method,
        level=1 - alpha,
    )


# --------------------------------------------------------------------------- #
#  NNT
# --------------------------------------------------------------------------- #


def number_needed_to_treat(
    a: Any,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    alpha: float = 0.05,
) -> NNTResult:
    """Number needed to treat (or harm), defined as |1 / RD|.

    Propagates the Wald CI for RD.  Interpretation convention:
    negative RD -> NNT-Benefit (treatment reduces risk);
    positive RD -> NNT-Harm.

    When the RD interval ``(L, U)`` excludes zero, ``ci`` is
    ``(1/|U|, 1/|L|)`` sorted, matching R ``epiR::epi.2by2``. When it
    includes zero the set of compatible NNTs is disjoint: ``1/|L|`` or more
    on one side and ``1/U`` or more on the other, reported by Altman
    (1998) as "NNTB 1/|L| to infinity to NNTH 1/U". ``ci_disjoint`` is then
    True, ``ci_signed`` is ``(1/L, 1/U)`` (epiR's pair) and ``ci`` is the
    interval for the magnitude, ``(1 / max(|L|, U), inf)``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.number_needed_to_treat(20, 80, 40, 60)
    >>> round(res.estimate, 2)
    5.0
    >>> round(res.risk_difference, 3)  # treated arm has lower risk
    -0.2

    References
    ----------
    Altman, D. G. (1998). Confidence intervals for the number needed to
    treat. BMJ 317(7168), 1309-1312. doi:10.1136/bmj.317.7168.1309
    """
    rd_res = risk_difference(a, b, c, d, alpha=alpha, method="wald")
    rd = rd_res.estimate
    lo, hi = (float(v) for v in rd_res.ci)

    def _recip(x: float, side: float) -> float:
        # 1/x with a limit of exactly zero sent to the infinity on ``side``.
        if x == 0:
            return float("inf") if side > 0 else float("-inf")
        return 1.0 / x

    nnt = 1 / abs(rd) if abs(rd) >= 1e-12 else float("inf")
    if lo == 0 and hi == 0:
        # Zero-variance table with RD = 0: no finite NNT is compatible.
        ci = (float("inf"), float("inf"))
        ci_signed = (float("inf"), float("inf"))
        disjoint = False
    elif lo < 0 < hi:
        # Spans both benefit and harm (Altman 1998): the compatible
        # |NNT| run from 1/max(|L|, U) to infinity.
        ci = (1.0 / max(abs(lo), hi), float("inf"))
        ci_signed = (1.0 / lo, 1.0 / hi)
        disjoint = True
    elif hi <= 0:
        # Benefit side; a limit at exactly 0 means NNT up to infinity
        # (until 1.40.2 it raised ZeroDivisionError).
        ci_signed = (_recip(hi, -1.0), _recip(lo, -1.0))
        ci = (abs(ci_signed[1]), abs(ci_signed[0]))
        disjoint = False
    else:
        ci_signed = (_recip(hi, 1.0), _recip(lo, 1.0))
        ci = (ci_signed[0], ci_signed[1])
        disjoint = False

    return NNTResult(
        estimate=float(nnt),
        ci=(float(ci[0]), float(ci[1])),
        risk_difference=float(rd),
        ci_signed=(float(ci_signed[0]), float(ci_signed[1])),
        ci_disjoint=disjoint,
        level=1 - alpha,
    )
