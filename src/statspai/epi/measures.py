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

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


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


def _coerce_2x2(a, b=None, c=None, d=None):
    """Accept either ``(a, b, c, d)`` counts or a 2x2 matrix.

    Returns floats to allow continuity corrections downstream without
    integer truncation issues.
    """
    if b is None:
        arr = np.asarray(a, dtype=float)
        if arr.shape != (2, 2):
            raise ValueError(
                "2x2 table expected; got shape %s" % (arr.shape,)
            )
        (a, b), (c, d) = arr
    if any(x < 0 for x in (a, b, c, d)):
        raise ValueError("Counts must be non-negative.")
    return float(a), float(b), float(c), float(d)


def _haldane_correction(a: float, b: float, c: float, d: float):
    """Haldane-Anscombe 0.5 continuity correction when any cell is zero."""
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return a, b, c, d


def _z_crit(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class OR2x2Result:
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

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Odds Ratio ({self.method})\n"
            f"  OR = {self.estimate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  2x2 table: [[{self.a:.0f}, {self.b:.0f}], "
            f"[{self.c:.0f}, {self.d:.0f}]]"
        )


@dataclass
class RR2x2Result:
    """Result of a 2x2 relative-risk (risk-ratio) calculation."""

    estimate: float
    se_log: float
    ci: tuple[float, float]
    p_value: float
    risk_exposed: float
    risk_unexposed: float
    method: str

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Relative Risk ({self.method})\n"
            f"  RR = {self.estimate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  Risk (exposed)   = {self.risk_exposed:.4f}\n"
            f"  Risk (unexposed) = {self.risk_unexposed:.4f}"
        )


@dataclass
class RD2x2Result:
    """Result of a 2x2 risk-difference calculation."""

    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    risk_exposed: float
    risk_unexposed: float
    method: str

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Risk Difference ({self.method})\n"
            f"  RD = {self.estimate:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
            f"p = {self.p_value:.4g}"
        )


@dataclass
class ARResult:
    """Attributable-risk quantities (Levin 1953, Miettinen 1974)."""

    ar_exposed: float  # AR% among the exposed (a.k.a. attributable fraction in exposed)
    paf: float         # Population attributable fraction (Levin)
    paf_ci: tuple[float, float]
    prevalence_exposed: float
    rr: float

    def summary(self) -> str:
        lo, hi = self.paf_ci
        return (
            "Attributable Risk\n"
            f"  AF (exposed) = {self.ar_exposed:.4f}\n"
            f"  PAF (Levin)  = {self.paf:.4f}   95% CI [{lo:.4f}, {hi:.4f}]\n"
            f"  Prevalence exposed = {self.prevalence_exposed:.4f}\n"
            f"  RR = {self.rr:.4f}"
        )


@dataclass
class IRRResult:
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

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Incidence Rate Ratio ({self.method})\n"
            f"  IRR = {self.estimate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  Rate (exposed)   = {self.rate_exposed:.6f}  "
            f"({self.events_exposed:.0f}/{self.pt_exposed:.1f} person-time)\n"
            f"  Rate (unexposed) = {self.rate_unexposed:.6f}  "
            f"({self.events_unexposed:.0f}/{self.pt_unexposed:.1f} person-time)"
        )


@dataclass
class NNTResult:
    estimate: float
    ci: tuple[float, float]
    risk_difference: float

    def summary(self) -> str:
        lo, hi = self.ci
        # NNT is reported as NNT-Benefit or NNT-Harm; the sign of RD drives it.
        tag = "NNT Benefit" if self.risk_difference < 0 else "NNT Harm"
        return (
            f"{tag}\n"
            f"  NNT = {self.estimate:.2f}   95% CI [{lo:.2f}, {hi:.2f}]"
        )


# --------------------------------------------------------------------------- #
#  Odds ratio
# --------------------------------------------------------------------------- #


def odds_ratio(
    a,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    method: str = "woolf",
    alpha: float = 0.05,
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
        Confidence-interval method.  "woolf" uses the asymptotic
        log-OR standard error; "exact" uses the Fisher-style
        conditional non-central hypergeometric CI (via
        :func:`scipy.stats.fisher_exact`).
    alpha : float, default 0.05

    Returns
    -------
    OR2x2Result

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.odds_ratio(50, 20, 30, 40)
    >>> round(res.estimate, 3)
    3.333
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    if method not in ("woolf", "exact"):
        raise ValueError("method must be 'woolf' or 'exact'")

    # Haldane correction for log-OR SE path.
    a_c, b_c, c_c, d_c = _haldane_correction(a, b, c, d)
    or_point = (a_c * d_c) / (b_c * c_c)
    log_or = np.log(or_point)
    se_log = float(np.sqrt(1 / a_c + 1 / b_c + 1 / c_c + 1 / d_c))

    if method == "woolf":
        z = _z_crit(alpha)
        ci = (float(np.exp(log_or - z * se_log)),
              float(np.exp(log_or + z * se_log)))
        # Two-sided chi-square test for H0: OR = 1
        z_stat = log_or / se_log if se_log > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    else:
        # Fisher's exact: use scipy, which gives unconditional MLE OR + exact CI.
        table = np.array([[a, b], [c, d]], dtype=int)
        try:
            res = stats.fisher_exact(table, alternative="two-sided")
            # scipy returns (odds_ratio, p_value)
            or_exact = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
            p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
            # Use scipy.stats.contingency.odds_ratio for exact CI if available
            try:
                from scipy.stats.contingency import odds_ratio as _scipy_or
                obj = _scipy_or(table, kind="conditional")
                ci_lo, ci_hi = obj.confidence_interval(confidence_level=1 - alpha)
                ci = (float(ci_lo), float(ci_hi))
                or_point = or_exact
            except Exception:
                # Fall back to Woolf CI.
                z = _z_crit(alpha)
                ci = (float(np.exp(log_or - z * se_log)),
                      float(np.exp(log_or + z * se_log)))
        except Exception as exc:  # pragma: no cover - scipy always has fisher_exact
            raise RuntimeError(f"scipy.stats.fisher_exact failed: {exc}")

    _result = OR2x2Result(
        estimate=float(or_point),
        se_log=se_log,
        ci=ci,
        p_value=p,
        a=a, b=b, c=c, d=d,
        method=method,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.epi.odds_ratio",
            params={"method": method, "alpha": alpha},
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
    a,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    alpha: float = 0.05,
) -> RR2x2Result:
    """Relative risk (risk ratio) with Katz log-RR confidence interval.

    Uses the Haldane correction when any cell is zero.
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        raise ValueError("Empty exposure row in 2x2 table.")

    a_c, b_c, c_c, d_c = _haldane_correction(a, b, c, d)
    p1 = a_c / (a_c + b_c)
    p0 = c_c / (c_c + d_c)
    rr = p1 / p0
    # Katz (1978) log-RR SE in its textbook form:
    #   sqrt(1/a_c - 1/(a_c+b_c) + 1/c_c - 1/(c_c+d_c))
    # This is equivalent to sqrt((1-p1)/a_c + (1-p0)/c_c) when the
    # counts are the Haldane-corrected values, and is stable for
    # sparse cells (no ad-hoc epsilon required).
    se_log = float(np.sqrt(
        1.0 / a_c - 1.0 / (a_c + b_c)
        + 1.0 / c_c - 1.0 / (c_c + d_c)
    ))
    z = _z_crit(alpha)
    log_rr = np.log(rr)
    ci = (float(np.exp(log_rr - z * se_log)),
          float(np.exp(log_rr + z * se_log)))
    z_stat = log_rr / se_log if se_log > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return RR2x2Result(
        estimate=float(rr),
        se_log=se_log,
        ci=ci,
        p_value=p,
        risk_exposed=float(a / n1),
        risk_unexposed=float(c / n0),
        method="katz-log",
    )


def prevalence_ratio(*args, **kwargs) -> RR2x2Result:
    """Prevalence ratio (cross-sectional RR); mathematically identical
    to :func:`relative_risk` when called on a 2x2 prevalence table.
    Distinguished for semantic clarity in cross-sectional studies.
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
    )


# --------------------------------------------------------------------------- #
#  Risk difference
# --------------------------------------------------------------------------- #


def risk_difference(
    a,
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
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    if method not in ("wald", "newcombe"):
        raise ValueError("method must be 'wald' or 'newcombe'")

    n1 = a + b
    n0 = c + d
    if n1 == 0 or n0 == 0:
        raise ValueError("Empty row in 2x2 table.")
    p1 = a / n1
    p0 = c / n0
    rd = p1 - p0
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
    z = _z_crit(alpha)

    if method == "wald":
        ci = (rd - z * se, rd + z * se)
    else:
        # Newcombe hybrid score: combine Wilson score intervals per arm.
        def _wilson(k, n):
            if n == 0:
                return 0.0, 1.0
            phat = k / n
            denom = 1 + z ** 2 / n
            centre = (phat + z ** 2 / (2 * n)) / denom
            half = (z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
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
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return RD2x2Result(
        estimate=float(rd),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        p_value=p_val,
        risk_exposed=float(p1),
        risk_unexposed=float(p0),
        method=method,
    )


# --------------------------------------------------------------------------- #
#  Attributable risk / population attributable fraction
# --------------------------------------------------------------------------- #


def attributable_risk(
    a,
    b: Optional[float] = None,
    c: Optional[float] = None,
    d: Optional[float] = None,
    *,
    alpha: float = 0.05,
) -> ARResult:
    """Attributable fractions in exposed + in population (Levin 1953).

    Computes:
      - AF_exposed = (RR - 1) / RR
      - PAF = P_e * (RR - 1) / [1 + P_e * (RR - 1)]

    where P_e is prevalence of exposure.  CI for PAF uses the delta
    method on log(1 - PAF).
    """
    a, b, c, d = _coerce_2x2(a, b, c, d)
    rr_res = relative_risk(a, b, c, d, alpha=alpha)
    rr = rr_res.estimate
    n_total = a + b + c + d
    p_exposed = (a + b) / n_total if n_total > 0 else 0.0
    af_exposed = (rr - 1) / rr if rr > 0 else np.nan
    numer = p_exposed * (rr - 1)
    paf = numer / (1 + numer) if (1 + numer) > 0 else np.nan

    # Delta-method CI on log(1 - PAF) via log(RR) variance (Greenland 2001)
    # 1 - PAF = 1 / (1 + P_e*(RR-1))
    # var[log(1 - PAF)] approximated via var[log(RR)] * (dlog(1-PAF)/dlog(RR))^2
    if rr > 0 and paf < 1:
        one_minus_paf = 1 - paf
        # d(log(1-PAF))/d(logRR) = -P_e * RR / (1 + P_e*(RR-1))
        d_deriv = -p_exposed * rr / (1 + numer)
        se_lnq = abs(d_deriv) * rr_res.se_log
        z = _z_crit(alpha)
        lo = 1 - np.exp(np.log(one_minus_paf) + z * se_lnq)
        hi = 1 - np.exp(np.log(one_minus_paf) - z * se_lnq)
        paf_ci = (float(lo), float(hi))
    else:
        paf_ci = (float("nan"), float("nan"))

    return ARResult(
        ar_exposed=float(af_exposed),
        paf=float(paf),
        paf_ci=paf_ci,
        prevalence_exposed=float(p_exposed),
        rr=float(rr),
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
        "exact" uses the F-distribution-based Poisson CI (Breslow-Day);
        "wald" uses log-rate SE.
    """
    if pt_exposed <= 0 or pt_unexposed <= 0:
        raise ValueError("Person-time must be positive.")
    if events_exposed < 0 or events_unexposed < 0:
        raise ValueError("Event counts must be non-negative.")

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
        )

    irr = rate1 / rate0 if rate0 > 0 else np.inf
    # Log-IRR SE
    e1 = max(events_exposed, 0.5)
    e0 = max(events_unexposed, 0.5)
    se_log = float(np.sqrt(1 / e1 + 1 / e0))

    if method == "wald":
        z = _z_crit(alpha)
        log_irr = np.log(irr) if irr > 0 else np.nan
        ci = (float(np.exp(log_irr - z * se_log)),
              float(np.exp(log_irr + z * se_log)))
    elif method == "exact":
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
    else:
        raise ValueError("method must be 'exact' or 'wald'")

    # Two-sided Wald p-value on log-IRR
    log_irr = np.log(irr) if irr > 0 else 0.0
    z_stat = log_irr / se_log if se_log > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

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
    )


# --------------------------------------------------------------------------- #
#  NNT
# --------------------------------------------------------------------------- #


def number_needed_to_treat(
    a,
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
    """
    rd_res = risk_difference(a, b, c, d, alpha=alpha, method="wald")
    rd = rd_res.estimate
    if abs(rd) < 1e-12:
        return NNTResult(
            estimate=float("inf"),
            ci=(float("inf"), float("inf")),
            risk_difference=float(rd),
        )

    # If CI crosses zero, NNT CI is a union of two half-lines.  Report
    # as (infinity, finite) with a signal via the result summary.
    lo, hi = rd_res.ci
    nnt = 1 / abs(rd)
    if lo < 0 < hi:
        # Spans both benefit and harm
        ci = (float("inf"), float("inf"))
    else:
        # NNT is 1/|RD|.  The CI bound *closer to zero* on the RD
        # scale gives the *larger* NNT (treatment less certainly
        # effective).  We therefore sort by NNT magnitude directly.
        nnt_endpoints = (1 / abs(hi), 1 / abs(lo))
        ci = (min(nnt_endpoints), max(nnt_endpoints))

    return NNTResult(
        estimate=float(nnt),
        ci=(float(ci[0]), float(ci[1])),
        risk_difference=float(rd),
    )
