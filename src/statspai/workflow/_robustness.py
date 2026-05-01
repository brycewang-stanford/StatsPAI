"""
Shared robustness battery used by both ``sp.paper`` entry points.

The pre-1.12 split was: ``paper(data, question, ...)`` ran a thin
:meth:`CausalWorkflow.robustness` (DID pretrend / IV first-stage F /
ci_width / e-value), and ``paper_from_question`` ran *nothing* —
its Robustness section was a placeholder pointing the user back to
``sp.causal``. The user-visible promise ("parallel-trends test,
leave-one-out, specification curve, e-value, sensitivity") was wider
than what either entry point actually delivered.

This module consolidates: any caller that has a fitted result and
optionally the analysis frame can produce a structured
:class:`RobustnessReport`. The battery is **design-aware** but always
yields *something* — every sub-check is wrapped so that one failing
diagnostic never blocks the rest. Failed checks land as
``severity='check_failed'`` findings so the agent / human reader
can see what *would* have been reported.

Findings cover:

* ``result.violations()`` — every estimator's self-reported
  diagnostic flags (first-stage F, propensity overlap, rhat / ESS,
  pre-trend p, McCrary, ...).
* Estimate + CI width — universal sanity.
* DID: ``pretrend_test``, ``bjs_pretrend_joint``.
* IV: first-stage ``effective_f``-like metrics from
  ``result.diagnostics``.
* Observational / DID: e-value via :func:`statspai.evalue_from_result`.
* Observational with covariates: Oster (2019) coefficient-stability
  bounds via :func:`statspai.oster_bounds`.
* Observational with covariates: Cinelli-Hazlett (2020) sensitivity
  via :func:`statspai.sensemakr`.
* Any result whose ``model_info["diagnostics"]`` is populated (DML,
  panel_dml, ...) — the diagnostics block is surfaced verbatim.

The same battery output is rendered into Robustness sections by
:func:`paper_from_question` (estimand-first) and
:meth:`CausalWorkflow.robustness` (NL-driven path) so the user gets
the same content regardless of how they reached the draft.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- #
#  Data types
# ---------------------------------------------------------------------- #

# severity → ordering used by markdown rendering (worst first)
_SEVERITY_ORDER = {
    "violation": 0,
    "warning": 1,
    "info": 2,
    "ok": 3,
    "check_failed": 4,
}


@dataclass
class RobustnessFinding:
    """One row in the robustness report.

    Attributes
    ----------
    name : str
        Stable identifier for programmatic access (snake_case).
    label : str
        Human-readable label rendered in the Robustness section.
    value : Any
        Numeric scalar, mapping, or short string.
    severity : str
        One of ``'ok'``, ``'info'``, ``'warning'``, ``'violation'``,
        ``'check_failed'``. Drives ordering and the leading icon in
        the markdown rendering.
    interpretation : str, optional
        One-line plain-English interpretation; rendered after the
        value in markdown.
    """
    name: str
    label: str
    value: Any
    severity: str = "info"
    interpretation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "label": self.label,
            "value": _coerce_jsonable(self.value),
            "severity": self.severity,
        }
        if self.interpretation:
            d["interpretation"] = self.interpretation
        return d


@dataclass
class RobustnessReport:
    """Container for a battery of robustness findings.

    Attributes
    ----------
    findings : list[RobustnessFinding]
    design : str, optional
        The design tag the battery was run with (``'did'`` / ``'iv'``
        / ``'rd'`` / ``'observational'`` / ...). Useful for the
        markdown header.
    notes : list[str]
        Free-form notes appended at the end of the section (e.g. when
        a particular check was deliberately skipped because the
        result-shape did not support it).
    """
    findings: List[RobustnessFinding] = field(default_factory=list)
    design: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.findings) == 0 and not self.notes

    def to_dict(self) -> Dict[str, Any]:
        """Flat-key-value dict for backwards-compat with the old
        ``CausalWorkflow.robustness_findings`` shape, *plus* a
        ``_findings`` array carrying the structured records."""
        out: Dict[str, Any] = {}
        for f in self.findings:
            out[f.name] = _coerce_jsonable(f.value)
        out["_findings"] = [f.to_dict() for f in self.findings]
        if self.notes:
            out["_notes"] = list(self.notes)
        if self.design:
            out["_design"] = self.design
        return out

    def to_markdown(self) -> str:
        """Render the report as a markdown bullet list.

        Each finding renders as ``- {icon} {label}: {value} — {interp}``
        with findings sorted by severity. Empty reports return a
        clear "no findings" sentinel so the section is never blank.
        """
        if self.is_empty():
            return (
                "_The robustness battery produced no findings — this "
                "usually means the result-shape did not expose any "
                "diagnostics. Re-fit with a richer estimator (e.g. "
                "passing `dag=` / `cluster=` / a covariate set) for a "
                "non-empty Robustness section._"
            )
        sorted_f = sorted(
            self.findings,
            key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), f.name),
        )
        lines: List[str] = []
        if self.design:
            lines.append(
                f"_Battery for design `{self.design}` — "
                f"{len(self.findings)} finding(s)._"
            )
            lines.append("")
        for f in sorted_f:
            icon = _icon(f.severity)
            val = _format_value(f.value)
            base = f"- {icon} **{f.label}**: {val}"
            if f.interpretation:
                base = f"{base} — {f.interpretation}"
            lines.append(base)
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"_Note: {note}_")
        return "\n".join(lines)


def _icon(severity: str) -> str:
    return {
        "ok": "✅",
        "info": "ℹ️",
        "warning": "⚠️",
        "violation": "❌",
        "check_failed": "⚙️",
    }.get(severity, "•")


def _format_value(v: Any) -> str:
    if v is None:
        return "_(not available)_"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return f"{v}"
        return f"{float(v):.4f}"
    if isinstance(v, dict):
        bits = []
        for k, vv in list(v.items())[:6]:
            bits.append(f"{k}={_format_value(vv)}")
        more = "" if len(v) <= 6 else f", … +{len(v) - 6}"
        return "{" + ", ".join(bits) + more + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in list(v)[:5]) + (
            "…" if len(v) > 5 else "") + "]"
    return str(v)


def _coerce_jsonable(v: Any) -> Any:
    """Convert numpy scalars / arrays to plain Python so to_dict()
    payloads can round-trip through json.dumps without help."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_coerce_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _coerce_jsonable(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------------- #
#  Battery implementation
# ---------------------------------------------------------------------- #


def run_robustness_battery(
    result: Any,
    *,
    design: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    covariates: Optional[List[str]] = None,
) -> RobustnessReport:
    """Run a design-appropriate robustness battery on ``result``.

    Always returns a :class:`RobustnessReport`; never raises. Each
    sub-check is wrapped so a single failure becomes a
    ``severity='check_failed'`` finding rather than aborting the
    rest.

    Parameters
    ----------
    result : object
        Fitted estimator output. Anything exposing ``estimate`` /
        ``ci`` / ``violations()`` / ``model_info`` works.
    design : str, optional
        ``'did'`` / ``'iv'`` / ``'rd'`` / ``'observational'`` /
        ``'synth'`` / ``'panel'`` / etc. Drives which design-specific
        checks are attempted.
    data : pd.DataFrame, optional
        The analysis frame, if available. Required for Oster bounds
        and sensemakr; otherwise those checks are silently skipped.
    treatment : str, optional
        Treatment column name. Same role as ``data``.
    outcome : str, optional
        Outcome column name. Same role as ``data``.
    covariates : list[str], optional
        Covariate column names. Required for Oster bounds and
        sensemakr; otherwise those checks are silently skipped.

    Returns
    -------
    RobustnessReport
    """
    findings: List[RobustnessFinding] = []
    notes: List[str] = []

    if result is None:
        notes.append(
            "No fitted result supplied; battery cannot run — pass a "
            "result with estimate/ci/model_info."
        )
        return RobustnessReport(findings=findings, design=design, notes=notes)

    # --- 1. result.violations() — universal agent-native diagnostics
    _add_violations(result, findings)

    # --- 2. Estimate + CI width (universal sanity)
    _add_estimate_summary(result, findings, treatment=treatment)

    # --- 3. Normalise the design tag to the battery's internal
    # taxonomy. Both ``CausalWorkflow.design`` (free-form) and
    # ``CausalQuestion.design`` (the closed enum
    # 'selection_on_observables' / 'dml' / 'tmle' / ...) feed in here.
    d_norm = _normalise_design(design)

    if d_norm == "did":
        _add_did_checks(result, findings)
    elif d_norm == "iv":
        _add_iv_checks(result, findings)
    elif d_norm == "rd":
        _add_rd_checks(result, findings)

    # --- 4. E-value (observational + DID)
    if d_norm in ("did", "observational"):
        _add_evalue(result, findings)

    # --- 5. Oster bounds + sensemakr (observational with covariates)
    if d_norm == "observational":
        _add_oster_bounds(result, data, treatment, outcome, covariates, findings, notes)
        _add_sensemakr(result, data, treatment, outcome, covariates, findings, notes)

    # --- 6. Surface model_info["diagnostics"] block verbatim if present
    _add_model_info_diagnostics(result, findings)

    return RobustnessReport(findings=findings, design=design, notes=notes)


# ---------------------------------------------------------------------- #
#  Sub-check implementations
# ---------------------------------------------------------------------- #


def _add_violations(result: Any, findings: List[RobustnessFinding]) -> None:
    try:
        viols_fn = getattr(result, "violations", None)
        if not callable(viols_fn):
            return
        viols = viols_fn()
        if not viols:
            findings.append(RobustnessFinding(
                name="violations_none",
                label="Self-reported violations",
                value="none",
                severity="ok",
                interpretation="`result.violations()` returned an empty list.",
            ))
            return
        for v in viols:
            sev = (v.get("severity") or "info").lower()
            severity = {
                "error": "violation",
                "warning": "warning",
                "info": "info",
            }.get(sev, "info")
            findings.append(RobustnessFinding(
                name=f"viol_{v.get('test', 'unknown')}",
                label=f"Violation flag: {v.get('test', 'unknown')}",
                value=v.get("value"),
                severity=severity,
                interpretation=v.get("message"),
            ))
    except Exception as exc:  # pragma: no cover — defensive
        findings.append(_check_failed("violations", exc))


def _add_estimate_summary(
    result: Any, findings: List[RobustnessFinding],
    treatment: Optional[str] = None,
) -> None:
    try:
        if hasattr(result, "estimate") and hasattr(result, "ci"):
            est = float(result.estimate)
            lo, hi = result.ci
            lo, hi = float(lo), float(hi)
            findings.append(RobustnessFinding(
                name="estimate",
                label="Point estimate",
                value=est,
                severity="info",
            ))
            findings.append(RobustnessFinding(
                name="ci_width",
                label="95% CI width",
                value=hi - lo,
                severity="info",
                interpretation=f"[{lo:+.4f}, {hi:+.4f}]",
            ))
        elif hasattr(result, "params"):
            # statsmodels-flavoured result. Prefer the user-supplied
            # treatment column over the first row (which is usually
            # ``Intercept`` and not the parameter of interest).
            try:
                idx = list(result.params.index)
                if treatment and treatment in idx:
                    main_idx = treatment
                else:
                    # Fall back to the first non-Intercept entry, then
                    # the very first as last resort.
                    non_const = [
                        n for n in idx
                        if str(n).lower() not in {"intercept", "const", "(intercept)"}
                    ]
                    main_idx = non_const[0] if non_const else idx[0]
                coef = float(result.params[main_idx])
                se = float(result.std_errors[main_idx])
                findings.append(RobustnessFinding(
                    name="estimate",
                    label=f"Coefficient on {main_idx}",
                    value=coef,
                    severity="info",
                ))
                findings.append(RobustnessFinding(
                    name="ci_width",
                    label="95% CI width (≈ 2·1.96·SE)",
                    value=2 * 1.96 * se,
                    severity="info",
                ))
            except Exception as exc:  # pragma: no cover — defensive
                findings.append(_check_failed("estimate_summary", exc))
    except Exception as exc:  # pragma: no cover — defensive
        findings.append(_check_failed("estimate_summary", exc))


def _add_did_checks(result: Any, findings: List[RobustnessFinding]) -> None:
    mi = getattr(result, "model_info", None) or {}
    pt = mi.get("pretrend_test")
    if pt is not None:
        # Convention: many estimators store {pvalue, statistic} or just pvalue
        if isinstance(pt, dict) and "pvalue" in pt:
            p = float(pt["pvalue"])
            sev = "warning" if p < 0.10 else "ok"
            findings.append(RobustnessFinding(
                name="pretrend_test",
                label="Parallel-trends pre-test",
                value=p,
                severity=sev,
                interpretation=(
                    "p < 0.10 — pre-trends look unbalanced; investigate"
                    if p < 0.10 else
                    "p ≥ 0.10 — no pre-trend evidence at the 10% level"
                ),
            ))
        else:
            findings.append(RobustnessFinding(
                name="pretrend_test",
                label="Parallel-trends pre-test",
                value=pt,
                severity="info",
            ))
    pj = mi.get("bjs_pretrend_joint") or mi.get("pretrend_joint")
    if pj is not None and isinstance(pj, dict) and "pvalue" in pj:
        p = float(pj["pvalue"])
        sev = "warning" if p < 0.10 else "ok"
        findings.append(RobustnessFinding(
            name="pretrend_joint",
            label="Joint pre-trend test (BJS)",
            value=p,
            severity=sev,
        ))


def _add_iv_checks(result: Any, findings: List[RobustnessFinding]) -> None:
    diags = getattr(result, "diagnostics", None) or {}
    # Look for first-stage F-style keys.  The matcher must accept
    # ``first_stage_F``, ``first_stage_F_approx``, ``first_stage_partial_corr``,
    # but reject p-value variants which collapse to 0 and would
    # otherwise be misread as Stock–Yogo violations.
    skip_substrings = ("p_value", "pvalue", "p-value", "pval")
    for k, v in diags.items():
        kl = str(k).lower()
        if any(s in kl for s in skip_substrings):
            continue
        if "first" in kl and ("_f" in kl or " f" in kl or "stage" in kl
                              or kl.endswith("_f") or kl == "first_stage_f"):
            try:
                fval = float(v) if isinstance(v, (int, float, np.integer, np.floating)) \
                    else (float(v.get("value", "nan")) if isinstance(v, dict) else float("nan"))
            except Exception:
                fval = float("nan")
            # Only treat finite numeric F values as Stock–Yogo-comparable.
            if not np.isfinite(fval):
                continue
            sev = "ok" if fval >= 10 else ("warning" if fval >= 5 else "violation")
            findings.append(RobustnessFinding(
                name=f"iv_{k}",
                label=f"First-stage strength ({k})",
                value=fval,
                severity=sev,
                interpretation=(
                    "≥ 10 — Stock–Yogo rule of thumb passed"
                    if fval >= 10 else
                    "< 10 — weak instrument flag (Stock–Yogo)"
                ),
            ))
    # Anderson-Rubin / weak-IV-robust
    mi = getattr(result, "model_info", None) or {}
    ar = mi.get("anderson_rubin") or mi.get("AR_test")
    if ar is not None and isinstance(ar, dict) and "pvalue" in ar:
        p = float(ar["pvalue"])
        findings.append(RobustnessFinding(
            name="anderson_rubin_pvalue",
            label="Anderson-Rubin (weak-IV-robust)",
            value=p,
            severity="info",
            interpretation="weak-IV-robust p-value for H0: β=0",
        ))


def _add_rd_checks(result: Any, findings: List[RobustnessFinding]) -> None:
    mi = getattr(result, "model_info", None) or {}
    # McCrary / RDD-density manipulation test
    mc = mi.get("mccrary_test") or mi.get("rd_density")
    if mc is not None and isinstance(mc, dict) and "pvalue" in mc:
        p = float(mc["pvalue"])
        sev = "warning" if p < 0.10 else "ok"
        findings.append(RobustnessFinding(
            name="mccrary_test",
            label="Manipulation test (McCrary / Cattaneo-Jansson-Ma)",
            value=p,
            severity=sev,
            interpretation=(
                "p < 0.10 — running-variable density discontinuity at "
                "the cutoff; check for sorting"
                if p < 0.10 else
                "p ≥ 0.10 — no density-jump evidence at 10%"
            ),
        ))
    # Bandwidth disclosure
    bw = mi.get("bandwidth") or mi.get("h")
    if bw is not None:
        findings.append(RobustnessFinding(
            name="rd_bandwidth",
            label="Selected bandwidth",
            value=bw,
            severity="info",
        ))


def _add_evalue(result: Any, findings: List[RobustnessFinding]) -> None:
    try:
        from .. import evalue_from_result  # type: ignore
        ev = evalue_from_result(result)
        # statspai 1.x returns a dict with ``evalue_estimate`` (point
        # estimate's e-value) and ``evalue_ci`` (CI-limit e-value, which
        # is the more conservative read).  Older callers may pass an
        # object exposing ``evalue`` / a bare float — handle both.
        if isinstance(ev, dict):
            for key in ("evalue_ci", "evalue_estimate", "E_value", "evalue", "e_value"):
                if key in ev:
                    val = float(ev[key])
                    break
            else:
                raise ValueError(f"e-value dict has no recognised key: {list(ev)}")
        elif hasattr(ev, "evalue"):
            val = float(ev.evalue)
        else:
            val = float(ev)
        # Conventional read: e-value ≥ 2 is "moderately robust", ≥ 4 strong.
        if val >= 4:
            sev, interp = "ok", "≥ 4 — strong robustness to unmeasured confounding"
        elif val >= 2:
            sev, interp = "info", "≥ 2 — moderate robustness; sensible to unmeasured confounders of similar strength"
        else:
            sev, interp = "warning", "< 2 — modest unmeasured confounding could overturn the result"
        findings.append(RobustnessFinding(
            name="evalue",
            label="E-value (CI-limit, VanderWeele-Ding)",
            value=val,
            severity=sev,
            interpretation=interp,
        ))
    except Exception as exc:
        # Don't swamp the section with check_failed when the result
        # simply doesn't carry the right shape — only record at a low
        # severity so the agent knows the check ran but didn't apply.
        findings.append(RobustnessFinding(
            name="evalue",
            label="E-value (VanderWeele-Ding)",
            value=None,
            severity="check_failed",
            interpretation=f"e-value computation failed: {type(exc).__name__}",
        ))


def _add_oster_bounds(
    result: Any, data: Optional[pd.DataFrame],
    treatment: Optional[str], outcome: Optional[str],
    covariates: Optional[List[str]],
    findings: List[RobustnessFinding], notes: List[str],
) -> None:
    if data is None or treatment is None or outcome is None or not covariates:
        return  # Required inputs missing — skip silently
    try:
        from .. import oster_bounds  # type: ignore
        # ``oster_bounds`` (1.x) takes ``controls=`` and returns a dict
        # whose key for δ* is ``delta_for_zero`` (the value of δ that
        # would drive the bias-adjusted β to zero); ``beta_adjusted`` is
        # the bias-adjusted point estimate at the supplied δ.
        ob = oster_bounds(
            data=data, y=outcome, treat=treatment, controls=list(covariates),
        )
        if not isinstance(ob, dict):
            ob = {k: getattr(ob, k) for k in dir(ob) if not k.startswith("_")}
        delta_star = float(ob.get("delta_for_zero", float("nan")))
        beta_adj = float(ob.get("beta_adjusted", float("nan")))
        robust = bool(ob.get("robust", abs(delta_star) >= 1.0))
        sev = "ok" if robust else ("info" if abs(delta_star) >= 0.5 else "warning")
        findings.append(RobustnessFinding(
            name="oster_delta_star",
            label="Oster δ*-for-zero (selection-ratio threshold)",
            value=delta_star,
            severity=sev,
            interpretation=(
                "|δ*| ≥ 1 — unobservables would have to be at least as "
                "important as observables to flip the sign"
                if abs(delta_star) >= 1 else
                f"|δ*| ≈ {abs(delta_star):.2f} — modest unobservable "
                "selection could drive β to zero"
            ),
        ))
        findings.append(RobustnessFinding(
            name="oster_beta_adjusted",
            label="Oster β* (bias-adjusted estimate at δ=1)",
            value=beta_adj,
            severity="info",
        ))
    except Exception as exc:
        notes.append(
            f"Oster bounds skipped: {type(exc).__name__}: {exc}"
        )


def _add_sensemakr(
    result: Any, data: Optional[pd.DataFrame],
    treatment: Optional[str], outcome: Optional[str],
    covariates: Optional[List[str]],
    findings: List[RobustnessFinding], notes: List[str],
) -> None:
    if data is None or treatment is None or outcome is None or not covariates:
        return
    try:
        from .. import sensemakr  # type: ignore
        # Cinelli-Hazlett (2020) sensemakr.  v1.x returns a dict with
        # ``rv_q`` (RV at q=1) and ``rv_qa`` (RV at q=1 with α=0.05).
        sm = sensemakr(
            data=data, y=outcome, treat=treatment, controls=list(covariates),
        )
        if not isinstance(sm, dict):
            sm = {k: getattr(sm, k) for k in dir(sm) if not k.startswith("_")}
        rv_q = float(sm.get("rv_q", float("nan")))
        rv_qa = float(sm.get("rv_qa", float("nan")))
        sev = "ok" if rv_q >= 0.10 else ("info" if rv_q >= 0.05 else "warning")
        findings.append(RobustnessFinding(
            name="sensemakr_rv",
            label="Sensemakr Robustness Value (RV, q=1)",
            value=rv_q,
            severity=sev,
            interpretation=(
                f"RV = {rv_q:.3f} — confounders explaining ≥ {rv_q*100:.1f}% "
                "of treatment & outcome residual variance would null the result"
            ),
        ))
        if np.isfinite(rv_qa):
            findings.append(RobustnessFinding(
                name="sensemakr_rv_qa",
                label="Sensemakr RV (q=1, α=0.05 — significance threshold)",
                value=rv_qa,
                severity="info",
            ))
    except Exception as exc:
        notes.append(
            f"Sensemakr skipped: {type(exc).__name__}: {exc}"
        )


def _add_model_info_diagnostics(result: Any, findings: List[RobustnessFinding]) -> None:
    """Surface ``result.model_info['diagnostics']`` if present (e.g. DML
    diagnostics block: propensity range, partial corr, n clipped, ...)."""
    mi = getattr(result, "model_info", None) or {}
    diags = mi.get("diagnostics")
    if not isinstance(diags, dict) or not diags:
        return
    findings.append(RobustnessFinding(
        name="estimator_diagnostics",
        label="Estimator self-diagnostics",
        value=diags,
        severity="info",
        interpretation=(
            "Block exposed by the estimator's `model_info['diagnostics']`; "
            "see the source estimator for interpretation."
        ),
    ))


def _check_failed(check_name: str, exc: Exception) -> RobustnessFinding:
    return RobustnessFinding(
        name=f"{check_name}_failed",
        label=f"Check `{check_name}` failed",
        value=None,
        severity="check_failed",
        interpretation=f"{type(exc).__name__}: {exc}",
    )


# Design tags shipped by the public API map to a smaller internal
# taxonomy that selects which sub-checks the battery runs.  Tags from
# both ``CausalWorkflow`` (free-form: 'observational' / 'did' / 'iv' /
# 'rd' / 'synth' / 'panel' / ...) and ``CausalQuestion`` (closed
# enum: 'selection_on_observables' / 'dml' / 'tmle' /
# 'regression_discontinuity' / 'natural_experiment' / ...) flow in;
# unknown tags collapse to "" — generic-only checks are still run.
_DESIGN_ALIASES: Dict[str, str] = {
    # observational family
    "observational": "observational",
    "selection_on_observables": "observational",
    "longitudinal_observational": "observational",
    "dml": "observational",
    "tmle": "observational",
    "metalearner": "observational",
    "causal_forest": "observational",
    "rct": "observational",  # randomised, but the obs-style checks still apply
    # DiD family
    "did": "did",
    "event_study": "did",
    # IV family
    "iv": "iv",
    "natural_experiment": "iv",
    "policy_shock": "iv",
    # RD family
    "rd": "rd",
    "regression_discontinuity": "rd",
    # synth — no design-specific battery yet (future: placebo, RMSPE)
    "synth": "",
    "synthetic_control": "",
}


def _normalise_design(design: Optional[str]) -> str:
    if not design:
        return ""
    return _DESIGN_ALIASES.get(str(design).lower(), str(design).lower())


__all__ = [
    "RobustnessFinding",
    "RobustnessReport",
    "run_robustness_battery",
]
