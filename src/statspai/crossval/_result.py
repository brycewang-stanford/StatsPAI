"""The result object returned by :func:`statspai.cross_validate`.

Bundles the per-engine estimates, the agreement verdict and reproducibility
provenance into one object that renders for humans (``.summary()``, ``.plot()``,
``.to_markdown()``, ``.to_latex()``) and for agents (``.to_dict()`` /
``.for_agent()`` carry the verdict, the per-engine table and concrete
next steps).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ._agreement import (
    VERDICT_AGREE,
    VERDICT_DISAGREE,
    VERDICT_INSUFFICIENT,
    VERDICT_PARTIAL,
    AgreementReport,
    EngineEstimate,
)

try:  # SummaryText gives nice terminal + notebook rendering; degrade to str.
    from ..core.results import SummaryText
except Exception:  # noqa: BLE001  pragma: no cover
    SummaryText = str  # type: ignore[assignment,misc]


_VERDICT_GLYPH = {
    VERDICT_AGREE: "✓ AGREE",
    VERDICT_PARTIAL: "~ PARTIAL",
    VERDICT_DISAGREE: "✗ DISAGREE",
    VERDICT_INSUFFICIENT: "? INSUFFICIENT",
}

_VERDICT_GLOSS = {
    VERDICT_AGREE: (
        "Independent engines reproduce the estimate within tolerance — the "
        "number is engine-robust."
    ),
    VERDICT_PARTIAL: (
        "Point estimates agree but inference (standard errors) does not — "
        "check that every engine used the same variance estimator."
    ),
    VERDICT_DISAGREE: (
        "Engines disagree beyond tolerance — the result is implementation-"
        "sensitive. Reconcile the model specification before trusting it."
    ),
    VERDICT_INSUFFICIENT: (
        "Fewer than two engines produced an estimate — cross-validation could "
        "not run. Install another backend (see notes) and retry."
    ),
}


class CrossValidationResult:
    """Outcome of cross-validating one estimand across several engines.

    Attributes
    ----------
    estimand : str
    term : str
        Focal coefficient that was reconciled.
    estimates : list of EngineEstimate
        Every engine that was requested (including unavailable / errored ones).
    agreement : AgreementReport
        Verdict + spread diagnostics.
    spec : dict
        Serialised :class:`EstimandSpec` (what was fit).
    provenance : dict
        Engine versions / environment captured for reproducibility.
    degradations : list of dict
        Structured records for every engine that could not contribute, mirrored
        from :func:`statspai.workflow.record_degradation`.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame(
    ...     {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 0.0, 1.0]}
    ... )
    >>> cv = sp.cross_validate(
    ...     df, "ols", formula="y ~ x", treatment="x", engines=["statspai"]
    ... )
    >>> isinstance(cv, sp.CrossValidationResult)
    True
    >>> cv.term
    'x'
    """

    def __init__(
        self,
        *,
        estimand: str,
        term: str,
        estimates: List[EngineEstimate],
        agreement: AgreementReport,
        spec: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
        degradations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.estimand = estimand
        self.term = term
        self.estimates = estimates
        self.agreement = agreement
        self.spec = spec
        self.provenance = provenance or {}
        self.degradations = degradations or []

    # -- verdict shortcuts ----------------------------------------------- #
    @property
    def verdict(self) -> str:
        return self.agreement.verdict

    @property
    def agree(self) -> bool:
        return self.agreement.verdict == VERDICT_AGREE

    # -- tables ----------------------------------------------------------- #
    @property
    def estimates_table(self) -> pd.DataFrame:
        """One row per requested engine (ok and not-ok alike)."""
        return pd.DataFrame([e.to_dict() for e in self.estimates])

    def ok_table(self) -> pd.DataFrame:
        """Only the engines that produced an estimate."""
        return pd.DataFrame([e.to_dict() for e in self.estimates if e.ok])

    @property
    def engine_status_counts(self) -> Dict[str, int]:
        """Count engines by status, including unavailable/error entries."""
        counts: Dict[str, int] = {}
        for e in self.estimates:
            counts[e.status] = counts.get(e.status, 0) + 1
        return counts

    @property
    def can_claim_cross_engine_agreement(self) -> bool:
        """Whether it is honest to report cross-engine agreement."""
        return self.verdict == VERDICT_AGREE and self.agreement.n_ok >= 2

    # -- rendering -------------------------------------------------------- #
    def summary(self) -> Any:
        a = self.agreement
        lines = [
            "=" * 72,
            f"Cross-engine validation — estimand '{self.estimand}', "
            f"focal term '{self.term}'",
            "=" * 72,
            f"{'Engine':<16s}{'Estimate':>12s}{'Std.Err':>12s}"
            f"{'95% CI':>22s}{'  status':>10s}",
            "-" * 72,
        ]
        for e in self.estimates:
            if e.ok:
                ci = (
                    f"[{e.ci_lower:.4f}, {e.ci_upper:.4f}]"
                    if e.ci_lower is not None and e.ci_upper is not None
                    else ""
                )
                lines.append(
                    f"{e.engine:<16s}{e.coef:>12.5f}"
                    f"{(e.se if e.se is not None else float('nan')):>12.5f}"
                    f"{ci:>22s}{'  ok':>10s}"
                )
            else:
                lines.append(
                    f"{e.engine:<16s}{'—':>12s}{'—':>12s}{'':>22s}"
                    f"{('  ' + e.status):>10s}"
                )
        lines.append("-" * 72)
        lines.append(
            f"VERDICT: {_VERDICT_GLYPH.get(a.verdict, a.verdict)}   "
            f"({a.n_ok}/{a.n_requested} engines ran)"
        )
        lines.append(f"  {_VERDICT_GLOSS.get(a.verdict, '')}")
        if a.max_rel_coef_diff is not None:
            lines.append(
                f"  max relative coef difference: {a.max_rel_coef_diff:.2e}"
                f"   (reference: {a.reference})"
            )
        if a.sign_agree is not None:
            lines.append(
                f"  sign agreement: {a.sign_agree}   "
                f"significance agreement: {a.significance_agree}   "
                f"CI overlap: {a.ci_overlap}"
            )
        lines.append(f"  tolerance: {a.policy.mode} mode — {a.policy.rationale}")
        if a.notes:
            lines.append("  notes:")
            for n in a.notes:
                lines.append(f"    • {n}")
        lines.append("=" * 72)
        return SummaryText("\n".join(lines))

    def to_markdown(self) -> str:
        df = self.estimates_table[
            ["engine", "coef", "se", "ci_lower", "ci_upper", "status"]
        ]
        head = (
            f"**Cross-engine validation** — estimand `{self.estimand}`, term "
            f"`{self.term}` → **{self.verdict}**\n\n"
        )
        return head + str(df.to_markdown(index=False))

    def to_latex(
        self, caption: Optional[str] = None, label: Optional[str] = None
    ) -> str:
        df = self.ok_table()[["engine", "coef", "se", "ci_lower", "ci_upper"]]
        cap = caption or (
            f"Cross-engine validation of {self.estimand} "
            f"(focal term {self.term}): {self.verdict}"
        )
        body = df.to_latex(index=False, float_format="%.5f", caption=cap, label=label)
        return str(body)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Forest plot of the engines' estimates with shared-range shading."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for .plot()") from e
        df = self.ok_table()
        if df.empty:  # pragma: no cover
            raise ValueError("No successful engine estimates to plot.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(2.5, len(df) * 0.6)))
        y = range(len(df))
        lo = (df["coef"] - df["ci_lower"]).fillna(0).values
        hi = (df["ci_upper"] - df["coef"]).fillna(0).values
        ax.errorbar(
            df["coef"].values,
            list(y),
            xerr=[lo, hi],
            fmt="o",
            capsize=4,
            color="steelblue",
            elinewidth=2,
            markersize=8,
        )
        ax.set_yticks(list(y))
        ax.set_yticklabels(df["engine"].values)
        ax.axvspan(df["coef"].min(), df["coef"].max(), alpha=0.12, color="orange")
        ax.set_xlabel(f"Estimate of {self.term}")
        ax.set_title(f"Cross-engine validation — {self.verdict}")
        return ax

    # -- agent surface ---------------------------------------------------- #
    def next_steps(self) -> List[str]:
        v = self.verdict
        if v == VERDICT_AGREE:
            return [
                "Estimate is engine-robust; safe to report. Cite the engines "
                "and versions from `.provenance` for reproducibility."
            ]
        if v == VERDICT_PARTIAL:
            return [
                "Align the variance estimator across engines (pass `vcov=`).",
                "If only standard errors differ, point identification is "
                "robust — clarify which SE you report and why.",
            ]
        if v == VERDICT_DISAGREE:
            return [
                "Inspect the per-engine specification: same sample, same "
                "controls, same fixed effects, same treatment coding?",
                "For randomised estimators (DML/forests) widen the tolerance "
                "with `tol={'se_band': 0.5}` or fix learners/seeds.",
            ]
        unavailable = [e.engine for e in self.estimates if e.status == "unavailable"]
        return [
            "Cross-validation needs ≥2 engines. Unavailable: "
            f"{', '.join(unavailable) or 'none reported'}.",
            "Install another backend (e.g. `pip install pyfixest linearmodels`, "
            "or put Rscript/Stata on PATH) and retry.",
        ]

    def to_dict(self, *, detail: str = "agent") -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "estimand": self.estimand,
            "term": self.term,
            "verdict": self.verdict,
            "engines": [e.to_dict() for e in self.estimates],
            "agreement": self.agreement.to_dict(),
            "engine_status_counts": self.engine_status_counts,
            "can_claim_cross_engine_agreement": self.can_claim_cross_engine_agreement,
        }
        if detail == "minimal":
            out["engines"] = [
                {"engine": e["engine"], "coef": e["coef"], "status": e["status"]}
                for e in out["engines"]
            ]
            out.pop("agreement", None)
            return out
        out["next_steps"] = self.next_steps()
        out["spec"] = self.spec
        out["provenance"] = self.provenance
        if self.degradations:
            out["degradations"] = self.degradations
        return out

    def for_agent(self) -> Dict[str, Any]:
        return self.to_dict(detail="agent")

    # -- dunders ---------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"CrossValidationResult(estimand={self.estimand!r}, "
            f"term={self.term!r}, verdict={self.verdict!r}, "
            f"n_ok={self.agreement.n_ok}/{self.agreement.n_requested})"
        )

    def _repr_html_(self) -> str:  # pragma: no cover - notebook nicety
        head = (
            f"<b>Cross-engine validation</b> — estimand <code>{self.estimand}"
            f"</code>, term <code>{self.term}</code>: "
            f"<b>{self.verdict}</b>"
        )
        return head + str(self.estimates_table.to_html(index=False))
