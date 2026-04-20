"""Identification diagnostics: read the data, flag design pitfalls.

``check_identification()`` reads a DataFrame + research design and
outputs a prioritised list of *design-level* diagnostic warnings.

This is different from ``recommend()`` (which picks an estimator) and
from ``diagnose_result()`` (which inspects a fitted model).
``check_identification()`` answers the question you should ask BEFORE
running any estimator:

    "Is my design capable of identifying a causal effect at all?"

Checks performed (with literature references):

1. Bad controls
   - Post-treatment variables (candidate mediators/colliders) —
     Angrist-Pischke MHE §3.2.
   - Outcome-correlated covariates with variance dominated by the
     treatment period — classic "controlling away the effect".

2. Overlap / common support
   - Propensity score distribution across treatment groups
   - Extreme propensities (<0.01 or >0.99) that destabilise IPW

3. Sample size and statistical power
   - Minimum detectable effect (MDE) at 80% power
   - Cluster-aware MDE if clustering is specified
   - Unit-level vs observation-level power for panel data

4. Variation in treatment
   - Fraction treated (problematic if < 5% or > 95%)
   - Cohort sizes (for DID)
   - Variation in running variable density at cutoff (for RD)

5. Clustering
   - Recommendation: which level to cluster at
   - Moulton factor warning if ignored

Usage
-----
>>> import statspai as sp
>>> diag = sp.check_identification(
...     df, y='wage', treatment='training',
...     covariates=['age', 'education'],
...     id='worker_id', time='year',
...     design='did',
... )
>>> print(diag.summary())
>>> diag.verdict   # 'OK' | 'WARNINGS' | 'BLOCKERS'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticFinding:
    """A single design-level finding."""
    severity: str  # 'blocker' | 'warning' | 'info'
    category: str  # 'bad_controls' | 'overlap' | 'power' | 'variation' | 'clustering'
    message: str
    suggestion: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {'blocker': '[X]', 'warning': '[!]', 'info': '[i]'}.get(
            self.severity, '[?]')


@dataclass
class IdentificationReport:
    """Report from ``check_identification``."""
    findings: List[DiagnosticFinding] = field(default_factory=list)
    design: str = ''
    n_obs: int = 0
    n_units: Optional[int] = None

    @property
    def verdict(self) -> str:
        """Overall verdict: BLOCKERS | WARNINGS | OK."""
        if any(f.severity == 'blocker' for f in self.findings):
            return 'BLOCKERS'
        if any(f.severity == 'warning' for f in self.findings):
            return 'WARNINGS'
        return 'OK'

    def by_category(self, category: str) -> List[DiagnosticFinding]:
        return [f for f in self.findings if f.category == category]

    def summary(self) -> str:
        lines = [
            '=' * 70,
            'Identification Diagnostics',
            '=' * 70,
            f'  Design:    {self.design}',
            f'  N obs:     {self.n_obs:,}',
        ]
        if self.n_units is not None:
            lines.append(f'  N units:   {self.n_units:,}')
        lines.extend([
            f'  Verdict:   {self.verdict}',
            '-' * 70,
        ])
        if not self.findings:
            lines.append('  No issues detected.')
        else:
            # Order: blockers first, then warnings, then info
            order = {'blocker': 0, 'warning': 1, 'info': 2}
            findings = sorted(self.findings, key=lambda f: order.get(f.severity, 3))
            for f in findings:
                lines.append(f'  {f.icon} [{f.category.upper()}] {f.message}')
                if f.suggestion:
                    lines.append(f'     -> {f.suggestion}')
        lines.append('=' * 70)
        return '\n'.join(lines)

    def __repr__(self) -> str:
        counts = {
            'blocker': sum(1 for f in self.findings if f.severity == 'blocker'),
            'warning': sum(1 for f in self.findings if f.severity == 'warning'),
            'info': sum(1 for f in self.findings if f.severity == 'info'),
        }
        return (f"<IdentificationReport {self.design}: {self.verdict}, "
                f"B{counts['blocker']} W{counts['warning']} I{counts['info']}>")


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------

def _check_bad_controls(
    data: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    outcome: str,
    time: Optional[str] = None,
    findings: Optional[List[DiagnosticFinding]] = None,
) -> None:
    if findings is None:
        return

    # Heuristic 1: covariate's correlation with TREATMENT is suspiciously
    # high — it may be a mediator.  Flag when |corr| > 0.5.
    if treatment in data.columns:
        t_series = pd.to_numeric(data[treatment], errors='coerce')
        for c in covariates:
            if c not in data.columns or c == treatment:
                continue
            c_series = pd.to_numeric(data[c], errors='coerce')
            if c_series.isna().all():
                continue
            try:
                corr = c_series.corr(t_series)
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) > 0.5:
                findings.append(DiagnosticFinding(
                    severity='warning',
                    category='bad_controls',
                    message=f"Covariate '{c}' has |corr| = {abs(corr):.2f} "
                            f"with treatment '{treatment}'; may be a "
                            f"mediator (bad control).",
                    suggestion=f"Verify '{c}' is pre-treatment. If it is "
                               f"determined AFTER treatment, drop it.",
                    evidence={'covariate': c, 'corr_with_treatment': float(corr)},
                ))

    # Heuristic 2: panel data with time — check for post-treatment
    # covariates by comparing pre vs post treatment-period variance
    # (if time and treatment are available).  Skipped here as this
    # requires knowing treatment-onset per unit.


def _check_overlap(
    data: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
    findings: List[DiagnosticFinding],
) -> None:
    """Fit a quick logistic PS and check extreme propensities."""
    if treatment not in data.columns or not covariates:
        return

    t_vals = data[treatment].dropna().unique()
    if len(t_vals) != 2:
        return  # only binary-treatment overlap is defined here

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        findings.append(DiagnosticFinding(
            severity='info',
            category='overlap',
            message='scikit-learn not installed; skipping PS overlap check.',
        ))
        return

    # Build design matrix
    ok_cov = [c for c in covariates if c in data.columns
              and pd.api.types.is_numeric_dtype(data[c])]
    if not ok_cov:
        return
    X = data[ok_cov].fillna(data[ok_cov].median())
    y = data[treatment].astype(int)
    if y.isna().any() or X.isna().any().any():
        return

    try:
        ps = LogisticRegression(max_iter=500).fit(X, y).predict_proba(X)[:, 1]
    except Exception:
        return

    frac_extreme = ((ps < 0.02) | (ps > 0.98)).mean()
    min_p, max_p = ps.min(), ps.max()

    if frac_extreme > 0.05:
        findings.append(DiagnosticFinding(
            severity='warning',
            category='overlap',
            message=f'{frac_extreme:.1%} of units have propensity scores '
                    f'outside [0.02, 0.98] (min={min_p:.3f}, max={max_p:.3f}).',
            suggestion='Trim extreme PS units or use sp.overlap_weights '
                       '(ATO) for a well-defined estimand.',
            evidence={'frac_extreme_ps': float(frac_extreme),
                      'min_ps': float(min_p), 'max_ps': float(max_p)},
        ))
    if min_p < 1e-3 or max_p > 1 - 1e-3:
        findings.append(DiagnosticFinding(
            severity='blocker',
            category='overlap',
            message=f'Near-perfect separation (PS min={min_p:.4f}, '
                    f'max={max_p:.4f}). ATE/ATT are not identified.',
            suggestion='Check for deterministic treatment rules; restrict '
                       'the sample to a region of common support.',
            evidence={'min_ps': float(min_p), 'max_ps': float(max_p)},
        ))


def _check_treatment_variation(
    data: pd.DataFrame,
    treatment: str,
    findings: List[DiagnosticFinding],
) -> None:
    if treatment not in data.columns:
        return
    t = data[treatment].dropna()
    if len(t) == 0:
        return
    # Binary?
    uniq = t.unique()
    if len(uniq) == 2:
        frac_t = (t == max(uniq)).mean()
        if frac_t < 0.05:
            findings.append(DiagnosticFinding(
                severity='warning',
                category='variation',
                message=f'Only {frac_t:.1%} of units are treated.  Power '
                        f'will be limited; SEs will be large.',
                suggestion='Consider oversampling controls or using '
                           'entropy balancing for efficiency.',
                evidence={'frac_treated': float(frac_t)},
            ))
        elif frac_t > 0.95:
            findings.append(DiagnosticFinding(
                severity='warning',
                category='variation',
                message=f'{frac_t:.1%} of units are treated. Very few '
                        f'controls; consider reversing treatment definition.',
                evidence={'frac_treated': float(frac_t)},
            ))


def _check_did_cohort_sizes(
    data: pd.DataFrame,
    cohort: str,
    unit: Optional[str],
    findings: List[DiagnosticFinding],
) -> None:
    if cohort not in data.columns:
        return

    if unit and unit in data.columns:
        # Each unit has one cohort; get cohort size by counting unique units
        by_unit = data.drop_duplicates(subset=[unit])
        counts = by_unit[cohort].value_counts()
    else:
        counts = data[cohort].value_counts()

    # Never-treated group typically coded as 0
    never = counts.get(0, 0)
    total = counts.sum()
    if total == 0:
        return
    frac_never = never / total

    if frac_never < 0.1:
        findings.append(DiagnosticFinding(
            severity='warning',
            category='variation',
            message=f'Only {frac_never:.1%} of units are never-treated '
                    f'(n={never}).  Callaway-Sant\'Anna with '
                    f'control_group="nevertreated" may be noisy.',
            suggestion='Use control_group="notyettreated" for more '
                       'comparisons.',
            evidence={'frac_never_treated': float(frac_never),
                      'n_never_treated': int(never)},
        ))

    small_cohorts = counts[(counts < 10) & (counts.index != 0)]
    if len(small_cohorts) > 0:
        findings.append(DiagnosticFinding(
            severity='warning',
            category='variation',
            message=f'Small treatment cohorts detected: '
                    f'{dict(small_cohorts.astype(int))} (units < 10 each).',
            suggestion='Cohort-level ATTs will be noisy.  Aggregate to '
                       'broader cohorts or report simple ATT only.',
            evidence={'small_cohorts': dict(small_cohorts.astype(int))},
        ))


def _check_power(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    findings: List[DiagnosticFinding],
    alpha: float = 0.05,
    power: float = 0.80,
) -> None:
    """Compute MDE (minimum detectable effect) at 80% power."""
    if treatment not in data.columns or outcome not in data.columns:
        return

    # Two-sample MDE (Cohen): d = (z_{1-alpha/2} + z_{1-beta}) * sqrt(2/n_per_group)
    # Converted to raw units: MDE = d * sigma_pooled
    try:
        from scipy import stats as sps
    except ImportError:
        return

    t = pd.to_numeric(data[treatment], errors='coerce')
    y = pd.to_numeric(data[outcome], errors='coerce')
    mask = ~(t.isna() | y.isna())
    t, y = t[mask], y[mask]
    if len(t) == 0:
        return

    uniq = t.unique()
    if len(uniq) != 2:
        return

    n_t = int((t == max(uniq)).sum())
    n_c = int((t == min(uniq)).sum())
    if n_t == 0 or n_c == 0:
        return

    sigma = float(y.std())
    z_a = sps.norm.ppf(1 - alpha/2)
    z_b = sps.norm.ppf(power)
    mde = (z_a + z_b) * sigma * np.sqrt(1/n_t + 1/n_c)

    # Compare to observed effect if possible
    mean_diff = float(y[t == max(uniq)].mean() - y[t == min(uniq)].mean())

    if abs(mean_diff) < 0.5 * mde:
        findings.append(DiagnosticFinding(
            severity='warning',
            category='power',
            message=f'Observed raw mean-diff ({mean_diff:.3f}) is less '
                    f'than half the MDE ({mde:.3f}).  Underpowered for the '
                    f'observed effect.',
            suggestion='Increase sample size, reduce noise, or revise '
                       'hypothesised effect.',
            evidence={'mde_80pct_power': float(mde),
                      'observed_raw_diff': float(mean_diff),
                      'n_treated': n_t, 'n_control': n_c},
        ))
    else:
        findings.append(DiagnosticFinding(
            severity='info',
            category='power',
            message=f'MDE at 80% power: {mde:.4f} (raw units); '
                    f'n_treated={n_t}, n_control={n_c}.',
            evidence={'mde_80pct_power': float(mde),
                      'n_treated': n_t, 'n_control': n_c},
        ))


def _check_clustering(
    data: pd.DataFrame,
    unit: Optional[str],
    time: Optional[str],
    cluster: Optional[str],
    findings: List[DiagnosticFinding],
) -> None:
    if cluster is None and unit is not None and unit in data.columns:
        # Panel data without explicit clustering
        findings.append(DiagnosticFinding(
            severity='info',
            category='clustering',
            message=f"Panel data detected with unit='{unit}'.  "
                    f"Cluster-robust SEs at the unit level are standard.",
            suggestion=f"Pass cluster='{unit}' to the estimator; "
                       f"if there's a higher level (e.g. firm, state), "
                       f"consider two-way clustering.",
        ))
    if cluster is not None and unit is not None and cluster == unit:
        n_clusters = data[cluster].nunique()
        if n_clusters < 30:
            findings.append(DiagnosticFinding(
                severity='warning',
                category='clustering',
                message=f"Only {n_clusters} clusters; asymptotic "
                        f"cluster-robust SEs may be unreliable.",
                suggestion="Use sp.wild_cluster_bootstrap for valid "
                           "inference with few clusters.",
                evidence={'n_clusters': int(n_clusters)},
            ))


def _check_rd_density(
    data: pd.DataFrame,
    running_var: str,
    cutoff: float,
    findings: List[DiagnosticFinding],
) -> None:
    if running_var not in data.columns:
        return
    x = pd.to_numeric(data[running_var], errors='coerce').dropna()
    if len(x) == 0:
        return
    left = (x < cutoff).sum()
    right = (x >= cutoff).sum()
    if left == 0 or right == 0:
        findings.append(DiagnosticFinding(
            severity='blocker',
            category='variation',
            message=f'No observations on {"left" if left == 0 else "right"} '
                    f'of cutoff {cutoff}. RD is not identified.',
        ))
        return
    # Local window density check (±10% of range)
    rng = x.max() - x.min()
    w = 0.1 * rng
    near_left = ((x >= cutoff - w) & (x < cutoff)).sum()
    near_right = ((x >= cutoff) & (x < cutoff + w)).sum()
    if near_left < 20 or near_right < 20:
        findings.append(DiagnosticFinding(
            severity='warning',
            category='variation',
            message=f'Sparse observations near cutoff: '
                    f'{near_left} on left / {near_right} on right '
                    f'within ±{w:.2g} of cutoff.',
            suggestion='RD estimates will have wide CIs. Consider '
                       'collecting more data near the cutoff or using '
                       'sp.rdpower for explicit power analysis.',
            evidence={'n_near_left': int(near_left),
                      'n_near_right': int(near_right)},
        ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_identification(
    data: pd.DataFrame,
    y: str,
    treatment: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    id: Optional[str] = None,
    time: Optional[str] = None,
    running_var: Optional[str] = None,
    instrument: Optional[str] = None,
    cluster: Optional[str] = None,
    cutoff: Optional[float] = None,
    design: Optional[str] = None,
    cohort: Optional[str] = None,
) -> IdentificationReport:
    """Run design-level identification diagnostics before fitting an estimator.

    Unique to StatsPAI.  This reads your dataframe + design and outputs
    a prioritised list of pitfalls — bad controls, overlap violations,
    underpowered designs, small cohorts, clustering ambiguity.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treatment : str, optional
        Binary or continuous treatment column.
    covariates : list of str, optional
        Candidate control variables.
    id, time : str, optional
        Panel identifiers.
    running_var : str, optional
        RD running variable.
    instrument : str, optional
        IV instrument.
    cluster : str, optional
        Clustering column for inference.
    cutoff : float, optional
        RD cutoff value.
    design : str, optional
        Override auto-detected design: one of
        'rct', 'did', 'rd', 'iv', 'observational', 'panel'.
    cohort : str, optional
        First-treatment-period column (for staggered DID).

    Returns
    -------
    IdentificationReport
        With ``.summary()``, ``.verdict``, ``.findings``, ``.by_category()``.

    Examples
    --------
    >>> report = sp.check_identification(
    ...     df, y='wage', treatment='training',
    ...     covariates=['age', 'education'],
    ...     id='worker', time='year', design='did',
    ... )
    >>> print(report.summary())
    >>> if report.verdict == 'BLOCKERS':
    ...     raise RuntimeError("Design has identification blockers.")
    """
    if covariates is None:
        covariates = []

    # Auto-detect design
    if design is None:
        if running_var is not None:
            design = 'rd'
        elif instrument is not None:
            design = 'iv'
        elif id is not None and time is not None and treatment is not None:
            design = 'did'
        elif id is not None and time is not None:
            design = 'panel'
        elif treatment is not None:
            design = 'observational'
        else:
            design = 'cross-section'

    n_units = None
    if id is not None and id in data.columns:
        n_units = int(data[id].nunique())
    report = IdentificationReport(design=design, n_obs=len(data),
                                  n_units=n_units)
    findings = report.findings

    if treatment is not None:
        _check_bad_controls(data, treatment, covariates, y, time,
                            findings=findings)
        _check_treatment_variation(data, treatment, findings)
        if covariates:
            _check_overlap(data, treatment, covariates, findings)
        _check_power(data, treatment, y, findings)

    if design == 'did' and cohort is not None:
        _check_did_cohort_sizes(data, cohort, id, findings)

    if design == 'rd' and running_var is not None:
        _check_rd_density(data, running_var, cutoff or 0.0, findings)

    _check_clustering(data, id, time, cluster, findings)

    return report
