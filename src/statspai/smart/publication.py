"""
Publication Readiness Checklist.

Journal-specific checklists for submitting empirical papers.
Checks that all required robustness tests, tables, and diagnostics
are present before submission.

**No other package does this.** Researchers manually track requirements.
This automates the process with venue-specific checklists.

Usage
-----
>>> import statspai as sp
>>> check = sp.pub_ready(results=[result1, result2],
...                      venue='top5_econ')
>>> print(check.summary())
"""

from typing import Optional, List, Dict, Any


class PubReadyResult:
    """Publication readiness checklist results."""

    def __init__(self, venue, checks, score, missing, present):
        self.venue = venue
        self.checks = checks
        self.score = score  # 0-100
        self.missing = missing
        self.present = present

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Publication Readiness: {self.venue}",
            "=" * 70,
            f"Score: {self.score}/100",
            "",
            "CHECKLIST",
            "─" * 70,
        ]

        for check in self.checks:
            status = "✓" if check['done'] else "○"
            lines.append(f"  {status} {check['item']}")
            if not check['done'] and check.get('how'):
                lines.append(f"    → {check['how']}")

        if self.missing:
            lines.append(f"\nMISSING ({len(self.missing)} items):")
            for m in self.missing:
                lines.append(f"  • {m}")

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)


# Journal-specific checklists
_CHECKLISTS = {
    'top5_econ': {
        'name': 'Top 5 Economics (AER/QJE/JPE/Econometrica/REStud)',
        'items': [
            {'item': 'Summary statistics table', 'category': 'tables',
             'how': 'sp.sumstats(df)'},
            {'item': 'Balance table (if treatment)', 'category': 'tables',
             'how': 'sp.balance_table(df, treatment=...)'},
            {'item': 'Main results table with multiple specs', 'category': 'tables',
             'how': 'sp.regtable([result1, result2, result3])'},
            {'item': 'Robust SE reported', 'category': 'estimation',
             'how': 'Use robust="hc1" or cluster SE'},
            {'item': 'Pre-trend test (if DID)', 'category': 'identification',
             'how': 'sp.pretrends_test(result)'},
            {'item': 'Event study plot (if DID)', 'category': 'identification',
             'how': 'sp.event_study(df, ...)'},
            {'item': 'McCrary density test (if RD)', 'category': 'identification',
             'how': 'sp.rddensity(df, x=...)'},
            {'item': 'First-stage F reported (if IV)', 'category': 'identification',
             'how': 'Check result.diagnostics["first_stage_F"]'},
            {'item': 'Sensitivity to unobservables', 'category': 'robustness',
             'how': 'sp.sensemakr(result) or sp.oster_bounds(result)'},
            {'item': 'Specification curve / robustness table', 'category': 'robustness',
             'how': 'sp.spec_curve(...) or sp.robustness_report(result)'},
            {'item': 'Heterogeneity / subgroup analysis', 'category': 'robustness',
             'how': 'sp.subgroup_analysis(result, ...)'},
            {'item': 'Placebo / falsification tests', 'category': 'robustness',
             'how': 'Use placebo outcomes or fake treatment timing'},
            {'item': 'Multiple hypothesis correction (if multiple outcomes)',
             'category': 'inference',
             'how': 'sp.romano_wolf([result1, result2, ...])'},
            {'item': 'Replication data and code available', 'category': 'reproducibility',
             'how': 'Package data + code for journal dataverse'},
        ],
    },
    'aej_applied': {
        'name': 'AEJ: Applied Economics',
        'items': [
            {'item': 'Summary statistics', 'category': 'tables', 'how': 'sp.sumstats(df)'},
            {'item': 'Main results (3+ specifications)', 'category': 'tables',
             'how': 'sp.regtable([r1, r2, r3])'},
            {'item': 'Robustness checks appendix', 'category': 'robustness',
             'how': 'sp.robustness_report(result)'},
            {'item': 'Mechanism / channel analysis', 'category': 'identification',
             'how': 'sp.mediate(df, ...)'},
            {'item': 'Power analysis (if null result)', 'category': 'inference',
             'how': 'sp.power(...) or sp.mde(...)'},
        ],
    },
    'rct': {
        'name': 'RCT / Field Experiment',
        'items': [
            {'item': 'CONSORT flow diagram', 'category': 'design', 'how': 'Manual'},
            {'item': 'Balance table (treatment vs control)', 'category': 'tables',
             'how': 'sp.balance_check(df, treatment=..., covariates=[...])'},
            {'item': 'Attrition analysis', 'category': 'design',
             'how': 'sp.attrition_test(df, treatment=..., observed=...)'},
            {'item': 'Lee bounds (if differential attrition)', 'category': 'bounds',
             'how': 'sp.lee_bounds(df, y=..., treatment=...)'},
            {'item': 'ITT estimate', 'category': 'estimation',
             'how': 'sp.regress("y ~ treatment + controls", data=df)'},
            {'item': 'LATE estimate (if non-compliance)', 'category': 'estimation',
             'how': 'sp.iv(df, y=..., x_endog=["takeup"], z=["assignment"])'},
            {'item': 'MHT correction (multiple outcomes)', 'category': 'inference',
             'how': 'sp.romano_wolf(results)'},
            {'item': 'Heterogeneous effects', 'category': 'robustness',
             'how': 'sp.subgroup_analysis() or sp.causal_forest()'},
            {'item': 'Power calculation', 'category': 'design',
             'how': 'sp.power_rct(mde=..., sigma=...)'},
            {'item': 'Pre-analysis plan alignment', 'category': 'reproducibility',
             'how': 'Compare results to PAP specifications'},
        ],
    },
}


def pub_ready(
    results: list = None,
    venue: str = 'top5_econ',
    design: str = None,
    has_balance: bool = False,
    has_pretrends: bool = False,
    has_robustness: bool = False,
    has_heterogeneity: bool = False,
    has_sensitivity: bool = False,
    has_placebo: bool = False,
    has_mht: bool = False,
) -> PubReadyResult:
    """
    Publication readiness checklist.

    **Unique to StatsPAI.** Generates a venue-specific checklist
    for empirical paper submission.

    Parameters
    ----------
    results : list, optional
        List of estimated result objects.
    venue : str, default 'top5_econ'
        Target venue: 'top5_econ', 'aej_applied', 'rct'.
    design : str, optional
        Research design: 'rct', 'did', 'rd', 'iv', 'observational'.
    has_balance : bool
        Already have balance table.
    has_pretrends : bool
        Already have pre-trend tests.
    has_robustness : bool
        Already have robustness checks.
    has_heterogeneity : bool
        Already have subgroup analysis.
    has_sensitivity : bool
        Already have sensitivity analysis.
    has_placebo : bool
        Already have placebo tests.
    has_mht : bool
        Already have MHT correction.

    Returns
    -------
    PubReadyResult

    Examples
    --------
    >>> import statspai as sp
    >>> check = sp.pub_ready(results=[r1, r2], venue='top5_econ', design='did')
    >>> print(check.summary())
    """
    if venue not in _CHECKLISTS:
        venue = 'top5_econ'

    checklist = _CHECKLISTS[venue]
    items = checklist['items']

    # Auto-detect what's been done
    done_flags = {
        'balance': has_balance,
        'pretrends': has_pretrends,
        'robustness': has_robustness,
        'heterogeneity': has_heterogeneity,
        'sensitivity': has_sensitivity,
        'placebo': has_placebo,
        'mht': has_mht,
    }

    # Check results for what's been computed
    if results:
        done_flags['main_results'] = True
        for r in results:
            mi = getattr(r, 'model_info', {})
            di = getattr(r, 'diagnostics', {})
            if 'robust' in str(mi) or 'hc1' in str(mi):
                done_flags['robust_se'] = True

    checks = []
    present = []
    missing = []

    for item_info in items:
        item = item_info['item']
        # Simple heuristic matching
        done = False
        if 'balance' in item.lower() and done_flags.get('balance'):
            done = True
        elif 'pre-trend' in item.lower() and done_flags.get('pretrends'):
            done = True
        elif 'robustness' in item.lower() and done_flags.get('robustness'):
            done = True
        elif 'heterogen' in item.lower() and done_flags.get('heterogeneity'):
            done = True
        elif 'sensitivity' in item.lower() and done_flags.get('sensitivity'):
            done = True
        elif 'placebo' in item.lower() and done_flags.get('placebo'):
            done = True
        elif 'multiple hypothesis' in item.lower() and done_flags.get('mht'):
            done = True
        elif 'main results' in item.lower() and done_flags.get('main_results'):
            done = True
        elif 'robust se' in item.lower() and done_flags.get('robust_se'):
            done = True

        # Filter by design relevance
        skip = False
        if 'DID' in item and design not in [None, 'did']:
            skip = True
        if 'RD' in item and design not in [None, 'rd']:
            skip = True
        if 'IV' in item and design not in [None, 'iv']:
            skip = True
        if 'attrition' in item.lower() and design not in [None, 'rct']:
            skip = True

        if skip:
            continue

        checks.append({'item': item, 'done': done, 'how': item_info.get('how')})
        if done:
            present.append(item)
        else:
            missing.append(item)

    total = len(checks)
    n_done = len(present)
    score = int(100 * n_done / total) if total > 0 else 0

    return PubReadyResult(
        venue=checklist['name'],
        checks=checks,
        score=score,
        missing=missing,
        present=present,
    )
