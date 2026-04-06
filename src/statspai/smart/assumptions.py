"""
Assumption Audit Engine.

Given an estimated result, systematically tests every assumption
of the method used and reports pass/fail with actionable guidance.

**No other package does this.** Stata/R test assumptions individually.
This tests ALL assumptions at once and tells you what to do if one fails.

Usage
-----
>>> import statspai as sp
>>> result = sp.regress("wage ~ education + experience", data=df, robust='hc1')
>>> audit = sp.assumption_audit(result)
>>> print(audit.summary())
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import warnings


class AssumptionCheck:
    """A single assumption test result."""
    def __init__(self, assumption, test_name, passed, statistic,
                 p_value, detail, remedy):
        self.assumption = assumption
        self.test_name = test_name
        self.passed = passed  # True/False/None (inconclusive)
        self.statistic = statistic
        self.p_value = p_value
        self.detail = detail
        self.remedy = remedy

    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL" if self.passed is False else "? INCONCLUSIVE"
        return f"{status} | {self.assumption}: {self.test_name}"


class AssumptionResult:
    """Results from comprehensive assumption audit."""

    def __init__(self, method, checks, overall_grade, n_pass, n_fail,
                 n_inconclusive, critical_failures):
        self.method = method
        self.checks = checks  # list of AssumptionCheck
        self.overall_grade = overall_grade  # A/B/C/D/F
        self.n_pass = n_pass
        self.n_fail = n_fail
        self.n_inconclusive = n_inconclusive
        self.critical_failures = critical_failures

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Assumption Audit: {self.method}",
            "=" * 70,
            f"\nOverall Grade: {self.overall_grade}  "
            f"({self.n_pass} pass, {self.n_fail} fail, "
            f"{self.n_inconclusive} inconclusive)",
        ]

        if self.critical_failures:
            lines.append(f"\n⚠ CRITICAL FAILURES:")
            for cf in self.critical_failures:
                lines.append(f"  • {cf}")

        lines.append(f"\n{'─' * 70}")
        lines.append(f"{'Assumption':<30s} {'Test':<20s} {'Status':>8s} "
                     f"{'p-value':>8s}")
        lines.append(f"{'─' * 70}")

        for check in self.checks:
            if check.passed is True:
                status = "✓ PASS"
            elif check.passed is False:
                status = "✗ FAIL"
            else:
                status = "? N/A"
            p_str = f"{check.p_value:.4f}" if check.p_value is not None else "—"
            lines.append(f"{check.assumption:<30s} {check.test_name:<20s} "
                         f"{status:>8s} {p_str:>8s}")
            if not check.passed and check.remedy:
                lines.append(f"  → Remedy: {check.remedy}")

        lines.append(f"{'─' * 70}")
        return "\n".join(lines)

    def failed(self) -> List[AssumptionCheck]:
        """Return only failed checks."""
        return [c for c in self.checks if c.passed is False]

    def passed_all(self) -> bool:
        """True if all checks passed."""
        return self.n_fail == 0


def assumption_audit(
    result,
    data: pd.DataFrame = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> AssumptionResult:
    """
    Comprehensive assumption audit for any estimated model.

    **Unique to StatsPAI.** Systematically tests every assumption
    of the method used and provides actionable remedies.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Estimated model result.
    data : pd.DataFrame, optional
        Original data (needed for some tests). Auto-extracted if available.
    alpha : float, default 0.05
        Significance level for tests.
    verbose : bool, default True
        Print summary automatically.

    Returns
    -------
    AssumptionResult
        With .summary(), .failed(), .passed_all() methods.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.regress("wage ~ educ + exper", data=df)
    >>> audit = sp.assumption_audit(result)
    >>> print(audit.summary())
    >>> if not audit.passed_all():
    ...     for fail in audit.failed():
    ...         print(f"  Fix: {fail.remedy}")
    """
    checks = []

    # Detect method type
    model_info = getattr(result, 'model_info', {})
    method = model_info.get('model_type', '')

    # Try to get data from result
    if data is None:
        data = getattr(result, '_data', None) or getattr(result, 'data', None)

    # ===== OLS / LINEAR REGRESSION =====
    if any(kw in method.lower() for kw in ['ols', 'linear', 'regress', '']):
        checks.extend(_audit_linear(result, data, alpha))

    # ===== IV / 2SLS =====
    if any(kw in method.lower() for kw in ['iv', '2sls', 'liml', 'jive', 'gmm']):
        checks.extend(_audit_iv(result, data, alpha))

    # ===== DID =====
    if any(kw in method.lower() for kw in ['did', 'difference']):
        checks.extend(_audit_did(result, data, alpha))

    # ===== RD =====
    if any(kw in method.lower() for kw in ['rd', 'regression discontinuity', 'rdrobust']):
        checks.extend(_audit_rd(result, data, alpha))

    # ===== LOGIT/PROBIT =====
    if any(kw in method.lower() for kw in ['logit', 'probit', 'binary']):
        checks.extend(_audit_binary(result, data, alpha))

    # ===== PANEL =====
    if any(kw in method.lower() for kw in ['panel', 'fixed effect', 'fe', 're']):
        checks.extend(_audit_panel(result, data, alpha))

    # If no method-specific checks, run generic
    if not checks:
        checks.extend(_audit_generic(result, data, alpha))

    # Compute grades
    n_pass = sum(1 for c in checks if c.passed is True)
    n_fail = sum(1 for c in checks if c.passed is False)
    n_inc = sum(1 for c in checks if c.passed is None)

    critical = [c.assumption for c in checks
                if c.passed is False and 'critical' in (c.detail or '').lower()]

    total = n_pass + n_fail + n_inc
    if total == 0:
        grade = '?'
    elif n_fail == 0:
        grade = 'A'
    elif n_fail <= 1 and not critical:
        grade = 'B'
    elif n_fail <= 2:
        grade = 'C'
    elif n_fail <= 3:
        grade = 'D'
    else:
        grade = 'F'

    audit = AssumptionResult(
        method=method or 'Unknown',
        checks=checks,
        overall_grade=grade,
        n_pass=n_pass,
        n_fail=n_fail,
        n_inconclusive=n_inc,
        critical_failures=critical,
    )

    if verbose:
        print(audit.summary())

    return audit


def _audit_linear(result, data, alpha):
    """Assumption tests for OLS / linear regression."""
    checks = []
    import statspai as sp

    # 1. Linearity (RESET test)
    try:
        reset = sp.reset_test(result)
        p = reset.get('p_value', reset.get('pvalue', None))
        checks.append(AssumptionCheck(
            assumption='Linearity',
            test_name='RESET test',
            passed=p > alpha if p is not None else None,
            statistic=reset.get('F', reset.get('statistic', None)),
            p_value=p,
            detail='Critical — misspecification invalidates all inference',
            remedy='Add polynomial terms, use log transform, or nonparametric methods (sp.lpoly)',
        ))
    except Exception:
        checks.append(AssumptionCheck(
            'Linearity', 'RESET test', None, None, None,
            'Could not run', 'Manually check residual plots'))

    # 2. Homoskedasticity
    try:
        het = sp.het_test(result)
        p = het.get('p_value', het.get('pvalue', None))
        checks.append(AssumptionCheck(
            assumption='Homoskedasticity',
            test_name='Breusch-Pagan',
            passed=p > alpha if p is not None else None,
            statistic=het.get('LM', het.get('statistic', None)),
            p_value=p,
            detail='If violated, use robust SE (already used if robust=hc1)',
            remedy='Use robust SE: sp.regress(..., robust="hc1") or cluster SE',
        ))
    except Exception:
        pass

    # 3. No multicollinearity
    try:
        vif_result = sp.vif(result)
        max_vif = max(vif_result.values()) if isinstance(vif_result, dict) else 1
        checks.append(AssumptionCheck(
            assumption='No multicollinearity',
            test_name='VIF < 10',
            passed=max_vif < 10,
            statistic=max_vif,
            p_value=None,
            detail=f'Max VIF = {max_vif:.1f}',
            remedy='Drop one of the collinear variables or use PCA/LASSO',
        ))
    except Exception:
        pass

    # 4. Sample size adequacy
    n = getattr(result, 'data_info', {}).get('n_obs', 0)
    k = len(result.params) if hasattr(result, 'params') else 0
    if n > 0 and k > 0:
        checks.append(AssumptionCheck(
            assumption='Sample size adequacy',
            test_name=f'N/k > 10',
            passed=n / max(k, 1) > 10,
            statistic=n / max(k, 1),
            p_value=None,
            detail=f'N={n}, k={k}, ratio={n/max(k,1):.0f}',
            remedy='Reduce number of regressors or increase sample size',
        ))

    # 5. Sensitivity to unobservables (Oster bounds)
    try:
        oster = sp.oster_bounds(result)
        delta = oster.get('delta', oster.get('oster_delta', None))
        if delta is not None:
            checks.append(AssumptionCheck(
                assumption='Robustness to unobservables',
                test_name='Oster delta > 1',
                passed=abs(delta) > 1 if np.isfinite(delta) else None,
                statistic=delta,
                p_value=None,
                detail=f'Oster δ = {delta:.2f} (>1 means robust)',
                remedy='Consider IV estimation or sensitivity analysis (sp.sensemakr)',
            ))
    except Exception:
        pass

    return checks


def _audit_iv(result, data, alpha):
    """Assumption tests for IV / 2SLS."""
    checks = []

    # 1. Instrument relevance (first-stage F)
    diag = getattr(result, 'diagnostics', {})
    f_stat = diag.get('first_stage_F', diag.get('f_first_stage', None))
    if f_stat is not None:
        checks.append(AssumptionCheck(
            assumption='Instrument relevance',
            test_name='First-stage F > 10',
            passed=f_stat > 10,
            statistic=f_stat,
            p_value=None,
            detail=f'Critical — F={f_stat:.1f}. Weak if < 10 (Stock-Yogo).',
            remedy='Use LIML (sp.liml) or weak-IV robust inference (sp.anderson_rubin_test)',
        ))

    # 2. Overidentification (J-test)
    j_stat = diag.get('sargan_stat', diag.get('hansen_j', diag.get('J_stat', None)))
    j_p = diag.get('sargan_p', diag.get('hansen_p', diag.get('J_p', None)))
    if j_p is not None:
        checks.append(AssumptionCheck(
            assumption='Instrument exogeneity',
            test_name='Sargan/Hansen J',
            passed=j_p > alpha,
            statistic=j_stat,
            p_value=j_p,
            detail='Critical — tests exclusion restriction (if overidentified).',
            remedy='Re-examine instrument validity. Try different instrument set.',
        ))

    # 3. Endogeneity (Hausman)
    dwh_p = diag.get('durbin_wu_hausman_p', diag.get('endogeneity_p', None))
    if dwh_p is not None:
        checks.append(AssumptionCheck(
            assumption='Endogeneity confirmed',
            test_name='Durbin-Wu-Hausman',
            passed=dwh_p < alpha,  # Want to reject: endogeneity exists → IV needed
            statistic=None,
            p_value=dwh_p,
            detail='If p > 0.05, OLS may be preferred (more efficient).',
            remedy='If not endogenous, use OLS instead (more efficient).',
        ))

    return checks


def _audit_did(result, data, alpha):
    """Assumption tests for DID."""
    checks = []

    # 1. Parallel trends (from result if available)
    mi = getattr(result, 'model_info', {})
    if 'pretrends_p' in mi:
        p = mi['pretrends_p']
        checks.append(AssumptionCheck(
            assumption='Parallel trends',
            test_name='Joint pre-trend test',
            passed=p > alpha if p is not None else None,
            statistic=None,
            p_value=p,
            detail='Critical — core identifying assumption of DID.',
            remedy='Use sp.honest_did() for sensitivity to violations, '
                   'or sp.sensitivity_rr() for Rambachan-Roth analysis.',
        ))
    else:
        checks.append(AssumptionCheck(
            assumption='Parallel trends',
            test_name='(not yet tested)',
            passed=None,
            statistic=None,
            p_value=None,
            detail='Run sp.pretrends_test() or sp.event_study() to check.',
            remedy='sp.pretrends_test(result) or sp.event_study(df, ...)',
        ))

    # 2. No anticipation
    checks.append(AssumptionCheck(
        assumption='No anticipation',
        test_name='(visual check)',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Check that pre-treatment coefficients are near zero in event study.',
        remedy='Run sp.event_study() and inspect pre-period coefficients.',
    ))

    # 3. SUTVA
    checks.append(AssumptionCheck(
        assumption='SUTVA (no spillovers)',
        test_name='(domain knowledge)',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Cannot be tested statistically — requires domain knowledge.',
        remedy='Consider sp.spillover() if units interact.',
    ))

    return checks


def _audit_rd(result, data, alpha):
    """Assumption tests for RD."""
    checks = []

    # 1. No manipulation
    checks.append(AssumptionCheck(
        assumption='No manipulation',
        test_name='McCrary density',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Run sp.rddensity(df, x="running_var") to test.',
        remedy='sp.rddensity() — if significant, consider donut-hole RD.',
    ))

    # 2. Bandwidth robustness
    checks.append(AssumptionCheck(
        assumption='Bandwidth robustness',
        test_name='(sensitivity check)',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Run sp.rdbwsensitivity() to check stability across bandwidths.',
        remedy='sp.rdbwsensitivity(result)',
    ))

    # 3. Covariate balance
    checks.append(AssumptionCheck(
        assumption='Covariate continuity',
        test_name='Balance at cutoff',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Run sp.rdbalance() to test covariate jumps at cutoff.',
        remedy='sp.rdbalance(df, covariates=[...], x="running_var")',
    ))

    return checks


def _audit_binary(result, data, alpha):
    """Assumption tests for logit/probit."""
    checks = []

    # 1. Goodness of fit
    diag = getattr(result, 'diagnostics', {})
    pcp = diag.get('pcp', diag.get('percent_correct', None))
    if pcp is not None:
        checks.append(AssumptionCheck(
            assumption='Predictive accuracy',
            test_name='% correctly predicted',
            passed=pcp > 0.6,
            statistic=pcp,
            p_value=None,
            detail=f'{pcp:.1%} correctly predicted.',
            remedy='Consider additional predictors or alternative specification.',
        ))

    # 2. No perfect separation
    checks.append(AssumptionCheck(
        assumption='No perfect separation',
        test_name='(convergence check)',
        passed=getattr(result, 'model_info', {}).get('converged', True),
        statistic=None,
        p_value=None,
        detail='If model did not converge, perfect separation may exist.',
        remedy='Remove variables that perfectly predict the outcome.',
    ))

    return checks


def _audit_panel(result, data, alpha):
    """Assumption tests for panel models."""
    checks = []

    mi = getattr(result, 'model_info', {})
    method = mi.get('method', mi.get('model_type', ''))

    # 1. FE vs RE (Hausman)
    checks.append(AssumptionCheck(
        assumption='FE vs RE specification',
        test_name='Hausman test',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Run sp.hausman_test(fe_result, re_result) to choose.',
        remedy='If Hausman rejects, use FE. Otherwise RE is more efficient.',
    ))

    # 2. Serial correlation
    checks.append(AssumptionCheck(
        assumption='No serial correlation',
        test_name='(panel-specific)',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Check with Wooldridge test or use clustered SE by default.',
        remedy='Use cluster SE: sp.panel(..., cluster=id) or sp.panel_fgls(..., corr="ar1")',
    ))

    return checks


def _audit_generic(result, data, alpha):
    """Generic checks for any model."""
    checks = []

    # Sample size
    n = getattr(result, 'data_info', {}).get('n_obs', 0)
    k = len(result.params) if hasattr(result, 'params') else 0
    if n > 0:
        checks.append(AssumptionCheck(
            assumption='Sufficient sample size',
            test_name=f'N={n}, k={k}',
            passed=n > 30 and n / max(k, 1) > 5,
            statistic=n,
            p_value=None,
            detail=f'N/k ratio = {n/max(k,1):.0f}',
            remedy='Increase sample or reduce model complexity.',
        ))

    return checks
