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

    # Detect method type. CausalResult stores the human-readable method
    # on ``.method`` (e.g. "Proximal Causal Inference (linear 2SLS)");
    # legacy EconometricResults put it in ``model_info['model_type']``.
    # Read both so Sprint-B modules actually match on their keyword
    # rather than falling silently into the OLS branch (which triggers
    # on the *empty string* under the old detection — the "''" kw list
    # member always matched).
    model_info = getattr(result, 'model_info', {}) or {}
    method_str = ' '.join([
        str(model_info.get('model_type', '') or ''),
        str(getattr(result, 'method', '') or ''),
        str(model_info.get('estimator', '') or ''),
    ]).lower()

    # Try to get data from result
    if data is None:
        data = getattr(result, '_data', None) or getattr(result, 'data', None)

    # ----- Sprint-B (0.9.6) methods: explicit match BEFORE the generic
    # OLS branch so they don't fall through into "looks like
    # regression → run RESET" by accident.
    sprint_b_matched = False
    if any(kw in method_str for kw in ('proximal',)):
        checks.extend(_audit_proximal(result, data, alpha))
        sprint_b_matched = True
    if any(kw in method_str
           for kw in ('marginal structural', 'iptw')):
        checks.extend(_audit_msm(result, data, alpha))
        sprint_b_matched = True
    # PrincipalStratResult stores 'monotonicity' or 'principal_score' on
    # .method (not a CausalResult label), so match both on the class name
    # and on the Zhang-Rubin / Ding-Lu keywords.
    if (type(result).__name__ == 'PrincipalStratResult'
            or any(kw in method_str for kw in (
                'principal stratif', 'zhang-rubin',
                'principal score', 'monotonicity'))):
        checks.extend(_audit_principal_strat(result, data, alpha))
        sprint_b_matched = True
    if any(kw in method_str
           for kw in ('g-computation', 'g computation',
                      'g-formula', 'g formula',
                      'standardization', 'standardisation')):
        checks.extend(_audit_g_computation(result, data, alpha))
        sprint_b_matched = True
    if any(kw in method_str
           for kw in ('front-door', 'front door', 'front_door')):
        checks.extend(_audit_front_door(result, data, alpha))
        sprint_b_matched = True
    if any(kw in method_str
           for kw in ('interventional mediation', 'iie')):
        checks.extend(_audit_mediation_interventional(result, data, alpha))
        sprint_b_matched = True
    # Natural mediation branch is guarded: if a future estimator labels
    # itself "Causal Mediation Analysis (interventional)" the
    # interventional branch above fires first and this `and not
    # sprint_b_matched` prevents double-firing the natural-effects
    # stub on the same result.
    if (not sprint_b_matched
            and 'causal mediation' in method_str):
        checks.extend(_audit_mediation_natural(result, data, alpha))
        sprint_b_matched = True

    # Legacy keyword detection runs ONLY when no Sprint-B branch matched.
    # The classical keywords are short substrings that leak into unrelated
    # method strings (e.g. "fe" and "re" appear inside "causal infeRence",
    # "iv" inside "interventional"). When the Sprint-B dispatcher has
    # already classified the method, skip all of these outright.
    if not sprint_b_matched:
        # Use word-boundary tokens to avoid the short-substring trap.
        def _has_token(*tokens):
            import re
            for tok in tokens:
                if re.search(r'\b' + re.escape(tok) + r'\b', method_str):
                    return True
            return False

        # ===== OLS / LINEAR REGRESSION =====
        if _has_token('ols', 'linear', 'regress', 'regression'):
            checks.extend(_audit_linear(result, data, alpha))

        # ===== IV / 2SLS =====
        if _has_token('iv', '2sls', 'liml', 'jive', 'gmm'):
            checks.extend(_audit_iv(result, data, alpha))

        # ===== DID =====
        if _has_token('did', 'difference') or 'diff-in-diff' in method_str:
            checks.extend(_audit_did(result, data, alpha))

        # ===== RD =====
        if (_has_token('rd', 'rdrobust')
                or 'regression discontinuity' in method_str):
            checks.extend(_audit_rd(result, data, alpha))

        # ===== LOGIT/PROBIT =====
        if _has_token('logit', 'probit', 'binary'):
            checks.extend(_audit_binary(result, data, alpha))

        # ===== PANEL =====
        if (_has_token('panel', 'fe', 're')
                or 'fixed effect' in method_str):
            checks.extend(_audit_panel(result, data, alpha))

    # If no method-specific checks, run generic
    if not checks:
        checks.extend(_audit_generic(result, data, alpha))

    # Method label surfaced on the report — use the richest available.
    method = (
        str(getattr(result, 'method', '') or '')
        or str(model_info.get('model_type', '') or '')
        or str(model_info.get('estimator', '') or '')
    )

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


# ====================================================================== #
#  Sprint-B (v0.9.6) method-aware audits                                 #
# ====================================================================== #
#
#  Each of these functions mirrors an estimator in the Sprint-B causal-
#  inference surface. They focus on the assumptions that are *specific*
#  to the method — we don't duplicate the generic sample-size / overlap
#  checks already in ``_audit_generic`` because ``assumption_audit``
#  appends those only when *no* method-specific branch matched.
#
#  Every check's ``remedy`` points to a concrete StatsPAI function or
#  a ROADMAP section so an agent consuming the report has an
#  actionable next step.


def _audit_proximal(result, data, alpha):
    """Proximal causal inference: bridge + proxy rank + weak-instrument F."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    bridge = info.get('bridge', 'linear')
    checks.append(AssumptionCheck(
        assumption='Outcome bridge functional form',
        test_name=f"bridge='{bridge}'",
        passed=bridge == 'linear',
        statistic=None,
        p_value=None,
        detail=('Linear bridge is the shipped option (Cui et al. 2024). '
                'Kernel/sieve bridges are on the roadmap.'),
        remedy=('See docs/ROADMAP.md §1 for non-linear bridge plans; '
                'inspect residuals for obvious non-linearity.'),
    ))

    k_z = info.get('n_proxy_z')
    k_w = info.get('n_proxy_w')
    if k_z is not None and k_w is not None:
        ok = int(k_z) >= int(k_w)
        checks.append(AssumptionCheck(
            assumption='Proxy order condition (k_z ≥ k_w)',
            test_name=f'{k_z} Z vs {k_w} W proxies',
            passed=ok,
            statistic=int(k_z) - int(k_w),
            p_value=None,
            detail='Need at least one excluded Z per endogenous W.',
            remedy='Add a treatment-side proxy or drop an outcome-side proxy.',
        ))

    fs_F = info.get('first_stage_F')
    if fs_F is not None:
        passed = float(fs_F) >= 10.0
        checks.append(AssumptionCheck(
            assumption='Proxy relevance (first-stage F)',
            test_name=f'F = {float(fs_F):.2f}',
            passed=passed,
            statistic=float(fs_F),
            p_value=None,
            detail='F ≥ 10 is the standard weak-IV rule of thumb.',
            remedy=('Find a stronger treatment-side proxy Z, or use '
                    'weak-IV-robust inference (sp.anderson_rubin_test).'),
        ))
    elif k_w is not None and int(k_w) > 1:
        checks.append(AssumptionCheck(
            assumption='Proxy relevance (multi-W)',
            test_name='Cragg-Donald / Kleibergen-Paap',
            passed=None,
            statistic=None,
            p_value=None,
            detail='Multi-W weak-IV statistic is not yet shipped.',
            remedy='See docs/ROADMAP.md §2 for planned implementation.',
        ))

    return checks


def _audit_msm(result, data, alpha):
    """Marginal structural model: positivity + sequential exchangeability."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    sw_mean = info.get('sw_mean')
    if sw_mean is not None:
        deviation = abs(float(sw_mean) - 1.0)
        checks.append(AssumptionCheck(
            assumption='Stabilized-weight mean centred at 1',
            test_name=f'sw_mean = {float(sw_mean):.3f}',
            passed=deviation < 0.2,
            statistic=float(sw_mean) - 1.0,
            p_value=None,
            detail=('Under correct nuisance specification, stabilized '
                    'IPTW has mean ≈ 1. Drift flags model misspec.'),
            remedy=('Refit the numerator / denominator nuisance with '
                    'more flexible covariates, or use trim_per_period=True.'),
        ))

    sw_max = info.get('sw_max')
    if sw_max is not None:
        checks.append(AssumptionCheck(
            assumption='Positivity (no extreme weights)',
            test_name=f'max sw = {float(sw_max):.2f}',
            passed=float(sw_max) < 50.0,
            statistic=float(sw_max),
            p_value=None,
            detail=('Near-positivity-violation produces extreme weights '
                    'which dominate the IPW fit.'),
            remedy=('Enable trim_per_period=True or raise trim quantile; '
                    'drop covariates that deterministically predict treatment.'),
        ))

    exposure = info.get('exposure_type')
    if exposure:
        checks.append(AssumptionCheck(
            assumption='Exposure summary is the target estimand',
            test_name=f"exposure='{exposure}'",
            passed=True,
            statistic=None,
            p_value=None,
            detail=('The reported coefficient is the marginal effect on '
                    f'the {exposure} exposure.'),
            remedy='If wrong target, rerun sp.msm with a different exposure=.',
        ))

    checks.append(AssumptionCheck(
        assumption='Sequential exchangeability (identification)',
        test_name='Design assumption (not testable from data)',
        passed=None,
        statistic=None,
        p_value=None,
        detail=('MSM requires no unmeasured time-varying confounders at '
                'each period (Robins 1998).'),
        remedy=('Document the DAG justifying inclusion of all '
                'time-varying confounders; consider sensitivity analysis.'),
    ))
    return checks


def _audit_principal_strat(result, data, alpha):
    """Principal stratification: monotonicity + principal ignorability."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    viol = info.get('mono_violation_frac')
    if viol is not None:
        checks.append(AssumptionCheck(
            assumption='Monotonicity S(1) ≥ S(0)',
            test_name=(f'fitted p11(x) < p10(x) fraction '
                       f'= {float(viol):.1%}'),
            passed=float(viol) <= 0.05,
            statistic=float(viol),
            p_value=None,
            detail=('Monotonicity is violated when the fitted cell '
                    'probabilities imply defiers. Small drift (≤5%) is '
                    'absorbed by clipping; larger is a red flag.'),
            remedy=('Revisit the stratum definition and the population: '
                    'monotonicity fails if some units systematically '
                    'respond in opposition to treatment.'),
        ))

    # Stratum proportions sum to 1
    strata = getattr(result, 'strata_proportions', None)
    if strata:
        total = sum(float(v) for v in strata.values())
        checks.append(AssumptionCheck(
            assumption='Stratum proportions sum to 1',
            test_name=f'sum = {total:.3f}',
            passed=abs(total - 1.0) < 0.05,
            statistic=total,
            p_value=None,
            detail='Simplex check: strata partition the population.',
            remedy='Re-examine stratum definition if the sum drifts far from 1.',
        ))

    # Principal ignorability is only needed for method='principal_score'
    method = info.get('estimator', '')
    if 'principal score' in method.lower():
        checks.append(AssumptionCheck(
            assumption='Principal ignorability (Y(d) ⊥ stratum | X, D=d)',
            test_name='Design assumption (not testable from data)',
            passed=None,
            statistic=None,
            p_value=None,
            detail=('Ding & Lu (2017) point identification hinges on X '
                    'absorbing all stratum-outcome dependence within '
                    'each treatment arm.'),
            remedy=('Pair the point estimate with a sensitivity analysis; '
                    'or fall back to method="monotonicity" bounds.'),
        ))
    return checks


def _audit_g_computation(result, data, alpha):
    """G-computation: outcome model correctness + positivity + non-DR warning."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    estimand = info.get('estimand', 'ATE')
    checks.append(AssumptionCheck(
        assumption='Outcome model correctly specified',
        test_name=f'ml_Q = {info.get("ml_Q", "OLS")}',
        passed=None,
        statistic=None,
        p_value=None,
        detail=('G-computation is consistent only if the outcome '
                f'regression Q(D, X) is correctly specified. '
                f'Not doubly robust — a misspecified model biases the '
                f'{estimand}.'),
        remedy=('Cross-check against sp.aipw or sp.tmle (doubly robust) '
                'for coverage under misspecification.'),
    ))

    n_boot = info.get('n_boot', 0)
    n_failed = info.get('n_boot_failed', 0)
    if n_boot:
        frac = n_failed / n_boot
        checks.append(AssumptionCheck(
            assumption='Bootstrap inference well-behaved',
            test_name=f'{n_failed}/{n_boot} replications failed',
            passed=frac < 0.1,
            statistic=frac,
            p_value=None,
            detail='High bootstrap failure rate distorts SE and CI.',
            remedy=('Increase sample or simplify the outcome model; '
                    'investigate first_bootstrap_error in model_info.'),
        ))
    return checks


def _audit_front_door(result, data, alpha):
    """Front-door: mediator exhaustiveness + unmeasured confounding blocked."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    m_type = info.get('mediator_type')
    if m_type:
        checks.append(AssumptionCheck(
            assumption='Mediator modelled with correct family',
            test_name=f"mediator_type='{m_type}'",
            passed=m_type in ('binary', 'continuous'),
            statistic=None,
            p_value=None,
            detail='Binary uses closed-form sums; continuous uses MC Gaussian.',
            remedy='If the mediator is categorical with >2 levels, not yet supported.',
        ))

    checks.append(AssumptionCheck(
        assumption='No direct D→Y path (full mediation)',
        test_name='Design assumption (verify with sp.dag)',
        passed=None,
        statistic=None,
        p_value=None,
        detail=('Pearl (1995) front-door identification requires that '
                'M fully transmits the effect of D on Y.'),
        remedy='Use sp.dag + .front_door_adjustment_sets() to verify.',
    ))

    checks.append(AssumptionCheck(
        assumption='No unmeasured M-Y confounder',
        test_name='Design assumption',
        passed=None,
        statistic=None,
        p_value=None,
        detail='Any unobserved path M←U→Y invalidates front-door.',
        remedy=('Probe with sp.evalue / sp.sensemakr-style sensitivity '
                'analysis (not yet wired to front-door — roadmap).'),
    ))
    return checks


def _audit_mediation_interventional(result, data, alpha):
    """Interventional (in)direct effects: decomposition identity + assumptions."""
    checks = []
    info = getattr(result, 'model_info', {}) or {}

    iie = info.get('iie')
    ide = info.get('ide')
    total = info.get('total_effect')
    if iie is not None and ide is not None and total is not None:
        drift = abs(float(iie) + float(ide) - float(total))
        checks.append(AssumptionCheck(
            assumption='Decomposition identity IIE + IDE = Total',
            test_name=f'residual = {drift:.2e}',
            passed=drift < 1e-6,
            statistic=drift,
            p_value=None,
            detail='Under OLS linearity the identity is algebraic.',
            remedy='Non-zero drift indicates a numerical or code bug.',
        ))
    else:
        # Explicit "can't verify" check rather than silent skip. Prevents
        # a future estimator that stores different keys (e.g. 'acme'
        # instead of 'iie') from producing an audit grade of 'A' with
        # zero identity checks performed.
        missing = [k for k in ('iie', 'ide', 'total_effect')
                   if info.get(k) is None]
        checks.append(AssumptionCheck(
            assumption='Decomposition identity IIE + IDE = Total',
            test_name='keys not found in model_info',
            passed=None,
            statistic=None,
            p_value=None,
            detail=f'Missing keys: {missing}. Identity check skipped.',
            remedy=('Ensure the estimator populates iie / ide / '
                    'total_effect in model_info for audit coverage.'),
        ))

    checks.append(AssumptionCheck(
        assumption='No unmeasured baseline D–Y confounder',
        test_name='Design assumption',
        passed=None,
        statistic=None,
        p_value=None,
        detail=('Interventional effects dodge cross-world independence '
                'but still need baseline confounder control.'),
        remedy='Verify with sp.dag; sensitivity analysis for omitted controls.',
    ))

    tv = info.get('tv_confounders')
    if tv:
        checks.append(AssumptionCheck(
            assumption='Treatment-induced confounders listed',
            test_name=f'{len(tv)} confounder(s): {list(tv)[:3]}',
            passed=True,
            statistic=len(tv),
            p_value=None,
            detail=('Interventional effects handle treatment-induced '
                    'mediator-outcome confounders explicitly.'),
            remedy='Confirm the list is exhaustive for the DGP.',
        ))
    return checks


def _audit_mediation_natural(result, data, alpha):
    """Classical natural (in)direct effects (Imai-Keele-Tingley)."""
    checks = []

    checks.append(AssumptionCheck(
        assumption='Cross-world independence (natural effects)',
        test_name='Design assumption — not testable from data',
        passed=None,
        statistic=None,
        p_value=None,
        detail=('Natural (in)direct effects are identified only if '
                'there is no treatment-induced mediator-outcome '
                'confounder. If that assumption is suspect, rerun '
                'with sp.mediate_interventional.'),
        remedy=('If treatment affects a mediator-outcome confounder, '
                'switch to sp.mediate_interventional (interventional '
                'effects).'),
    ))

    info = getattr(result, 'model_info', {}) or {}
    n_boot = info.get('n_boot', 0)
    if n_boot:
        # Track failure rate the same way _audit_g_computation does —
        # previously this check was vacuously ``passed=True`` which
        # masked broken bootstraps.
        n_failed = info.get('n_boot_failed', 0) or 0
        frac = (n_failed / n_boot) if n_boot else 0.0
        checks.append(AssumptionCheck(
            assumption='Bootstrap inference well-behaved',
            test_name=f'{n_failed}/{n_boot} replications failed',
            passed=(frac < 0.1),
            statistic=frac,
            p_value=None,
            detail=('Percentile CIs derived from the bootstrap; failure '
                    'rate above 10% signals numerical instability.'),
            remedy=('If skewed, consider pvalue_method="wald"; '
                    'if failure_rate is high, increase sample or '
                    'simplify the mediator/outcome model.'),
        ))
    return checks
