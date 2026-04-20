"""Tests for sp.check_identification — design-level identification diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.smart.identification import (
    check_identification,
    IdentificationReport,
    DiagnosticFinding,
)


# ---------------------------------------------------------------------------
# Clean DGP: should produce minimal findings
# ---------------------------------------------------------------------------

def test_clean_rct_has_no_blockers():
    """A clean RCT with balanced treatment should have no blockers."""
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        'd': rng.binomial(1, 0.5, n),
        'X1': rng.normal(size=n),
        'X2': rng.normal(size=n),
    })
    df['y'] = 1.0 + 0.5 * df['X1'] + 1.0 * df['d'] + rng.normal(size=n)
    rep = check_identification(df, y='y', treatment='d',
                               covariates=['X1', 'X2'], design='observational')
    assert rep.verdict in ('OK', 'WARNINGS'), (
        f"Expected OK/WARNINGS, got {rep.verdict}: "
        f"{[f.message for f in rep.findings if f.severity == 'blocker']}"
    )
    # No blockers expected on a well-balanced RCT
    blockers = [f for f in rep.findings if f.severity == 'blocker']
    assert len(blockers) == 0


# ---------------------------------------------------------------------------
# Overlap blocker: near-deterministic treatment
# ---------------------------------------------------------------------------

def test_near_deterministic_treatment_flagged_as_blocker():
    """If treatment is nearly determined by covariates, must flag as blocker."""
    rng = np.random.default_rng(2)
    n = 500
    X = rng.normal(size=n)
    # Near-perfect separation: d = 1 iff X > 0 (no overlap)
    d = (X > 0).astype(int)
    y = 1.0 + X + d + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'X': X})
    rep = check_identification(df, y='y', treatment='d',
                               covariates=['X'], design='observational')
    # Either blocker (perfect separation) or warning (extreme PS)
    flagged = [f for f in rep.findings
               if f.category == 'overlap'
               and f.severity in ('blocker', 'warning')]
    assert len(flagged) > 0, (
        f"Near-deterministic treatment must flag overlap issue. "
        f"Findings: {[(f.category, f.severity, f.message) for f in rep.findings]}"
    )


# ---------------------------------------------------------------------------
# Bad control detection
# ---------------------------------------------------------------------------

def test_mediator_as_control_flagged():
    """A covariate highly correlated with treatment (candidate mediator)
    must be flagged."""
    rng = np.random.default_rng(3)
    n = 500
    d = rng.binomial(1, 0.5, n)
    # mediator M = 0.9 * d + noise
    m = 0.9 * d + rng.normal(scale=0.2, size=n)
    y = 1.0 + 2.0 * d + 0.5 * m + rng.normal(size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'm': m})
    rep = check_identification(df, y='y', treatment='d',
                               covariates=['m'], design='observational')
    flagged = [f for f in rep.findings
               if f.category == 'bad_controls']
    assert len(flagged) > 0, "Should flag highly-correlated covariate"
    # Should mention 'm'
    assert any('m' in f.message for f in flagged)


# ---------------------------------------------------------------------------
# Treatment variation: too few treated
# ---------------------------------------------------------------------------

def test_tiny_treatment_fraction_flagged():
    rng = np.random.default_rng(4)
    n = 1000
    d = rng.binomial(1, 0.02, n)  # only 2% treated
    y = 1.0 + d + rng.normal(size=n)
    df = pd.DataFrame({'y': y, 'd': d})
    rep = check_identification(df, y='y', treatment='d',
                               design='observational')
    variation = rep.by_category('variation')
    assert len(variation) > 0
    assert any('treated' in f.message.lower() for f in variation)


# ---------------------------------------------------------------------------
# DID cohort sizes
# ---------------------------------------------------------------------------

def test_did_cohort_sizes_flagged():
    """Small DID cohorts should be flagged."""
    rng = np.random.default_rng(5)
    rows = []
    # 3 units in cohort 5, rest never-treated
    for i in range(100):
        g = 5 if i < 3 else 0
        for t in range(1, 9):
            rows.append({'i': i, 't': t, 'g': g, 'y': rng.normal()})
    df = pd.DataFrame(rows)
    rep = check_identification(df, y='y', cohort='g', id='i', time='t',
                               design='did')
    # Must flag small cohort
    variation = rep.by_category('variation')
    assert any('Small treatment cohort' in f.message or
               'small cohort' in f.message.lower()
               for f in variation), (
        f"Small cohort not flagged: {[f.message for f in rep.findings]}"
    )


# ---------------------------------------------------------------------------
# RD density near cutoff
# ---------------------------------------------------------------------------

def test_rd_sparse_near_cutoff_flagged():
    """Sparse observations near the cutoff should be flagged."""
    rng = np.random.default_rng(6)
    # Running variable clustered away from cutoff
    x = np.concatenate([rng.uniform(-1, -0.5, 100), rng.uniform(0.5, 1, 100)])
    y = rng.normal(size=200)
    df = pd.DataFrame({'x': x, 'y': y})
    rep = check_identification(df, y='y', running_var='x', cutoff=0.0,
                               design='rd')
    variation = rep.by_category('variation')
    assert any('cutoff' in f.message.lower() for f in variation)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_panel_without_cluster_raises_info():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(50):
        for t in range(1, 5):
            rows.append({'i': i, 't': t, 'y': rng.normal()})
    df = pd.DataFrame(rows)
    rep = check_identification(df, y='y', id='i', time='t', design='panel')
    cl = rep.by_category('clustering')
    assert len(cl) > 0
    assert any("cluster" in f.message.lower() for f in cl)


def test_few_clusters_flagged():
    rng = np.random.default_rng(8)
    rows = []
    # Only 10 clusters
    for i in range(10):
        for t in range(1, 10):
            rows.append({'i': i, 't': t, 'y': rng.normal()})
    df = pd.DataFrame(rows)
    rep = check_identification(df, y='y', id='i', time='t', cluster='i',
                               design='panel')
    cl = rep.by_category('clustering')
    assert any(f.severity == 'warning' and 'clusters' in f.message.lower()
               for f in cl)


# ---------------------------------------------------------------------------
# Report API
# ---------------------------------------------------------------------------

def test_report_summary_is_string():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({'d': rng.binomial(1, 0.5, 100),
                       'y': rng.normal(size=100)})
    rep = check_identification(df, y='y', treatment='d',
                               design='observational')
    s = rep.summary()
    assert isinstance(s, str)
    assert 'Identification Diagnostics' in s


def test_report_repr():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({'d': rng.binomial(1, 0.5, 100),
                       'y': rng.normal(size=100)})
    rep = check_identification(df, y='y', treatment='d',
                               design='observational')
    r = repr(rep)
    assert 'IdentificationReport' in r
    assert rep.verdict in r


def test_top_level_api_exposed():
    """sp.check_identification should be importable from top-level."""
    assert hasattr(sp, 'check_identification')
    assert sp.check_identification is check_identification


# ---------------------------------------------------------------------------
# strict mode: IdentificationError
# ---------------------------------------------------------------------------

def test_strict_mode_raises_on_blocker():
    """strict=True must raise IdentificationError when a blocker is present."""
    from statspai.smart.identification import IdentificationError
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(size=n)
    d = (X > 0).astype(int)  # near-perfect separation => blocker
    y = 1.0 + X + d + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'X': X})
    with pytest.raises(IdentificationError) as exc_info:
        check_identification(df, y='y', treatment='d',
                             covariates=['X'], design='observational',
                             strict=True)
    # Exception should carry the report and blocker messages
    err = exc_info.value
    assert err.report.verdict == 'BLOCKERS'
    assert 'blocker' in str(err).lower() or 'identifi' in str(err).lower()


def test_strict_mode_allows_warnings():
    """strict=True must NOT raise when only warnings/info are present."""
    rng = np.random.default_rng(12)
    n = 500
    df = pd.DataFrame({
        'd': rng.binomial(1, 0.5, n),
        'X1': rng.normal(size=n),
    })
    df['y'] = 1.0 + 0.5 * df['X1'] + 1.0 * df['d'] + rng.normal(size=n)
    rep = check_identification(df, y='y', treatment='d', covariates=['X1'],
                               design='observational', strict=True)
    # Should reach here without raising
    assert rep.verdict in ('OK', 'WARNINGS')


def test_strict_mode_default_is_non_strict():
    """Default strict=False must return a report even with blockers."""
    rng = np.random.default_rng(13)
    n = 400
    X = rng.normal(size=n)
    d = (X > 0).astype(int)
    y = 1.0 + X + d + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'X': X})
    rep = check_identification(df, y='y', treatment='d',
                               covariates=['X'], design='observational')
    # Returns, doesn't raise
    assert rep.verdict == 'BLOCKERS'


def test_identification_error_exposed_at_sp_top_level():
    assert hasattr(sp, 'IdentificationError')
    from statspai.smart.identification import IdentificationError as _IE
    assert sp.IdentificationError is _IE


# ---------------------------------------------------------------------------
# IV strength check
# ---------------------------------------------------------------------------

def test_iv_weak_instrument_flagged():
    """Near-zero correlation between instrument and treatment => weak IV warning/blocker."""
    rng = np.random.default_rng(14)
    n = 600
    Z = rng.normal(size=n)
    # D essentially unrelated to Z
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 + D + rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z})
    rep = check_identification(df, y='y', treatment='d', instrument='z',
                               design='iv')
    # Some finding about IV strength should exist
    msgs = ' '.join(f.message.lower() for f in rep.findings)
    assert ('weak' in msgs or 'first-stage' in msgs or 'instrument' in msgs), (
        f"Expected IV-strength finding; got: {[f.message for f in rep.findings]}"
    )


def test_iv_strong_instrument_no_weak_finding():
    """A strong first stage (F >> 10) should not emit weak-IV warnings."""
    rng = np.random.default_rng(15)
    n = 800
    Z = rng.normal(size=n)
    # D strongly driven by Z
    D = (0.9 * Z + rng.normal(scale=0.3, size=n) > 0).astype(float)
    Y = 1.0 + D + rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z})
    rep = check_identification(df, y='y', treatment='d', instrument='z',
                               design='iv')
    weak_flagged = [f for f in rep.findings
                    if 'weak' in f.message.lower()
                    and f.severity in ('blocker', 'warning')]
    assert len(weak_flagged) == 0, (
        f"Unexpected weak-IV finding on strong-instrument DGP: {weak_flagged}"
    )


# ---------------------------------------------------------------------------
# DAG integration (trade-off #2 — principled collider / mediator detection)
# ---------------------------------------------------------------------------

class TestDAGIntegration:
    """When a DAG is supplied, check_identification must catch structural
    bad controls that the correlation heuristic misses."""

    def _mbias_dag(self):
        """M-bias DAG: X is a collider on U1 -> X <- U2, both U1 and U2
        are ancestors of D and Y respectively.  Conditioning on X opens
        a non-causal backdoor U1 -> X <- U2 -> Y."""
        g = sp.DAG()
        g.add_edge('U1', 'D')
        g.add_edge('U1', 'X')
        g.add_edge('U2', 'X')
        g.add_edge('U2', 'Y')
        g.add_edge('D', 'Y')
        return g

    def test_mbias_collider_flagged_as_blocker(self):
        """M-bias scenario: including the collider X must be flagged BLOCKER."""
        rng = np.random.default_rng(1)
        n = 500
        U1 = rng.normal(size=n)
        U2 = rng.normal(size=n)
        D = (U1 + rng.normal(size=n) > 0).astype(int)
        X = U1 + U2 + rng.normal(size=n)
        Y = D + U2 + rng.normal(size=n)
        df = pd.DataFrame({'D': D, 'Y': Y, 'X': X, 'U1': U1, 'U2': U2})
        g = self._mbias_dag()

        rep = sp.check_identification(
            df, y='Y', treatment='D', covariates=['X'],
            design='observational', dag=g,
        )
        blockers = [f for f in rep.findings
                    if f.category == 'bad_controls' and f.severity == 'blocker']
        assert len(blockers) > 0, (
            f"M-bias collider not flagged as blocker: "
            f"{[(f.severity, f.message) for f in rep.findings]}"
        )
        assert any('collider' in b.message.lower() for b in blockers)

    def test_mediator_flagged_as_blocker(self):
        """Mediator detection: M = f(D), Y = g(M). Conditioning on M blocks
        the indirect effect."""
        g = sp.DAG()
        g.add_edge('D', 'M')
        g.add_edge('M', 'Y')
        g.add_edge('D', 'Y')

        rng = np.random.default_rng(2)
        n = 300
        D = rng.binomial(1, 0.5, n)
        M = D + rng.normal(scale=0.3, size=n)
        Y = D + 0.5 * M + rng.normal(size=n)
        df = pd.DataFrame({'D': D, 'M': M, 'Y': Y})

        rep = sp.check_identification(
            df, y='Y', treatment='D', covariates=['M'],
            design='observational', dag=g,
        )
        blockers = [f for f in rep.findings
                    if f.category == 'bad_controls' and f.severity == 'blocker']
        assert len(blockers) > 0, (
            f"Mediator not flagged: {[f.message for f in rep.findings]}"
        )
        assert any('mediator' in b.message.lower() or
                   'descendant' in b.message.lower()
                   for b in blockers)

    def test_good_adjustment_set_no_warning(self):
        """Covariates that form a valid adjustment set: no DAG warning."""
        # U is a confounder: U -> D, U -> Y.
        # Adjusting for U is the correct backdoor adjustment.
        g = sp.DAG()
        g.add_edge('U', 'D')
        g.add_edge('U', 'Y')
        g.add_edge('D', 'Y')
        rng = np.random.default_rng(3)
        n = 300
        U = rng.normal(size=n)
        D = (U + rng.normal(size=n) > 0).astype(int)
        Y = U + D + rng.normal(size=n)
        df = pd.DataFrame({'U': U, 'D': D, 'Y': Y})

        rep = sp.check_identification(
            df, y='Y', treatment='D', covariates=['U'],
            design='observational', dag=g,
        )
        # Adjustment set {U} is valid; no blocker from DAG
        dag_blockers = [f for f in rep.findings
                        if f.category == 'bad_controls'
                        and f.severity == 'blocker']
        # The correlation heuristic may still flag U (|corr| high),
        # but the DAG check should NOT produce a blocker.
        assert not any("DAG-flagged" in b.message for b in dag_blockers)


# ---------------------------------------------------------------------------
# tidy(conf_level) uses t-distribution when df_resid is available
# ---------------------------------------------------------------------------

class TestTidyTDistribution:
    """Trade-off #3: small-sample CausalResult.tidy(conf_level) should use
    the t-distribution critical value when df_resid is recorded."""

    def test_t_dist_wider_than_normal_small_sample(self):
        """With df_resid = 10 and conf_level=0.95, the t-critical is
        2.228 vs 1.96 for normal; CIs must be wider accordingly."""
        from statspai.core.results import CausalResult

        # Build a CausalResult with df_resid in model_info
        r_t = CausalResult(
            method='Test DID',
            estimand='ATT',
            estimate=1.0,
            se=0.1,
            pvalue=0.001,
            ci=(0.7,  1.3),
            alpha=0.1,         # so 1 - alpha = 0.90, different from tidy's 0.95
            n_obs=20,
            model_info={'df_resid': 10},
        )
        # Without df_resid
        r_n = CausalResult(
            method='Test async',
            estimand='ATT',
            estimate=1.0,
            se=0.1,
            pvalue=0.001,
            ci=(0.7, 1.3),
            alpha=0.1,
            n_obs=1_000_000,
            model_info={},
        )
        t_t = r_t.tidy(conf_level=0.95).query("type == 'main'").iloc[0]
        t_n = r_n.tidy(conf_level=0.95).query("type == 'main'").iloc[0]

        w_t = t_t['conf_high'] - t_t['conf_low']
        w_n = t_n['conf_high'] - t_n['conf_low']

        # t-based width must be STRICTLY wider (2.228 vs 1.96 at df=10)
        assert w_t > w_n, (
            f"Expected t-CI ({w_t:.4f}) > normal-CI ({w_n:.4f})"
        )
        # Check specific numbers: t.ppf(0.975, df=10) = 2.228139
        # Expected half-width: 2.228139 * 0.1 = 0.2228; full = 0.4456
        expected_w = 2.0 * 2.2281 * 0.1
        assert abs(w_t - expected_w) < 0.01, (
            f"t-width {w_t:.4f} vs expected {expected_w:.4f}"
        )

    def test_large_df_converges_to_normal(self):
        """With df_resid = 1e6, t and normal must coincide."""
        from statspai.core.results import CausalResult
        r = CausalResult(
            method='Test', estimand='ATT', estimate=0, se=1.0,
            pvalue=1.0, ci=(-1.96, 1.96), alpha=0.1,
            n_obs=1_000_001, model_info={'df_resid': 1_000_000},
        )
        t = r.tidy(conf_level=0.95).query("type == 'main'").iloc[0]
        w = t['conf_high'] - t['conf_low']
        # Normal 95% CI: 1.96 * 2 = 3.92
        assert abs(w - 3.92) < 0.01, (
            f"Large-df CI width {w:.4f} should approach 3.92"
        )
