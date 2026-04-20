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
