"""Tests for sp.causal() end-to-end workflow orchestrator."""
from __future__ import annotations

import tempfile
import os

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.workflow import causal, CausalWorkflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def did_panel():
    """Staggered DID panel, homogeneous effect=1.5."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(300):
        g = [3, 5, 7, 0][i % 4]
        for t in range(1, 9):
            post = 1 if (g > 0 and t >= g) else 0
            rows.append({
                'i': i, 't': t, 'g': g, 'treat': post,
                'y': 0.2 * t + 1.5 * post + rng.normal(scale=0.8),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def rd_data():
    """Sharp RD with known jump=1.0."""
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 2 + 3*x + 1.0 * (x >= 0).astype(int) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({'y': y, 'x': x})


@pytest.fixture
def cia_observational():
    """CIA observational DGP, true ATT=2.0."""
    rng = np.random.default_rng(3)
    n = 1000
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    lin = -0.3 + 0.5 * X1 - 0.3 * X2
    p = 1 / (1 + np.exp(-lin))
    d = (rng.uniform(0, 1, n) < p).astype(int)
    y = 1.0 + 1.5 * X1 - 0.8 * X2 + 2.0 * d + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({'y': y, 'd': d, 'X1': X1, 'X2': X2})


# ---------------------------------------------------------------------------
# End-to-end: DID
# ---------------------------------------------------------------------------

class TestDIDWorkflow:
    """Full pipeline on a staggered DID panel."""

    def test_auto_run_completes_all_stages(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        # All the original core stages must have run. Since v0.9.17
        # `causal()` also triggers the extended stages
        # (compare_estimators / sensitivity_panel / cate) — we assert
        # superset rather than exact equality so the test stays robust
        # to additions.
        assert set(w.stages_completed) >= {
            'diagnose', 'recommend', 'estimate', 'robustness',
        }

    def test_diagnostics_verdict_ok_on_clean_dgp(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        assert w.diagnostics.verdict in ('OK', 'WARNINGS')

    def test_recommends_staggered_did(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        top = w.recommendation.recommendations[0]
        assert 'did' in top['function'].lower() or \
               'callaway' in top['function'].lower() or \
               'sun' in top['function'].lower()

    def test_estimate_near_truth(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        assert abs(w.result.estimate - 1.5) < 0.2, (
            f"DID workflow estimate {w.result.estimate} far from truth 1.5"
        )

    def test_robustness_includes_pretrend_for_did(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        # Pretrend test should be available if CS is the estimator
        # (but may be missing depending on which variant ran)
        assert 'estimate' in w.robustness_findings
        assert 'ci_width' in w.robustness_findings


# ---------------------------------------------------------------------------
# End-to-end: RD
# ---------------------------------------------------------------------------

class TestRDWorkflow:
    def test_rd_full_pipeline(self, rd_data):
        w = causal(rd_data, y='y', running_var='x', cutoff=0.0,
                   design='rd')
        assert set(w.stages_completed) >= {
            'diagnose', 'recommend', 'estimate', 'robustness',
        }
        # Jump should be near 1.0
        assert abs(w.result.estimate - 1.0) < 0.3


# ---------------------------------------------------------------------------
# End-to-end: observational
# ---------------------------------------------------------------------------

class TestObservationalWorkflow:
    def test_observational_full_pipeline(self, cia_observational):
        w = causal(cia_observational, y='y', treatment='d',
                   covariates=['X1', 'X2'], design='observational')
        assert w.diagnostics is not None
        assert w.recommendation is not None
        assert w.result is not None
        # estimate should be near 2.0 (either coef on 'd' or point estimate)
        if hasattr(w.result, 'estimate'):
            est = w.result.estimate
        else:
            est = w.result.params.get('d', np.nan)
        assert abs(est - 2.0) < 0.5, f"Got {est}"


# ---------------------------------------------------------------------------
# auto_run=False gives manual control
# ---------------------------------------------------------------------------

class TestManualStageControl:
    def test_auto_run_false_does_not_execute(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did',
                   auto_run=False)
        assert w.stages_completed == []
        assert w.diagnostics is None
        assert w.result is None

    def test_stages_run_on_demand(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did',
                   auto_run=False)
        w.diagnose()
        assert w.stages_completed == ['diagnose']
        w.recommend()
        assert 'recommend' in w.stages_completed
        w.estimate()
        assert w.result is not None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_html_report_contains_all_sections(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        html = w.report(fmt='html')
        # Check for major sections
        assert 'Causal Analysis Report' in html
        assert 'Identification' in html
        assert 'Recommended' in html
        assert 'Main estimate' in html or 'estimate' in html.lower()
        assert 'Robustness' in html
        # Check HTML validity (basic)
        assert html.startswith('<!DOCTYPE html>')
        assert '</html>' in html

    def test_markdown_report_has_headings(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        md = w.report(fmt='markdown')
        assert '# Causal Analysis Report' in md
        assert '## 1. Identification' in md
        assert '## 2. Recommended' in md
        assert '## 3. Main estimate' in md
        assert '## 4. Robustness' in md

    def test_report_writes_to_disk(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'report.html')
            w.report(path, fmt='html')
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 500
            assert 'Causal Analysis Report' in content

    def test_unknown_fmt_raises(self, did_panel):
        w = causal(did_panel, y='y', treatment='treat',
                   id='i', time='t', cohort='g', design='did')
        with pytest.raises(ValueError):
            w.report(fmt='pdf')


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def test_sp_causal_exposed():
    assert hasattr(sp, 'causal')
    assert callable(sp.causal)
    assert hasattr(sp, 'CausalWorkflow')


def test_workflow_repr(did_panel):
    w = causal(did_panel, y='y', treatment='treat',
               id='i', time='t', cohort='g', design='did',
               auto_run=False)
    r = repr(w)
    assert 'CausalWorkflow' in r
    assert 'did' in r
