"""Tests for sp.paper_tables() — AER/QJE/Econometrica multi-panel bundle."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output.paper_tables import paper_tables, PaperTables, TEMPLATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def results_pair():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        'd': rng.binomial(1, 0.5, n),
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
    })
    df['y'] = 1 + 2 * df.d + 0.5 * df.x1 + 0.3 * df.x2 + rng.normal(size=n)
    df['y2'] = 0.5 + df.x1 + rng.normal(size=n)
    return {
        'r1': sp.regress('y ~ d', data=df, robust='hc1'),
        'r2': sp.regress('y ~ d + x1', data=df, robust='hc1'),
        'r3': sp.regress('y ~ d + x1 + x2', data=df, robust='hc1'),
        'r_placebo': sp.regress('y2 ~ d', data=df, robust='hc1'),
    }


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_main_only(self, results_pair):
        pt = paper_tables(main=[results_pair['r1'], results_pair['r2']],
                           template='aer')
        assert pt.main is not None
        assert pt.placebo is None
        assert pt.template == 'aer'

    def test_all_four_panels(self, results_pair):
        pt = paper_tables(
            main=[results_pair['r1'], results_pair['r3']],
            heterogeneity=[results_pair['r2']],
            robustness=[results_pair['r3']],
            placebo=[results_pair['r_placebo']],
            template='aer',
        )
        assert set(pt.panels().keys()) == {'main', 'heterogeneity',
                                            'robustness', 'placebo'}

    def test_unknown_template_raises(self, results_pair):
        with pytest.raises(ValueError) as exc:
            paper_tables(main=[results_pair['r1']], template='nature')
        assert 'template' in str(exc.value).lower()

    def test_all_templates_registered(self):
        assert 'aer' in TEMPLATES
        assert 'qje' in TEMPLATES
        assert 'econometrica' in TEMPLATES
        assert 'restat' in TEMPLATES


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_aer_has_three_star_levels(self, results_pair):
        pt = paper_tables(main=[results_pair['r1']], template='aer')
        assert pt.template == 'aer'
        # Render and check for at least three stars in the LaTeX
        tex = pt.to_latex()
        assert '***' in tex or '***' in pt.to_text()

    def test_econometrica_uses_two_star_levels(self, results_pair):
        pt = paper_tables(main=[results_pair['r1']], template='econometrica')
        assert TEMPLATES['econometrica']['star_levels'] == (0.01, 0.05)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_latex_includes_all_panels(self, results_pair):
        pt = paper_tables(
            main=[results_pair['r1']],
            placebo=[results_pair['r_placebo']],
            template='aer',
        )
        tex = pt.to_latex()
        assert 'MAIN' in tex
        assert 'PLACEBO' in tex
        assert r'\begin{table}' in tex
        assert r'\caption{' in tex

    def test_to_markdown_has_table_headings(self, results_pair):
        pt = paper_tables(
            main=[results_pair['r1'], results_pair['r3']],
            robustness=[results_pair['r2']],
            template='aer',
        )
        md = pt.to_markdown()
        assert '## Table: Main' in md
        assert '## Table: Robustness' in md
        # Markdown table rows
        assert '|' in md

    def test_to_text_has_panel_separators(self, results_pair):
        pt = paper_tables(main=[results_pair['r1']], template='aer')
        txt = pt.to_text()
        assert '=== MAIN ===' in txt

    def test_writes_latex_to_disk(self, results_pair):
        pt = paper_tables(main=[results_pair['r1']], template='aer')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tables.tex')
            pt.to_latex(path)
            assert os.path.exists(path)
            with open(path) as f:
                assert r'\begin{table}' in f.read()

    def test_writes_markdown_via_filename(self, results_pair):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tables.md')
            paper_tables(main=[results_pair['r1']],
                         template='aer', filename=path)
            assert os.path.exists(path)
            with open(path) as f:
                assert '## Table:' in f.read()


# ---------------------------------------------------------------------------
# Coefficient labels propagate
# ---------------------------------------------------------------------------

class TestCoefficientLabels:
    def test_coef_labels_applied(self, results_pair):
        pt = paper_tables(
            main=[results_pair['r3']],
            template='aer',
            coef_labels={'d': 'Treatment', 'x1': 'Age'},
        )
        txt = pt.to_text()
        assert 'Treatment' in txt
        assert 'Age' in txt


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def test_sp_paper_tables_exposed():
    assert hasattr(sp, 'paper_tables')
    assert hasattr(sp, 'PaperTables')
    assert hasattr(sp, 'PAPER_TABLE_TEMPLATES')


def test_paper_tables_repr(results_pair):
    pt = paper_tables(main=[results_pair['r1']],
                       placebo=[results_pair['r_placebo']],
                       template='qje')
    r = repr(pt)
    assert 'PaperTables' in r
    assert 'qje' in r
    assert 'main' in r
    assert 'placebo' in r
