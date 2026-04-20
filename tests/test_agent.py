"""Tests for sp.agent — LLM tool-definition surface + error remediation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import (
    tool_manifest,
    execute_tool,
    TOOL_REGISTRY,
    remediate,
    REMEDIATIONS,
)


@pytest.fixture(scope='module')
def sample_df():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        'd': rng.binomial(1, 0.5, n),
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
    })
    df['y'] = 1 + 2 * df.d + 0.5 * df.x1 + rng.normal(size=n)
    return df


# ---------------------------------------------------------------------------
# Tool manifest validity
# ---------------------------------------------------------------------------

class TestToolManifest:
    def test_manifest_is_list(self):
        m = tool_manifest()
        assert isinstance(m, list)
        assert len(m) >= 6

    def test_every_tool_has_required_keys(self):
        for t in tool_manifest():
            assert 'name' in t
            assert 'description' in t
            assert 'input_schema' in t
            assert isinstance(t['input_schema'], dict)
            assert t['input_schema'].get('type') == 'object'
            assert 'properties' in t['input_schema']

    def test_manifest_is_json_serializable(self):
        m = tool_manifest()
        # Should round-trip through JSON without errors
        s = json.dumps(m)
        m2 = json.loads(s)
        assert len(m) == len(m2)

    def test_every_tool_name_is_unique(self):
        names = [t['name'] for t in tool_manifest()]
        assert len(names) == len(set(names))

    def test_core_tools_present(self):
        names = {t['name'] for t in tool_manifest()}
        required = {'regress', 'did', 'callaway_santanna', 'rdrobust',
                    'ivreg', 'ebalance', 'check_identification', 'causal'}
        missing = required - names
        assert not missing, f"Missing tools: {missing}"

    def test_every_input_schema_has_required_array(self):
        for t in tool_manifest():
            assert 'required' in t['input_schema']
            assert isinstance(t['input_schema']['required'], list)


# ---------------------------------------------------------------------------
# Tool execution — happy paths
# ---------------------------------------------------------------------------

class TestExecuteToolHappy:
    def test_regress(self, sample_df):
        out = execute_tool('regress', {'formula': 'y ~ d + x1'},
                           data=sample_df)
        assert 'error' not in out
        assert 'coefficients' in out
        assert 'd' in out['coefficients']
        assert abs(out['coefficients']['d']['estimate'] - 2.0) < 0.3

    def test_did(self, sample_df):
        # Build a 2-period DID dataset
        rng = np.random.default_rng(5)
        rows = []
        for i in range(200):
            tr = 1 if i < 100 else 0
            for t in [0, 1]:
                y = 1.0 + 0.3*t + 0.5*tr + 2.0*tr*t + rng.normal(scale=0.5)
                rows.append({'i': i, 't': t, 'treated': tr,
                             'post': t, 'y': y})
        df = pd.DataFrame(rows)
        out = execute_tool(
            'did',
            {'y': 'y', 'treat': 'treated', 'time': 't', 'post': 'post'},
            data=df,
        )
        assert 'error' not in out
        assert 'estimate' in out
        assert abs(out['estimate'] - 2.0) < 0.2

    def test_check_identification(self, sample_df):
        out = execute_tool(
            'check_identification',
            {'y': 'y', 'treatment': 'd',
             'covariates': ['x1', 'x2'],
             'design': 'observational'},
            data=sample_df,
        )
        assert 'error' not in out
        assert 'verdict' in out
        assert out['verdict'] in ('OK', 'WARNINGS', 'BLOCKERS')
        assert 'findings' in out

    def test_causal_end_to_end(self, sample_df):
        out = execute_tool(
            'causal',
            {'y': 'y', 'treatment': 'd',
             'covariates': ['x1', 'x2'],
             'design': 'observational'},
            data=sample_df,
        )
        assert 'error' not in out
        assert 'verdict' in out
        assert 'estimate' in out
        # estimate might be None if result doesn't expose .estimate (regress)
        # but verdict should always be populated


# ---------------------------------------------------------------------------
# Tool execution — error path
# ---------------------------------------------------------------------------

class TestExecuteToolErrors:
    def test_unknown_tool_returns_error_dict(self, sample_df):
        out = execute_tool('not_a_real_tool', {}, data=sample_df)
        assert 'error' in out
        assert 'available_tools' in out

    def test_formula_error_triggers_remediation(self, sample_df):
        out = execute_tool('regress', {'formula': 'not a valid formula'},
                           data=sample_df)
        assert 'error' in out
        assert 'remediation' in out
        assert out['remediation']['category'] in ('formula', 'unknown')

    def test_missing_column_triggers_remediation(self, sample_df):
        out = execute_tool(
            'regress',
            {'formula': 'y ~ nonexistent_column'},
            data=sample_df,
        )
        assert 'error' in out
        assert 'remediation' in out


# ---------------------------------------------------------------------------
# Remediation registry
# ---------------------------------------------------------------------------

class TestRemediation:
    def test_remediate_returns_dict_with_required_keys(self):
        err = KeyError('my_column')
        out = remediate(err)
        for key in ('category', 'diagnosis', 'fix', 'matched'):
            assert key in out

    def test_missing_column_pattern_matches(self):
        err = KeyError('my_column')
        out = remediate(err)
        assert out['matched']
        assert out['category'] == 'missing_column'

    def test_unknown_error_falls_back_gracefully(self):
        class CustomError(Exception):
            pass
        out = remediate(CustomError('something truly novel'))
        assert out['matched'] is False
        assert out['category'] == 'unknown'
        assert out['exception_type'] == 'CustomError'

    def test_weak_instrument_pattern(self):
        err = ValueError('weak instrument F=2.1')
        out = remediate(err)
        assert out['matched']
        assert out['category'] == 'weak_instrument'

    def test_context_is_attached(self):
        err = KeyError('x')
        out = remediate(err, context={'tool': 'regress'})
        assert 'context' in out
        assert out['context']['tool'] == 'regress'


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def test_agent_module_importable():
    assert hasattr(sp, 'agent')
    assert hasattr(sp.agent, 'tool_manifest')
    assert hasattr(sp.agent, 'execute_tool')
    assert hasattr(sp.agent, 'remediate')


def test_tool_registry_exposed():
    from statspai.agent import TOOL_REGISTRY
    assert isinstance(TOOL_REGISTRY, list)
    assert len(TOOL_REGISTRY) >= 6
    for spec in TOOL_REGISTRY:
        assert 'statspai_fn' in spec
        assert 'serializer' in spec
