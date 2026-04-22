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
# Result-object agent surface (to_dict / for_agent / to_json)
# ---------------------------------------------------------------------------

class TestResultToDict:
    """EconometricResults.to_dict() / for_agent() / to_json()."""

    def test_econometric_to_dict_has_required_keys(self, sample_df):
        r = sp.regress('y ~ d + x1', data=sample_df)
        d = r.to_dict()
        for k in ('method', 'coefficients', 'diagnostics', 'n_obs',
                  'glance'):
            assert k in d, f"{k!r} missing from to_dict()"

    def test_econometric_coefficients_have_full_schema(self, sample_df):
        r = sp.regress('y ~ d + x1', data=sample_df)
        d = r.to_dict()
        assert 'd' in d['coefficients']
        cell = d['coefficients']['d']
        for k in ('estimate', 'std_error', 't_statistic', 'p_value',
                  'conf_low', 'conf_high'):
            assert k in cell, f"{k!r} missing from coefficient row"
            assert cell[k] is None or isinstance(cell[k],
                                                  (int, float, bool))

    def test_econometric_to_json_round_trips(self, sample_df):
        r = sp.regress('y ~ d + x1', data=sample_df)
        payload = r.to_json()
        reloaded = json.loads(payload)
        assert reloaded['method']
        assert 'd' in reloaded['coefficients']

    def test_econometric_for_agent_extends_to_dict(self, sample_df):
        r = sp.regress('y ~ d + x1', data=sample_df)
        base = r.to_dict()
        fa = r.for_agent()
        assert set(base.keys()).issubset(set(fa.keys()))
        for k in ('violations', 'warnings', 'next_steps',
                  'suggested_functions'):
            assert k in fa, f"for_agent missing {k!r}"
            assert isinstance(fa[k], list)


class TestCausalResultToDict:
    """CausalResult.to_dict() / for_agent() / to_json()."""

    @pytest.fixture(scope='class')
    def did_result(self):
        rng = np.random.default_rng(5)
        rows = []
        for i in range(200):
            tr = 1 if i < 100 else 0
            for t in [0, 1]:
                y = 1.0 + 0.3*t + 0.5*tr + 2.0*tr*t + rng.normal(scale=0.5)
                rows.append({'i': i, 't': t, 'treated': tr,
                             'post': t, 'y': y})
        df = pd.DataFrame(rows)
        return sp.did(df, y='y', treat='treated', time='t', post='post')

    def test_causal_to_dict_has_required_keys(self, did_result):
        d = did_result.to_dict()
        for k in ('method', 'estimand', 'estimate', 'se', 'pvalue',
                  'ci', 'alpha', 'n_obs', 'diagnostics'):
            assert k in d, f"{k!r} missing"
        assert isinstance(d['ci'], list) and len(d['ci']) == 2

    def test_causal_estimate_close_to_truth(self, did_result):
        # DGP: ATT = 2.0 with treatment-period interaction
        d = did_result.to_dict()
        assert abs(d['estimate'] - 2.0) < 0.2

    def test_causal_to_json_round_trips(self, did_result):
        payload = did_result.to_json()
        reloaded = json.loads(payload)
        assert reloaded['method']
        assert reloaded['estimate'] is not None

    def test_causal_for_agent_includes_violations(self, did_result):
        fa = did_result.for_agent()
        for k in ('violations', 'warnings', 'next_steps',
                  'suggested_functions'):
            assert k in fa
            assert isinstance(fa[k], list)

    def test_causal_detail_head_opt_out(self, did_result):
        d = did_result.to_dict(detail_head=0)
        assert 'detail_head' not in d

    def test_causal_to_dict_is_flat_vs_agent_summary_nested(self,
                                                             did_result):
        # Documented API contract: to_dict is flat, to_agent_summary is
        # nested under 'point'. They should both survive json.dumps.
        flat = did_result.to_dict()
        nested = did_result.to_agent_summary()
        assert 'estimate' in flat
        assert 'point' in nested
        json.dumps(flat, default=str)
        json.dumps(nested, default=str)


class TestExecuteToolUsesToDict:
    """`execute_tool` should route results through result.to_dict()."""

    def test_did_result_matches_to_dict(self, sample_df):
        rng = np.random.default_rng(5)
        rows = []
        for i in range(200):
            tr = 1 if i < 100 else 0
            for t in [0, 1]:
                y = 1.0 + 0.3*t + 0.5*tr + 2.0*tr*t + rng.normal(scale=0.5)
                rows.append({'i': i, 't': t, 'treated': tr,
                             'post': t, 'y': y})
        df = pd.DataFrame(rows)
        out = execute_tool('did',
                           {'y': 'y', 'treat': 'treated', 't': 't',
                            'time': 't', 'post': 'post'},
                           data=df)
        # Should have the flat to_dict shape, not the legacy shape.
        assert 'estimate' in out
        assert 'method' in out
        assert 'diagnostics' in out


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


# ---------------------------------------------------------------------------
# Auto-generated MCP manifest (registry coverage expansion)
# ---------------------------------------------------------------------------

class TestAutoManifest:
    def test_auto_manifest_covers_many_tools(self):
        from statspai.agent.auto_tools import auto_tool_manifest
        m = auto_tool_manifest()
        # Registry has 800+ fns; after whitelist + class filter we should
        # still be well above the 8-tool baseline.
        assert len(m) >= 80, f"only {len(m)} auto tools; expected ≥ 80"

    def test_auto_manifest_tools_have_valid_schema(self):
        from statspai.agent.auto_tools import auto_tool_manifest
        for t in auto_tool_manifest():
            assert 'name' in t and t['name']
            assert 'description' in t
            assert isinstance(t['input_schema'], dict)
            assert t['input_schema'].get('type') == 'object'
            assert 'properties' in t['input_schema']
            assert 'required' in t['input_schema']

    def test_auto_manifest_is_json_serializable(self):
        from statspai.agent.auto_tools import auto_tool_manifest
        m = auto_tool_manifest()
        # No dataclass sentinels or callables leaking into defaults.
        json.dumps(m)

    def test_auto_manifest_skips_classes(self):
        from statspai.agent.auto_tools import auto_tool_manifest
        names = [t['name'] for t in auto_tool_manifest()]
        # PascalCase → class, shouldn't appear
        pascal = [n for n in names if n[:1].isupper()]
        assert not pascal, f"class names leaked into manifest: {pascal[:5]}"

    def test_merged_manifest_preserves_curated_eight(self):
        from statspai.agent import tool_manifest
        names = {t['name'] for t in tool_manifest()}
        required = {
            'regress', 'did', 'callaway_santanna', 'rdrobust', 'ivreg',
            'ebalance', 'check_identification', 'causal',
        }
        assert required <= names, f"missing curated: {required - names}"

    def test_merged_manifest_no_duplicates(self):
        from statspai.agent import tool_manifest
        names = [t['name'] for t in tool_manifest()]
        assert len(names) == len(set(names))

    def test_merged_manifest_is_json_serializable(self):
        from statspai.agent import tool_manifest
        json.dumps(tool_manifest())

    def test_curated_only_flag(self):
        from statspai.agent import tool_manifest
        # Curated registry grew from 8 → 13 when we added recommend,
        # honest_did, bacon_decomposition, sensitivity, spec_curve in
        # P0.  Keep as a floor so future additions don't break the
        # contract.
        assert len(tool_manifest(curated_only=True)) >= 13
        assert len(tool_manifest()) >= 80
        # Merged >= curated (merge adds at least the auto layer).
        assert len(tool_manifest()) > len(tool_manifest(curated_only=True))

    def test_auto_manifest_agent_card_enrichment(self):
        """Tools whose spec has agent-card metadata should carry a
        richer description (assumptions / alternatives)."""
        from statspai.agent.auto_tools import auto_tool_manifest
        m = {t['name']: t for t in auto_tool_manifest()}
        iv = m.get('iv')
        if iv is None:
            import pytest
            pytest.skip("iv not in registry in this environment")
        # The merge adds 'Assumptions' when spec.agent_card() has any.
        assert 'Assumptions:' in iv['description']

    def test_auto_manifest_strips_data_param(self):
        from statspai.agent.auto_tools import auto_tool_manifest
        for t in auto_tool_manifest()[:20]:
            assert 'data' not in (t['input_schema'].get('properties') or {}), (
                f"{t['name']} still exposes 'data' (must be stripped, "
                "MCP server injects data_path)"
            )


# ---------------------------------------------------------------------------
# Expanded remediation registry (causal-specific + exception bridge)
# ---------------------------------------------------------------------------

class TestCausalRemediations:
    """Regex patterns for causal-specific failure modes."""

    @pytest.mark.parametrize('msg,expected', [
        ('parallel trends violated at p=0.01', 'parallel_trends_fail'),
        ('negative weights on 15% of comparisons',
         'negative_weights_twfe'),
        ('McCrary density test rejects', 'mccrary_reject'),
        ('rhat_max > 1.05 on 3 chains', 'bayes_convergence'),
        ('ESS < 200 on posterior', 'bayes_convergence'),
        ('overlap violated: extreme propensity scores',
         'overlap_violation'),
        ('SBW infeasible at tolerance 0.001', 'sbw_infeasible'),
        ('SMD > 0.2 remaining after match', 'matching_unbalanced'),
        ('Hansen J reject at 1%', 'iv_exclusion_fail'),
        ('Hausman reject FE vs RE', 'hausman_reject'),
        ('placebo significant at -3 periods', 'placebo_fail'),
        ('conformal coverage fail: empirical 0.78', 'ci_coverage_fail'),
        ('orthogonality fail in DML score', 'dml_ortho_fail'),
        ('design not inferred from columns', 'identification_unknown'),
        ('pre-treatment RMSE large relative to sd_y',
         'synth_no_pretrend_fit'),
    ])
    def test_pattern_matches_expected_category(self, msg, expected):
        out = remediate(ValueError(msg))
        assert out['matched'], f"no match for {msg!r}"
        assert out['category'] == expected, (
            f"{msg!r} → {out['category']!r}, expected {expected!r}")

    def test_registry_is_large_enough(self):
        # P0 target: ≥ 25 entries. Regression guard.
        assert len(REMEDIATIONS) >= 25


class TestStatsPAIErrorBridge:
    """StatsPAIError subclass → remediate() surfaces hint + alts."""

    def test_assumption_violation_routes_to_assumption_category(self):
        from statspai.exceptions import AssumptionViolation
        err = AssumptionViolation(
            'generic assumption message — no pattern hit',
            recovery_hint='switch to sp.callaway_santanna',
            alternative_functions=['sp.callaway_santanna',
                                    'sp.honest_did'],
        )
        out = remediate(err)
        assert out['matched']
        assert out['category'] in (
            'assumption_violation', 'parallel_trends_fail',
        )
        assert 'callaway_santanna' in out['fix']
        assert out['alternative_functions'] == [
            'sp.callaway_santanna', 'sp.honest_did']

    def test_identification_failure_routes(self):
        from statspai.exceptions import IdentificationFailure
        err = IdentificationFailure(
            'estimand not identified without a valid instrument',
            recovery_hint='provide a valid instrument or use bounds',
            alternative_functions=['sp.bounds'],
        )
        out = remediate(err)
        assert out['matched']
        assert out['category'] in (
            'identification_failure', 'assumption_violation',
        )

    def test_convergence_failure_routes(self):
        from statspai.exceptions import ConvergenceFailure
        err = ConvergenceFailure(
            'NUTS did not reach tolerance',
            recovery_hint='raise tune to 4000',
            diagnostics={'iterations': 1000},
        )
        out = remediate(err)
        assert out['matched']
        assert out['category'] == 'convergence_failure'
        assert out['diagnostics'] == {'iterations': 1000}

    def test_method_incompatibility_routes(self):
        from statspai.exceptions import MethodIncompatibility
        err = MethodIncompatibility(
            'HAC requires a time index which is not set',
        )
        out = remediate(err)
        assert out['matched']
        assert out['category'] == 'method_incompatibility'

    def test_numerical_instability_routes(self):
        from statspai.exceptions import NumericalInstability
        err = NumericalInstability(
            'singular design matrix: rank 4 < 5',
        )
        out = remediate(err)
        assert out['matched']
        # Either the specific taxonomy or the pre-existing
        # 'collinearity' pattern — both are valid outcomes.
        assert out['category'] in (
            'numerical_instability', 'collinearity')

    def test_data_insufficient_routes(self):
        from statspai.exceptions import DataInsufficient
        err = DataInsufficient('too few treated cohorts')
        out = remediate(err)
        assert out['matched']
        assert out['category'] in (
            'data_insufficient', 'sample_size', 'small_cohort')


# ---------------------------------------------------------------------------
# Housekeeping: MCP server version sync + curated tool additions
# ---------------------------------------------------------------------------

class TestMCPServerVersion:
    def test_server_version_matches_package(self):
        from statspai.agent.mcp_server import SERVER_VERSION
        assert SERVER_VERSION == sp.__version__

    def test_server_name_unchanged(self):
        from statspai.agent.mcp_server import SERVER_NAME
        assert SERVER_NAME == 'statspai'


class TestCuratedToolAdditions:
    """P0 added recommend / honest_did / bacon / sensitivity / spec_curve."""

    def test_recommend_in_curated(self):
        from statspai.agent import tool_manifest
        names = {t['name'] for t in tool_manifest(curated_only=True)}
        for n in ('recommend', 'honest_did', 'bacon_decomposition',
                  'sensitivity', 'spec_curve'):
            assert n in names, f"{n} missing from curated manifest"

    def test_curated_tools_have_valid_schemas(self):
        from statspai.agent import TOOL_REGISTRY
        for spec in TOOL_REGISTRY:
            assert spec['name']
            assert spec['description']
            assert isinstance(spec['input_schema'], dict)
            assert spec['input_schema'].get('type') == 'object'
            assert 'properties' in spec['input_schema']
            assert 'required' in spec['input_schema']
            assert callable(spec['serializer'])
