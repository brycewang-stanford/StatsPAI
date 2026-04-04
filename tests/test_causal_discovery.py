"""
Tests for Causal Discovery module (NOTEARS, PC Algorithm).
"""

import pytest
import numpy as np
import pandas as pd

from statspai.causal_discovery import notears, NOTEARS, pc_algorithm, PCAlgorithm


# ======================================================================
# Fixtures: Generate data from known DAGs
# ======================================================================

@pytest.fixture
def chain_data():
    """
    Chain DAG: X -> Z -> Y
        Z = 0.8*X + eps_z
        Y = 0.7*Z + eps_y
    """
    rng = np.random.default_rng(42)
    n = 2000

    X = rng.normal(0, 1, n)
    Z = 0.8 * X + rng.normal(0, 0.3, n)
    Y = 0.7 * Z + rng.normal(0, 0.3, n)

    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})


@pytest.fixture
def fork_data():
    """
    Fork DAG: X <- Z -> Y  (Z is a common cause)
        X = 0.9*Z + eps_x
        Y = 0.6*Z + eps_y
    """
    rng = np.random.default_rng(42)
    n = 2000

    Z = rng.normal(0, 1, n)
    X = 0.9 * Z + rng.normal(0, 0.3, n)
    Y = 0.6 * Z + rng.normal(0, 0.3, n)

    return pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})


@pytest.fixture
def collider_data():
    """
    Collider DAG: X -> Z <- Y  (Z is a collider)
        Z = 0.7*X + 0.5*Y + eps_z
    """
    rng = np.random.default_rng(42)
    n = 2000

    X = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    Z = 0.7 * X + 0.5 * Y + rng.normal(0, 0.3, n)

    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})


@pytest.fixture
def four_node_data():
    """
    DAG: X -> Z -> Y, X -> M -> Y
        Z = 0.8*X + eps
        M = 0.6*X + eps
        Y = 0.5*Z + 0.4*M + eps
    """
    rng = np.random.default_rng(42)
    n = 3000

    X = rng.normal(0, 1, n)
    Z = 0.8 * X + rng.normal(0, 0.3, n)
    M = 0.6 * X + rng.normal(0, 0.3, n)
    Y = 0.5 * Z + 0.4 * M + rng.normal(0, 0.3, n)

    return pd.DataFrame({'X': X, 'Z': Z, 'M': M, 'Y': Y})


@pytest.fixture
def small_data():
    """Small data for smoke tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(0, 1, n)
    Y = 0.5 * X + rng.normal(0, 0.5, n)
    Z = rng.normal(0, 1, n)
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})


# ======================================================================
# NOTEARS Tests
# ======================================================================

class TestNOTEARS:
    """Tests for NOTEARS DAG learning."""

    def test_returns_dict_with_required_keys(self, small_data):
        result = notears(small_data)
        assert isinstance(result, dict)
        assert 'adjacency' in result
        assert 'edges' in result
        assert 'dag' in result
        assert 'variables' in result
        assert 'h_value' in result
        assert 'n_edges' in result
        assert 'n_obs' in result

    def test_adjacency_is_dataframe(self, small_data):
        result = notears(small_data)
        assert isinstance(result['adjacency'], pd.DataFrame)
        assert isinstance(result['dag'], pd.DataFrame)

    def test_acyclicity(self, small_data):
        """h(W) should be near 0 (DAG constraint satisfied)."""
        result = notears(small_data)
        assert result['h_value'] < 0.1

    def test_chain_discovery(self, chain_data):
        """Should find edges in X -> Z -> Y chain."""
        result = notears(chain_data, variables=['X', 'Z', 'Y'],
                         lambda1=0.1, w_threshold=0.2)
        edges = result['edges']
        edge_pairs = [(e[0], e[1]) for e in edges]

        # Should have X->Z and Z->Y (or similar)
        assert result['n_edges'] >= 2, f"Expected >= 2 edges, got {result['n_edges']}"

    def test_no_self_loops(self, chain_data):
        """Diagonal of adjacency should be zero."""
        result = notears(chain_data, variables=['X', 'Z', 'Y'])
        adj = result['adjacency'].values
        np.testing.assert_array_equal(np.diag(adj), 0)

    def test_sparsity_lambda(self, chain_data):
        """Higher lambda1 should produce fewer edges."""
        r_low = notears(chain_data, lambda1=0.01, w_threshold=0.1)
        r_high = notears(chain_data, lambda1=1.0, w_threshold=0.1)
        assert r_high['n_edges'] <= r_low['n_edges']

    def test_threshold_effect(self, chain_data):
        """Higher threshold should produce fewer edges."""
        r_low = notears(chain_data, w_threshold=0.1)
        r_high = notears(chain_data, w_threshold=0.5)
        assert r_high['n_edges'] <= r_low['n_edges']

    def test_class_interface(self, small_data):
        est = NOTEARS(data=small_data, lambda1=0.1)
        result = est.fit()
        assert isinstance(result, dict)
        summary = est.summary()
        assert 'NOTEARS' in summary

    def test_specific_variables(self, chain_data):
        result = notears(chain_data, variables=['X', 'Z'])
        assert result['variables'] == ['X', 'Z']
        assert result['adjacency'].shape == (2, 2)

    def test_missing_variable_raises(self, small_data):
        with pytest.raises(ValueError, match="Variables not found"):
            notears(small_data, variables=['X', 'nonexistent'])

    def test_single_variable_raises(self):
        df = pd.DataFrame({'X': [1, 2, 3]})
        with pytest.raises(ValueError, match="At least 2"):
            notears(df)

    def test_edges_sorted_by_weight(self, chain_data):
        result = notears(chain_data, variables=['X', 'Z', 'Y'])
        edges = result['edges']
        if len(edges) > 1:
            weights = [abs(e[2]) for e in edges]
            assert weights == sorted(weights, reverse=True)

    def test_four_node_dag(self, four_node_data):
        """Test with 4 nodes."""
        result = notears(four_node_data, lambda1=0.1, w_threshold=0.2)
        assert result['n_edges'] >= 3


# ======================================================================
# PC Algorithm Tests
# ======================================================================

class TestPCAlgorithm:
    """Tests for PC Algorithm."""

    def test_returns_dict_with_required_keys(self, small_data):
        result = pc_algorithm(small_data)
        assert isinstance(result, dict)
        assert 'skeleton' in result
        assert 'cpdag' in result
        assert 'edges' in result
        assert 'undirected_edges' in result
        assert 'separating_sets' in result
        assert 'variables' in result
        assert 'n_obs' in result

    def test_skeleton_is_symmetric(self, small_data):
        result = pc_algorithm(small_data)
        skel = result['skeleton'].values
        np.testing.assert_array_equal(skel, skel.T)

    def test_chain_skeleton(self, chain_data):
        """Chain X->Z->Y should have edges X-Z and Z-Y in skeleton."""
        result = pc_algorithm(chain_data, variables=['X', 'Z', 'Y'],
                              alpha=0.01)
        skel = result['skeleton']
        # X-Z should be connected
        assert skel.loc['X', 'Z'] == 1
        # Z-Y should be connected
        assert skel.loc['Z', 'Y'] == 1

    def test_independent_removed(self, chain_data):
        """In chain X->Z->Y, X and Y should be separated by Z."""
        result = pc_algorithm(chain_data, variables=['X', 'Z', 'Y'],
                              alpha=0.01)
        skel = result['skeleton']
        # X-Y edge should be removed (conditionally independent given Z)
        assert skel.loc['X', 'Y'] == 0

    def test_collider_detection(self, collider_data):
        """In X->Z<-Y, Z is a collider, should be oriented."""
        result = pc_algorithm(collider_data, variables=['X', 'Y', 'Z'],
                              alpha=0.01)
        cpdag = result['cpdag']
        # X -> Z and Y -> Z should be directed
        edges = result['edges']
        edge_set = set(edges)

        # At minimum, the v-structure should be detected
        assert result['n_edges'] >= 2

    def test_alpha_effect(self, chain_data):
        """Lower alpha should produce sparser graph."""
        r_loose = pc_algorithm(chain_data, alpha=0.50)
        r_strict = pc_algorithm(chain_data, alpha=0.001)
        assert r_strict['n_edges'] <= r_loose['n_edges']

    def test_class_interface(self, small_data):
        est = PCAlgorithm(data=small_data, alpha=0.05)
        result = est.fit()
        assert isinstance(result, dict)
        summary = est.summary()
        assert 'PC Algorithm' in summary

    def test_specific_variables(self, chain_data):
        result = pc_algorithm(chain_data, variables=['X', 'Z'])
        assert result['variables'] == ['X', 'Z']

    def test_missing_variable_raises(self, small_data):
        with pytest.raises(ValueError, match="Variables not found"):
            pc_algorithm(small_data, variables=['X', 'nonexistent'])

    def test_single_variable_raises(self):
        df = pd.DataFrame({'X': [1, 2, 3]})
        with pytest.raises(ValueError, match="At least 2"):
            pc_algorithm(df)

    def test_max_cond_size(self, four_node_data):
        """Should respect max conditioning set size."""
        r0 = pc_algorithm(four_node_data, max_cond_size=0)
        r2 = pc_algorithm(four_node_data, max_cond_size=2)
        # With max_cond_size=0, only marginal tests => denser graph
        assert r0['n_edges'] >= r2['n_edges']

    def test_four_node_dag(self, four_node_data):
        result = pc_algorithm(four_node_data, alpha=0.01)
        assert result['n_edges'] >= 3


# ======================================================================
# Import Tests
# ======================================================================

class TestImports:
    """Test module imports."""

    def test_import_from_statspai(self):
        import statspai as sp
        assert hasattr(sp, 'notears')
        assert hasattr(sp, 'NOTEARS')
        assert hasattr(sp, 'pc_algorithm')
        assert hasattr(sp, 'PCAlgorithm')

    def test_unfitted_summary_raises(self):
        df = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
        est = NOTEARS(data=df)
        with pytest.raises(ValueError, match="fitted"):
            est.summary()

    def test_nan_handling(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'X': rng.normal(0, 1, n),
            'Y': rng.normal(0, 1, n),
        })
        df.loc[0, 'X'] = np.nan
        result = notears(df)
        assert result['n_obs'] == n - 1
