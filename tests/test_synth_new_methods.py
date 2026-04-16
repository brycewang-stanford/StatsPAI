"""
Tests for new Synthetic Control methods (v0.9):

- Bayesian SCM (bayesian.py)
- BSTS / CausalImpact (bsts.py)
- Penalized SCM — Abadie & L'Hour 2021 (penscm.py)
- Forward DID (fdid.py)
- Cluster SCM (cluster.py)
- Sparse SCM (sparse.py)
- Kernel SCM (kernel.py)
"""

import pytest
import numpy as np
import pandas as pd
from statspai.core.results import CausalResult


# ====================================================================== #
#  Shared fixtures
# ====================================================================== #

@pytest.fixture
def panel_data():
    """
    Simulated panel: 1 treated + 10 donors, 20 periods.
    Treatment at period 11 with effect = 5.0.

    DGP: Y_it = alpha_i + beta_i * t + eps_it
          Treated unit gets +5.0 after t >= 11.
    """
    rng = np.random.default_rng(42)
    n_units = 11
    n_periods = 20
    treatment_time = 11

    records = []
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)

    for i in range(n_units):
        unit_name = f'unit_{i}'
        for t in range(1, n_periods + 1):
            y = alphas[i] + betas[i] * t + rng.normal(0, 0.3)
            if i == 0 and t >= treatment_time:
                y += 5.0
            records.append({
                'unit': unit_name,
                'time': t,
                'outcome': y,
            })

    return pd.DataFrame(records)


@pytest.fixture
def panel_no_effect():
    """Panel with no treatment effect (placebo check)."""
    rng = np.random.default_rng(99)
    n_units = 8
    n_periods = 15

    records = []
    for i in range(n_units):
        alpha = rng.normal(5, 1)
        for t in range(1, n_periods + 1):
            y = alpha + 0.3 * t + rng.normal(0, 0.2)
            records.append({'unit': f'u{i}', 'time': t, 'outcome': y})

    return pd.DataFrame(records)


# ====================================================================== #
#  Bayesian SCM
# ====================================================================== #

class TestBayesianSCM:

    def test_basic_bayesian(self, panel_data):
        """Bayesian SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.bayesian import bayesian_synth

        result = bayesian_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_iter=500, n_warmup=200, n_chains=1,
            seed=42, alpha=0.05,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 3.0, (
            f"Bayesian ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_returns_causal_result(self, panel_data):
        """Output should be a CausalResult with required fields."""
        from statspai.synth.bayesian import bayesian_synth

        result = bayesian_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_iter=300, n_warmup=100, n_chains=1, seed=42,
        )

        assert hasattr(result, 'estimate')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci')
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_posterior_diagnostics(self, panel_data):
        """model_info should contain posterior diagnostics."""
        from statspai.synth.bayesian import bayesian_synth

        result = bayesian_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_iter=400, n_warmup=150, n_chains=1, seed=42,
        )

        mi = result.model_info
        assert 'weights_posterior_mean' in mi
        assert 'sigma_posterior_mean' in mi
        assert mi['sigma_posterior_mean'] > 0

    def test_via_dispatcher(self, panel_data):
        """synth(method='bayesian') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='bayesian',
            n_iter=300, n_warmup=100, n_chains=1, seed=42,
        )

        assert isinstance(result, CausalResult)
        assert result.estimate != 0

    def test_citation(self, panel_data):
        """Bayesian SCM should have a citation."""
        from statspai.synth.bayesian import bayesian_synth

        result = bayesian_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_iter=300, n_warmup=100, n_chains=1, seed=42,
        )

        cite = result.cite()
        assert isinstance(cite, str)
        assert len(cite) > 0


# ====================================================================== #
#  BSTS / CausalImpact
# ====================================================================== #

class TestBSTS:

    def test_basic_bsts(self, panel_data):
        """BSTS should recover treatment effect ≈ 5.0."""
        from statspai.synth.bsts import bsts_synth

        result = bsts_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_simulations=200, seed=42,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 4.0, (
            f"BSTS ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_causal_impact_interface(self, panel_data):
        """causal_impact() wide-format interface should work."""
        from statspai.synth.bsts import causal_impact

        # Reshape to wide format for causal_impact
        wide = panel_data.pivot_table(
            index='time', columns='unit', values='outcome',
        )
        wide.columns.name = None

        result = causal_impact(
            wide, pre_period=(1, 10), post_period=(11, 20),
            outcome='unit_0', n_simulations=200, seed=42,
        )

        assert isinstance(result, CausalResult)
        assert result.estimate != 0

    def test_model_info(self, panel_data):
        """model_info should contain BSTS diagnostics."""
        from statspai.synth.bsts import bsts_synth

        result = bsts_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_simulations=100, seed=42,
        )

        mi = result.model_info
        assert 'sigma_obs' in mi
        assert 'model_type' in mi
        assert mi['model_type'] in ('local_level', 'local_linear_trend')

    def test_via_dispatcher(self, panel_data):
        """synth(method='bsts') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='bsts',
            n_simulations=100, seed=42,
        )

        assert isinstance(result, CausalResult)

    def test_cumulative_effect(self, panel_data):
        """BSTS should report cumulative effect."""
        from statspai.synth.bsts import bsts_synth

        result = bsts_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_simulations=200, seed=42,
        )

        mi = result.model_info
        assert 'cumulative_effect' in mi


# ====================================================================== #
#  Penalized SCM (Abadie & L'Hour 2021)
# ====================================================================== #

class TestPenalizedSCM:

    def test_basic_penscm(self, panel_data):
        """Penalized SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.penscm import penalized_synth

        result = penalized_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False, alpha=0.05,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 3.0, (
            f"PenSCM ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_pairwise_distances(self, panel_data):
        """model_info should contain pairwise distances."""
        from statspai.synth.penscm import penalized_synth

        result = penalized_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'pairwise_distances' in mi
        assert isinstance(mi['pairwise_distances'], dict)

    def test_penalty_types(self, panel_data):
        """All 3 penalty types should work."""
        from statspai.synth.penscm import penalized_synth

        for ptype in ('pairwise', 'max_dev', 'l1_pairwise'):
            result = penalized_synth(
                panel_data,
                outcome='outcome', unit='unit', time='time',
                treated_unit='unit_0', treatment_time=11,
                penalty_type=ptype, placebo=False,
            )
            assert isinstance(result, CausalResult)
            assert result.estimate != 0, f"penalty_type={ptype} returned 0"

    def test_weights_valid(self, panel_data):
        """Weights should be non-negative and sum to 1."""
        from statspai.synth.penscm import penalized_synth

        result = penalized_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        weights = result.model_info['weights']
        w_arr = np.array(list(weights.values()))
        assert np.all(w_arr >= -1e-6), "Weights should be non-negative"
        assert abs(np.sum(w_arr) - 1.0) < 1e-4, "Weights should sum to 1"

    def test_via_dispatcher(self, panel_data):
        """synth(method='penscm') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='penscm', placebo=False,
        )

        assert isinstance(result, CausalResult)


# ====================================================================== #
#  Forward DID
# ====================================================================== #

class TestForwardDID:

    def test_basic_fdid(self, panel_data):
        """FDID should recover treatment effect ≈ 5.0."""
        from statspai.synth.fdid import fdid

        result = fdid(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 3.0, (
            f"FDID ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_selected_donors(self, panel_data):
        """FDID should select a subset of donors."""
        from statspai.synth.fdid import fdid

        result = fdid(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'selected_donors' in mi
        assert len(mi['selected_donors']) > 0
        assert len(mi['selected_donors']) <= 10  # at most all donors

    def test_selection_methods(self, panel_data):
        """All selection methods should work."""
        from statspai.synth.fdid import fdid

        for method in ('forward', 'forward_cv'):
            result = fdid(
                panel_data,
                outcome='outcome', unit='unit', time='time',
                treated_unit='unit_0', treatment_time=11,
                method=method, placebo=False,
            )
            assert isinstance(result, CausalResult)
            assert result.estimate != 0

    def test_via_dispatcher(self, panel_data):
        """synth(method='fdid') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='fdid', placebo=False,
        )

        assert isinstance(result, CausalResult)

    def test_selection_path(self, panel_data):
        """model_info should contain the selection path."""
        from statspai.synth.fdid import fdid

        result = fdid(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'selection_path' in mi


# ====================================================================== #
#  Cluster SCM
# ====================================================================== #

class TestClusterSCM:

    def test_basic_cluster(self, panel_data):
        """Cluster SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.cluster import cluster_synth

        result = cluster_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False, seed=42,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 4.0, (
            f"Cluster ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_cluster_methods(self, panel_data):
        """All clustering methods should work."""
        from statspai.synth.cluster import cluster_synth

        for method in ('kmeans', 'hierarchical'):
            result = cluster_synth(
                panel_data,
                outcome='outcome', unit='unit', time='time',
                treated_unit='unit_0', treatment_time=11,
                cluster_method=method, placebo=False, seed=42,
            )
            assert isinstance(result, CausalResult)

    def test_cluster_info(self, panel_data):
        """model_info should contain cluster assignments."""
        from statspai.synth.cluster import cluster_synth

        result = cluster_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False, seed=42,
        )

        mi = result.model_info
        assert 'cluster_labels' in mi
        assert 'n_clusters' in mi

    def test_via_dispatcher(self, panel_data):
        """synth(method='cluster') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='cluster', placebo=False, seed=42,
        )

        assert isinstance(result, CausalResult)

    def test_augment_mode(self, panel_data):
        """Augment mode should add extra donors from other clusters."""
        from statspai.synth.cluster import cluster_synth

        result = cluster_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            augment=True, max_augment=2,
            placebo=False, seed=42,
        )

        assert isinstance(result, CausalResult)


# ====================================================================== #
#  Sparse SCM
# ====================================================================== #

class TestSparseSCM:

    def test_basic_sparse(self, panel_data):
        """Sparse SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.sparse import sparse_synth

        result = sparse_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 4.0, (
            f"Sparse ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_sparsity(self, panel_data):
        """Sparse SCM should produce sparse weights."""
        from statspai.synth.sparse import sparse_synth

        result = sparse_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'n_nonzero_weights' in mi
        # Should have fewer non-zero weights than total donors
        assert mi['n_nonzero_weights'] <= 10

    def test_modes(self, panel_data):
        """All sparse modes should work."""
        from statspai.synth.sparse import sparse_synth

        for mode in ('lasso', 'constrained_lasso'):
            result = sparse_synth(
                panel_data,
                outcome='outcome', unit='unit', time='time',
                treated_unit='unit_0', treatment_time=11,
                mode=mode, placebo=False,
            )
            assert isinstance(result, CausalResult)

    def test_via_dispatcher(self, panel_data):
        """synth(method='sparse') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='sparse', placebo=False,
        )

        assert isinstance(result, CausalResult)

    def test_cv_lambda(self, panel_data):
        """CV should auto-select lambda."""
        from statspai.synth.sparse import sparse_synth

        result = sparse_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'lambda_w' in mi
        assert mi['lambda_w'] > 0


# ====================================================================== #
#  Kernel SCM
# ====================================================================== #

class TestKernelSCM:

    def test_basic_kernel(self, panel_data):
        """Kernel SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.kernel import kernel_synth

        result = kernel_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 4.0, (
            f"Kernel ATT={result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_kernel_types(self, panel_data):
        """All kernel types should work."""
        from statspai.synth.kernel import kernel_synth

        for kernel in ('rbf', 'polynomial', 'laplacian'):
            result = kernel_synth(
                panel_data,
                outcome='outcome', unit='unit', time='time',
                treated_unit='unit_0', treatment_time=11,
                kernel=kernel, placebo=False,
            )
            assert isinstance(result, CausalResult)
            assert result.estimate != 0, f"kernel={kernel} returned 0"

    def test_kernel_ridge(self, panel_data):
        """Kernel ridge SCM should work."""
        from statspai.synth.kernel import kernel_ridge_synth

        result = kernel_ridge_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 4.0

    def test_weights_valid(self, panel_data):
        """Kernel SCM weights should sum to 1 and be non-negative."""
        from statspai.synth.kernel import kernel_synth

        result = kernel_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        weights = result.model_info['weights']
        w_arr = np.array(list(weights.values()))
        assert np.all(w_arr >= -1e-6), "Weights should be non-negative"
        assert abs(np.sum(w_arr) - 1.0) < 1e-4, "Weights should sum to 1"

    def test_via_dispatcher(self, panel_data):
        """synth(method='kernel') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='kernel', placebo=False,
        )

        assert isinstance(result, CausalResult)

    def test_kernel_ridge_via_dispatcher(self, panel_data):
        """synth(method='kernel_ridge') should route correctly."""
        from statspai.synth import synth

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='kernel_ridge',
        )

        assert isinstance(result, CausalResult)

    def test_median_heuristic(self, panel_data):
        """sigma=None should trigger median heuristic."""
        from statspai.synth.kernel import kernel_synth

        result = kernel_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            sigma=None, placebo=False,
        )

        mi = result.model_info
        assert 'sigma' in mi
        assert mi['sigma'] > 0


# ====================================================================== #
#  Unified dispatcher coverage
# ====================================================================== #

class TestUnifiedDispatcher:

    @pytest.mark.parametrize("method", [
        'bayesian', 'bsts', 'penscm', 'fdid',
        'cluster', 'sparse', 'kernel', 'kernel_ridge',
    ])
    def test_all_new_methods_return_causal_result(self, panel_data, method):
        """All new methods via synth() should return CausalResult."""
        from statspai.synth import synth

        kwargs = {'placebo': False}
        if method == 'bayesian':
            kwargs.update(n_iter=300, n_warmup=100, n_chains=1, seed=42)
        elif method == 'bsts':
            kwargs.update(n_simulations=100, seed=42)
        elif method == 'cluster':
            kwargs.update(seed=42)

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method=method, **kwargs,
        )

        assert isinstance(result, CausalResult)
        assert result.estimate is not None
        assert not np.isnan(result.estimate)

    @pytest.mark.parametrize("method", [
        'penscm', 'fdid', 'cluster', 'sparse', 'kernel',
    ])
    def test_new_methods_positive_effect(self, panel_data, method):
        """All new methods should detect a positive effect (true ATT = 5.0)."""
        from statspai.synth import synth

        kwargs = {'placebo': False}
        if method == 'cluster':
            kwargs.update(seed=42)

        result = synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method=method, **kwargs,
        )

        assert result.estimate > 0, (
            f"method={method}: ATT={result.estimate:.2f} should be positive"
        )
