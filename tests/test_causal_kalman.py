"""Tests for causal_kalman / assimilative_causal (Nature Comms 2026)."""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from statspai.assimilation.kalman import (
    causal_kalman, assimilative_causal, AssimilationResult,
)


def test_kalman_shrinks_sd_with_more_obs():
    r = causal_kalman(
        estimates=[2.0, 2.1, 1.95, 2.05, 1.98],
        standard_errors=[0.3, 0.3, 0.3, 0.3, 0.3],
    )
    # Final posterior SD should be smaller than any single SE
    assert r.final_sd < 0.3
    # Posterior mean should be near the average
    assert abs(r.final_mean - 2.016) < 0.1


def test_kalman_posterior_within_ci():
    r = causal_kalman(
        estimates=[1.0, 0.9, 1.1],
        standard_errors=[0.2, 0.2, 0.2],
    )
    lo, hi = r.final_ci
    assert lo < r.final_mean < hi


def test_kalman_trajectory_shapes():
    n = 5
    r = causal_kalman(
        estimates=list(np.linspace(1, 2, n)),
        standard_errors=[0.25] * n,
    )
    assert len(r.posterior_mean) == n
    assert len(r.posterior_sd) == n
    assert r.posterior_ci.shape == (n, 2)
    # Posterior SD should be monotonically non-increasing
    for i in range(1, n):
        assert r.posterior_sd[i] <= r.posterior_sd[i - 1] + 1e-9


def test_kalman_with_process_variance_tracks_drift():
    """If process_var > 0, Kalman allows for drift and weighs recent obs more."""
    # First 5 estimates around 0, then drift to 2
    ests = [0.0, 0.05, -0.05, 0.02, 1.5, 1.9, 2.0]
    r = causal_kalman(
        estimates=ests,
        standard_errors=[0.2] * len(ests),
        process_var=0.5,
    )
    # Final mean should land somewhere between the prior 0 and the recent 2
    assert 0.5 < r.final_mean < 2.2


def test_assimilative_causal_with_estimator_callback():
    """Test the higher-level wrapper that accepts data batches + estimator."""
    batches = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.5, 3.5]),
        np.array([1.2, 2.2, 3.2]),
    ]
    def estimator(batch):
        return float(batch.mean()), float(batch.std() / np.sqrt(len(batch)))
    r = assimilative_causal(batches, estimator=estimator, backend="kalman")
    assert isinstance(r, AssimilationResult)
    assert 1.5 < r.final_mean < 2.8


def test_kalman_raises_on_length_mismatch():
    with pytest.raises((ValueError, TypeError)):
        causal_kalman(
            estimates=[1.0, 2.0, 3.0],
            standard_errors=[0.1, 0.2],  # one fewer
        )
