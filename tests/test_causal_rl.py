"""Smoke tests for sp.causal_rl namespace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def test_causal_rl_benchmark():
    res = sp.causal_rl_benchmark(
        name="confounded_bandit", n_episodes=200, seed=0,
    )
    assert isinstance(res, sp.BanditBenchmarkResult)
    assert len(res.transitions) == 200
    assert res.optimal_policy[0] == 1


def test_causal_rl_benchmark_unknown():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        sp.causal_rl_benchmark(name="invalid_name", n_episodes=10)


def test_causal_dqn():
    bench = sp.causal_rl_benchmark(
        name="confounded_routing", n_episodes=300, seed=42,
    )
    res = sp.causal_dqn(
        bench.transitions, state='state', action='action',
        reward='reward', next_state='next_state',
        gamma_bound=0.05, n_iter=200, lr=0.1,
    )
    assert isinstance(res, sp.CausalDQNResult)
    assert res.policy.shape == (3,)
    assert res.q_table.shape[1] == 2


def test_causal_dqn_invalid_gamma():
    df = pd.DataFrame({
        's': [0, 1], 'a': [0, 1], 'r': [1.0, 0.5], 'sp': [1, 0],
    })
    with pytest.raises(ValueError, match="gamma_bound"):
        sp.causal_dqn(df, state='s', action='a', reward='r',
                       next_state='sp', gamma_bound=1.5)


def test_offline_safe_policy():
    rng = np.random.default_rng(0)
    n = 200
    s = rng.integers(0, 3, size=n)
    a = rng.integers(0, 2, size=n)
    r = (a == 1).astype(float) + 0.1 * rng.standard_normal(n)
    c = (a == 1).astype(float) * 0.8 + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({'s': s, 'a': a, 'r': r, 'cost': c})
    res = sp.offline_safe_policy(
        df, state='s', action='a', reward='r', cost='cost',
        cost_threshold=0.5,
    )
    assert isinstance(res, sp.OfflineSafeResult)
    # With threshold=0.5 and a=1 cost ~0.8, policy should pick a=0 (cheaper)
    assert (res.policy == 0).all()
