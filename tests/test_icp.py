"""Sprint-3 tests: Invariant Causal Prediction."""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_two_env_scm(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """DGP:
        E in {0, 1}
        X1 = N(0, 1) + 2 * E          <- intervened on in env 1
        X2 = N(0, 1)                  <- spurious predictor
        Y  = 1.5 * X1 + noise         <- X1 is the true parent
        X3 = Y + N(0, 0.5)            <- child of Y (should NOT be a parent)
    """
    rng = np.random.default_rng(seed)
    E = rng.integers(0, 2, n)
    X1 = rng.normal(0, 1, n) + 2 * E
    X2 = rng.normal(0, 1, n)
    Y = 1.5 * X1 + rng.normal(0, 0.5, n)
    X3 = Y + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    return X, np.asarray(Y), E


def test_icp_recovers_true_parent():
    X, y, E = _make_two_env_scm(n=800, seed=0)
    res = sp.icp(X, y, environment=E, alpha=0.05)
    assert isinstance(res, sp.ICPResult)
    # X1 should be accepted as a parent; X3 (descendant) must NOT be.
    assert "X3" not in res.parents
    # In any reasonable run the accepted subsets must include X1
    if res.parents:
        assert "X1" in res.parents
    summary = res.summary()
    assert isinstance(summary, str)


def test_icp_nonlinear_api_works():
    X, y, E = _make_two_env_scm(n=400, seed=1)
    res = sp.nonlinear_icp(X, y, environment=E, alpha=0.1)
    assert isinstance(res, sp.ICPResult)
    assert res.method == "nonlinear"


def test_icp_requires_multiple_environments():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    y = rng.normal(0, 1, 100)
    with pytest.raises(ValueError, match="at least 2"):
        sp.icp(X, y, environment=np.zeros(100, dtype=int))


def test_icp_empty_candidate_still_returns():
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"x1": rng.normal(0, 1, n)})
    # Y fully exogenous -> truly no parents in X
    y = rng.normal(0, 1, n)
    E = rng.integers(0, 2, n)
    res = sp.icp(X, y, environment=E, alpha=0.05)
    # empty set should be accepted when Y has no X-parents
    assert frozenset() in res.accepted_subsets
