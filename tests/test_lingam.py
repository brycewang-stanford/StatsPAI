"""DirectLiNGAM tests — cross-validated against the ``lingam`` package."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.causal_discovery import lingam


@pytest.fixture(scope="module")
def chain_dgp():
    """x1 → x2 → x3 with non-Gaussian (uniform) disturbances."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(-1, 1, n)
    x2 = 0.8 * x1 + 0.3 * rng.uniform(-1, 1, n)
    x3 = -0.5 * x2 + 0.2 * x1 + 0.3 * rng.uniform(-1, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def test_recovers_causal_order(chain_dgp):
    res = lingam(chain_dgp)
    assert [res.names[i] for i in res.order] == ["x1", "x2", "x3"]


def test_matches_reference_lingam_package(chain_dgp):
    """Our B should match the published `lingam` package bit-for-bit."""
    try:
        import lingam as _lingam
    except ImportError:
        pytest.skip("lingam package not installed")
    ours = lingam(chain_dgp).adjacency
    ref = _lingam.DirectLiNGAM()
    ref.fit(chain_dgp.values)
    np.testing.assert_allclose(ours, ref.adjacency_matrix_, atol=1e-8)


def test_edges_detected(chain_dgp):
    res = lingam(chain_dgp)
    edges = {(src, dst) for src, dst, _ in res.edges(threshold=0.1)}
    assert ("x1", "x2") in edges
    assert ("x2", "x3") in edges
    assert ("x1", "x3") in edges          # direct path


def test_summary_prints(chain_dgp):
    res = lingam(chain_dgp)
    s = res.summary()
    assert "DirectLiNGAM" in s
    assert "x1" in s
    assert "x2" in s


def test_rejects_no_relationship():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "a": rng.uniform(size=n),
        "b": rng.uniform(size=n),
    })
    res = lingam(df)
    # Any detected edge should have tiny magnitude
    for _, _, w in res.edges(threshold=0.0):
        assert abs(w) < 0.15
