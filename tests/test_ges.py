"""GES tests."""
import numpy as np, pandas as pd, pytest
from statspai.causal_discovery.ges import ges


def test_ges_recovers_chain_skeleton():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.5
    x3 = 0.4 * x2 + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    res = ges(df)
    edges = {(s, d) for s, d, _ in res.edges()}
    assert ("x1", "x2") in edges or ("x2", "x1") in edges
    assert ("x2", "x3") in edges or ("x3", "x2") in edges
    assert ("x1", "x3") not in edges and ("x3", "x1") not in edges


def test_ges_independent_vars_no_edges():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.standard_normal(300),
                       "b": rng.standard_normal(300)})
    res = ges(df)
    assert len(res.edges()) == 0


def test_ges_summary():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
    assert "GES" in ges(df).summary()


def test_exported():
    import statspai as sp
    assert callable(sp.ges)
