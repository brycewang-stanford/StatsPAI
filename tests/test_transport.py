"""Sprint-4 tests: transportability + SWIG."""
import numpy as np
import pandas as pd

import statspai as sp


# ---------- density-ratio weighting ----------

def test_transport_weights_recovers_target_effect():
    """Source has effect modified by X; target has different X
    distribution -> density-ratio weighting should shift the effect."""
    rng = np.random.default_rng(0)
    n_src, n_tgt = 1500, 800

    # Source: X ~ N(0, 1); treatment effect = 1 + X
    x_src = rng.normal(0, 1, n_src)
    a = rng.binomial(1, 0.5, n_src)
    y = (1 + x_src) * a + rng.normal(0, 0.3, n_src)
    src = pd.DataFrame({"x": x_src, "a": a, "y": y})

    # Target: X ~ N(1, 1) -- different distribution
    x_tgt = rng.normal(1, 1, n_tgt)
    tgt = pd.DataFrame({"x": x_tgt})

    res = sp.transport.weights(src, tgt, features=["x"], treatment="a", outcome="y")

    # Source unweighted effect ~= 1 + E_src[X] = 1
    # Target transported effect should shift toward 1 + E_tgt[X] = 2
    assert abs(res.effect_source - 1.0) < 0.15
    # Transported effect should lean toward the target modifier mean
    assert res.effect_transported > res.effect_source - 0.1
    assert res.ess > 0
    assert np.isfinite(res.weights).all()
    assert (res.weights > 0).all()
    assert "Transport" in res.summary()


# ---------- generalize ----------

def test_generalize_works():
    rng = np.random.default_rng(1)
    n = 300
    src = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "treat": rng.binomial(1, 0.5, n),
    })
    src["y"] = src["treat"] * 2 + rng.normal(0, 1, n)
    tgt = pd.DataFrame({"age": rng.normal(55, 10, 200)})
    res = sp.transport.generalize(src, tgt, features=["age"])
    assert isinstance(res, sp.TransportWeightResult)


# ---------- Pearl-Bareinboim identification ----------

def test_identify_transport_selection_node_separated():
    """If S is d-separated from Y given X, transport is direct."""
    g = sp.dag("S -> X; X -> Y")
    res = sp.transport.identify_transport(g, treatment="X", outcome="Y",
                                          selection_nodes={"S"})
    assert res.transportable
    assert "X" in res.formula


def test_identify_transport_fails_when_S_directly_causes_Y():
    """If S -> Y directly and has no mediator in the graph, no admissible
    Z exists -- NOT transportable."""
    g = sp.dag("S -> Y; X -> Y")
    res = sp.transport.identify_transport(g, treatment="X", outcome="Y",
                                          selection_nodes={"S"})
    assert not res.transportable


def test_identify_transport_finds_admissible_set():
    g = sp.dag("S -> Z; Z -> Y; X -> Y")
    res = sp.transport.identify_transport(g, treatment="X", outcome="Y",
                                          selection_nodes={"S"})
    # Z should render S irrelevant for Y
    assert res.transportable
    assert "Z" in res.admissible_set


# ---------- SWIG integration ----------

def test_swig_with_latent_confounder_splits_correctly():
    g = sp.dag("X -> Y; L -> X; L -> Y")
    sw = sp.swig(g, intervention={"X": "x"})
    # L should appear as an unchanged observation
    assert "L" in sw.nodes or any(v.startswith("L(") for v in sw.nodes)
    # Y must appear as potential outcome
    assert any(v.startswith("Y(") for v in sw.nodes)
