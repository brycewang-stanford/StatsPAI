import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _network_frame(n: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(136)
    x = rng.normal(size=n)
    treat = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 1.0 + 0.5 * treat + 0.4 * x + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"y": y, "treat": treat, "x": x})


def _ring_adjacency(n: int) -> np.ndarray:
    adjacency = np.eye(n)
    for i in range(n):
        adjacency[i, (i - 1) % n] = 1.0
        adjacency[i, (i + 1) % n] = 1.0
    return adjacency


def test_gnn_causal_accepts_scalar_covariate() -> None:
    df = _network_frame()
    res = sp.gnn_causal(
        df,
        y="y",
        treat="treat",
        covariates="x",
        adjacency=_ring_adjacency(len(df)),
        n_layers=1,
        n_trees=5,
        min_leaf=2,
        random_state=136,
    )

    assert res.n_obs == len(df)
    assert res.feature_map.shape == (len(df), 2)
    assert np.isfinite(res.ate)


def test_gnn_causal_rejects_adjacency_mismatch_with_taxonomy() -> None:
    df = _network_frame()
    with pytest.raises(MethodIncompatibility, match="adjacency"):
        sp.gnn_causal(
            df,
            y="y",
            treat="treat",
            covariates="x",
            adjacency=np.eye(len(df) - 1),
            n_trees=5,
            min_leaf=2,
        )


def test_gnn_causal_rejects_single_arm_with_taxonomy() -> None:
    df = _network_frame()
    df["treat"] = 1
    with pytest.raises(DataInsufficient, match="both treatment arms"):
        sp.gnn_causal(
            df,
            y="y",
            treat="treat",
            covariates="x",
            adjacency=_ring_adjacency(len(df)),
            n_trees=5,
            min_leaf=2,
        )


def test_gnn_causal_rejects_bad_options_with_taxonomy() -> None:
    df = _network_frame()
    adjacency = _ring_adjacency(len(df))
    with pytest.raises(MethodIncompatibility, match="n_layers"):
        sp.gnn_causal(
            df,
            y="y",
            treat="treat",
            covariates="x",
            adjacency=adjacency,
            n_layers=-1,
        )
    with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
        sp.gnn_causal(
            df,
            y="y",
            treat="treat",
            covariates="x",
            adjacency=adjacency,
            propensity_bounds=(0.9, 0.2),
        )


@pytest.mark.parametrize("codes", [(0.0, 0.5), (0.0, 1.5)])
def test_gnn_causal_refuses_treatment_not_coded_zero_one(codes) -> None:
    """``astype(int)`` turned 0.5 into 0 and 1.5 into 1 before the check."""
    df = _network_frame()
    df["treat"] = np.where(df["treat"] == 1, codes[1], codes[0])
    with pytest.raises(MethodIncompatibility, match="binary"):
        sp.gnn_causal(
            df,
            y="y",
            treat="treat",
            covariates="x",
            adjacency=_ring_adjacency(len(df)),
            n_trees=5,
            min_leaf=2,
        )
