"""Torch-free contract tests for neural causal export helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def _fake_neural_result():
    cate = np.array([0.5, 1.0, 1.5])
    return sp.CausalResult(
        method="DragonNet",
        estimand="ATE",
        estimate=float(cate.mean()),
        se=0.1,
        pvalue=0.01,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs=len(cate),
        model_info={
            "neural_causal": True,
            "architecture": "DragonNet",
            "n_epochs_trained": 2,
            "cate": cate,
            "mu0": np.array([1.0, 2.0, 3.0]),
            "mu1": np.array([1.5, 3.0, 4.5]),
            "treatment": np.array([0, 1, 1]),
            "propensity": np.array([0.2, 0.6, 0.8]),
            "loss_history": [
                {"epoch": 1, "loss": 2.0},
                {"epoch": 2, "loss": 1.0},
            ],
        },
    )


def test_neural_export_helpers_match_result_metadata_without_torch():
    result = _fake_neural_result()

    effects = sp.neural_effects_frame(result)
    summary = sp.neural_summary_frame(result)
    training = sp.neural_training_frame(result)

    np.testing.assert_allclose(effects["cate"].to_numpy(), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(summary.loc[0, "estimate"], 1.0)
    np.testing.assert_allclose(training["loss"].to_numpy(), [2.0, 1.0])
    assert isinstance(effects, pd.DataFrame)
