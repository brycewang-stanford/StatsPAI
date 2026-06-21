"""Tests for the optional synthdid R backend."""

import subprocess

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.sdid import _find_rscript


def _exact_synthetic_panel(tau: float = 2.5):
    rows = []
    times = range(1, 9)
    offsets = [0.0, 1.0, -0.5, 2.0]
    slopes = [0.2, -0.1, 0.4, 0.0]

    def control_value(index: int, t: int) -> float:
        return 5.0 + offsets[index] + slopes[index] * t + 0.05 * t * t

    for index in range(4):
        for t in times:
            rows.append(
                {
                    "unit": f"c{index + 1}",
                    "time": t,
                    "y": control_value(index, t),
                }
            )
    for t in times:
        untreated = np.mean([control_value(index, t) for index in range(4)])
        rows.append(
            {
                "unit": "treated",
                "time": t,
                "y": untreated + 1.7 + (tau if t >= 6 else 0.0),
            }
        )
    return pd.DataFrame(rows)


def test_native_sdid_matches_synthdid_reference_fixture():
    result = sp.sdid(
        sp.datasets.california_prop99(),
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="native",
        seed=42,
    )
    assert result.model_info["backend"] == "native"
    assert result.model_info["validation_tier"] == "T2_native_reference_parity"
    assert result.model_info["reference_backend"] == "synthdid"
    assert "Frank-Wolfe weight solver" in result.model_info["validation_note"]
    assert np.isclose(result.estimate, -15.94838884672099)
    assert np.isclose(result.se, 2.6040746521236526)


def test_native_sdid_recovers_constant_effect_on_exact_synthetic_dgp():
    """Native SDID has a non-circular known-truth guard."""
    truth = 2.5
    result = sp.sdid(
        _exact_synthetic_panel(truth),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=6,
        backend="native",
        method="sdid",
    )
    assert result.model_info["backend"] == "native"
    assert result.model_info["validation_tier"] == "T2_native_reference_parity"
    assert abs(result.estimate - truth) < 1e-10
    weights = result.model_info["unit_weights"].set_index("unit")["weight"]
    assert np.allclose(weights.sort_index().to_numpy(), np.repeat(0.25, 4))


def _skip_unless_synthdid_available():
    try:
        rscript = _find_rscript()
    except RuntimeError:
        pytest.skip("Rscript is not installed")
    probe = subprocess.run(
        [
            rscript,
            "-e",
            (
                "quit(status = as.integer("
                "!requireNamespace('synthdid', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages synthdid/jsonlite are not installed")


def test_synthdid_backend_matches_reference_fixture():
    _skip_unless_synthdid_available()
    result = sp.sdid(
        sp.datasets.california_prop99(),
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="synthdid",
        seed=42,
    )
    assert np.isclose(result.estimate, -15.94838884672099)
    assert np.isclose(result.se, 2.6266066920828113)
    assert result.model_info["backend"] == "synthdid"
    assert result.model_info["validation_tier"] == "reference_backend_bridge"
    assert (
        "not counted as native Python parity evidence"
        in result.model_info["validation_note"]
    )
    assert result.model_info["n_control"] == 38
    assert result.model_info["T_pre"] == 19


def test_sdid_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        sp.sdid(
            sp.datasets.california_prop99(),
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            backend="unknown",
        )
