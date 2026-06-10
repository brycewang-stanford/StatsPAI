"""Tests for the optional Synth R backend."""

import subprocess

import numpy as np
import pytest

import statspai as sp
from statspai.synth.scm import _find_rscript


def test_native_synth_exposes_identification_boundary():
    result = sp.synth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="native",
        placebo=False,
    )
    assert result.model_info["backend"] == "native"
    assert result.model_info["validation_tier"] == "identification_dependent_native"
    assert result.model_info["reference_backend"] == "Synth"
    assert "T4 non-uniqueness disclosures" in result.model_info["validation_note"]
    diag = result.model_info["solver_start_diagnostics"]
    assert list(diag["start"]) == ["equal"]
    assert result.model_info["solver_best_start"] == "equal"
    assert result.model_info["solver_near_best_start_count"] == 1
    assert result.model_info["solver_near_best_weight_class_count"] == 1
    assert result.model_info["weight_solution_nonunique"] is False


def test_nested_scm_exposes_multistart_weight_class_diagnostics():
    result = sp.synth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="native",
        special_predictors=[("gdppc", yr, "mean") for yr in range(1955, 1970)],
        v_method="nested",
        n_random_starts=0,
        placebo=False,
    )
    diag = result.model_info["solver_start_diagnostics"]
    weights = result.model_info["solver_start_weights"]
    v_weights = result.model_info["solver_start_v_weights"]

    assert list(diag["start"]) == ["equal", "regression"]
    assert result.model_info["solver_best_start"] == "regression"
    assert result.model_info["solver_near_best_start_count"] == 2
    assert result.model_info["solver_near_best_weight_class_count"] == 2
    assert result.model_info["solver_near_best_weight_l1_max"] > 0.004
    assert result.model_info["weight_solution_nonunique"] is True
    assert set(weights["start"]) == {"equal", "regression"}
    assert set(v_weights["start"]) == {"equal", "regression"}
    assert {"Asturias", "Cataluna", "Madrid"} <= set(
        weights.loc[weights["weight"] > 0.01, "unit"]
    )


def _skip_unless_synth_available():
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
                "!requireNamespace('Synth', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages Synth/jsonlite are not installed")


def test_synth_backend_matches_reference_fixture():
    _skip_unless_synth_available()
    result = sp.synth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="synth",
    )
    assert np.isclose(result.estimate, -0.687789209705457)
    assert np.isclose(result.model_info["pre_treatment_rmse"], 0.0793738251107324)
    weights = result.model_info["weights"].set_index("unit")["weight"]
    assert np.isclose(weights.loc["Madrid"], 0.537261315298536)
    assert np.isclose(weights.loc["Cataluna"], 0.451608208273253)
    assert result.model_info["backend"] == "synth"
    assert result.model_info["validation_tier"] == "reference_backend_bridge"
    assert (
        "not counted as native Python parity evidence"
        in result.model_info["validation_note"]
    )


def test_synth_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        sp.synth(
            sp.datasets.basque_terrorism(),
            outcome="gdppc",
            unit="region",
            time="year",
            treated_unit="Basque Country",
            treatment_time=1970,
            method="classic",
            backend="unknown",
        )
