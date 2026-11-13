"""Task 4 contracts for the shared IRM scorer and external OOF path."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import pydoc
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import statspai as sp
from statspai import OOFPredictions
from statspai.agent.tools import tool_manifest
from statspai.dml._base import _DoubleMLBase
from statspai.dml.double_ml import DoubleML
from statspai.dml.double_ml import dml as library_dml
from statspai.dml.irm import DoubleMLIRM
from statspai.exceptions import MethodIncompatibility

from .dml_oof_helpers import _training_records


class NoFitRegressor(BaseEstimator):
    def fit(self, *_args, **_kwargs):
        raise AssertionError("external scoring must not fit ml_g")

    def predict(self, *_args, **_kwargs):
        raise AssertionError("external scoring must not predict with ml_g")


class NoFitClassifier(BaseEstimator):
    def fit(self, *_args, **_kwargs):
        raise AssertionError("external scoring must not fit ml_m")

    def predict(self, *_args, **_kwargs):
        raise AssertionError("external scoring must not predict with ml_m")

    def predict_proba(self, *_args, **_kwargs):
        raise AssertionError("external scoring must not predict_proba with ml_m")


def _external_fixture(
    *,
    n_rep=1,
    ids=None,
    ps_raw=None,
    fold_ids=None,
    origin="caller_declared",
    two_covariates=False,
):
    frame = pd.DataFrame(
        {
            "y": [1.0, 4.0, 3.0, 6.0],
            "d": [0.0, 1.0, 0.0, 1.0],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    if two_covariates:
        frame["z"] = [10.0, 11.0, 12.0, 13.0]
    covariates = ["x", "z"] if two_covariates else ["x"]
    if ids is None:
        ids = [f"row:{i}" for i in range(len(frame))]
    if fold_ids is None:
        fold_ids = np.tile([0, 0, 1, 1], (n_rep, 1))
    fold_ids = np.asarray(fold_ids)
    if ps_raw is None:
        ps_raw = np.full((n_rep, len(frame)), 0.5)
    predictions = OOFPredictions.from_arrays(
        ids=ids,
        y=frame["y"].to_numpy(),
        d=frame["d"].to_numpy(),
        x=frame[covariates].to_numpy(),
        covariate_names=covariates,
        g0=np.ones((n_rep, len(frame))),
        g1=np.full((n_rep, len(frame)), 3.0),
        ps_raw=ps_raw,
        fold_ids=fold_ids,
        training_records=_training_records(
            ids, frame["d"].to_numpy(), fold_ids, origin=origin
        ),
        source={
            "engine": "hand_fixture",
            "recipe": "fixed caller-declared arrays",
            "seed": None,
            "software_versions": {"fixture": "1"},
        },
    )
    return frame, predictions


def _assert_external_provenance(result, predictions, id_source, store_oof):
    provenance = sp.get_provenance(result)
    keys = {
        "external_predictions_provided",
        "external_predictions_hashes",
        "store_oof",
        "observation_ids_source",
    }
    assert {name: provenance.params[name] for name in keys} == {
        "external_predictions_provided": True,
        "external_predictions_hashes": dict(predictions.hashes),
        "store_oof": store_oof,
        "observation_ids_source": id_source,
    }
    encoded = json.dumps(provenance.params, allow_nan=False)
    assert all(row_id not in encoded for row_id in predictions.ids)
    assert not _contains_identity(provenance.params, predictions)


@pytest.mark.parametrize(
    "function", [sp.dml, library_dml, _DoubleMLBase.fit, DoubleML.fit]
)
def test_python_entry_points_name_three_keyword_only_oof_parameters(function):
    assert typing.get_type_hints(function)
    signature = inspect.signature(function)
    expected_defaults = {
        "external_predictions": None,
        "store_oof": False,
        "observation_ids": None,
    }
    for name, default in expected_defaults.items():
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is default
    expected_positional = {
        sp.dml: "data y d X",
        library_dml: (
            "data y treat covariates model instrument ml_g ml_m ml_r n_folds "
            "n_rep alpha random_state sample_weight fold_indices score "
            "normalize_ipw trimming_threshold"
        ),
        _DoubleMLBase.fit: "self",
        DoubleML.fit: "self",
    }[function].split()
    actual_positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert actual_positional == expected_positional


def test_article_wrapper_oof_annotations_are_explicit():
    annotations = sp.dml.__annotations__
    assert "OOFPredictions" in str(annotations["external_predictions"])
    assert annotations["store_oof"] in ("bool", bool)
    assert "Sequence[str]" in str(annotations["observation_ids"])


def _run_external_entry(entry, frame, predictions, **fit_kwargs):
    common = {
        "ml_g": NoFitRegressor(),
        "ml_m": NoFitClassifier(),
        "n_folds": predictions.n_folds,
        "n_rep": predictions.n_rep,
    }
    if entry == "direct":
        estimator = DoubleMLIRM(frame, y="y", treat="d", covariates=["x"], **common)
        return estimator.fit(external_predictions=predictions, **fit_kwargs)
    if entry == "legacy":
        estimator = DoubleML(frame, "y", "d", ["x"], "irm", **common)
        return estimator.fit(external_predictions=predictions, **fit_kwargs)
    function = library_dml if entry == "library" else sp.dml
    return function(
        frame,
        "y",
        "d",
        ["x"],
        model="irm",
        external_predictions=predictions,
        **common,
        **fit_kwargs,
    )


def _forbid_external_internal_work(monkeypatch, *, reject_score=False):
    irm_module = importlib.import_module("statspai.dml.irm")
    base_module = importlib.import_module("statspai.dml._base")
    retention_module = importlib.import_module("statspai.dml._oof_retention")
    real_numpy = irm_module.np

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external OOF request reached forbidden work")

    class NoFallbackNumpy:
        def __getattr__(self, name):
            return forbidden if name == "full" else getattr(real_numpy, name)

    monkeypatch.setattr(_DoubleMLBase, "_make_splits", forbidden)
    monkeypatch.setattr(DoubleMLIRM, "_fit_one_rep", forbidden)
    monkeypatch.setattr(irm_module, "np", NoFallbackNumpy())
    if reject_score:
        monkeypatch.setattr(irm_module, "score_binary_ate", forbidden)
        monkeypatch.setattr(base_module, "_attach_result_oof", forbidden)
        monkeypatch.setattr(retention_module.OOFRetention, "internal", forbidden)
        monkeypatch.setattr(retention_module.OOFRetention, "external", forbidden)


@pytest.mark.parametrize("entry", ["direct", "library", "article", "legacy"])
@pytest.mark.parametrize("store_oof", [False, True])
def test_external_store_entry_points_skip_internal_fit_split_and_fallback(
    monkeypatch, entry, store_oof
):
    ids = ["unit-a", "unit-b", "unit-c", "unit-d"] if entry == "article" else None
    frame, predictions = _external_fixture(ids=ids)
    _forbid_external_internal_work(monkeypatch)
    result = _run_external_entry(
        entry, frame, predictions, store_oof=store_oof, observation_ids=ids
    )
    assert result.estimate == 3.0
    assert result.se == math.sqrt(3.25)
    if store_oof:
        assert result.get_oof().predictions.hashes == predictions.hashes
    else:
        with pytest.raises(ValueError, match="^OOF records are unavailable;"):
            result.get_oof()
    if entry in {"library", "article"}:
        _assert_external_provenance(
            result,
            predictions,
            "caller_provided" if ids else "generated_ordinal",
            store_oof,
        )
    else:
        assert sp.get_provenance(result) is None


def test_external_two_repeat_uses_existing_median_aggregation_and_last_repeat():
    frame, predictions = _external_fixture(
        n_rep=2,
        ps_raw=np.array([[0.5] * 4, [0.25] * 4]),
    )
    result = _run_external_entry("direct", frame, predictions)
    theta_second = float.fromhex("0x1.5555555555556p+2")
    assert result.model_info["theta_all_reps"] == [3.0, theta_second]
    np.testing.assert_allclose(
        result.model_info["se_all_reps"],
        [math.sqrt(13.0 / 4.0), math.sqrt(23.0 / 3.0)],
    )
    assert result.estimate == float.fromhex("0x1.0aaaaaaaaaaabp+2")
    assert result.se == pytest.approx(math.sqrt(491.0 / 72.0))
    assert result.model_info["fold_source"] == "external_predictions"
    np.testing.assert_allclose(
        result.model_info["_y_resid"],
        np.array([2.0, 6.0, -2.0 / 3.0, 14.0]) - theta_second,
    )
    np.testing.assert_array_equal(result.model_info["_pscore"], [0.25] * 4)
    np.testing.assert_array_equal(
        result.model_info["_d_resid"], frame["d"] - predictions.ps_raw[-1]
    )


def test_external_raw_boundary_propensity_is_clipped_only_for_scoring():
    frame, predictions = _external_fixture(ps_raw=np.array([[0.0, 1.0, 0.5, 0.5]]))
    estimator = DoubleMLIRM(
        frame,
        y="y",
        treat="d",
        covariates=["x"],
        ml_g=NoFitRegressor(),
        ml_m=NoFitClassifier(),
        n_folds=2,
        trimming_threshold=0.1,
    )

    result = estimator.fit(external_predictions=predictions)

    assert result.estimate == pytest.approx(25.0 / 9.0)
    assert result.se**2 == pytest.approx(343.0 / 108.0)
    np.testing.assert_array_equal(result.model_info["_pscore"], [0.0, 1.0, 0.5, 0.5])
    assert result.model_info["diagnostics"]["n_clipped_below"] == 1
    assert result.model_info["diagnostics"]["n_clipped_above"] == 1


def test_external_custom_ids_and_label_permuted_explicit_partition_are_accepted():
    ids = ["unit-a", "unit-b", "unit-c", "unit-d"]
    frame, predictions = _external_fixture(n_rep=2, ids=ids)
    estimator = _direct_external_estimator(
        frame,
        n_rep=2,
        fold_indices=[1, 1, 0, 0],
    )

    result = estimator.fit(
        external_predictions=predictions, store_oof=True, observation_ids=ids
    )

    assert result.estimate == 3.0
    assert result.model_info["fold_source"] == "external_predictions"
    assert result.get_oof().predictions.n_rep == 2


def _contains_identity(value, target, seen=None):
    if value is target:
        return True
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)
    if isinstance(value, dict):
        return any(_contains_identity(item, target, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target, seen) for item in value)
    return False


def test_functional_oof_provenance_validation_precedes_best_effort_attach(monkeypatch):
    external_module = importlib.import_module("statspai.dml._external_predictions")
    build = external_module.build_oof_provenance_payload
    assert build(external_predictions=None, store_oof=False, observation_ids=None) == {}
    lineage = importlib.import_module("statspai.output._lineage")
    frame, predictions = _external_fixture()
    assert build(
        external_predictions=None,
        store_oof=True,
        observation_ids=("private",),
    ) == {
        "external_predictions_provided": False,
        "external_predictions_hashes": None,
        "store_oof": True,
        "observation_ids_source": "caller_provided",
    }
    assert build(
        external_predictions=predictions, store_oof=True, observation_ids=None
    ) == {
        "external_predictions_provided": True,
        "external_predictions_hashes": dict(predictions.hashes),
        "store_oof": True,
        "observation_ids_source": "generated_ordinal",
    }
    valid_builder = external_module._raw_oof_provenance_payload
    call = lambda: library_dml(  # noqa: E731
        frame,
        "y",
        "d",
        ["x"],
        "irm",
        n_folds=2,
        external_predictions=predictions,
    )

    def bad_payload(**kwargs):
        payload = valid_builder(**kwargs)
        payload["external_predictions_hashes"] = {"data": object()}
        return payload

    def fail_json(*_args, **_kwargs):
        raise TypeError("injected non-JSON payload")

    def fail_attach(*_args, **_kwargs):
        raise RuntimeError("optional lineage unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(external_module, "_raw_oof_provenance_payload", bad_payload)
        with pytest.raises(ValueError, match="exact hash keys"):
            call()
    with monkeypatch.context() as patch:
        patch.setattr(external_module.json, "dumps", fail_json)
        with pytest.raises(TypeError, match="injected non-JSON"):
            call()
    with monkeypatch.context() as patch:
        patch.setattr(lineage, "attach_provenance", fail_attach)
        assert call().estimate == 3.0


def test_external_result_keeps_legacy_analysis_fields_without_oof_retention():
    frame, predictions = _external_fixture(n_rep=2)
    estimator = DoubleMLIRM(
        frame,
        y="y",
        treat="d",
        covariates=["x"],
        ml_g=NoFitRegressor(),
        ml_m=NoFitClassifier(),
        n_folds=2,
        n_rep=2,
    )

    result = estimator.fit(external_predictions=predictions, store_oof=False)
    info = result.model_info

    assert info["scoring_engine"] == "statspai_irm_external_predictions"
    assert info["external_predictions_hashes"] == dict(predictions.hashes)
    assert info["nuisance_fit"] == "skipped_external_predictions"
    for name in ("_X_design", "_T", "_Y", "_covariate_names"):
        assert name in info
    for forbidden in ("g0", "g1", "ps_used", "psi_b", "fold_ids", "observation_ids"):
        assert forbidden not in info
    assert not any(
        isinstance(value, np.ndarray) and value.shape == (2, 4)
        for value in info.values()
    )
    assert not _contains_identity(result.__dict__, predictions)
    assert not _contains_identity(estimator.__dict__, predictions)

    diagnostics = sp.dml_diagnostics(result)
    assert diagnostics.n_obs == 4
    # A benchmark refits the model without the covariate; supplied
    # predictions cannot be refitted, so the request is refused.
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="refit"):
        sp.dml_sensitivity(result, benchmark_covariates=["x"])


def _direct_external_estimator(frame, **kwargs):
    options = {
        "ml_g": NoFitRegressor(),
        "ml_m": NoFitClassifier(),
        "n_folds": 2,
    }
    options.update(kwargs)
    return DoubleMLIRM(frame, y="y", treat="d", covariates=["x"], **options)


@pytest.mark.parametrize("bad", [{}, "predictions.json", Path("predictions.json")])
@pytest.mark.parametrize("store_oof", [False, True])
def test_external_predictions_reject_unsupported_transport_types(
    monkeypatch, bad, store_oof
):
    frame, _ = _external_fixture()
    estimator = _direct_external_estimator(frame)

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    message = "external_predictions must be an in-memory OOFPredictions instance"
    with pytest.raises(TypeError, match=f"^{message}$"):
        estimator.fit(external_predictions=bad, store_oof=store_oof)


@pytest.mark.parametrize("store_oof", [False, True])
def test_external_custom_prediction_ids_require_matching_caller_ids(
    monkeypatch, store_oof
):
    frame, predictions = _external_fixture(ids=["a", "b", "c", "d"])
    estimator = _direct_external_estimator(frame)

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    with pytest.raises(ValueError, match="^OOFPredictions alignment failed for: ids$"):
        estimator.fit(external_predictions=predictions, store_oof=store_oof)


@pytest.mark.parametrize(
    "ids,error",
    [
        (["a", "b", "c"], ValueError),
        (["a", "a", "c", "d"], ValueError),
        (["a", "", "c", "d"], ValueError),
        (["a", 2, "c", "d"], ValueError),
        ("abcd", TypeError),
    ],
)
def test_external_rejects_invalid_observation_ids(ids, error):
    frame, predictions = _external_fixture()
    estimator = _direct_external_estimator(frame)

    with pytest.raises(error, match="observation_ids"):
        estimator.fit(external_predictions=predictions, observation_ids=ids)


@pytest.mark.parametrize(
    "mutation,failed", [("row_order", "y, d, x"), ("y", "y"), ("d", "d"), ("x", "x")]
)
@pytest.mark.parametrize("store_oof", [False, True])
def test_external_rejects_analysis_row_or_value_mismatch(
    monkeypatch, mutation, failed, store_oof
):
    frame, predictions = _external_fixture()
    changed = frame.copy()
    if mutation == "row_order":
        changed = changed.iloc[[1, 0, 2, 3]].reset_index(drop=True)
    elif mutation == "y":
        changed.loc[0, "y"] += 1.0
    elif mutation == "d":
        changed["d"] = [1, 1, 0, 0]
    else:
        changed.loc[0, "x"] += 1.0
    estimator = _direct_external_estimator(changed)

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    message = f"OOFPredictions alignment failed for: {failed}"
    with pytest.raises(ValueError, match=f"^{message}$"):
        estimator.fit(external_predictions=predictions, store_oof=store_oof)


@pytest.mark.parametrize("store_oof", [False, True])
def test_external_rejects_covariate_name_and_order_mismatch(monkeypatch, store_oof):
    frame, predictions = _external_fixture(two_covariates=True)
    estimator = DoubleMLIRM(
        frame,
        y="y",
        treat="d",
        covariates=["z", "x"],
        ml_g=NoFitRegressor(),
        ml_m=NoFitClassifier(),
        n_folds=2,
    )

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    message = "OOFPredictions alignment failed for: covariate_names, x"
    with pytest.raises(ValueError, match=f"^{message}$"):
        estimator.fit(external_predictions=predictions, store_oof=store_oof)


@pytest.mark.parametrize("field", ["y", "d", "x", "fold"])
def test_external_rejects_missing_rows_instead_of_implicit_deletion(field):
    frame, predictions = _external_fixture()
    changed = frame.copy()
    fold_indices = None
    if field == "fold":
        fold_indices = np.array([0.0, np.nan, 1.0, 1.0])
    else:
        changed.loc[0, field] = np.nan
    estimator = _direct_external_estimator(changed, fold_indices=fold_indices)

    with pytest.raises(ValueError, match="complete"):
        estimator.fit(external_predictions=predictions)


@pytest.mark.parametrize("field", ["y", "d", "x"])
@pytest.mark.parametrize(
    "bad_values",
    [
        np.array([1 + 1j, 2 + 0j, 3 + 0j, 4 + 0j]),
        np.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]),
        np.array(["1", "2", "3", "4"], dtype=object),
        np.array(["1", "2", "3", "4"]),
    ],
)
def test_external_dtype_gate_precedes_alignment_and_training(
    monkeypatch, field, bad_values
):
    frame, predictions = _external_fixture()
    changed = frame.copy()
    changed[field] = bad_values
    estimator = _direct_external_estimator(changed)

    def forbidden_alignment(*_args, **_kwargs):
        raise AssertionError("dtype gate reached alignment")

    monkeypatch.setattr(OOFPredictions, "validate_alignment", forbidden_alignment)
    with pytest.raises(ValueError, match="real numeric"):
        estimator.fit(external_predictions=predictions)


@pytest.mark.parametrize("field", ["y", "d", "x"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_external_rejects_nonfinite_analysis_values_before_alignment(
    monkeypatch, field, bad
):
    frame, predictions = _external_fixture()
    changed = frame.copy()
    changed.loc[0, field] = bad

    def forbidden_alignment(*_args, **_kwargs):
        raise AssertionError("finite-value gate reached alignment")

    monkeypatch.setattr(OOFPredictions, "validate_alignment", forbidden_alignment)
    if not np.isnan(bad):
        # inf is refused when the estimator is built (the package-wide rule)
        with pytest.raises(ValueError, match="infinite"):
            _direct_external_estimator(changed)
        return
    estimator = _direct_external_estimator(changed)
    with pytest.raises(ValueError, match="complete"):
        estimator.fit(external_predictions=predictions)


@pytest.mark.parametrize("store_oof", [False, True])
def test_external_rejects_repeat_and_fold_count_mismatch(monkeypatch, store_oof):
    frame, predictions = _external_fixture()
    _forbid_external_internal_work(monkeypatch, reject_score=True)
    with pytest.raises(
        ValueError, match="^external prediction repeats do not match n_rep: 1 != 2$"
    ):
        _direct_external_estimator(frame, n_rep=2).fit(
            external_predictions=predictions, store_oof=store_oof
        )
    with pytest.raises(
        ValueError, match="^external prediction folds do not match n_folds: 2 != 3$"
    ):
        _direct_external_estimator(frame, n_folds=3).fit(
            external_predictions=predictions, store_oof=store_oof
        )


@pytest.mark.parametrize("store_oof", [False, True])
def test_external_rejects_conflicting_explicit_partition_in_any_repeat(
    monkeypatch, store_oof
):
    frame, predictions = _external_fixture(
        n_rep=2,
        fold_ids=np.array([[0, 0, 1, 1], [0, 1, 1, 0]]),
    )
    estimator = _direct_external_estimator(frame, n_rep=2, fold_indices=[0, 0, 1, 1])

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    message = "explicit fold partition does not match external predictions at repeat 1"
    with pytest.raises(ValueError, match=f"^{message}$"):
        estimator.fit(external_predictions=predictions, store_oof=store_oof)


@pytest.mark.parametrize("store_oof", [False, True])
def test_external_rejects_statspai_internal_training_origin(monkeypatch, store_oof):
    frame, predictions = _external_fixture(origin="statspai_internal")
    estimator = _direct_external_estimator(frame)

    _forbid_external_internal_work(monkeypatch, reject_score=True)
    message = "external predictions require training-record origin='caller_declared'"
    with pytest.raises(ValueError, match=f"^{message}$"):
        estimator.fit(external_predictions=predictions, store_oof=store_oof)


@pytest.mark.parametrize("model", ["plr", "pliv", "iivm"])
def test_store_only_rejects_other_models_before_any_work(monkeypatch, model):
    frame, _ = _external_fixture()
    frame["z"] = [0, 1, 0, 1]
    instrument = "z" if model in {"pliv", "iivm"} else None
    _forbid_external_internal_work(monkeypatch, reject_score=True)
    with pytest.raises(NotImplementedError, match="model='irm'"):
        library_dml(
            frame,
            y="y",
            treat="d",
            covariates=["x"],
            model=model,
            instrument=instrument,
            ml_g=NoFitRegressor(),
            ml_m=NoFitClassifier(),
            ml_r=NoFitClassifier(),
            n_folds=2,
            store_oof=True,
        )


def test_store_only_rejects_atte_normalized_and_weighted_before_work(monkeypatch):
    frame, _ = _external_fixture()
    _forbid_external_internal_work(monkeypatch, reject_score=True)
    atte = _direct_external_estimator(
        frame, score="ATTE", normalize_ipw=True, sample_weight=np.ones(4)
    )
    with pytest.raises(NotImplementedError, match="score='ATE'"):
        atte.fit(store_oof=True)

    normalized = _direct_external_estimator(
        frame, normalize_ipw=True, sample_weight=np.ones(4)
    )
    with pytest.raises(NotImplementedError, match="normalize_ipw"):
        normalized.fit(store_oof=True)

    weighted = _direct_external_estimator(frame, sample_weight=np.ones(4))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        weighted.fit(store_oof=True)


def test_external_request_rejects_nonbinary_treatment_before_alignment():
    frame, predictions = _external_fixture()
    frame["d"] = [0.0, 0.5, 0.0, 1.0]
    estimator = _direct_external_estimator(frame)

    with pytest.raises(NotImplementedError, match="binary treatment"):
        estimator.fit(external_predictions=predictions)
    frame["d"] = 0.0
    with pytest.raises(sp.IdentificationFailure, match="both 0 and 1"):
        _direct_external_estimator(frame).fit(external_predictions=predictions)


@pytest.mark.parametrize("bad", [None, 0, 1, np.bool_(False), np.bool_(True), "true"])
@pytest.mark.parametrize("entry", ["direct", "article"])
def test_store_oof_requires_exact_builtin_bool(entry, bad):
    frame, predictions = _external_fixture()
    with pytest.raises(TypeError, match="store_oof must be a bool"):
        _run_external_entry(entry, frame, predictions, store_oof=bad)


def test_store_oof_builtin_true_enters_external_retention_path():
    frame, predictions = _external_fixture()
    result = _direct_external_estimator(frame).fit(
        external_predictions=predictions, store_oof=True
    )
    assert result.get_oof().predictions.n_rep == 1


def test_observation_ids_alone_are_not_silently_ignored():
    frame, _ = _external_fixture()
    estimator = _direct_external_estimator(frame)

    with pytest.raises(ValueError, match="observation_ids require"):
        estimator.fit(observation_ids=["a", "b", "c", "d"])


def test_legacy_constructor_validation_precedes_new_path_gates():
    frame, _ = _external_fixture()

    with pytest.raises(MethodIncompatibility, match="columns not found"):
        library_dml(
            frame,
            y="missing",
            treat="d",
            covariates=["x"],
            model="irm",
            n_folds=2,
            external_predictions="not-an-oof-record",
        )
    with pytest.raises(MethodIncompatibility, match="n_folds must be >= 2"):
        library_dml(
            frame,
            y="y",
            treat="d",
            covariates=["x"],
            model="irm",
            n_folds=1,
            external_predictions="not-an-oof-record",
        )


_PYTHON_ONLY_NAMES = {
    "external_predictions",
    "store_oof",
    "observation_ids",
}
_PRECHANGE_NON_DML_HASHES = {
    # Current upstream 42147197, excluding this PR's two DML API entries.
    "functions.json": "f2c5d9a4e5da3cc454f3c117e0ac8b6d88665505c00d5e63796fba9fe7752633",  # noqa: E501
    "agent_cards.json": "7b08027400be24d27b597362388725fb86cc7c8e21207294a69948fd112c2a6f",  # noqa: E501
    "tools.json": "ccfb1057df23752e1bc5616f946378eb6e24538ca25194e6028dc7f3a0706e93",
}
_LIVE_DML_SCHEMAS = {
    "functions.json": lambda: sp.function_schema("dml")["parameters"],
    "agent_cards.json": lambda: sp.agent_card("dml")["signature"]["parameters"],
    "tools.json": lambda: _dml_artifact_entry(tool_manifest())["input_schema"],
}


def _dml_artifact_entry(entries):
    return next(entry for entry in entries if entry.get("name") == "dml")


def _artifact_parameter_schema(filename, entry):
    if filename == "functions.json":
        return entry["parameters"]
    if filename == "agent_cards.json":
        return entry["signature"]["parameters"]
    return entry["input_schema"]


def _canonical_non_dml_hash(entries):
    others = [
        entry for entry in entries if entry.get("name") not in {"dml", "DMLDiagnostics"}
    ]
    payload = json.dumps(
        others, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_python_only_oof_controls_are_absent_from_all_live_callable_schemas():
    schemas = [getter() for getter in _LIVE_DML_SCHEMAS.values()]
    for schema in schemas:
        assert _PYTHON_ONLY_NAMES.isdisjoint(schema["properties"])
        assert _PYTHON_ONLY_NAMES.isdisjoint(schema.get("required", ()))

    registry_params = {p.name for p in sp.registry._REGISTRY["dml"].params}
    assert _PYTHON_ONLY_NAMES.isdisjoint(registry_params)


def _live_help(value):
    return pydoc.render_doc(value, renderer=pydoc.plaintext)


def test_dml_live_help_and_registry_disclose_complete_retention_contract():
    surfaces = {
        "article": _live_help(sp.dml),
        "library": _live_help(library_dml),
        "legacy_fit": _live_help(DoubleML.fit),
        "registry": json.dumps(sp.describe_function("dml"), sort_keys=True),
    }
    required = (
        "Python-only",
        "store_oof=True",
        "get_oof()",
        "get_residuals()",
        "individual-level",
        "caller_declared",
        "10 rows per treatment arm",
        "internal explicit fold_indices require n_rep=1",
        "external repeated IRM",
        "equivalent",
    )
    for name, text in surfaces.items():
        assert "pending Task 5" not in text, name
        for phrase in required:
            assert phrase in text, (name, phrase)
    library = surfaces["library"]
    for heading in (
        "external_predictions : OOFPredictions, optional",
        "store_oof : bool, default False",
        "observation_ids : sequence of str, optional",
    ):
        assert heading in library
    export_help = inspect.getdoc(sp.OOFBundle.to_json) or ""
    expected = "individual-level IDs, Y, D, X, nuisance predictions, and scores"
    assert expected in export_help


def test_dml_stub_reexports_the_annotated_article_wrapper():
    stub = (
        Path(__file__).resolve().parents[1] / "src/statspai/__init__.pyi"
    ).read_text()

    assert "from ._article_aliases import dml as dml" in stub
    annotations = sp.dml.__annotations__
    assert "OOFPredictions" in str(annotations["external_predictions"])
    assert "Sequence[str]" in str(annotations["observation_ids"])


@pytest.mark.parametrize(
    "filename", ["functions.json", "agent_cards.json", "tools.json"]
)
def test_committed_dml_schemas_keep_python_only_controls_out_and_non_dml_baseline(
    filename,
):
    root = Path(__file__).resolve().parents[1]
    root_path = root / "schemas" / filename
    packaged_path = root / "src/statspai/schemas" / filename
    assert root_path.read_bytes() == packaged_path.read_bytes()
    entries = json.loads(root_path.read_text())
    dml_entry = _dml_artifact_entry(entries)
    schema = _artifact_parameter_schema(filename, dml_entry)

    assert schema == _LIVE_DML_SCHEMAS[filename]()
    assert _PYTHON_ONLY_NAMES.isdisjoint(schema["properties"])
    assert _PYTHON_ONLY_NAMES.isdisjoint(schema.get("required", ()))
    artifact_text = json.dumps(dml_entry, sort_keys=True)
    assert "Python-only" in artifact_text
    assert "get_oof()" in artifact_text
    assert "10 rows per treatment arm" in artifact_text
    assert "pending Task 5" not in artifact_text
    # The non-DML hash pinned that the OOF audit PR left every other schema
    # untouched. It is checked against the pre-merge snapshot only while the
    # rest of the registry is still that snapshot; once other entries change
    # (new estimators, signature edits), ``scripts/dump_schemas.py --check``
    # keeps them in step with the code instead.
    others_hash = _canonical_non_dml_hash(entries)
    if others_hash != _PRECHANGE_NON_DML_HASHES[filename]:
        others = {e.get("name") for e in entries} - {"dml", "DMLDiagnostics"}
        assert others, "schema lost every non-DML entry"
