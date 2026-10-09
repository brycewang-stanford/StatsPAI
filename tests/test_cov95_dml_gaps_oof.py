"""Coverage tests for the DML out-of-fold (OOF) audit helpers.

Targets validation branches in ``statspai.dml._oof_score``,
``_score_concentration``, ``_oof_validation``, ``_external_predictions``
and ``_oof_retention`` that the main OOF contract tests do not reach.
Every branch here is a loud rejection of a malformed audit payload, so
each test pins the exception type and the message that names the field.
"""

import copy

import numpy as np
import pytest

from statspai import OOFBundle, OOFPredictions
from statspai.dml import _external_predictions as _ext
from statspai.dml import _oof_retention as _ret
from statspai.dml import _oof_validation as _val
from statspai.dml._oof_score import binary_ate_pseudo_outcome
from statspai.dml._score_concentration import score_concentration
from statspai.exceptions import MethodIncompatibility

from .dml_oof_helpers import bundle_metadata, tiny_bundle, tiny_inputs

# ---------------------------------------------------------------------
# _oof_score.binary_ate_pseudo_outcome
# ---------------------------------------------------------------------

_Y = np.array([1.0, 4.0, 3.0, 6.0])
_D = np.array([0.0, 1.0, 0.0, 1.0])
_G0 = np.ones(4)
_G1 = np.full(4, 3.0)
_PS = np.full(4, 0.5)


def test_pseudo_outcome_matches_the_aipw_formula():
    out = binary_ate_pseudo_outcome(_Y, _D, _G0, _G1, _PS)
    expected = _G1 - _G0 + _D * (_Y - _G1) / _PS - (1 - _D) * (_Y - _G0) / (1 - _PS)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(out, [2.0, 4.0, -2.0, 8.0], rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "y, exc, fragment",
    [
        (np.zeros((2, 2, 1)), ValueError, "y must be one- or two-dimensional"),
        (_Y.astype(complex), ValueError, "y must not contain complex values"),
        (np.array(["a", "b", "c", "d"]), TypeError, "y must be numeric"),
        (np.array([1.0, np.nan, 3.0, 6.0]), ValueError, "only finite values"),
        (np.ones(3), ValueError, "identical shapes"),
    ],
)
def test_pseudo_outcome_rejects_malformed_outcome(y, exc, fragment):
    with pytest.raises(exc, match=fragment):
        binary_ate_pseudo_outcome(y, _D, _G0, _G1, _PS)


def test_pseudo_outcome_rejects_nonbinary_treatment_and_boundary_propensity():
    with pytest.raises(ValueError, match="d must be binary"):
        binary_ate_pseudo_outcome(_Y, np.array([0.0, 2.0, 0.0, 1.0]), _G0, _G1, _PS)
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        binary_ate_pseudo_outcome(_Y, _D, _G0, _G1, np.array([0.5, 1.0, 0.5, 0.5]))


# ---------------------------------------------------------------------
# _score_concentration.score_concentration
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "fraction", [True, "0.1", np.array([0.1]), 1j, None, 0.0, 1.5, np.nan]
)
def test_score_concentration_rejects_bad_top_fraction(fraction):
    with pytest.raises(ValueError, match=r"top_fraction must be a finite scalar"):
        score_concentration(np.ones((1, 4)), top_fraction=fraction)


@pytest.mark.parametrize(
    "psi, exc, fragment",
    [
        (np.ones(4), ValueError, r"shape \(n_rep, n_obs\)"),
        (np.ones((1, 4), dtype=complex), ValueError, "complex"),
        (np.array([["a", "b"]]), TypeError, "psi must be numeric"),
        (np.ones((0, 4)), ValueError, "at least one repeat and observation"),
    ],
)
def test_score_concentration_rejects_malformed_scores(psi, exc, fragment):
    with pytest.raises(exc, match=fragment):
        score_concentration(psi)


def test_score_concentration_single_observation_is_unavailable_not_an_alarm():
    (record,) = score_concentration(np.array([[2.0]]))
    assert record["status"] == "diagnostic_unavailable"
    assert record["failure_reason"] == "insufficient_score_count"
    assert record["n_obs"] == 1 and record["top_count"] == 1
    assert record["cmax"] is None and record["effective_score_count"] is None


# ---------------------------------------------------------------------
# _oof_validation primitives
# ---------------------------------------------------------------------


def test_array_view_descriptor_is_immutable_and_returns_itself_on_the_class():
    descriptor = OOFPredictions.__dict__["g0"]
    assert isinstance(descriptor, _val.ArrayView)
    assert OOFPredictions.g0 is descriptor
    with pytest.raises(AttributeError, match="descriptors are immutable"):
        descriptor._storage_name = "other"
    _, predictions = tiny_bundle()
    assert not predictions.g0.flags.writeable
    np.testing.assert_array_equal(predictions.g0, np.ones((1, 4)))


def test_immutable_array_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="y must be numeric"):
        _val.immutable_array(np.array(["a", "b"]), "y", 1)


def test_validated_names_rejects_non_sequence():
    with pytest.raises(TypeError, match="covariate_names must be a sequence"):
        _val.validated_names(5, "covariate_names", 1)


@pytest.mark.parametrize(
    "value, exc, fragment",
    [
        (("a",), TypeError, "train_ids must be a list"),
        (["a", ""], ValueError, "non-empty string IDs"),
        (["a", "a"], ValueError, "unique IDs"),
        (["a", "z"], ValueError, "unknown IDs"),
    ],
)
def test_id_list_rejects_malformed_ids(value, exc, fragment):
    with pytest.raises(exc, match=fragment):
        _val._id_list(value, {"a", "b"}, "train_ids")
    assert _val._id_list(["b", "a"], {"a", "b"}, "train_ids") == ["b", "a"]


@pytest.mark.parametrize(
    "patch, fragment",
    [
        ({"engine": "  "}, "source.engine must be a non-empty string"),
        ({"recipe": 3}, "source.recipe must be a non-empty string"),
        ({"seed": 1.5}, "source.seed must be an integer or None"),
        ({"seed": True}, "source.seed must be an integer or None"),
        ({"software_versions": {"pkg": ""}}, "must map non-empty strings"),
        ({"software_versions": ["pkg"]}, "must map non-empty strings"),
    ],
)
def test_predictions_reject_malformed_source(patch, fragment):
    _, kwargs = tiny_inputs()
    kwargs["source"] = {**kwargs["source"], **patch}
    with pytest.raises(ValueError, match=fragment):
        OOFPredictions.from_arrays(**kwargs)


def test_predictions_reject_training_records_that_are_not_a_list():
    _, kwargs = tiny_inputs()
    kwargs["training_records"] = {"rep": 0}
    with pytest.raises(TypeError, match="training_records must be a list"):
        OOFPredictions.from_arrays(**kwargs)


def test_predictions_reject_non_integer_and_out_of_range_record_keys():
    _, kwargs = tiny_inputs()
    bad = copy.deepcopy(kwargs)
    bad["training_records"][0]["fold_id"] = "0"
    with pytest.raises(ValueError, match="rep and fold_id must be integers"):
        OOFPredictions.from_arrays(**bad)
    bad = copy.deepcopy(kwargs)
    bad["training_records"][1]["fold_id"] = 7
    with pytest.raises(MethodIncompatibility, match="outside fold_ids"):
        OOFPredictions.from_arrays(**bad)


@pytest.mark.parametrize(
    "kept, dropped, fragment",
    [
        ([0, 1, 2], [], "invalid length or negative ordinals"),
        ([0, 1, 2, -1], [], "invalid length or negative ordinals"),
        ([0, 1, 2, 5], [], "cover contiguous original ordinals"),
        ([1, 0, 2, 3], [], "strictly increasing"),
    ],
)
def test_input_mapping_rejections(kept, dropped, fragment):
    with pytest.raises(ValueError, match=fragment):
        _val.validated_input_mapping(kept, dropped, 4)


def test_input_mapping_accepts_interleaved_dropped_rows():
    kept, dropped = _val.validated_input_mapping([0, 2, 3, 5], [1, 4], 4)
    np.testing.assert_array_equal(kept, [0, 2, 3, 5])
    np.testing.assert_array_equal(dropped, [1, 4])


def _bundle_kwargs(metadata_patch=None, *, engine=None, origin=None):
    df, kwargs = tiny_inputs()
    if origin is not None:
        for record in kwargs["training_records"]:
            record["origin"] = origin
    predictions = OOFPredictions.from_arrays(**kwargs)
    metadata = (
        bundle_metadata(predictions)
        if engine is None
        else bundle_metadata(predictions, scoring_engine=engine)
    )
    if metadata_patch is not None:
        metadata_patch(metadata)
    psi_b = np.array([[2.0, 4.0, -2.0, 8.0]])
    return {
        "predictions": predictions,
        "ps_used": np.full((1, 4), 0.5),
        "psi_b": psi_b,
        "psi": psi_b - psi_b.mean(axis=1, keepdims=True),
        "theta": np.array([3.0]),
        "se": np.array([np.sqrt(3.25)]),
        "input_positions": np.arange(4),
        "dropped_positions": np.array([], dtype=int),
        "aggregation": {
            "rule": "median_theta_median_variance_plus_split_deviation",
            "theta": 3.0,
            "se": float(np.sqrt(3.25)),
        },
        "metadata": metadata,
    }


def test_bundle_fixture_is_valid_before_it_is_broken():
    bundle = OOFBundle.from_arrays(**_bundle_kwargs())
    np.testing.assert_allclose(bundle.theta, [3.0])


def test_bundle_rejects_metadata_that_is_not_a_mapping():
    kwargs = _bundle_kwargs()
    kwargs["metadata"] = [1, 2]
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        OOFBundle.from_arrays(**kwargs)


def _set(key, value):
    def patch(metadata):
        metadata[key] = value

    return patch


@pytest.mark.parametrize(
    "patch, exc, fragment",
    [
        (_set("score", "ATTE"), MethodIncompatibility, "score='ATE' and psi_a=-1"),
        (_set("psi_a", 1.0), MethodIncompatibility, "score='ATE' and psi_a=-1"),
        (_set("variance_policy", "ddof1"), MethodIncompatibility, "must be 'ddof0'"),
        (_set("normalize_ipw", True), MethodIncompatibility, "must be false"),
        (_set("trimming_threshold", 0.5), ValueError, "between 0 and 0.5"),
        (_set("trimming_threshold", True), ValueError, "between 0 and 0.5"),
        (_set("clipping_counts", []), ValueError, "one row per repeat"),
        (_set("fit_records", []), ValueError, "cover every repeat/fold"),
    ],
)
def test_bundle_rejects_inconsistent_metadata(patch, exc, fragment):
    with pytest.raises(exc, match=fragment):
        OOFBundle.from_arrays(**_bundle_kwargs(patch))


def test_bundle_rejects_unsorted_fit_records():
    def swap(metadata):
        metadata["fit_records"] = metadata["fit_records"][::-1]

    with pytest.raises(ValueError, match="sorted by repeat/fold"):
        OOFBundle.from_arrays(**_bundle_kwargs(swap))


def test_internal_bundle_requires_integer_seed_and_non_negative_fallbacks():
    internal = dict(engine="statspai_irm_internal", origin="statspai_internal")

    def bad_seed(metadata):
        metadata["fit_records"][0]["fit_seed"] = 1.5

    with pytest.raises(ValueError, match="internal fit_seed must be an integer"):
        OOFBundle.from_arrays(**_bundle_kwargs(bad_seed, **internal))

    def bad_fallback(metadata):
        metadata["fit_records"][0]["subgroup_fallback_counts"] = {"g0": -1, "g1": 0}

    with pytest.raises(ValueError, match="fallback counts must be non-negative"):
        OOFBundle.from_arrays(**_bundle_kwargs(bad_fallback, **internal))

    bundle = OOFBundle.from_arrays(**_bundle_kwargs(**internal))
    assert bundle.metadata["scoring_engine"] == "statspai_irm_internal"


# ---------------------------------------------------------------------
# _external_predictions: hashes, partitions, provenance payload
# ---------------------------------------------------------------------

_GOOD_HASHES = {key: "a" * 64 for key in sorted(_ext._HASH_KEYS)}


def test_copy_hashes_rejects_non_mapping_and_non_sha256():
    with pytest.raises(TypeError, match="hashes must be a mapping"):
        _ext._copy_hashes(["a" * 64])
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _ext._copy_hashes({**_GOOD_HASHES, "data": "A" * 64})
    copied = _ext._copy_hashes(_GOOD_HASHES)
    assert copied == _GOOD_HASHES and copied is not _GOOD_HASHES


def test_partitions_equivalent_is_label_invariant_but_not_merge_invariant():
    left = np.array([0, 0, 1, 1, 2])
    assert _ext.partitions_equivalent(left, np.array([5, 5, 9, 9, 7]))
    # different shapes
    assert not _ext.partitions_equivalent(left, np.array([0, 0, 1, 1]))
    # right merges two left blocks: the reverse map catches it
    assert not _ext.partitions_equivalent(left, np.array([0, 0, 1, 1, 1]))
    # right splits a left block: the forward map catches it
    assert not _ext.partitions_equivalent(left, np.array([0, 3, 1, 1, 2]))


def _payload(**patch):
    payload = {
        "external_predictions_provided": True,
        "external_predictions_hashes": dict(_GOOD_HASHES),
        "store_oof": False,
        "observation_ids_source": "generated_ordinal",
    }
    payload.update(patch)
    return payload


def test_provenance_payload_round_trips_when_valid():
    payload = _payload()
    assert _ext._validate_oof_provenance_payload(payload, requested=True) is payload
    assert _ext._validate_oof_provenance_payload({}, requested=False) == {}


@pytest.mark.parametrize(
    "payload, requested, exc, fragment",
    [
        ([("store_oof", True)], True, TypeError, "must be a plain dict"),
        ({"store_oof": True}, False, ValueError, "legacy DML calls must not add"),
        ({"store_oof": True}, True, ValueError, "must use the exact keys"),
        (
            _payload(external_predictions_provided=1),
            True,
            TypeError,
            "external_predictions_provided must be a bool",
        ),
        (
            _payload(external_predictions_hashes=tuple(_GOOD_HASHES)),
            True,
            TypeError,
            "external_predictions_hashes must be a plain dict",
        ),
        (
            _payload(external_predictions_provided=False),
            True,
            ValueError,
            "must be None without predictions",
        ),
        (
            _payload(observation_ids_source="guessed"),
            True,
            ValueError,
            "observation_ids_source is invalid",
        ),
    ],
)
def test_provenance_payload_rejections(payload, requested, exc, fragment):
    with pytest.raises(exc, match=fragment):
        _ext._validate_oof_provenance_payload(payload, requested=requested)


# ---------------------------------------------------------------------
# _oof_retention: analysis identity
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask", [np.array([1, 0, 1]), np.ones((3, 1), dtype=bool), np.ones(2, dtype=bool)]
)
def test_internal_identity_rejects_a_mask_that_is_not_a_bool_vector(mask):
    with pytest.raises(ValueError, match="one-dimensional bool array"):
        _ret.internal_analysis_identity(
            n_input=3, complete_mask=mask, observation_ids=None
        )


def test_external_identity_rejects_row_count_mismatch():
    _, predictions = tiny_bundle()
    with pytest.raises(ValueError, match="rows do not match input rows"):
        _ret.external_analysis_identity(
            predictions=predictions,
            n_input=5,
            observation_ids_source="generated_ordinal",
        )
