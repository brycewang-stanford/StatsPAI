"""Tests for ``sp.output._lineage`` — numerical lineage / provenance.

The module is foundational: ``sp.replication_pack`` and the upcoming
Quarto emitter both consume ``result._provenance``. These tests cover:

- Hashing semantics for DataFrame / Series / ndarray / bytes.
- ``Provenance`` dataclass round-trip + ``to_dict`` / ``short``.
- ``attach_provenance`` happy path, no-overwrite default,
  ``enabled=False`` short-circuit, immutable-target no-op.
- ``get_provenance`` direct + container walk.
- ``lineage_summary`` aggregation + dedup.
- Param summarisation: scalars pass through, frames become
  fingerprint dicts, long sequences are truncated.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from statspai.output._lineage import (
    Provenance,
    attach_provenance,
    compute_data_hash,
    format_provenance,
    get_provenance,
    lineage_summary,
)


# ---------------------------------------------------------------------------
# compute_data_hash
# ---------------------------------------------------------------------------

class TestComputeDataHash:
    def test_dataframe_stable(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h1 = compute_data_hash(df)
        h2 = compute_data_hash(df)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 12

    def test_dataframe_sensitive_to_content(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        assert compute_data_hash(df1) != compute_data_hash(df2)

    def test_dataframe_sensitive_to_column_name(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        assert compute_data_hash(df1) != compute_data_hash(df2)

    def test_series_supported(self):
        s = pd.Series([1.0, 2.0, 3.0], name="x")
        h = compute_data_hash(s)
        assert isinstance(h, str) and len(h) == 12

    def test_ndarray_supported(self):
        a = np.arange(12).reshape(3, 4)
        h1 = compute_data_hash(a)
        h2 = compute_data_hash(a.copy())
        assert h1 == h2
        # Different shape with same bytes → different hash.
        h3 = compute_data_hash(a.reshape(4, 3))
        assert h1 != h3

    def test_bytes_supported(self):
        h = compute_data_hash(b"hello world")
        assert isinstance(h, str) and len(h) == 12

    def test_none_returns_none(self):
        assert compute_data_hash(None) is None

    def test_unknown_type_returns_none(self):
        assert compute_data_hash({"some": "dict"}) is None
        assert compute_data_hash("a string") is None

    def test_custom_length(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert len(compute_data_hash(df, length=8)) == 8
        assert len(compute_data_hash(df, length=20)) == 20


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------

class TestProvenanceDataclass:
    def test_construct_minimal(self):
        p = Provenance(function="sp.did.canonical_did")
        assert p.function == "sp.did.canonical_did"
        assert p.params == {}
        assert p.data_hash is None
        # auto-populated:
        assert isinstance(p.run_id, str) and len(p.run_id) == 12
        assert p.statspai_version  # non-empty
        assert p.python_version.count(".") == 2
        assert "T" in p.timestamp  # ISO-8601

    def test_to_dict_round_trip(self):
        p = Provenance(
            function="f",
            params={"a": 1, "b": "hello"},
            data_hash="abc123",
            data_shape=[100, 5],
        )
        d = p.to_dict()
        assert d["function"] == "f"
        assert d["params"] == {"a": 1, "b": "hello"}
        assert d["data_hash"] == "abc123"
        assert d["data_shape"] == [100, 5]
        assert "run_id" in d and "timestamp" in d

    def test_short_format(self):
        p = Provenance(function="sp.iv.tsls", data_hash="cafe", run_id="deadbeef")
        s = p.short()
        assert "sp.iv.tsls" in s
        assert "data:cafe" in s
        assert "run:deadbeef" in s

    def test_unique_run_ids(self):
        p1 = Provenance(function="f")
        p2 = Provenance(function="f")
        # Two independent constructions get distinct ids.
        assert p1.run_id != p2.run_id


# ---------------------------------------------------------------------------
# attach_provenance
# ---------------------------------------------------------------------------

class TestAttachProvenance:
    def test_basic_attach(self):
        result = SimpleNamespace(estimate=0.5, se=0.1)
        df = pd.DataFrame({"y": [1, 2], "x": [3, 4]})
        attach_provenance(
            result,
            function="sp.did.test",
            params={"y": "y", "x": "x"},
            data=df,
        )
        assert hasattr(result, "_provenance")
        prov = result._provenance
        assert isinstance(prov, Provenance)
        assert prov.function == "sp.did.test"
        assert prov.params == {"y": "y", "x": "x"}
        assert prov.data_hash is not None
        assert prov.data_shape == [2, 2]

    def test_returns_same_object(self):
        result = SimpleNamespace()
        out = attach_provenance(result, function="f")
        assert out is result

    def test_disabled_is_noop(self):
        result = SimpleNamespace()
        attach_provenance(result, function="f", enabled=False)
        assert not hasattr(result, "_provenance")

    def test_no_overwrite_by_default(self):
        result = SimpleNamespace()
        attach_provenance(result, function="inner")
        first_id = result._provenance.run_id
        attach_provenance(result, function="outer")
        # Inner record preserved.
        assert result._provenance.run_id == first_id
        assert result._provenance.function == "inner"

    def test_overwrite_when_requested(self):
        result = SimpleNamespace()
        attach_provenance(result, function="inner")
        first_id = result._provenance.run_id
        attach_provenance(result, function="outer", overwrite=True)
        assert result._provenance.function == "outer"
        assert result._provenance.run_id != first_id

    def test_immutable_target_silent_noop(self):
        # Tuples do not accept attribute assignment — must not raise.
        result = (1, 2, 3)
        out = attach_provenance(result, function="f")
        assert out is result  # unchanged

    def test_none_target_returns_none(self):
        assert attach_provenance(None, function="f") is None

    def test_handles_non_serialisable_params_gracefully(self):
        result = SimpleNamespace()

        class NotSerialisable:
            def __repr__(self):
                return "<weird>"

        attach_provenance(
            result,
            function="f",
            params={"weird": NotSerialisable(), "n": 42},
        )
        prov = result._provenance
        # Scalar passes through; opaque object becomes its repr.
        assert prov.params["n"] == 42
        assert prov.params["weird"] == "<weird>"

    def test_dataframe_param_becomes_fingerprint(self):
        result = SimpleNamespace()
        df = pd.DataFrame({"a": [1, 2, 3]})
        attach_provenance(result, function="f", params={"data": df})
        cap = result._provenance.params["data"]
        assert isinstance(cap, dict)
        assert cap["_kind"] == "DataFrame"
        assert cap["shape"] == [3, 1]
        assert "hash" in cap

    def test_long_list_truncated(self):
        result = SimpleNamespace()
        attach_provenance(result, function="f", params={"big": list(range(200))})
        cap = result._provenance.params["big"]
        assert len(cap) == 51  # 50 entries + "...(+150 more)" sentinel
        assert isinstance(cap[-1], str) and "+150 more" in cap[-1]


# ---------------------------------------------------------------------------
# get_provenance
# ---------------------------------------------------------------------------

class TestGetProvenance:
    def test_direct(self):
        result = SimpleNamespace()
        attach_provenance(result, function="f")
        prov = get_provenance(result)
        assert isinstance(prov, Provenance)
        assert prov.function == "f"

    def test_missing(self):
        assert get_provenance(SimpleNamespace()) is None
        assert get_provenance(None) is None

    def test_via_dict(self):
        prov = Provenance(function="f")
        assert get_provenance({"_provenance": prov}) is prov

    def test_via_tuple(self):
        result = SimpleNamespace()
        attach_provenance(result, function="f")
        diagnostics = SimpleNamespace()  # no provenance
        # Walks to find first item with provenance.
        assert get_provenance((diagnostics, result)) is result._provenance


# ---------------------------------------------------------------------------
# format_provenance
# ---------------------------------------------------------------------------

class TestFormatProvenance:
    def test_minimal(self):
        p = Provenance(function="sp.did.test")
        out = format_provenance(p)
        assert "function   : sp.did.test" in out
        assert "run_id     :" in out
        assert "StatsPAI v" in out

    def test_with_data_and_params(self):
        p = Provenance(
            function="f",
            params={"y": "wage", "n": 100},
            data_hash="abcdef",
            data_shape=[100, 5],
        )
        out = format_provenance(p)
        assert "SHA256:abcdef" in out
        assert "100×5" in out
        assert "y = 'wage'" in out
        assert "n = 100" in out


# ---------------------------------------------------------------------------
# lineage_summary
# ---------------------------------------------------------------------------

class TestLineageSummary:
    def test_aggregates_runs(self):
        df = pd.DataFrame({"a": [1, 2]})
        r1 = SimpleNamespace()
        r2 = SimpleNamespace()
        attach_provenance(r1, function="f1", data=df)
        attach_provenance(r2, function="f2", data=df)
        summary = lineage_summary(r1, r2)
        assert summary["n_runs"] == 2
        assert len(summary["runs"]) == 2
        # Both consumed the same data hash.
        assert len(summary["data_inputs"]) == 1
        consumers = summary["data_inputs"][0]["consumers"]
        assert len(consumers) == 2
        funcs = {c["function"] for c in consumers}
        assert funcs == {"f1", "f2"}

    def test_skips_unprovenanced(self):
        r1 = SimpleNamespace()
        attach_provenance(r1, function="f1")
        r2 = SimpleNamespace()  # no provenance
        summary = lineage_summary(r1, r2)
        assert summary["n_runs"] == 1

    def test_empty(self):
        summary = lineage_summary()
        assert summary["n_runs"] == 0
        assert summary["runs"] == {}
        assert summary["data_inputs"] == []
        assert "statspai_version" in summary
