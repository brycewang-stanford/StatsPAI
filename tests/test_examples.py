"""Tests for ``sp.examples(name)`` — runnable code-snippet surface.

Pins: registry-driven dispatch, curated-snippet coverage for the
flagship surface, fallback to ``registry.example`` for unknown names,
JSON-safety, and bounded payload size.
"""

from __future__ import annotations

import json

import pytest

import statspai as sp


# ---------------------------------------------------------------------------
#  Top-level export
# ---------------------------------------------------------------------------


class TestExport:

    def test_callable(self):
        assert callable(sp.examples)

    def test_in_all(self):
        assert "examples" in sp.__all__


# ---------------------------------------------------------------------------
#  Return shape
# ---------------------------------------------------------------------------


class TestReturnShape:

    def test_top_level_keys(self):
        ex = sp.examples("did")
        for k in ("name", "category", "description", "signature",
                  "examples", "pre_conditions", "assumptions",
                  "alternatives", "known_function"):
            assert k in ex, f"missing key {k!r}"

    def test_each_example_has_title_and_code(self):
        ex = sp.examples("did")
        for e in ex["examples"]:
            assert "title" in e
            assert "code" in e
            assert isinstance(e["code"], str)
            assert len(e["code"]) > 0

    def test_payload_strict_json_safe(self):
        ex = sp.examples("did")
        json.dumps(ex)

    def test_payload_bounded(self):
        # Each method's example block should fit comfortably in an
        # agent's tool result — pin a 4K-char ceiling.
        ex = sp.examples("did")
        assert len(json.dumps(ex)) < 4000


# ---------------------------------------------------------------------------
#  Curated-snippet coverage
# ---------------------------------------------------------------------------


class TestCuratedCoverage:

    @pytest.mark.parametrize("name", [
        "regress", "did", "callaway_santanna", "rdrobust", "ivreg",
        "ebalance", "synth", "audit", "preflight", "detect_design",
    ])
    def test_flagship_has_curated_snippet(self, name):
        ex = sp.examples(name)
        assert len(ex["examples"]) >= 1, (
            f"flagship method {name!r} has no curated snippet")
        # Every snippet must be valid Python (parses without
        # SyntaxError). We compile in 'exec' mode.
        for snippet in ex["examples"]:
            compile(snippet["code"], f"<{name}>", "exec")

    def test_did_snippet_uses_sp_alias(self):
        ex = sp.examples("did")
        assert any("import statspai as sp" in e["code"]
                   for e in ex["examples"])


# ---------------------------------------------------------------------------
#  Registry fallback
# ---------------------------------------------------------------------------


class TestRegistryFallback:

    def test_unknown_function_returns_empty_examples(self):
        ex = sp.examples("absolutely_not_a_real_function")
        assert ex["known_function"] is False
        assert ex["examples"] == []

    def test_registered_function_without_curated_snippet_uses_field(self):
        # Pick a registered function we know has an `example` field
        # but no hand-curated snippet (e.g. 'iv' if not curated, or
        # any other registered fn). Skip if we can't find one.
        from statspai.registry import _REGISTRY
        for name, spec in (_REGISTRY or {}).items():
            from statspai.smart.examples import _CURATED
            if name not in _CURATED and getattr(spec, "example", ""):
                ex = sp.examples(name)
                assert ex["known_function"] is True
                assert len(ex["examples"]) == 1
                assert "Registry quick-start" == ex["examples"][0]["title"]
                return
        pytest.skip("no registered fn with example but no curated snippet")


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_name_lower_cased(self):
        a = sp.examples("DID")
        b = sp.examples("did")
        assert a["name"] == b["name"]

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            sp.examples("")
        with pytest.raises(ValueError):
            sp.examples("   ")

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            sp.examples(123)
