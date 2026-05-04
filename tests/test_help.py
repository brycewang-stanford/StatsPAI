"""Tests for sp.help() unified entry point and CLI."""
from __future__ import annotations

import io
import json
import sys

import pytest


# --------------------------------------------------------------------------- #
#  sp.help() — Python API
# --------------------------------------------------------------------------- #

class TestTopLevelHelp:
    def test_help_no_args_returns_overview(self):
        import statspai as sp
        r = sp.help()
        # HelpResult prints as text; should mention the package and version.
        text = str(r)
        assert "StatsPAI" in text
        assert sp.__version__ in text
        assert "CATEGORIES" in text
        assert "sp.help" in text

    def test_help_overview_dict_format(self):
        import statspai as sp
        d = sp.help(format="dict")
        assert d["kind"] == "overview"
        assert d["version"] == sp.__version__
        assert isinstance(d["categories"], dict)
        assert sum(d["categories"].values()) >= 40  # at least hand-written specs

    def test_help_bad_format_raises(self):
        import statspai as sp
        with pytest.raises(ValueError):
            sp.help(format="xml")


class TestFunctionLookup:
    def test_help_by_function_name(self):
        import statspai as sp
        r = sp.help("did")
        text = str(r)
        assert "sp.did" in text
        assert "causal" in text

    def test_help_function_dict_format(self):
        import statspai as sp
        d = sp.help("regress", format="dict")
        assert d["name"] == "regress"
        assert d["category"] == "regression"
        assert any(p["name"] == "formula" for p in d["params"])

    def test_help_scoped_category_dot_name(self):
        import statspai as sp
        r = sp.help("causal.did")
        assert "sp.did" in str(r)

    def test_help_callable_object(self):
        import statspai as sp
        r = sp.help(sp.regress)
        text = str(r)
        # Should fall through to registry detail (has one).
        assert "sp.regress" in text or "regress" in text


class TestCategoryListing:
    def test_help_by_category_name(self):
        import statspai as sp
        r = sp.help("causal")
        text = str(r)
        assert "Category: causal" in text
        assert "did" in text


class TestSearch:
    def test_help_search_keyword(self):
        import statspai as sp
        r = sp.help(search="treatment")
        text = str(r)
        assert "Search:" in text
        assert "match" in text

    def test_help_search_dict_format(self):
        import statspai as sp
        d = sp.help(search="synthetic control", format="dict")
        assert d["query"] == "synthetic control"
        assert isinstance(d["results"], list)


class TestNotFound:
    def test_help_unknown_topic_returns_suggestions(self):
        import statspai as sp
        r = sp.help("diid")  # typo
        text = str(r)
        assert "No match" in text
        # Should suggest 'did'
        assert "did" in text.lower()

    def test_help_unknown_dict_format(self):
        import statspai as sp
        d = sp.help("totallynosuchthing", format="dict")
        assert d["kind"] == "not_found"
        assert isinstance(d["function_suggestions"], list)

    def test_help_bad_topic_type_raises(self):
        import statspai as sp
        with pytest.raises(TypeError):
            sp.help(42)


# --------------------------------------------------------------------------- #
#  Registry auto-registration
# --------------------------------------------------------------------------- #

class TestRegistryExpansion:
    def test_list_functions_covers_more_than_hand_written(self):
        import statspai as sp
        names = sp.list_functions()
        # Hand-written baseline was 41; auto-registration should lift this
        # substantially (hundreds of exports in __all__).
        assert len(names) > 100, f"only {len(names)} functions registered"

    def test_list_functions_includes_new_modules(self):
        import statspai as sp
        names = set(sp.list_functions())
        # These were previously unregistered.
        for n in ("bayes_mte", "frontier", "mixed", "rdrobust"):
            assert n in names, f"{n} missing from expanded registry"

    def test_describe_auto_registered_returns_params(self):
        import statspai as sp
        # Pick a function that likely wasn't hand-registered.
        info = sp.describe_function("frontier")
        assert info["name"] == "frontier"
        # Auto-registered specs get category via __module__ introspection.
        assert info["category"] in {"frontier", "regression", "panel", "other"}

    def test_all_schemas_stable(self):
        import statspai as sp
        schemas = sp.all_schemas()
        assert len(schemas) > 100
        for s in schemas:
            assert "name" in s
            assert "parameters" in s

    def test_top_level_all_has_no_duplicates(self):
        import statspai as sp
        assert len(sp.__all__) == len(set(sp.__all__))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

class TestCLI:
    def _run(self, *argv):
        from statspai.cli import main
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = main(list(argv))
        finally:
            sys.stdout = old
        return rc, buf.getvalue()

    def test_cli_version(self):
        import statspai as sp
        rc, out = self._run("version")
        assert rc == 0
        assert sp.__version__ in out

    def test_cli_version_flag(self):
        import statspai as sp
        rc, out = self._run("--version")
        assert rc == 0
        assert sp.__version__ in out

    def test_cli_list(self):
        rc, out = self._run("list")
        assert rc == 0
        assert "did" in out
        assert "regress" in out

    def test_cli_list_category(self):
        rc, out = self._run("list", "--category", "causal")
        assert rc == 0
        assert "did" in out
        assert "regress" not in out

    def test_cli_list_json(self):
        rc, out = self._run("list", "--json")
        assert rc == 0
        data = json.loads(out)
        assert isinstance(data, list)
        assert "did" in data

    def test_cli_describe_text(self):
        rc, out = self._run("describe", "did")
        assert rc == 0
        assert "did" in out

    def test_cli_describe_json(self):
        rc, out = self._run("describe", "regress", "--json")
        assert rc == 0
        data = json.loads(out)
        assert data["name"] == "regress"

    def test_cli_describe_unknown_returns_nonzero(self):
        rc, _ = self._run("describe", "nonexistent_fn_xyz")
        assert rc == 2

    def test_cli_search(self):
        rc, out = self._run("search", "treatment")
        assert rc == 0
        assert "match" in out

    def test_cli_search_json(self):
        rc, out = self._run("search", "synthetic", "--json")
        assert rc == 0
        data = json.loads(out)
        assert isinstance(data, list)

    def test_cli_help_subcommand(self):
        rc, out = self._run("help")
        assert rc == 0
        assert "StatsPAI" in out

    def test_cli_help_topic(self):
        rc, out = self._run("help", "did")
        assert rc == 0
        assert "did" in out

    def test_cli_default_prints_overview(self):
        rc, out = self._run()
        assert rc == 0
        assert "StatsPAI" in out
