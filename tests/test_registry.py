"""Tests for the AI function registry."""

import pytest


class TestRegistry:
    def test_list_functions(self):
        from statspai import list_functions
        funcs = list_functions()
        assert len(funcs) > 0
        assert "regress" in funcs
        assert "did" in funcs

    def test_list_functions_by_category(self):
        from statspai import list_functions
        causal = list_functions(category="causal")
        assert "did" in causal
        assert "regress" not in causal

    def test_describe_function(self):
        from statspai import describe_function
        info = describe_function("regress")
        assert info["name"] == "regress"
        assert info["category"] == "regression"
        assert len(info["params"]) > 0

    def test_describe_unknown_raises(self):
        from statspai import describe_function
        with pytest.raises(KeyError, match="Unknown function"):
            describe_function("nonexistent_function_xyz")

    def test_function_schema(self):
        from statspai import function_schema
        schema = function_schema("regress")
        assert schema["name"] == "regress"
        assert "parameters" in schema
        assert "properties" in schema["parameters"]
        assert "formula" in schema["parameters"]["properties"]

    def test_search_functions(self):
        from statspai import search_functions
        results = search_functions("treatment")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert any("did" in n or "dml" in n or "match" in n for n in names)

    def test_all_schemas(self):
        from statspai import all_schemas
        schemas = all_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 5
        for s in schemas:
            assert "name" in s
            assert "parameters" in s

    def test_schema_has_required(self):
        from statspai import function_schema
        schema = function_schema("did")
        required = schema["parameters"]["required"]
        assert "data" in required
        assert "y" in required

    def test_survey_in_registry(self):
        from statspai import list_functions
        funcs = list_functions(category="survey")
        assert "svydesign" in funcs
