"""Agent-facing schema shape: enums, column roles, spellings, returns, aliases.

Covers the W10 discovery pass on ``sp.function_schema`` / ``sp.agent_schema``
/ ``sp.agent_card`` / ``sp.describe_function`` / ``sp.all_schemas``:

* ``enum`` derived from ``Literal[...]`` annotations, dispatcher registries
  and grounded docstring choice lists (never for SE keywords from prose);
* ``x-statspai-role`` on data / formula / column / column-list parameters,
  with ``data`` typed ``object``;
* ``x-aliases`` / ``x-canonical`` spelling hints, and canonical names in the
  agent-native schema only when the call accepts them;
* array item types implied by numeric defaults;
* ``x_statspai.returns`` / ``card["returns"]`` with the result class fields;
* ``alias_of`` and :func:`statspai.registry.is_alias`;
* per-field card provenance;
* classes kept out of the bulk tool export.
"""

from __future__ import annotations

import inspect

import jsonschema
import pytest

import statspai as sp
from statspai import _schema_enrich as E
from statspai import registry

registry._ensure_full_registry()


def _props(name, **kw):
    return sp.function_schema(name, **kw)["parameters"]["properties"]


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class TestEnumDerivation:
    def test_literal_annotation_becomes_enum(self):
        assert _props("synthdid_placebo")["method"]["enum"] == ["sdid", "sc", "did"]

    @pytest.mark.parametrize(
        "annotation, expected",
        [
            ("Literal['a', 'b']", ["a", "b"]),
            ("Optional[Literal['x', 'y']]", ["x", "y"]),
            ("Union[Literal['x'], str]", None),  # free strings also accepted
            ("str", None),
        ],
    )
    def test_literal_text_parsing(self, annotation, expected):
        assert E.literal_choices(annotation) == expected

    def test_live_literal_object(self):
        from typing import Literal, Optional

        assert E.literal_choices(Optional[Literal["p", "q"]]) == ["p", "q"]

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Kernel function: 'triangular', 'uniform', or 'epanechnikov'.",
                ["triangular", "uniform", "epanechnikov"],
            ),
            ("One of ``'rct'``, ``'did'``, ``'rd'``.", ["rct", "did", "rd"]),
            ("'kink' or 'notch'.", ["kink", "notch"]),
            ("One of: - ``'a'`` : first - ``'b'``, ``'c'`` : others", ["a", "b", "c"]),
        ],
    )
    def test_doc_choice_shapes(self, text, expected):
        found = E.doc_choices(text)
        assert found is not None and found[0] == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Method, e.g. 'a' or 'b'.",
            "'nbinomial' (alias 'negbin') or 'poisson'.",
            "Kernel such as 'a', 'b', etc.",
            "Free text describing 'a' and then more words.",
        ],
    )
    def test_open_sets_are_rejected(self, text):
        assert E.doc_choices(text) is None

    def test_default_must_be_in_doc_enum(self):
        assert (
            E.grounded_doc_enum("'kink' or 'notch'.", "other", False, sp.bunching)
            is None
        )

    def test_se_keywords_never_take_doc_enums(self):
        desc = "Standard error type: 'nonrobust', 'robust' or 'hc1'."
        assert E.grounded_doc_enum(desc, "nonrobust", False, sp.logit, "robust") is None

    def test_values_must_appear_in_source(self):
        desc = "Kernel function: 'triangular' or 'zzz_not_in_code'."
        assert E.grounded_doc_enum(desc, "triangular", False, sp.rkd, "kernel") is None

    def test_grounded_doc_enum_reaches_schema(self):
        prop = _props("bunching")["design"]
        assert prop["enum"] == ["kink", "notch"] and prop["default"] == "kink"

    def test_dispatcher_registry_enum(self):
        enum = _props("synth")["method"]["enum"]
        assert "classic" in enum and "sdid" in enum and "augmented" in enum

    def test_every_enum_contains_its_default(self):
        for schema in sp.all_schemas():
            for pname, prop in schema["parameters"]["properties"].items():
                if "enum" in prop and "default" in prop:
                    assert prop["default"] in prop["enum"], (schema["name"], pname)


# --------------------------------------------------------------------------
# Column roles / spellings / arrays
# --------------------------------------------------------------------------


class TestColumnRoles:
    def test_data_is_a_dataframe_object(self):
        prop = _props("did")["data"]
        assert prop["type"] == "object"
        assert prop["x-statspai-role"] == "dataframe"

    def test_column_and_columns_roles(self):
        props = _props("callaway_santanna")
        assert props["y"]["x-statspai-role"] == "column"
        props = _props("regress")
        assert props["formula"]["x-statspai-role"] == "formula"
        props = _props("dml")
        assert props["covariates"]["x-statspai-role"] == "columns"

    @pytest.mark.parametrize(
        "name, typ, jtype, expected",
        [
            ("y", "str", "string", "column"),
            ("y", "np.ndarray", "string", None),
            ("x", "str", "string", "column"),
            ("x", "Any", "string", None),
            ("covariates", "Optional[List[str]]", "array", "columns"),
            ("formula", "str", "string", "formula"),
            ("data", "pd.DataFrame", "string", "dataframe"),
            ("method", "str", "string", None),
        ],
    )
    def test_role_rules(self, name, typ, jtype, expected):
        assert E.column_role(name, typ, jtype) == expected

    def test_schemas_stay_valid_json_schema(self):
        for schema in sp.all_schemas(agent_native=True)[:400]:
            jsonschema.Draft202012Validator.check_schema(schema["parameters"])


class TestSpellings:
    def test_legacy_name_points_to_accepted_canonical(self):
        prop = _props("synth")["unit"]
        assert prop["x-canonical"] == "id"
        assert "id" in prop["x-aliases"]

    def test_agent_schema_exports_canonical_name(self):
        props = sp.agent_schema("synth")["parameters"]["properties"]
        assert "id" in props and "unit" not in props
        assert "unit" in props["id"]["x-aliases"]

    def test_plain_schema_keeps_signature_names(self):
        props = _props("synth")
        assert "unit" in props and "id" not in props

    def test_canonical_only_renamed_when_accepted(self):
        # A spelling hint without the alias stays a hint: every key the
        # agent schema adds over the plain schema is an accepted alias.
        for name in ("synth", "callaway_santanna", "logit", "sun_abraham"):
            plain = _props(name)
            agent = sp.agent_schema(name)["parameters"]
            aliases = getattr(getattr(sp, name), "__statspai_aliases__", {}) or {}
            for key in set(agent["properties"]) - set(plain):
                assert key in aliases, (name, key)
            assert set(agent["required"]) <= set(agent["properties"])

    def test_bool_robust_gets_no_vce_hint(self):
        prop = _props("did").get("robust")
        if prop is not None:
            assert "x-canonical" not in prop


class TestArrayItems:
    def test_numeric_default_sets_item_type(self):
        prop = _props("gnn_causal")["propensity_bounds"]
        assert prop["items"]["type"] == "number"

    def test_no_string_items_with_numeric_defaults(self):
        bad = []
        for schema in sp.all_schemas():
            for pname, prop in schema["parameters"]["properties"].items():
                d = prop.get("default")
                if (
                    isinstance(d, (list, tuple))
                    and d
                    and prop.get("items", {}).get("type") == "string"
                    and not all(isinstance(v, str) for v in d)
                ):
                    bad.append((schema["name"], pname))
        assert not bad, bad

    @pytest.mark.parametrize(
        "default, expected",
        [
            ([0.05, 0.95], "number"),
            ((64, 64), "integer"),
            ([1, 0.5], "number"),
            (["a"], "string"),
            ([], None),
            (None, None),
        ],
    )
    def test_item_type_from_default(self, default, expected):
        assert E.item_type_from_default(default) == expected


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------


class TestReturns:
    def test_agent_schema_carries_result_shape(self):
        ret = sp.agent_schema("did")["x_statspai"]["returns"]
        assert ret["class"] == "CausalResult"
        assert {"estimate", "se", "ci", "pvalue"} <= set(ret["fields"])
        assert ret["payload_schema"] == "result.schema.json"

    def test_econometric_results_fields(self):
        ret = sp.agent_card("regress")["returns"]
        assert ret["class"] == "EconometricResults"
        assert "params" in ret["fields"]

    def test_dataframe_and_dict_returns(self):
        assert sp.agent_card("rdbalance")["returns"]["class"] == "DataFrame"
        assert sp.agent_card("evalue_rd")["returns"]["class"] == "dict"

    def test_transplanted_annotation_resolves(self):
        ret = sp.agent_card("anderson_rubin_ci")["returns"]
        assert ret["class"] == "WeakIVConfidenceSet"
        assert "lower" in ret["fields"]

    def test_describe_function_result_schema(self):
        d = sp.describe_function("kitagawa_test")
        assert d["result_class"] == "KitagawaResult"
        assert d["result_schema"]["class"] == "KitagawaResult"
        assert d["result_schema"]["fields"]

    def test_any_is_not_a_class(self):
        for spec in registry._REGISTRY.values():
            assert spec.result_class not in ("Any", "object"), spec.name


# --------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------


class TestAliases:
    def test_alias_of_is_exposed(self):
        from statspai.registry import is_alias

        assert sp.describe_function("rdd")["alias_of"] == "rdrobust"
        assert sp.agent_card("psm")["alias_of"] == "match"
        assert sp.agent_schema("rosenbaum_gamma")["x_statspai"]["alias_of"] == (
            "rosenbaum_bounds"
        )
        assert is_alias("rdd") and not is_alias("rdrobust")

    def test_every_shared_callable_is_declared(self):
        from statspai._canonical_aliases import FUNCTION_ALIAS_OF

        by_id = {}
        for name in registry._REGISTRY:
            obj = getattr(sp, name, None)
            if obj is not None and callable(obj) and not inspect.isclass(obj):
                by_id.setdefault(id(obj), []).append(name)
        undeclared = []
        for names in by_id.values():
            if len(names) < 2:
                continue
            canon = [n for n in names if n not in FUNCTION_ALIAS_OF]
            if len(canon) != 1:
                undeclared.append(names)
        assert not undeclared, undeclared

    def test_alias_targets_are_registered_and_not_aliases(self):
        from statspai._canonical_aliases import FUNCTION_ALIAS_OF

        for alias, target in FUNCTION_ALIAS_OF.items():
            assert alias in registry._REGISTRY, alias
            assert target in registry._REGISTRY, target
            assert target not in FUNCTION_ALIAS_OF, target

    def test_card_less_alias_inherits_canonical_card(self):
        card = sp.agent_card("causal_survival")
        assert card["inherits_from"] == "causal_survival_forest"
        assert card["assumptions"]


# --------------------------------------------------------------------------
# Provenance / bulk export
# --------------------------------------------------------------------------


class TestProvenance:
    ALLOWED = registry.PROVENANCE_SOURCES | {"curated+family"}

    def test_values_are_from_the_documented_set(self):
        for name in (
            "did",
            "gsynth",
            "callaway_santanna",
            "rd_honest",
            "kitagawa_test",
        ):
            prov = sp.agent_card(name)["provenance"]
            assert prov and set(prov.values()) <= self.ALLOWED, (name, prov)

    def test_family_template_is_labelled_family(self):
        prov = sp.agent_card("gsynth")["provenance"]
        assert prov["assumptions"] == "family"

    def test_new_estimator_card_is_curated(self):
        prov = sp.agent_card("kitagawa_test")["provenance"]
        assert prov["assumptions"] == "curated"

    def test_inherited_items_are_family(self):
        prov = sp.describe_function("rd_honest")["provenance"]
        assert prov["assumptions"] in {"family", "curated+family"}

    def test_docstring_harvest_is_labelled(self):
        prov = sp.agent_card("gsynth")["provenance"]
        assert prov.get("returns") == "docstring"


class TestBulkExport:
    def test_classes_are_not_exported_as_tools(self):
        names = {s["name"] for s in sp.all_schemas()}
        assert "CausalResult" not in names and "did" in names

    def test_classes_available_on_request(self):
        names = {s["name"] for s in sp.all_schemas(include_classes=True)}
        assert "CausalResult" in names

    def test_describe_function_still_explains_classes(self):
        assert sp.describe_function("CausalResult")["kind"] == "class"
