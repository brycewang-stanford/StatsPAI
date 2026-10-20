"""``str | list[str]`` parameters export as ``anyOf`` string / array."""

from __future__ import annotations

import pytest

import statspai as sp
from statspai.registry import _union_members, _union_with_scalars


@pytest.mark.parametrize(
    "text,members",
    [
        ("Union[str, List[str], NoneType]", ["str", "List[str]", "NoneType"]),
        ("Optional[Union[str, Sequence[str]]]", ["str", "Sequence[str]", "None"]),
        ("list[str] | str", ["list[str]", "str"]),
        ("Dict[str, List[int]]", ["Dict[str, List[int]]"]),
    ],
)
def test_union_members_split_at_top_level(text, members):
    assert _union_members(text) == members


def test_string_or_list_becomes_anyof():
    prop = {"type": "array", "items": {"type": "string"}, "enum": ["a"]}
    _union_with_scalars(prop, "str | list[str] | dict")
    assert "type" not in prop
    assert prop["anyOf"] == [
        {"type": "string", "enum": ["a"]},
        {"type": "object"},
        {"type": "array", "items": {"type": "string"}},
    ]


@pytest.mark.parametrize(
    "text",
    ["List[str]", "Union[np.ndarray, pd.DataFrame, List[str]]", "Optional[List[str]]"],
)
def test_unions_without_scalar_members_stay_arrays(text):
    prop = {"type": "array", "items": {"type": "string"}}
    _union_with_scalars(prop, text)
    assert prop == {"type": "array", "items": {"type": "string"}}


@pytest.mark.parametrize(
    "fn,param", [("weakrobust", "endog"), ("dml", "cluster"), ("predictions", "by")]
)
def test_exported_schema_accepts_a_single_name(fn, param):
    prop = sp.function_schema(fn)["parameters"]["properties"][param]
    kinds = [b["type"] for b in prop["anyOf"]]
    assert "string" in kinds and "array" in kinds


def test_no_exported_property_mixes_type_and_anyof():
    for name in sp.list_functions():
        try:
            props = sp.function_schema(name)["parameters"]["properties"]
        except Exception:  # classes and non-tool entries
            continue
        for p, prop in props.items():
            assert not ("anyOf" in prop and "type" in prop), (name, p)
