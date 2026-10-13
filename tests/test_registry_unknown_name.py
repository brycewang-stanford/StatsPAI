"""A name the registry does not know is answered with the nearest ones."""

from __future__ import annotations

import pytest

import statspai as sp

LOOKUPS = [
    sp.describe_function,
    sp.function_schema,
    sp.agent_card,
    sp.support_tier,
]


@pytest.mark.parametrize("lookup", LOOKUPS)
def test_misspelling_names_the_function_meant(lookup):
    with pytest.raises(KeyError) as err:
        lookup("callaway_santana")
    assert err.value.suggestions[0] == "callaway_santanna"
    assert "did you mean 'callaway_santanna'" in str(err.value)


def test_sp_prefix_and_case_are_forgiven():
    with pytest.raises(KeyError) as err:
        sp.describe_function("sp.RDrobust")
    assert err.value.suggestions[0] == "rdrobust"


def test_nothing_close_points_at_search():
    with pytest.raises(KeyError) as err:
        sp.describe_function("zzzzqqq")
    assert err.value.suggestions == []
    assert "sp.search_functions('zzzzqqq')" in str(err.value)


def test_kind_filter_separates_callables_from_classes():
    everything = sp.list_functions()
    functions = sp.list_functions(kind="function")
    classes = sp.list_functions(kind="class")
    assert sorted(functions + classes) == sorted(everything)
    assert "regress" in functions and "CausalResult" in classes
    assert not set(functions) & set(classes)
    # The bulk tool export and the filter agree on what is callable.
    assert set(functions) == {s["name"] for s in sp.all_schemas()}
    assert sp.list_functions(core=True, kind="class") == []
    with pytest.raises(ValueError, match="kind="):
        sp.list_functions(kind="estimator")
