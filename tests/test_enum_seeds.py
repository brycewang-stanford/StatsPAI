"""Every seeded ``enum`` is re-read from the place the code states it.

``statspai._enum_seeds`` holds choice lists the registry cannot derive from
an annotation or a docstring. A list that drifts from the code is worse than
no list (a strict tool-caller would refuse a valid value or send a dead one),
so each entry is checked against its stated source.
"""

from __future__ import annotations

import inspect
import re
import sys

import pytest

import statspai as sp
from statspai import _enum_seeds as seeds


def _source(name: str) -> str:
    return inspect.getsource(inspect.unwrap(getattr(sp, name)))


def _module(name: str):
    return sys.modules[inspect.unwrap(getattr(sp, name)).__module__]


def _enum(func: str, param: str):
    return sp.function_schema(func)["parameters"]["properties"][param].get("enum")


def _all_keys():
    keys = list(seeds.DIRECT) + list(seeds.CONSTANT) + list(seeds.FORWARDED)
    keys += [(f, "kernel") for f in seeds.KERNEL_USERS]
    return keys


@pytest.mark.parametrize("func,param", _all_keys())
def test_seed_reaches_the_schema(func, param):
    assert _enum(func, param), f"sp.{func}({param}=) has no enum"


@pytest.mark.parametrize("key", list(seeds.DIRECT))
def test_direct_values_are_literals_in_the_function(key):
    func, _param = key
    src = _source(func)
    for value in seeds.DIRECT[key]:
        assert f'"{value}"' in src or f"'{value}'" in src, (func, value)


@pytest.mark.parametrize("key", list(seeds.CONSTANT))
def test_constant_values_equal_the_module_constant(key):
    func, _param = key
    const, values = seeds.CONSTANT[key]
    live = getattr(_module(func), const)
    assert set(values) == set(live), (func, const)


@pytest.mark.parametrize("key", list(seeds.FORWARDED))
def test_forwarded_argument_is_passed_unchanged(key):
    func, param = key
    target, tparam = seeds.FORWARDED[key]
    src = _source(func)
    # The callee may be imported under a private name (``_frontier``) or
    # reached through a helper that takes the keywords as a dict; the
    # function must at least name it, in code or in its docstring.
    assert re.search(rf"{target}\b", src), f"{func} never names {target}"
    assert re.search(rf"\b{tparam}\s*=\s*{param}\b", src), (func, param)
    assert _enum(func, param) == _enum(target, tparam)


@pytest.mark.parametrize("alias", list(seeds.SAME_CALL))
def test_same_call_alias_really_is_the_target(alias):
    target = seeds.SAME_CALL[alias]
    if getattr(sp, alias) is getattr(sp, target):
        return
    body = _source(alias)
    assert re.search(rf"return\s+_?{target}\(\*args,\s*\*\*kwargs\)", body), alias


def test_kernel_choices_are_what_kernel_fn_implements():
    from statspai.rd import _core

    implemented = re.findall(r'kernel == "(\w+)"', inspect.getsource(_core._kernel_fn))
    assert sorted(implemented) == sorted(seeds.KERNEL_FN_CHOICES)


@pytest.mark.parametrize("func", seeds.KERNEL_USERS)
def test_kernel_user_hands_kernel_to_kernel_fn(func):
    assert re.search(r"_kernel_fn\([^()]*(\([^()]*\)[^()]*)*\bkernel\)", _source(func))


def test_a_seeded_enum_matches_what_the_function_rejects():
    """One live probe: a value outside the list is refused by the call."""
    import numpy as np

    with pytest.raises(ValueError, match="default.*fixed.*random"):
        sp.mr_ivw(np.ones(4), np.ones(4), np.ones(4), np.ones(4), model="mixed")
    with pytest.raises(ValueError, match="Unknown adjustment method"):
        sp.adjust_pvalues([0.01, 0.04], method="tukey")
