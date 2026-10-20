"""Every runnable card example runs as written.

2026-10-19 audit: the agent-card examples of the most used entry points
assumed a ``df`` no dataset has (``sp.did(df, y="wage", ...)``) or used
keywords the function does not take, so an agent that copied one failed
before learning anything. ``statspai._card_examples`` now holds a
self-contained example for each of the 70 audited entry points; this test
executes them in a fresh namespace holding only ``sp``.
"""

from __future__ import annotations

import ast
import inspect
import sys
import warnings
from pathlib import Path

import pytest

import statspai as sp
from statspai._card_examples import RUNNABLE_EXAMPLES

#: Examples that take more than a few seconds (learners, bootstrap).
SLOW = {
    "tmle",
    "metalearner",
    "dml",
    "did_multiplegt",
    "honest_did",
    "xtintreg",
    "bart_causal",
}


def _audited():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import agent_card_audit
    finally:
        sys.path.remove(str(scripts))
    return list(agent_card_audit.AUDITED)


def test_every_audited_entry_point_has_a_runnable_example():
    assert set(_audited()) <= set(RUNNABLE_EXAMPLES)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n
        for n in sorted(RUNNABLE_EXAMPLES)
    ],
)
def test_example_runs(name):
    code = RUNNABLE_EXAMPLES[name]
    namespace = {"sp": sp}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, f"<card example {name}>", "exec"), namespace)
    assert namespace.get("r") is not None
    # The example calls the function whose card it is on.
    assert f"sp.{name}(" in code or f"sp.iv.{name}(" in code


@pytest.mark.parametrize("name", ["did", "psm", "callaway_santanna", "rdrobust"])
def test_cards_serve_the_runnable_example(name):
    card = sp.describe_function(name)
    assert card["example"] == RUNNABLE_EXAMPLES[name].rstrip("\n")


def _sp_calls(code):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain = ast.unparse(node.func)
            if chain.startswith("sp.") and not chain.startswith(
                ("sp.datasets.", "sp.dgp_")
            ):
                yield chain, node


@pytest.mark.parametrize("name", sorted(RUNNABLE_EXAMPLES))
def test_example_keywords_are_schema_parameters(name):
    # The names an agent sees in function_schema(); aliases would run but
    # would teach a spelling the schema does not list.
    for chain, node in _sp_calls(RUNNABLE_EXAMPLES[name]):
        sig = inspect.signature(eval(chain, {"sp": sp}))
        params = {
            p.name for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD
        }
        varkw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        for kw in node.keywords:
            assert varkw or kw.arg in params, (chain, kw.arg)
