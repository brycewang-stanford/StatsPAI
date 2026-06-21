"""Guard: every registered ``example`` must parse and bind to its signature.

StatsPAI is agent-native — an agent reads ``sp.describe_function(name)`` /
``sp.function_schema(name)`` and copies the registered ``example`` string. If
that example references a keyword the function does not accept, or does not even
parse, the agent gets a deterministic ``TypeError`` / ``SyntaxError`` on the
exact happy path the package is built around.

This test statically (a) parses each example and (b) binds the keyword
arguments of the ``sp.<name>(...)`` call against the real signature. It does
*not* execute examples (many are illustrative snippets with undefined symbols),
so it is fast and deterministic. It exists to stop the registry/example drift
fixed in this change from silently coming back.
"""

import ast
import inspect

import pytest

import statspai as sp
from statspai import registry

registry._ensure_full_registry()
_REGISTRY = registry._REGISTRY


def _resolve(name):
    fn = getattr(sp, name, None)
    if callable(fn):
        return fn
    # submodule-scoped functions (e.g. sp.causal_llm.openai_client)
    for mod_name in dir(sp):
        mod = getattr(sp, mod_name, None)
        if mod is not None and hasattr(mod, name):
            cand = getattr(mod, name)
            if callable(cand):
                return cand
    return None


def _accepted(fn):
    """Return (kwarg_names, has_var_keyword) or (None, False) if introspection
    fails (built-ins / C funcs)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None, False
    names, var_kw = set(), False
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            var_kw = True
        elif p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY):
            names.add(p.name)
    return names, var_kw


_EXAMPLES = [
    (name, spec.example)
    for name, spec in _REGISTRY.items()
    if isinstance(getattr(spec, "example", None), str) and spec.example.strip()
]


@pytest.mark.parametrize("name,example", _EXAMPLES, ids=[n for n, _ in _EXAMPLES])
def test_registered_example_parses_and_binds(name, example):
    # (a) must parse
    try:
        tree = ast.parse(example.strip(), mode="exec")
    except SyntaxError as exc:  # pragma: no cover - failure message path
        pytest.fail(
            f"registered example for {name!r} is not valid Python: "
            f"{exc}\n    {example!r}"
        )

    # (b) locate the sp.<name>(...) call and bind its kwargs
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = (
                f.attr
                if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else None
            )
            if fname == name:
                target = node
                break
    if target is None:
        return  # example does not directly call the function — nothing to bind

    fn = _resolve(name)
    if fn is None:
        return
    accepted, var_kw = _accepted(fn)
    if accepted is None or var_kw:
        return  # **kwargs sink or un-introspectable — cannot mismatch

    used = [kw.arg for kw in target.keywords if kw.arg is not None]
    bad = [k for k in used if k not in accepted]
    assert not bad, (
        f"registered example for {name!r} passes unknown keyword(s) {bad}; "
        f"signature accepts {sorted(accepted)}\n    {example!r}"
    )
