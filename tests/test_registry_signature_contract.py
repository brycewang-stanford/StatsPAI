"""Contract: every hand-written ``FunctionSpec`` is callable *exactly as advertised*.

An agent reads ``sp.describe_function(name)`` / ``sp.function_schema(name)`` and then
calls ``sp.name(**kwargs)`` using only the documented parameters. Two airtight
invariants guarantee that round-trip never dies on a ``TypeError`` from an unknown
or missing keyword:

* **Invariant A — no phantom params.** Every parameter the spec advertises either
  exists in the real signature *or* is absorbed by a ``**kwargs`` (the dispatcher
  pattern, e.g. ``sp.synth(method=...)``). A spec param that is neither is a lie:
  an agent that passes it gets ``TypeError: unexpected keyword argument``.
* **Invariant B — no missing required params.** Every *required* signature
  parameter (no default) appears in the spec. If the spec hides a required param,
  an agent that follows the schema omits it and gets
  ``TypeError: missing required argument``.

Auto-registered specs (``_auto`` flag) are derived from the signature by
construction and cannot violate these; the drift surface is the hand-written tier,
which is what this module audits exhaustively.

This is the CI guard for the agent-infra schema-drift work line. If it goes red,
a public signature changed without its hand-written spec following — fix the spec
in ``registry.py`` so ``describe_function`` stops lying to agents.
"""

import inspect

import pytest

import statspai as sp
from statspai import registry as R


def _hand_written_callable_specs():
    """Hand-written specs whose name resolves to an introspectable callable.

    Non-callable / non-exported spec names (e.g. internal client stubs) cannot be
    audited for signature drift and are reported separately by
    :func:`test_hand_written_specs_are_callable`.
    """
    R._ensure_full_registry()
    names = []
    for name, spec in R._REGISTRY.items():
        if getattr(spec, "_auto", False):
            continue
        obj = getattr(sp, name, None)
        if obj is None or not callable(obj):
            continue
        try:
            inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        names.append(name)
    return sorted(names)


def _sig_facts(obj):
    """Return ``(param_names, required_names, accepts_extra)``.

    ``accepts_extra`` is True when the function has ``**kwargs`` *or* ``*args`` —
    either one means the keyword-correctness of an advertised param cannot be
    falsified by introspection (dispatchers route through ``**kwargs``; variadic
    table helpers like ``esttab`` slurp model results through ``*args``), so the
    phantom-param check (Invariant A) is skipped for them. Invariant B still
    applies: any *required keyword* param after the variadic must be documented.
    """
    sig = inspect.signature(obj)
    names, required, accepts_extra = [], [], False
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            accepts_extra = True
            continue
        if p.name == "self":
            continue
        names.append(p.name)
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return names, required, accepts_extra


HAND_WRITTEN = _hand_written_callable_specs()


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_no_phantom_params(name):
    """Invariant A: no advertised param the function cannot accept."""
    obj = getattr(sp, name)
    sig_names, _required, accepts_extra = _sig_facts(obj)
    if accepts_extra:
        pytest.skip(f"{name} accepts *args/**kwargs — phantom check N/A")
    spec_params = [p.name for p in R._REGISTRY[name].params]
    phantom = [p for p in spec_params if p not in sig_names]
    assert not phantom, (
        f"{name}: spec advertises param(s) {phantom} that are NOT in the signature "
        f"{sig_names!r} and there is no **kwargs to absorb them. An agent following "
        f"describe_function('{name}') would call sp.{name}({phantom[0]}=...) and get "
        f"TypeError. Fix the ParamSpec name(s) in registry.py to match the signature."
    )


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_no_missing_required_params(name):
    """Invariant B: every required signature param is documented."""
    obj = getattr(sp, name)
    _sig_names, required, _accepts_extra = _sig_facts(obj)
    spec_params = {p.name for p in R._REGISTRY[name].params}
    missing = [p for p in required if p not in spec_params]
    assert not missing, (
        f"{name}: required signature param(s) {missing} are missing from the spec. "
        f"An agent following describe_function('{name}') would omit them and get "
        f"TypeError: missing required argument. Add the ParamSpec(s) in registry.py."
    )


def test_hand_written_specs_are_callable():
    """Every hand-written spec name should resolve to a callable ``sp.<name>``.

    A spec for a name an agent cannot import or call is dead weight in
    ``sp.help`` / ``sp.list_functions``. This is a softer hygiene check than the
    signature contract; it allowlists the small set of known internal stubs.
    """
    R._ensure_full_registry()
    allow = {"particle_filter", "openai_client", "anthropic_client", "echo_client"}
    orphans = []
    for name, spec in R._REGISTRY.items():
        if getattr(spec, "_auto", False):
            continue
        obj = getattr(sp, name, None)
        if (obj is None or not callable(obj)) and name not in allow:
            orphans.append(name)
    assert not orphans, (
        f"Hand-written specs with no callable sp.<name>: {orphans}. Either export the "
        f"function, remove the spec, or add to the allowlist with justification."
    )
