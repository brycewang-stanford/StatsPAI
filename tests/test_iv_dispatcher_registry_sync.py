"""Guard: ``sp.iv``'s registry ``method`` enum must match the live dispatcher.

``sp.iv`` is a callable subpackage and the entry point for the whole IV
family — ``sp.iv(..., method='ujive')``, ``method='mte'``,
``method='post_lasso'`` and 23 others all route through
``statspai.iv._METHOD_ALIASES``.

The registry advertised only five of them (``2sls``, ``liml``, ``fuller``,
``gmm``, ``jive``).  Because ``sp.describe_function('iv')`` /
``sp.function_schema('iv')`` are how an agent discovers what a function can
do, 21 working estimators were invisible: an agent reading the schema would
conclude StatsPAI has no MTE, no MST bounds, no plausibly-exogenous
sensitivity and no jackknife-IV variants, when every one of them ships and
is tested.

The fix is the enum, not new top-level exports: routing a family through one
dispatcher is the documented house pattern (CLAUDE.md §3 rule 4), so adding
``sp.mte`` / ``sp.ujive`` / … beside ``sp.iv(method=...)`` would duplicate
the entry point rather than close the gap.
"""

from __future__ import annotations

import statspai as sp
from statspai.iv import _METHOD_ALIASES


def _registry_method_enum() -> list:
    spec = sp.describe_function("iv")
    for param in spec["params"]:
        if param["name"] == "method":
            return param["enum"] or []
    raise AssertionError("sp.iv has no 'method' parameter in the registry")


def test_dispatcher_exposes_a_broad_family():
    """Guard the guard: if the alias table collapsed, everything below would
    pass vacuously."""
    assert len(set(_METHOD_ALIASES.values())) >= 20


def test_every_dispatcher_method_is_advertised():
    live = set(_METHOD_ALIASES.values())
    advertised = set(_registry_method_enum())
    missing = sorted(live - advertised)
    assert not missing, (
        "sp.iv routes these methods but the registry does not advertise them, "
        "so agents cannot discover them: " + ", ".join(missing)
    )


def test_no_advertised_method_is_dead():
    """The reverse: the enum must not promise a method that does not route."""
    live = set(_METHOD_ALIASES.values())
    advertised = set(_registry_method_enum())
    dead = sorted(advertised - live)
    assert not dead, "the registry advertises methods sp.iv cannot route: " + ", ".join(
        dead
    )


def test_advertised_methods_all_resolve_through_the_alias_table():
    """Each advertised name must be usable verbatim as ``method=``."""
    unresolvable = [
        m
        for m in _registry_method_enum()
        if _METHOD_ALIASES.get(m.lower().strip().replace("-", "_")) is None
    ]
    assert (
        not unresolvable
    ), "advertised method names that sp.iv would reject: " + ", ".join(unresolvable)
