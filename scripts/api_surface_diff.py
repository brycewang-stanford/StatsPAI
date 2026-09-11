"""Print the asymmetry between ``statspai.__all__`` and the registry.

Developer ergonomic helper. ``tests/test_api_surface_consistency.py`` enforces
that no *new* drift slips in; this script gives a local view of the current
asymmetry plus a copy-pasteable update for the test baselines.

Usage::

    python scripts/api_surface_diff.py             # summary + lists
    python scripts/api_surface_diff.py --baseline  # print baselines that
                                                  # match current state,
                                                  # ready to paste into the
                                                  # consistency test
"""

from __future__ import annotations

import argparse
import importlib
from typing import Iterable

import statspai
import statspai as sp


def _registry_example_module(example: str | None) -> str | None:
    if not example or not isinstance(example, str):
        return None
    head = example.strip().split("(", 1)[0]
    parts = head.split(".")
    if len(parts) >= 3 and parts[0] in {"sp", "statspai"}:
        return parts[1]
    return None


def _classify():
    all_set = frozenset(getattr(statspai, "__all__", []))
    reg_set = frozenset(sp.list_functions())

    all_not_registered = sorted(all_set - reg_set)
    registered_not_in_all = sorted(reg_set - all_set)

    truly_unreachable: list[str] = []
    submodule_only: list[str] = []
    for name in reg_set:
        if hasattr(statspai, name):
            continue
        spec = sp.describe_function(name) or {}
        submod = _registry_example_module(spec.get("example"))
        if submod is None:
            truly_unreachable.append(name)
            continue
        try:
            mod = importlib.import_module(f"statspai.{submod}")
        except ImportError:
            truly_unreachable.append(name)
            continue
        if hasattr(mod, name):
            submodule_only.append(name)
        else:
            truly_unreachable.append(name)
    submodule_only.sort()
    truly_unreachable.sort()

    return {
        "all_total": len(all_set),
        "registry_total": len(reg_set),
        "all_not_registered": all_not_registered,
        "registered_not_in_all": registered_not_in_all,
        "submodule_only": submodule_only,
        "truly_unreachable": truly_unreachable,
    }


def _print_block(title: str, items: Iterable[str]) -> None:
    items = list(items)
    print(f"\n{title} ({len(items)}):")
    if not items:
        print("  (none)")
        return
    for x in items:
        print(f"  {x}")


def _print_baseline(state: dict) -> None:
    print("\n# Paste-ready baselines for tests/test_api_surface_consistency.py")
    print("ALL_NOT_REGISTERED_BASELINE = frozenset({")
    for x in state["all_not_registered"]:
        print(f'    "{x}",')
    print("})\n")
    print("REGISTERED_NOT_IN_ALL_BASELINE = frozenset({")
    for x in state["registered_not_in_all"]:
        print(f'    "{x}",')
    print("})\n")
    print("SUBMODULE_ONLY_BASELINE = frozenset({")
    for x in state["submodule_only"]:
        print(f'    "{x}",')
    print("})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print frozenset literals matching current state (ready to "
        "paste into tests/test_api_surface_consistency.py).",
    )
    args = parser.parse_args(argv)
    state = _classify()

    print(f"__all__:  {state['all_total']}")
    print(f"registry: {state['registry_total']}")

    _print_block(
        "in __all__ but NOT registered (modules / constants)",
        state["all_not_registered"],
    )
    _print_block(
        "registered but NOT in __all__ (missing top-level re-export)",
        state["registered_not_in_all"],
    )
    _print_block(
        "registered, top-level absent, submodule available " "(documented in example)",
        state["submodule_only"],
    )
    _print_block(
        "registered but neither sp.<name> nor documented submodule resolves " "— BUG",
        state["truly_unreachable"],
    )

    if args.baseline:
        _print_baseline(state)

    return 0 if not state["truly_unreachable"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
