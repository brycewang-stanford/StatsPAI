#!/usr/bin/env python3
"""Report every registered function that takes a seed, and its default.

The default seed is not uniform across StatsPAI (``None``, ``0``, ``42``,
``12345`` ...). Unifying it would change seeded numbers and is a decision
per estimator, not a clean-up; what must hold everywhere is that the
*name* of the seed argument is one the result card recognises, so that
``sp.result_card(fit)["provenance"]`` can say whether a stochastic output
is reproducible. This script prints the inventory that decision needs;
``tests/test_seed_contract.py`` enforces the naming rule.

Usage
-----
    python scripts/seed_inventory.py            # summary
    python scripts/seed_inventory.py --all      # one row per function
    python scripts/seed_inventory.py --json
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import re
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: A parameter is seed-like when its name says so.
SEED_LIKE = re.compile(r"(^|_)(seed|random_state|rng)$")


def inventory() -> List[Dict[str, Any]]:
    import statspai as sp
    from statspai._result_contract import _SEED_PARAMS

    rows: List[Dict[str, Any]] = []
    for name in sp.list_functions():
        obj = getattr(sp, name, None)
        if obj is None or inspect.isclass(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        for pname, param in params.items():
            if not SEED_LIKE.search(pname):
                continue
            default = param.default
            rows.append(
                {
                    "function": name,
                    "parameter": pname,
                    "default": (
                        "required"
                        if default is inspect.Parameter.empty
                        else repr(default)
                    ),
                    "recognised_by_result_card": pname in _SEED_PARAMS,
                }
            )
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_default: Dict[str, int] = {}
    by_name: Dict[str, int] = {}
    for row in rows:
        by_default[row["default"]] = by_default.get(row["default"], 0) + 1
        by_name[row["parameter"]] = by_name.get(row["parameter"], 0) + 1
    return {
        "functions": len({r["function"] for r in rows}),
        "by_default": dict(sorted(by_default.items(), key=lambda kv: -kv[1])),
        "by_parameter": dict(sorted(by_name.items(), key=lambda kv: -kv[1])),
        "unrecognised": sorted(
            f"{r['function']}({r['parameter']}=)"
            for r in rows
            if not r["recognised_by_result_card"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="one row per function")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    rows = inventory()
    summary = summarise(rows)
    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, indent=2))
        return 0
    print(f"functions taking a seed: {summary['functions']}")
    print("default            functions")
    for default, n in summary["by_default"].items():
        print(f"  {default:<16} {n}")
    print("parameter name     functions")
    for pname, n in summary["by_parameter"].items():
        print(f"  {pname:<16} {n}")
    if summary["unrecognised"]:
        print("not read by the result card:", ", ".join(summary["unrecognised"]))
    if args.all:
        for row in sorted(rows, key=lambda r: (r["default"], r["function"])):
            print(f"  {row['function']:<40} {row['parameter']:<14} {row['default']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
