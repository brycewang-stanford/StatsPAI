"""Examples-coverage audit for the public registry surface.

Every registered function should carry a NumPy-style ``Examples``
section in its docstring (JOSS review checklist: "example usage") and,
ideally, a registry-level ``example`` snippet for the agent surface
(``sp.help`` / MCP cards). This script measures both, so the gap list
is precise and the campaign that closes it can be ratcheted in CI.

Usage
-----
python scripts/examples_coverage.py                  # per-category summary
python scripts/examples_coverage.py --missing causal # gaps in one category
python scripts/examples_coverage.py --markdown PATH  # full inventory table
python scripts/examples_coverage.py --check --max-missing N
                                                     # CI ratchet: fail if the
                                                     # number of registered
                                                     # functions whose
                                                     # docstring lacks an
                                                     # Examples section
                                                     # exceeds N
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

# Resolve registered functions via their real source (not just top-level
# getattr), so the intentionally submodule-scoped functions (e.g.
# sp.causal_llm.echo_client) are measured against their actual docstring.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ...and make ``import statspai`` resolve from THIS checkout's source, the
# way dump_schemas / registry_stats already do. Without it the audit runs
# against whatever editable install is on the path -- in a git worktree that
# is the main checkout, so the gate silently measures someone else's tree.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from _resolve import resolve_registered  # noqa: E402

_EXAMPLES_HEADER = re.compile(r"^\s*Examples?\s*\n\s*-{3,}", re.MULTILINE)


def _has_docstring_examples(obj) -> Optional[bool]:
    """True/False for resolvable callables; None when unresolvable."""
    try:
        doc = inspect.getdoc(obj)
    except Exception:
        return None
    if not doc:
        return False
    return bool(_EXAMPLES_HEADER.search(doc))


def collect() -> List[Dict]:
    import statspai as sp

    rows: List[Dict] = []
    for name in sorted(sp.list_functions()):
        try:
            spec = sp.describe_function(name)
        except Exception:
            spec = {}
        obj = resolve_registered(sp, name)
        if obj is None or not callable(obj):
            resolved, has_doc_ex = False, None
        else:
            resolved, has_doc_ex = True, _has_docstring_examples(obj)
        rows.append(
            {
                "name": name,
                "category": (spec or {}).get("category") or "(uncategorised)",
                "resolved": resolved,
                "docstring_examples": has_doc_ex,
                "registry_example": bool((spec or {}).get("example")),
            }
        )
    return rows


def summarise(rows: List[Dict]) -> str:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = [
        f"{'category':<22} {'n':>5} {'doc-ex':>7} {'doc-ex %':>9} "
        f"{'reg-ex':>7} {'unresolved':>11}"
    ]
    total = doc_ok = reg_ok = unresolved = 0
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        rs = by_cat[cat]
        n = len(rs)
        d = sum(1 for r in rs if r["docstring_examples"])
        g = sum(1 for r in rs if r["registry_example"])
        u = sum(1 for r in rs if not r["resolved"])
        total += n
        doc_ok += d
        reg_ok += g
        unresolved += u
        lines.append(f"{cat:<22} {n:>5} {d:>7} {100 * d / n:>8.1f}% {g:>7} {u:>11}")
    lines.append("-" * len(lines[0]))
    lines.append(
        f"{'TOTAL':<22} {total:>5} {doc_ok:>7} "
        f"{100 * doc_ok / total:>8.1f}% {reg_ok:>7} {unresolved:>11}"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--missing", metavar="CATEGORY", default=None)
    ap.add_argument("--markdown", metavar="PATH", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-missing", type=int, default=None)
    args = ap.parse_args(argv)

    rows = collect()

    if args.missing is not None:
        gaps = [
            r["name"]
            for r in rows
            if r["category"] == args.missing and not r["docstring_examples"]
        ]
        print(f"# {args.missing}: {len(gaps)} without docstring Examples")
        for n in gaps:
            print(n)
        return 0

    if args.markdown is not None:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(
                "| function | category | docstring Examples " "| registry example |\n"
            )
            fh.write("| --- | --- | --- | --- |\n")
            for r in rows:
                fh.write(
                    f"| `{r['name']}` | {r['category']} "
                    f"| {'✅' if r['docstring_examples'] else '❌'} "
                    f"| {'✅' if r['registry_example'] else '❌'} |\n"
                )
        print(f"wrote {len(rows)} rows to {args.markdown}")
        return 0

    print(summarise(rows))

    if args.check:
        missing = sum(1 for r in rows if not r["docstring_examples"])
        budget = args.max_missing
        if budget is None:
            print("--check requires --max-missing N", file=sys.stderr)
            return 2
        print(f"\n[examples-coverage] missing={missing} budget={budget}")
        if missing > budget:
            print(
                "[examples-coverage] REGRESSION: new public functions must "
                "ship a docstring Examples section (CLAUDE.md §4).",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
