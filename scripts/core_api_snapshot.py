"""Frozen core API: snapshot the signatures of ``sp.CORE_FUNCTIONS``.

The everyday entry points (``sp.list_functions(core=True)``) promise that
code written against them keeps running. ``tests/core_api_snapshot.json``
records their signatures, and ``tests/test_core_api_freeze.py`` fails on
any change that would break a caller:

* a parameter disappears (renaming is fine when the old spelling stays
  accepted through ``@accepts_aliases``);
* a default value changes, or an optional parameter becomes required;
* the positional parameters are reordered or one becomes keyword-only;
* a new parameter has no default;
* a core function is removed.

New parameters with defaults are not breaks. They join the snapshot the
next time it is written.

A break that is intended goes through the deprecation process (a
``DeprecationWarning`` release, then the change) and is recorded::

    python scripts/core_api_snapshot.py --write \\
        --ack "kaplan_meier.conf_type=<text of the MIGRATION.md entry>"

``--write`` refuses while a break is not acknowledged, and every
acknowledgement must quote text that appears in ``MIGRATION.md``. The
acknowledgements are kept in the snapshot's ``history``.

Usage::

    python scripts/core_api_snapshot.py --check
    python scripts/core_api_snapshot.py --write [--ack FN.PARAM=TEXT ...]
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "tests" / "core_api_snapshot.json"
MIGRATION = REPO / "MIGRATION.md"

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _default(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "<required>"
    return _ADDRESS.sub("", repr(value))


def signature_record(func: Any) -> Dict[str, Any]:
    """The parts of a signature a caller depends on."""
    sig = inspect.signature(func)
    params = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        params.append(
            {
                "name": p.name,
                "positional": p.kind in _POSITIONAL,
                "default": _default(p.default),
            }
        )
    aliases = getattr(func, "__statspai_aliases__", None) or {}
    return {
        "params": params,
        # canonical name -> legacy spelling still accepted
        "aliases": {str(k): str(v) for k, v in sorted(aliases.items())},
    }


def current() -> Dict[str, Dict[str, Any]]:
    sys.path.insert(0, str(REPO / "src"))
    import statspai as sp
    from statspai.registry import CORE_FUNCTIONS

    return {name: signature_record(getattr(sp, name)) for name in CORE_FUNCTIONS}


def breaks(old: Dict[str, Any], new: Dict[str, Any]) -> List[Dict[str, str]]:
    """Changes from ``old`` to ``new`` that would break an existing call."""
    out: List[Dict[str, str]] = []

    def add(fn: str, param: str, change: str) -> None:
        out.append({"function": fn, "param": param, "change": change})

    for fn, rec in old.items():
        if fn not in new:
            add(fn, "*", "core function removed")
            continue
        cur = new[fn]
        by_name = {p["name"]: p for p in cur["params"]}
        legacy = set(cur["aliases"].values())
        for p in rec["params"]:
            q = by_name.get(p["name"])
            if q is None:
                if p["name"] not in legacy:
                    add(fn, p["name"], "parameter removed or renamed")
                continue
            if p["default"] != q["default"]:
                if q["default"] == "<required>":
                    add(fn, p["name"], "optional parameter became required")
                elif p["default"] != "<required>":
                    add(
                        fn,
                        p["name"],
                        f"default changed from {p['default']} to {q['default']}",
                    )
        old_pos = [p["name"] for p in rec["params"] if p["positional"]]
        new_pos = [p["name"] for p in cur["params"] if p["positional"]]
        if new_pos[: len(old_pos)] != old_pos:
            add(
                fn,
                "*",
                f"positional order changed from {old_pos} to {new_pos}",
            )
        known = {p["name"] for p in rec["params"]}
        for q in cur["params"]:
            if q["name"] not in known and q["default"] == "<required>":
                add(fn, q["name"], "new required parameter")
    return out


def load() -> Dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def migration_text() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    ap.add_argument(
        "--ack",
        action="append",
        default=[],
        metavar="FN.PARAM=TEXT",
        help="acknowledge a break; TEXT must appear in MIGRATION.md",
    )
    a = ap.parse_args()

    old = load() if SNAPSHOT.exists() else {"functions": {}, "history": []}
    now = current()
    found = breaks(old["functions"], now)

    if a.check:
        for b in found:
            print(f"BREAK {b['function']}.{b['param']}: {b['change']}")
        extra = sorted(set(now) - set(old["functions"]))
        for fn in extra:
            print(f"NOT IN SNAPSHOT {fn}: run --write")
        ok = not found and not extra
        print("OK" if ok else "core API snapshot check failed")
        return 0 if ok else 1

    acks: Dict[str, str] = {}
    for item in a.ack:
        key, _, text = item.partition("=")
        if not text:
            ap.error(f"--ack {item!r}: expected FN.PARAM=TEXT")
        acks[key.strip()] = text.strip()
    migration = migration_text()
    history = list(old.get("history", []))
    missing = []
    for b in found:
        key = f"{b['function']}.{b['param']}"
        text = acks.get(key)
        if text is None:
            missing.append(f"{key}: {b['change']}")
            continue
        if text not in migration:
            ap.error(f"--ack {key}: the text is not in MIGRATION.md")
        history.append({**b, "migration": text, "date": date.today().isoformat()})
    if missing:
        print("Unacknowledged breaking changes to the frozen core API:")
        for m in missing:
            print(f"  {m}")
        print("Deprecate first, write MIGRATION.md, then pass --ack.")
        return 1
    SNAPSHOT.write_text(
        json.dumps({"functions": now, "history": history}, indent=1, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SNAPSHOT.relative_to(REPO)} ({len(now)} functions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
