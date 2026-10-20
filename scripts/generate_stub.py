"""Regenerate ``src/statspai/__init__.pyi`` from the live runtime namespace.

Run this whenever ``__all__`` or the lazy import maps change so that IDEs and
static type-checkers (Pyright, mypy, PyCharm) keep seeing every public symbol,
including lazy-loaded ones routed through ``statspai.__getattr__``.

Usage::

    python3 scripts/generate_stub.py           # rewrite the stub
    python3 scripts/generate_stub.py --check   # exit 1 if it is stale

The output is committed to the repo (black / isort format it at commit
time); it is not generated at install time. ``--check`` and
``tests/test_type_stub.py`` compare the declarations, not the formatting.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "statspai"


# Names whose runtime binding involves an ``X as Y`` alias inside __init__.py.
# We hard-code these because ``obj.__name__`` only reports the original symbol
# and we need the public alias on the LHS of the stub line.
ALIAS_LINES: dict[str, str] = {
    "PAPER_TABLE_TEMPLATES": "from .output.paper_tables import TEMPLATES as PAPER_TABLE_TEMPLATES",
    "JOURNAL_PRESETS": "from .output._journals import JOURNALS as JOURNAL_PRESETS",
    "list_journal_templates": "from .output._journals import list_templates as list_journal_templates",
    "get_journal_template": "from .output._journals import get_template as get_journal_template",
    "transport_weights_fn": "from .transport import weights as transport_weights_fn",
    "transport_generalize": "from .transport import generalize as transport_generalize",
    "longitudinal_analyze": "from .longitudinal import analyze as longitudinal_analyze",
    "longitudinal_contrast": "from .longitudinal import contrast as longitudinal_contrast",
    "gformula_ice_fn": "from .gformula import ice as gformula_ice_fn",
    "target_trial_protocol": "from .target_trial import protocol as target_trial_protocol",
    "target_trial_emulate": "from .target_trial import emulate as target_trial_emulate",
    "target_trial_report": "from .target_trial import to_paper as target_trial_report",
    "target_trial_checklist": "from .target_trial import target_checklist as target_trial_checklist",
    "tte": "from . import target_trial as tte",
    "dag_recommend_estimator": "from .dag import recommend_estimator as dag_recommend_estimator",
    "gt": "from .output._gt import to_gt as gt",
}

CONSTANTS: dict[str, str] = {
    "__version__": "__version__: str",
    "__author__": "__author__: str",
    "__email__": "__email__: str",
    "__citation__": "__citation__: str",
    "STABILITY_TIERS": "STABILITY_TIERS: frozenset[str]",
    "TARGET_ITEMS": "TARGET_ITEMS: list[str]",
}


def _relative(modname: str) -> str:
    if modname == "statspai":
        return "."
    return "." + modname[len("statspai.") :]


def build_stub(sp: types.ModuleType) -> tuple[str, dict[str, int]]:
    """The stub text for the imported ``statspai`` package, and counts."""
    submodule_lines: set[str] = set()
    leaf_lines: set[str] = set()
    unresolved: list[str] = []

    for name in sorted(set(sp.__all__)):
        if name in CONSTANTS:
            continue
        if name in ALIAS_LINES:
            leaf_lines.add(ALIAS_LINES[name])
            continue
        obj = getattr(sp, name, None)
        if obj is None:
            unresolved.append(name)
            continue
        if isinstance(obj, types.ModuleType):
            rel = _relative(obj.__name__)
            if rel.startswith(".") and "." not in rel[1:]:
                submodule_lines.add(f"from . import {rel.lstrip('.')} as {name}")
            else:
                parent, leaf = rel.rsplit(".", 1)
                submodule_lines.add(f"from {parent} import {leaf} as {name}")
            continue
        src_mod = getattr(obj, "__module__", None)
        src_name = getattr(obj, "__name__", None) or getattr(obj, "__qualname__", None)
        if not src_mod or not src_mod.startswith("statspai") or not src_name:
            unresolved.append(name)
            continue
        leaf_lines.add(f"from {_relative(src_mod)} import {src_name} as {name}")

    out: list[str] = [
        '"""Auto-generated type stub — do not edit by hand."""',
        "",
        "# This stub mirrors the runtime ``statspai`` namespace so that IDEs and",
        "# static type-checkers can statically see every public symbol, even the",
        "# lazy-loaded ones routed through ``__getattr__``. The real bindings",
        "# happen at runtime per src/statspai/__init__.py.",
        "#",
        "# Regenerate with: python3 scripts/generate_stub.py",
        "",
    ]
    for sig in CONSTANTS.values():
        out.append(sig)
    out.append("")
    out.append("# Submodule re-exports (preserve ``sp.X.Y`` access)")
    out.extend(sorted(submodule_lines))
    out.append("")
    out.append("# Public function / class re-exports")
    out.extend(sorted(leaf_lines))
    out.append("")
    out.append("__all__: list[str]")
    out.append("")
    if unresolved:
        out.append("# Names without a discoverable source module — declared as Any")
        out.append("from typing import Any")
        for n in sorted(unresolved):
            out.append(f"{n}: Any")
        out.append("")

    counts = {
        "submodules": len(submodule_lines),
        "leaves": len(leaf_lines),
        "unresolved": len(unresolved),
    }
    return "\n".join(out), counts


def declarations(text: str) -> set[str]:
    """The stub's declarations, independent of how they are formatted."""
    out: set[str] = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            for a in node.names:
                out.add(f"from {mod} import {a.name} as {a.asname or a.name}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(f"{node.target.id}: {ast.unparse(node.annotation)}")
    return out


def stale(sp: types.ModuleType, target: Path = ROOT / "__init__.pyi") -> dict:
    """Declarations missing from / extra in the committed stub."""
    want = declarations(build_stub(sp)[0])
    have = declarations(target.read_text(encoding="utf-8"))
    return {"missing": sorted(want - have), "extra": sorted(have - want)}


def main() -> None:
    # Force a clean import so ``__all__`` reflects current source.
    for k in list(sys.modules):
        if k.startswith("statspai"):
            del sys.modules[k]
    import statspai as sp  # noqa: E402

    target = ROOT / "__init__.pyi"
    if "--check" in sys.argv[1:]:
        diff = stale(sp, target)
        if diff["missing"] or diff["extra"]:
            print(
                f"{target} is stale: {len(diff['missing'])} declarations "
                f"missing, {len(diff['extra'])} extra; run "
                "python3 scripts/generate_stub.py"
            )
            for line in diff["missing"][:20]:
                print(f"  + {line}")
            for line in diff["extra"][:20]:
                print(f"  - {line}")
            sys.exit(1)
        print(f"{target} is current.")
        return
    text, counts = build_stub(sp)
    target.write_text(text, encoding="utf-8")
    print(
        f"submodules: {counts['submodules']}\n"
        f"leaves:     {counts['leaves']}\n"
        f"unresolved: {counts['unresolved']}\n"
        f"wrote:      {target}"
    )


if __name__ == "__main__":
    main()
