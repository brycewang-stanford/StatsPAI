"""Generate ``src/statspai/_source_failure_modes.py`` from ``raise`` sites.

A third of the callable registry entries had no ``failure_modes``: nothing
told an agent what the function raises or what to do next. For many of them
the function already says both, in code: it raises a typed
:class:`statspai.exceptions.StatsPAIError` with a ``recovery_hint``.

This script reads those ``raise`` statements out of each function's own body
and records them as failure modes::

    raise MethodIncompatibility(
        f"method must be 'iterative'; got {method!r}",
        recovery_hint="Use method='iterative'.",
        alternative_functions=["sp.panel"],
    )

becomes ``{symptom: "method must be 'iterative'; got <method>", exception:
"statspai.MethodIncompatibility", remedy: "Use method='iterative'.",
alternative: "sp.panel"}``.

Rules that keep the output a statement of fact:

1. Only a ``raise`` in the function's **own** source counts (after
   ``inspect.unwrap``); helpers it calls are not followed.
2. The exception must be a ``StatsPAIError`` subclass and must carry a
   ``recovery_hint`` made of string literals. A hint computed at run time is
   skipped: there is no remedy text to quote.
3. The message keeps its literal text; each interpolated value becomes
   ``<expression>``.
4. The registry only uses these where a function has no failure mode at all,
   hand-written or inherited (``registry._apply_source_failure_modes``), and
   tags them ``source`` in the card provenance.

Usage::

    python scripts/gen_source_failure_modes.py           # regenerate
    python scripts/gen_source_failure_modes.py --check   # fail on drift

``tests/test_source_failure_modes.py`` runs the check.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import pprint
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "src" / "statspai" / "_source_failure_modes.py"
MAX_PER_FUNCTION = 4
MAX_TEXT = 220

HEADER = '''"""Failure modes read from each function's own ``raise`` statements.

DO NOT EDIT BY HAND. Regenerate with::

    python scripts/gen_source_failure_modes.py

Each entry quotes a typed ``StatsPAIError`` the function raises together
with the ``recovery_hint`` written next to it. The registry uses an entry
only when the function has no hand-written or inherited failure mode.
"""

from __future__ import annotations

from typing import Dict, List

SOURCE_FAILURE_MODES: Dict[str, List[Dict[str, str]]] = '''


def _literal_text(node: Optional[ast.AST]) -> Optional[str]:
    """Text of a string literal (implicit concatenation included)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_text(node.left), _literal_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _template_text(node: Optional[ast.AST]) -> Optional[str]:
    """Message text with ``<expr>`` in place of each interpolated value."""
    plain = _literal_text(node)
    if plain is not None:
        return plain
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                parts.append(str(piece.value))
            elif isinstance(piece, ast.FormattedValue):
                expr = ast.unparse(piece.value)
                parts.append(f"<{expr}>" if len(expr) <= 40 else "<...>")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template_text(node.left), _template_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _clean(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 3].rstrip() + "..."
    return text


def _taxonomy() -> Dict[str, type]:
    from statspai import exceptions as exc

    return {
        name: obj
        for name, obj in vars(exc).items()
        if inspect.isclass(obj) and issubclass(obj, exc.StatsPAIError)
    }


def failure_modes_for(obj: Any, taxonomy: Dict[str, type]) -> List[Dict[str, str]]:
    """Failure modes stated by the ``raise`` sites in ``obj``'s own body."""
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(obj)))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        cls = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if cls not in taxonomy:
            continue
        keywords = {kw.arg: kw.value for kw in node.exc.keywords if kw.arg}
        remedy = _literal_text(keywords.get("recovery_hint"))
        message = _template_text(node.exc.args[0]) if node.exc.args else None
        if not remedy or not message:
            continue
        entry = {
            "symptom": _clean(message),
            "exception": f"statspai.{cls}",
            "remedy": _clean(remedy),
        }
        alts = keywords.get("alternative_functions")
        if isinstance(alts, (ast.List, ast.Tuple)) and alts.elts:
            first = _literal_text(alts.elts[0])
            if first:
                entry["alternative"] = first
        key = (entry["symptom"], entry["exception"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= MAX_PER_FUNCTION:
            break
    return out


def collect() -> Dict[str, List[Dict[str, str]]]:
    warnings.simplefilter("ignore")
    import statspai as sp
    from statspai import registry

    registry._ensure_full_registry()
    taxonomy = _taxonomy()
    table: Dict[str, List[Dict[str, str]]] = {}
    for name in sorted(registry._REGISTRY):
        spec = registry._REGISTRY[name]
        if getattr(spec, "_kind", None) == "class":
            continue
        obj = getattr(sp, name, None)
        if obj is None or inspect.isclass(obj) or not callable(obj):
            continue
        # Only functions with no failure mode of their own or inherited:
        # the rest never read this table, and shipping their entries would
        # triple the module for nothing.
        has_other = name not in registry._SOURCE_FAILURE_MODES_FILLED and bool(
            spec.agent_card(merge_inherited=True)["failure_modes"]
        )
        if has_other:
            continue
        modes = failure_modes_for(obj, taxonomy)
        if modes:
            table[name] = modes
    return table


def render(table: Dict[str, List[Dict[str, str]]]) -> str:
    body = pprint.pformat(table, width=80, sort_dicts=True)
    text = HEADER + body + "\n"
    try:
        import black
    except ImportError:  # the layout is cosmetic; --check compares the table
        return text
    return black.format_str(text, mode=black.Mode())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args(argv)
    text = render(collect())
    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if _normalise(current) != _normalise(text):
            print(
                "src/statspai/_source_failure_modes.py is stale; run "
                "python scripts/gen_source_failure_modes.py",
                file=sys.stderr,
            )
            return 1
        print("OK: source failure modes are current")
        return 0
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _normalise(text: str) -> Any:
    """The table itself, so a formatter's layout is not drift."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return ast.literal_eval(node.value)
    return text


if __name__ == "__main__":
    sys.exit(main())
