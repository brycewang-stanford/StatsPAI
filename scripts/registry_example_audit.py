"""Static-check the registry's one-line ``example=`` strings.

``sp.describe_function(name)["example"]`` is often the only call syntax an
agent sees before dispatching. Unlike docstring ``Examples`` blocks — which
:mod:`scripts.check_example_execution` actually runs — these one-liners
reference undefined names by design (``sp.modelsummary([r1, r2, r3])``), so
they cannot be executed. They can still be checked structurally, and the
three failure modes below are all silent today:

1. **Syntax** — the string does not parse at all.
2. **Unknown callee** — ``sp.<name>`` that no longer exists (renamed or
   removed), so the example dispatches into an AttributeError.
3. **Unknown keyword** — a keyword argument absent from the real
   signature. This is the common one: ``sp.psm`` takes ``y, d, X`` while
   its sibling ``sp.ipw`` takes ``y, treat, covariates``, and an example
   written for one reads plausibly for the other.

Coverage is deliberately partial and fails open. Checks 1 and 2 apply to
every example. Check 3 applies to functions with a closed signature, and
to those whose ``**kwargs`` is the ``@accepts_aliases`` mechanism (whose
accepted names are recoverable from ``__statspai_aliases__``). It stands
down for functions that forward ``**kwargs`` to a delegate — ``sp.psm``
hands them to ``sp.match`` — because the delegate is not knowable from
the call site. Those still reject unknown keywords at runtime; they are
simply outside what a static pass can promise.

Usage
-----
python scripts/registry_example_audit.py             # report
python scripts/registry_example_audit.py --check     # exit 1 on any finding
python scripts/registry_example_audit.py --max-findings N
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from typing import Any, Dict, List, Tuple

import statspai as sp
from statspai import registry as R

#: Public functions whose ``**kwargs`` is forwarded to another estimator, so
#: the accepted names are the delegate's, not knowable from this signature.
_FORWARDS_KWARGS = {
    "psm": "sp.match",
    "cs_jackknife": "sp.callaway_santanna",
}


def _accepted_keywords(target: Any, signature: inspect.Signature) -> Any:
    """The keyword names *target* really accepts, or None if unknowable.

    A bare ``**kwargs`` means anything goes and the check must stand down.
    Most of this package's ``**kwargs`` is not bare, though: the
    ``@accepts_aliases`` decorator adds it to accept alternative spellings
    while still rejecting genuinely unknown names at runtime, and it
    records what it accepts on ``__statspai_aliases__``. Reading that back
    keeps the check alive for the majority of the public surface — which
    is exactly where a stale example would otherwise hide.
    """
    names = set(signature.parameters)
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if not has_var_keyword:
        return names

    aliases = getattr(target, "__statspai_aliases__", None)
    if not aliases:
        return None  # genuinely open-ended
    # A function can carry both: ``@accepts_aliases`` on the outside and its
    # own ``**kwargs`` forwarded to a delegate. The signature cannot tell that
    # apart from a ``**kwargs`` the function validates itself (``sp.regress``
    # rejects unknown names at runtime), so forwarders are listed by name.
    if getattr(target, "__name__", "") in _FORWARDS_KWARGS:
        return None
    names.discard("kwargs")
    return names | set(aliases)


def _callee_name(node: ast.Call) -> str:
    """``sp.foo(...)`` -> ``"foo"``; ``foo(...)`` -> ``"foo"``; else ``""``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_statspai_call(node: ast.Call) -> bool:
    """True for ``sp.foo(...)`` / ``statspai.foo(...)`` only.

    Method calls on intermediate objects (``result.summary()``) are out of
    scope: the receiver's type is not knowable statically.
    """
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and (func.value.id in {"sp", "statspai"})
    )


def audit_one(name: str, example: str) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    try:
        tree = ast.parse(example)
    except SyntaxError as exc:
        return [("syntax", f"{example!r} does not parse ({exc.msg})")]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_statspai_call(node):
            continue
        callee = _callee_name(node)
        target = getattr(sp, callee, None)
        if target is None:
            findings.append(("unknown-callee", f"sp.{callee} does not exist"))
            continue
        if not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        valid = _accepted_keywords(target, signature)
        if valid is None:
            continue
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat in the example
                continue
            if kw.arg not in valid:
                findings.append(
                    (
                        "unknown-keyword",
                        f"sp.{callee}(..., {kw.arg}=...) — not in signature "
                        f"({', '.join(sorted(valid)[:8])}...)",
                    )
                )
    return findings


def collect() -> Dict[str, List[Tuple[str, str]]]:
    R._ensure_full_registry()
    out: Dict[str, List[Tuple[str, str]]] = {}
    for name, spec in R._REGISTRY.items():
        example = getattr(spec, "example", "") or ""
        if not example.strip():
            continue
        findings = audit_one(name, example)
        if findings:
            out[name] = findings
    return out


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-findings", type=int, default=0)
    args = parser.parse_args(argv)

    findings = collect()
    total = sum(len(v) for v in findings.values())
    for name in sorted(findings):
        for kind, detail in findings[name]:
            print(f"{name}: [{kind}] {detail}")
    print(f"\n{total} finding(s) across {len(findings)} registry entr(ies).")

    if args.check and total > args.max_findings:
        print(
            f"FAIL: {total} > --max-findings {args.max_findings}. "
            "Fix the registry example so agents are not handed a call that "
            "cannot work.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
