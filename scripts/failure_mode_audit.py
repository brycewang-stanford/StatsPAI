#!/usr/bin/env python3
"""Audit the *consistency* between StatsPAI's declared failure modes and its
runtime reality.

StatsPAI's registry carries a rich, agent-native failure-mode contract: each
``FunctionSpec`` may declare ``failure_modes`` (``FailureMode(symptom,
exception, remedy, alternative)``) and a ranked ``alternatives`` list. The
:mod:`statspai.exceptions` taxonomy gives every estimator a typed,
``recovery_hint``-carrying way to *raise* those failures. This script checks
that the two halves line up, and surfaces where the runtime still lags the
declaration. It is both a CI gate and a trust artifact: a reader can run it to
confirm that what the package *promises* on failure is what it actually does.

Three checks
------------
1. **Dangling alternatives** — every ``sp.<name>`` referenced by a
   ``FailureMode.alternative`` or a spec's ``alternatives`` list must resolve to
   a *registered* function. A dangling pointer sends an agent (or human) to a
   function that does not exist. This is a hard error (``--check`` fails).
2. **Unknown exception names** — every ``FailureMode.exception`` must name a
   real, catchable class: a StatsPAI exception/warning, a builtin exception, or
   be intentionally blank. A typo here means an agent's ``except`` clause never
   fires. Hard error.
3. **Cryptic-raise hotspots** — for the flagship estimator modules, count how
   many ``raise`` sites still use bare ``ValueError`` / ``RuntimeError`` /
   ``KeyError`` / ``Exception`` versus the typed :class:`statspai.StatsPAIError`
   taxonomy. This is the backlog for the actionable-error migration campaign;
   reported as guidance, not a hard failure.

Usage
-----
::

    python scripts/failure_mode_audit.py            # human-readable report
    python scripts/failure_mode_audit.py --json     # machine-readable
    python scripts/failure_mode_audit.py --check     # exit 1 on any hard error
    python scripts/failure_mode_audit.py --markdown out.md
"""
from __future__ import annotations

import argparse
import builtins
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "statspai"

# Flagship estimator modules whose runtime raise-hygiene we track. These are the
# families CLAUDE.md §5 commits to aligning with Stata/R first, so they are the
# priority surface for typed, actionable failures.
CORE_MODULES: Dict[str, List[str]] = {
    "regression": ["regression"],
    "did": ["did"],
    "iv": ["iv"],
    "rd": ["rd"],
    "synth": ["synth"],
    "dml": ["dml"],
    "panel": ["panel"],
    "matching": ["matching"],
}

# Bare exception types that carry no recovery payload. A ``raise`` of one of
# these on an *assumption / data / identification* failure is a migration
# target — it should become a typed StatsPAIError subclass.
CRYPTIC_RAISE_RE = re.compile(
    r"\braise\s+(ValueError|RuntimeError|KeyError|Exception|TypeError|"
    r"IndexError|AssertionError)\b"
)
TYPED_RAISE_RE = re.compile(
    r"\braise\s+(StatsPAIError|AssumptionViolation|IdentificationFailure|"
    r"DataInsufficient|ConvergenceFailure|NumericalInstability|"
    r"MethodIncompatibility)\b"
)
# A soft, payload-carrying diagnostic. Counts as "good" runtime hygiene.
SOFT_WARN_RE = re.compile(
    r"(spx?\.warn\(|warnings\.warn\([^)]*(AssumptionWarning|ConvergenceWarning|"
    r"StatsPAIWarning))"
)


@dataclass
class AuditReport:
    n_specs: int = 0
    n_specs_with_failure_modes: int = 0
    n_specs_with_alternatives: int = 0
    dangling_alternatives: List[Dict[str, str]] = field(default_factory=list)
    imprecise_alternatives: List[Dict[str, str]] = field(default_factory=list)
    unknown_exceptions: List[Dict[str, str]] = field(default_factory=list)
    taxonomy_gaps: List[Dict[str, str]] = field(default_factory=list)
    cryptic_hotspots: List[Dict[str, object]] = field(default_factory=list)

    @property
    def hard_error_count(self) -> int:
        # taxonomy_gaps are soft (a real class, just outside the central
        # taxonomy) — reported but not a CI failure on their own.
        return len(self.dangling_alternatives) + len(self.unknown_exceptions)


def _load_registry():
    sys.path.insert(0, str(SRC.parent))
    import statspai.registry as R  # noqa: E402

    if hasattr(R, "_ensure_full_registry"):
        R._ensure_full_registry()
    return R._REGISTRY


def _taxonomy_names() -> set:
    """The central, agent-catchable taxonomy in :mod:`statspai.exceptions`."""
    import statspai.exceptions as spx

    names = set(getattr(spx, "__all__", []))
    names.add("WorkflowDegradedWarning")
    return names


def _valid_exception_names() -> set:
    """All exception / warning identifiers a FailureMode may legitimately name:
    the central taxonomy plus any genuine builtin exception."""
    names = _taxonomy_names()
    names.update(
        n
        for n in dir(builtins)
        if isinstance(getattr(builtins, n), type)
        and issubclass(getattr(builtins, n), BaseException)
    )
    return names


def _resolve_exception_class(name: str):
    """Resolve an exception identifier to its class via the public ``statspai``
    namespace (covers re-exported names like ``IdentificationError``). Returns
    the class or ``None``."""
    import statspai as sp

    obj = getattr(sp, name, None)
    if isinstance(obj, type) and issubclass(obj, BaseException):
        return obj
    return None


def _is_taxonomy_catchable(name: str) -> bool:
    """True if ``name`` resolves to a class that an agent can catch through the
    central taxonomy (a :class:`statspai.StatsPAIError` /
    :class:`statspai.StatsPAIWarning` subclass), even if it lives in another
    module. This is the property that actually matters for agent recovery."""
    import statspai.exceptions as spx

    cls = _resolve_exception_class(name)
    if cls is None:
        return False
    return issubclass(cls, (spx.StatsPAIError, spx.StatsPAIWarning))


def _resolvable_elsewhere(name: str) -> bool:
    """True if ``name`` is a real exception class reachable from statspai but
    NOT catchable through the central taxonomy (a genuine unification target)."""
    cls = _resolve_exception_class(name)
    return cls is not None and not _is_taxonomy_catchable(name)


def _is_informational_sentinel(exc: str) -> bool:
    """Some FailureModes use the ``exception`` field as a free-text marker —
    ``"(none — informational)"``, ``"(none — returns converged=False)"`` — to
    say *this failure surfaces as a diagnostic, not a raise*. Treat those as a
    legitimate "no exception" sentinel rather than a typo."""
    e = exc.strip().lower()
    return (not e) or e.startswith("(") or e.startswith("none")


# Capture the full dotted path after ``sp.`` so ``sp.fast.feols`` is resolved
# as a namespace attribute, not mistaken for a top-level function ``fast``.
_ALT_NAME_RE = re.compile(r"sp\.([A-Za-z_][A-Za-z0-9_.]*)")


def _extract_alt_names(text: str) -> List[str]:
    """Pull every ``sp.<path>`` target out of an alternative string.

    Handles ``"sp.iv(..., method='liml')"`` (→ ``iv``), bare ``"sp.synth"``,
    and dotted ``"sp.fast.feols"`` (→ ``fast.feols``). Non-``sp.`` free text
    (e.g. "switch to a panel design") yields nothing and is advisory.
    """
    return [m.rstrip(".") for m in _ALT_NAME_RE.findall(text or "")]


def _alt_status(path: str, registered: set, statspai_mod) -> str:
    """Classify an ``sp.<path>`` alternative.

    Returns one of:
    * ``"function"`` — a registered estimator or a callable namespace attribute
      (e.g. ``sp.fast.feols``); an agent can call it directly. Good.
    * ``"module"`` — resolves only to a bare submodule (e.g. ``sp.bounds``); the
      pointer is *imprecise* — it should name a concrete callable. Soft.
    * ``"missing"`` — does not resolve at all. Hard error.
    """
    import types

    if path.split(".")[0] in registered:
        return "function"
    obj = statspai_mod
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return "missing"
    if isinstance(obj, types.ModuleType):
        return "module"
    return "function" if callable(obj) else "module"


def audit_registry(registry) -> AuditReport:
    import statspai as statspai_mod

    rep = AuditReport()
    valid_names = set(registry.keys())
    valid_exc = _valid_exception_names()

    rep.n_specs = len(registry)
    for spec in registry.values():
        fms = getattr(spec, "failure_modes", None) or []
        alts = getattr(spec, "alternatives", None) or []
        if fms:
            rep.n_specs_with_failure_modes += 1
        if alts:
            rep.n_specs_with_alternatives += 1

        # Check 1 + 2: failure modes
        for fm in fms:
            exc = (getattr(fm, "exception", "") or "").strip()
            exc_tail = exc.split(".")[-1] if exc else ""
            if not _is_informational_sentinel(exc):
                if exc_tail in valid_exc or _is_taxonomy_catchable(exc_tail):
                    pass  # central taxonomy, a builtin, or a catchable subclass
                elif _resolvable_elsewhere(exc_tail):
                    rep.taxonomy_gaps.append(
                        {
                            "function": spec.name,
                            "exception": exc,
                            "symptom": getattr(fm, "symptom", ""),
                        }
                    )
                else:
                    rep.unknown_exceptions.append(
                        {
                            "function": spec.name,
                            "exception": exc,
                            "symptom": getattr(fm, "symptom", ""),
                        }
                    )
            _classify_alts(
                rep,
                spec.name,
                "failure_mode.alternative",
                getattr(fm, "alternative", ""),
                valid_names,
                statspai_mod,
            )

        # Check 1: top-level alternatives list
        for alt_raw in alts:
            _classify_alts(
                rep, spec.name, "alternatives", alt_raw, valid_names, statspai_mod
            )

    return rep


def _classify_alts(rep, fn_name, source, raw, valid_names, statspai_mod) -> None:
    for alt in _extract_alt_names(raw):
        status = _alt_status(alt, valid_names, statspai_mod)
        if status == "missing":
            rep.dangling_alternatives.append(
                {"function": fn_name, "source": source, "target": alt, "raw": raw}
            )
        elif status == "module":
            rep.imprecise_alternatives.append(
                {"function": fn_name, "source": source, "target": alt, "raw": raw}
            )


def _iter_module_files(module_dirs: List[str]):
    for md in module_dirs:
        base = SRC / md
        if base.is_dir():
            yield from base.rglob("*.py")
        elif (SRC / f"{md}.py").is_file():
            yield SRC / f"{md}.py"


def audit_runtime_hygiene(rep: AuditReport) -> None:
    """Check 3: per-family ratio of cryptic vs typed raises."""
    for family, dirs in CORE_MODULES.items():
        cryptic = typed = soft = 0
        worst: List[Tuple[str, int]] = []
        for f in _iter_module_files(dirs):
            if "__pycache__" in f.parts:
                continue
            text = f.read_text(encoding="utf-8")
            c = len(CRYPTIC_RAISE_RE.findall(text))
            t = len(TYPED_RAISE_RE.findall(text))
            s = len(SOFT_WARN_RE.findall(text))
            cryptic += c
            typed += t
            soft += s
            if c:
                worst.append((str(f.relative_to(REPO_ROOT)), c))
        worst.sort(key=lambda x: -x[1])
        total = cryptic + typed
        rep.cryptic_hotspots.append(
            {
                "family": family,
                "cryptic_raises": cryptic,
                "typed_raises": typed,
                "soft_warns": soft,
                "typed_ratio": round(typed / total, 3) if total else None,
                "top_files": worst[:5],
            }
        )


def render_markdown(rep: AuditReport) -> str:
    lines: List[str] = []
    lines.append("# StatsPAI failure-mode consistency audit\n")
    lines.append(
        f"- registry functions: **{rep.n_specs}**  \n"
        f"- with declared `failure_modes`: **{rep.n_specs_with_failure_modes}**  \n"
        f"- with declared `alternatives`: **{rep.n_specs_with_alternatives}**\n"
    )
    lines.append("## Hard errors (must be zero)\n")
    lines.append(
        f"- dangling `sp.*` alternatives: **{len(rep.dangling_alternatives)}**\n"
        f"- unknown exception names: **{len(rep.unknown_exceptions)}**\n"
        f"- imprecise (module-only) alternatives (soft): "
        f"{len(rep.imprecise_alternatives)}\n"
        f"- taxonomy gaps (soft): {len(rep.taxonomy_gaps)}\n"
    )
    if rep.dangling_alternatives:
        lines.append("\n### Dangling alternatives\n")
        for d in rep.dangling_alternatives:
            lines.append(
                f"- `{d['function']}` → `sp.{d['target']}` "
                f"(via {d['source']}: `{d['raw']}`) — **not registered**"
            )
    if rep.imprecise_alternatives:
        lines.append(
            "\n## Imprecise alternatives (soft — points at a module, not a "
            "callable)\n"
        )
        for d in rep.imprecise_alternatives:
            lines.append(
                f"- `{d['function']}` → `sp.{d['target']}` "
                f"(via {d['source']}) — names a submodule; point at a concrete "
                f"`sp.<function>` instead"
            )
    if rep.unknown_exceptions:
        lines.append("\n### Unknown exception names\n")
        for d in rep.unknown_exceptions:
            lines.append(
                f"- `{d['function']}`: `{d['exception']}` — not a known "
                f"exception/warning class (symptom: {d['symptom']!r})"
            )
    if rep.taxonomy_gaps:
        lines.append(
            "\n## Taxonomy gaps (soft — real class, outside the central taxonomy)\n"
        )
        lines.append(
            "These name a real exception that lives *outside* "
            "`statspai.exceptions`, so `except sp.StatsPAIError` / "
            "`except sp.IdentificationFailure` will not catch them. Unify them "
            "into the taxonomy (or alias) so agents can catch by kind.\n"
        )
        for d in rep.taxonomy_gaps:
            lines.append(
                f"- `{d['function']}`: `{d['exception']}` "
                f"(symptom: {d['symptom']!r})"
            )
    lines.append("\n## Runtime raise-hygiene by family (migration backlog)\n")
    lines.append("| family | typed | cryptic | soft warns | typed-ratio |")
    lines.append("| --- | --: | --: | --: | --: |")
    for h in rep.cryptic_hotspots:
        lines.append(
            f"| {h['family']} | {h['typed_raises']} | {h['cryptic_raises']} | "
            f"{h['soft_warns']} | "
            f"{h['typed_ratio'] if h['typed_ratio'] is not None else '—'} |"
        )
    lines.append(
        "\n*typed-ratio = typed StatsPAIError raises / (typed + cryptic). "
        "Higher is better; the campaign raises it family by family.*\n"
    )
    for h in rep.cryptic_hotspots:
        if h["top_files"]:
            lines.append(f"\n### {h['family']}: top cryptic-raise files")
            for path, n in h["top_files"]:
                lines.append(f"- `{path}` — {n} cryptic raises")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--markdown", metavar="PATH", help="write markdown to PATH")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any hard error (dangling alt / unknown exception)",
    )
    args = ap.parse_args(argv)

    registry = _load_registry()
    rep = audit_registry(registry)
    audit_runtime_hygiene(rep)

    if args.json:
        print(
            json.dumps(
                {
                    "n_specs": rep.n_specs,
                    "n_specs_with_failure_modes": rep.n_specs_with_failure_modes,
                    "n_specs_with_alternatives": rep.n_specs_with_alternatives,
                    "dangling_alternatives": rep.dangling_alternatives,
                    "imprecise_alternatives": rep.imprecise_alternatives,
                    "unknown_exceptions": rep.unknown_exceptions,
                    "taxonomy_gaps": rep.taxonomy_gaps,
                    "cryptic_hotspots": rep.cryptic_hotspots,
                    "hard_error_count": rep.hard_error_count,
                },
                indent=2,
            )
        )
    else:
        md = render_markdown(rep)
        if args.markdown:
            Path(args.markdown).write_text(md, encoding="utf-8")
            print(f"wrote {args.markdown}")
        else:
            print(md)

    if args.check and rep.hard_error_count:
        print(
            f"\nFAIL: {rep.hard_error_count} hard error(s) "
            f"({len(rep.dangling_alternatives)} dangling, "
            f"{len(rep.unknown_exceptions)} unknown exception).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
