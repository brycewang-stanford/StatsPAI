"""Public-API signature house-style lint.

Stata's real genius is a *consistent grammar*: ``vce(cluster id)``, ``robust``,
``weights`` mean the same thing across two hundred commands.  StatsPAI exposes
~1,130 flat functions, so the risk is the opposite — the same idea spelled five
different ways (``robust`` / ``vcov`` / ``vce`` / ``se_type`` for the SE type;
``y`` / ``outcome`` / ``Y`` for the dependent variable).

This lint reads the canonical vocabulary from ``statspai._house_style`` and, by
*live introspection* of every public callable, reports where a parameter uses a
non-canonical spelling of a known theme.  Acknowledged false friends (spatial
``w``/``W``, mixed-model ``cov_type``, EconML-mirrored ``Y``/``T``) are excluded
so the signal stays trustworthy.

It is intentionally a **ratchet**, not a hard zero: the surface has historical
drift.  ``--check`` fails only if a theme's violation count rises above the
frozen baseline, so convergence can land incrementally without blocking work.

Usage
-----
::

    python scripts/signature_house_style.py            # human report
    python scripts/signature_house_style.py --json     # machine-readable
    python scripts/signature_house_style.py --details   # per-site list
    python scripts/signature_house_style.py --check     # CI ratchet
    python scripts/signature_house_style.py --update-baseline   # refreeze floor
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
BASELINE_PATH = Path(__file__).resolve().parent / "signature_house_style_baseline.json"

# Make ``import statspai`` resolve from the in-repo source without an install.
sys.path.insert(0, str(SRC_ROOT))


def _load_house_style():
    import statspai._house_style as hs  # noqa: WPS433 (local import by design)

    return hs


def _public_callables() -> Dict[str, Any]:
    """Return ``{name: callable}`` for the public function surface.

    Classes (result containers, namespaces) are skipped — the house-style
    contract here is about *function* signatures.  Anything that fails to
    resolve or introspect (optional-dependency modules, namespace objects) is
    silently dropped and counted as unresolved by the caller.
    """
    import statspai as sp

    out: Dict[str, Any] = {}
    for name in sp.list_functions():
        try:
            obj = getattr(sp, name)
        except Exception:  # pragma: no cover - optional dep import guards
            continue
        if obj is None or inspect.isclass(obj):
            continue
        if not callable(obj):
            continue
        out[name] = obj
    return out


def _signature_params(obj: Any) -> List[str]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    params = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        params.append(p.name)
    return params


def _robust_default_kind(obj: Any) -> str | None:
    """Classify the default of a ``robust`` parameter: bool / str / none."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    p = sig.parameters.get("robust")
    if p is None:
        return None
    default = p.default
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, str):
        return "str"
    if default is None:
        return "none"
    return "other"


def collect() -> Dict[str, Any]:
    hs = _load_house_style()
    alias_to_canonical = hs.alias_index()
    canonical_names = {str(t["canonical"]) for t in hs.THEMES.values()}
    # spelling -> theme key, for participation accounting.
    spelling_theme: Dict[str, str] = {}
    for key, theme in hs.THEMES.items():
        spelling_theme[str(theme["canonical"])] = key
        for alias in theme["aliases"]:  # type: ignore[union-attr]
            spelling_theme[str(alias)] = key

    callables = _public_callables()

    violations: List[Dict[str, Any]] = []
    acknowledged: List[Dict[str, Any]] = []
    # theme -> {"canonical": n, "legacy": n} over participating functions.
    participation: Dict[str, Counter] = defaultdict(Counter)
    robust_kinds: Counter = Counter()

    introspected = 0
    for name, obj in sorted(callables.items()):
        params = _signature_params(obj)
        if not params:
            continue
        introspected += 1
        module = getattr(obj, "__module__", "") or ""

        kind = _robust_default_kind(obj)
        if kind is not None:
            robust_kinds[kind] += 1

        for param in params:
            theme = spelling_theme.get(param)
            if theme is None:
                continue
            if hs.is_false_friend(param, name, module):
                acknowledged.append(
                    {"function": name, "param": param, "module": module}
                )
                continue
            if param in canonical_names:
                participation[theme]["canonical"] += 1
            else:
                participation[theme]["legacy"] += 1
                violations.append(
                    {
                        "function": name,
                        "param": param,
                        "theme": theme,
                        "found": param,
                        "canonical": alias_to_canonical.get(param, "?"),
                        "module": module,
                    }
                )

    by_theme = Counter(v["theme"] for v in violations)
    by_spelling = Counter(v["found"] for v in violations)

    coverage = {}
    for theme in hs.THEMES:
        counts = participation.get(theme, Counter())
        total = counts["canonical"] + counts["legacy"]
        pct = (counts["canonical"] / total * 100.0) if total else 100.0
        coverage[theme] = {
            "canonical": counts["canonical"],
            "legacy": counts["legacy"],
            "total": total,
            "canonical_pct": round(pct, 1),
        }

    return {
        "totals": {
            "public_callables": len(callables),
            "introspected": introspected,
            "violations": len(violations),
            "acknowledged_false_friends": len(acknowledged),
        },
        "by_theme": dict(sorted(by_theme.items())),
        "by_spelling": dict(
            sorted(by_spelling.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "coverage": coverage,
        "robust_default_kinds": dict(sorted(robust_kinds.items())),
        "violations": sorted(violations, key=lambda v: (v["theme"], v["function"])),
        "acknowledged": sorted(acknowledged, key=lambda v: (v["param"], v["function"])),
    }


def _baseline_floor() -> Dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.get("by_theme", {}).items()}


def check(report: Dict[str, Any]) -> int:
    floor = _baseline_floor()
    if not floor:
        print(
            "[signature_house_style] no baseline frozen yet; run "
            "--update-baseline first.",
            file=sys.stderr,
        )
        return 1
    observed = report["by_theme"]
    failures: List[str] = []
    for theme, ceiling in floor.items():
        seen = observed.get(theme, 0)
        if seen > ceiling:
            failures.append(f"{theme}: observed={seen} > baseline={ceiling}")
    # New themes that appear with violations but are absent from the floor.
    for theme, seen in observed.items():
        if theme not in floor and seen > 0:
            failures.append(f"{theme}: observed={seen} (no baseline entry)")
    if failures:
        print("[signature_house_style] REGRESSION", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        "[signature_house_style] OK - "
        f"{report['totals']['violations']} legacy-spelling sites "
        "(<= baseline across every theme)."
    )
    return 0


def update_baseline(report: Dict[str, Any]) -> int:
    payload = {
        "_comment": (
            "Frozen ceiling of legacy-spelling sites per theme for the "
            "signature house-style ratchet. Lower these as convergence lands; "
            "regenerate with `python scripts/signature_house_style.py "
            "--update-baseline`."
        ),
        "totals": report["totals"],
        "by_theme": report["by_theme"],
        "coverage": report["coverage"],
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[signature_house_style] baseline written to {BASELINE_PATH.name}")
    return 0


def render(report: Dict[str, Any]) -> str:
    t = report["totals"]
    lines: List[str] = []
    lines.append("StatsPAI signature house-style lint")
    lines.append("=" * 50)
    lines.append(f"Public callables          : {t['public_callables']}")
    lines.append(f"  introspected w/ params  : {t['introspected']}")
    lines.append(f"Legacy-spelling sites     : {t['violations']}")
    lines.append(f"Acknowledged false friends: {t['acknowledged_false_friends']}")
    lines.append("")
    lines.append("Canonical coverage by theme")
    lines.append("-" * 50)
    for theme, cov in report["coverage"].items():
        lines.append(
            f"  {theme:10s}: {cov['canonical_pct']:5.1f}% canonical "
            f"({cov['canonical']}/{cov['total']}; {cov['legacy']} legacy)"
        )
    lines.append("")
    lines.append("Legacy-spelling sites by theme")
    lines.append("-" * 50)
    for theme, n in sorted(report["by_theme"].items()):
        lines.append(f"  {theme:10s}: {n}")
    lines.append("")
    lines.append("Legacy spellings encountered")
    lines.append("-" * 50)
    for spelling, n in report["by_spelling"].items():
        lines.append(f"  {spelling:16s}: {n}")
    lines.append("")
    lines.append("`robust` parameter default types (overload hazard)")
    lines.append("-" * 50)
    for kind, n in report["robust_default_kinds"].items():
        lines.append(f"  {kind:8s}: {n}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--check", action="store_true", help="Run CI ratchet.")
    parser.add_argument(
        "--details", action="store_true", help="Include every site in --json."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Freeze the current per-theme counts as the ratchet floor.",
    )
    args = parser.parse_args(argv)

    report = collect()
    if args.update_baseline:
        return update_baseline(report)
    if args.check:
        return check(report)
    if args.json:
        payload = dict(report)
        if not args.details:
            payload.pop("violations", None)
            payload.pop("acknowledged", None)
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
