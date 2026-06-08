#!/usr/bin/env python3
"""Tier-D validation-evidence classifier for the StatsPAI registry.

Part of the P1 "Tier D analytic special-cases" campaign. This is a *read-only*
diagnostic: it never imports or runs estimators with side effects beyond
``sp.list_functions`` / ``sp.describe_function`` and a static scan of ``tests/``.

It answers one question per registered function:

    *Does this estimator already have a real numerical-assertion test, or does
    it need an analytic / known-DGP recovery test (a "Tier D" special case)?*

Evidence ladder (strongest first)
---------------------------------
- ``reference`` : referenced under tests/reference_parity, external_parity,
                  r_parity, orig_parity, or stata_parity (Tier A/B already).
- ``anchored``  : a tolerance/closeness assertion (assert_allclose, approx,
                  ``abs(... - ...) < tol``, ``rel <``, "recover", ...) lives in
                  the enclosing test body of a call site — a known-truth guard.
- ``weak``      : an assertion exists in the enclosing test body, but only a
                  boolean / shape / not-None check (no known-truth anchor).
- ``smoke``     : referenced in tests but no assertion in the enclosing body.
- ``untested``  : not referenced by any test file.

Tier D worklist (estimator-like, not already certified/validated):
- **P1** (floor)   : ``smoke`` / ``untested`` — no numeric guard at all.
- **P2** (upgrade) : ``weak`` — has an assert but no known-truth anchor.

Estimator-like excludes infra/presentation categories and name patterns
(CamelCase result classes, ``*plot``, ``*_report``, ``*_to_latex``, ``*_simulate`` ...).

Usage
-----
    python scripts/tierd_classify.py report                 # summary tables
    python scripts/tierd_classify.py worklist               # candidate list (md)
    python scripts/tierd_classify.py worklist --priority P1
    python scripts/tierd_classify.py worklist --category causal
    python scripts/tierd_classify.py json > out.json        # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Directories whose mere presence of the function name counts as Tier A/B
# (cross-language parity or published-replication evidence).
REFERENCE_DIRS = (
    "reference_parity",
    "external_parity",
    "r_parity",
    "orig_parity",
    "stata_parity",
)

# Categories that are infrastructure / presentation, not numeric estimators.
# Tier D analytic special-cases do not apply to these.
NON_ESTIMATOR_CATEGORIES = frozenset(
    {
        "output",
        "plots",
        "utils",
        "core",
        "agent",
        "workflow",
        "smart",
        "validation",
        "datasets",
        "compat",
        "experimental_infra",
    }
)

# Validation grades that already carry strong evidence — out of Tier D scope.
STRONG_GRADES = frozenset({"certified", "validated", "deprecated"})

# Name-based exclusions: result classes (CamelCase) and presentation / IO /
# comparison helpers that are not numeric estimators even when their registry
# category is an estimator family. Analytic special-cases don't apply to these.
_CLASS_NAME_RE = re.compile(r"^[A-Z]")  # NotchResult, RDMultiResult, ...
_PRESENTATION_RE = re.compile(
    r"(plot$|^plot_|_plot_|plotdensity|_map$|_report|_summary$|_table$|_to_|"
    r"_dashboard|_compare$|^compare_|_examples?$|_card$|marginsplot|psplot|"
    r"_to_latex|_to_word|_to_excel|_to_markdown|_to_file|"
    r"_simulate$|_positions$|^dag_example)"
)


def _is_presentation_name(name):
    return bool(_CLASS_NAME_RE.match(name) or _PRESENTATION_RE.search(name))


ASSERT_RE = re.compile(
    r"\bassert\b|np\.testing\.|assert_allclose|pytest\.approx|\.approx\b"
)
# "Anchored" assertion idioms: the test compares an estimate to a numeric truth
# within a tolerance, rather than only checking a boolean / shape / not-None.
ANCHORED_RE = re.compile(
    r"assert_allclose|assert_almost_equal|approx\(|"
    r"<\s*tol|rtol|atol|abs\([^)]*-[^)]*\)\s*<|"
    r"\brel\b\s*<|<\s*0?\.\d|within|recover"
)
_DEF_RE = re.compile(r"^(\s*)def\s")


def _enclosing_def_body(lines, call_idx):
    """Return the (start, end) line range of the test-function enclosing a call.

    Walks back to the nearest ``def`` header, then forward to the next ``def``
    at the same-or-shallower indent. Falls back to a +/-12 line window if no
    enclosing ``def`` is found (e.g. module-level smoke).
    """
    start = None
    indent = None
    for j in range(call_idx, -1, -1):
        m = _DEF_RE.match(lines[j])
        if m:
            start = j
            indent = len(m.group(1))
            break
    if start is None:
        return max(0, call_idx - 12), min(len(lines), call_idx + 13)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _DEF_RE.match(lines[j])
        if m and len(m.group(1)) <= indent:
            end = j
            break
    return start, end


def _assert_kind_in_enclosing_def(lines, call_idx):
    """Return 'anchored' | 'weak' | None for the enclosing test body.

    'anchored' = a tolerance/closeness assertion (recovery against a numeric
    truth); 'weak' = an assertion exists but only boolean/shape/not-None; None
    = no assertion at all (smoke).
    """
    lo, hi = _enclosing_def_body(lines, call_idx)
    has_assert = False
    for k in range(lo, hi):
        ln = lines[k]
        if ANCHORED_RE.search(ln):
            return "anchored"
        if ASSERT_RE.search(ln):
            has_assert = True
    return "weak" if has_assert else None


def _iter_test_files():
    for p in TESTS.rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p
    # parity files are not always named test_*.py
    for p in TESTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        if any(d in p.parts for d in REFERENCE_DIRS) and p.suffix == ".py":
            yield p


def _build_test_index():
    """Return (reference_hits, asserted_hits, smoke_hits) as name->set(files)."""
    files = sorted(set(_iter_test_files()))
    # Pre-read every file once.
    contents = {}
    for f in files:
        try:
            contents[f] = f.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

    reference = defaultdict(set)
    anchored = defaultdict(set)
    asserted = defaultdict(set)  # weak (boolean/shape) asserts
    smoke = defaultdict(set)

    # call patterns: sp.name(  |  statspai.name(  |  .name(  |  name(
    # A preceding "." is allowed (sp.name / result.method); a preceding word
    # char is rejected so we don't match a longer identifier ending in `name`
    # (e.g. `xtnbreg` must not match `nbreg`, `my_did` must not match `did`).
    def call_re(name):
        return re.compile(rf"(?<![\w]){re.escape(name)}\s*\(")

    # Cache compiled regexes lazily per name during scan.
    return files, contents, reference, anchored, asserted, smoke, call_re


def classify(names_meta):
    """names_meta: dict name -> {category, validation_status}. Returns records."""
    files, contents, reference, anchored, asserted, smoke, call_re = _build_test_index()

    # Build a single regex alternation is too broad; iterate names but reuse
    # the pre-read line lists. For ~1000 names x ~600 files this is a few
    # seconds — acceptable for an offline diagnostic.
    name_re = {n: call_re(n) for n in names_meta}

    for f, lines in contents.items():
        is_ref = any(d in f.parts for d in REFERENCE_DIRS)
        joined = "\n".join(lines)
        # quick membership filter to skip names absent from the file
        present = [n for n in names_meta if n in joined]
        for n in present:
            rgx = name_re[n]
            hit_lines = [i for i, ln in enumerate(lines) if rgx.search(ln)]
            if not hit_lines:
                continue
            if is_ref:
                reference[n].add(f)
                continue
            # Grade the strongest assertion across this file's call sites.
            kinds = {_assert_kind_in_enclosing_def(lines, i) for i in hit_lines}
            if "anchored" in kinds:
                anchored[n].add(f)
            elif "weak" in kinds:
                asserted[n].add(f)
            else:
                smoke[n].add(f)

    records = []
    for n, meta in names_meta.items():
        cat = meta["category"]
        grade = meta["validation_status"]
        if reference[n]:
            evidence = "reference"
        elif anchored[n]:
            evidence = "anchored"
        elif asserted[n]:
            evidence = "weak"
        elif smoke[n]:
            evidence = "smoke"
        else:
            evidence = "untested"
        is_estimator = (
            cat not in NON_ESTIMATOR_CATEGORIES and not _is_presentation_name(n)
        )
        # Tier D worklist priorities:
        #   P1 (floor)    : no numeric guard at all  -> smoke / untested
        #   P2 (upgrade)  : has only a weak/shape assert, no known-truth anchor
        needs_tierd = (
            is_estimator
            and grade not in STRONG_GRADES
            and evidence in ("smoke", "untested", "weak")
        )
        priority = None
        if needs_tierd:
            priority = "P1" if evidence in ("smoke", "untested") else "P2"
        records.append(
            {
                "name": n,
                "category": cat,
                "grade": grade,
                "evidence": evidence,
                "n_ref_files": len(reference[n]),
                "n_anchored_files": len(anchored[n]),
                "n_weak_files": len(asserted[n]),
                "n_smoke_files": len(smoke[n]),
                "is_estimator": is_estimator,
                "needs_tierd": needs_tierd,
                "priority": priority,
            }
        )
    return records


def load_registry():
    import statspai as sp

    meta = {}
    for name in sp.list_functions():
        try:
            d = sp.describe_function(name)
        except Exception:  # pragma: no cover - defensive
            continue
        meta[name] = {
            "category": d.get("category") or "?",
            "validation_status": d.get("validation_status") or "?",
        }
    return meta


def cmd_report(records, args):
    n = len(records)
    print(f"# Tier-D classification — {n} registered functions\n")
    ev = Counter(r["evidence"] for r in records)
    print("## Evidence distribution (all functions)")
    for k in ("reference", "anchored", "weak", "smoke", "untested"):
        print(f"  {k:10s} {ev.get(k,0):4d}")
    cand = [r for r in records if r["needs_tierd"]]
    p1 = [r for r in cand if r["priority"] == "P1"]
    p2 = [r for r in cand if r["priority"] == "P2"]
    print(
        f"\n## Tier D worklist: {len(cand)} estimator-like fns "
        f"({len(p1)} P1 zero-guard, {len(p2)} P2 weak-assert upgrade)\n"
    )
    print("### P1 (no numeric guard) by category")
    for cat, c in Counter(r["category"] for r in p1).most_common():
        print(f"  {cat:22s} {c:4d}")
    print("\n### P2 (weak assert, needs known-truth anchor) by category")
    for cat, c in Counter(r["category"] for r in p2).most_common():
        print(f"  {cat:22s} {c:4d}")


def cmd_worklist(records, args):
    cand = [r for r in records if r["needs_tierd"]]
    if args.category:
        cand = [r for r in cand if r["category"] == args.category]
    if args.priority:
        cand = [r for r in cand if r["priority"] == args.priority]
    cand.sort(key=lambda r: (r["priority"], r["category"], r["name"]))
    print(
        f"# Tier D worklist ({len(cand)} functions"
        + (f", category={args.category}" if args.category else "")
        + (f", {args.priority}" if args.priority else "")
        + ")\n"
    )
    print("| priority | function | category | grade | evidence |")
    print("|---|---|---|---|---|")
    for r in cand:
        print(
            f"| {r['priority']} | `{r['name']}` | {r['category']} "
            f"| {r['grade']} | {r['evidence']} |"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    wl = sub.add_parser("worklist")
    wl.add_argument("--category", default=None)
    wl.add_argument("--priority", default=None, choices=["P1", "P2"])
    sub.add_parser("json")
    args = ap.parse_args(argv)

    meta = load_registry()
    records = classify(meta)

    if args.cmd == "report":
        cmd_report(records, args)
    elif args.cmd == "worklist":
        cmd_worklist(records, args)
    elif args.cmd == "json":
        json.dump(records, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
