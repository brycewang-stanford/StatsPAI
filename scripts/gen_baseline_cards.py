"""Generate ``src/statspai/_baseline_cards.py`` from docstrings.

This is the L0 auto-enrichment pass described in
``docs/agent_cards_spec.md``.  For every registered function whose
``FunctionSpec`` is missing one of the easy Tier-B fields (``example``,
``reference``, ``tags``), we try to fill it mechanically from the
docstring or from the function name / category.

Three rules carry the whole design:

1. **Never overwrite curated content.**  The generated apply() helper
   only writes into a field if that field is currently empty on the
   spec.  Hand-written ``register(FunctionSpec(...))`` calls always
   win.

2. **Zero-hallucination references.**  The only reference strings we
   emit are bib keys.  A bib key gets emitted iff (a) the docstring
   contains an explicit ``[@<bib_key>]`` marker and (b) that key
   exists in ``paper.bib``.  No DOI / no author-year guessing — see
   §10 in ``CLAUDE.md``.

3. **Tag vocabulary stays conservative.**  We pull from a controlled
   list that matches what already exists in the curated specs (see
   ``CURATED_TAG_VOCAB`` below).  New tags must be added explicitly.

Usage
-----
::

    python scripts/gen_baseline_cards.py            # regenerate the module
    python scripts/gen_baseline_cards.py --dry-run  # show diff w/o writing

Run this script after editing docstrings or after registering new
functions.  CI does NOT auto-run it (deterministic codegen lives in
the repo and is reviewed in the diff).
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "src" / "statspai" / "_baseline_cards.py"
PAPER_BIB = REPO_ROOT / "paper.bib"


# --------------------------------------------------------------------- #
#  Tag vocabulary — stay close to the curated registry's existing tags.
# --------------------------------------------------------------------- #
#
# Each entry is (substring_in_name, tag_to_emit).  Order matters — the
# first match wins.  Always paired with the category tag.

NAME_TAG_RULES: Tuple[Tuple[str, str], ...] = (
    # DiD family
    ("callaway_santanna", "did"),
    ("sun_abraham", "did"),
    ("did_multiplegt", "did"),
    ("did_imputation", "did"),
    ("event_study", "event_study"),
    ("eventstudy", "event_study"),
    ("staggered", "staggered"),
    ("did_", "did"),
    # IV family
    ("ivreg", "iv"),
    ("iv_2sls", "2sls"),
    ("late_", "late"),
    # RD family
    ("rdrobust", "rd"),
    ("rdbwselect", "rd"),
    ("rdplot", "rd"),
    ("rd_", "rd"),
    # Synth family
    ("synth_did", "synth"),
    ("synth_", "synth"),
    ("synthetic_control", "synth"),
    # DML / ML causal
    ("dml_", "dml"),
    ("causal_forest", "ml"),
    ("xlearner", "metalearners"),
    ("tlearner", "metalearners"),
    ("slearner", "metalearners"),
    # Bayesian
    ("bayes_", "bayesian"),
    ("bcf_", "bayesian"),
    # Mendelian randomization
    ("mr_", "mendelian_randomization"),
    # TMLE
    ("tmle", "tmle"),
    ("ltmle", "tmle"),
    ("hal_tmle", "tmle"),
    # Sensitivity
    ("sensemakr", "sensitivity"),
    ("oster_delta", "sensitivity"),
    ("e_value", "sensitivity"),
    ("evalue", "sensitivity"),
    # Misc design
    ("conformal", "conformal"),
    ("policy_", "policy_learning"),
    ("dynamic_treatment", "dtr"),
    ("dtr_", "dtr"),
    ("proximal", "proximal"),
    ("bridge_", "proximal"),
    ("interference", "interference"),
    ("spillover", "interference"),
    ("transport", "transport"),
    ("dag_", "dag"),
)


# Bib-key marker in docstrings: looks like ``[@authorYEARkeyword]``.
BIB_MARKER_RE = re.compile(r"\[@([A-Za-z][A-Za-z0-9_]*)\]")

# Inside-line patterns for example extraction (NumPy / doctest style).
DOCTEST_LINE_RE = re.compile(r"^\s*>>>\s+(.*)$")
SP_CALL_RE = re.compile(r"(sp\.\w+\([^\n]*\))")


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def load_bib_keys() -> set[str]:
    """Return the set of bib keys present in ``paper.bib``."""
    if not PAPER_BIB.exists():
        return set()
    text = PAPER_BIB.read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\{([^,\s]+),", text, flags=re.MULTILINE))


def _continuation_lines(doc_lines: List[str], start: int) -> List[str]:
    """Return continuation-line bodies starting at ``doc_lines[start]``.

    A NumPy-style doctest continues with ``... `` prefixes; we pull
    those out until we hit a non-continuation line.  Returns the
    *content* (no ``... `` prefix).
    """
    cont = []
    for line in doc_lines[start:]:
        s = line.lstrip()
        if s.startswith("..."):
            cont.append(s[3:].lstrip())
            continue
        break
    return cont


def _balance_parens(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _shorten_call(call: str, max_len: int = 200) -> str:
    """Compress a multi-line call into one line and truncate gently."""
    one = re.sub(r"\s+", " ", call).strip()
    if len(one) <= max_len:
        return one
    # Try to keep just the function head: ``sp.xxx(...)``.
    m = re.match(r"(sp\.\w+)\s*\(", one)
    if m:
        return f"{m.group(1)}(...)"
    return one[: max_len - 3] + "..."


def extract_first_example(doc: str, fn_name: str) -> Optional[str]:
    """Pull the most informative ``sp.<fn>(...)`` call from the docstring.

    Strategy, in order:

    1. **Same-line complete call** — ``>>> ... sp.<fn>(...complete...)``
       on a single doctest line.
    2. **Multi-line call** — ``>>> ... sp.<fn>(`` followed by ``...``
       continuation lines until parens balance.  We join and shorten.
    3. **Same-line related call** — any complete ``sp.<other>(...)``.
    4. **Synthetic placeholder** — if a ``sp.<fn>(`` opener exists
       anywhere in the doctest, emit ``sp.<fn>(...)`` as a hint.

    Imports (``>>> import statspai as sp``) are always skipped.
    """
    if not doc:
        return None
    target_re = re.compile(rf"sp\.{re.escape(fn_name)}\s*\(")
    target_complete = re.compile(rf"sp\.{re.escape(fn_name)}\s*\([^\n]*\)")
    lines = doc.splitlines()

    # Pre-filter to doctest lines while keeping indices for continuation lookahead.
    doctest_idx: List[int] = []
    for i, raw in enumerate(lines):
        m = DOCTEST_LINE_RE.match(raw)
        if not m:
            continue
        content = m.group(1).strip()
        if content.startswith("import") or content.startswith("from"):
            continue
        doctest_idx.append(i)

    def _line_content(i: int) -> str:
        return DOCTEST_LINE_RE.match(lines[i]).group(1).strip()  # type: ignore[union-attr]

    # Strategy 1: one-line complete target call.
    for i in doctest_idx:
        c = _line_content(i)
        tm = target_complete.search(c)
        if tm:
            return _shorten_call(tm.group(0))

    # Strategy 2: multi-line target call.
    for i in doctest_idx:
        c = _line_content(i)
        if not target_re.search(c):
            continue
        # Grab from the sp.<fn>( onward, then append continuations.
        head_match = target_re.search(c)
        if head_match is None:
            continue
        head = c[head_match.start() :]
        cont = _continuation_lines(lines, i + 1)
        joined = head + " " + " ".join(cont)
        if _balance_parens(joined):
            return _shorten_call(joined)
        # Even unbalanced — fall through, will produce synthetic in step 4.

    # Strategy 3: any one-line sp.X(...) call.
    for i in doctest_idx:
        c = _line_content(i)
        am = SP_CALL_RE.search(c)
        if am:
            return _shorten_call(am.group(1))

    # Strategy 4: synthetic placeholder if the target appears anywhere.
    for i in doctest_idx:
        if target_re.search(_line_content(i)):
            return f"sp.{fn_name}(...)"

    return None


def extract_first_bib_key(doc: str, valid_keys: set[str]) -> Optional[str]:
    """Return the first ``[@bib_key]`` marker that is present in paper.bib.

    Markers that don't resolve to an actual bib entry are dropped —
    this is the §10 red line: never emit an unverified citation.
    """
    if not doc:
        return None
    for m in BIB_MARKER_RE.finditer(doc):
        key = m.group(1)
        if key in valid_keys:
            return key
    return None


def derive_tags(name: str, category: str) -> List[str]:
    """Heuristic tag derivation from function name + category.

    Returns at most ~3 tags.  Always includes the category itself as
    the first tag (so the controlled vocab matches the existing
    curated set).  Order: ``[category, family_tag?, ...]``.
    """
    tags: List[str] = []
    if category and category != "other":
        tags.append(category)
    lname = name.lower()
    seen: set[str] = set(tags)
    for needle, tag in NAME_TAG_RULES:
        if needle in lname and tag not in seen:
            tags.append(tag)
            seen.add(tag)
            if len(tags) >= 3:
                break
    return tags


def collect_enrichments() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Walk the registry and compute the enrichment dict.

    Returns (enrichments, stats).  Only entries with at least one
    non-empty field are emitted.

    Chicken-and-egg note: ``_ensure_full_registry`` normally applies
    the currently committed baseline cards on top of the auto-pass.
    On a re-gen, that would make every previously-filled field look
    "already curated" and we'd emit an empty file.  We short-circuit
    the baseline-card application by setting its sentinel before
    triggering registry construction.
    """
    import statspai  # noqa: F401
    from statspai import registry as _reg

    # Bypass the baseline-card pass so our own scan sees the raw
    # auto-pass + hand-curated specs (no auto-baseline re-injected).
    _reg._BASELINE_CARDS_APPLIED = True
    _reg._ensure_full_registry()
    _REGISTRY = _reg._REGISTRY

    valid_bib = load_bib_keys()

    enrichments: Dict[str, Dict[str, Any]] = {}
    stats = {
        "n_total": len(_REGISTRY),
        "examples_filled": 0,
        "references_filled": 0,
        "tags_filled": 0,
        "skipped_already_curated": 0,
    }

    for name, spec in _REGISTRY.items():
        fn = getattr(statspai, name, None)
        if fn is None:
            continue
        doc = inspect.getdoc(fn) or ""

        enrich: Dict[str, Any] = {}

        # example — only if spec.example is currently empty
        if not spec.example:
            ex = extract_first_example(doc, name)
            if ex:
                enrich["example"] = ex
                stats["examples_filled"] += 1
        else:
            stats["skipped_already_curated"] += 1

        # reference — only if spec.reference is currently empty AND the
        # bib key exists in paper.bib.  Zero hallucination guard.
        if not spec.reference:
            key = extract_first_bib_key(doc, valid_bib)
            if key:
                enrich["reference"] = key
                stats["references_filled"] += 1

        # tags — only if spec.tags is currently empty
        if not spec.tags:
            tags = derive_tags(name, spec.category)
            if tags:
                enrich["tags"] = tags
                stats["tags_filled"] += 1

        if enrich:
            enrichments[name] = enrich

    return enrichments, stats


# --------------------------------------------------------------------- #
#  Codegen
# --------------------------------------------------------------------- #


def render_module(enrichments: Dict[str, Dict[str, Any]]) -> str:
    """Emit the Python source of ``src/statspai/_baseline_cards.py``."""
    header = '''"""Auto-generated Tier-B enrichment for the function registry.

DO NOT EDIT BY HAND — regenerate via::

    python scripts/gen_baseline_cards.py

Every entry below is mechanically extracted from a docstring or from
the function name + category.  The ``apply()`` helper only fills a
:class:`statspai.registry.FunctionSpec` field if that field is
currently empty, so curated content in ``registry.py`` always wins.

See ``docs/agent_cards_spec.md`` for the tier definitions.
"""
from __future__ import annotations

from typing import Any, Dict


BASELINE_CARDS: Dict[str, Dict[str, Any]] = {
'''
    body_lines: List[str] = []
    for name in sorted(enrichments.keys()):
        entry = enrichments[name]
        body_lines.append(f"    {name!r}: {{")
        if "example" in entry:
            body_lines.append(f"        'example': {entry['example']!r},")
        if "reference" in entry:
            body_lines.append(f"        'reference': {entry['reference']!r},")
        if "tags" in entry:
            tag_repr = ", ".join(repr(t) for t in entry["tags"])
            body_lines.append(f"        'tags': [{tag_repr}],")
        body_lines.append("    },")
    body = "\n".join(body_lines)

    footer = '''
}


def apply(registry: Dict[str, Any]) -> None:
    """Fill empty Tier-B fields on registered specs.

    ``registry`` is the live ``_REGISTRY`` dict from
    :mod:`statspai.registry`.  We mutate :class:`FunctionSpec`
    instances in place but only when the target field is empty —
    curated specs are never overwritten.

    Idempotent: running twice has no further effect because once a
    field is populated, the empty check stops re-application.
    """
    for name, enrich in BASELINE_CARDS.items():
        spec = registry.get(name)
        if spec is None:
            continue
        ex = enrich.get('example')
        if ex and not spec.example:
            spec.example = ex
        ref = enrich.get('reference')
        if ref and not spec.reference:
            spec.reference = ref
        tags = enrich.get('tags')
        if tags and not spec.tags:
            spec.tags = list(tags)
'''
    return header + body + footer


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats but do not write the output file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Output path (default: {OUT_PATH}).",
    )
    args = parser.parse_args(argv)

    enrichments, stats = collect_enrichments()

    print(
        f"Scanned {stats['n_total']} registered functions.\n"
        f"  examples to fill   : {stats['examples_filled']:4d}\n"
        f"  references to fill : {stats['references_filled']:4d}\n"
        f"  tags to fill       : {stats['tags_filled']:4d}\n"
        f"  entries to emit    : {len(enrichments):4d}",
        file=sys.stderr,
    )

    source = render_module(enrichments)

    if args.dry_run:
        print(
            f"[dry-run] Would write {len(source)} chars to {args.out}",
            file=sys.stderr,
        )
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
