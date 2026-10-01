#!/usr/bin/env python3
"""How much of a set of do-files does ``sp.from_stata`` translate, and how?

Reads every ``.do`` file under the given paths, resolves comments,
continuations and text-defined macros the way ``sp.stata`` does, and runs
each estimation command through ``sp.from_stata``. The report separates a
faithful translation from one that says what it lost and from a refusal,
and lists the commands and options that account for the losses -- by the
number of projects (top-level folders) they appear in, so one large
package does not set the priorities.

It is a detector, not a test: fix what it finds against Stata's
documented grammar and pin the fix with synthetic commands in
``tests/test_stata_translation_grammar.py``.

    python scripts/stata_corpus_scan.py ~/replications --csv scan.csv

No do-file is shipped with StatsPAI; point it at your own.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

#: Estimation commands counted even though they are not translated, so the
#: report shows what is missing. Everything in the translator's own map is
#: counted as well.
_ESTIMATION = set("""
areg xtivreg xtivreg2 xtpoisson xtlogit xtprobit ologit clogit qreg sqreg
reg2hdfe acreg csdid2 did_multiplegt did_multiplegt_dyn eventstudyinteract
jwdid xthdidregress sdid synth_runner rdbwselect kmatch ebalance xtabond2
xtscc newey gmm sureg reg3 glm fracreg stcox mixed xtevent did2s lpdid
stackedev eventdd leebounds rwolf
""".split())
_SKIP = {
    "su",
    "sum",
    "summarize",
    "sum2docx",
    "xtset",
    "tsset",
    "test",
    "lincom",
    "margins",
    "marginsplot",
    "contrast",
    "mi",
}


def _read(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _statements(path: Path) -> Iterator[Tuple[str, str]]:
    """``(as written, macros expanded or '')`` for each command of a file."""
    from statspai.agent._translation._stata_script import (
        MacroTable,
        ScriptError,
        split_commands,
    )

    macros = MacroTable()
    for command in split_commands(_read(path)):
        try:
            if macros.define(command):
                continue
            yield command, macros.expand(command)
        except ScriptError:
            yield command, ""


def _outcome(line: str, expanded: str) -> Tuple[str, Dict[str, Any]]:
    from statspai.agent._translation._stata import from_stata

    if not expanded:
        return "refused: macro not resolvable by reading", {}
    out = from_stata(expanded)
    if not out.get("ok"):
        error = str(out.get("error"))
        if "unsupported" in error or "line by line" in error:
            return "command not translated", out
        if "prefix" in error:
            return "refused: prefix changes the estimate", out
        if "macro" in error:
            return "refused: macro not resolvable by reading", out
        if "names dataset columns" in error:
            # x* / a-b need the dataset; sp.stata passes its columns
            return "needs the dataset's columns (sp.stata has them)", out
        return "refused: other", out
    if out.get("untranslated_options"):
        return "translated, option loss reported", out
    if out.get("unapplied_sample"):
        return "translated, `if`/`in` to apply first", out
    return "translated faithfully", out


def scan(paths: List[Path], exclude: Tuple[str, ...] = ()) -> List[Dict[str, str]]:
    from statspai.agent._translation._stata import STATA_COMMAND_MAP
    from statspai.agent._translation._stata_options import peel_prefixes

    wanted = (set(STATA_COMMAND_MAP) | _ESTIMATION) - _SKIP
    rows: List[Dict[str, str]] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.do"))
        for path in files:
            if any(part in str(path) for part in exclude):
                continue
            relative = path.relative_to(root) if root.is_dir() else Path(path.name)
            project = relative.parts[0] if len(relative.parts) > 1 else root.name
            for line, expanded in _statements(path):
                _, refused, core = peel_prefixes(line)
                if refused:
                    core = core.split(":", 1)[-1].strip()
                word = re.match(r"[A-Za-z_]\w*", core)
                if not word or word.group(0).lower() not in wanted:
                    continue
                outcome, out = _outcome(line, expanded)
                lost = [str(o) for o in out.get("untranslated_options") or []]
                rows.append(
                    {
                        "project": project,
                        "file": str(relative),
                        "command": word.group(0).lower(),
                        "outcome": outcome,
                        "untranslated": " ".join(lost),
                        "error": str(out.get("error") or "")[:200],
                        "statement": line[:300],
                    }
                )
    return rows


def report(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "no estimation commands found"
    n = len(rows)
    lines = [
        f"{n} estimation commands in {len({r['project'] for r in rows})} projects",
        "",
    ]
    for outcome, count in collections.Counter(r["outcome"] for r in rows).most_common():
        lines.append(f"  {count:6d}  {100 * count / n:5.1f}%  {outcome}")

    def by_projects(pairs: List[Tuple[str, str]], title: str) -> None:
        projects = collections.defaultdict(set)
        counts: collections.Counter = collections.Counter()
        for key, project in pairs:
            projects[key].add(project)
            counts[key] += 1
        if not counts:
            return
        lines.extend(["", title])
        ranked = sorted(counts, key=lambda k: (-len(projects[k]), -counts[k]))
        for key in ranked[:20]:
            lines.append(f"  {len(projects[key]):3d} projects  {counts[key]:5d}  {key}")

    by_projects(
        [
            (r["command"], r["project"])
            for r in rows
            if r["outcome"] == "command not translated"
        ],
        "Commands not translated:",
    )
    by_projects(
        [
            (f"{r['command']}: {opt}", r["project"])
            for r in rows
            for opt in r["untranslated"].split()
        ],
        "Options reported as not carried over:",
    )
    quoted = re.compile(r"'[^']*'|\"[^\"]*\"")
    by_projects(
        [
            (f"{r['command']}: {quoted.sub('…', r['error'])[:90]}", r["project"])
            for r in rows
            if r["outcome"] == "refused: other"
        ],
        "Other refusals:",
    )
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help=".do files or folders")
    parser.add_argument("--csv", type=Path, help="write one row per command")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="skip files whose path contains this text (repeatable)",
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = scan(args.paths, tuple(args.exclude))
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
