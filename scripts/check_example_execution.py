"""Execute every runnable docstring ``Examples`` block in the package.

The companion to :mod:`scripts.examples_coverage` (which measures
*presence* of an ``Examples`` section). This script measures
*runnability*: it extracts the ``>>>`` / ``...`` source lines from every
public function's docstring, drops any line flagged ``# doctest: +SKIP``,
and executes the remainder in a shared ``{sp, np}`` namespace. A block
that raises is a failure.

This is the gate behind the project promise that copy-pasteable examples
actually run (JOSS review checklist: "example usage"). Blocks that
genuinely cannot run in a hermetic test environment — heavy optional
deps (PyTorch/PyMC), external data files — must mark their executable
lines ``# doctest: +SKIP`` so they are illustrative-only and skipped
here.

Usage
-----
python scripts/check_example_execution.py            # run all, list failures
python scripts/check_example_execution.py --quiet    # only the summary line
python scripts/check_example_execution.py --max-failures N
                                                     # CI ratchet: exit 1 if
                                                     # failures exceed N
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import os
import re
import sys
import warnings
from typing import List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

_HEADER = re.compile(r"^\s*Examples?\s*\n\s*-{3,}", re.MULTILINE)
_PROMPT = re.compile(r"^\s*>>> (.*)$")
_CONT = re.compile(r"^\s*\.\.\. (.*)$")


def _extract(doc: str) -> str:
    """Return the executable source of a docstring's example blocks.

    Joins each ``>>>`` statement with its ``...`` continuations and drops
    any statement whose first line carries ``# doctest: +SKIP``.
    """
    lines = doc.splitlines()
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        m = _PROMPT.match(lines[i])
        if not m:
            i += 1
            continue
        stmt = [m.group(1)]
        i += 1
        while i < n:
            c = _CONT.match(lines[i])
            if not c:
                break
            stmt.append(c.group(1))
            i += 1
        block = "\n".join(stmt)
        if "+SKIP" not in block:
            out.append(block)
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="exit non-zero if the number of failing examples exceeds this",
    )
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import numpy as np
    import statspai as sp

    seen = set()
    ran = 0
    failures: List[str] = []
    for name in sorted(sp.list_functions()):
        obj = getattr(sp, name, None)
        if not callable(obj):
            continue
        oid = id(inspect.unwrap(obj))
        if oid in seen:
            continue
        seen.add(oid)
        doc = inspect.getdoc(obj) or ""
        if not _HEADER.search(doc):
            continue
        code = _extract(doc)
        if not code.strip():
            continue
        ns = {"sp": sp, "np": np, "__name__": "__docexample__"}
        try:
            # Examples print freely; keep that noise out of the gate log.
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"<{name}>", "exec"), ns)  # noqa: S102
            ran += 1
        except BaseException as e:  # noqa: BLE001 - report, don't crash
            first = str(e).splitlines()[0] if str(e) else ""
            failures.append(f"{name}: {type(e).__name__}: {first[:100]}")

    if not args.quiet:
        for f in failures:
            print(f"FAIL {f}")
    print(f"ran_ok={ran} failed={len(failures)}")

    if args.max_failures is not None and len(failures) > args.max_failures:
        print(
            f"RATCHET FAIL: {len(failures)} failing examples exceeds "
            f"the allowed maximum of {args.max_failures}."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
