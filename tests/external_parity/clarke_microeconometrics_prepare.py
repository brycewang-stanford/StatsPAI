"""Prepare the Stata chapters of Clarke's *Applied Microeconometrics* for a
logged Stata run.

The book's companion site (the Quarto project ``Microeconometrics_QuartoExample``)
gives every code call-out in R, Stata and Python. The Stata chapters
(``02_02_Chapter2Stata.qmd`` ... ``02_04_Chapter4Stata.qmd``) hold the code
in fenced blocks, each of which Quarto runs in a session of its own. The
replay test needs Stata's output as its answer key, so the blocks have to be
run once with a log. This script writes, into ``<folder>/run``:

* ``Chapter_02.do`` ... ``Chapter_04.do``: the blocks of each chapter in
  order. Before each block a pending ``preserve`` and the frames are
  dropped inside a ``quietly`` block (which the log does not echo) and the
  data are cleared (which it does), because the blocks were written to
  start from an empty session. The one block
  that continues the data of the block before it (the simulation of
  chapter 2) is left alone. ``quietly { ... }`` wrappers around data steps
  are removed so the log shows the commands they run; loops keep theirs
  (the replay reads the body of a quiet block from the do-file, which is
  why ``Chapter_NN.do`` has to stay beside ``Chapter_NN.log``);
* a link to the ``Datasets`` folder;
* ``_master.do``, which runs each chapter under ``log using
  Chapter_NN.log, text`` with ``nostop``: chapter 4 has one line that
  stops Stata (``list ..., sepby`` without its argument).

Then, in Stata 18, from ``<folder>/run``::

    do _master.do

Community commands the chapters use: ``psmatch2``, ``boottest``, ``synth``,
``sdid`` and the ``plottig`` / ``plotplainblind`` schemes (``blindschemes``),
all from SSC. The run takes about a minute.

Usage::

    python tests/external_parity/clarke_microeconometrics_prepare.py <folder>

``<folder>`` is the Quarto project (the one holding the ``.qmd`` files and
``Datasets``).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

CHAPTERS = {
    "Chapter_02": "02_02_Chapter2Stata.qmd",
    "Chapter_03": "02_03_Chapter3Stata.qmd",
    "Chapter_04": "02_04_Chapter4Stata.qmd",
}
#: (chapter, block number) that goes on from the data of the block before it
_CONTINUES = {("Chapter_02", 2)}
_BLOCK = re.compile(r"^```\{stata[^\n]*\n(.*?)^```", re.S | re.M)
_QUIET_OPEN = re.compile(r"^(?:qui|quietly)\s*\{$")
_LOOP = re.compile(r"^(?:forv\w*|foreach|while)\b")
_QUIET_STEP = re.compile(
    r"^(\s*)(?:qui|quietly)\s+(?=(?:import|keep|use|drop|gen|replace)\b)"
)
_RESET = "quietly {\n    capture restore\n    capture frames reset\n}\nclear"


def _unwrap(lines: list) -> list:
    """Drop ``quietly {`` ... ``}`` around plain commands; keep other braces,
    and a quiet block inside a loop or a program.

    A loop that was inside such a block stays quiet (``quietly forvalues``):
    its thousand regressions are not what the log is for.
    """
    out, open_blocks = [], []
    for line in lines:
        bare = line.strip()
        if _QUIET_OPEN.match(bare) and "other" not in open_blocks:
            open_blocks.append("quiet")
            continue
        if bare.endswith("{"):
            if "quiet" in open_blocks and _LOOP.match(bare):
                line = line.replace(bare, "quietly " + bare, 1)
            open_blocks.append("other")
        elif bare == "}" and open_blocks and open_blocks.pop() == "quiet":
            continue
        out.append(_QUIET_STEP.sub(r"\1", line))
    return out


def prepare(folder: Path) -> Path:
    run = folder / "run"
    run.mkdir(exist_ok=True)
    link = run / "Datasets"
    if not link.exists():
        os.symlink(os.path.relpath(folder / "Datasets", run), link)
    master = ["clear all", "set more off", "set graphics off", "set linesize 255"]
    for name, source in CHAPTERS.items():
        text = (folder / source).read_text(encoding="utf-8")
        out: list = []
        for number, block in enumerate(_BLOCK.findall(text), 1):
            lines = [ln for ln in block.split("\n") if not ln.lstrip().startswith("#|")]
            out.append(f"* ---- block {number}")
            if (name, number) not in _CONTINUES:
                out.append(_RESET)
            out.extend(_unwrap(lines))
        (run / f"{name}.do").write_text("\n".join(out) + "\n", encoding="utf-8")
        master += [
            "capture log close _all",
            f'log using "{name}.log", text replace name({name})',
            f'capture noisily do "{name}.do", nostop',
            f"log close {name}",
            "clear all",
            "set graphics off",
        ]
    (run / "_master.do").write_text("\n".join(master) + "\n", encoding="utf-8")
    return run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(prepare(Path(sys.argv[1])))
