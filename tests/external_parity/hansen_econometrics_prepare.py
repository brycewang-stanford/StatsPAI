"""Prepare the Stata programs of Hansen's *Econometrics* for a logged run.

Bruce Hansen's *Econometrics* (Princeton University Press, 2022) comes with
the programs behind its tables and figures: sixteen Stata do-files
(``chapter3.do`` ... ``chapter26.do``), R and MATLAB versions of some of
them, and the data. The do-files carry no output, so the answer key for the
replay test is a log made by running them in Stata. This script writes,
into ``<folder>/run``:

* ``Chapter_NN.do``: the chapter's do-file with the lines that only manage
  logs and graphs removed (``log using``, ``translate``, ``irf graph``,
  ``graph export``, ``set scheme``), relative paths to the author's own
  folders replaced by the file name, ``use ..., replace`` written as
  ``use ..., clear``, and the number of bootstrap replications cut from
  10,000 to 200. A bootstrap number depends on Stata's random-number
  stream and cannot be compared in any case; the cut only keeps the run
  short;
* a copy of every ``.dta`` file the chapters read;
* ``_master.do``, which runs each chapter under ``log using
  Chapter_NN.log, text`` with ``nostop``.

Then, in Stata 18, from ``<folder>/run``::

    do _master.do

Usage::

    python tests/external_parity/hansen_econometrics_prepare.py <folder>

``<folder>`` is the one holding ``Econometrics Programs`` and
``Econometrics Data`` as downloaded from the book's page.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

CHAPTERS = [3, 4, 8, 10, 11, 12, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26]

_DROP = re.compile(
    r"^\s*(log\s+(using|close)|translate\b|irf\s+graph\b|graph\s+export\b"
    r"|set\s+scheme\b)",
    re.I,
)
_PATH = re.compile(r'use\s+"(?:[^"]*/)?([^"/]+\.dta)"')
_USE_REPLACE = re.compile(r"^(\s*use\s+[^,]+),\s*replace\s*$")
_REPS = re.compile(r"reps\(10000\)")


def clean(text: str) -> str:
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if _DROP.match(line):
            continue
        line = _PATH.sub(r"use \1", line)
        line = _USE_REPLACE.sub(r"\1, clear", line)
        line = _REPS.sub("reps(200)", line)
        out.append(line.rstrip())
    return "\n".join(out).rstrip() + "\n"


def prepare(folder: Path) -> Path:
    programs = folder / "Econometrics Programs"
    run = folder / "run"
    run.mkdir(exist_ok=True)
    for source in list(programs.glob("*.dta")) + list(
        (folder / "Econometrics Data").rglob("*.dta")
    ):
        target = run / source.name
        if not target.exists():
            shutil.copyfile(source, target)
    master = ["clear all", "set more off", "set graphics off", "set linesize 255"]
    for number in CHAPTERS:
        name = f"Chapter_{number:02d}"
        text = (programs / f"chapter{number}.do").read_text(
            encoding="utf-8", errors="replace"
        )
        (run / f"{name}.do").write_text(clean(text), encoding="utf-8")
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
