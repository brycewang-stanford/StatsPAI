"""Prepare the do-files of Chen Qiang's textbook for a logged Stata run.

The programs and data of *Econometrics and Stata Applications* (2nd edition)
ship as do-files and .dta files, without the output. The replay test needs
Stata's own output as its answer key, so the do-files have to be run once
with a log. This script writes, into ``<folder>/run``:

* a copy of every ``Chapter_NN.do`` with the lines that stop a batch run
  removed (``help``, ``set more on``, ``sysdir``, the interactive ``log``
  demonstrations, ``exit``), two option lines of chapter 15 that the source
  file has on a line of their own joined back to their ``twoway`` command,
  and ``capture noisily`` put in front of the one command the chapter runs
  to show an error (the caliper with too few matches);
* a copy of the datasets;
* ``_master.do``, which runs each chapter under ``log using
  Chapter_NN.log, text``.

Then, in Stata 18, from ``<folder>/run``::

    do _master.do

Community commands the chapters use: ``estout``, ``xtoverid``, ``reghdfe``,
``ftools``, ``coefplot``, ``synth``, ``synth2``, ``rcm`` (SSC), ``xtserial``
(``net install st0039, from(http://www.stata-journal.com/software/sj3-2)``),
``rdrobust`` / ``rddensity`` / ``lpdensity`` and the ``rdrobust_senate``
data (https://github.com/rdpackages). Chapter 18 takes about twenty minutes
(the nested synthetic-control placebo runs).

Usage::

    python tests/external_parity/chen_qiang_2e_prepare.py <folder>

``<folder>`` holds the extracted ``Chapter_*.do`` and ``*.dta`` files.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_DROPPED = ("exit", "set more on", "sysdir")
_DROPPED_PREFIX = ("help ", "log ", "capture log")
_ORPHAN_OPTIONS = ("ytitle(Density)", "legend(label(1 Control)")
_STOPS_BY_DESIGN = "caliper(0.0135) osample(outside)"


def prepare(folder: Path) -> Path:
    run = folder / "run"
    run.mkdir(exist_ok=True)
    for data in folder.glob("*.dta"):
        shutil.copy2(data, run / data.name)
    chapters = sorted(folder.glob("Chapter_*.do"))
    for source in chapters:
        text = source.read_text(encoding="utf-8", errors="replace").replace("\r", "")
        out: list[str] = []
        for line in text.split("\n"):
            bare = line.strip()
            if bare in _DROPPED or bare.startswith(_DROPPED_PREFIX):
                continue
            if bare.startswith(_ORPHAN_OPTIONS) and out:
                out[-1] = out[-1].rstrip() + " " + bare
                continue
            if _STOPS_BY_DESIGN in bare and not bare.startswith("capture"):
                line = "capture noisily " + bare
            out.append(line)
        (run / source.name).write_text("\n".join(out) + "\n", encoding="utf-8")
    master = ["clear all", "set more off", "set graphics off", "set linesize 255"]
    for source in chapters:
        name = source.stem
        master += [
            "capture log close _all",
            f'log using "{name}.log", text replace name({name})',
            f'capture noisily do "{source.name}"',
            f"log close {name}",
            "clear all",
            "set graphics off",
        ]
    (run / "_master.do").write_text("\n".join(master) + "\n", encoding="utf-8")
    return run


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit(__doc__)
    print("prepared", prepare(Path(sys.argv[1])))
