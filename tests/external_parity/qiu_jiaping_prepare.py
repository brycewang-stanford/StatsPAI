"""Prepare the Stata code of Qiu Jiaping's textbook for a logged Stata run.

*Practical Econometric Methods for Causal Inference* (Qiu Jiaping) ships
its Stata code as one transcribed text file, ``stata_code.txt``, with a
fenced block per chapter (3 to 11), and its data as ``data/dataNN*.dta``.
The replay test needs Stata's own output as the answer key, so the code has
to be run once with a log. This script writes, into ``<folder>/run``:

* ``Chapter_NN.do`` for chapters 3 to 11, cut from the fenced blocks, with
  the file names pointed at the shipped data (the text says ``data3.dta``
  and ``data7-1.dta``; the files are ``data03.dta`` and ``data07_1.dta``)
  and the transcription slips that stop Stata repaired (listed in
  ``_REPAIRS`` below, each with what the book prints);
* ``Chapter_12.do``. The book's regression discontinuity example uses union
  election data that were never released, and the text file has no code for
  it. The commands printed in the chapter (``ttest``, ``rdplot`` with
  evenly spaced and quantile spaced bins, ``DCdensity``, ``rddensity``, the
  fourth-order global polynomial, ``rdrobust``) are applied here to the two
  Lee (2008) House election files that ship in their place;
* a copy of the datasets;
* ``_master.do``, which runs each chapter under ``log using
  Chapter_NN.log, text`` with ``do ..., nostop``.

Then, in Stata 18, from ``<folder>/run``::

    do _master.do

Community commands the chapters use: ``pscore`` / ``attnd`` (``net install
st0026_2, from(http://www.stata-journal.com/software/sj5-3)``),
``psmatch2`` / ``pstest``, ``estout`` (SSC), ``rdrobust`` / ``rddensity`` /
``lpdensity`` (https://github.com/rdpackages) and McCrary's ``DCdensity``.

Usage::

    python tests/external_parity/qiu_jiaping_prepare.py <folder>

``<folder>`` holds ``stata_code.txt`` and ``data/``.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

_CHAPTERS = {
    "第三章": 3,
    "第四章": 4,
    "第五章": 5,
    "第六章": 6,
    "第七章": 7,
    "第八章": 8,
    "第九章": 9,
    "第十章": 10,
    "第十一章": 11,
}

#: (chapter, text as transcribed, text as run, why)
_REPAIRS = [
    (4, "gen err = 0 in 1/1", "gen e = 0 in 1/1", "the next line replaces e"),
    (5, "1.g1schid, cluster", "i.g1schid, cluster", "school dummies, as above it"),
    (
        11,
        "gen work = (wage ~= .)",
        "capture drop work\ngen work = (wage ~= .)",
        "the substitute data already hold a variable of that name",
    ),
    (11, "normalden(2)", "normalden(z)", "the density at the probit index"),
    (11, "normal(2)", "normal(z)", "the distribution function at the index"),
    (
        11,
        "treat(union = south black tenure) twostep \nest store etrMLE",
        "treat(union = south black tenure)\nest store etrMLE",
        "the block is headed maximum likelihood",
    ),
]

_CHAPTER_12 = """\
* Regression discontinuity: the commands of chapter 12 on Lee (2008)
use "data12_1_lee.dta", clear
des margin vote
sum margin vote
gen win = margin >= 0
ttest vote, by(win)

rdplot vote margin, nbins(10 10) binselect(es) c(0) p(4)
rdplot vote margin, nbins(10 10) binselect(qs) c(0) p(4)
rdplot vote margin, binselect(es) c(0) p(4)
rdplot vote margin, binselect(qs) c(0) p(4)

DCdensity margin, breakpoint(0) generate(Xj Yj r0 fhat se_fhat) nograph
rddensity margin, c(0)

gen X1 = margin
gen X2 = X1^2
gen X3 = X1^3
gen X4 = X1^4
gen win_X1 = X1*win
gen win_X2 = X2*win
gen win_X3 = X3*win
gen win_X4 = X4*win
reg vote win X1 X2 X3 X4 win_X1 win_X2 win_X3 win_X4, robust
reg vote win X1 win_X1 if abs(margin) <= 0.1, robust

rdrobust vote margin, c(0) kernel(triangular) p(1) bwselect(mserd)
rdrobust vote margin, c(0) kernel(triangular) p(1) bwselect(mserd) all
rdrobust vote margin, c(0) kernel(epanechnikov) p(2) bwselect(mserd)
rdrobust vote margin, c(0) kernel(uniform) p(1) h(0.1)
rdbwselect vote margin, c(0) kernel(triangular) p(1) all

use "data12_2_group_final.dta", clear
keep if use == 1
sum difdemshare mdemsharenext mdemshareprev yearel
gen win = difdemshare >= 0
gen X1 = difdemshare
gen X2 = X1^2
gen X3 = X1^3
gen X4 = X1^4
gen win_X1 = X1*win
gen win_X2 = X2*win
gen win_X3 = X3*win
gen win_X4 = X4*win
reg mdemsharenext win X1 X2 X3 X4 win_X1 win_X2 win_X3 win_X4, cluster(yearel)
rdrobust mdemsharenext difdemshare, c(0) kernel(triangular) p(1) bwselect(mserd) vce(cluster yearel)
rdrobust mdemshareprev difdemshare, c(0) kernel(triangular) p(1) bwselect(mserd) vce(cluster yearel)
rdrobust mdemsharenext difdemshare, c(0) kernel(triangular) p(1) bwselect(mserd) covs(mdemshareprev) vce(cluster yearel)
rddensity difdemshare, c(0)
"""


def _data_name(match: "re.Match[str]") -> str:
    """``data7-1.dta`` in the text is ``data07_1.dta`` on disk."""
    chapter, part = match.group(1), match.group(2)
    name = f"data{int(chapter):02d}" + (f"_{part}" if part else "")
    return f'"{name}.dta"'


def chapters(text: str) -> dict[int, str]:
    """The fenced Stata block under each chapter heading."""
    out: dict[int, str] = {}
    pattern = re.compile(r"^## (\S+)\s*\n+```stata\n(.*?)^```", re.M | re.S)
    for heading, body in pattern.findall(text):
        if heading in _CHAPTERS:
            out[_CHAPTERS[heading]] = body
    return out


def prepare(folder: Path) -> Path:
    run = folder / "run"
    run.mkdir(exist_ok=True)
    for data in (folder / "data").glob("*.dta"):
        shutil.copy2(data, run / data.name)
    text = (folder / "stata_code.txt").read_text(encoding="utf-8").replace("\r", "")
    blocks = chapters(text)
    for number, body in blocks.items():
        body = re.sub(r'"\$libname\\data(\d+)(?:-(\d+))?\.dta"', _data_name, body)
        for chapter, old, new, _why in _REPAIRS:
            if chapter == number:
                if old not in body:
                    raise SystemExit(f"chapter {number}: {old!r} not found")
                body = body.replace(old, new)
        (run / f"Chapter_{number:02d}.do").write_text(body, encoding="utf-8")
    blocks[12] = _CHAPTER_12
    (run / "Chapter_12.do").write_text(_CHAPTER_12, encoding="utf-8")
    master = ["clear all", "set more off", "set graphics off", "set linesize 255"]
    for number in sorted(blocks):
        name = f"Chapter_{number:02d}"
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
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit(__doc__)
    print("prepared", prepare(Path(sys.argv[1])))
