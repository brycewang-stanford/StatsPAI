"""Stata 18 reference numbers for the commands of Kohler, Kreuter and
Haensch, *Data Analysis Using Stata* (4th ed.), on committed synthetic data.

``_fixtures/kk_syllabus_reference.do`` was run in Stata 18 on
``_fixtures/kk_syllabus.csv`` and wrote ``_fixtures/kk_syllabus_stata.txt``:
one ``name value`` pair per ``emit`` line, at full precision. This test
runs the same do-file through one ``sp.stata`` session and evaluates each
``emit`` expression where Stata evaluated it.

Evidence level: T2. Same bytes, same estimand, deterministic algorithms.
The tolerance is 1e-9 relative. Two groups of names are relaxed to 1e-5,
each for a documented reason that bounds the gap:

* ``ML``: quantities behind a maximum-likelihood fit. Stata's ``logit``
  stops at ``nrtolerance(1e-5)``; ``sp.logit`` iterates further.
* ``SINGLE``: numbers Stata passes through a single-precision variable on
  the way. ``kwallis`` and ``estat ovtest`` build their working variables
  as ``float``; ``dfbeta``, ``statsby`` and ``collapse`` store their
  results as ``float`` (``collapse`` because ``import delimited`` typed
  the integer source as ``int``). The do-file reads and predicts in double precision
  everywhere it can, so nothing else is affected.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.agent._translation._stata_run import StataSession
from statspai.agent._translation._stata_script import split_commands

FIXTURES = Path(__file__).parent / "_fixtures"

#: names whose value depends on where an iterative fit stopped
ML = re.compile(r"^(logit_|gof_|hl_|roc_|linfl_|linktest_|lrtest_|mlogit_|ologit_)")
#: names Stata computes through a single-precision variable
SINGLE = re.compile(r"^(kwallis_|reset_rhs_|infl__dfbeta_|collapse_|statsby_)")
#: read from the output of the command before the `emit`, not from r()
FROM_OUTPUT = {
    "prop_ll2": lambda out: out["ci_lower"].iloc[1],
    "prop_ul2": lambda out: out["ci_upper"].iloc[1],
    "svy_deff": lambda out: out["DEFF"].iloc[0],
    "svy_deft": lambda out: out["DEFT"].iloc[0],
    "margins_at_b1": lambda out: out["margin"].iloc[0],
    "margins_at_se3": lambda out: out["se"].iloc[2],
    "margins_f_b4": lambda out: out["margin"].iloc[3],
    "margins_f_se4": lambda out: out["se"].iloc[3],
    "margins_dydx_b2": lambda out: out["dy/dx"].iloc[1],
    "margins_dydx_se2": lambda out: out["se"].iloc[1],
    "nest_F1": lambda out: out["F"].iloc[0],
    "nest_F2": lambda out: out["F"].iloc[1],
    "nest_p2": lambda out: out["p"].iloc[1],
    "nest_r2": lambda out: out["r2"].iloc[1],
    "nest_change": lambda out: out["change_r2"].iloc[1],
}


def _reference() -> dict:
    out = {}
    text = (FIXTURES / "kk_syllabus_stata.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        name, value = line.split()
        out[name] = float(value)
    return out


def _unrolled(commands: list) -> list:
    """The do-file's ``foreach v in a b c { ... }`` loops written out, so
    that each ``emit`` inside one is a line of its own."""
    out, at = [], 0
    while at < len(commands):
        head = re.match(r"\s*foreach\s+(\w+)\s+in\s+(.+?)\s*\{\s*$", commands[at])
        if head is None:
            out.append(commands[at])
            at += 1
            continue
        end = commands.index("}", at)
        for value in head.group(2).split():
            out += [
                c.replace(f"`{head.group(1)}'", value) for c in commands[at + 1 : end]
            ]
        at = end + 1
    return out


def _replay() -> dict:
    """Run the reference do-file; {name: our value} for every `emit`."""
    data = pd.read_csv(FIXTURES / "kk_syllabus.csv")
    session = StataSession(data)
    source = (FIXTURES / "kk_syllabus_reference.do").read_text(encoding="utf-8")
    ours = {}
    last_output = None
    skip = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for command in _unrolled([c.strip() for c in split_commands(source)]):
            if command.startswith("import delimited"):
                skip = False
                continue
            if skip or command.startswith(("file ", "matrix ", "list")):
                continue
            emitted = re.match(r"\s*emit\s+(\w+)\s+(.+)$", command)
            if emitted is None:
                if session.run(command):
                    last_output = session.output
                continue
            name, expr = emitted.group(1), emitted.group(2)
            if name in FROM_OUTPUT:
                ours[name] = float(FROM_OUTPUT[name](last_output))
            else:
                ours[name] = session.value(session._macros.expand(expr))
    return ours


@pytest.fixture(scope="module")
def numbers() -> tuple:
    return _reference(), _replay()


def test_every_reference_number_is_computed(numbers: tuple) -> None:
    reference, ours = numbers
    assert set(reference) == set(ours)
    assert len(reference) > 190


def test_numbers_match_stata(numbers: tuple) -> None:
    reference, ours = numbers
    wrong = []
    for name, value in reference.items():
        rtol = 1e-5 if ML.match(name) or SINGLE.match(name) else 1e-9
        if name in ("cc_lb", "cc_ub"):
            rtol = 1e-4
        if not np.isclose(ours[name], value, rtol=rtol, atol=1e-10):
            wrong.append(f"{name}: Stata {value!r}, ours {ours[name]!r}")
    assert not wrong, "\n".join(wrong)
