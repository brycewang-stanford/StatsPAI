"""Replay the Stata logs of Chen Qiang, *Econometrics and Stata
Applications* (2nd edition), through ``sp.stata`` and compare every number.

The textbook's programs cover the undergraduate sequence end to end: OLS and
its tests, heteroskedasticity and serial correlation, specification,
instrumental variables, binary choice, panel data, time series, unit roots
and cointegration, matching, regression discontinuity, difference in
differences, synthetic control and the regression control method. Its
do-files come without output, so the answer key is a log made by running
them in Stata (``chen_qiang_2e_prepare.py`` next to this file says how).
``scripts/stata_log_replay.py`` then runs every logged command through one
``sp.stata`` session and compares what Stata printed with what StatsPAI
returns, to the precision Stata printed it.

Neither the programs nor the data are redistributed here. Point
``STATSPAI_CHENQIANG_DIR`` at the folder holding ``Chapter_NN.log`` and the
``.dta`` files to run this; it is skipped otherwise.

    STATSPAI_CHENQIANG_DIR=/path/to/run \\
        pytest tests/external_parity/test_chen_qiang_2e_logs.py

Chapter 18 fits one nested synthetic control (about two and a half minutes).
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_CHENQIANG_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_CHENQIANG_DIR to the logs of Chen Qiang's do-files",
)

#: Numbers reproduced per chapter log on the day this test was written. A
#: drop means a command stopped running or stopped being compared.
REPRODUCED = {
    "Chapter_02.log": 10,
    "Chapter_03.log": 45,
    "Chapter_04.log": 27,
    "Chapter_05.log": 126,
    "Chapter_06.log": 33,
    "Chapter_07.log": 129,
    "Chapter_08.log": 138,
    "Chapter_09.log": 310,
    "Chapter_10.log": 86,
    "Chapter_11.log": 132,
    "Chapter_12.log": 545,
    "Chapter_13.log": 316,
    "Chapter_14.log": 122,
    "Chapter_15.log": 128,
    "Chapter_16.log": 120,
    "Chapter_17.log": 112,
    "Chapter_18.log": 42,
    "Chapter_19.log": 490,
}

#: Printed numbers StatsPAI does not reproduce, each with the reason. The
#: key is (log, start of the command); anything else that differs fails.
DIFFERENT = {
    ("Chapter_09.log", "estat ovtest,rhs"): (
        "One regressor is the square of another. Stata drops the original "
        "regressor from the augmented regression and tests its replacement, "
        "11 restrictions against the model without it (F = 1.73). The RESET "
        "test of the fitted model has 10 restrictions (F = 1.27); the "
        "mechanism is reproduced in docs/dev/2026-10-03-chen-qiang-2e-review.md."
    ),
    ("Chapter_15.log", "sum pscore pscore_match"): (
        "pscore[match1] reads the first neighbour teffects lists. Stata's "
        "order among the four nearest neighbours is not the distance order; "
        "StatsPAI lists the nearest first. The matched set and the ATET are "
        "the same."
    ),
    ("Chapter_18.log", "synth cigsale"): (
        "Nested predictor weights are a non-convex search. StatsPAI's "
        "solution has the lower pre-treatment MSPE (3.086 against Stata's "
        "3.227), so the donor weights and the predictor balance differ."
    ),
}

#: Numbers Stata prints that the StatsPAI result has no counterpart for.
NO_COUNTERPART: dict = {}

#: Commands sp.stata declines.
DECLINED = {
    "synth2": (
        "synth plus nested placebo and leave-one-out runs: hours with the "
        "nested search; the translation names the three sp calls instead."
    ),
}


@pytest.fixture(scope="module")
def frame():
    script = Path(__file__).resolve().parents[2] / "scripts" / "stata_log_replay.py"
    spec = importlib.util.spec_from_file_location("stata_log_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(ROOT)
    logs = sorted(root.glob("Chapter_*.log"))
    out = module.replay(logs, [root])
    out["log"] = out["file"].str.split("/").str[-1]
    return out


def _listed(row, ledger):
    return any(row.log == log and row.command.startswith(cmd) for log, cmd in ledger)


def test_every_difference_is_a_documented_one(frame):
    diff = frame[frame.status == "DIFF"]
    stray = diff[[not _listed(r, DIFFERENT) for r in diff.itertuples()]]
    assert stray.empty, stray[["log", "command", "what", "stata", "ours"]].to_string()
    # a ledger entry that no longer differs should be removed
    for log, cmd in DIFFERENT:
        hit = diff[(diff.log == log) & diff.command.str.startswith(cmd)]
        assert not hit.empty, f"{log}: {cmd!r} is reproduced now; drop the entry"


@pytest.mark.parametrize("log, expected", sorted(REPRODUCED.items()))
def test_each_log_reproduces_its_numbers(frame, log, expected):
    ok = int(((frame.log == log) & (frame.status == "ok")).sum())
    assert ok >= expected, f"{log}: {ok} numbers reproduced, {expected} before"


def test_only_documented_commands_are_declined(frame):
    notrun = frame[frame.status == "NOT RUN"]
    stray = [c for c in notrun.command if c.split()[0] not in DECLINED]
    assert not stray, stray


def test_only_documented_numbers_lack_a_counterpart(frame):
    missing = frame[frame.status == "no output"]
    stray = missing[[not _listed(r, NO_COUNTERPART) for r in missing.itertuples()]]
    assert stray.empty, stray[["log", "command", "what"]].to_string()


def test_simulated_data_are_marked_not_compared(frame):
    # chapters 4, 6 and 14 draw random numbers; numpy's are not Stata's
    random = frame[frame.status == "random"]
    assert set(random.log) == {"Chapter_04.log", "Chapter_06.log", "Chapter_14.log"}
