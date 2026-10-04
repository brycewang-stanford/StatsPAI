"""Replay the Stata logs of Clarke's *Applied Microeconometrics* through
``sp.stata`` and compare every number.

The book's companion site gives each code call-out in R, Stata and Python.
Its Stata chapters cover randomisation inference and the bootstrap
(chapter 2), matching and weighting (chapter 3) and, in chapter 4,
clustered inference with the wild bootstrap, the two-way fixed effects
estimator taken apart by hand, synthetic control and synthetic difference
in differences. The code comes without output, so the answer key is a log
made by running it in Stata (``clarke_microeconometrics_prepare.py`` next
to this file says how). ``scripts/stata_log_replay.py`` then runs every
logged command through one ``sp.stata`` session and compares what Stata
printed with what StatsPAI returns, to the precision Stata printed it.

Neither the programs nor the data are redistributed here. Point
``STATSPAI_CLARKE_DIR`` at the folder holding ``Chapter_NN.log`` (its
parent holds ``Datasets``) to run this; it is skipped otherwise.

    STATSPAI_CLARKE_DIR=/path/to/Microeconometrics_QuartoExample/run \\
        pytest tests/external_parity/test_clarke_microeconometrics_logs.py

What the replay found is in
``docs/dev/2026-10-04-clarke-applied-microeconometrics-review.md``.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_CLARKE_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_CLARKE_DIR to the logs of Clarke's Stata chapters",
)

#: Numbers reproduced per chapter log on the day this test was written. A
#: drop means a command stopped running or stopped being compared.
REPRODUCED = {
    "Chapter_02.log": 37,
    "Chapter_03.log": 79,
    "Chapter_04.log": 547,
}

#: Printed numbers StatsPAI does not reproduce, each with the reason. The
#: key is (log, start of the command); anything else that differs fails.
DIFFERENT = {
    ("Chapter_04.log", "synth cigsale"): (
        "Regression-based predictor weights with one predictor listed twice "
        "(cigsale(1985)), so the weights are not unique. StatsPAI's solution "
        "has the lower pre-treatment RMSPE (1.657031 against Stata's "
        "1.657121); the donor weights differ in the third decimal."
    ),
}

#: First words of the commands sp.stata declines on these logs, with the
#: reason. A command that depends on a declined one (a variable a loop would
#: have created) fails with it; those are listed under FOLLOWS.
DECLINED = {
    "frame": "Stata 16 frames",
    "frlink": "Stata 16 frames",
    "frget": "Stata 16 frames",
    "forvalues": "loops are written in Python around the call",
    "forval": "loops are written in Python around the call",
    "foreach": "loops are written in Python around the call",
    "if": "an `if` block is control flow",
    "program": "a program with arguments",
    "randomization_inference": "the program the chapter defines",
    "svmat": "matrices are not held",
    "tempfile": "files are not written",
    "save": "files are not written",
    "/*": "a comment the log wraps over several lines",
}
FOLLOWS = {"gen", "reg", "summarize", "sum", "_pctile", "scalar", "count", "use", "di"}
FOLLOWS |= {"kdensity"}


@pytest.fixture(scope="module")
def frame():
    script = Path(__file__).resolve().parents[2] / "scripts" / "stata_log_replay.py"
    spec = importlib.util.spec_from_file_location("stata_log_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(ROOT)
    logs = sorted(root.glob("Chapter_*.log"))
    out = module.replay(logs, [root.parent / "Datasets", root])
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
    words = {c.split()[0] for c in notrun.command}
    assert words <= set(DECLINED) | FOLLOWS, sorted(words - set(DECLINED) - FOLLOWS)
    # the estimation commands of the chapters all run
    for command in ("teffects", "psmatch2", "boottest", "sdid", "synth", "logit"):
        assert command not in words


def test_no_number_lacks_a_counterpart(frame):
    assert (frame.status == "no output").sum() == 0


def _row(frame, log, start, what):
    hit = frame[
        (frame.log == log) & frame.command.str.startswith(start) & (frame.what == what)
    ]
    assert len(hit) >= 1, (log, start, what)
    return hit


def test_synthetic_difference_in_differences(frame):
    # method(sdid), method(sc), method(did): -15.60383, -19.61966, -27.34911
    rows = _row(frame, "Chapter_04.log", "sdid cigsale", "ATT")
    assert len(rows) == 3 and (rows.status == "ok").all()


def test_probit_weighting_and_matching(frame):
    # teffects ipw (..., probit), atet: 1177.529 (616.7019)
    for what in ("effect", "se"):
        rows = _row(frame, "Chapter_03.log", "teffects ipw", what)
        assert (rows.status == "ok").all()
        rows = _row(frame, "Chapter_03.log", "teffects psmatch", what)
        assert len(rows) == 2 and (rows.status == "ok").all()


def test_boottest_statistic(frame):
    rows = _row(frame, "Chapter_04.log", "boottest treat2016", "t")
    assert (rows.status == "ok").all()


def test_resampled_data_are_marked_not_compared(frame):
    # chapter 2 simulates and bootstraps; numpy's draws are not Stata's
    random = frame[frame.status == "random"]
    assert set(random.log) == {"Chapter_02.log"}
