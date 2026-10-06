"""Replay the Stata logs of Qiu Jiaping, *Practical Econometric Methods
for Causal Inference*, through ``sp.stata`` and compare every number.

The book goes from regression and standard errors (heteroskedasticity,
serial correlation, clustering) through randomised experiments, propensity
score matching (Becker and Ichino's ``pscore`` / ``attnd``, ``psmatch2``,
``teffects psmatch``), panel data, difference in differences, instrumental
variables and selection models to regression discontinuity. Its Stata code
ships as one text file without output, so the answer key is a log made by
running it in Stata 18 (``qiu_jiaping_prepare.py`` next to this file says
how, and what was repaired in the transcription).
``scripts/stata_log_replay.py`` then runs every logged command through one
``sp.stata`` session and compares what Stata printed with what StatsPAI
returns, to the precision Stata printed it.

Neither the code nor the data are redistributed here. Point
``STATSPAI_QIU_DIR`` at the folder holding ``Chapter_NN.log``,
``Chapter_NN.do`` and the ``.dta`` files to run this; it is skipped
otherwise.

    STATSPAI_QIU_DIR=/path/to/run \\
        pytest tests/external_parity/test_qiu_jiaping_logs.py

The numbers this replay found wrong are pinned, on data that can be
redistributed, in ``tests/reference_parity/test_qiu_jiaping_methods_stata.py``
and ``tests/test_stata_qiu_jiaping_syntax.py``.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_QIU_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_QIU_DIR to the logs of Qiu Jiaping's Stata code",
)

#: Numbers reproduced per chapter log on the day this test was written. A
#: drop means a command stopped running or stopped being compared.
REPRODUCED = {
    "Chapter_03.log": 281,
    "Chapter_04.log": 26,
    "Chapter_05.log": 573,
    "Chapter_06.log": 309,
    "Chapter_07.log": 81,
    "Chapter_08.log": 110,
    "Chapter_09.log": 238,
    "Chapter_10.log": 29,
    "Chapter_11.log": 145,
    "Chapter_12.log": 215,
}

#: Printed numbers StatsPAI does not reproduce, each with the reason. The
#: key is (log, start of the command); anything else that differs fails.
DIFFERENT = {
    ("Chapter_12.log", "reg mdemsharenext win X1 X2 X3 X4"): (
        "The model F with 15 clusters and nine restrictions, on powers of "
        "the running variable up to the fourth: the covariance matrix of "
        "the nine slopes has a condition number of 5e7. Stata prints "
        "42621.11; the statistic computed in 60-digit arithmetic from the "
        "same rows is 42620.8909, and StatsPAI returns 42620.8899. The "
        "coefficients and standard errors agree to every printed digit."
    ),
}

#: Numbers Stata prints that the StatsPAI result has no counterpart for.
NO_COUNTERPART = {
    ("Chapter_04.log", "reg score"): (
        "A regression on the constant alone. Stata prints F(0, 29) = 0.00 "
        "with a missing p-value; a test of no restrictions has no "
        "statistic, and StatsPAI reports it as missing."
    ),
}

#: Commands sp.stata declines.
DECLINED: dict = {}


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


def test_the_matching_chapter_is_compared_number_by_number(frame):
    """pscore, attnd, psmatch2, teffects psmatch and pstest: the commands
    of chapter 6 all gave Stata's numbers, and the three matching commands
    the same effect."""
    six = frame[frame.log == "Chapter_06.log"]
    for word in ("pscore", "attnd", "psmatch2", "teffects", "pstest"):
        rows = six[six.command.str.split().str[0] == word]
        assert len(rows) and (rows.status == "ok").all(), word
    att = six[six.what.isin(["ATT", "effect"]) & (six.status == "ok")]
    assert att.stata.round(3).nunique() == 1 and len(att) >= 6


def test_simulated_data_are_marked_not_compared(frame):
    # chapter 4 simulates its heteroskedastic and AR(1) errors; chapter 6
    # bootstraps. numpy's draws are not Stata's
    random = frame[frame.status == "random"]
    assert set(random.log) <= {"Chapter_04.log", "Chapter_06.log"}
