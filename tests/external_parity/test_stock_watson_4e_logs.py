"""Replay the Stata logs of Stock & Watson, *Introduction to Econometrics*
(4th edition), through ``sp.stata`` and compare every printed number.

The replication files on the authors' site include, for chapters 2 to 13,
the Stata logs that produced the book's tables: genuine Stata output for
OLS with robust and clustered errors, fixed effects, probit and logit, 2SLS,
F tests, t tests and detailed summary statistics on nine public datasets. ``scripts/stata_log_replay.py``
runs each logged command and compares what Stata printed with what StatsPAI
returns, to the precision Stata printed it.

The files are not redistributed here. Point ``STATSPAI_SW4E_DIR`` at a
folder holding the unzipped chapter folders and ``SW_4E_Replication_Data``
(from https://www.princeton.edu/~mwatson/Stock-Watson_4E/) to run this; it
is skipped otherwise.

    STATSPAI_SW4E_DIR=/path/to/files \
        pytest tests/external_parity/test_stock_watson_4e_logs.py
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_SW4E_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_SW4E_DIR to the Stock & Watson 4E replication files",
)

#: Numbers reproduced per log on the day this test was written. A drop means
#: a command stopped running or stopped being compared.
REPRODUCED = {
    "chapter2/ch2_4e_cps_earnings_box.log": 70,
    "chapter2/ch2_4e_djia_box.log": 2,
    "chapter4/SW_4E_ch4.log": 8,
    "chapter5/SW_4E_ch5.log": 8,
    "chapter5/ch5_4e_economic_value_box.log": 38,
    "chapter6/ch6_7_ex1_4.log": 53,
    "chapter6/ch6_caschools.log": 35,
    "chapter7/ch7_caschools.log": 61,
    "chapter9/ch9.log": 160,
    "chapter10/ch10.log": 178,
    "chapter11/ch11.log": 457,
    "chapter12/ch12.log": 82,
    "chapter13/ch13.log": 148,
}

#: Commands sp.stata declines, each for a stated reason. Anything else that
#: is not run is a regression.
DECLINED = ("pctile", "ch6_caschools")


@pytest.fixture(scope="module")
def frame():
    script = Path(__file__).resolve().parents[2] / "scripts" / "stata_log_replay.py"
    spec = importlib.util.spec_from_file_location("stata_log_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(ROOT)
    return module.replay(module.collect([str(root)]), [root])


def test_no_printed_number_differs(frame):
    diff = frame[frame.status == "DIFF"]
    assert diff.empty, diff[["file", "command", "what", "stata", "ours"]].to_string()


@pytest.mark.parametrize("log, expected", sorted(REPRODUCED.items()))
def test_each_log_reproduces_its_numbers(frame, log, expected):
    ok = int(((frame.file == log) & (frame.status == "ok")).sum())
    assert ok >= expected, f"{log}: {ok} numbers reproduced, {expected} before"


def test_only_documented_commands_are_declined(frame):
    notrun = frame[frame.status == "NOT RUN"]
    stray = [c for c in notrun.command if not any(key in c for key in DECLINED)]
    assert not stray, stray


def test_the_only_missing_output_is_the_fixed_effects_constant(frame):
    missing = frame[frame.status == "no output"]
    assert set(missing.what) <= {"b[_cons]", "se[_cons]"}
    assert missing.command.str.startswith("xtreg").all()
