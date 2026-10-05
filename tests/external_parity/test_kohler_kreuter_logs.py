"""Replay the chapter logs of Kohler, Kreuter and Haensch, *Data Analysis
Using Stata* (4th ed.) through ``sp.stata`` and compare every number.

The do-files are in ``kohler_kreuter_syllabus/`` next to this file (its
README says how the logs are made). Neither the book's datasets nor the
logs are redistributed. Point ``STATSPAI_KK4E_DIR`` at the folder holding
``ch01.log`` ... ``ch12.log`` and the datasets to run this; it is skipped
otherwise.

    STATSPAI_KK4E_DIR=/path/to/run \\
        pytest tests/external_parity/test_kohler_kreuter_logs.py

What the replay found is in
``docs/dev/2026-10-05-kohler-kreuter-4e-review.md``.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_KK4E_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_KK4E_DIR to the chapter logs and datasets of the book",
)

#: Numbers reproduced per chapter log on the day this test was written. A
#: drop means a command stopped running or stopped being compared.
REPRODUCED = {
    "ch01.log": 176,
    "ch03.log": 378,
    "ch05.log": 583,
    "ch07.log": 1108,
    "ch08.log": 870,
    "ch09.log": 1275,
    "ch10.log": 589,
    "ch11.log": 178,
    "ch12.log": 115,
}

#: Printed numbers StatsPAI does not reproduce, by the start of the
#: command, each with the reason. Anything else that differs fails.
DIFFERENT = {
    "summarize income": (
        "follows `mvencode income, mv(.c=0)`, which is declined: the data "
        "do not keep which rows held .c"
    ),
    "summarize hhn hhrank": "cuminc is a running sum of that same income",
    "summarize len comma": (
        "mdb.dta stores its names in Latin-1, so an umlaut is one byte in "
        "Stata's strlen() and two in the UTF-8 text pandas hands over"
    ),
    "swilk rent": (
        "2,160 observations: outside the range (4 to 2,000) for which the "
        "approximation of W is defined; V differs in the sixth digit"
    ),
    "svy: regress": "the model F test of svy: regress is not computed",
    "linktest": (
        "Stata stores _hat in single precision before refitting, so its "
        "coefficients move in the seventh digit; p-values are not compared"
    ),
    "regress D.lsat D.age, noconstant": (
        "the regressor is a constant: Stata reports the uncentered "
        "R-squared of a noconstant model, StatsPAI the centered one of a "
        "model that holds a constant (as statsmodels does)"
    ),
    "logit survived i.class men age": (
        "one coefficient differs by five units of the seventh digit: the "
        "two optimisers stop at different points"
    ),
    "margins, dydx(age_c) over(east)": "the header's number of observations",
    'display "`typ\'"': (
        "the storage type of a numeric variable with missing values is not "
        "kept by the frame: Stata says long, the frame holds float64"
    ),
    "display _rc": (
        "the replay strips `capture` before running a command, so _rc is "
        "that of an earlier line"
    ),
}

#: Commands sp.stata declines on these logs, by their start, with the reason.
DECLINED = {
    "bysort sex edu: summarize": "edu holds .a and .b: Stata makes a group of each",
    "count if income == .c": "one kind of extended missing value",
    "mvencode income, mv(.c=0)": "one kind of extended missing value",
    "tabulate edu edu3, missing": "edu holds .a and .b: one row each in Stata",
    "collapse (mean) income [aweight = xweights], by(sex edu)": "by(edu) as above",
    "misstable": "counts `.` apart from .a-.z; `patterns` is not implemented",
    "egen inc_rank_u": "rank(), unique breaks ties arbitrarily in Stata",
    "anova": "only the one-way layout (oneway) is implemented",
    "tabstat income, statistics(mean sd) by(state) missing": "by() missing group",
    "mi ": "multiple imputation by chained equations draws random numbers",
    "test _b[c.income": "tests after mean / proportion are not implemented",
    "lincom _b[c.income": "tests after mean / proportion are not implemented",
    "svy: logit pia": "the outcome does not vary: Stata stops too (r(2000))",
    "teffects ra (survived men age) (third), pomeans": "pomeans is not translated",
    "tebalance summarize": "after teffects ipw: implemented after psmatch only",
    "nestreg": "nested-model F tests are not implemented",
    "cc ": "epitab tables are not translated",
    "cs ": "epitab tables are not translated",
    "tabodds": "epitab tables are not translated",
    "margins east, pwcompare": "pairwise comparisons of margins",
    "margins, dydx(yedu)": "margins after mlogit / ologit",
    "predict pm1": "predicted probabilities of every outcome after mlogit",
    "predict po1": "predicted probabilities of every outcome after ologit",
    "test [1]yedu": "tests across equations of mlogit",
    "import ": "sp.stata does not read files; pass the DataFrame",
    "infile": "sp.stata does not read files; pass the DataFrame",
    "infix": "sp.stata does not read files; pass the DataFrame",
    "input str10": "the replay reads numeric `input` blocks only",
    "merge 1:1 pid using _m_income, keep(match) nogenerate keepusing(yedu) update": (
        "merge, update is not implemented"
    ),
    "statsby": "the statsby prefix is not implemented",
    "p2": "an ado-file of the book; programs are run when they are defined in the text",
    "xi i.edu, noomit": "xi, noomit is not implemented",
    # variables or data an earlier declined command would have made
    "summarize$": "follows a declined import",
    "summarize kx kd": "kdensity, generate() draws no graph and makes no variable",
    "summarize pm1-pm6": "follows the declined predict",
    "summarize po1-po4": "follows the declined predict",
    "summarize _Iedu*": "follows the declined xi",
    "summarize score": "follows the declined input",
    "summarize nosuchvar": "no such variable: Stata stops too (r(111))",
    "count if name": "follows the declined input",
}


@pytest.fixture(scope="module")
def frame():
    script = Path(__file__).resolve().parents[2] / "scripts" / "stata_log_replay.py"
    spec = importlib.util.spec_from_file_location("stata_log_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(ROOT)
    out = module.replay(sorted(root.glob("ch*.log")), [root])
    out["log"] = out["file"].str.split("/").str[-1]
    return out


def _listed(command, ledger):
    """Whether a ledger entry covers the command: by its start, or the
    whole command for an entry that ends with ``$``."""
    for start in ledger:
        if start.endswith("$"):
            if command.strip() == start[:-1]:
                return True
        elif command.startswith(start):
            return True
    return False


def test_every_log_is_replayed(frame):
    assert set(frame["log"]) == set(REPRODUCED)


def test_numbers_are_reproduced(frame):
    ok = frame[frame["status"] == "ok"].groupby("log").size().to_dict()
    short = {k: (ok.get(k, 0), v) for k, v in REPRODUCED.items() if ok.get(k, 0) < v}
    assert not short, f"fewer numbers reproduced than before (now, then): {short}"


def test_differences_are_the_documented_ones(frame):
    diff = frame[frame["status"] == "DIFF"]
    new = sorted({c for c in diff["command"] if not _listed(c, DIFFERENT)})
    assert not new, "\n".join(new)


def test_declined_commands_are_the_documented_ones(frame):
    declined = frame[frame["status"] == "NOT RUN"]
    new = sorted({c for c in declined["command"] if not _listed(c, DECLINED)})
    assert not new, "\n".join(new)
