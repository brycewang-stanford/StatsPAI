"""Replay the Stata programs of Hansen's *Econometrics* through ``sp.stata``
and compare every number.

The book (Princeton University Press, 2022) comes with sixteen Stata
do-files, chapters 3 to 26: least squares and its covariance estimators,
constrained regression, the jackknife and the bootstrap, principal
components and factors, instrumental variables, autoregressions, VARs and
structural VARs, unit roots and cointegration, panels, difference in
differences, nonlinear least squares, quantile regression, binary and
multiple choice. The do-files carry no output, so the answer key is a log
made by running them in Stata (``hansen_econometrics_prepare.py`` next to
this file says how). ``scripts/stata_log_replay.py`` then runs every logged
command through one ``sp.stata`` session and compares what Stata printed
with what StatsPAI returns, to the precision Stata printed it.

Neither the programs nor the data are redistributed here. Point
``STATSPAI_HANSEN_DIR`` at the folder holding ``Chapter_NN.log`` and the
``.dta`` files to run this; it is skipped otherwise.

    STATSPAI_HANSEN_DIR="/path/to/14-Hansen-Econometrics/run" \\
        pytest tests/external_parity/test_hansen_econometrics_logs.py

The replay takes about fifteen minutes: chapters 12 and 24 jackknife and
bootstrap instrumental-variable and quantile regressions. What it found is
in ``docs/dev/2026-10-05-hansen-econometrics-review.md``.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("STATSPAI_HANSEN_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_HANSEN_DIR to the logs of Hansen's Stata programs",
)

#: Numbers reproduced per chapter log on the day this test was written. A
#: drop means a command stopped running or stopped being compared.
REPRODUCED = {
    "Chapter_03.log": 39,
    "Chapter_04.log": 126,
    "Chapter_08.log": 36,
    "Chapter_10.log": 38,
    "Chapter_11.log": 116,
    "Chapter_12.log": 660,
    "Chapter_14.log": 468,
    "Chapter_15.log": 981,
    "Chapter_16.log": 659,
    "Chapter_17.log": 206,
    "Chapter_18.log": 415,
    "Chapter_20.log": 32,
    "Chapter_23.log": 16,
    "Chapter_24.log": 30,
    "Chapter_25.log": 68,
    "Chapter_26.log": 35,
}

#: Printed numbers StatsPAI does not reproduce, each with the reason. The
#: key is (log, start of the command); anything else that differs fails.
DIFFERENT = {
    ("Chapter_23.log", "nl (gdp ="): (
        "A threshold parameter makes the sum of squares non-smooth. "
        "StatsPAI stops at a residual sum of squares of 3738.2673, Stata at "
        "3738.271; the threshold agrees to five digits, the slopes to three."
    ),
    ("Chapter_12.log", "estat overid, forcenonrobust"): (
        "One of the five calls. With `perfect`, an instrument (age) is an "
        "exact combination of the fitted regressors, and Stata's robust "
        "score statistic then depends on the order of the instruments: "
        "5.379 as written, 6.467 and 7.866 in other orders. StatsPAI "
        "returns 7.866, Hansen's J at the 2SLS residuals. The Sargan and "
        "Basmann statistics of the same call agree."
    ),
    ("Chapter_17.log", "xi: xthtaylor"): (
        "Table 17.2, last column. The panel is unbalanced and the model has "
        "year dummies. Stata's instruments are the unit means of the dummies "
        "it kept, so its estimates depend on the year it omits (1962 here). "
        "StatsPAI adds the constant to the instruments, which makes the fit "
        "the same for every base year; leaving the constant out and "
        "omitting 1962 reproduces Stata's column to the last printed digit. "
        "The variance components and the coefficients agree to four or five "
        "digits either way."
    ),
}

#: Commands sp.stata declines on these logs, by their start, with the reason.
DECLINED = {
    "mata{": "mata is not translated (the block is the minimum distance estimator)",
    "mat list b_emd": "defined inside the mata block",
    "mat list std_emd": "defined inside the mata block",
    "estat bootstrap": "percentile / BCa intervals of the last bootstrap",
    "disp c, pc": "two expressions in one display",
    "reg edu black smsa married i.yob i.region": (
        "i.qob#i.yob without the main effect of qob: Stata fits one "
        "indicator per cell, which the formula does not reproduce "
        "coefficient by coefficient"
    ),
    "testparm i.qob#i.yob": "follows the regression above",
    "matrix list e(Sigma)": "display of a stored matrix",
    "xi: xtdpd": "the dgmmiv() / lgmmiv() grammar is not translated",
    "cmmprobit": "no multinomial probit",
    "cmmixlogit": "choice-model commands are not translated",
    "nlogitgen": "choice-model commands are not translated",
    "nlogit": "choice-model commands are not translated",
    "margins, dydx(": "after cmmprobit / cmmixlogit, which were not fitted",
    "estat covariance": "after cmmprobit",
    "estat correlation": "after cmmprobit",
}


@pytest.fixture(scope="module")
def frame():
    script = Path(__file__).resolve().parents[2] / "scripts" / "stata_log_replay.py"
    spec = importlib.util.spec_from_file_location("stata_log_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(ROOT)
    out = module.replay(sorted(root.glob("Chapter_*.log")), [root])
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
    # of the five overid calls of chapter 12 only the score of one differs
    overid = diff[diff.log == "Chapter_12.log"]
    assert set(overid.what) == {"Score statistic", "Score p"} and len(overid) == 2


@pytest.mark.parametrize("log, expected", sorted(REPRODUCED.items()))
def test_each_log_reproduces_its_numbers(frame, log, expected):
    ok = int(((frame.log == log) & (frame.status == "ok")).sum())
    assert ok >= expected, f"{log}: {ok} numbers reproduced, {expected} before"


def test_only_documented_commands_are_declined(frame):
    notrun = frame[frame.status == "NOT RUN"]
    stray = [
        (r.log, r.command)
        for r in notrun.itertuples()
        if not any(r.command.startswith(d) for d in DECLINED)
    ]
    assert not stray, stray
    # outside chapter 26 (choice models) a handful of lines are left
    assert (notrun.log != "Chapter_26.log").sum() <= 22


def _rows(frame, log, start, what=None):
    hit = frame[(frame.log == log) & frame.command.str.startswith(start)]
    if what is not None:
        hit = hit[hit.what == what]
    assert len(hit) >= 1, (log, start, what)
    return hit


def test_random_effects_with_industry_dummies(frame):
    # Table 17.2, column 2: counted regressors that are constant within
    # firm as within parameters before
    rows = _rows(frame, "Chapter_17.log", "xtreg inva L.vala L.debta L.cfa nyseamex")
    assert len(rows) > 46 and (rows.status == "ok").all()
    # sigma_u, sigma_e and rho of the same fit
    assert {"sigma_u", "sigma_e", "rho"} <= set(rows.what)


def test_dickey_fuller_p_value_of_the_participation_rate(frame):
    # statistic +1.145 with a trend; Stata prints 1.0000
    rows = _rows(frame, "Chapter_16.log", "dfuller Y9", "p-value")
    assert (rows.status == "ok").all()


def test_cubic_in_class_size(frame):
    rows = _rows(frame, "Chapter_20.log", "ivregress 2sls avgverb (c1 c2 c3 cd1")
    assert (rows.status == "ok").all()


def test_angrist_krueger_with_interacted_instruments(frame):
    rows = _rows(frame, "Chapter_12.log", "ivregress 2sls logwage", "b[edu]")
    assert len(rows) == 3 and (rows.status == "ok").all()


def test_constrained_and_nonlinear_regression(frame):
    rows = _rows(frame, "Chapter_08.log", "cnsreg lndY")
    assert len(rows) == 10 and (rows.status == "ok").all()
    for start in ("nl ( risk", "nl (y ="):
        assert (_rows(frame, "Chapter_23.log", start).status == "ok").all()


def test_principal_components_and_factors(frame):
    for start in ("pca wordscore", "factor wordscore"):
        assert (_rows(frame, "Chapter_11.log", start).status == "ok").all()


def test_jackknife_is_reproduced_and_bootstrap_is_marked_random(frame):
    jack = frame[
        frame.command.str.contains("jackknife") & frame.what.str.startswith("se[")
    ]
    assert len(jack) >= 9 and (jack.status == "ok").all()
    boot = frame[
        frame.command.str.contains("bootstrap") & frame.what.str.startswith("se[")
    ]
    assert len(boot) >= 10 and (boot.status == "random").all()


def test_impulse_responses_of_structural_vars(frame):
    rows = frame[(frame.log == "Chapter_15.log") & frame.command.str.startswith("irf table")]
    assert len(rows) > 200 and (rows.status == "ok").all()
    # Blanchard-Perotti: short-run restrictions with a quadratic trend
    assert (_rows(frame, "Chapter_15.log", "svar gov tax gdp").status == "ok").all()
