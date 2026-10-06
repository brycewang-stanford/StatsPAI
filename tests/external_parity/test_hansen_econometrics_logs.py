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
    "Chapter_10.log": 46,
    "Chapter_11.log": 116,
    "Chapter_12.log": 767,
    "Chapter_14.log": 468,
    "Chapter_15.log": 981,
    "Chapter_16.log": 659,
    "Chapter_17.log": 342,
    "Chapter_18.log": 507,
    "Chapter_20.log": 32,
    "Chapter_23.log": 16,
    "Chapter_24.log": 40,
    "Chapter_25.log": 68,
    "Chapter_26.log": 190,
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
    "reg edu black smsa married i.yob i.region i.state": (
        "i.qob#i.yob and i.qob#i.state share qob, whose main effect is "
        "absent: Stata and the formula drop different cells, so the "
        "coefficients are not Stata's one by one"
    ),
    "testparm i.qob#i.yob i.qob#i.state": "follows the regression above",
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
    assert len(notrun) <= 5


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
    # Blanchard-Perotti: short-run restrictions with a quadratic trend. The
    # replay compares no number of the `svar` table itself; the structural
    # responses computed from it are the check.
    for shock in ("gov", "tax"):
        hit = _rows(frame, "Chapter_15.log", f"irf table sirf, impulse({shock})")
        assert len(hit) >= 17 and (hit.status == "ok").all()


def test_threshold_model_of_figure_23_3():
    """Card, Mas and Rothstein's tipping model as the book fits it
    (``figure23_3.R``): MSA fixed effects, a jump and a change of slope in
    the minority share, 100 grid points, 99% interval. The numbers are the
    ones that program prints; its covariance has no small-sample factor."""
    import numpy as np

    import statspai as sp

    data = sp.read_data(str(Path(ROOT) / "CMR2008.dta"))
    data = data[data["samp_70"] == 1]
    keep = [
        "msa", "chg_white_7080", "fr_min_70", "unem_70", "pubtran_70",
        "faminc_70", "vac_70", "rent_70", "oneunit_70",
    ]  # fmt: skip
    data = data[keep].dropna().copy()
    data["fr_min_70_sq"] = data["fr_min_70"] ** 2
    fit = sp.threshold(
        "chg_white_7080 ~ fr_min_70 + fr_min_70_sq + unem_70 + pubtran_70 "
        "+ faminc_70 + vac_70 + rent_70 + oneunit_70",
        data,
        "fr_min_70",
        regime=["fr_min_70"],
        grid=100,
        absorb="msa",
        cluster="msa",
        alpha=0.01,
    )
    info = fit.model_info
    n, groups, k = len(data), info["n_clusters"], len(fit.params)
    assert (n, groups) == (35656, 104)
    assert np.isclose(info["threshold"], 0.197894376637, rtol=1e-10)
    assert np.allclose(info["threshold_ci"], (0.197894376637, 0.208681391044))
    assert np.isclose(fit.diagnostics["Residual SS"] / n, 3766.70153667, rtol=1e-10)
    factor = np.sqrt(groups / (groups - 1) * (n - 1) / (n - k - 1))
    printed = {
        "above:fr_min_70": (-74.12890644125, 42.62738178697),
        "fr_min_70": (-54.42579737974, 28.76921252410),
        "fr_min_70_sq": (142.26834955528, 23.85218117459),
        "unem_70": (-81.06343605628, 38.83130422449),
        "vac_70": (324.89988093385, 40.19216581109),
        "oneunit_70": (-4.78595972893, 9.49959774095),
    }
    for name, (b, se) in printed.items():
        assert np.isclose(fit.params[name], b, rtol=1e-9), name
        assert np.isclose(fit.std_errors[name] / factor, se, rtol=1e-9), name
    # the program centres the slope change at the threshold; here it is not
    shift = fit.params["above"] + info["threshold"] * fit.params["above:fr_min_70"]
    assert np.isclose(shift, -11.64998623084, rtol=1e-9)


def test_choice_models_of_chapter_26(frame):
    """Conditional and nested logit are compared digit for digit (nested
    logit to Stata's `ml` tolerance). Stata simulates the multinomial
    probit and the mixed logit: a coefficient counts as reproduced within
    2% of Stata's standard error, a standard error within 2%."""
    rows = frame[frame.log == "Chapter_26.log"]
    assert (rows.status == "ok").all()
    for start, least in (
        ("cmclogit", 20),
        ("nlogit choice", 26),
        ("cmmprobit", 50),
        ("cmmixlogit", 24),
        ("estat cov", 6),
        ("estat cor", 6),
        ("margins, dydx(", 36),
    ):
        assert rows.command.str.startswith(start).sum() >= least, start


def test_series_cross_validation_of_figure_20_6():
    """The cross-validation sums printed by the book's `figure20_6.R`:
    polynomial orders 1 to 8 for college-educated white and Black women."""
    import numpy as np

    import statspai as sp

    data = sp.read_data(str(Path(ROOT) / "cps09mar.dta"))
    data = data[(data["female"] == 1) & (data["education"] == 16)].copy()
    data["lwage"] = np.log(data["earnings"] / (data["hours"] * data["week"]))
    data["exper"] = data["age"] - data["education"] - 6
    printed = {
        1: [1261.7069898193, 1226.8063686149, 1224.7454086314, 1225.8652145768,
            1226.120941825, 1227.2771566001, 1228.1442711188, 1232.9528450091],
        2: [125.16576004144, 124.34593921719, 124.6221802564, 124.64677655466,
            125.10932430571, 125.11906222811, 125.76412784607, 129.88436227138],
    }  # fmt: skip
    for race, order in ((1, 3), (2, 2)):
        fit = sp.series("lwage ~ 1", data[data["race"] == race], "exper")
        assert fit.model_info["order"] == order
        assert np.allclose(fit.model_info["cv"]["cv"], printed[race], rtol=1e-11)
