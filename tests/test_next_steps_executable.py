"""``result.next_steps()`` must hand an agent calls that run.

An agent treats each ``action`` as its next command. Before this test the
list recommended calls that could not work as written:
``sp.rdbalance(covariates=...)`` (the argument is ``covs``),
``sp.hausman_test(result_fe, result_re)`` (it takes data and columns),
``sp.blp_test(result)`` without its data, ``sp.cate_summary`` on a
partially linear DML fit, ``sp.evalue(result.estimate, result.se)`` on a
DiD on log employment (refused: a risk ratio must be positive) and on an
RD jump (accepted, and wrong), and RD steps with ``x='...'`` although the
fit records its running variable.

Contract: every step marked ``executable`` runs verbatim with ``result``
bound to the fit and ``df`` to the data it was fitted on. Steps that need
something the fit does not know keep a ``'...'`` placeholder and are
marked ``executable: False``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _fits():
    mp = sp.datasets.mpdta()
    mpe = mp.assign(tt=mp["first_treat"].replace(0, np.nan))
    nsw = sp.datasets.nsw_dw()
    lee = sp.datasets.lee_2008_senate()
    card = sp.datasets.card_1995()
    ca = sp.datasets.california_prop99()
    covs = ["age", "education", "re74", "re75"]
    return {
        "callaway_santanna": (
            mp,
            lambda: sp.callaway_santanna(
                mp, y="lemp", g="first_treat", t="year", i="countyreal"
            ),
        ),
        "event_study": (
            mpe,
            lambda: sp.event_study(
                mpe, y="lemp", treat_time="tt", time="year", unit="countyreal"
            ),
        ),
        "rdrobust": (lee, lambda: sp.rdrobust(lee, y="y", x="x", c=0)),
        "regress": (
            nsw,
            lambda: sp.regress(
                "re78 ~ treat + age + education + re74 + re75", data=nsw
            ),
        ),
        "ivreg": (card, lambda: sp.ivreg("lwage ~ (educ ~ nearc4) + exper", data=card)),
        "match": (nsw, lambda: sp.match(nsw, y="re78", treat="treat", covariates=covs)),
        "aipw": (nsw, lambda: sp.aipw(nsw, y="re78", treat="treat", covariates=covs)),
        "dml": (nsw, lambda: sp.dml(nsw, y="re78", treat="treat", covariates=covs)),
        "metalearner": (
            nsw,
            lambda: sp.metalearner(
                nsw,
                y="re78",
                treat="treat",
                covariates=covs,
                learner="t",
                n_bootstrap=20,
            ),
        ),
        "synth": (
            ca,
            lambda: sp.synth(
                ca,
                outcome="cigsale",
                unit="state",
                time="year",
                treated_unit="California",
                treatment_time=1989,
            ),
        ),
    }


_FITS = _fits()


@pytest.mark.parametrize("name", sorted(_FITS))
def test_every_executable_step_runs_verbatim(name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # export steps write files
    df, fit = _FITS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit()
        steps = result.next_steps()
        executable = [s for s in steps if s["executable"]]
        assert executable, f"{name}: no step can be run as written"
        failures = []
        for step in executable:
            try:
                eval(step["action"], {"sp": sp, "result": result, "df": df})
            except Exception as exc:  # pragma: no cover - reported below
                failures.append(f"{step['action']} -> {type(exc).__name__}: {exc}")
    capsys.readouterr()
    assert not failures, f"{name}: " + "\n".join(failures)
    for step in steps:
        assert step["executable"] == ("..." not in step["action"]), step


def test_recorded_columns_are_filled_in():
    df, fit = _FITS["rdrobust"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actions = [s["action"] for s in fit().next_steps()]
    assert "sp.rd_honest(data=df, y='y', x='x', c=0)" in actions


def test_evalue_is_not_suggested_for_designs_it_does_not_bound():
    # DiD and RD do not rest on no-unmeasured-confounding; an E-value of
    # their estimate answers a question the design does not ask.
    for name in ("callaway_santanna", "rdrobust", "synth"):
        df, fit = _FITS[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actions = " ".join(s["action"] for s in fit().next_steps())
        assert "evalue" not in actions, name


def test_did_steps_point_to_the_rambachan_roth_sets():
    df, fit = _FITS["callaway_santanna"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actions = [s["action"] for s in fit().next_steps()]
    assert "sp.honest_did(result)" in actions
    assert not any("sensitivity_rr" in a for a in actions)
