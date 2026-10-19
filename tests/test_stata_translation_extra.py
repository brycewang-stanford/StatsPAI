"""Stata commands translated from 2026-10-19: ``stcox``, ``stcrreg``,
``eventstudyinteract`` and ``power twomeans | onemean | pairedmeans``.

Reference numbers are Stata 18 MP, ``set type double``, on the data built
below (written with ``%.17g`` and read back with ``import delimited ...,
asdouble``), printed with ``%21.16e``::

    stset time, failure(d)
    stcox x1 x2                         // breslow
    stcox x1 x2, efron
    stcox x1 x2, vce(cluster g)
    stcox x1 x2, strata(st)
    stset time, failure(cause == 1)
    stcrreg x1 x2, compete(cause == 2) tolerance(1e-12) ltolerance(1e-14) ///
        nrtolerance(1e-14)    // Stata's default stops two iterations early

    eventstudyinteract y rel_m7 ... rel_m2 rel_p0 ... rel_p6, cohort(g) ///
        control_cohort(nt) absorb(i.id i.year) vce(cluster id) [covariates(x)]

    power twomeans 0 0.5, sd(1) power(0.8)      // N = 128
    power twomeans 0 0.5, sd(1) n(100)          // power .6968934055333842
    power twomeans 0 0.5, sd(2) n(100) alpha(0.01)   // .0890843265940530
    power twomeans 0 0.5, sd(1) power(0.9) onesided  // N = 140
    power onemean 0 0.5, sd(1) power(0.8)       // N = 34
    power pairedmeans 0 0.5, sddiff(1) power(0.8)    // N = 34

Tolerance 1e-8: above the 15 printed digits and the Newton tolerance of
the partial likelihood, far below any difference between estimators.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RTOL = 1e-8

COX = {
    "stcox x1 x2": {
        "x1": (4.02681810467657e-01, 8.36005232117656e-02),
        "x2": (-5.65390599113054e-01, 1.60569545553688e-01),
    },
    "stcox x1 x2, efron": {
        "x1": (4.21351972926047e-01, 8.36641208094031e-02),
        "x2": (-5.91105068614476e-01, 1.60561680044026e-01),
    },
    "stcox x1 x2, vce(cluster g)": {
        "x1": (4.02681810467657e-01, 6.79504565287975e-02),
        "x2": (-5.65390599113054e-01, 1.82227325596649e-01),
    },
    "stcox x1 x2, strata(st)": {
        "x1": (4.03083556553255e-01, 8.45581870461380e-02),
        "x2": (-5.52889978073537e-01, 1.60874018694551e-01),
    },
}
CRR = {
    "x1": (3.66087469668843e-01, 1.02985118533208e-01),
    "x2": (-6.64920174127074e-01, 2.09404226759654e-01),
}
ESI = {
    -7: (3.95735794904471e-01, 3.00883682985597e-01),
    -2: (-2.23837721859125e-01, 2.02025512567819e-01),
    0: (2.96299182723924e-01, 1.91666652691033e-01),
    3: (1.56298549138028e00, 2.54210340458480e-01),
    6: (3.29101292088969e00, 3.86457143006649e-01),
}
ESI_COV = {
    -7: (3.78796453308406e-01, 2.75399951694918e-01),
    0: (2.57892208263477e-01, 1.86854191871255e-01),
    6: (3.35469774466387e00, 3.64889684576750e-01),
}
RELS = [f"rel_m{k}" for k in range(7, 1, -1)] + [f"rel_p{k}" for k in range(7)]


@pytest.fixture(scope="module")
def surv():
    r = np.random.default_rng(11)
    n = 300
    x1 = r.normal(size=n)
    x2 = r.binomial(1, 0.4, n)
    g = r.integers(1, 31, n)
    st = r.integers(0, 2, n)
    t = np.ceil(r.exponential(1 / np.exp(0.5 * x1 - 0.4 * x2)) * 10)
    c = np.ceil(r.exponential(1.5, n) * 10)
    time = np.minimum(t, c)
    d = (t <= c).astype(int)
    cause = np.where(d == 1, r.choice([1, 2], n, p=[0.6, 0.4]), 0)
    return pd.DataFrame(dict(time=time, d=d, cause=cause, x1=x1, x2=x2, g=g, st=st))


@pytest.fixture(scope="module")
def esi():
    r = np.random.default_rng(3)
    rows = []
    for i in range(150):
        c = r.choice([0, 4, 6, 8])
        a = r.normal()
        for t in range(1, 11):
            te = (t - c + 1) * 0.5 if (c and t >= c) else 0
            x = r.normal()
            rows.append((i, t, c, x, a + 0.1 * t + 0.3 * x + te + r.normal()))
    d = pd.DataFrame(rows, columns=["id", "year", "cohort", "x", "y"])
    d["nt"] = (d.cohort == 0).astype(int)
    d["g"] = d.cohort.where(d.cohort > 0)
    rel = d.year - d.cohort
    for k in range(-7, 7):
        if k != -1:
            name = f"rel_m{-k}" if k < 0 else f"rel_p{k}"
            d[name] = ((rel == k) & (d.cohort > 0)).astype(int)
    return d


def _quiet(script, data=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(script, data=data)


@pytest.mark.parametrize("command", sorted(COX))
def test_stcox_reproduces_stata(surv, command):
    result = _quiet(f"stset time, failure(d)\n{command}", surv)
    for name, (b, se) in COX[command].items():
        assert np.isclose(result.params[name], b, rtol=RTOL, atol=0)
        assert np.isclose(result.std_errors[name], se, rtol=RTOL, atol=0)


def test_stcox_ties_default_is_written_out():
    out = sp.from_stata("stcox x1 x2")
    assert out["arguments"]["ties"] == "breslow"
    assert sp.from_stata("stcox x1 x2, efron")["arguments"]["ties"] == "efron"


def test_stcox_robust_maps_to_the_stata_small_sample_factor():
    """stcox vce(robust) is the sandwich times N/(N-1): sp.cox(ssc='stata').
    The numbers are pinned against Stata 18 in
    tests/reference_parity/test_survival_weighted_parity.py."""
    out = sp.from_stata("stcox x1 x2, vce(robust)")
    assert out["complete"] is True
    assert out["arguments"]["robust"] == "hc0" and out["arguments"]["ssc"] == "stata"
    out = sp.from_stata("stcox x1 x2, vce(cluster g)")
    assert out["arguments"]["cluster"] == "g" and out["arguments"]["ssc"] == "stata"


def test_stcrreg_reproduces_stata(surv):
    result = _quiet(
        "stset time, failure(cause == 1)\nstcrreg x1 x2, compete(cause == 2)", surv
    )
    for k, name in enumerate(("x1", "x2")):
        b, se = CRR[name]
        assert np.isclose(np.asarray(result.params)[k], b, rtol=RTOL, atol=0)
        assert np.isclose(np.asarray(result.bse)[k], se, rtol=RTOL, atol=0)


def test_event_codes_the_call_would_misread_are_refused(surv):
    other = surv.assign(cause=surv["cause"].where(surv.index >= 5, 3))
    with pytest.raises(Exception, match="codes"):
        _quiet(
            "stset time, failure(cause == 1)\nstcrreg x1 x2, compete(cause == 2)",
            other,
        )
    # stcox on a variable with a competing code: sp.cox reads a 0/1 indicator
    with pytest.raises(Exception, match="codes"):
        _quiet("stset time, failure(cause == 1)\nstcox x1 x2", surv)


def _es(result):
    es = result.model_info["event_study"]
    return {int(t): (b, s) for t, b, s in zip(es.relative_time, es.att, es.se)}


@pytest.mark.parametrize("extra, ref", [("", ESI), (" covariates(x)", ESI_COV)])
def test_eventstudyinteract_reproduces_stata(esi, extra, ref):
    line = (
        f"eventstudyinteract y {' '.join(RELS)}, cohort(g) control_cohort(nt) "
        f"absorb(i.id i.year) vce(cluster id){extra}"
    )
    out = sp.from_stata(line)
    assert out["tool"] == "sun_abraham"
    # a single line cannot show that the varlist is every relative period
    assert out["dropped"] == ["varlist"] and out["complete"] is False
    es = _es(_quiet(line, esi))
    for k, (b, se) in ref.items():
        assert np.isclose(es[k][0], b, rtol=RTOL, atol=0), k
        assert np.isclose(es[k][1], se, rtol=RTOL, atol=0), k


def test_eventstudyinteract_with_another_varlist_is_refused(esi):
    """Leaving rel_m7 out of the varlist pools it into the reference in
    eventstudyinteract; sp.sun_abraham would estimate it."""
    line = (
        f"eventstudyinteract y {' '.join(RELS[1:])}, cohort(g) "
        "control_cohort(nt) absorb(i.id i.year) vce(cluster id)"
    )
    with pytest.raises(Exception, match="missing"):
        _quiet(line, esi)


@pytest.mark.parametrize(
    "line, n, power",
    [
        ("power twomeans 0 0.5, sd(1) power(0.8)", 128, None),
        ("power twomeans 0 0.5, sd(1) n(100)", 100, 0.6968934055333842),
        ("power twomeans 0 0.5, sd(2) n(100) alpha(0.01)", 100, 0.0890843265940530),
        ("power twomeans 0 0.5, sd(1) power(0.9) onesided", 140, None),
        ("power onemean 0 0.5, sd(1) power(0.8)", 34, None),
        ("power pairedmeans 0 0.5, sddiff(1) power(0.8)", 34, None),
    ],
)
def test_power_reproduces_stata(line, n, power):
    result = _quiet(line)
    assert int(result.n) == n
    if power is not None:
        assert np.isclose(result.power, power, rtol=1e-8, atol=0)


@pytest.mark.parametrize(
    "line",
    ["power twomeans 0 0.5, sd1(1) sd2(2) n(100)", "power twoproportions 0.1 0.2"],
)
def test_power_designs_outside_the_mapping_are_refused(line):
    assert sp.from_stata(line)["ok"] is False


@pytest.mark.parametrize(
    "command, function",
    [
        ("xtabond2 y L.y x, gmm(L.y)", "xtabond"),
        ("leebounds y d", "lee_bounds"),
        ("lpirf y x", "local_projections"),
        ("rwolf y1 y2, indepvar(d)", "romano_wolf"),
        ("wyoung y1 y2, cmd(regress OUTCOMEVAR d) familyp(d)", "westfall_young"),
        ("spregress y x, ml dvarlag(W)", "sar"),
    ],
)
def test_guidance_names_an_existing_function(command, function):
    out = sp.from_stata(command)
    assert out["ok"] is False
    assert function in out["statspai_functions"]
    assert callable(getattr(sp, function))
