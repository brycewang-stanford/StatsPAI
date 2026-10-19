"""``sp.weakrobust`` in Stata ``weakiv``'s form, and the joint AR test.

Reference: Stata 18 MP, weakiv 2.4.07 / ivreg2 4.1.11 / avar on
``_fixtures/weakiv_fx.csv`` (``_fixtures/_generate_weakiv_Stata.do``).

DGP (``numpy.random.default_rng(20261019)``, rounded to 6dp): 400 obs in 40
clusters ``g``; ``z1..z3`` normal plus a cluster shock; error
``u = N(0,1)(1 + 0.5|z1|) + 0.8 c_g``; ``d1 = .25 z1 + .15 z2 + .05 z3 +
.3 x1 + .6 u + e1``, ``d2 = .05 z1 + .3 z2 + .2 z3 - .2 x1 + .4 u + e2``,
``y = 1 + .5 d1 - .3 d2 + .4 x1 + u``. The instruments are weak, so the sets
are wide, disjoint or unbounded -- the cases worth pinning.

What is pinned and to what:

* AR, K, J, rk and CLR statistics under i.i.d., robust and cluster variances,
  one and two endogenous regressors, and the subset AR statistic: T2 against
  ``weakiv``'s ``e()`` (rel 1e-9; observed <= 1e-12).
* Set endpoints: Stata's own p-value at each StatsPAI endpoint is 0.05 to
  1e-11 (CLR: 1.5e-7, the error of weakiv's Simpson-rule p-value).
* CLR p-values: exact conditional integral. With one regressor weakiv's
  20-node Simpson rule (reproduced here) gives Stata's number from our
  statistic to 1e-12. With several, weakiv simulates AR and K as independent
  chi-squares although AR = K + J; Stata's p-values follow that law, ours
  the correct one (checked by simulation).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp
from statspai.diagnostics._weakiv_lm import SubsetAR, WeakIVLM, clr_pvalue
from statspai.iv.weak_iv_ci import _clr_conditional_pvalue

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "weakiv_fx.csv")
INST = ["z1", "z2", "z3"]
REL = 1e-9
VCOV = {
    "iid": {"test_vcov": "classic"},
    "robust": {"test_vcov": "robust"},
    "cluster": {"cluster": "g"},
}

# (ar, k, j, rk, clr, clr_p, kj_p) from weakiv ... null(0.5)
ONE = {
    "iid": (
        2.069544022507372,
        1.6331876376605441,
        0.43635638484682809,
        38.016065818350519,
        1.6600334089692159,
        0.20972881587629921,
        0.24145172915157029,
    ),
    "robust": (
        2.1112754517061592,
        1.7197246675268569,
        0.39155078417930178,
        32.905240138748241,
        1.739383036693015,
        0.20107092026429449,
        0.2281613876943813,
    ),
    "cluster": (
        1.5375583306983509,
        0.96488068555292861,
        0.57267764514542274,
        18.67456972768289,
        0.99381732190121497,
        0.34644493168626261,
        0.38088778951120378,
    ),
}
# (ar, k, j, rk, clr) from weakiv ... (d1 d2 = ...), null(b)
TWO = {
    ((0.5, -0.3), "iid"): (
        0.052147488417260898,
        0.0085181880814698997,
        0.043629300335791002,
        7.0508498710858269,
        0.0085722848751381992,
    ),
    ((0.5, -0.3), "robust"): (
        0.050775446352751601,
        0.0075555694363024003,
        0.043219876916449199,
        7.0778859784616994,
        0.0076019395516069002,
    ),
    ((0.5, -0.3), "cluster"): (
        0.060988414970423302,
        0.0058082815649791002,
        0.055180133405444198,
        6.7936608023082252,
        0.0058558031785489002,
    ),
    ((0.1, 0.1), "iid"): (
        0.71136483262557881,
        0.66411444629701766,
        0.047250386328561197,
        6.5700977186621783,
        0.66967604725579521,
    ),
    ((0.1, 0.1), "robust"): (
        0.67380243704742027,
        0.63012586160019424,
        0.043676575447226001,
        6.5923239956765318,
        0.63395758278267511,
    ),
    ((0.1, 0.1), "cluster"): (
        0.88940766381307235,
        0.82570589702083641,
        0.063701766792235895,
        6.0063159052726904,
        0.8334683356705499,
    ),
}
# weakiv clr_p with clrsims(400000) at null (0.1, 0.1)
STATA_CLR_P_SIM = {"iid": 0.8027975, "robust": 0.8130875, "cluster": 0.762995}


def _panel(endog, h0, vcov, **kw):
    return sp.weakrobust(
        DATA,
        y="y",
        endog=endog,
        instruments=INST,
        exog=["x1"],
        h0=h0,
        **VCOV[vcov],
        **kw,
    )


def _new_try(k, qt, lrstat):
    """weakiv's ``new_try`` (k = 3 branch): 20-node Simpson rule."""
    gamma = 2 * np.exp(special.gammaln(k / 2)) / np.sqrt(np.pi)
    gamma /= np.exp(special.gammaln((k - 1) / 2))
    ni = 20
    s2 = np.arange(ni + 1) / ni
    qs = (qt + lrstat) / (1 + (qt / lrstat) * s2 * s2)
    farg = gamma * stats.chi2.cdf(qs, k)
    wt = np.full(ni + 1, 2.0)
    wt[0] = wt[-1] = 1.0
    wt[1:-1:2] = 4.0
    return 1 - float(wt @ farg) / 3 / ni


@pytest.mark.parametrize("vcov", ["iid", "robust", "cluster"])
def test_one_endogenous_matches_weakiv(vcov):
    r = _panel("d1", 0.5, vcov)
    ar, k, j, rk, clr, clr_p, kj_p = ONE[vcov]
    assert r["convention"] == "weakiv"
    assert r["ar_stat"] == pytest.approx(ar, rel=REL)
    assert r["k_stat"] == pytest.approx(k, rel=REL)
    assert r["j_stat"] == pytest.approx(j, rel=REL)
    assert r["rk_stat"] == pytest.approx(rk, rel=REL)
    assert r["clr_stat"] == pytest.approx(clr, rel=REL)
    assert r["kj_pvalue"] == pytest.approx(kj_p, rel=REL)
    assert (r["ar_df"], r["k_df"], r["j_df"]) == (3, 1, 2)
    # weakiv's quadrature applied to our statistic gives Stata's p-value;
    # ours is the exact integral of the same function
    assert _new_try(3, r["rk_stat"], r["clr_stat"]) == pytest.approx(clr_p, abs=1e-12)
    assert r["clr_pvalue"] == pytest.approx(
        _clr_conditional_pvalue(clr, rk, 3), abs=1e-12
    )
    assert r["clr_pvalue"] == pytest.approx(clr_p, abs=5e-7)


@pytest.mark.parametrize("vcov", ["iid", "robust", "cluster"])
@pytest.mark.parametrize("h0", [(0.5, -0.3), (0.1, 0.1)])
def test_two_endogenous_joint_matches_weakiv(h0, vcov):
    r = _panel(["d1", "d2"], list(h0), vcov)
    ar, k, j, rk, clr = TWO[(h0, vcov)]
    assert r["ar_stat"] == pytest.approx(ar, rel=REL)
    assert r["k_stat"] == pytest.approx(k, rel=REL)
    assert r["j_stat"] == pytest.approx(j, rel=REL)
    assert r["rk_stat"] == pytest.approx(rk, rel=REL)
    assert r["clr_stat"] == pytest.approx(clr, rel=REL)
    assert (r["ar_df"], r["k_df"], r["j_df"], r["rk_df"]) == (3, 2, 1, 2)
    assert r["clr_pvalue"] == pytest.approx(clr_pvalue(clr, rk, 3, 2), abs=1e-12)


@pytest.mark.parametrize("vcov", ["iid", "robust", "cluster"])
def test_multi_endogenous_clr_pvalue_is_the_conditional_law(vcov):
    """Given rk, K ~ chi2(2) and J ~ chi2(1) independent under H0. Our
    p-value is that law's tail (simulated here); weakiv draws AR ~ chi2(3)
    independently of K instead, and its p-value is that law's tail."""
    _, _, _, rk, clr = TWO[((0.1, 0.1), vcov)]
    rng = np.random.default_rng(7)
    m = 1_000_000
    K = rng.chisquare(2, m)
    J = rng.chisquare(1, m)
    ar = K + J
    sim = 0.5 * (ar - rk + np.sqrt((ar + rk) ** 2 - 4 * J * rk))
    p_exact = clr_pvalue(clr, rk, 3, 2)
    se = np.sqrt(p_exact * (1 - p_exact) / m)
    assert p_exact == pytest.approx(float((sim > clr).mean()), abs=4 * se)
    ar_ind = rng.chisquare(3, m)
    sim_ind = 0.5 * (ar_ind - rk + np.sqrt((ar_ind + rk) ** 2 - 4 * (ar_ind - K) * rk))
    # Stata's 400k draws carry their own error (sd ~6e-4, one seed for all)
    assert STATA_CLR_P_SIM[vcov] == pytest.approx(
        float((sim_ind > clr).mean()), abs=3e-3
    )


def test_clr_pvalue_one_regressor_identity():
    """The p >= 2 integral with p = 1 is Moreira's conditional p-value."""
    from scipy import integrate

    for c, rk, k in [(1.66, 38.0, 3), (4.2, 2.5, 5), (0.3, 0.7, 2), (9.0, 15.0, 4)]:
        shrink = c / (rk + c)
        val = integrate.quad(
            lambda j: stats.chi2.sf(max(c - j * shrink, 0), 1)
            * stats.chi2.pdf(j, k - 1),
            0,
            rk + c,
            epsabs=1e-13,
            epsrel=1e-11,
        )[0] + stats.chi2.sf(rk + c, k - 1)
        assert val == pytest.approx(_clr_conditional_pvalue(c, rk, k), abs=1e-11)


ENDPOINTS = {
    # Stata's own p-value at each endpoint (do-file): 0.05 to 1e-11
    ("cluster", "ar"): [(-np.inf, 0.8412897662787029), (454.3508781183183, np.inf)],
    ("cluster", "k"): [
        (-3.5951021980971385, 0.7028643424980318),
        (1.4868174669321081, 1.5706403210601543),
    ],
    ("cluster", "clr"): [(-4.161639895298505, 0.7212766642723389)],
    ("robust", "ar"): [(-2.740055503827746, 0.7819187728639636)],
    ("robust", "k"): [
        (-1.0974187706394274, 0.6377454878116505),
        (1.704569132792757, 1.8766640289962833),
    ],
    ("robust", "clr"): [(-1.1657537963934554, 0.6481812220749339)],
    ("iid", "ar"): [(-2.515302179256940, 0.7864793095366956)],
    ("iid", "clr"): [(-1.0995121696141026, 0.6510374321281086)],
}


@pytest.mark.parametrize("key", sorted(ENDPOINTS))
def test_confidence_sets(key):
    vcov, test = key
    r = _panel("d1", 0.5, vcov)
    got = r[f"{test}_set"]
    want = ENDPOINTS[key]
    assert len(got) == len(want)
    for (a, b), (c, d) in zip(got, want):
        for x, y in ((a, c), (b, d)):
            if np.isinf(y):
                assert x == y
            else:
                assert x == pytest.approx(y, rel=1e-8)
    assert r[f"{test}_is_disjoint"] == (len(want) > 1)
    assert r[f"{test}_ci"] == (got[0][0], got[-1][1])


def test_iid_closed_form_sets():
    """weakiv's closed-form CLR set is the inversion of its test (display
    precision); its AR set inverts the Wald form ``N e'P e / e'M e``, not
    the LM statistic it reports (Stata's AR p-value at its own endpoint
    .7814155 is 0.05347; at ours, 0.05)."""
    r = _panel("d1", 0.5, "iid")
    lo, hi = r["clr_set"][0]
    assert lo == pytest.approx(-1.09951, abs=6e-6)
    assert hi == pytest.approx(0.651037, abs=6e-7)
    W = np.column_stack([np.ones(len(DATA)), DATA.x1])

    def po(A):
        A = np.asarray(A, float)
        return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]

    y, d, Z = po(DATA.y), po(DATA.d1), po(DATA[INST])
    crit = stats.chi2.ppf(0.95, 3)
    for b, stata in ((-2.428634, None), (0.7814155, None)):
        e = y - d * b
        pe = Z @ np.linalg.lstsq(Z, e, rcond=None)[0]
        wald = len(y) * (pe @ e) / ((e - pe) @ (e - pe))
        assert wald == pytest.approx(crit, rel=2e-6)


def test_subset_ar_matches_weakiv():
    stata = {
        0.2: 0.24697597977602631,
        0.5: 0.048849245147953899,
        1.5: 3.951670029531646,
    }
    for b, val in stata.items():
        r = _panel(["d1", "d2"], [b, 0.0], "iid", subset="d1")
        s = r["subset_ar"]["d1"]
        assert s["stat"] == pytest.approx(val, rel=REL)
        assert s["df"] == 2
    r = _panel(["d1", "d2"], [0.0, 0.1], "iid", subset=["d2"])
    assert r["subset_ar"]["d2"]["stat"] == pytest.approx(0.66281510557761547, rel=REL)
    # Stata's subset p-value at these endpoints is 0.05 to 1e-13
    assert r["subset_ar"]["d2"]["set"][1][0] == pytest.approx(
        -1.7349621864108762, rel=1e-8
    )
    r = _panel(["d1", "d2"], 0.1, "iid", subset="d1")
    (lo1, hi1), (lo2, hi2) = r["subset_ar"]["d1"]["set"]
    assert lo1 == -np.inf and hi2 == np.inf
    assert hi1 == pytest.approx(2.116846691048375, rel=1e-8)
    assert lo2 == pytest.approx(65.70888769155566, rel=1e-8)


def test_projection_ar_is_the_minimum_of_the_joint_ar():
    """Projection statistic at b1 = min over b2 of the joint LM AR."""
    from scipy import optimize

    W = np.column_stack([np.ones(len(DATA)), DATA.x1])

    def po(A):
        A = np.asarray(A, float)
        return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]

    y, X, Z = po(DATA.y), po(DATA[["d1", "d2"]]), po(DATA[INST])
    joint = WeakIVLM(y, X, Z, vcov="iid")
    proj = SubsetAR(y, X[:, 0], X[:, 1:], Z, L=5)
    for b1 in (-1.0, 0.2, 0.9, 3.0):
        grid = np.linspace(-20, 20, 801)
        vals = [joint.at([b1, b2], clr_p=False)["ar"] for b2 in grid]
        i = int(np.argmin(vals))
        best = optimize.minimize_scalar(
            lambda b2: joint.at([b1, b2], clr_p=False)["ar"],
            bracket=(grid[i - 1], grid[i], grid[i + 1]),
            tol=1e-12,
        )
        assert proj.projection(b1)["stat"] == pytest.approx(best.fun, rel=1e-9)
    r = _panel(["d1", "d2"], [0.2, 0.0], "iid", project=["d1"])
    assert r["projection_ar"]["d1"]["stat"] == pytest.approx(
        proj.projection(0.2)["stat"], rel=1e-12
    )
    assert r["projection_ar"]["d1"]["df"] == 3


def test_joint_ar_f_identity_and_refusals():
    """Joint AR F = F test of Z in the regression of y - X h0 on [1, x1, Z]."""
    h0 = [0.3, -0.1]
    r = sp.anderson_rubin_test(
        DATA, y="y", endog=["d1", "d2"], instruments=INST, exog=["x1"], h0=h0
    )
    e = DATA.y - DATA.d1 * h0[0] - DATA.d2 * h0[1]
    A = np.column_stack([np.ones(len(DATA)), DATA.x1])
    B = np.column_stack([A, DATA[INST]])
    rss_r = np.sum((e - A @ np.linalg.lstsq(A, e, rcond=None)[0]) ** 2)
    rss_u = np.sum((e - B @ np.linalg.lstsq(B, e, rcond=None)[0]) ** 2)
    F = ((rss_r - rss_u) / 3) / (rss_u / (len(DATA) - 5))
    assert r["ar_stat"] == pytest.approx(F, rel=1e-10)
    assert r["ar_df"] == (3, 395)
    assert r["ar_ci"] is None and r["h0"] == {"d1": 0.3, "d2": -0.1}

    exc = sp.exceptions
    with pytest.raises(exc.MethodIncompatibility, match="h0 has 3"):
        sp.anderson_rubin_test(
            DATA, y="y", endog=["d1", "d2"], instruments=INST, h0=[0, 0, 0]
        )
    with pytest.raises(exc.IdentificationFailure, match="Under-identified"):
        sp.weakrobust(DATA, y="y", endog=["d1", "d2"], instruments=["z1"])
    with pytest.raises(exc.MethodIncompatibility, match="i.i.d. errors only"):
        _panel(["d1", "d2"], 0.0, "robust", subset="d1")
    with pytest.raises(exc.MethodIncompatibility, match="two or more"):
        _panel("d1", 0.0, "iid", project="d1")
    with pytest.raises(exc.MethodIncompatibility, match="not endogenous"):
        _panel(["d1", "d2"], 0.0, "iid", subset="x1")
    with pytest.raises(exc.MethodIncompatibility, match="one-way"):
        sp.weakrobust(
            DATA.assign(g2=DATA.g % 3),
            y="y",
            endog="d1",
            instruments=INST,
            cluster=["g", "g2"],
        )
    with pytest.raises(exc.MethodIncompatibility, match="does not use it"):
        sp.weakrobust(
            DATA,
            y="y",
            endog="d1",
            instruments=INST,
            cluster="g",
            test_vcov="robust",
        )


def test_legacy_single_endogenous_path_unchanged():
    """No cluster / robust / subset / project: the ivmodel convention."""
    r = sp.weakrobust(DATA, y="y", endog="d1", instruments=INST, exog=["x1"])
    a = sp.anderson_rubin_test(DATA, y="y", endog="d1", instruments=INST, exog=["x1"])
    assert r["convention"] == "ivmodel"
    assert r["ar_stat"] == a["ar_stat"]
    r2 = sp.weakrobust(
        DATA,
        y="y",
        endog="d1",
        instruments=INST,
        exog=["x1"],
        vcov="cluster",
        cluster="g",
    )
    assert r2["test_vcov"] == "cluster"
    assert r2["effective_F"] == pytest.approx(
        sp.effective_f_test(DATA, "d1", INST, ["x1"], cluster="g")["F_eff"], rel=1e-12
    )
    assert "weakiv LM form" in r2.summary()
    assert list(r2.to_frame()["statistic"])[0].startswith("First-stage F")
