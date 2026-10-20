"""Identities and known values for ``diagnostics._weakiv_lm`` (clean-room
rewrite, ``docs/dev/cleanroom/weakiv_lm.md``). Stata parity lives in
``test_weakiv_stata.py``; this file pins what that file does not.

* ``clr_pvalue`` against the conditional law evaluated in 40-digit
  arithmetic (mpmath, integrating over ``J`` -- the other ordering from the
  implementation's integral over ``K``), abs 1e-14;
* its limits (``rk = 0``: chi2(k); ``rk = inf`` and ``k = p``: chi2(p));
* closed forms of the AR statistic (i.i.d. LM form ``N e'P e / e'e``;
  robust ``e'Z (Z' diag(e^2) Z)^{-1} Z'e``);
* ``K <= CLR <= AR`` and the ``clr_alpha`` shortcut never changing the
  decision;
* just-identified models: no J, CLR test = AR test.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.diagnostics._weakiv_lm import WeakIVLM, clr_pvalue

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "weakiv_fx.csv")

# mpmath (dps = 40) of int_0^{rk+c} P(chi2(p) > c - j c/(rk+c)) f_{k-p}(j) dj
# + P(chi2(k-p) > rk + c)
MP40 = [
    ((0.0085, 38.0, 5, 1), 0.93057011478813839067),
    ((0.3, 7.05, 3, 2), 0.87857033812294432323),
    ((0.3, 7.05, 6, 3), 0.97880944460387363354),
    ((4.2, 7.05, 6, 3), 0.39633353990408747824),
    ((1.66, 38.0, 3, 1), 0.20973859345012894489),
    ((9.0, 1e4, 4, 1), 0.0027037861230528218751),
    ((30.0, 0.7, 3, 2), 1.2399750491584034201e-6),
    ((2.0, 5.0, 3, 1), 0.26368799607836423382),
    ((80.0, 1e4, 10, 4), 1.7829201009751402329e-16),
]


def _partialled():
    W = np.column_stack([np.ones(len(DATA)), DATA.x1])

    def po(A):
        A = np.asarray(A, float)
        return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]

    return (
        po(DATA.y),
        po(DATA[["d1", "d2"]]),
        po(DATA[["z1", "z2", "z3"]]),
        pd.factorize(DATA.g)[0],
    )


@pytest.mark.parametrize("args,want", MP40)
def test_clr_pvalue_matches_40_digit_integral(args, want):
    assert clr_pvalue(*args) == pytest.approx(want, abs=1e-14)


def test_clr_pvalue_limits():
    for c in (0.5, 3.0, 11.0):
        assert clr_pvalue(c, 0.0, 4, 1) == pytest.approx(stats.chi2.sf(c, 4), abs=1e-14)
        assert clr_pvalue(c, 0.0, 5, 2) == pytest.approx(stats.chi2.sf(c, 5), abs=1e-14)
        assert clr_pvalue(c, np.inf, 4, 2) == stats.chi2.sf(c, 2)
        assert clr_pvalue(c, 3.0, 2, 2) == stats.chi2.sf(c, 2)
    assert clr_pvalue(0.0, 3.0, 3, 1) == 1.0
    assert clr_pvalue(np.inf, 3.0, 3, 1) == 0.0
    assert np.isnan(clr_pvalue(np.nan, 3.0, 3, 1))
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        clr_pvalue(1.0, 2.0, 1, 2)


def test_ar_closed_forms():
    y, X, Z, _ = _partialled()
    n = len(y)
    for b in (-1.0, 0.5, 2.0):
        e = y - X[:, 0] * b
        pe = Z @ np.linalg.lstsq(Z, e, rcond=None)[0]
        lm = n * (pe @ e) / (e @ e)
        assert WeakIVLM(y, X[:, :1], Z).at(b)["ar"] == pytest.approx(lm, rel=1e-12)
        ze = Z.T @ e
        meat = (Z * (e**2)[:, None]).T @ Z
        rob = ze @ np.linalg.solve(meat, ze)
        got = WeakIVLM(y, X[:, :1], Z, vcov="robust").at(b)["ar"]
        assert got == pytest.approx(rob, rel=1e-12)


@pytest.mark.parametrize("vcov", ["robust", "cluster"])
def test_clr_between_k_and_ar_and_shortcut_keeps_decision(vcov):
    y, X, Z, g = _partialled()
    for Xs in (X[:, :1], X):
        eng = WeakIVLM(y, Xs, Z, vcov=vcov, cluster_codes=g)
        p = Xs.shape[1]
        for b in np.linspace(-3, 3, 13):
            b0 = [b] if p == 1 else [b, -0.3]
            exact = eng.at(b0)
            assert exact["k"] <= exact["clr"] * (1 + 1e-12)
            assert exact["clr"] <= exact["ar"] * (1 + 1e-12)
            fast = eng.at(b0, clr_alpha=0.05)
            assert (fast["clr_p"] > 0.05) == (exact["clr_p"] > 0.05)


def test_just_identified_has_no_j_and_clr_is_ar():
    y, X, Z, _ = _partialled()
    r = WeakIVLM(y, X, Z[:, :2]).at([0.5, -0.3])
    assert r["j"] is None and r["j_df"] is None and r["j_p"] is None
    assert r["clr_p"] == r["ar_p"]
    assert r["k"] == pytest.approx(r["ar"], rel=1e-12)
    assert np.isnan(WeakIVLM(y, X, Z[:, :2]).at([0.5, -0.3], clr_p=False)["clr_p"])


def test_refusals():
    y, X, Z, _ = _partialled()
    exc = sp.exceptions
    with pytest.raises(exc.IdentificationFailure, match="Under-identified"):
        WeakIVLM(y, X, Z[:, :1])
    with pytest.raises(exc.MethodIncompatibility, match="unknown vcov"):
        WeakIVLM(y, X, Z, vcov="hc3")
    with pytest.raises(exc.MethodIncompatibility, match="cluster_codes"):
        WeakIVLM(y, X, Z, vcov="cluster")
    with pytest.raises(exc.MethodIncompatibility, match="b0 has 1 value"):
        WeakIVLM(y, X, Z).at(0.5)
    with pytest.raises(exc.DataInsufficient, match="clusters"):
        WeakIVLM(y, X, Z, vcov="cluster", cluster_codes=np.arange(len(y)) % 2)
