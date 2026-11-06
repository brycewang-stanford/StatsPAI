"""``sp.pscore`` against Stata 18 ``pscore`` (Becker and Ichino, st0026_2).

Fixture: ``_fixtures/pscore_Stata.json`` from
``_fixtures/_generate_pscore_Stata.do`` on ``_fixtures/matching_lalonde.csv``
(``MatchIt::lalonde``), five specifications: logit and probit, with and
without the common-support restriction, three starting block counts and
three test levels. ``pscore`` is run as a black box (outputs only).

Compared per specification: the number of blocks, every row's block,
common-support flag and score, the "not balanced" lines pscore prints
(and, for each, the p-value of Stata's equal-variance ``ttest`` in that
block), the treatment-model coefficients and log likelihood. All of these
agree to 1e-8 relative or better (scores, coefficients) and exactly
(blocks, support, the list of rejections).

Standard errors are the one gap. ``pscore`` reports the variance of its
``logit`` / ``probit`` call at Stata's default convergence, and the
standard errors differ from ours by up to 2.4e-5 relative while the
coefficients agree to 4e-9. For the logit, the same specifications fitted
by R ``glm`` with the deviance converged to 1e-14
(``_fixtures/matching_balance_R.json``) give our standard errors to 1e-10
(the inverse information at glm's fitted values), so the gap is the
stopping point of Stata's fit, not the variance formula.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
X8 = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
SPECS = {
    "1": dict(cov=X8, ps_model="logit", common_support=True),
    "2": dict(cov=X8, ps_model="probit"),
    "3": dict(cov=["age", "educ", "married", "re74"], ps_model="logit", n_blocks=3,
              level=0.05),
    "4": dict(cov=X8, ps_model="probit", common_support=True, level=0.05),
    "5": dict(cov=["age", "educ", "black", "married", "re74"], ps_model="logit",
              n_blocks=8, level=0.1),
}  # fmt: skip


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "pscore_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lalonde():
    return pd.read_csv(FIX / "matching_lalonde.csv")


def _fit(lalonde, key):
    kw = dict(SPECS[key])
    cov = kw.pop("cov")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.pscore(lalonde, "treat", cov, **kw)


def _vec(values):
    return np.array([np.nan if v is None else float(v) for v in values])


@pytest.mark.parametrize("key", sorted(SPECS))
def test_blocks_support_and_scores_match_pscore(ref, lalonde, key):
    r, s = _fit(lalonde, key), ref[key]
    assert r.n_blocks == s["n_blocks"]
    np.testing.assert_array_equal(
        np.nan_to_num(r.block.to_numpy(dtype=float), nan=-1),
        np.nan_to_num(_vec(s["block"]), nan=-1),
    )
    np.testing.assert_array_equal(r.support.to_numpy().astype(int), s["comsup"])
    want = _vec(s["ps"])
    got = r.pscore.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-8, atol=0)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_balancing_test_matches_pscore(ref, lalonde, key):
    r, s = _fit(lalonde, key), ref[key]
    assert r.unbalanced[["variable", "block"]].to_numpy().tolist() == s["not_balanced"]
    assert r.balanced is (not s["not_balanced"])
    p = {(v, int(b)): float(pv) for v, b, pv in s["ttest"]}
    level = SPECS[key].get("level", 0.01)
    # the lines pscore prints are exactly the ttest rejections at `level`
    assert {k for k, v in p.items() if v < level} == {
        tuple(x) for x in s["not_balanced"]
    }
    for row in r.unbalanced.itertuples():
        assert row.pvalue == pytest.approx(p[(row.variable, int(row.block))], rel=1e-10)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_treatment_model_matches_pscore(ref, lalonde, key):
    r, s = _fit(lalonde, key), ref[key]
    assert r.loglik == pytest.approx(s["ll"], rel=1e-9)
    coef = r.coefficients
    for name, b in s["b"].items():
        assert coef.loc[name, "coef"] == pytest.approx(b, rel=1e-8), name
        # Stata's stopping point; see the module docstring
        assert coef.loc[name, "se"] == pytest.approx(s["se"][name], rel=5e-5), name


@pytest.mark.parametrize("key, spec", [("1", "x8"), ("5", "x5")])
def test_logit_standard_errors_are_those_at_the_maximum(lalonde, key, spec):
    glm = json.loads((FIX / "matching_balance_R.json").read_text(encoding="utf-8"))
    g = glm["lalonde_logit"][spec]
    coef = _fit(lalonde, key).coefficients
    for name, se in g["se_info"].items():
        ours = "_cons" if name == "(Intercept)" else name
        assert coef.loc[ours, "coef"] == pytest.approx(g["b"][name], rel=1e-9), name
        assert coef.loc[ours, "se"] == pytest.approx(se, rel=1e-10), name
        # vcov(glm) takes the IRLS weights of its last iteration, before
        # the final update: 2e-9 here
        assert coef.loc[ours, "se"] == pytest.approx(g["se"][name], rel=1e-8), name
