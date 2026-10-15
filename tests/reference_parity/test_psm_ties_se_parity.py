"""The Abadie-Imbens SE under ``ties='all'`` is ``psmatch2 , ai(1) ties``.

Through 1.39.6 the matches kept every tied control but the same-arm
neighbours behind sigma^2(X) were taken in row order, so the default
``sp.psm`` standard error on ``MatchIt::lalonde`` moved between 1174 and
1198 when the rows were shuffled. Stata 18 ``psmatch2 , ai(1) ties`` (run
by ``_fixtures/_generate_psm_ties_stata.do`` under ``set type double``,
on the propensity score ``sp.psm`` fits) gives one number in every order,
and so does ``sp.psm`` now.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching._matched_frame import _within_group_self_outcome_all_ties

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "psm_ties_Stata.json").read_text(encoding="utf-8"))
X = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


@pytest.fixture(scope="module")
def lalonde():
    sys.path.insert(0, str(Path(__file__).parents[1] / "orig_parity"))
    from _common import read_csv

    return read_csv("04_lalonde_original")


def test_stata_ties_result_does_not_depend_on_row_order():
    ses = {REF[f"ties_order{k}"]["se"] for k in range(4)}
    assert len(ses) == 1
    # Without `ties` Stata itself moves with the order.
    assert len({REF[f"noties_order{k}"]["se"] for k in range(4)}) == 4


@pytest.mark.parametrize("order", [None, 0, 1, 2])
def test_default_psm_matches_psmatch2_ties_in_any_order(lalonde, order):
    data = lalonde if order is None else lalonde.sample(frac=1, random_state=order)
    res = sp.psm(data, y="re78", d="treat", X=X, method="nn")
    # T2: same score, same matches; the SE agrees to the last digits.
    assert res.estimate == pytest.approx(REF["ties_order0"]["est"], rel=1e-12)
    assert res.se == pytest.approx(REF["ties_order0"]["se"], rel=1e-12)


def test_same_arm_means_equal_psmatch2_self_outcome():
    d = pd.read_csv(FIX / "psm_ties_lalonde.csv")
    got = _within_group_self_outcome_all_ties(
        d["re78"].to_numpy(float),
        d["treat"].to_numpy(int),
        d["_pscore"].to_numpy(float),
        1,
    )
    np.testing.assert_allclose(got, REF["ties_self_re78"], rtol=1e-13)


def test_psmatch2_without_ties_keeps_row_order(lalonde):
    # psmatch2 without `ties` breaks ties in the matches and the same-arm
    # neighbours by row order; `sp.psmatch2(ties=False)` does the same on
    # the order Stata saw.
    data = lalonde.assign(
        ps=pd.read_csv(FIX / "psm_ties_lalonde.csv")["_pscore"].to_numpy()
    )
    res = sp.psmatch2(data, treat="treat", y="re78", pscore="ps", ai=1)
    assert res.att == pytest.approx(REF["noties_order0"]["est"], rel=1e-12)
    assert res.se == pytest.approx(REF["noties_order0"]["se"], rel=1e-12)
