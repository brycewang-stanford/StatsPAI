"""``sp.bitest`` against Stata 18 ``bitest`` / ``bitesti``.

The reference numbers are ``return list`` after each command, at full
precision, run on 2026-10-04 (StataNow 18 MP). The binomial distribution is
exact arithmetic on both sides, so the tolerance is 1e-12.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# (N, k, p0) -> r(p), r(p_l), r(p_u), r(k_opp)
STATA = {
    (41, 25, 0.5): (0.211023597608801, 0.9413623970212937, 0.1055117988044005, 16),
    (30, 4, 0.3): (0.0470922545946388, 0.0301549431020893, 0.9906834326888229, 15),
    (29, 4, 0.3): (0.067160348713326, 0.0378949112529996, 0.9879049825833679, 14),
}


@pytest.mark.parametrize("case", sorted(STATA))
def test_immediate_form_matches_stata(case):
    n, k, p0 = case
    p, p_l, p_u, k_opp = STATA[case]
    res = sp.bitest(n=n, successes=k, p=p0)
    assert res.pvalue == pytest.approx(p, rel=1e-12)
    assert res.pvalue_less == pytest.approx(p_l, rel=1e-12)
    assert res.pvalue_greater == pytest.approx(p_u, rel=1e-12)
    assert res.k_opposite == k_opp
    assert res.expected == pytest.approx(n * p0)


def test_variable_form_drops_missing_values_as_stata_does():
    """``bitest z == 0.3`` on 4 ones, 25 zeros and one missing value."""
    df = pd.DataFrame({"z": [1.0] * 4 + [0.0] * 25 + [np.nan]})
    res = sp.bitest(df, "z", p=0.3)
    assert (res.n_obs, res.successes) == (29, 4)
    assert res.pvalue == pytest.approx(STATA[(29, 4, 0.3)][0], rel=1e-12)


def test_two_sided_is_not_twice_the_smaller_tail_off_one_half():
    res = sp.bitest(n=30, successes=4, p=0.3)
    assert res.pvalue != pytest.approx(2 * res.pvalue_less)
    assert res.pvalue == pytest.approx(stats.binomtest(4, 30, 0.3).pvalue, rel=1e-12)


def test_interval_is_clopper_pearson():
    res = sp.bitest(n=41, successes=25)
    want = stats.binomtest(25, 41).proportion_ci(0.95, method="exact")
    assert res.ci == pytest.approx((want.low, want.high), rel=1e-12)
    assert sp.bitest(n=10, successes=0).ci[0] == 0.0
    assert sp.bitest(n=10, successes=10).ci[1] == 1.0


def test_extreme_outcome_has_an_empty_opposite_tail():
    res = sp.bitest(n=5, successes=0, p=0.1)
    # nothing in the upper tail is as unlikely as... every k >= 1 is checked
    assert res.pvalue == pytest.approx(stats.binomtest(0, 5, 0.1).pvalue, rel=1e-12)
    assert "two-sided" in res.summary()


def test_result_serializes():
    import json

    json.dumps(sp.bitest(n=41, successes=25).to_dict(detail="agent"))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=10, successes=11),
        dict(n=10),
        dict(n=10, successes=3, p=1.0),
        dict(),
    ],
)
def test_bad_input_is_refused(kwargs):
    with pytest.raises(MethodIncompatibility):
        sp.bitest(**kwargs)


def test_non_binary_column_is_refused():
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.bitest(pd.DataFrame({"z": [0, 1, 2]}), "z")


def test_stata_commands_run_and_store_r():
    assert sp.from_stata("bitesti 41 25 1/2")["arguments"] == {
        "n": 41,
        "successes": 25,
        "p": 0.5,
    }
    assert sp.from_stata("bitesti 41 25 1/2, detail")["ignored_display_options"] == [
        "detail"
    ]
    assert not sp.from_stata("bitesti 4 9 0.5")["ok"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("bitesti 41 25 1/2")
        df = pd.DataFrame({"z": [1.0] * 4 + [0.0] * 25 + [np.nan]})
        shown = sp.stata("bitest z == 0.3\ndisplay r(k_opp)", data=df)
    assert res.pvalue == pytest.approx(STATA[(41, 25, 0.5)][0], rel=1e-12)
    assert shown == 14.0
