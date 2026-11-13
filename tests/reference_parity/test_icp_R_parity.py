"""``sp.icp(test="normal")`` against R ``InvariantCausalPrediction::ICP``.

Reference: InvariantCausalPrediction 0.8 (GPL, run as a black box) with
its default invariance test (``test = "normal"``) and every subset tested
(``selection = "all"``), at ``alpha`` 0.05 and 0.01, on the 16 designs of
``_fixtures/icp_data.csv`` (two to four environments, three to five
predictors; ``_generate_icp_R.R`` -> ``icp_R.json``). In a quarter of the
designs the environments shift the response directly and the reference
rejects every subset.

Compared, for each of the 32 fits:

* the accepted subsets: exact;
* the parents, their intersection: exact;
* the confidence interval of each parent's coefficient: 1e-9 absolute
  (observed 5e-15).

The reference's test compares each environment with the rest (Welch t test
and F test on the residuals of the pooled fit) and multiplies the smallest
p-value by ``2 * (K - 1)``. ``sp.icp`` defaults to another test of the same
hypothesis, ``test="anova"`` (analysis of variance and Levene's test over
all environments at once). On these 32 fits the default accepts a
different collection of subsets in 8 and names the same parents in all,
which is why the comparison is on ``test="normal"`` and the grade says so.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "icp_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "icp_data.csv")
CASES = REF["cases"]


def _fit(k, **kw):
    case = CASES[k]
    rows = DATA[DATA["case"] == case["case"]]
    return sp.icp(
        rows[case["columns"]],
        rows["y"].to_numpy(),
        rows["env"].to_numpy(),
        alpha=case["alpha"],
        **kw,
    )


def test_fixture_is_icp_output_with_accepting_and_rejecting_designs():
    assert REF["versions"]["InvariantCausalPrediction"] == "0.8"
    assert len(CASES) == 32
    assert {c["n_env"] for c in CASES} == {2, 3, 4}
    assert sum(c["model_reject"] for c in CASES) == 8
    assert sum(len(c["accepted"]) > 0 for c in CASES) == 24


@pytest.mark.parametrize("k", range(len(CASES)))
def test_accepted_sets_parents_and_intervals_match_r(k):
    case = CASES[k]
    res = _fit(k, test="normal")
    ours = sorted(sorted(s) for s in res.accepted_subsets)
    assert ours == sorted(sorted(s) for s in case["accepted"])
    assert (len(res.accepted_subsets) == 0) is case["model_reject"]
    want_parents = (
        set.intersection(*(set(s) for s in case["accepted"]))
        if case["accepted"]
        else set()
    )
    assert res.parents == want_parents
    assert set(res.coefficients) == want_parents
    for name, (lo, hi) in res.coefficients.items():
        j = case["columns"].index(name)
        assert lo == pytest.approx(case["ci_lower"][j], abs=1e-9)
        assert hi == pytest.approx(case["ci_upper"][j], abs=1e-9)
        # a parent's interval excludes zero in the reference
        assert case["ci_lower"][j] * case["ci_upper"][j] > 0


def test_unknown_test_is_refused_and_default_is_anova():
    with pytest.raises(MethodIncompatibility, match="test"):
        _fit(0, test="chow")
    default = _fit(0)
    explicit = _fit(0, test="anova")
    assert default.accepted_subsets == explicit.accepted_subsets
    assert default.coefficients == explicit.coefficients
