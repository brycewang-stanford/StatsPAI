"""``sp.cdlz_bunching`` reproduces Cengiz, Dube, Lindner & Zipperer (2019),
Table 1 column 1.

``data/cdlz_table1_col1.json`` holds the 63 bin x event-year coefficients
and their covariance from the Minimum Wages replication (``sp.hdfe_ols`` of
the authors' regression on their data, whose summaries equal the published
table to three decimals), with the table's constants (E, B, EWB, %dMW,
mean new MW). The coefficients are the paper's; the fixture spares the
15 GB of data.

* Published Table 1, col 1 (three decimals): every statistic and SE.
* The replication's unrounded numbers (numerical-gradient delta method):
  estimates to 1e-12; SEs to 1e-8 for the linear statistics and 1e-6 for
  the nonlinear ones (affected wage, wage elasticity) -- the rounding of a
  central finite difference, which ``sp.cdlz_bunching`` replaces with the
  analytic gradient.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = json.loads(
    (pathlib.Path(__file__).parent / "data" / "cdlz_table1_col1.json").read_text(
        encoding="utf-8"
    )
)
MAP = {
    "below": "missing_jobs_below",
    "above": "excess_jobs_above",
    "wage": "pct_change_affected_wage",
    "emp": "pct_change_affected_employment",
    "elas_mw": "elasticity_wrt_mw",
    "elas_wage": "elasticity_wrt_affected_wage",
}


@pytest.fixture(scope="module")
def result():
    terms = pd.DataFrame(FIX["terms"])
    params = pd.Series(FIX["params"], index=terms["term"])
    vcov = pd.DataFrame(FIX["vcov"], index=terms["term"], columns=terms["term"])
    return sp.cdlz_bunching(
        terms=terms, params=params, covariance=vcov, **FIX["constants"]
    )


def test_published_table1_col1(result):
    s = result.model_info["summary"]
    for key, (est, se) in FIX["published"].items():
        row = s.loc[MAP[key]]
        assert round(row["estimate"], 3) == pytest.approx(est, abs=1e-12), key
        assert round(row["se"], 3) == pytest.approx(se, abs=1e-12), key


def test_replication_unrounded(result):
    s = result.model_info["summary"]
    rep = FIX["replication"]
    for key, name in MAP.items():
        assert s.loc[name, "estimate"] == pytest.approx(rep[key], rel=1e-12), key
        tol = 1e-8 if key in ("below", "above", "emp", "elas_mw") else 1e-6
        assert s.loc[name, "se"] == pytest.approx(rep[key + "_se"], rel=tol), key
    assert result.estimate == pytest.approx(rep["emp"], rel=1e-12)


def test_event_path_and_bin_profile(result):
    path = result.model_info["event_path"]
    assert set(path["year"]) == {-3, -2, 0, 1, 2, 3, 4}
    prof = result.model_info["bin_profile"]
    assert list(prof["bin"]) == [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # The running sum of the bin profile over the post years is Da + Db.
    s = result.model_info["summary"]
    total = s.loc["missing_jobs_below", "estimate"] + s.loc["excess_jobs_above", "estimate"]
    assert prof["running_sum"].iloc[-1] == pytest.approx(total, rel=1e-12)


def test_input_validation():
    terms = pd.DataFrame(FIX["terms"])
    with pytest.raises(sp.MethodIncompatibility, match="params"):
        sp.cdlz_bunching(terms=terms, **FIX["constants"])
    params = pd.Series(np.zeros(len(terms)), index=terms["term"])
    vcov = pd.DataFrame(np.eye(len(terms)), index=terms["term"], columns=terms["term"])
    with pytest.raises(sp.MethodIncompatibility, match="nonzero"):
        sp.cdlz_bunching(
            terms=terms, params=params, covariance=vcov, **{**FIX["constants"], "epop": 0.0}
        )
