"""Survey domain (subpopulation) estimation: ``subpop=`` vs R and Stata.

Estimates use the domain rows; the variance linearises over the whole
design with zero scores outside the domain -- filtering the data first
would drop strata / PSUs and understate the variance.

References on ``_fixtures/survey_calib_data.csv``:

* ``survey_domain_R.json`` (``_generate_survey_domain_R.R``): R ``survey``
  ``subset(design, ...)`` then ``svymean`` / ``svytotal`` / ``svyglm`` for
  males, for ``dom2`` (age group a in strata 1-2 only, so strata 3-4 have
  no members) and for males in the raking-calibrated design.
* Stata 18 ``svyset psu [pw=d], strata(stratum)`` then ``svy, subpop()``
  (numbers inline below).

The references agree on estimates and SEs (asserted 1e-10; observed 3e-14)
and on df when every PSU of a stratum with domain members has some. When
one does not (``dom3``), Stata counts all PSUs of such strata (df 8) and R
only PSUs with members (df 7): ``subpop_df="stata"`` (default) / ``"r"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_domain_R.json").read_text(encoding="utf-8"))
T = json.loads((_FIX / "survey_calib_R.json").read_text(encoding="utf-8"))["targets"]


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "survey_calib_data.csv")
    d["male"] = (d["sex"] == "M").astype(int)
    d["dom2"] = ((d["agegrp"] == "a") & (d["stratum"] <= 2)).astype(int)
    d["dom3"] = ((d["dom2"] == 1) & ~((d["stratum"] == 1) & (d["psu"] == 1))).astype(
        int
    )
    return d


@pytest.fixture(scope="module")
def designs(data):
    des = sp.svydesign(data, weights="d", strata="stratum", cluster="psu", nest=True)
    cal = des.calibrate(margins={"sex": T["sex"], "agegrp": T["agegrp"]}, tol=1e-14)
    return des, cal


@pytest.mark.parametrize(
    "key,which,sub",
    [("male", 0, "male"), ("dom2", 0, "dom2"), ("calib_male", 1, "male")],
)
def test_domain_estimates_match_R_survey(designs, key, which, sub):
    d = designs[which]
    ref = R[key]
    m = d.mean("y", subpop=sub, subpop_df="r")
    t = d.total("y", subpop=sub)
    g = d.glm("y ~ income + age", subpop=sub)
    got = np.r_[
        m.estimate.iloc[0],
        m.std_error.iloc[0],
        t.estimate.iloc[0],
        t.std_error.iloc[0],
        g.estimate.values,
        g.std_error.values,
    ]
    want = np.r_[
        ref["mean"],
        ref["se_mean"],
        ref["total"],
        ref["se_total"],
        ref["glm_coef"],
        ref["glm_se"],
    ]
    np.testing.assert_allclose(got, want, rtol=1e-10)
    assert m.dof == ref["degf"]


def test_domain_estimates_match_stata_svy_subpop(designs):
    des = designs[0]
    m = des.mean("y", subpop="dom2")
    assert m.estimate.iloc[0] == pytest.approx(6.62835725938913, rel=1e-10)
    assert m.std_error.iloc[0] == pytest.approx(0.133368800299639, rel=1e-10)
    assert m.dof == 8
    t = des.total("y", subpop="dom2")
    assert t.estimate.iloc[0] == pytest.approx(15443.0357393013, rel=1e-10)
    assert t.std_error.iloc[0] == pytest.approx(1605.71207931403, rel=1e-10)
    g = des.glm("y ~ income + age", subpop="dom2")
    assert g.estimate["income"] == pytest.approx(0.115976273678693, rel=1e-10)
    assert g.std_error["income"] == pytest.approx(0.020152386203656, rel=1e-10)
    assert g.dof == 8
    m_male = des.mean("y", subpop="male")
    assert m_male.estimate.iloc[0] == pytest.approx(7.21419559525776, rel=1e-10)
    assert m_male.std_error.iloc[0] == pytest.approx(0.168057051300918, rel=1e-10)
    assert m_male.dof == 16


def test_df_rules_differ_only_on_psus_without_members(designs):
    des = designs[0]
    stata = des.mean("y", subpop="dom3")
    r = des.mean("y", subpop="dom3", subpop_df="r")
    # Stata: mean 6.62081937458253, se .152433026274666, df 8; R degf 7
    assert stata.estimate.iloc[0] == pytest.approx(6.62081937458253, rel=1e-10)
    assert stata.std_error.iloc[0] == pytest.approx(0.152433026274666, rel=1e-10)
    assert (stata.dof, r.dof) == (8, 7)


def test_filtering_first_is_not_the_domain_estimate(data, designs):
    """The classic mistake the option exists to prevent. It bites when a PSU
    of a domain stratum has no members (dom3): filtering deletes that PSU,
    the domain estimate keeps it with a zero total (Stata / R)."""
    des = designs[0]
    sub = data[data["dom3"] == 1]
    filtered = sp.svydesign(
        sub, weights="d", strata="stratum", cluster="psu", nest=True
    ).mean("y")
    domain = des.mean("y", subpop="dom3")
    assert filtered.estimate.iloc[0] == pytest.approx(domain.estimate.iloc[0])
    assert filtered.std_error.iloc[0] != pytest.approx(domain.std_error.iloc[0])


def test_svyglm_with_missing_covariates_matches_R(data):
    """Regression: rows patsy drops for missing values used to crash svyglm
    (weights and scores of different lengths). R svyglm's na.action treats
    them as outside the domain; so does sp.svyglm now.

    R: d$income[1:5] <- NA; svyglm(y ~ income + age, design = des)
    """
    holes = data.copy()
    holes.loc[holes.index[:5], "income"] = np.nan
    des = sp.svydesign(holes, weights="d", strata="stratum", cluster="psu", nest=True)
    g = des.glm("y ~ income + age")
    np.testing.assert_allclose(
        g.estimate.values,
        [3.20758548908184, 0.0875351730835878, 0.0220339638286511],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        g.std_error.values,
        [0.293439414726701, 0.00945978216599002, 0.00305486484202395],
        rtol=1e-10,
    )
