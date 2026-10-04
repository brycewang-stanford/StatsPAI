"""The TWFE coefficient taken apart by hand on a three-unit panel.

Clarke's *Applied Microeconometrics* (code call-out 4.2) builds a panel of
three units over 2000-2009 -- one never treated, one treated from 2006, one
from 2003 -- with no noise, and computes by hand every piece of the
Goodman-Bacon (2021) decomposition and of the de Chaisemartin-D'Haultfoeuille
(2020) weights. The panel is deterministic, so every number is known exactly:

* TWFE coefficient with a constant effect per unit: 27/11.
* Goodman-Bacon weights: 7/22 (2003 cohort vs never), 8/22 (2006 vs never),
  3/22 (2003 vs 2006 before 2006), 4/22 (2006 vs 2003 after 2003); the 2x2
  estimates are 3, 2, 3 and 2.
* dCDH weights: 5/33 on each of the 2003 cohort's cells in 2003-2005, zero on
  its cells from 2006 on, 3/22 on each of the 2006 cohort's four cells.

Stata 18 prints the same numbers when the chapter's do-file is run
(2.4545456, .31818181, .36363635, .13636363, .18181817, 0.1515152,
0.1363636), and 3.8045454 for the second outcome, whose effect grows with
time since treatment.

Until 2026-10 ``sp.twfe_decomposition`` weighted its 2x2 rows by unit
counts and did not restrict the timing comparisons to their windows; its
headline on this panel was 1.839.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    unit = np.repeat([1, 2, 3], 10)
    year = np.tile(np.arange(2000, 2010), 3)
    first = np.where(unit == 2, 2006, np.where(unit == 3, 2003, 0))
    post = ((first > 0) & (year >= first)).astype(int)
    since = np.where(post == 1, year - first, 0)
    df = pd.DataFrame({"unit": unit, "year": year, "first": first, "post": post})
    df["y1"] = 2 + (year - 2000) * 0.2 + unit + post * unit
    df["y2"] = df["y1"] + 0.45 * post * unit * since
    return df


BACON = {
    # (treated cohort, control): (2x2 estimate, weight)
    (2003, "Never"): (3.0, 7 / 22),
    (2006, "Never"): (2.0, 8 / 22),
    (2003, 2006): (3.0, 3 / 22),
    (2006, 2003): (2.0, 4 / 22),
}


def test_twfe_decomposition_headline_is_the_twfe_coefficient(panel):
    r = sp.twfe_decomposition(
        panel, y="y1", group="unit", time="year", first_treat="first"
    )
    assert abs(r.estimate - 27 / 11) < 1e-12
    assert abs(r.model_info["twfe_beta"] - 27 / 11) < 1e-12
    assert abs(r.model_info["bacon_att"] - 27 / 11) < 1e-12
    assert abs(r.detail["weighted_est"].sum() - r.estimate) < 1e-12


def test_twfe_decomposition_rows_are_goodman_bacons(panel):
    r = sp.twfe_decomposition(
        panel, y="y1", group="unit", time="year", first_treat="first"
    )
    rows = {
        (row.treated_cohort, row.control_cohort): (row.estimate, row.weight)
        for row in r.detail.itertuples()
    }
    assert set(rows) == set(BACON)
    for key, (estimate, weight) in BACON.items():
        assert abs(rows[key][0] - estimate) < 1e-12, key
        assert abs(rows[key][1] - weight) < 1e-12, key
    assert abs(r.detail["weight"].sum() - 1.0) < 1e-12


def test_bacon_decomposition_agrees(panel):
    out = sp.bacon_decomposition(panel, y="y1", treat="post", time="year", id="unit")
    assert abs(out["beta_twfe"] - 27 / 11) < 1e-12
    assert abs(out["weighted_sum"] - 27 / 11) < 1e-12
    assert sorted(out["decomposition"]["weight"]) == pytest.approx(
        sorted(w for _, w in BACON.values()), abs=1e-12
    )


def test_dcdh_weights_are_the_books(panel):
    r = sp.twfe_decomposition(
        panel, y="y1", group="unit", time="year", first_treat="first"
    )
    w = r.model_info["dcdh_weights"].set_index(["cohort", "period"])["dcdh_weight"]
    for year in (2003, 2004, 2005):
        assert abs(w[(2003, year)] - 5 / 33) < 1e-12
    for year in (2006, 2007, 2008, 2009):
        assert abs(w[(2003, year)]) < 1e-12
        assert abs(w[(2006, year)] - 3 / 22) < 1e-12
    assert abs(w.sum() - 1.0) < 1e-12
    assert r.model_info["n_negative_weights_dcdh"] == 0
    # the coefficient is the weights applied to the cell effects (3 and 2)
    effect = np.where(w.index.get_level_values("cohort") == 2003, 3.0, 2.0)
    assert abs(float((w.to_numpy() * effect).sum()) - 27 / 11) < 1e-12


def test_growing_effect_matches_stata(panel):
    """Second outcome of the chapter: Stata prints 3.8045454."""
    r = sp.twfe_decomposition(
        panel, y="y2", group="unit", time="year", first_treat="first"
    )
    assert abs(r.estimate - 3.8045454) < 2e-7
    assert abs(r.detail["weighted_est"].sum() - r.estimate) < 1e-12
    # the later cohort is compared with a unit whose effect is still growing
    later = r.detail[r.detail["type"] == "Later vs Earlier"]
    assert float(later["estimate"].iloc[0]) < 0


def test_standard_error_is_the_clustered_one_of_xtreg():
    """``se`` is the unit-clustered standard error of the coefficient, with
    the small-sample factor of Stata ``xtreg, fe vce(cluster)``; the
    translated command is the Stata-validated path to the same number."""
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    r = sp.twfe_decomposition(
        df, y="y", group="unit", time="time", first_treat="first_treat"
    )
    df["post"] = (df["first_treat"].notna() & (df["time"] >= df["first_treat"])) * 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fe = sp.stata(
            "xtset unit time\nxtreg y post i.time, fe vce(cluster unit)", data=df
        )
    assert abs(r.estimate - float(fe.params["post"])) < 1e-10
    assert abs(r.se - float(fe.std_errors["post"])) < 1e-10
    assert abs(r.detail["weighted_est"].sum() - r.estimate) < 1e-10


def test_unbalanced_panel_keeps_the_coefficient_and_the_weights():
    df = sp.dgp_did(n_units=40, n_periods=6, staggered=True, seed=3)
    df = df.drop(df.index[[5, 77, 140]])
    with pytest.warns(UserWarning, match="unbalanced"):
        r = sp.twfe_decomposition(
            df, y="y", group="unit", time="time", first_treat="first_treat"
        )
    assert r.detail.empty and np.isnan(r.model_info["bacon_att"])
    df["post"] = (df["first_treat"].notna() & (df["time"] >= df["first_treat"])) * 1.0
    fit = sp.regress("y ~ post + C(unit) + C(time)", data=df)
    assert abs(r.estimate - float(fit.params["post"])) < 1e-9
    assert abs(r.model_info["dcdh_weights"]["dcdh_weight"].sum() - 1.0) < 1e-10


def test_no_treated_unit_is_an_error(panel):
    never = panel.assign(first=0)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.twfe_decomposition(
            never, y="y1", group="unit", time="year", first_treat="first"
        )
