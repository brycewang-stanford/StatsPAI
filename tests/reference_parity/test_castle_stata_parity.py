"""Stata / R parity for the Cheng-Hoekstra (2013) castle-doctrine panel.

Reference numbers were produced on this machine with **Stata 18 MP** and
**R 4.x `did`** against the same bundled CSV; the exact commands are
recorded next to each constant so they can be re-run.  Nothing here is
mocked — every target is real reference-software output.

Stata (mixtape ``Do/castle_1.do`` specification)::

    use castle.dta, clear
    xtset sid year
    global demo blackm_15_24 whitem_15_24 blackm_25_44 whitem_25_44
    global lintrend trend_1-trend_51
    global region r20001-r20104
    global spending l_exp_subsidy l_exp_pubwelfare
    global xvar l_police unemployrt poverty l_income l_prisoner ///
                l_lagprisoner $demo $spending
    xi: xtreg l_homicide i.year $region $xvar $lintrend post ///
        [aweight=popwt], fe vce(cluster sid)

Tolerance note: the bundled CSV round-trips Stata ``float`` storage
through text, so agreement is ~1e-7 rather than machine epsilon.  1e-6
still pins five significant digits of every coefficient.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

# Reference values -------------------------------------------------------
# (coefficient on `post`, cluster-robust SE), Stata 18 MP.
STATA_TWFE_UNWEIGHTED = (0.069398429, 0.055859635)
STATA_TWFE_WEIGHTED = (0.075533239, 0.033193606)
STATA_TWFE_WEIGHTED_CONTROLS = (0.079634870, 0.030875590)
STATA_TWFE_FULL = (0.076948986, 0.033937717)

# Stata `bacondecomp l_homicide post` (Goodman-Bacon 2021).
STATA_BACON_TWFE = 0.069398429
STATA_BACON_NEVER_TREATED_WEIGHT = 0.8988088336
STATA_BACON_N_COMPARISONS = 25

# Stata `csdid ..., agg(simple)` == R `did::aggte(type="simple")`.
CSDID_SIMPLE_ATT_GVAR_EFFYEAR = 0.110383035
CSDID_SIMPLE_ATT_GVAR_EFFYEAR_PLUS1 = 0.019402808

ATOL = 1e-6

XVAR = [
    "l_police",
    "unemployrt",
    "poverty",
    "l_income",
    "l_prisoner",
    "l_lagprisoner",
    "blackm_15_24",
    "whitem_15_24",
    "blackm_25_44",
    "whitem_25_44",
    "l_exp_subsidy",
    "l_exp_pubwelfare",
]


@pytest.fixture(scope="module")
def castle():
    return sp.datasets.castle_doctrine(
        region_year_fe=True, state_trends=True, event_time=True
    )


def _post(result):
    return float(result.params["post"]), float(result.std_errors["post"])


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


def test_panel_shape_and_treatment_structure(castle):
    """50 states x 11 years, 21 staggered adopters, 29 never-treated."""
    assert castle.shape[0] == 550
    assert castle["sid"].nunique() == 50
    assert sorted(castle["year"].unique()) == list(range(2000, 2011))

    treated = castle.loc[castle["effyear"].notna(), "sid"].nunique()
    never = castle.loc[castle["effyear"].isna(), "sid"].nunique()
    assert (treated, never) == (21, 29)
    assert sorted(castle["effyear"].dropna().unique()) == [
        2005.0,
        2006.0,
        2007.0,
        2008.0,
        2009.0,
    ]


def test_post_excludes_the_adoption_year(castle):
    """`post` is 1{year > effyear}, NOT 1{year >= effyear}.

    Cheng & Hoekstra code the adoption year as untreated because the law
    was in force for only part of it.  This guards the 21 observations
    that a naive reconstruction would flip.
    """
    eff = castle["effyear"].fillna(9999)
    assert np.array_equal((castle["year"] > eff).astype(float), castle["post"].values)

    naive = (castle["year"] >= eff).astype(float)
    assert int((naive != castle["post"]).sum()) == 21


def test_cdl_carries_fractional_exposure_in_adoption_year(castle):
    """`cdl` is fractional in the adoption year and 1 once post turns on."""
    adoption = castle[castle["year"] == castle["effyear"]]
    assert len(adoption) == 21
    assert ((adoption["cdl"] > 0) & (adoption["cdl"] < 1)).all()
    assert (adoption["post"] == 0).all()
    assert (castle.loc[castle["post"] == 1, "cdl"] == 1).all()


def test_reconstructed_design_matrix_columns(castle):
    """44 region x year dummies and 51 state trends, matching Stata's varlists."""
    region = [c for c in castle.columns if c.startswith("r20")]
    trends = [c for c in castle.columns if c.startswith("trend_")]
    assert len(region) == 44
    assert len(trends) == 51

    # trend_j = 1{sid == j} * (year - 1999); sid 9 (DC) is absent from the
    # panel, so trend_9 is identically zero — as in the published file.
    assert (castle["trend_9"] == 0).all()
    alabama = castle[castle["sid"] == 1]
    assert np.array_equal(
        alabama["trend_1"].values, (alabama["year"] - 1999).astype(float).values
    )


# ---------------------------------------------------------------------------
# TWFE ladder — Stata parity
# ---------------------------------------------------------------------------


def test_twfe_unweighted_matches_stata(castle):
    b, se = _post(
        sp.feols("l_homicide ~ post | sid + year", data=castle, vcov={"CRV1": "sid"})
    )
    assert b == pytest.approx(STATA_TWFE_UNWEIGHTED[0], abs=ATOL)
    assert se == pytest.approx(STATA_TWFE_UNWEIGHTED[1], abs=ATOL)


def test_twfe_weighted_matches_stata(castle):
    """Exercises Stata `aweight` semantics, not just the point estimate."""
    b, se = _post(
        sp.feols(
            "l_homicide ~ post | sid + year",
            data=castle,
            weights="popwt",
            vcov={"CRV1": "sid"},
        )
    )
    assert b == pytest.approx(STATA_TWFE_WEIGHTED[0], abs=ATOL)
    assert se == pytest.approx(STATA_TWFE_WEIGHTED[1], abs=ATOL)


def test_twfe_weighted_with_controls_matches_stata(castle):
    fml = "l_homicide ~ post + " + " + ".join(XVAR) + " | sid + year"
    b, se = _post(sp.feols(fml, data=castle, weights="popwt", vcov={"CRV1": "sid"}))
    assert b == pytest.approx(STATA_TWFE_WEIGHTED_CONTROLS[0], abs=ATOL)
    assert se == pytest.approx(STATA_TWFE_WEIGHTED_CONTROLS[1], abs=ATOL)


def test_twfe_full_specification_matches_stata(castle):
    """The paper's headline spec: 19 of the 95 extra regressors are collinear.

    This is the strongest of the four checks — it only passes if StatsPAI
    drops exactly the same collinear columns Stata does.
    """
    region = [c for c in castle.columns if c.startswith("r20")]
    trends = [c for c in castle.columns if c.startswith("trend_")]
    fml = "l_homicide ~ post + " + " + ".join(XVAR + region + trends) + " | sid + year"
    b, se = _post(sp.feols(fml, data=castle, weights="popwt", vcov={"CRV1": "sid"}))
    assert b == pytest.approx(STATA_TWFE_FULL[0], abs=ATOL)
    assert se == pytest.approx(STATA_TWFE_FULL[1], abs=ATOL)


# ---------------------------------------------------------------------------
# Goodman-Bacon decomposition — Stata `bacondecomp` parity
# ---------------------------------------------------------------------------


def test_bacon_decomposition_matches_stata(castle):
    bacon = sp.bacon_decomposition(
        castle, y="l_homicide", treat="post", time="year", id="sid"
    )
    assert bacon["beta_twfe"] == pytest.approx(STATA_BACON_TWFE, abs=ATOL)
    assert bacon["n_comparisons"] == STATA_BACON_N_COMPARISONS

    dec = bacon["decomposition"]
    # Weights must sum to 1 and reproduce the TWFE coefficient.
    assert dec["weight"].sum() == pytest.approx(1.0, abs=1e-9)
    assert float((dec["estimate"] * dec["weight"]).sum()) == pytest.approx(
        STATA_BACON_TWFE, abs=ATOL
    )

    clean = dec.loc[dec["type"] == "Treated vs Untreated", "weight"].sum()
    assert float(clean) == pytest.approx(STATA_BACON_NEVER_TREATED_WEIGHT, abs=ATOL)


# ---------------------------------------------------------------------------
# Callaway-Sant'Anna — Stata `csdid` / R `did` parity
# ---------------------------------------------------------------------------


def test_callaway_santanna_simple_att_matches_csdid(castle):
    """Point estimate parity with both csdid and R did (they agree)."""
    cs = sp.callaway_santanna(
        castle,
        y="l_homicide",
        g="gvar",
        t="year",
        i="sid",
        control_group="nevertreated",
    )
    att = float(sp.aggte(cs, type="simple", bstrap=False).estimate)
    assert att == pytest.approx(CSDID_SIMPLE_ATT_GVAR_EFFYEAR, abs=ATOL)


def test_cohort_coding_changes_the_answer(castle):
    """gvar = effyear vs effyear+1 is consequential — pin both.

    This is a documented modelling trap, not a defect: coding the cohort
    as ``effyear + 1`` (consistent with ``post``) pushes the partially
    treated adoption year into the Callaway-Sant'Anna base period.
    """
    df = castle.copy()
    df["gvar_plus1"] = (df["effyear"] + 1).fillna(0)
    cs = sp.callaway_santanna(
        df,
        y="l_homicide",
        g="gvar_plus1",
        t="year",
        i="sid",
        control_group="nevertreated",
    )
    att = float(sp.aggte(cs, type="simple", bstrap=False).estimate)
    assert att == pytest.approx(CSDID_SIMPLE_ATT_GVAR_EFFYEAR_PLUS1, abs=ATOL)

    # The two conventions differ by far more than sampling noise.
    assert abs(att - CSDID_SIMPLE_ATT_GVAR_EFFYEAR) > 0.05


def test_twfe_understates_relative_to_callaway_santanna(castle):
    """The teaching payload: heterogeneity-robust ATT exceeds TWFE here."""
    twfe, _ = _post(
        sp.feols("l_homicide ~ post | sid + year", data=castle, vcov={"CRV1": "sid"})
    )
    cs = sp.callaway_santanna(
        castle,
        y="l_homicide",
        g="gvar",
        t="year",
        i="sid",
        control_group="nevertreated",
    )
    att = float(sp.aggte(cs, type="simple", bstrap=False).estimate)
    assert att > twfe
    assert att / twfe > 1.5


# ---------------------------------------------------------------------------
# Replication-guide wiring
# ---------------------------------------------------------------------------


def test_replication_entry_is_registered_and_renders():
    table = sp.list_replications()
    row = table.loc[table["key"] == "castle_2013"]
    assert len(row) == 1
    assert bool(row["has_real_data"].iloc[0])
    assert bool(row["has_classic_track"].iloc[0])
    assert bool(row["has_modern_track"].iloc[0])

    data, guide = sp.replicate("castle_2013")
    assert data.shape[0] == 550
    assert "Cheng" in guide
    assert "CAVEATS" in guide
    # The cohort-coding trap must survive into the rendered guide.
    assert "effyear + 1" in guide


def test_golden_numbers_in_guide_match_this_modules_constants():
    """The guide's pinned numbers and this test file must not drift apart."""
    from statspai.smart.replicate import _REPLICATIONS

    entry = _REPLICATIONS["castle_2013"]
    gold = {
        label: value
        for label, value, _paper, _cite in entry["classic"]["golden_numbers"]
    }
    assert gold["TWFE unweighted beta_post"] == pytest.approx(
        STATA_TWFE_UNWEIGHTED[0], abs=1e-9
    )
    assert gold["TWFE full (region x year + trends)"] == pytest.approx(
        STATA_TWFE_FULL[0], abs=1e-9
    )

    pinned = {label: value for label, value, _note in entry["modern"]["pinned_numbers"]}
    assert pinned["Bacon never-treated weight"] == pytest.approx(
        STATA_BACON_NEVER_TREATED_WEIGHT, abs=1e-9
    )
    assert pinned["Callaway-Sant'Anna simple ATT (gvar=effyear)"] == pytest.approx(
        CSDID_SIMPLE_ATT_GVAR_EFFYEAR, abs=1e-9
    )
