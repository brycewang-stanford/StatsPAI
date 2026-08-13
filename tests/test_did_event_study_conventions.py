"""Event-study reference conventions: Stata parity and analytic identities.

Roth (2026) shows that recent DiD methods build their pre-treatment
event-study coefficients differently from their post-treatment ones, so
the plotted path can break at the treatment date with no treatment effect
present.  These tests pin StatsPAI's three imputation conventions against
the reference implementation and against the analytic identities the note
derives, and check that the convention registry describes what the code
actually does rather than what it is supposed to do.

Stata reference provenance
--------------------------
Stata 18 MP, ``did_imputation`` version 22 November 2023 (Kirill
Borusyak, SSC S458957), executed 2026-08-11 on the two fixtures built
below by :func:`_roth_panel`.  Commands::

    import delimited "roth.csv", clear case(preserve)
    replace g = . if g == 0
    did_imputation y unit time g, pretrends(15) horizons(0/9) autosample

    import delimited "roth2.csv", clear case(preserve)
    replace g = . if g == 0
    did_imputation y unit time g, pretrends(9)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# --------------------------------------------------------------------- #
# Fixtures: Roth (2026) figure-1 design — non-staggered, zero treatment
# effect, treated group on a linear relative trend of slope gamma.
# --------------------------------------------------------------------- #


def _roth_panel(
    n_units: int = 100,
    treated_frac: float = 0.5,
    t_min: int = -15,
    t_max: int = 10,
    gamma: float = 0.5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_treated = int(round(n_units * treated_frac))
    times = np.arange(t_min, t_max + 1)
    offset = 1 - t_min
    rows = []
    for i in range(n_units):
        treated = 1 if i < n_treated else 0
        for t in times:
            rows.append(
                (i, t + offset, treated, gamma * t * treated + rng.standard_normal())
            )
    df = pd.DataFrame(rows, columns=["unit", "time", "treated", "y"])
    treat_period = 1 + offset
    df["g"] = np.where(df["treated"] == 1, treat_period, 0)
    df["g_nan"] = np.where(df["treated"] == 1, float(treat_period), np.nan)
    return df


def _staggered_panel(n_units: int = 60, n_times: int = 12, seed: int = 11):
    rng = np.random.default_rng(seed)
    cohorts = [5, 7, 9, 0]
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        u = rng.normal()
        for t in range(1, n_times + 1):
            eff = 0.0 if (g == 0 or t < g) else 1.0 + 0.3 * (t - g)
            rows.append((i, t, g, u + 0.05 * t + eff + rng.normal()))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "y"])


# Stata did_imputation, pretrends(15): pre1 .. pre15 == relative time -1 .. -15
_STATA_A_PRE_B = [
    7.440786425517536,
    6.817529709276191,
    6.530321036474332,
    6.068492046425464,
    5.5279864364767475,
    5.323100272267267,
    4.335305263605482,
    3.846821978856736,
    3.1551893018607506,
    2.847332865871384,
    2.546454140906984,
    2.031564734139339,
    1.7221515359177515,
    0.852268271343511,
    0.32821655836491376,
]
_STATA_A_PRE_SE = [
    0.292064487435709,
    0.2787582515334544,
    0.30417689537258535,
    0.2643300799558943,
    0.2930748150251891,
    0.2740532094500764,
    0.2622565157898193,
    0.26880857667726277,
    0.28599504018503874,
    0.2844870949555121,
    0.28857278896141864,
    0.28756881564209674,
    0.27503697656642373,
    0.26076225291435035,
    0.30575051022578054,
]
_STATA_A_TAU = [
    3.9085393213067863,
    4.66223889407801,
    5.60037008659233,
    5.391093506614911,
    6.326328422535254,
    6.823505939967774,
    7.50279019028167,
    8.050485531452523,
    8.259795094606337,
    8.664977810330306,
]
# Second design: n=80, 25% treated, t in [-9, 6], gamma=0.3, seed 21, pretrends(9)
_STATA_B_PRE_B = [
    2.59458681993218,
    2.37664397222243,
    2.31557468595827,
    1.56197426242044,
    1.47892678406271,
    1.49593968120492,
    1.08366578659271,
    0.796443823583626,
    0.183559844803602,
]
_STATA_B_PRE_SE = [
    0.347483062411143,
    0.306046159159309,
    0.328268561843719,
    0.348559277149934,
    0.307991239248573,
    0.307428757298222,
    0.291720897979889,
    0.364667112877595,
    0.324524614358386,
]


@pytest.fixture(scope="module")
def roth_a():
    return _roth_panel(seed=7)


@pytest.fixture(scope="module")
def roth_b():
    return _roth_panel(
        n_units=80, treated_frac=0.25, t_min=-9, t_max=6, gamma=0.3, seed=21
    )


# --------------------------------------------------------------------- #
# Stata parity
# --------------------------------------------------------------------- #


def test_bjs_pretrends_match_stata_design_a(roth_a):
    """Leads and their SEs reproduce Stata did_imputation, pretrends(15)."""
    res = sp.did_imputation(
        roth_a,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        horizon=list(range(-15, 10)),
        cluster="unit",
    )
    es = res.model_info["event_study"].set_index("relative_time")
    assert res.model_info["pretrend_method"] == "bjs"

    for k, (b, s) in enumerate(zip(_STATA_A_PRE_B, _STATA_A_PRE_SE), start=1):
        # 1e-9 absolute is four orders looser than the observed 1e-12 gap;
        # the slack absorbs the iterative sparse solve's tolerance.
        assert es.loc[-k, "att"] == pytest.approx(b, abs=1e-9)
        assert es.loc[-k, "se"] == pytest.approx(s, rel=1e-8)
    for k, tau in enumerate(_STATA_A_TAU):
        assert es.loc[k, "att"] == pytest.approx(tau, abs=1e-9)


def test_bjs_pretrends_match_stata_design_b(roth_b):
    """A second design pins the degrees-of-freedom convention, not one draw."""
    res = sp.did_imputation(
        roth_b,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        pretrends=9,
        cluster="unit",
    )
    es = res.model_info["event_study"].set_index("relative_time")
    for k, (b, s) in enumerate(zip(_STATA_B_PRE_B, _STATA_B_PRE_SE), start=1):
        assert es.loc[-k, "att"] == pytest.approx(b, abs=1e-9)
        assert es.loc[-k, "se"] == pytest.approx(s, rel=1e-8)


def test_bjs_joint_test_uses_full_covariance(roth_b):
    res = sp.did_imputation(
        roth_b,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        pretrends=9,
        cluster="unit",
    )
    test = res.model_info["pretrend_test"]
    assert "wald-cluster" in test["method"]
    assert test["df"] == 9
    assert test["statistic"] > 0


# --------------------------------------------------------------------- #
# Analytic identities from Roth (2026)
# --------------------------------------------------------------------- #


def _twfe_path(df, window):
    res = sp.event_study(
        df,
        y="y",
        treat_time="g_nan",
        time="time",
        unit="unit",
        window=window,
        cluster="unit",
    )
    es = res.model_info["event_study"]
    return {int(k): float(v) for k, v in zip(es["relative_time"], es["att"])}


def test_symmetric_equals_twfe_up_to_a_common_shift(roth_a):
    """beta^{BJS,new} = beta^{TWFE} + c for every relative time."""
    horizon = list(range(-16, 10))
    res = sp.did_imputation(
        roth_a,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        horizon=horizon,
        cluster="unit",
        pretrend_method="symmetric",
    )
    es = res.model_info["event_study"].set_index("relative_time")["att"]
    twfe = _twfe_path(roth_a, (-16, 9))
    shifts = [es.loc[k] - twfe[k] for k in es.index if k in twfe]
    assert max(shifts) - min(shifts) == pytest.approx(0.0, abs=1e-8)


def test_in_sample_leads_are_attenuated_by_the_untreated_share(roth_a):
    """Li-Strezhnev / Roth: in-sample leads equal (N0/N) times symmetric."""
    horizon = list(range(-16, 10))
    kwargs = dict(
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        horizon=horizon,
        cluster="unit",
    )
    legacy = sp.did_imputation(roth_a, pretrend_method="in-sample", **kwargs)
    symmetric = sp.did_imputation(roth_a, pretrend_method="symmetric", **kwargs)
    a = legacy.model_info["event_study"].set_index("relative_time")["att"]
    b = symmetric.model_info["event_study"].set_index("relative_time")["att"]
    pre = [k for k in a.index if k < 0]
    ratios = np.array([a.loc[k] / b.loc[k] for k in pre])
    n_units = roth_a["unit"].nunique()
    n_untreated = int((roth_a.groupby("unit")["g"].first() == 0).sum())
    assert np.allclose(ratios, n_untreated / n_units, atol=1e-10)
    # Post-treatment coefficients are the same object under both.
    post = [k for k in a.index if k >= 0]
    assert np.allclose([a.loc[k] for k in post], [b.loc[k] for k in post], atol=1e-12)


def test_cs_universal_reproduces_dynamic_twfe(roth_a):
    """Roth's CS recommendation: a universal base period restores symmetry."""
    res = sp.callaway_santanna(
        roth_a,
        y="y",
        g="g",
        t="time",
        i="unit",
        estimator="reg",
        base_period="universal",
    )
    tidy = sp.aggte(res, type="dynamic").tidy()
    rows = tidy[tidy["type"] == "event_study"]
    got = {
        int(t.replace("event_", "")): float(v)
        for t, v in zip(rows["term"], rows["estimate"])
    }
    twfe = _twfe_path(roth_a, (-16, 9))
    for k, v in got.items():
        assert v == pytest.approx(twfe[k], abs=1e-8)


# --------------------------------------------------------------------- #
# The convention registry has to describe the code
# --------------------------------------------------------------------- #


def test_registry_claims_hold_on_roths_design(roth_a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.compare_event_study_conventions(
            roth_a,
            y="y",
            unit="unit",
            time="time",
            first_treat="g",
        )
    drift = [w for w in caught if "convention registry" in str(w.message)]
    assert not drift, [str(w.message) for w in drift]

    table = res.table.set_index("key")
    # Every recorded twfe_comparable claim must match what the data show.
    assert (table["twfe_comparable"] == table["matches_twfe"]).all()

    # The three asymmetric conventions have three distinct signatures.
    kink = table.loc["callaway_santanna[base_period=varying]"]
    assert abs(kink["asymmetry"]) > 1.0 and kink["shape_gap"] > 1.0

    jump = table.loc["did_imputation[pretrend_method=bjs]"]
    assert abs(jump["asymmetry"]) > 1.0
    assert jump["shape_gap"] == pytest.approx(0.0, abs=1e-6)

    attenuated = table.loc["did_imputation[pretrend_method=in-sample]"]
    assert attenuated["asymmetry"] == pytest.approx(0.0, abs=1e-6)
    assert attenuated["shape_gap"] > 1.0

    assert "asym" in res.summary()
    assert set(res.paths["key"]) == set(table.index)


def test_event_study_convention_lookup():
    full = sp.event_study_convention()
    assert isinstance(full, pd.DataFrame) and len(full) > 5
    one = sp.event_study_convention("callaway_santanna[base_period=varying]")
    assert one["symmetric"] is False
    by_name = sp.event_study_convention("did_imputation")
    assert len(by_name) == 3
    with pytest.raises(KeyError, match="No event-study convention"):
        sp.event_study_convention("not_an_estimator")


# --------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------- #


def test_bjs_refuses_when_leads_exhaust_the_pre_history(roth_b):
    """All leads requested -> collinear with the unit effects, so refuse."""
    n_pre = 10  # t in [-9, 6] with treatment at 0 -> relative times -10..-1
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.did_imputation(
            roth_b,
            y="y",
            group="unit",
            time="time",
            first_treat="g",
            pretrends=n_pre,
            cluster="unit",
        )


def test_symmetric_refuses_on_staggered_designs():
    df = _staggered_panel()
    with pytest.raises(MethodIncompatibility, match="non-staggered"):
        sp.did_imputation(
            df,
            y="y",
            group="unit",
            time="time",
            first_treat="g",
            pretrends=2,
            pretrend_method="symmetric",
        )


def test_bad_pretrend_method_is_rejected(roth_a):
    with pytest.raises(ValueError, match="pretrend_method must be"):
        sp.did_imputation(
            roth_a,
            y="y",
            group="unit",
            time="time",
            first_treat="g",
            pretrend_method="fect",
        )


def test_compare_refuses_staggered_and_all_treated():
    df = _staggered_panel()
    with pytest.raises(MethodIncompatibility, match="non-staggered"):
        sp.compare_event_study_conventions(
            df, y="y", unit="unit", time="time", first_treat="g"
        )

    always = _roth_panel(treated_frac=1.0, seed=3)
    with pytest.raises(MethodIncompatibility, match="never-treated"):
        sp.compare_event_study_conventions(
            always, y="y", unit="unit", time="time", first_treat="g"
        )


def test_sp_did_forwards_the_pretrend_options(roth_a):
    """An allowlisted argument must reach the estimator, not just be accepted."""
    with pytest.raises(ValueError, match="pretrend_method must be"):
        sp.did(
            roth_a,
            y="y",
            treat="g",
            time="time",
            id="unit",
            method="bjs",
            pretrends=3,
            pretrend_method="fect",
        )
    res = sp.did(
        roth_a, y="y", treat="g", time="time", id="unit", method="bjs", pretrends=3
    )
    assert np.isfinite(float(res.estimate))


# --------------------------------------------------------------------- #
# R did2s reference
#
# did2s 1.2.1 (Kyle Butts), R 4.5.2, executed 2026-08-12 on the
# _roth_panel(seed=7) fixture dumped to CSV:
#
#     df$g[df$g == 0] <- NA
#     df$rel <- ifelse(is.na(df$g), -Inf, df$time - df$g)
#     df$treat <- !is.na(df$g) & df$time >= df$g
#     did2s(df, yname = "y", first_stage = ~ 0 | unit + time,
#           second_stage = ~ i(rel, ref = c(-Inf)), treatment = "treat",
#           cluster_var = "unit")
#
# Keys are relative time.
# --------------------------------------------------------------------- #
_DID2S_COEF = {
    -16: -1.85542251804078,
    -15: -1.69131423885834,
    -14: -1.42928838236901,
    -13: -0.994346750081898,
    -12: -0.83964015097109,
    -11: -0.582195447587294,
    -10: -0.431756085105091,
    -9: -0.277827867110392,
    -8: 0.0679884713875875,
    -7: 0.312230113761986,
    -6: 0.80612761809288,
    -5: 0.908570700197536,
    -4: 1.178823505172,
    -3: 1.40973800019639,
    -2: 1.55334233659746,
    -1: 1.86497069471807,
    0: 3.90853932130679,
    1: 4.66223889407801,
    2: 5.60037008659233,
    3: 5.39109350661491,
    4: 6.32632842253525,
    5: 6.82350593996777,
    6: 7.50279019028167,
    7: 8.05048553145252,
    8: 8.25979509460634,
    9: 8.66497781033031,
}
_DID2S_SE = {
    -16: 0.210059868942641,
    -15: 0.193310350316733,
    -14: 0.172840471256294,
    -13: 0.142342116977372,
    -12: 0.126810044323152,
    -11: 0.107226349689829,
    -10: 0.0970128795891082,
    -9: 0.0901029789466948,
    -8: 0.0834864829809693,
    -7: 0.0952633160148166,
    -6: 0.121612199075206,
    -5: 0.12827502469317,
    -4: 0.146689874048547,
    -3: 0.173641500728118,
    -2: 0.175943795015964,
    -1: 0.207078394446953,
    0: 0.202953471251438,
    1: 0.21507324988493,
    2: 0.191248039157718,
    3: 0.215442999691519,
    4: 0.221887931917314,
    5: 0.211588887529905,
    6: 0.193082587600291,
    7: 0.237752791499147,
    8: 0.195066591820072,
    9: 0.222382370399876,
}


def test_gardner_event_study_matches_r_did2s(roth_a):
    """sp.gardner_did reproduces did2s coefficients at every horizon.

    This is the control for the did_imputation finding: the same
    in-sample lead construction is *correct* here, because did2s is what
    gardner_did documents alignment with. A convention is only a defect
    when it is not the one the function claims.
    """
    res = sp.gardner_did(
        roth_a,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        event_study=True,
        horizon=list(range(-16, 10)),
    )
    es = res.model_info["event_study"]
    coef = {
        int(label.replace("D_k", "").replace("+", "")): float(es["coef"][label])
        for label in es["horizon"]
    }
    for k, expected in _DID2S_COEF.items():
        assert coef[k] == pytest.approx(expected, abs=1e-10)


def test_in_sample_leads_are_the_did2s_convention(roth_a):
    """pretrend_method='in-sample' is did2s's construction, not a label."""
    res = sp.did_imputation(
        roth_a,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        horizon=list(range(-16, 10)),
        cluster="unit",
        pretrend_method="in-sample",
    )
    es = res.model_info["event_study"].set_index("relative_time")["att"]
    for k, expected in _DID2S_COEF.items():
        assert float(es.loc[k]) == pytest.approx(expected, abs=1e-9)


def test_gardner_analytic_event_study_ses_are_smaller_than_did2s(roth_a):
    """The documented SE understatement, measured against the reference.

    ``sp.gardner_did`` warns that its analytic standard error ignores the
    variance from estimating the stage-one fixed effects. That warning
    has never been priced against did2s, which carries the two-stage
    correction. It is priced here so the caveat is a number rather than
    an adjective, and so a future change that silently closes or widens
    the gap has to move this test.
    """
    res = sp.gardner_did(
        roth_a,
        y="y",
        group="unit",
        time="time",
        first_treat="g",
        event_study=True,
        horizon=list(range(-16, 10)),
    )
    es = res.model_info["event_study"]
    ratios = np.array(
        [
            float(es["se"][label]) / _DID2S_SE[k]
            for label in es["horizon"]
            for k in [int(label.replace("D_k", "").replace("+", ""))]
        ]
    )
    # Predominantly, but not uniformly, too small: the median horizon is
    # about 0.71 of the reference, roughly seven horizons in ten are
    # below it, and no horizon exceeds 1.6. The non-uniform direction is
    # itself informative --- an omitted positive variance term would be
    # uniformly downward, so something else is also moving.
    assert np.median(ratios) < 0.80
    assert ratios.min() > 0.55
    assert ratios.max() < 1.60
    assert (ratios < 1.0).mean() > 0.60
