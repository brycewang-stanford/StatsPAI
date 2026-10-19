"""R ``fixest`` / Stata ``did_imputation`` numbers for the 2026-10-19 DiD fixes.

Three behaviours added or changed on that date are pinned against the
reference implementations on identical data (generated below; the R and
Stata runs read the same rows written with ``DataFrame.to_csv``, which
round-trips every double):

1. ``sp.sun_abraham`` drops a cohort treated from the first period. The
   reference is ``fixest::sunab`` on the panel **without** that cohort:
   fixest itself keeps the cohort and drops a collinear cell, which moves
   the leads of the other cohorts (t::-7 goes from -0.18 to +0.57 on this
   panel), so it is not a usable reference on the full panel.
2. ``sp.sun_abraham(i=None)`` (repeated cross sections) is
   ``feols(y ~ sunab(g2, t) | g2 + t, vcov = "hetero")``; ``anticipation=``
   and ``ref_p=`` are ``sunab(..., ref.p = )``.
3. ``sp.did_imputation(shift=)`` is Stata ``did_imputation, shift()``.

Reference scripts (fixest 0.14.0, R 4.5; Stata 18 MP, did_imputation of
2023-11-22, ``set type double`` + ``import delimited ..., asdouble``)::

    m  <- feols(y ~ sunab(g2, t) | id + t, d, cluster = ~id)        # (1)
    m1 <- feols(y ~ sunab(g2, t) | g2 + t, d, vcov = "hetero")     # (2)
    m2 <- feols(y ~ sunab(g2, t, ref.p = -2) | state + t, d, cluster = ~state)
    m3 <- feols(y ~ x + sunab(g2, t, ref.p = c(-2, -4)) | state + t, d,
                cluster = ~state)
    summary(m, agg = "att")
    did_imputation y id t g, horizons(0/7) shift(1) minn(0)           # (3)
    did_imputation y id t g, shift(2) minn(0)

with ``g2 = ifelse(g == 0, 10000, g)``. Point estimates and SEs agree to
~1e-11 for fixest. Stata's ``reghdfe`` demeans iteratively (tolerance
1e-8), so the BJS rows agree to ~4e-8 and are held to the T2 budget of
1e-6.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RTOL_FIXEST = 1e-8
RTOL_STATA = 1e-6  # T2 budget; reghdfe's iterative demeaning, see docstring


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _always_treated_panel():
    rng = np.random.default_rng(1)
    n, T = 400, 10
    g = rng.choice([0, 4, 6, 8], size=n)
    a = rng.normal(size=n)
    ids = np.repeat(np.arange(n), T)
    tt = np.tile(np.arange(1, T + 1), n)
    gg = np.repeat(g, T)
    d = ((gg > 0) & (tt >= gg)).astype(int)
    eff = np.where(d == 1, 2.0 * (1 + 0.2 * (tt - gg)), 0.0)
    x = rng.normal(size=n * T)
    y = np.repeat(a, T) + 0.3 * tt + 0.5 * x + eff + rng.normal(size=n * T)
    df = pd.DataFrame(dict(id=ids, t=tt, g=gg, d=d, y=y, x=x))
    first20 = df.loc[df.g == 4, "id"].unique()[:20]
    df.loc[df.id.isin(first20), "g"] = 1
    return df


def _rcs_states():
    rng = np.random.default_rng(7)
    S, T, n = 40, 8, 15
    g_state = rng.choice([0, 3, 5, 6], size=S)
    a = rng.normal(size=S)
    rows = []
    for s in range(S):
        for t in range(1, T + 1):
            for _ in range(n):
                g = g_state[s]
                e = t - g
                # one period of anticipation
                eff = (1.0 + 0.3 * e) if (g > 0 and t >= g - 1) else 0.0
                x = rng.normal()
                rows.append((s, t, g, a[s] + 0.2 * t + 0.5 * x + eff + rng.normal(), x))
    return pd.DataFrame(rows, columns=["state", "t", "g", "y", "x"])


def _shift_panel():
    rng = np.random.default_rng(11)
    n, T = 300, 10
    g = rng.choice([0, 4, 6, 8], size=n)
    a = rng.normal(size=n)
    ids = np.repeat(np.arange(n), T)
    tt = np.tile(np.arange(1, T + 1), n)
    gg = np.repeat(g, T)
    e = tt - gg
    eff = np.where((gg > 0) & (e >= -1), 1.0 + 0.25 * e, 0.0)
    y = np.repeat(a, T) + 0.3 * tt + eff + rng.normal(size=n * T)
    return pd.DataFrame(dict(id=ids, t=tt, g=gg, y=y))


# fixest coeftable on the panel without the always-treated cohort -----------
SA_DROP = {
    -7: (-0.1828502989849867, 0.212177416114145),
    -6: (-0.0816881660295066, 0.228643686629224),
    -5: (-0.3132102822128514, 0.164670903176945),
    -4: (-0.1166621800628581, 0.167468385183162),
    -3: (-0.0926573311573336, 0.109459354680200),
    -2: (-0.1249096283790652, 0.135497820621157),
    0: (1.9014151602723794, 0.135571832999265),
    1: (2.4966969366694505, 0.112755789991924),
    2: (2.7089855164317211, 0.130825280855620),
    3: (3.1509205612365259, 0.165801467980872),
    4: (3.5329561133410081, 0.165413467128284),
    5: (4.2484928855218209, 0.240513682890127),
    6: (4.8870164567275340, 0.218200665115907),
}
SA_DROP_ATT = (2.890026184934193, 0.102747571285153)

RCS_HETERO = {
    -5: (-0.9884221434718177, 0.228488451103517),
    -4: (-0.9175620728737622, 0.174417671100991),
    -3: (-0.8678376515055994, 0.168591549249397),
    -2: (-0.9975014896411106, 0.126635905747585),
    0: (0.0768977404432926, 0.126710447155675),
    1: (0.2947861309669713, 0.131021609284673),
    2: (0.7416767907141784, 0.126500869692829),
    3: (0.9419134484972386, 0.179107123369717),
    4: (1.3184542973612230, 0.252876726376602),
    5: (1.4790566502262286, 0.259770916294859),
}
RCS_HETERO_ATT = (0.604867743353104, 0.116576294726120)

REFP2 = {
    -5: (0.0331343027451293, 0.2482180561980770),
    -4: (-0.0108370330437507, 0.1981008877986833),
    -3: (0.0388873883237182, 0.1201907966605817),
    -1: (0.9975014896394001, 0.0986191523465395),
    0: (1.0743992300828955, 0.0629277289263679),
    1: (1.2922876206066580, 0.0901131562506122),
    2: (1.7391782803536351, 0.1342146864902573),
    3: (1.9233783004200116, 0.1668570244933118),
    4: (2.5140134956771210, 0.1911442504348185),
    5: (2.6746158485421141, 0.2214410399524478),
}
REFP2_ATT = (1.6295347748766447, 0.0900839354774482)

REFP24_X = {
    -5: (-0.0454426330208759, 0.1535189328842108),
    -3: (-0.0193828919600739, 0.0768585502107077),
    -1: (0.9796855408028993, 0.0801323327970539),
    0: (1.0557840745285361, 0.0573822988286412),
    1: (1.3375147464964330, 0.0707007879792751),
    2: (1.6902349827247112, 0.0982712005298514),
    3: (1.9503051325084204, 0.1314330558587063),
    4: (2.5065700344108333, 0.1356203491492777),
    5: (2.5589745274188580, 0.1610863451141389),
}
REFP24_X_ATT = (1.618926223608394, 0.0585896091491389)


def _check(result, table, att):
    es = result.detail.set_index("relative_time")
    assert sorted(es.index) == sorted(table)
    for e, (b, se) in table.items():
        assert es.loc[e, "att"] == pytest.approx(b, rel=RTOL_FIXEST, abs=1e-12)
        assert es.loc[e, "se"] == pytest.approx(se, rel=RTOL_FIXEST)
    mi = result.model_info
    assert mi["att_fixest_att"] == pytest.approx(att[0], rel=RTOL_FIXEST)
    assert mi["se_fixest_att"] == pytest.approx(att[1], rel=RTOL_FIXEST)


def test_sunab_always_treated_cohort_dropped_matches_fixest_without_it():
    df = _always_treated_panel()
    r = _quiet(lambda: sp.sun_abraham(df, "y", "g", "t", "id", share_variance=False))
    assert r.model_info["dropped_unidentified_cohorts"] == [1]
    _check(r, SA_DROP, SA_DROP_ATT)


def test_sunab_repeated_cross_sections_match_fixest_hetero():
    df = _rcs_states()
    r = _quiet(lambda: sp.sun_abraham(df, "y", "g", "t", share_variance=False))
    _check(r, RCS_HETERO, RCS_HETERO_ATT)


def test_sunab_anticipation_matches_fixest_ref_p():
    df = _rcs_states()
    r = _quiet(
        lambda: sp.sun_abraham(
            df, "y", "g", "t", "state", anticipation=1, share_variance=False
        )
    )
    _check(r, REFP2, REFP2_ATT)


def test_sunab_multiple_ref_p_with_covariate_matches_fixest():
    df = _rcs_states()
    r = _quiet(
        lambda: sp.sun_abraham(
            df,
            "y",
            "g",
            "t",
            "state",
            covariates=["x"],
            anticipation=1,
            ref_p=[-2, -4],
            share_variance=False,
        )
    )
    _check(r, REFP24_X, REFP24_X_ATT)


# Stata did_imputation y id t g, horizons(0/7) shift(1) minn(0) -------------
STATA_SHIFT1_B = [
    0.689147351336833,
    0.89699595703215,
    1.04676703643893,
    1.3527822354098,
    1.48477500815287,
    1.98039442730066,
    2.01130248289965,
    2.69908347605197,
]
STATA_SHIFT1_VDIAG = [
    0.00901409310761203,
    0.00951619474564556,
    0.0108047000860267,
    0.0110081396920509,
    0.0186957449074752,
    0.0194926289327677,
    0.0391931516015801,
    0.0372351139778792,
]
# did_imputation y id t g, shift(2) minn(0)
STATA_SHIFT2 = (1.14568505282199, 0.00685307100447066)


def test_did_imputation_shift_event_study_matches_stata():
    df = _shift_panel()
    r = _quiet(
        lambda: sp.did_imputation(
            df, "y", "id", "t", "g", horizon=list(range(8)), shift=1
        )
    )
    es = r.model_info["event_study"].set_index("relative_time")
    for h in range(8):
        assert es.loc[h, "att"] == pytest.approx(STATA_SHIFT1_B[h], rel=RTOL_STATA)
        assert es.loc[h, "se"] == pytest.approx(
            np.sqrt(STATA_SHIFT1_VDIAG[h]), rel=RTOL_STATA
        )


def test_did_imputation_shift_overall_matches_stata():
    df = _shift_panel()
    r = _quiet(lambda: sp.did_imputation(df, "y", "id", "t", "g", shift=2))
    assert r.estimate == pytest.approx(STATA_SHIFT2[0], rel=RTOL_STATA)
    assert r.se == pytest.approx(np.sqrt(STATA_SHIFT2[1]), rel=RTOL_STATA)
