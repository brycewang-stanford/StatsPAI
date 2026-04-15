"""Pinned numerical fixtures: regression guards against silent drift.

Every number in this module is the *exact* output of the current
implementation on a fixed-seed, fully-deterministic DGP (the helper
``_fixture_panel()`` below).  Any future refactor that unintentionally
changes a result will fail these tests and force a conscious decision
either to update the pinned value (with a commit explaining the
numerical change) or to revert the behavioural drift.

The fixtures are checked to 4 decimal places so bit-level floating-point
differences across BLAS backends do not cause spurious failures.
"""
import numpy as np
import pandas as pd
import pytest

from statspai.did import aggte, callaway_santanna


# --------------------------------------------------------------------------- #
# Deterministic DGP                                                           #
# --------------------------------------------------------------------------- #

def _fixture_panel() -> pd.DataFrame:
    """60 units × 8 periods staggered panel, seed = 12345."""
    rng = np.random.default_rng(12345)
    rows = []
    for u in range(60):
        g = [3, 5, 7, 0][u // 15]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def cs_fixture():
    df = _fixture_panel()
    return callaway_santanna(df, y='y', g='g', t='t', i='i',
                             estimator='reg')


# --------------------------------------------------------------------------- #
# Pinned ATT(g, t) values                                                     #
# --------------------------------------------------------------------------- #

# (group, time) -> (att, se).  Generated from the current implementation.
PINNED_ATT_GT = {
    (3, 1): (-0.583666, 0.377119),
    (3, 3): (0.162054,  0.332874),
    (3, 4): (1.146706,  0.365457),
    (3, 5): (0.810629,  0.416610),
    (3, 6): (1.340394,  0.405758),
    (3, 7): (2.299304,  0.378036),
    (3, 8): (2.862484,  0.385927),
    (5, 1): (-0.392430, 0.318560),
    (5, 2): (0.346434,  0.339810),
    (5, 3): (-0.174793, 0.343675),
    (5, 5): (0.289302,  0.359778),
    (5, 6): (0.956634,  0.455490),
    (5, 7): (1.945725,  0.437620),
    (5, 8): (1.986254,  0.295077),
    (7, 1): (0.082153,  0.323557),
    (7, 2): (0.284830,  0.340380),
    (7, 3): (0.663454,  0.417548),
    (7, 4): (0.507629,  0.334881),
    (7, 5): (0.206995,  0.400476),
    (7, 7): (0.617812,  0.295688),
    (7, 8): (0.968689,  0.315171),
}


def test_att_gt_matches_pinned_values(cs_fixture):
    detail = cs_fixture.detail.set_index(['group', 'time'])
    for (g, t), (att_exp, se_exp) in PINNED_ATT_GT.items():
        row = detail.loc[(g, t)]
        assert row['att'] == pytest.approx(att_exp, abs=1e-4), (
            f"ATT(g={g}, t={t}) drifted: "
            f"{row['att']:.6f} vs pinned {att_exp:.6f}"
        )
        assert row['se'] == pytest.approx(se_exp, abs=1e-4), (
            f"SE(g={g}, t={t}) drifted: "
            f"{row['se']:.6f} vs pinned {se_exp:.6f}"
        )


def test_overall_att_matches_pinned(cs_fixture):
    assert cs_fixture.estimate == pytest.approx(1.282166, abs=1e-4)
    assert cs_fixture.se == pytest.approx(0.101724, abs=1e-4)


# --------------------------------------------------------------------------- #
# Pinned aggte(dynamic) values                                                #
# --------------------------------------------------------------------------- #

PINNED_EVENT_STUDY = {
    -6: (0.082153,  0.161875),
    -5: (0.284830,  0.155933),
    -4: (0.135512,  0.129498),
    -3: (0.427031,  0.121281),
    -2: (-0.183822, 0.101445),
     0: (0.356390,  0.093998),
     1: (1.024010,  0.107502),
     2: (1.378177,  0.157336),
     3: (1.663324,  0.122397),
     4: (2.299304,  0.191350),
     5: (2.862484,  0.200262),
}


def test_aggte_dynamic_matches_pinned_values(cs_fixture):
    es = aggte(cs_fixture, type='dynamic',
               n_boot=500, random_state=42)
    got = es.detail.set_index('relative_time')
    for e, (att_exp, se_exp) in PINNED_EVENT_STUDY.items():
        row = got.loc[e]
        assert row['att'] == pytest.approx(att_exp, abs=1e-4), (
            f"dynamic ATT(e={e}) drifted: "
            f"{row['att']:.6f} vs pinned {att_exp:.6f}"
        )
        assert row['se'] == pytest.approx(se_exp, abs=1e-3), (
            f"dynamic SE(e={e}) drifted: "
            f"{row['se']:.6f} vs pinned {se_exp:.6f}"
        )


def test_aggte_simple_matches_cs_overall(cs_fixture):
    simple = aggte(cs_fixture, type='simple',
                   n_boot=500, random_state=42)
    assert simple.estimate == pytest.approx(1.282166, abs=1e-4)
