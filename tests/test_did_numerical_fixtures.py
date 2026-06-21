"""Pinned numerical fixtures: regression guards against silent drift.

Every number in this module is the *exact* output of the current
implementation on a fixed-seed, fully-deterministic DGP (the helper
``_fixture_panel()`` below).  Any future refactor that unintentionally
changes a result will fail these tests and force a conscious decision
either to update the pinned value (with a commit explaining the
numerical change) or to revert the behavioural drift.

The fixtures are checked to 4 decimal places so bit-level floating-point
differences across BLAS backends do not cause spurious failures.

History note (CHANGELOG ``[1.13.1]``): the SEs in ``PINNED_ATT_GT``,
``PINNED_EVENT_STUDY``, and the overall-ATT SE in
``test_overall_att_matches_pinned`` were re-pinned in v1.13 to absorb
the simple-ATT influence-function scaling fix
(``Fix CS-DiD parity inference``).  Each group-time IF is now
multiplied by ``n_total / n_relevant`` when embedded in the full unit
universe, and the outcome-regression IF carries the
control-regression uncertainty term.  The point estimates are
unchanged at 4 decimals; the SEs grew by roughly the
``sqrt(n_total / n_relevant)`` factor, which is the corrected size.
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
            rows.append({"i": u, "t": t, "g": g, "y": ui + 0.2 * t + te + rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cs_fixture():
    df = _fixture_panel()
    return callaway_santanna(df, y="y", g="g", t="t", i="i", estimator="reg")


# --------------------------------------------------------------------------- #
# Pinned ATT(g, t) values                                                     #
# --------------------------------------------------------------------------- #

# (group, time) -> (att, se).  Generated from the current implementation.
PINNED_ATT_GT = {
    # Re-pinned 2026-05-05 after the v1.13 simple-ATT IF-scaling fix.
    # ATT point estimates unchanged; SEs grew by the
    # n_total/n_relevant correction.
    (3, 1): (-0.583666, 0.552161),
    (3, 3): (0.162054, 0.546096),
    (3, 4): (1.146706, 0.490623),
    (3, 5): (0.810629, 0.567576),
    (3, 6): (1.340394, 0.580295),
    (3, 7): (2.299304, 0.521175),
    (3, 8): (2.862484, 0.509427),
    (5, 1): (-0.392430, 0.477550),
    (5, 2): (0.346434, 0.471830),
    (5, 3): (-0.174793, 0.530724),
    (5, 5): (0.289302, 0.575977),
    (5, 6): (0.956634, 0.603753),
    (5, 7): (1.945725, 0.601489),
    (5, 8): (1.986254, 0.428450),
    (7, 1): (0.082153, 0.581038),
    (7, 2): (0.284830, 0.536620),
    (7, 3): (0.663454, 0.586537),
    (7, 4): (0.507629, 0.518836),
    (7, 5): (0.206995, 0.545912),
    (7, 7): (0.617812, 0.426407),
    (7, 8): (0.968689, 0.502857),
}


def test_att_gt_matches_pinned_values(cs_fixture):
    detail = cs_fixture.detail.set_index(["group", "time"])
    for (g, t), (att_exp, se_exp) in PINNED_ATT_GT.items():
        row = detail.loc[(g, t)]
        assert row["att"] == pytest.approx(att_exp, abs=1e-4), (
            f"ATT(g={g}, t={t}) drifted: " f"{row['att']:.6f} vs pinned {att_exp:.6f}"
        )
        assert row["se"] == pytest.approx(se_exp, abs=1e-4), (
            f"SE(g={g}, t={t}) drifted: " f"{row['se']:.6f} vs pinned {se_exp:.6f}"
        )


def test_overall_att_matches_pinned(cs_fixture):
    assert cs_fixture.estimate == pytest.approx(1.282166, abs=1e-4)
    # SE re-pinned to 0.289142 (was 0.101724 pre-v1.13) following the
    # simple-ATT IF-scaling fix; see module docstring.
    assert cs_fixture.se == pytest.approx(0.289142, abs=1e-4)


# --------------------------------------------------------------------------- #
# Pinned aggte(dynamic) values                                                #
# --------------------------------------------------------------------------- #

PINNED_EVENT_STUDY = {
    # Re-pinned 2026-05-05 after the v1.13 simple-ATT IF-scaling fix.
    -6: (0.082153, 0.602161),
    -5: (0.284830, 0.536783),
    -4: (0.135512, 0.419463),
    -3: (0.427031, 0.295390),
    -2: (-0.183822, 0.342370),
    0: (0.356390, 0.303102),
    1: (1.024010, 0.265289),
    2: (1.378177, 0.445609),
    3: (1.663324, 0.368396),
    4: (2.299304, 0.553597),
    5: (2.862484, 0.511114),
}


def test_aggte_dynamic_matches_pinned_values(cs_fixture):
    es = aggte(cs_fixture, type="dynamic", n_boot=500, random_state=42)
    got = es.detail.set_index("relative_time")
    for e, (att_exp, se_exp) in PINNED_EVENT_STUDY.items():
        row = got.loc[e]
        assert row["att"] == pytest.approx(att_exp, abs=1e-4), (
            f"dynamic ATT(e={e}) drifted: " f"{row['att']:.6f} vs pinned {att_exp:.6f}"
        )
        assert row["se"] == pytest.approx(se_exp, abs=1e-3), (
            f"dynamic SE(e={e}) drifted: " f"{row['se']:.6f} vs pinned {se_exp:.6f}"
        )


def test_aggte_simple_matches_cs_overall(cs_fixture):
    simple = aggte(cs_fixture, type="simple", n_boot=500, random_state=42)
    assert simple.estimate == pytest.approx(1.282166, abs=1e-4)
