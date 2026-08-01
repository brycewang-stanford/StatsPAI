"""Reference check: reverting treatment, and what it does to cohort DiD.

``sp.check_absorbing`` guards a failure that is otherwise silent. Cohort-based
DiD estimators represent treatment by the period a unit is *first* treated,
which is lossless only when treatment is absorbing. Under reversal they treat
post-reversal periods as still-treated and collapse toward zero — and they
cannot warn, because ``(y, g, t, i)`` does not contain the reversal.

This module pins the consequence against an external reference. On the
committed reverting panel (150 units x 10 periods, a third switching on at
t=4 and off again at t=7, true ATT 1.5):

===========================  =========
estimator                    estimate
===========================  =========
R ``fect`` 2.4.1             1.405412
``sp.did_multiplegt``        ~1.43
``sp.lp_did``                ~1.43
``sp.callaway_santanna``     ~0.71
===========================  =========

The reversal-capable estimators cluster near the truth; the cohort estimator
is off by more than half. ``sp.check_absorbing`` is what tells a caller which
group they are in.

Fixture
-------
``_fixtures/reverting_panel.csv``. The R reference was produced with::

    out <- fect(y ~ d, data = d, index = c("i", "t"), method = "fe",
                se = TRUE, nboots = 200, parallel = FALSE, force = "two-way")
    out$att.avg   # 1.4054124586

References
----------
- de Chaisemartin, C. and D'Haultfœuille, X. (2020). "Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects." *American Economic
  Review*, 110(9), 2964-2996. [@dechaisemartin2020two]
- Liu, L., Wang, Y. and Xu, Y. (2024). *AJPS*, 68(1), 160-176.
  [@liu2024practical]
"""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "reverting_panel.csv"

TRUE_ATT = 1.5
R_FECT_ATT = 1.4054124586


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


def test_check_absorbing_flags_the_panel(panel):
    """Known truth: 50 of 150 units revert, first at t=7."""
    chk = sp.check_absorbing(panel, unit="i", time="t", treat="d")
    assert chk.is_absorbing is False
    assert chk.n_reverting_units == 50
    assert chk.n_units == 150
    assert chk.first_reversal_period == 7


def test_reversal_capable_estimators_track_the_r_reference(panel):
    """dCDH and LP-DiD land near R ``fect`` and near the true ATT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcdh = sp.did_multiplegt(
            panel, y="y", group="i", time="t", treatment="d"
        ).estimate
        lp = sp.lp_did(panel, y="y", unit="i", time="t", treatment="d").estimate

    for name, est in (("did_multiplegt", dcdh), ("lp_did", lp)):
        assert abs(est - R_FECT_ATT) < 0.15, (
            f"{name} = {est:.6f} drifted from the R fect reference " f"{R_FECT_ATT:.6f}"
        )
        assert abs(est - TRUE_ATT) < 0.2, f"{name} = {est:.6f} vs truth 1.5"


def test_dcdh_and_lp_did_coincide_at_h0_by_construction(panel):
    """Not a bug: LP-DiD's h=0 contrast *is* dCDH's switcher-vs-stayer cell.

    With not-yet-treated clean controls the two point estimates are the same
    object, so they agree to machine precision. Their standard errors do not,
    because the inference procedures differ. Pinned so the identity is not
    mistaken for a wiring bug and "fixed".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.did_multiplegt(panel, y="y", group="i", time="t", treatment="d")
        b = sp.lp_did(panel, y="y", unit="i", time="t", treatment="d")

    assert b.estimand == "ATT at event-time h=0"
    assert a.estimate == pytest.approx(b.estimate, abs=1e-9)
    assert a.se != pytest.approx(b.se, abs=1e-9)


def test_cohort_estimator_collapses_toward_zero(panel):
    """The silent failure the guard exists for."""
    first = panel[panel.d == 1].groupby("i")["t"].min().rename("g")
    df = panel.merge(first, on="i", how="left")
    df["g"] = df["g"].fillna(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(df, y="y", g="g", t="t", i="i").estimate

    assert cs < 0.9, f"expected collapse toward zero, got {cs:.6f}"
    assert abs(cs - TRUE_ATT) > abs(R_FECT_ATT - TRUE_ATT), (
        "the cohort estimator is no longer worse than the reversal-capable "
        "reference; if that changed, revisit sp.recommend's routing"
    )
