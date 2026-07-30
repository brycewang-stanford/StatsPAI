"""Reference parity on canonical ``did::mpdta``: Sun-Abraham and Gardner.

Closes two coverage rows that previously had no matched-option runner against
their R reference implementations:

* ``sp.sun_abraham``  vs  ``fixest::sunab`` (fixest 0.14.0)
* ``sp.gardner_did``  vs  ``did2s::did2s``  (did2s 1.2.1)

Two convention notes, both verified rather than assumed:

1. **Sun-Abraham overall aggregation.** ``sp.sun_abraham`` defaults to
   ``aggregation='event_time'``, which equal-weights the post-treatment
   relative-time IW effects. ``fixest::summary(..., agg='att')`` instead
   weights every post cohort-time cell by its treated cohort size. Both are
   legitimate summaries of the same event study; they differ substantially on
   unbalanced panels (−0.0772 vs −0.0400 here, because e=0 carries 191 treated
   observations while e=2 and e=3 carry 20 each). ``aggregation='fixest_att'``
   selects the R convention and is what this module pins. The *event-study
   vector itself* is convention-free and is pinned coefficient by coefficient.

2. **Gardner two-stage standard errors.** The point estimate matches R to
   ~1e-8. The default ``vce='analytic'`` SE clusters the stage-2 residuals and
   ignores the variance from estimating the stage-1 fixed effects, so it comes
   in ~18% below R's (which propagates both stages); ``vce='bootstrap'``
   recovers it to within ~3%. ``sp.gardner_did`` already warns about this, and
   the bands below pin the known gap so a drift in either direction is caught.

Data provenance
---------------
``tests/orig_parity/data/02_mpdta_original.csv``, SHA256
``1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661`` —
verified byte-identical to a rebuild of ``did::mpdta`` from the R package.

References
----------
- Sun, L. and Abraham, S. (2021). "Estimating dynamic treatment effects in
  event studies with heterogeneous treatment effects." *Journal of
  Econometrics*, 225(2), 175-199. [@sun2021estimating]
- Gardner, J. (2022). "Two-stage differences in differences."
  [@gardner2022stage]
"""

from __future__ import annotations

import hashlib
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)
_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

# ---------------------------------------------------------------------------
# R reference values, generated on the locked CSV above.
#   fixest 0.14.0:
#     feols(lemp ~ sunab(g_sa, year) | countyreal + year, cluster = ~countyreal)
#     where g_sa recodes never-treated (first_treat == 0) to 10000
#   did2s 1.2.1:
#     did2s(yname="lemp", first_stage = ~0 | countyreal + year,
#           second_stage = ~i(dpost, ref=FALSE), treatment="dpost",
#           cluster_var="countyreal")
# ---------------------------------------------------------------------------
R_SUNAB_EVENT = {
    -4: (0.00330636, 0.02455510),
    -3: (0.02502183, 0.01815434),
    -2: (0.02445874, 0.01426679),
    0: (-0.01993182, 0.01185754),
    1: (-0.05095737, 0.01687068),
    2: (-0.13725874, 0.03658948),
    3: (-0.10081136, 0.03450427),
}
R_SUNAB_AGG_ATT = (-0.0399512752, 0.0117962774)
R_DID2S_STATIC = (-0.0477099151, 0.0134784088)


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    if not _MPDTA.exists():  # pragma: no cover - fixture shipped with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert digest == _MPDTA_SHA256, (
        "the mpdta fixture changed — the pinned R reference numbers in this "
        f"module were locked against {_MPDTA_SHA256}, got {digest}"
    )
    return pd.read_csv(_MPDTA)


# ===========================================================================
# Sun & Abraham vs fixest::sunab
# ===========================================================================


def test_sunab_event_study_matches_fixest(mpdta):
    """Every IW event-study coefficient matches fixest::sunab."""
    res = sp.sun_abraham(mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    got = {int(e): a for e, a in zip(res.detail["relative_time"], res.detail["att"])}

    assert set(got) == set(R_SUNAB_EVENT), (
        f"event-time grid drifted: got {sorted(got)}, "
        f"expected {sorted(R_SUNAB_EVENT)}"
    )
    for e, (att_r, _) in R_SUNAB_EVENT.items():
        assert got[e] == pytest.approx(
            att_r, abs=1e-6
        ), f"e={e}: StatsPAI {got[e]:.8f} vs fixest {att_r:.8f}"


def test_sunab_event_study_ses_match_fixest(mpdta):
    """Cluster-robust SEs on the event study agree to 1%."""
    res = sp.sun_abraham(mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    got = {int(e): s for e, s in zip(res.detail["relative_time"], res.detail["se"])}
    for e, (_, se_r) in R_SUNAB_EVENT.items():
        assert got[e] == pytest.approx(
            se_r, rel=0.01
        ), f"e={e}: SE {got[e]:.8f} vs fixest {se_r:.8f}"


def test_sunab_fixest_att_aggregation_matches_r(mpdta):
    """``aggregation='fixest_att'`` reproduces fixest's agg='att' summary."""
    res = sp.sun_abraham(
        mpdta,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        aggregation="fixest_att",
    )
    att_r, se_r = R_SUNAB_AGG_ATT
    assert res.estimate == pytest.approx(att_r, abs=1e-8)
    assert res.se == pytest.approx(se_r, rel=0.01)


def test_sunab_default_aggregation_is_event_time_not_fixest(mpdta):
    """Pin the documented default so the convention gap stays visible.

    The default equal-weights event times; fixest weights by treated cohort
    size.  On this unbalanced panel that is a ~2x difference, which is a real
    migration footgun worth keeping under test rather than discovering in a
    replication.
    """
    default = sp.sun_abraham(mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    post = default.detail.loc[default.detail["relative_time"] >= 0, "att"]

    assert default.estimate == pytest.approx(float(post.mean()), rel=1e-12)
    assert default.model_info["summary_aggregation"] == "event_time"
    # Both conventions are carried on the result so either can be reported.
    assert default.model_info["att_fixest_att"] == pytest.approx(
        R_SUNAB_AGG_ATT[0], abs=1e-8
    )


# ===========================================================================
# Gardner two-stage vs did2s
# ===========================================================================


def test_gardner_point_estimate_matches_did2s(mpdta):
    """Static two-stage ATT matches R did2s to ~1e-8."""
    res = sp.gardner_did(
        mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
    )
    assert res.estimate == pytest.approx(R_DID2S_STATIC[0], abs=1e-7)


def test_gardner_analytic_se_understates_and_bootstrap_recovers(mpdta):
    """Pin the documented two-stage SE convention gap.

    ``vce='analytic'`` ignores stage-1 estimation error and lands ~18% below
    R's; ``vce='bootstrap'`` propagates both stages and recovers it.  Bands are
    deliberately loose enough for bootstrap noise but tight enough to catch a
    regression in either direction.
    """
    se_r = R_DID2S_STATIC[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analytic = sp.gardner_did(
            mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
        )
        boot = sp.gardner_did(
            mpdta,
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
            vce="bootstrap",
        )

    assert 0.75 < analytic.se / se_r < 0.90, (
        f"analytic/R SE ratio {analytic.se / se_r:.4f} left the known band — "
        "the two-stage variance convention changed"
    )
    assert 0.90 < boot.se / se_r < 1.10, (
        f"bootstrap/R SE ratio {boot.se / se_r:.4f}: the bootstrap should "
        "recover R's two-stage SE"
    )
    assert boot.estimate == pytest.approx(analytic.estimate, rel=1e-12)
