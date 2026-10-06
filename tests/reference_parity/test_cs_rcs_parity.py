"""Reference parity: ``sp.callaway_santanna(panel=False)`` vs R ``did``.

Repeated cross-sections. Until now ``panel=False`` accepted only
``estimator='reg'``, forced ``control_group='nevertreated'``, and refused
``bstrap``, which left CPS/ACS/DHS-style data with no usable estimator.

The (g, t) loop now hands each two-period sub-sample to the matching
Sant'Anna-Zhao estimator in :mod:`statspai.did._rcs`, which is what R
``did::att_gt(panel = FALSE)`` does (confirmed from ``did:::compute.att_gt``):

    est_method = "dr"   ->  DRDID::drdid_rc
    est_method = "ipw"  ->  DRDID::std_ipw_did_rc
    est_method = "reg"  ->  DRDID::reg_did_rc

Reference generation (R 4.5.2, did 2.5.1, DRDID 1.3.0), on the locked
``mpdta`` CSV::

    a <- att_gt(yname="lemp", tname="year", idname="countyreal",
                gname="first.treat", data=mpdta, control_group=<cg>,
                panel=FALSE, est_method=<est>, xformla=~lpop,
                bstrap=FALSE, cband=FALSE)
    aggte(a, type="simple", bstrap=FALSE, cband=FALSE)

Point estimates and standard errors are both pinned at 1e-8.

``mpdta`` is a panel, so under ``panel=False`` the county id repeats across
years. On such data did 2.3.0 reported aggregate standard errors 0.003% to
0.08% away from the values below while every ATT(g, t) cell and its standard
error agreed. These tests carried a 1% tolerance for that. did 2.5.0 reworked
the aggregation of the influence function on the observation-level paths, and
did 2.5.1 returns what StatsPAI already computed, to 1e-9 or better.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
- Sant'Anna, P.H.C. and Zhao, J. (2020). "Doubly robust
  difference-in-differences estimators." *Journal of Econometrics*, 219(1),
  101-122. [@santanna2020doubly]
"""

from __future__ import annotations

import hashlib
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)
_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

# (est_method, control_group) -> (simple ATT, SE) from R did 2.5.1
R_RCS = {
    ("dr", "nevertreated"): (-0.041751772061081, 0.0460317504889275),
    ("dr", "notyettreated"): (-0.0413516292999427, 0.0474329482129552),
    ("ipw", "nevertreated"): (-0.0417770821895978, 0.167226232158267),
    ("ipw", "notyettreated"): (-0.0413894097517419, 0.171058030727509),
    ("reg", "nevertreated"): (-0.041968612421581, 0.149762154610337),
    ("reg", "notyettreated"): (-0.0413747697923552, 0.150232773334959),
}


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    if not _MPDTA.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert (
        digest == _MPDTA_SHA256
    ), f"mpdta fixture changed; expected {_MPDTA_SHA256}, got {digest}"
    return pd.read_csv(_MPDTA)


def _rcs_fit(df: pd.DataFrame, estimator: str, control_group: str, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            df,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            panel=False,
            base_period="varying",
            estimator=estimator,
            control_group=control_group,
            **kw,
        )


@pytest.mark.parametrize("key", sorted(R_RCS))
def test_rcs_simple_att_matches_r_did(mpdta, key):
    estimator, control_group = key
    att_r, _ = R_RCS[key]
    fit = _rcs_fit(mpdta, estimator, control_group, x=["lpop"])
    agg = sp.aggte(fit, type="simple", bstrap=False)
    assert agg.estimate == pytest.approx(att_r, abs=1e-8), (
        f"{estimator}/{control_group}: StatsPAI {agg.estimate:.10f} "
        f"vs R {att_r:.10f}"
    )


@pytest.mark.parametrize("key", sorted(R_RCS))
def test_rcs_simple_se_matches_r_did(mpdta, key):
    estimator, control_group = key
    _, se_r = R_RCS[key]
    fit = _rcs_fit(mpdta, estimator, control_group, x=["lpop"])
    agg = sp.aggte(fit, type="simple", bstrap=False)
    assert agg.se == pytest.approx(
        se_r, rel=1e-8
    ), f"{estimator}/{control_group}: SE {agg.se:.12f} vs R {se_r:.12f}"


@pytest.mark.parametrize("estimator", ["dr", "ipw", "reg"])
def test_rcs_estimators_are_now_accepted(mpdta, estimator):
    """All three est_method values must run; previously only 'reg' did."""
    fit = _rcs_fit(mpdta, estimator, "nevertreated", x=["lpop"])
    assert fit.model_info["panel"] is False
    assert "RCS" in fit.model_info["estimator"]


def test_rcs_not_yet_treated_is_now_accepted(mpdta):
    """control_group='notyettreated' used to raise under panel=False."""
    fit = _rcs_fit(mpdta, "dr", "notyettreated", x=["lpop"])
    assert fit.model_info["control_group"] == "notyettreated"


def test_rcs_bootstrap_is_now_accepted(mpdta):
    """bstrap used to raise under panel=False; it must now run."""
    fit = _rcs_fit(
        mpdta,
        "dr",
        "nevertreated",
        x=["lpop"],
        bstrap=True,
        biters=200,
        random_state=0,
    )
    assert fit.se > 0


def test_rcs_influence_functions_feed_aggte_bootstrap(mpdta):
    """The cell influence functions must support the multiplier bootstrap.

    This is the point of carrying them: ``aggte`` resamples them to get
    uniform bands, so a per-cell SE alone would not be enough.
    """
    fit = _rcs_fit(mpdta, "dr", "nevertreated", x=["lpop"])
    boot = sp.aggte(fit, type="simple", bstrap=True, n_boot=300, random_state=1)
    analytic = sp.aggte(fit, type="simple", bstrap=False)
    assert boot.se == pytest.approx(analytic.se, rel=0.35)

    event = sp.aggte(fit, type="dynamic", bstrap=False)
    assert len(event.detail) > 1


def test_unconditional_reg_rcs_matches_the_panel_simple_att(mpdta):
    """Sanity anchor: with no covariates the RCS reg estimand coincides.

    Unconditional cell-mean DiD on repeated cross-sections and the panel
    CS simple ATT target the same quantity on this balanced panel, so a large
    gap would mean the (g, t) cell construction is wrong.
    """
    fit = _rcs_fit(mpdta, "reg", "nevertreated")
    agg = sp.aggte(fit, type="simple", bstrap=False)
    assert agg.estimate == pytest.approx(-0.0399512752, abs=1e-6)


def test_clustervars_under_rcs_give_clustered_analytic_ses(mpdta):
    """RCS clusters with the analytic SEs too (Stata csdid, cluster())."""
    r = _rcs_fit(
        mpdta,
        "dr",
        "nevertreated",
        clustervars=["countyreal", "first_treat"],
    )
    plain = _rcs_fit(mpdta, "dr", "nevertreated")
    assert r.model_info["se_method"] == "analytic"
    np.testing.assert_allclose(r.detail["att"], plain.detail["att"])
    assert not np.allclose(r.detail["se"], plain.detail["se"])
