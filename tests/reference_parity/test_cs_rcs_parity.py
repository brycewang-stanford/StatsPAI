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

Reference generation (R 4.5.2, did 2.3.0), on the package's own ``mpdta`` so
the never-treated group survives (see the integer-column trap noted in
``test_aggte_mpdta_parity.py``)::

    a <- att_gt(yname="lemp", tname="year", idname="countyreal",
                gname="first.treat", data=mpdta, control_group=<cg>,
                panel=FALSE, est_method=<est>, xformla=~lpop,
                bstrap=FALSE, cband=FALSE)
    aggte(a, type="simple", bstrap=FALSE, cband=FALSE)

Point estimates are pinned tightly; standard errors carry a ~0.15% gap from
small differences in how the aggregation denominator is formed, so they use a
1% relative tolerance.

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

# (est_method, control_group) -> (simple ATT, SE) from R did 2.3.0
R_RCS = {
    ("dr", "nevertreated"): (-0.0417517721, 0.0460680525),
    ("dr", "notyettreated"): (-0.0413516293, 0.0474594680),
    ("ipw", "nevertreated"): (-0.0417770822, 0.1672310175),
    ("ipw", "notyettreated"): (-0.0413894098, 0.1710682403),
    ("reg", "nevertreated"): (-0.0419686124, 0.1497787162),
    ("reg", "notyettreated"): (-0.0413747698, 0.1502463356),
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
        se_r, rel=0.01
    ), f"{estimator}/{control_group}: SE {agg.se:.8f} vs R {se_r:.8f}"


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


def test_clustervars_still_raises_under_rcs(mpdta):
    """Observations are not nested in units here — fail loudly, not silently."""
    from statspai.did.callaway_santanna import CallawayNotImplemented

    with pytest.raises(CallawayNotImplemented, match="clustervars"):
        _rcs_fit(mpdta, "dr", "nevertreated", clustervars="countyreal")
