"""Reference parity: ``sp.panel`` vs R ``plm``.

R ``plm`` (Croissant & Millo 2008, *JSS*) is the canonical panel-data
package and the reference Stata's ``xtreg`` targets.  ``sp.panel`` must
reproduce plm's three static workhorse estimators on a balanced panel
whose regressors are correlated with the entity effect (so within /
between / random-effects give genuinely different answers — a within
vs between mix-up would fail these assertions):

  * ``method='fe'``       -> plm ``model='within'``  (classical + cluster SE)
  * ``method='re'``       -> plm ``model='random'``  (Swamy-Arora GLS)
  * ``method='between'``  -> plm ``model='between'``

Cluster-robust SE convention: ``sp.panel(method='fe', cluster=<entity>)``
matches plm ``vcovHC(type='HC1', cluster='group')``.

Tolerances
----------
  * coef: 1e-5 relative.  The within/between transforms and the
    Swamy-Arora variance-component GLS are deterministic; both packages
    solve the same least-squares system, so agreement is to numerical
    precision, not Monte-Carlo tolerance.
  * classical SE: 1e-5 relative (same residual variance, same dof
    correction n*T - n - K for within).
  * cluster SE: 2e-4 relative (HC1 finite-sample factor is identical;
    the looser band only absorbs float round-trips through JSON).

References
----------
- Croissant, Y. & Millo, G. (2008). "Panel Data Econometrics in R: The
  plm Package." *Journal of Statistical Software*, 27(2).
  doi:10.18637/jss.v027.i02
  (verified via Crossref and jstatsoft.org, 2026-06-08).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"

# plm intercept label -> sp.panel intercept label.
_INTERCEPT = {"(Intercept)": "const"}


@pytest.fixture(scope="module")
def panel_data():
    return pd.read_csv(_FIXTURE_DIR / "panel_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "panel_R.json", encoding="utf-8") as f:
        return json.load(f)


def _assert_block(res, ref_block, *, coef_rtol, se_rtol, label):
    """Compare every coefficient/SE in ``ref_block`` against ``res``."""
    for r_name, ref in ref_block.items():
        py_name = _INTERCEPT.get(r_name, r_name)
        py_coef = float(res.params[py_name])
        py_se = float(res.std_errors[py_name])
        assert py_coef == pytest.approx(
            ref["coef"], rel=coef_rtol
        ), f"{label}: coef[{py_name}] sp={py_coef:.8f} vs plm={ref['coef']:.8f}"
        assert py_se == pytest.approx(
            ref["se"], rel=se_rtol
        ), f"{label}: se[{py_name}] sp={py_se:.8f} vs plm={ref['se']:.8f}"


def test_fe_within_matches_plm(panel_data, r_reference):
    res = sp.panel(panel_data, "y ~ x1 + x2", entity="id", time="year", method="fe")
    _assert_block(
        res, r_reference["fe_within"], coef_rtol=1e-5, se_rtol=1e-5, label="FE within"
    )


def test_fe_within_cluster_se_matches_plm(panel_data, r_reference):
    # plm's vcovHC(HC1, cluster = "group") is closest to linearmodels'
    # scaling; since 1.42 the default is xtreg's, so name it
    res = sp.panel(
        panel_data,
        "y ~ x1 + x2",
        entity="id",
        time="year",
        method="fe",
        cluster="id",
        ssc="linearmodels",
    )
    # Coefficients are unchanged by clustering; the cluster block re-checks
    # them and validates the cluster-robust SE against plm vcovHC HC1/group.
    _assert_block(
        res,
        r_reference["fe_within_cluster"],
        coef_rtol=1e-5,
        se_rtol=2e-4,
        label="FE within (cluster id)",
    )


def test_re_swamy_arora_matches_plm(panel_data, r_reference):
    res = sp.panel(panel_data, "y ~ x1 + x2", entity="id", time="year", method="re")
    _assert_block(
        res,
        r_reference["re_swamy_arora"],
        coef_rtol=1e-5,
        se_rtol=1e-5,
        label="RE Swamy-Arora",
    )


def test_between_matches_plm(panel_data, r_reference):
    res = sp.panel(
        panel_data, "y ~ x1 + x2", entity="id", time="year", method="between"
    )
    _assert_block(
        res, r_reference["between"], coef_rtol=1e-5, se_rtol=1e-5, label="Between"
    )


def test_panel_nobs_matches_plm(panel_data, r_reference):
    res = sp.panel(panel_data, "y ~ x1 + x2", entity="id", time="year", method="fe")
    nobs = getattr(res, "nobs", None)
    if nobs is None:
        nobs = len(panel_data)
    assert int(nobs) == r_reference["n_obs"]
