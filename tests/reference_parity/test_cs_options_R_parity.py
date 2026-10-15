"""``sp.callaway_santanna`` option values against R ``did`` 2.5.1, one at a time.

Each case below is an option value that had no same-byte reference of its
own in ``sp.validation_scope``'s Callaway-Sant'Anna map:

* ``estimator='drimp'`` (the improved doubly robust estimator) on a panel
  under both control groups and on repeated cross-sections, and
  ``estimator='ipw_abadie'`` (Abadie 2005, Stata ``csdid``'s ``method(ipw)``)
  on both. R ``did``'s ``att_gt`` accepts a function as ``est_method``, so the
  references are ``DRDID::drdid_imp_panel`` / ``drdid_imp_rc`` and
  ``DRDID::ipw_did_panel`` / ``ipw_did_rc`` run through the canonical
  ``att_gt`` + ``aggte`` pipeline. Until now ``drimp`` was compared only
  with Stata through ``sp.stata`` at 1e-5, and ``ipw_abadie`` only on point
  estimates.
* ``estimator='stdipw'``, which is ``'ipw'`` under Stata's name; both are
  compared with ``est_method = "ipw"``, with clustering on a panel and on
  repeated cross-sections.
* ``anticipation=1`` with weights and a covariate (never-treated, universal
  base) and with clustering (not-yet-treated, varying base), through all four
  aggregations; the earlier anticipation references compare cells and the
  simple aggregate only.

Every ATT(g, t), its standard error, the four overall aggregates and the
per-cell aggregates (event time, cohort, calendar period) with their
standard errors, at ``rtol = 1e-7``. Measured worst case: 2.1e-8, a drimp
cell standard error (R solves the tilting propensity score with
``trust::trust`` and stops at its default tolerance); everything else is
within 1e-8.

⚠️ The repeated cross-section cases found a silent error. On
``panel=False`` the estimators ``'stdipw'`` and ``'ipw_abadie'`` were missing
from the set routed to the DRDID ports and fell through to the cell-mean
outcome regression, so they returned the unconditional 2x2 means under an
IPW name: on this fixture the simple ATT was -0.0399513 (SE 0.0473928) for
both, against R's -0.0417771 (SE 0.1672262) for ``std_ipw_did_rc`` and
-0.0418352 (SE 0.7218281) for ``ipw_did_rc``. ``'ipw'`` itself was always
routed correctly.

Fixture ``_fixtures/cs_options_R.json`` from
``_fixtures/_generate_cs_options_R.R`` on ``did::mpdta``
(``tests/orig_parity/data/02_mpdta_original.csv``) with two columns derived
by integer arithmetic on ``countyreal``, the same on both sides: ``state``
(``countyreal // 1000``, 29 clusters) and a time-invariant weight ``w``
(``1 + (countyreal % 7) / 7``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_ROOT = Path(__file__).resolve().parents[2]
_FIX = Path(__file__).parent / "_fixtures"
_REF = json.loads((_FIX / "cs_options_R.json").read_text(encoding="utf-8"))
_CASES = _REF["cases"]
_RTOL = 1e-7

#: R est_method -> StatsPAI estimator names that must reproduce it.
_ESTIMATORS = {
    "drimp": ("drimp",),
    "drimp_rc": ("drimp",),
    "ipw_abadie": ("ipw_abadie",),
    "ipw_abadie_rc": ("ipw_abadie",),
    "ipw": ("ipw", "stdipw"),
    "dr": ("dr",),
    "reg": ("reg",),
}
_PARAMS = [(cid, est) for cid, c in _CASES.items() for est in _ESTIMATORS[c["est"]]]


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    df = pd.read_csv(_ROOT / "tests" / "orig_parity" / "data" / "02_mpdta_original.csv")
    df["state"] = df["countyreal"] // 1000
    df["w"] = 1 + (df["countyreal"] % 7) / 7
    return df


def _fit(df: pd.DataFrame, case: dict, estimator: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            df,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            x=["lpop"] if case["x"] else None,
            estimator=estimator,
            panel=case["panel"],
            control_group=case["control_group"],
            base_period=case["base_period"],
            anticipation=case.get("anticipation", 0),
            weights=case.get("weights"),
            clustervars=case.get("clustervars"),
        )


@pytest.mark.parametrize("cid, estimator", _PARAMS)
def test_cells_match_r(mpdta, cid, estimator):
    case = _CASES[cid]
    fit = _fit(mpdta, case, estimator)
    ref = pd.DataFrame({k: case[k] for k in ("group", "time", "att", "se")}).set_index(
        ["group", "time"]
    )
    got = fit.detail.set_index(["group", "time"])[["att", "se"]]
    # R reports the normalised reference cells (se = NA, dropped by the
    # generator); every cell either side estimates must be in the other.
    assert set(got.index) == set(ref.index), cid
    joined = got.join(ref, rsuffix="_r")
    np.testing.assert_allclose(joined["att"], joined["att_r"], rtol=_RTOL, atol=1e-12)
    np.testing.assert_allclose(joined["se"], joined["se_r"], rtol=_RTOL)
    assert sp.validation_scope(fit)["status"] == "covered", (cid, estimator)


@pytest.mark.parametrize("agg_type", ["simple", "dynamic", "group", "calendar"])
@pytest.mark.parametrize("cid, estimator", _PARAMS)
def test_aggregates_match_r(mpdta, cid, estimator, agg_type):
    case = _CASES[cid]
    ref = case["agg"][agg_type]
    agg = sp.aggte(_fit(mpdta, case, estimator), type=agg_type, bstrap=False)
    assert agg.estimate == pytest.approx(ref["att"], rel=_RTOL)
    assert agg.se == pytest.approx(ref["se"], rel=_RTOL)
    if agg_type == "simple":
        return
    key = {"dynamic": "relative_time", "group": "group", "calendar": "time"}[agg_type]
    got = agg.detail.set_index(key)[["att", "se"]]
    want = pd.DataFrame(
        {"att": ref["att_egt"], "se": ref["se_egt"]}, index=ref["egt"]
    ).dropna()
    want = want[want["se"] > 0]  # the normalised reference event time
    got = got.loc[want.index]
    np.testing.assert_allclose(got["att"], want["att"], rtol=_RTOL, atol=1e-12)
    np.testing.assert_allclose(got["se"], want["se"], rtol=_RTOL)


@pytest.mark.parametrize("cid, estimator", _PARAMS)
def test_headline_is_the_simple_aggregate(mpdta, cid, estimator):
    case = _CASES[cid]
    fit = _fit(mpdta, case, estimator)
    assert fit.estimate == pytest.approx(case["agg"]["simple"]["att"], rel=_RTOL)
    assert fit.se == pytest.approx(case["agg"]["simple"]["se"], rel=_RTOL)


def test_stdipw_is_ipw_bit_for_bit(mpdta):
    case = _CASES["ipw_cluster"]
    a, b = _fit(mpdta, case, "ipw"), _fit(mpdta, case, "stdipw")
    pd.testing.assert_frame_equal(a.detail, b.detail)
    assert sp.validation_scope(b)["configuration"]["estimator"] == "ipw"


def test_drimp_is_not_dr_with_a_covariate(mpdta):
    """Not vacuous: with ``lpop`` the improved and the traditional doubly
    robust estimators differ, so the drimp cases test their own path."""
    case = _CASES["drimp_nev_universal"]
    imp, dr = _fit(mpdta, case, "drimp"), _fit(mpdta, case, "dr")
    assert (
        np.max(np.abs(imp.detail["att"].to_numpy() - dr.detail["att"].to_numpy()))
        > 1e-5
    )
