"""``sp.gardner_did(vce='stage2')`` is the convention it documents.

``vce='stage2'`` is the second-stage-only clustered SE: the number a user
gets by running Gardner's two stages by hand and reading the second
regression's clustered SE. It ignores the estimation of the first-stage
fixed effects and understates (the corrected two-stage variance is the
default ``vce='analytic'``, pinned against R and Stata ``did2s`` by Track A
module 73). It is kept to reproduce earlier output, and this test pins that
it reproduces what it says it is:

* static: ``lm(y_tilde ~ D)`` with ``sandwich::vcovCL(type = "HC1",
  cadjust = TRUE)`` on the county, the ``G/(G-1) (n-1)/(n-k)`` factor of
  Stata's ``regress, vce(cluster)``;
* event study: per relative time, the clustered SE of the bin mean with no
  small-sample factor, ``lm(y_tilde ~ 0 + factor(rel))`` with
  ``sandwich::vcovCL(type = "HC0", cadjust = FALSE)``.

The event-study headline SE under ``vce='stage2'`` treats the horizons as
independent and has no reference; it is not compared.

Fixture: ``_fixtures/gardner_stage2_R.json`` from
``_fixtures/_generate_gardner_stage2_R.R`` (R 4.5.2, sandwich 3.1.1), on
module 73's CSV bytes, with the first stage fitted by dense ``lm``.
Tolerance: relative 1e-9; observed 4.5e-11 (static), 3.3e-10 (event study).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = json.loads(
    (pathlib.Path(__file__).parent / "_fixtures" / "gardner_stage2_R.json").read_text(
        encoding="utf-8"
    )
)
DATA = pd.read_csv(_ROOT / "tests" / "r_parity" / "data" / "73_did2s.csv")
KEYS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
RTOL = 1e-9


def _fit(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.gardner_did(DATA, **KEYS, vce="stage2", **kw)


def test_static_stage2_se_matches_the_hand_run_second_stage():
    res = _fit()
    assert float(res.estimate) == pytest.approx(REF["static"]["att"], rel=RTOL)
    assert float(res.se) == pytest.approx(REF["static"]["se"], rel=RTOL)


def test_event_study_stage2_ses_match_the_bin_means():
    res = _fit(event_study=True)
    es = res.model_info["event_study"]
    want = {int(k): v for k, v in REF["event_study"].items()}
    got = {
        int(name.replace("D_k", "")): (es["coef"][name], es["se"][name])
        for name in es["coef"]
    }
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k][0] == pytest.approx(v["att"], rel=RTOL), k
        assert got[k][1] == pytest.approx(v["se"], rel=RTOL), k


def test_stage2_understates_the_corrected_variance():
    """The legacy SE is smaller than the did2s two-stage SE, as documented."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected = sp.gardner_did(DATA, **KEYS)
    assert float(_fit().se) < 0.9 * float(corrected.se)
