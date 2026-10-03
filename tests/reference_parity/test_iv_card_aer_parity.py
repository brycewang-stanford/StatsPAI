"""``sp.iv`` 2SLS on the original Card (1995) extract vs R ``AER::ivreg``.

Track A module 02 pins 2SLS with HC1 errors on a calibrated replica. The
JSS Card listing, however, prints ``sp.iv``'s *default* -- the classical
2SLS covariance -- and ``sp.validation_scope`` reported that configuration
as not covered by any deterministic reference. This module closes it on the
original ``wooldridge::card`` bytes: classical, HC1 and CR1 standard errors,
just-identified (``nearc4``) and over-identified (``nearc4 + nearc2``), and
the Sargan statistic of the over-identified fit.

Fixture: ``_fixtures/iv_card_R.json`` from ``_generate_iv_card_R.R``
(AER 1.2.16, sandwich 3.1.1). Observed agreement: <= 8e-11 relative.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "iv_card_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "iv_card.csv")
X = "exper + expersq + black + south + smsa"
INSTRUMENTS = {"just": "nearc4", "over": "nearc4 + nearc2"}
RTOL = 1e-8


def _fit(key, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.iv(f"lwage ~ {X} + (educ ~ {INSTRUMENTS[key]})", data=DATA, **kw)


@pytest.mark.parametrize("key", sorted(INSTRUMENTS))
@pytest.mark.parametrize(
    "kw,field",
    [
        ({}, "se_classical"),
        ({"robust": "hc1"}, "se_hc1"),
        ({"cluster": "cl"}, "se_cr1"),
    ],
)
def test_2sls_coefficient_and_se_match_aer(key, kw, field):
    fit = _fit(key, **kw)
    ref = REF[key]
    assert float(fit.params["educ"]) == pytest.approx(ref["beta"], rel=RTOL)
    assert float(fit.std_errors["educ"]) == pytest.approx(ref[field], rel=RTOL)


def test_sargan_matches_aer():
    fit = _fit("over")
    assert fit.diagnostics["Sargan statistic"] == pytest.approx(
        REF["over"]["sargan"], rel=RTOL
    )


def test_scope_map_now_covers_the_card_listing_configuration():
    scope = sp.validation_scope(_fit("just"))
    assert scope["configuration"] == {
        "estimator": "2sls",
        "vce": "classical",
        "identification": "just",
        "absorb": "none",
        "small": "true",
    }
    assert scope["status"] == "covered"
