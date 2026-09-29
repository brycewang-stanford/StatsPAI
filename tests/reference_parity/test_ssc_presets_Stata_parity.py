"""``sp.ssc(preset)`` reproduces Stata's small-sample conventions in ``sp.feols``.

``sp.feols`` (pyfixest backend) takes an ``ssc=`` dictionary; matching a
Stata table meant knowing which switches each command implies (Kinship,
QJE 2019; Web of Power, QJE 2023). The presets were read off Stata 18 on
``_fixtures/ssc_presets.csv`` (firm effects nested in industry clusters),
from ``_generate_ssc_presets_Stata.do``; each reproduces its command's SE of
``x`` to 1e-10.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "ssc_presets_Stata.json").read_text(encoding="utf-8"))

CASES = [
    ("regress_cl", "regress", "y ~ x + w", {"CRV1": "ind"}),
    ("regress_hc1", "regress", "y ~ x + w", "hetero"),
    ("areg_cl", "areg", "y ~ x + w | firm", {"CRV1": "ind"}),
    ("reghdfe_cl", "reghdfe", "y ~ x + w | firm", {"CRV1": "ind"}),
    ("xtreg_fe_cl", "xtreg", "y ~ x + w | firm", {"CRV1": "ind"}),
    ("ivregress_cl", "ivregress", "y ~ w | x ~ z", {"CRV1": "ind"}),
    ("ivregress_small_cl", "ivregress_small", "y ~ w | x ~ z", {"CRV1": "ind"}),
    ("ivregress_hc", "ivregress", "y ~ w | x ~ z", "hetero"),
]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "ssc_presets.csv")


@pytest.mark.parametrize("key, preset, fml, vcov", CASES, ids=[c[0] for c in CASES])
def test_preset_matches_stata(data, key, preset, fml, vcov):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.feols(fml, data, vcov=vcov, ssc=sp.ssc(preset))
    assert float(r.std_errors["x"]) == pytest.approx(R[key], rel=1e-10)


def test_overrides_and_unknown_preset():
    assert sp.ssc("reghdfe", G_adj=False)["G_adj"] is False
    assert sp.ssc("stata_areg") == sp.ssc("areg")
    with pytest.raises(sp.MethodIncompatibility, match="preset"):
        sp.ssc("spss")
