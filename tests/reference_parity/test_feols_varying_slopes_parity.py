"""``sp.feols`` varying-slope fixed effects vs R ``fixest`` (review §4.4).

pyfixest parses fixest's ``f[x]`` / ``f[[x]]`` syntax but does not absorb the
slope, so until 1.32 ``sp.feols`` refused it. It now routes to StatsPAI's own
HDFE kernel (``sp.hdfe_ols``), translating ``f[x]`` to ``f + i.f#c.x`` and
``f[[x]]`` to ``i.f#c.x``. One convention differs between the references:
with only slopes absorbed, R fixest keeps (and reports) an intercept while
Stata ``reghdfe absorb(i.f#c.x)`` does not; ``sp.feols`` follows fixest,
``sp.hdfe_ols`` follows reghdfe.

Reference: ``_fixtures/feols_varying_slopes_R.json`` from
``_generate_feols_varying_slopes_R.R`` (R 4.5.2, fixest 0.14.0): six
formulas x {iid, hetero, cluster, weighted cluster}. Coefficients and SEs of
every reported regressor agree to <= 1.6e-9 (observed; <= 5e-15 for a single
absorbed dimension -- the rest is fixest's own demeaning tolerance with two
FE dimensions); asserted at 1e-8.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "feols_varying_slopes_R.json").read_text(encoding="utf-8"))
R.pop("provenance", None)
VCOV = {
    "iid": dict(vcov="iid"),
    "hetero": dict(vcov="hetero"),
    "cl": dict(cluster="cl"),
    "w_cl": dict(cluster="cl", weights="w"),
}


def _L(v):
    return v if isinstance(v, list) else [v]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "feols_varying_slopes_data.csv")


@pytest.mark.parametrize("key", list(R))
def test_matches_fixest(data, key):
    fml, v = key.rsplit(" ", 1)
    ref = R[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.feols(fml, data, **VCOV[v])
    names = [n.replace("(Intercept)", "Intercept") for n in _L(ref["names"])]
    got = np.r_[res.params[names].to_numpy(), res.std_errors[names].to_numpy()]
    want = np.r_[_L(ref["coef"]), _L(ref["se"])]
    np.testing.assert_allclose(got, want, rtol=1e-8)
    assert res.model_info["backend"] == "statspai-native"


def test_translation():
    from statspai.fixest.wrapper import _translate_varying_slopes as tr

    assert tr("y ~ x | g[z]") == ("y ~ x | g + i.g#c.z", False)
    assert tr("y ~ x | g[[z]]") == ("y ~ x | i.g#c.z", True)
    assert tr("y ~ x | h + g[z, w]") == ("y ~ x | h + g + i.g#c.z + i.g#c.w", False)
    assert tr("y ~ x | h + g[[z]]") == ("y ~ x | h + i.g#c.z", False)


def test_unsupported_vcov_is_refused(data):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="varying slopes"):
        sp.feols("y ~ x | g[z]", data, vcov="CRV3")
