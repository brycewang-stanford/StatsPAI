"""``sp.ltmle(design=)`` vs R ``ltmle`` 1.3-0 + ``survey::svyrecvar``.

Fixture: ``_fixtures/ltmle_design_R.json`` from
``_fixtures/_generate_ltmle_design_R.R``. The data are the wide fixtures
of ``test_ltmle_weights_parity.py`` with the same observation weights and
a design added on both sides (10 strata of 100 rows, PSUs of 20 rows,
fpc = 40 PSUs per stratum). The reference ATE influence curve is the
difference of the ``IC$tmle`` of two single-regime ltmle fits (its
sd / sqrt(n) reproduces ltmle's own ATE std.dev, stored as a check), and
the design SE is ``svyrecvar(IC / n)``. ltmle and survey are GPL: used as
black boxes (documentation and outputs only).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "ltmle_design_R.json").read_text(encoding="utf-8"))
DESIGNS = {
    "full": dict(strata="stratum", cluster="psu", fpc="fpc"),
    "nofpc": dict(strata="stratum", cluster="psu"),
    "psu_only": dict(cluster="psu"),
}


def _data(cens: bool) -> pd.DataFrame:
    d = pd.read_csv(_FIX / ("teffects_wide_cens.csv" if cens else "teffects_wide.csv"))
    i = np.arange(1, len(d) + 1)
    d["sw"] = 0.4 + (i % 5) * 0.3
    d["stratum"] = (i - 1) // 100 + 1
    d["psu"] = (i - 1) // 20 + 1
    d["fpc"] = 40
    return d


def _fit(d: pd.DataFrame, cens: bool, y: str, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ltmle(
            d,
            y=y,
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            baseline=["v"],
            censoring=["C0", "C1"] if cens else None,
            **kw,
        )


@pytest.mark.parametrize("cens", [False, True])
@pytest.mark.parametrize("y", ["Yb", "Y"])
@pytest.mark.parametrize("name", list(DESIGNS))
def test_ltmle_design_se_matches_svyrecvar(cens, y, name):
    ref = REF[("cens_" if cens else "nocens_") + y]
    # the reference IC is ltmle's own: its sd reproduces ltmle's ATE SE
    assert abs(ref["ic_se_check"] - ref["ate_se_ic"]) < 1e-9 * ref["ate_se_ic"]
    d = _data(cens)
    r = _fit(d, cens, y, design=sp.svydesign(d, weights="sw", **DESIGNS[name]))
    # point estimate: the weights= path, already at 1e-8 of R
    assert abs(r.ate - ref["ate"]) <= 1e-8 * max(1.0, abs(ref["ate"]))
    # SE: same 1e-8 budget as the weighted parity (R's glm tolerance)
    assert abs(r.se - ref[name]["se"]) <= 1e-7 * ref[name]["se"], (r.se, ref[name])
    df = ref[name]["degf"]
    assert r.detail["survey_design"]["df"] == df
    crit = stats.t.ppf(0.975, df)
    np.testing.assert_allclose(r.ci[1] - r.ate, crit * r.se, rtol=1e-12)


def test_row_level_design_equals_ic_variance():
    """Each row its own PSU, no strata, no fpc: the linearisation variance
    is n / (n - 1) * sum (u - ubar)^2 with u = IC / n, i.e. var(IC) / n --
    the ``weights=`` standard error (an identity, independent of R)."""
    d = _data(False)
    a = _fit(d, False, "Y", weights="sw")
    b = _fit(d, False, "Y", design=sp.svydesign(d, weights="sw"))
    assert abs(a.ate - b.ate) < 1e-12
    assert abs(a.se - b.se) < 1e-12 * a.se


def test_design_with_weights_refused():
    d = _data(False)
    with pytest.raises(MethodIncompatibility, match="weights"):
        _fit(d, False, "Y", weights="sw", design=sp.svydesign(d, weights="sw"))
