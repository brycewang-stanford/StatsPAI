"""``sp.ltmle(weights=)`` vs R ``ltmle`` 1.3-0 ``observation.weights``.

Fixture: ``_fixtures/ltmle_weights_R.json`` from
``_fixtures/_generate_ltmle_weights_R.R`` (glm learners, default gbounds,
``variance.method = "ic"``) on the wide fixtures of
``test_teffects_R_parity.py``, with weights ``sw = 0.4 + (row % 5) * 0.3``
(rows from 1). ltmle is GPL-2 and was used as a black box (outputs only).
R's glms stop at their default deviance tolerance; the censored rows are
held to 1e-8 as in the unweighted test (observed ~1e-11).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "ltmle_weights_R.json").read_text(encoding="utf-8"))


def _fit(cens: bool, y: str, w: str = "sw"):
    d = pd.read_csv(_FIX / ("teffects_wide_cens.csv" if cens else "teffects_wide.csv"))
    d["sw"] = 0.4 + (np.arange(1, len(d) + 1) % 5) * 0.3
    d["one"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ltmle(
            d,
            y=y,
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            baseline=["v"],
            censoring=["C0", "C1"] if cens else None,
            weights=w,
        )


@pytest.mark.parametrize("cens", [False, True])
@pytest.mark.parametrize("y", ["Yb", "Y"])
def test_weighted_ltmle_matches_r(cens, y):
    r = _fit(cens, y)
    ref = REF[("cens_" if cens else "nocens_") + y]
    for got, want in [
        (r.psi_treated, ref["psi1"]),
        (r.psi_control, ref["psi0"]),
        (r.ate, ref["ate"]),
        (r.se, ref["ate_se"]),
    ]:
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (got, want)


@pytest.mark.parametrize("cens", [False, True])
def test_unit_weights_are_the_unweighted_estimator(cens):
    a = _fit(cens, "Y", w="one")
    d = pd.read_csv(_FIX / ("teffects_wide_cens.csv" if cens else "teffects_wide.csv"))
    b = sp.ltmle(
        d,
        y="Y",
        treatments=["A0", "A1"],
        covariates_time=[["L0"], ["L1"]],
        baseline=["v"],
        censoring=["C0", "C1"] if cens else None,
    )
    assert abs(a.ate - b.ate) < 1e-12 and abs(a.se - b.se) < 1e-12


def test_weights_are_scale_free():
    a = _fit(False, "Y")
    d = pd.read_csv(_FIX / "teffects_wide.csv")
    d["sw"] = 3.0 * (0.4 + (np.arange(1, len(d) + 1) % 5) * 0.3)
    b = sp.ltmle(
        d,
        y="Y",
        treatments=["A0", "A1"],
        covariates_time=[["L0"], ["L1"]],
        baseline=["v"],
        weights="sw",
    )
    assert abs(a.ate - b.ate) < 1e-10 and abs(a.se - b.se) < 1e-10 * a.se


def test_rejects_nonpositive_weights():
    d = pd.read_csv(_FIX / "teffects_wide.csv")
    d["sw"] = 0.0
    with pytest.raises(Exception, match="positive"):
        sp.ltmle(
            d,
            y="Y",
            treatments=["A0", "A1"],
            covariates_time=[["L0"], ["L1"]],
            baseline=["v"],
            weights="sw",
        )
