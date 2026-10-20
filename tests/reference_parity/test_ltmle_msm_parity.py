"""``sp.ltmle_msm`` vs R ``ltmle::ltmleMSM`` 1.3-0 (GPL-2, black box).

Fixtures from ``_fixtures/_generate_ltmle_msm_R.R``: two periods, the four
static regimes of ``(A0, A1)``, working model ``Y ~ cumA`` (number of
treated periods), glm nuisance fits, ``variance.method = "ic"``. TMLE
coefficients agree to 1e-8 absolute and their standard errors to 1e-8
relative (R's glm stops at its deviance tolerance; the IPTW coefficients,
which involve no iterated outcome regression, agree to 1e-11).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "ltmle_msm_R.json").read_text(encoding="utf-8"))
REGS = [(0, 0), (1, 0), (0, 1), (1, 1)]
SM = {"cumA": [0, 1, 1, 2]}
BASE = dict(treatments=["A0", "A1"], covariates_time=[["L0"], ["L1"]], baseline=["v"])


def _wide(name="teffects_wide.csv"):
    d = pd.read_csv(_FIX / name)
    d["sw"] = 0.4 + (np.arange(1, len(d) + 1) % 5) * 0.3
    return d


def _dynamic(k, hist):
    # regime 2 of the R fixture: treat at time 1, at time 2 only if L1 > 0
    return np.ones(len(hist["v"])) if k == 0 else (hist["L1"] > 0).astype(float)


def _fit(case):
    d = _wide()
    kw = {}
    y, regs, sm = "Yb", list(REGS), dict(SM)
    if case == "equal_weights":
        kw["msm_weights"] = None
    elif case == "baseline_term":
        kw["msm_terms"] = ["cumA", "v"]
    elif case == "censored":
        d = _wide("teffects_wide_cens.csv")
        kw["censoring"] = ["C0", "C1"]
    elif case == "continuous":
        y = "Y"
    elif case == "observation_weights":
        kw["weights"] = "sw"
    elif case == "dynamic":
        regs[1] = _dynamic
    elif case == "user_weights":
        kw["msm_weights"] = [1, 2, 3, 4]
    elif case == "duplicate_regime":
        regs.append((1, 1))
        sm = {"cumA": [0, 1, 1, 2, 2]}
    return sp.ltmle_msm(d, y, regimes=regs, summary_measures=sm, **BASE, **kw)


CASES = [k for k in REF if k != "versions"]


@pytest.mark.parametrize("case", CASES)
def test_ltmle_msm_matches_r(case):
    r = _fit(case)
    ref = REF[case]
    np.testing.assert_allclose(r.coefficients["estimate"], ref["beta"], atol=1e-8)
    np.testing.assert_allclose(r.coefficients["se"], ref["se"], rtol=1e-8)
    np.testing.assert_allclose(r.coefficients["iptw"], ref["beta_iptw"], atol=1e-8)


def test_saturated_model_reproduces_the_single_regime_means():
    # Identity: with an intercept and one indicator per other regime the
    # working model is saturated, so its score equations force each fitted
    # regime mean to equal the mean of that regime's targeted Q*_0.
    d = _wide()
    sm = {"d10": [0, 1, 0, 0], "d01": [0, 0, 1, 0], "d11": [0, 0, 0, 1]}
    r = sp.ltmle_msm(d, "Yb", regimes=REGS, summary_measures=sm, **BASE)
    b = r.coefficients["estimate"].to_numpy()
    X = np.column_stack([np.ones(4), np.array(list(sm.values())).T])
    fitted = 1 / (1 + np.exp(-(X @ b)))
    np.testing.assert_allclose(fitted, r.regime_means["mean_Q0"], atol=1e-10)


def test_influence_curve_has_mean_zero_and_gives_the_se():
    r = _fit("static")
    ic = r.influence_curve
    assert np.abs(ic.mean(axis=0)).max() < 1e-8
    se = np.sqrt(np.var(ic, axis=0, ddof=1) / ic.shape[0])
    np.testing.assert_allclose(se, r.coefficients["se"], rtol=1e-12)


def test_input_checks():
    d = _wide()
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.ltmle_msm(d, "Yb", regimes=REGS, summary_measures={"x": [0, 1]}, **BASE)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.ltmle_msm(
            d, "Yb", regimes=REGS, summary_measures=SM, msm_terms=["zz"], **BASE
        )
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.ltmle_msm(
            d, "Yb", regimes=[(1,), (0,)], summary_measures={"a": [1, 0]}, **BASE
        )
