"""Sanderson-Windmeijer conditional first stage in ``sp.hdfe_ols`` IV.

With several endogenous regressors the plain first-stage F of each one does
not say whether the instruments separate it from the others; referees ask
for the Sanderson-Windmeijer F that ``ivreg2, first`` / ``ivreghdfe, first``
print per endogenous regressor. ``sp.hdfe_ols`` reported only the plain F.

Reference: Stata 18, ivreg2 4.1.12 (no fixed effects) and ivreghdfe 1.1.4 /
reghdfe 6.13.1 (``absorb(f g)``), ranktest 2.0.04, ``e(first)`` rows SWF,
SWFdf2, SWFp, SWchi2, SWchi2p, SWr2 and F, on ``_fixtures/iv_sw.csv``
(``_generate_iv_sw_Stata.do``). Two and three endogenous regressors; iid,
robust and cluster variances; a cluster variable ``fc`` in which the
absorbed ``f`` is nested (its levels are then not charged). The no-FE rows
are fitted here as ``| one`` (a constant absorbed), which is ivreg2's model
with a constant: ``L`` loses the constant and ``sdof`` gains it.

Tolerance rel 1e-7: the absorber is run to 1e-14 and the observed maximum
over every statistic is 6e-10; the values in the JSON carry 17 digits.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "iv_sw_Stata.json").read_text(encoding="utf-8"))
SPECS = {
    "k2": (["d1", "d2"], "z1 + z2 + z3"),
    "k3": (["d1", "d2", "d3"], "z1 + z2 + z3 + z4"),
}
VCE = {
    "iid": {},
    "robust": {"vce": "robust"},
    "cluster": {"cluster": "c"},
    "nested": {"cluster": "fc"},
}
REL = 1e-7


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "iv_sw.csv", float_precision="round_trip")


def _fit(data, k, fe, v):
    endog, inst = SPECS[k]
    absorb = "f + g" if fe == "fe" else "one"
    return sp.hdfe_ols(
        f"y ~ w | {absorb} | {' + '.join(endog)} ~ {inst}",
        data,
        tol=1e-14,
        **VCE[v],
    )


@pytest.mark.parametrize("case", sorted(REF))
def test_matches_ivreg2_first(data, case):
    k, fe, v = case.split("_")
    ref = REF[case]
    r = _fit(data, k, fe, v)
    assert r.n_obs == ref["N"]
    names = SPECS[k][0]
    fs = r.iv_diagnostics["first_stage"]
    pairs = [
        ("sw_F", "SWF"),
        ("sw_F_p", "SWFp"),
        ("sw_chi2", "SWchi2"),
        ("sw_chi2_p", "SWchi2p"),
        ("sw_partial_r2", "SWr2"),
        ("F", "F"),
    ]
    for ours, theirs in pairs:
        np.testing.assert_allclose(
            [fs[n][ours] for n in names], ref[theirs], rtol=REL, err_msg=ours
        )
    assert [fs[n]["sw_F_df"] for n in names] == [
        (int(a), int(b)) for a, b in zip(ref["SWFdf1"], ref["SWFdf2"])
    ]
    assert all(fs[n]["sw_chi2_df"] == len(SPECS[k][1].split("+")) - len(names) + 1
               for n in names)  # fmt: skip
    assert "Sanderson-Windmeijer F:" in r.summary()


def test_one_endogenous_regressor_sw_is_the_first_stage(data):
    # ivreg2: with K1 = 1, SWF = F, SWchi2 = chi2 and SWr2 = the partial R2
    for v in VCE:
        r = sp.hdfe_ols("y ~ w | f + g | d1 ~ z1 + z2", data, tol=1e-14, **VCE[v])
        fs = r.iv_diagnostics["first_stage"]["d1"]
        assert fs["sw_F"] == pytest.approx(fs["F"], rel=1e-12)
        assert fs["sw_chi2"] == pytest.approx(fs["chi2"], rel=1e-12)
        assert fs["sw_partial_r2"] == pytest.approx(fs["partial_r2"], rel=1e-12)
        assert fs["sw_F_df"] == fs["F_df"]
        assert "Sanderson-Windmeijer" not in r.summary()


def test_exactly_identified_has_one_numerator_df(data):
    # L1 = K1: delta uses all but one instrument direction, df = 1
    r = sp.hdfe_ols("y ~ w | f + g | d1 + d2 ~ z1 + z3", data, tol=1e-14)
    fs = r.iv_diagnostics["first_stage"]
    assert fs["d1"]["sw_F_df"][0] == fs["d2"]["sw_F_df"][0] == 1
    # with one numerator df under iid the F is chi2 / q, q = N / (N - L - sdof)
    N, L, sdof = r.n_obs, 3, r.iv_diagnostics["sdofminus"]
    for n in ("d1", "d2"):
        assert fs[n]["sw_F"] == pytest.approx(
            fs[n]["sw_chi2"] * (N - L - sdof) / N, rel=1e-12
        )
