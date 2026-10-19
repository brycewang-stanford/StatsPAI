"""``sp.spatial_panel(effects='re')`` against Stata 18 ``spxtregress, re``.

Reference: ``spxtregress y x x2, re`` with ``dvarlag(W)`` (SAR),
``errorlag(W)`` (SEM), both (SAC) and ``dvarlag(W) ivarlag(W: x x2)``
(SDM), tight ``ml`` tolerances, from
``_fixtures/_generate_spatial_panel_re_stata.do``: 48 units on a rook grid,
5 periods. Stata's ``dvarlag`` coefficient is our ``rho`` and its
``errorlag`` coefficient our ``lambda`` (the package's naming, as
``sp.sac``). Log-likelihoods agree to 1e-10; coefficients, the two
standard deviations and OIM standard errors to 1e-5 (observed 1e-7).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_NAMES = {
    "const": "y:_cons",
    "x": "y:x",
    "x2": "y:x2",
    "W_x": "W:x",
    "W_x2": "W:x2",
    "rho": "W:y",
    "lambda": "W:e.y",
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "spatial_panel_re_Stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(_FIX / "spatial_panel_re.csv", float_precision="round_trip")
    w = pd.read_csv(_FIX / "spatial_panel_re_W.csv")
    A = w[[f"w{j + 1}" for j in range(len(w))]].to_numpy()
    return df, A


@pytest.mark.parametrize("model", ["sar", "sem", "sac", "sdm"])
def test_matches_spxtregress_re(ref, panel, model):
    df, A = panel
    r = sp.spatial_panel(
        df, "y ~ x + x2", entity="id", time="t", W=A, model=model, effects="re"
    )
    R = ref[model]
    assert r.log_likelihood == pytest.approx(R["ll"], rel=1e-10)
    for ours in r.params.index:
        b, se = R["coef"][_NAMES[ours]]
        assert r.params[ours] == pytest.approx(b, rel=1e-5, abs=1e-7), ours
        assert r.std_errors[ours] == pytest.approx(se, rel=1e-5), ours
    for name in ("sigma_u", "sigma_e"):
        b, se = R["coef"][f"{name}:_cons"]
        assert r.variance_components[name] == pytest.approx(b, rel=1e-5)
        assert r.variance_components[f"se_{name}"] == pytest.approx(se, rel=1e-5)


def test_sac_needs_random_effects(panel):
    df, A = panel
    with pytest.raises(sp.MethodIncompatibility, match="effects='re'"):
        sp.spatial_panel(df, "y ~ x", entity="id", time="t", W=A, model="sac")
