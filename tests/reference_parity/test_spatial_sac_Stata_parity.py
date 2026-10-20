"""``sp.sac`` against Stata 18 ``spregress, ml dvarlag(W) errorlag(W)``.

Same two fixtures as the SDEM / GNS check (rook grid n = 80, asymmetric
4-nearest-neighbour graph n = 70), reference from
``_fixtures/_generate_spatial_sac_stata.do`` with tight ``ml`` tolerances.
Log-likelihoods agree to 1e-12; coefficients, OIM standard errors and the
error variance to 1e-6 (observed 3.5e-7 on the constant's SE: the
likelihood is flat along the rho-lambda trade-off).

Through 1.39 ``sp.sac`` reported the conditional ``sigma2 (X~'X~)^-1``
regression SEs and the diagonal of a numerical Hessian of the concentrated
likelihood for rho / lambda: the constant's SE was 0.109 against Stata's
0.344 on fixture 1, and on fixture 2 the optimiser stopped 0.024 log-
likelihood units short of the maximum.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_NAMES = {
    "const": "y:_cons",
    "x1": "y:x1",
    "x2": "y:x2",
    "rho": "W:y",
    "lambda": "W:e.y",
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "spatial_sac_Stata.json").read_text(encoding="utf-8"))


def _load(s):
    df = pd.read_csv(
        _FIX / f"spatial_durbin_error_{s}.csv", float_precision="round_trip"
    )
    A = df[[f"w{j + 1}" for j in range(len(df))]].to_numpy()
    return df, A


@pytest.mark.parametrize("sample", [1, 2])
def test_sac_matches_spregress_ml(ref, sample):
    df, A = _load(sample)
    r = sp.sac(A, df, "y ~ x1 + x2")
    R = ref[f"{sample}_sac"]
    assert r.model_info["loglik"] == pytest.approx(R["ll"], rel=1e-12)
    for ours, theirs in _NAMES.items():
        b, se = R["coef"][theirs]
        assert r.params[ours] == pytest.approx(b, rel=1e-6, abs=1e-8), ours
        assert r.std_errors[ours] == pytest.approx(se, rel=1e-6), ours
    s2, se_s2 = R["coef"]["/var(e.y)"]
    assert r.model_info["sigma2"] == pytest.approx(s2, rel=1e-6)
    assert r.model_info["se_sigma2"] == pytest.approx(se_s2, rel=1e-6)


def test_sac_vcov_diagonal_is_the_reported_se():
    df, A = _load(1)
    r = sp.sac(A, df, "y ~ x1 + x2")
    V = r.model_info["vcov"]
    assert list(V.index) == list(r.params.index)
    np.testing.assert_allclose(np.sqrt(np.diag(V.to_numpy())), r.std_errors, rtol=1e-12)


def test_sac_is_gns_without_the_durbin_terms():
    """The SAC likelihood is the GNS likelihood at theta = 0, so SAC's
    maximum cannot exceed the GNS maximum on the same data."""
    df, A = _load(2)
    s = sp.sac(A, df, "y ~ x1 + x2").model_info["loglik"]
    g = sp.gns(A, df, "y ~ x1 + x2").model_info["loglik"]
    assert s <= g + 1e-9
