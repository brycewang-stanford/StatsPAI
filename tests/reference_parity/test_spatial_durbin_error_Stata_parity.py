"""``sp.sdem`` / ``sp.gns`` against Stata 18 ``spregress, ml``.

Reference: ``spregress y x1 x2, ml errorlag(W) ivarlag(W: x1 x2)`` (SDEM)
and ``... dvarlag(W) errorlag(W) ivarlag(W: x1 x2)`` (GNS) with tight
``ml`` tolerances, on a symmetric rook grid (n = 80) and an asymmetric
4-nearest-neighbour graph (n = 70), from
``_fixtures/_generate_spatial_durbin_error_stata.do``. The log-likelihoods
agree to 1e-12; coefficients, OIM standard errors and the error variance
to 1e-6 (observed 3e-7: the GNS likelihood is flat along the rho-lambda
trade-off, so equal likelihoods leave the coefficients that far apart).
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
    "x1": "y:x1",
    "x2": "y:x2",
    "W_x1": "W:x1",
    "W_x2": "W:x2",
    "rho": "W:y",
    "lambda": "W:e.y",
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "spatial_durbin_error_Stata.json").read_text(encoding="utf-8")
    )


def _load(s):
    df = pd.read_csv(
        _FIX / f"spatial_durbin_error_{s}.csv", float_precision="round_trip"
    )
    A = df[[f"w{j + 1}" for j in range(len(df))]].to_numpy()
    return df, A


@pytest.mark.parametrize("model", ["sdem", "gns"])
@pytest.mark.parametrize("sample", [1, 2])
def test_matches_spregress_ml(ref, model, sample):
    df, A = _load(sample)
    r = getattr(sp, model)(A, df, "y ~ x1 + x2")
    R = ref[f"{sample}_{model}"]
    assert r.model_info["loglik"] == pytest.approx(R["ll"], rel=1e-12)
    for ours, theirs in _NAMES.items():
        if ours not in r.params.index:
            assert model == "sdem" and ours == "rho"
            continue
        b, se = R["coef"][theirs]
        assert r.params[ours] == pytest.approx(b, rel=1e-6, abs=1e-8), ours
        assert r.std_errors[ours] == pytest.approx(se, rel=1e-6), ours
    s2, se_s2 = R["coef"]["/var(e.y)"]
    assert r.model_info["sigma2"] == pytest.approx(s2, rel=1e-6)
    assert r.model_info["se_sigma2"] == pytest.approx(se_s2, rel=1e-6)


def test_gns_nests_sdem_and_sdm_likelihoods():
    """GNS maximises over a superset, so its log-likelihood is at least that
    of SDEM (rho = 0) on the same data."""
    df, A = _load(1)
    g = sp.gns(A, df, "y ~ x1 + x2").model_info["loglik"]
    e = sp.sdem(A, df, "y ~ x1 + x2").model_info["loglik"]
    assert g >= e - 1e-9


def test_collinear_lag_is_refused():
    df, A = _load(1)
    df = df.assign(c=1.0)
    with pytest.raises(sp.NumericalInstability, match="collinear"):
        sp.sdem(A, df, "y ~ x1 + c")
