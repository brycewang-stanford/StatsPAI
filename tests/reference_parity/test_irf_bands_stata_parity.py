"""Asymptotic standard errors of ``sp.irf`` against Stata 18 ``irf create``.

Reference: ``_fixtures/irf_bands_Stata.csv`` (the saved ``.irf`` file) from
``_fixtures/_generate_irf_bands_Stata.do`` on the simulated
``_fixtures/irf_bands.csv``. Simple, orthogonalised and cumulative
responses with their delta-method standard errors, for ``var`` and for
``var, dfk``. Closed-form linear algebra on both sides, but Stata keeps
the ``.irf`` file in single precision (the differences sit at 1e-7
relative on entries of order one), so the tolerance is that of a float:
1e-6 relative, 1e-7 absolute.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
NAMES = ["y1", "y2", "y3"]
KINDS = {
    ("irf", "stdirf"): dict(orthogonal=False),
    ("oirf", "stdoirf"): dict(orthogonal=True),
    ("cirf", "stdcirf"): dict(orthogonal=False, cumulative=True),
    ("coirf", "stdcoirf"): dict(orthogonal=True, cumulative=True),
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "irf_bands.csv")


@pytest.fixture(scope="module")
def stata():
    return pd.read_csv(FIX / "irf_bands_Stata.csv")


@pytest.mark.parametrize("irfname,se_df", [("base", "stata"), ("dfk", "unbiased")])
@pytest.mark.parametrize("cols", list(KINDS))
def test_responses_and_standard_errors(data, stata, irfname, se_df, cols):
    fit = sp.var(data, variables=NAMES, lags=2, se_df=se_df)
    out = sp.irf(fit, periods=8, ci="asymptotic", **KINDS[cols])
    ref = stata[stata["irfname"] == irfname]
    for imp in NAMES:
        for resp in NAMES:
            rows = ref[(ref["impulse"] == imp) & (ref["response"] == resp)]
            rows = rows.sort_values("step")
            key = f"{imp} -> {resp}"
            np.testing.assert_allclose(
                out["irf"][key], rows[cols[0]].to_numpy(), rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                out["se"][key], rows[cols[1]].to_numpy(), rtol=1e-6, atol=1e-7
            )


@pytest.mark.parametrize("irfname,se_df", [("base", "stata"), ("dfk", "unbiased")])
def test_variance_decomposition_and_its_standard_errors(data, stata, irfname, se_df):
    # Stata: irf table fevd, stderr. Same single-precision file.
    fit = sp.var(data, variables=NAMES, lags=2, se_df=se_df)
    out = fit.fevd(8, ci="asymptotic")
    ref = stata[stata["irfname"] == irfname]
    for imp in NAMES:
        for resp in NAMES:
            rows = ref[(ref["impulse"] == imp) & (ref["response"] == resp)]
            rows = rows.sort_values("step")
            ours = out[(out["shock"] == imp) & (out["response"] == resp)]
            np.testing.assert_allclose(
                ours["fevd"], rows["fevd"].to_numpy(), rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                ours["se"], rows["stdfevd"].to_numpy(), rtol=1e-6, atol=1e-7
            )
