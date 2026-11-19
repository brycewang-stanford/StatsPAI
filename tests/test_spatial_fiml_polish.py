"""The spatial ML optimum does not depend on where the simplex stops.

The GNS likelihood is flat along the rho-lambda trade-off. Nelder-Mead
stopped wherever its simplex shrank below tolerance, which moved with the
platform's rounding: one coefficient was 1e-6 apart between two CI
runners (Parity guards, 2026-11-19). Newton steps on the analytic gradient
of the concentrated likelihood now finish the search at the stationary
point.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
import statspai.spatial.models._fiml as fiml

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


def _load(sample):
    df = pd.read_csv(_FIX / f"spatial_durbin_error_{sample}.csv")
    return df, df[[f"w{j + 1}" for j in range(len(df))]].to_numpy()


@pytest.mark.parametrize("model", ["gns", "sdem"])
@pytest.mark.parametrize("sample", [1, 2])
def test_estimates_do_not_depend_on_the_simplex_tolerance(monkeypatch, model, sample):
    df, A = _load(sample)
    tight = getattr(sp, model)(A, df, "y ~ x1 + x2").params
    real_minimize = fiml.minimize

    def loose(fun, x0, method=None, options=None, **kw):
        return real_minimize(
            fun, x0, method=method, options={"xatol": 1e-4, "fatol": 1e-6}
        )

    monkeypatch.setattr(fiml, "minimize", loose)
    rough = getattr(sp, model)(A, df, "y ~ x1 + x2").params
    np.testing.assert_allclose(rough.to_numpy(), tight.to_numpy(), rtol=1e-10)
