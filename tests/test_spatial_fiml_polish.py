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


@pytest.mark.parametrize("lag", [True, False])
@pytest.mark.parametrize("sample", [1, 2])
def test_observed_information_is_the_hessian_of_the_log_likelihood(lag, sample):
    """The analytic observed information equals a finite-difference Hessian
    of the full log-likelihood, written out here independently, to that
    difference's own error. A step-based Hessian lost ~1e-6 relative to
    rounding, and that part differed between CI runners (nightly
    2026-11-20/21, GNS const SE 1.2e-6 from Stata on Linux only)."""
    df, A = _load(sample)
    Wd = A / A.sum(axis=1, keepdims=True)
    X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])
    Z = np.column_stack([X, Wd @ X[:, 1:]])
    y = df["y"].to_numpy(dtype=float)
    fit = fiml._fit(y, Z, Wd, lag)
    n = len(y)
    k = Z.shape[1]

    def loglik(p):
        g, r = p[:k], (p[k] if lag else 0.0)
        lam, s = p[-2], p[-1]
        e = (np.eye(n) - lam * Wd) @ (y - r * (Wd @ y) - Z @ g)
        ld = np.linalg.slogdet(np.eye(n) - lam * Wd)[1]
        if lag:
            ld += np.linalg.slogdet(np.eye(n) - r * Wd)[1]
        return -n / 2 * np.log(2 * np.pi * s) + ld - e @ e / (2 * s)

    p = np.concatenate(
        [fit["g"], [fit["rho"]] if lag else [], [fit["lam"], fit["sigma2"]]]
    )
    H_num = fiml._hessian(loglik, p)
    H = -np.linalg.inv(fit["V"])
    np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-5 * np.abs(H).max())
    assert np.allclose(H, H.T, rtol=0, atol=1e-12 * np.abs(H).max())
