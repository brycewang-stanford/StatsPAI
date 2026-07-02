"""Analytical parity: sp.anderson_rubin_test weak-IV-robust behaviour.

The Anderson-Rubin (1949) test is size-correct regardless of instrument
strength. On a known DGP with true beta = 1 and strong instruments:

* it does NOT reject the true null (p > alpha) and DOES reject a far null;
* the AR confidence set contains the true beta;
* the 2SLS point estimate recovers beta;
* the reported effective F equals sp.effective_f_test's value (consistency).

Analytical evidence tier (known-truth behaviour on a deterministic DGP).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 1.0


def _data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    u = rng.normal(0, 1, n)
    D = 0.7 * Z[:, 0] + 0.5 * Z[:, 1] + 0.6 * u + rng.normal(0, 0.5, n)
    Y = BETA * D + 0.6 * u + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "d": D, "z1": Z[:, 0], "z2": Z[:, 1]})


def test_accepts_true_rejects_false_null():
    df = _data()
    at_true = sp.anderson_rubin_test(
        df, y="y", endog="d", instruments=["z1", "z2"], h0=BETA
    )
    at_false = sp.anderson_rubin_test(
        df, y="y", endog="d", instruments=["z1", "z2"], h0=3.0
    )
    assert float(at_true["ar_pvalue"]) > 0.05
    assert float(at_false["ar_pvalue"]) < 0.01
    lo, hi = at_true["ar_ci"]
    assert float(lo) <= BETA <= float(hi)


def test_2sls_recovers_beta_and_strong_id():
    df = _data()
    r = sp.anderson_rubin_test(df, y="y", endog="d", instruments=["z1", "z2"])
    assert float(r["beta_2sls"]) == pytest.approx(BETA, abs=0.05)
    assert float(r["effective_F"]) > 23.1  # Olea-Pflueger strong threshold


def test_effective_f_consistent_across_entry_points():
    df = _data()
    ar = sp.anderson_rubin_test(df, y="y", endog="d", instruments=["z1", "z2"])
    ef = sp.effective_f_test(df, endog="d", instruments=["z1", "z2"])
    assert float(ar["effective_F"]) == pytest.approx(float(ef["F_eff"]), rel=1e-9)
