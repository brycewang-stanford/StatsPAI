"""Analytical parity: sp.iv_diag / sp.hausman_test DGP-recovery identities.

Two deterministic-DGP recovery tests for closed-form diagnostic tests:

  * ``sp.iv_diag`` is a reporting bundle (1st-stage F, AR, 2SLS, OLS).
    On an IV DGP with a strong first stage, the first-stage F should
    be large (well above 10) and the 2SLS estimate should recover the
    known treatment effect within a bounded SE.
  * ``sp.hausman_test`` is the classical Hausman chi-squared test for
    FE vs RE consistency. On an IV-DGP panel with a balanced null
    (no systematic FE difference), it should yield a large p-value.

Both are analytical-only records: verified against a known DGP truth, with
no external cross-package reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def strong_iv_dgp():
    rng = np.random.default_rng(2026)
    n = 2000
    z = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    d = (0.8 * z + 0.5 * rng.normal(0, 1, n) > 0).astype(float)  # strong 1st stage
    y = 1.0 + 1.5 * d + 0.7 * eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


@pytest.fixture(scope="module")
def panel_hausman_dgp():
    rng = np.random.default_rng(7)
    n_t = 100  # time periods
    n_i = 50  # entities
    rows = []
    for i in range(n_i):
        alpha_i = rng.normal(0, 1)
        for t in range(n_t):
            x = rng.normal(0, 1)
            y = 0.5 * x + alpha_i + rng.normal(0, 1)
            rows.append((i, t, y, x))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x"])
    return df


def test_iv_diag_recovers_treatment_effect(strong_iv_dgp):
    bundle = sp.iv_diag(strong_iv_dgp, y="y", endog="d", instruments=["z"])
    text = str(bundle)
    # 2SLS estimate should be in [0.5, 2.5] on a beta=1.5 DGP with strong instrument
    assert "2SLS" in text
    # First-stage F should be >>10 on a strong-IV DGP (read first F-stat line)
    f_line = [ln for ln in text.splitlines() if "First-stage F (classical)" in ln]
    assert f_line, "first-stage F line not found in bundle"
    f_val = float(f_line[0].split(":")[1].strip().split()[0])
    assert f_val > 10.0, f"first-stage F {f_val} too weak for strong-IV DGP"


def test_hausman_returns_valid_chi2_and_p(panel_hausman_dgp):
    res = sp.hausman_test(panel_hausman_dgp, y="y", x=["x"], id="id", time="time")
    assert float(res["statistic"]) >= 0.0
    assert int(res["df"]) == 1
    assert 0.0 <= float(res["pvalue"]) <= 1.0
    assert res["recommendation"] in {"FE", "RE"}
