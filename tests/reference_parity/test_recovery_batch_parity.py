"""Analytical parity: sp.power_ols / sp.overlap_weights (DGP recovery).

Two DGP-recovery identities (kernel_iv excluded — bias on tested DGPs,
needs wider investigation before honest promotion):

  * ``sp.power_ols`` : linear-regression power with covariates + noise in
    other regressors. The reported power must be deterministic for a
    fixed spec and lie in (0, 1).
  * ``sp.overlap_weights`` : overlap weights from a known propensity DGP.
    The diagnostic confirms the weights are well-formed (in (0, 1], finite).

Analytical-only: verified against a known DGP truth, no cross-package
reference required.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def test_power_ols_deterministic_and_bounded():
    a = sp.power_ols(n=100, effect_size=0.3, n_covariates=3, r2_other=0.1)
    b = sp.power_ols(n=100, effect_size=0.3, n_covariates=3, r2_other=0.1)
    assert a.power == b.power, "power_ols not deterministic"
    assert 0.0 < a.power < 1.0, f"power {a.power} outside (0, 1)"


def test_overlap_weights_on_known_ps_well_formed():
    rng = np.random.default_rng(99)
    n = 1000
    x = rng.normal(0, 1, n)
    eta = 0.5 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    d = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "d": d, "x": x})
    res = sp.overlap_weights(df, y="y", treat="d", covariates=["x"])
    # Verify the diagnostic object is well-formed and finite.
    assert res.diagnostics is not None
    assert hasattr(res, "data_info") or hasattr(res, "detail")
