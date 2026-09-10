"""Coverage round-2 — ``statspai.rd.bandwidth`` (rdbwselect internals).

Round 1 covered all eight selectors / fuzzy / covs / cluster / kernels.
This file adds:

- the public input-validation error branches (bad kernel, bad bwselect,
  negative deriv, p<1, q<=p);
- the too-few-obs / too-small-sample errors;
- the internal local-poly helper "too-few-observations-in-bandwidth"
  early-return branches, driven on real RD data with a tiny bandwidth so the
  defensive fallbacks (return zeros / raw variance / 0.0) are exercised.

Real synthetic RD data; assertions check the helpers return finite, sane
fallbacks (non-negative variances, zero derivative) — never fabricated
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _df(seed=2, n=2000, tau=3.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + tau * (x >= 0) + rng.normal(0, 0.3, n)
    z = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.2 + 0.6 * (x >= 0)).astype(float)
    g = (np.arange(n) // 25).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z, "d": d, "g": g})


def test_rdbwselect_validation_errors():
    df = _df()
    with pytest.raises(ValueError, match="kernel"):
        sp.rdbwselect(df, y="y", x="x", c=0, kernel="bad")
    with pytest.raises(ValueError, match="bwselect"):
        sp.rdbwselect(df, y="y", x="x", c=0, bwselect="bad")
    with pytest.raises(ValueError, match="deriv"):
        sp.rdbwselect(df, y="y", x="x", c=0, deriv=-1)
    with pytest.raises(ValueError, match="p must"):
        sp.rdbwselect(df, y="y", x="x", c=0, p=0)
    with pytest.raises(ValueError, match="q must"):
        sp.rdbwselect(df, y="y", x="x", c=0, p=2, q=2)


def test_rdbwselect_too_few_observations():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 12)
    y = rng.normal(size=12)
    df = pd.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="20 observations"):
        sp.rdbwselect(df, y="y", x="x", c=0)


def test_rdbwselect_one_side_too_few():
    rng = np.random.default_rng(1)
    # 40 obs but only one on the left side -> per-side check fails
    x = np.concatenate([np.array([-0.5]), rng.uniform(0.01, 1, 39)])
    y = rng.normal(size=40)
    df = pd.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="each side"):
        sp.rdbwselect(df, y="y", x="x", c=0)


def test_rdbwselect_deriv_auto_bumps_p():
    # deriv=1 with p=1 forces p up to deriv+1; should still produce output
    df = _df()
    out = sp.rdbwselect(df, y="y", x="x", c=0, deriv=1, p=1)
    assert (out["h_left"] > 0).all()


# The four tests that used to follow here exercised `_local_poly_fit`,
# `_local_residual_var`, `_estimate_deriv` and
# `_covariate_adjusted_variance` -- private helpers of the single-step
# rule of thumb this module used to run. That implementation was removed
# in 1.27.0 once `sp.rdbwselect` was rewired onto the CCT cascade (Track A
# module 88), so the tests went with it rather than being retargeted:
# they asserted degenerate-bandwidth fallbacks of code that no longer
# exists, and keeping them alive would have meant keeping the code alive.
# The cascade's own behaviour is covered by `88_rdbwselect` and
# `tests/reference_parity/test_rdbwselect_comb_rules.py`.
