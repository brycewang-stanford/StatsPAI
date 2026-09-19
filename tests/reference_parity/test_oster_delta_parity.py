"""Analytical parity: sp.oster_delta structural identities (Oster 2019).

Coefficient-stability bounds report quantities that are exact functions of
two OLS fits and the delta grid:

    beta_short / r2_short  == the base OLS fit          (machine-exact)
    beta_full  / r2_full   == the controlled OLS fit    (machine-exact)
    upper == beta_full;  lower == beta_star(delta=1)
    beta*(delta_star) == 0 -- delta* is exactly the zero of the (exact)
    bias-adjusted coefficient.

Verified against hand-rolled lstsq fits. Cross-language parity against
Oster's Stata psacalc: test_inference_sens_stata_parity.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

N = 2000


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, N)
    x2 = rng.normal(0, 1, N)
    d = 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, N)
    y = 1.0 * d + 0.8 * x1 + 0.5 * x2 + rng.normal(0, 1, N)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    res = sp.oster_delta(df, y="y", x_base=["d"], x_controls=["x1", "x2"], n_boot=20)
    return df, res


def _ols(df, cols):
    X = np.column_stack([np.ones(N)] + [df[c].to_numpy() for c in cols])
    y = df["y"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ b
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(b[1]), float(r2)


def test_ols_inputs_match_hand_fits(fitted):
    df, r = fitted
    mi = r.model_info
    b_s, r2_s = _ols(df, ["d"])
    b_f, r2_f = _ols(df, ["d", "x1", "x2"])
    assert mi["beta_short"] == pytest.approx(b_s, abs=1e-12)
    assert mi["r2_short"] == pytest.approx(r2_s, abs=1e-12)
    assert mi["beta_full"] == pytest.approx(b_f, abs=1e-12)
    assert mi["r2_full"] == pytest.approx(r2_f, abs=1e-12)


def test_bound_endpoints_are_named_quantities(fitted):
    _, r = fitted
    mi = r.model_info
    assert r.upper == pytest.approx(mi["beta_full"], abs=1e-15)
    assert r.lower == pytest.approx(mi["beta_star_delta1"], abs=1e-15)


def test_delta_star_is_zero_crossing(fitted):
    """delta* is where the bias-adjusted coefficient reaches zero: fed back
    into Oster's cubic, it yields a root at 0 (exact solution; the grid is
    only a plotting aid and is non-linear in delta)."""
    from statspai.diagnostics._oster import oster_beta_exact, oster_inputs

    df, r = fitted
    mi = r.model_info
    inp = oster_inputs(df, "y", "d", ["x1", "x2"])
    roots = oster_beta_exact(inp, mi["r_max"], delta=mi["delta_star"])["roots"]
    assert min(abs(x) for x in roots) == pytest.approx(0.0, abs=1e-10)
