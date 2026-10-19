"""``weights=`` added (or refused with a typed error) on estimators that
lacked it: ``sp.gformula_mc``, ``sp.causal_impact``, ``sp.sar``.
``sp.finegray(weights=)`` is pinned to Stata in
``tests/reference_parity/test_finegray_weighted_stata_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _informative_sample() -> pd.DataFrame:
    """Population E[Y(1,1)] = 1 + 1.6 + 0.5 * 0.5 = 2.85 (E[L0] = 0); the
    sample over-represents high L0 (inclusion logit -2.5 + 1.5 L0), so the
    unweighted g-formula is pulled up by 0.75 E_sample[L0]."""
    rng = np.random.default_rng(11)
    N = 60000
    L0 = rng.normal(size=N)
    A0 = (rng.random(N) < 1 / (1 + np.exp(-L0))).astype(float)
    L1 = 0.5 * L0 + 0.5 * A0 + rng.normal(size=N)
    A1 = (rng.random(N) < 1 / (1 + np.exp(-L1))).astype(float)
    Y = 1 + 0.8 * (A0 + A1) + 0.5 * L1 + 0.5 * L0 + rng.normal(size=N)
    pi = 1 / (1 + np.exp(-(-2.5 + 1.5 * L0)))
    keep = rng.random(N) < pi
    df = pd.DataFrame({"L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y, "w": 1 / pi})
    return df[keep].reset_index(drop=True)


def test_gformula_mc_weights_recover_the_population_mean():
    df = _informative_sample()
    kw = dict(
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        strategy=[1, 1],
        n_simulations=200000,
        bootstrap=0,
        seed=0,
    )
    weighted = sp.gformula_mc(df, weights="w", **kw).value
    unweighted = sp.gformula_mc(df, **kw).value
    # sampling SE of the weighted estimate is ~0.06 (bootstrap, 30 reps)
    assert abs(weighted - 2.85) < 0.15
    assert unweighted - 2.85 > 0.5  # the bias the weights remove


def test_gformula_mc_weighted_fits_equal_replicated_rows():
    """Integer weights are frequency counts for the models' coefficients."""
    from statspai.gformula.mc import _fit_models

    df = _informative_sample().iloc[:400].reset_index(drop=True)
    k = np.arange(len(df)) % 3 + 1
    rep = df.loc[np.repeat(np.arange(len(df)), k)].reset_index(drop=True)
    a = _fit_models(df, ["A0", "A1"], [["L0"], ["L1"]], "Y", k.astype(float))
    b = _fit_models(rep, ["A0", "A1"], [["L0"], ["L1"]], "Y")
    np.testing.assert_allclose(a.outcome_model[1], b.outcome_model[1], rtol=1e-9)
    np.testing.assert_allclose(
        a.conf_models[1][0][1], b.conf_models[1][0][1], rtol=1e-9
    )


def test_gformula_mc_rejects_non_positive_weights():
    df = _informative_sample().iloc[:200].copy()
    df.loc[0, "w"] = 0.0
    with pytest.raises(MethodIncompatibility):
        sp.gformula_mc(
            df,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=[1, 1],
            weights="w",
            bootstrap=0,
            n_simulations=500,
        )


def test_causal_impact_and_sar_refuse_weights():
    ts = pd.DataFrame({"y": np.arange(20.0), "t": np.arange(20)})
    with pytest.raises(MethodIncompatibility, match="single"):
        sp.causal_impact(ts, "y", "t", 10, weights="w")
    W = np.eye(4, k=1) + np.eye(4, k=-1)
    d = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 0.0, 1.0]})
    with pytest.raises(MethodIncompatibility, match="joint"):
        sp.sar(W, d, "y ~ x", weights="x")
