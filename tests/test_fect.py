"""sp.fect: counterfactual estimators for TSCS data (Liu, Wang and Xu 2024).

Correctness is pinned by parity module 86 (R fect on identical bytes);
these tests cover the analytic identities and the API contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _staggered_panel(seed: int = 3, N: int = 30, T: int = 12, factors: bool = False):
    rng = np.random.default_rng(seed)
    first = np.zeros(N, dtype=int)
    first[10:20] = 6
    first[20:30] = 9
    alpha = rng.normal(size=N)
    xi = rng.normal(size=T)
    F = rng.normal(size=(T, 2))
    L = rng.normal(size=(N, 2))
    rows = []
    for i in range(N):
        for t in range(1, T + 1):
            d = int(first[i] > 0 and t >= first[i])
            y = 2.0 + alpha[i] + xi[t - 1] + (F[t - 1] @ L[i] if factors else 0.0)
            y += 1.5 * d + rng.normal(scale=0.3)
            rows.append({"id": i + 1, "time": t, "y": y, "d": d, "g": first[i]})
    return pd.DataFrame(rows)


def test_fe_matches_did_imputation_on_staggered_panel():
    """fect's fe model is the imputation estimator: untreated-only TWFE
    fit, ATT = mean(Y - Y0) over treated cells (BJS / Gardner)."""
    df = _staggered_panel()
    res = sp.fect(df, y="y", treat="d", unit="id", time="time", method="fe", tol=1e-10)
    ref = sp.did_imputation(df, y="y", group="id", time="time", first_treat="g")
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-8)
    assert res.model_info["niter"] >= 1
    assert res.se is None and res.ci is None


def test_fe_recovers_constant_effect_without_factors():
    df = _staggered_panel(factors=False)
    res = sp.fect(df, y="y", treat="d", unit="id", time="time", method="fe")
    assert abs(res.estimate - 1.5) < 0.15


def test_ife_beats_fe_when_factors_load_on_treatment():
    """With a factor structure the two-way model is biased; r=2 removes it."""
    df = _staggered_panel(factors=True, seed=11)
    fe = sp.fect(df, y="y", treat="d", unit="id", time="time", method="fe")
    ife = sp.fect(
        df, y="y", treat="d", unit="id", time="time", method="ife", r=2, tol=1e-8
    )
    assert ife.model_info["r"] == 2
    assert ife.model_info["factors"].shape == (12, 2)
    assert abs(ife.estimate - 1.5) < abs(fe.estimate - 1.5) + 0.05


def test_relative_time_table_uses_fect_coding():
    df = _staggered_panel()
    res = sp.fect(df, y="y", treat="d", unit="id", time="time")
    d = res.detail
    assert list(d.columns[:4]) == ["fect_time", "relative_time", "att", "count"]
    assert (d["relative_time"] == d["fect_time"] - 1).all()
    # cohort 6 has 5 untreated periods (fect_time -4..0), cohort 9 has 8.
    assert d["fect_time"].min() == -7
    assert d["fect_time"].max() == 7  # cohort 6 treated in periods 6..12
    assert int(d.loc[d["fect_time"] == 1, "count"].iloc[0]) == 20
    assert int(d.loc[d["fect_time"] == 0, "count"].iloc[0]) == 20


def test_mc_requires_lambda_and_ife_requires_r():
    df = _staggered_panel()
    with pytest.raises(ValueError):
        sp.fect(df, y="y", treat="d", unit="id", time="time", method="mc")
    with pytest.raises(ValueError):
        sp.fect(df, y="y", treat="d", unit="id", time="time", method="ife", r=0)
    with pytest.raises(ValueError):
        sp.fect(df, y="y", treat="d", unit="id", time="time", method="pca")


def test_mc_large_lambda_collapses_to_fe():
    """A penalty above the largest singular value removes every factor,
    so the matrix-completion fit is the two-way fit (fect behaves the same)."""
    df = _staggered_panel(factors=True)
    fe = sp.fect(df, y="y", treat="d", unit="id", time="time", method="fe", tol=1e-10)
    mc = sp.fect(
        df, y="y", treat="d", unit="id", time="time", method="mc", lam=10.0, tol=1e-10
    )
    assert mc.model_info["lambda_norm"] > 1
    assert mc.estimate == pytest.approx(fe.estimate, rel=1e-6)


def test_min_t0_drops_short_pretreatment_units_with_warning():
    df = _staggered_panel()
    df.loc[
        (df["id"] == 11) & (df["time"] < 6), "y"
    ] = np.nan  # unit 11 has 0 untreated periods
    with pytest.warns(UserWarning, match="dropped"):
        res = sp.fect(
            df, y="y", treat="d", unit="id", time="time", method="fe", min_t0=1
        )
    assert res.model_info["dropped_units"] == [11]
    assert res.model_info["n_units"] == 29


def test_bootstrap_and_jackknife_se_are_positive_and_seeded():
    df = _staggered_panel()
    a = sp.fect(
        df, y="y", treat="d", unit="id", time="time", vce="bootstrap", n_boot=30, seed=1
    )
    b = sp.fect(
        df, y="y", treat="d", unit="id", time="time", vce="bootstrap", n_boot=30, seed=1
    )
    assert a.se is not None and a.se > 0 and a.se == b.se
    assert a.ci[0] < a.estimate < a.ci[1]
    assert "se" in a.detail.columns
    j = sp.fect(df, y="y", treat="d", unit="id", time="time", vce="jackknife")
    assert j.se is not None and j.se > 0
    assert j.model_info["n_boot_success"] == 30


def test_citation_and_registry():
    df = _staggered_panel()
    res = sp.fect(df, y="y", treat="d", unit="id", time="time")
    assert "liu2024practical" in res.cite()
    spec = sp.describe_function("fect")
    assert spec["name"] == "fect"
