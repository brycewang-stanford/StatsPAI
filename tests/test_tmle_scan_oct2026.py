"""Option scan (Oct 2026) of ``statspai.tmle`` (survival LTMLE, Super Learner)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import statspai as sp


def _survival_panel(n: int, seed: int) -> pd.DataFrame:
    """Two intervals, sequentially randomised treatment, lagged effect.

    Event in interval 1 with hazard expit(-1 - 0.8 A1); among survivors,
    event in interval 2 with hazard expit(-1 - A1 - 0.3 A2). L1, L2 are
    independent noise covariates.
    """
    rng = np.random.default_rng(seed)
    A1 = rng.integers(0, 2, n)
    A2 = rng.integers(0, 2, n)
    L1 = rng.normal(size=n)
    L2 = rng.normal(size=n)
    T1 = (rng.uniform(size=n) < expit(-1 - 0.8 * A1)).astype(int)
    T2 = ((T1 == 0) & (rng.uniform(size=n) < expit(-1 - A1 - 0.3 * A2))).astype(int)
    return pd.DataFrame({"A1": A1, "A2": A2, "L1": L1, "L2": L2, "T1": T1, "T2": T2})


def test_ltmle_survival_recovers_static_regime_survival():
    df = _survival_panel(20000, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ltmle_survival(
            df,
            event_indicators=["T1", "T2"],
            treatments=["A1", "A2"],
            covariates_time=[["L1"], ["L2"]],
        )
    s11 = (1 - expit(-1.8)) * (1 - expit(-2.3))  # 0.7799
    s00 = (1 - expit(-1.0)) * (1 - expit(-1.0))  # 0.5345
    s1_1 = 1 - expit(-1.8)
    s0_1 = 1 - expit(-1.0)
    # Interval 1 is a single randomised step; interval 2 needs A1 set too.
    assert abs(res.survival_treated[0] - s1_1) < 0.015
    assert abs(res.survival_control[0] - s0_1) < 0.015
    assert abs(res.survival_treated[1] - s11) < 0.02
    assert abs(res.survival_control[1] - s00) < 0.02
    assert abs(res.risk_difference_final - (s00 - s11)) < 0.025


def test_superlearner_fit_does_not_mutate_library():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n)
    sl = sp.SuperLearner(n_folds=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sl.fit(X, y)
        assert sl.get_params()["library"] is None
        # A weighted refit of the same object must work as on a fresh one.
        sl.fit(X, y, sample_weight=w)
        fresh = sp.SuperLearner(n_folds=3).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(sl.predict(X), fresh.predict(X), rtol=1e-12)


def test_ltmle_survival_mcar_censoring_and_standard_error():
    """Independent censoring at interval 2 leaves the estimand unchanged,
    and the influence-curve SE of S(2) tracks its Monte Carlo SD."""
    s11 = (1 - expit(-1.8)) * (1 - expit(-2.3))
    est, ses = [], []
    for r in range(40):
        df = _survival_panel(2000, 100 + r)
        rng = np.random.default_rng(500 + r)
        df["C1"] = 1
        df["C2"] = (rng.uniform(size=len(df)) > 0.2).astype(int)
        df.loc[df["C2"] == 0, "T2"] = np.nan  # not observed once censored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.ltmle_survival(
                df,
                event_indicators=["T1", "T2"],
                treatments=["A1", "A2"],
                covariates_time=[["L1"], ["L2"]],
                censoring=["C1", "C2"],
            )
        est.append(res.survival_treated[1])
        ses.append(res.detail["survival_se_treated"][1])
    est = np.asarray(est)
    # Mean over 40 replications: MC SE of the mean is about sd / sqrt(40).
    assert abs(est.mean() - s11) < 4 * est.std(ddof=1) / np.sqrt(len(est))
    ratio = np.mean(ses) / est.std(ddof=1)
    # 40 replications: the SD itself is known to about +-11%.
    assert 0.7 < ratio < 1.4, ratio


def test_ltmle_survival_risk_difference_is_curve_difference():
    df = _survival_panel(3000, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ltmle_survival(
            df,
            event_indicators=["T1", "T2"],
            treatments=["A1", "A2"],
            covariates_time=[["L1"], ["L2"]],
        )
    assert res.risk_difference_final == pytest.approx(
        res.survival_control[-1] - res.survival_treated[-1], abs=1e-14
    )
    assert res.rmst_treated == pytest.approx(res.survival_treated.sum(), abs=1e-14)
    # Survival never rises.
    assert np.all(np.diff(res.survival_treated) <= 1e-12)
    assert np.all(np.diff(res.survival_control) <= 1e-12)
