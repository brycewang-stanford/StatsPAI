"""TVP-VAR with stochastic volatility: known truth and a reference screen.

Nothing here is a parity. The estimator is a Gibbs sampler; its exact
checks (each block against a closed form, the sweep against the joint
distribution) are in ``tests/test_tvp_var_sv.py``. This file adds

* known truth (T1): on the committed simulated file, whose own-lag
  coefficient drifts from 0.2 to 0.8 and whose first shock doubles its
  standard deviation halfway, the posterior bands cover the true paths
  and the medians track them;
* a stochastic screen (S, not T3): posterior medians against
  ``bvarsv::bvar.sv.tvp`` 1.1 on the same file with the same prior
  constants. Four seeds on the R side, one short chain here; the bounds
  are several times the differences seen with long chains (reported at
  each assert) and are not equivalence tests.

Regenerate the inputs with ``_fixtures/_generate_tvp_var_sv_data.py`` and
``_fixtures/_generate_tvp_var_sv_R.R``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries.tvp_var_sv import TVPVARSVResult, tvp_var_sv

FIX = Path(__file__).parent / "_fixtures"
TAU = 40


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIX / "tvp_var_sv.csv")


@pytest.fixture(scope="module")
def reference() -> dict:
    out: dict = json.loads((FIX / "tvp_var_sv_R.json").read_text(encoding="utf-8"))
    return out


def _fit(data: pd.DataFrame, **kw: float) -> TVPVARSVResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        return tvp_var_sv(
            data,
            variables=["x", "z"],
            lags=1,
            training=TAU,
            draws=800,
            burnin=700,
            seed=7,
            **kw,
        )


@pytest.fixture(scope="module")
def fit_loose(data: pd.DataFrame) -> TVPVARSVResult:
    # k_Q = 0.1: a prior that allows the amount of drift in the data
    return _fit(data, k_Q=0.1)


@pytest.fixture(scope="module")
def fit_default(data: pd.DataFrame) -> TVPVARSVResult:
    return _fit(data)


# --------------------------------------------------------------------------
# Known truth
# --------------------------------------------------------------------------
def test_volatility_path_recovers_the_doubling(
    data: pd.DataFrame, fit_loose: TVPVARSVResult
) -> None:
    truth = data["sd_x"].to_numpy()[TAU:]
    v = fit_loose.volatility()
    vx = v[v["variable"] == "x"]
    med = vx["median"].to_numpy()
    cover = (
        (vx["lower"].to_numpy() <= truth) & (truth <= vx["upper"].to_numpy())
    ).mean()
    # long chains (1500 draws, two seeds): coverage 0.945 and 0.96 of the
    # 95% band, RMSE 0.18; the misses are at the break, which a random
    # walk smooths. Bounds with headroom for the short chain.
    assert cover >= 0.85
    assert np.sqrt(np.mean((med - truth) ** 2)) < 0.30
    # first 60 dates sd 1, last 60 dates sd 2 (long chains: 0.99 and 1.96)
    assert 0.8 < med[:60].mean() < 1.2
    assert 1.6 < med[-60:].mean() < 2.4
    # the second shock has constant sd 0.7 (long chains: 0.66 .. 0.73)
    vz = v[v["variable"] == "z"]
    assert 0.55 < vz["median"].min() and vz["median"].max() < 0.9
    assert ((vz["lower"] <= 0.7) & (0.7 <= vz["upper"])).mean() >= 0.9


def test_coefficient_path_recovers_the_drift(
    data: pd.DataFrame, fit_loose: TVPVARSVResult
) -> None:
    truth = data["a11"].to_numpy()[TAU:]
    c = fit_loose.coefficients()
    c = c[(c["equation"] == "x") & (c["term"] == "L1.x")]
    med = c["median"].to_numpy()
    cover = ((c["lower"].to_numpy() <= truth) & (truth <= c["upper"].to_numpy())).mean()
    # long chains: coverage 1.0, RMSE 0.10, medians 0.10 -> 0.73 between
    # the first and the last 20 dates (truth 0.23 -> 0.77)
    assert cover >= 0.9
    assert np.sqrt(np.mean((med - truth) ** 2)) < 0.20
    assert med[-20:].mean() - med[:20].mean() > 0.35
    # the other coefficients are constant: 0.1, 0 / 0, 0.3, 0
    rest = fit_loose.coefficients()
    for eq, term, value in (("x", "L1.z", 0.1), ("z", "L1.z", 0.3), ("z", "L1.x", 0.0)):
        part = rest[(rest["equation"] == eq) & (rest["term"] == term)]
        assert ((part["lower"] <= value) & (value <= part["upper"])).mean() >= 0.9


def test_default_prior_shrinks_the_drift_away(
    data: pd.DataFrame, fit_default: TVPVARSVResult
) -> None:
    # k_Q = 0.01 is a strong prior for constant coefficients: on these
    # data the posterior median stays near 0.5 at every date, in both
    # implementations (bvarsv: 0.49 .. 0.51). Pinned so that a change of
    # the default is noticed; it is the reason the tests above raise k_Q.
    c = fit_default.coefficients()
    med = c[(c["equation"] == "x") & (c["term"] == "L1.x")]["median"].to_numpy()
    assert med.max() - med.min() < 0.1
    assert 0.4 < med.mean() < 0.6


# --------------------------------------------------------------------------
# Screen against bvarsv (S: report-level bounds, not an equivalence test)
# --------------------------------------------------------------------------
def test_screen_against_bvarsv(reference: dict, fit_default: TVPVARSVResult) -> None:
    runs = reference["runs"]
    assert reference["versions"]["bvarsv"] == "1.1"
    # bvarsv estimates on rows tau + p + 1 .. n, one date fewer
    n_ref = reference["n_dates"]
    assert n_ref == fit_default.n_obs - 1
    sd_ref = np.array([r["sd_median"] for r in runs])
    sd_seed = sd_ref.std(axis=0, ddof=1).mean()
    sd = np.median(np.exp(fit_default.logsig_draws.astype(float)), axis=0)[1:]
    rel = np.abs(sd / sd_ref.mean(axis=0) - 1.0)
    # with four long chains on each side the seed-mean paths differ by at
    # most 3.2% (x) and 4.1% (z), mean 1.3%; the R seeds differ among
    # themselves by 1.2% on average. Bound: 12% anywhere, 5% on average.
    assert sd_seed < 0.03
    assert rel.max() < 0.12, rel.max()
    assert rel.mean() < 0.05, rel.mean()
    a_ref = np.array([r["a11_median"] for r in runs]).mean(axis=0)
    a = np.median(fit_default.coef_draws[:, 1:, 0, 0], axis=0)
    # long chains: 0.009 at most; posterior sd of the coefficient 0.08
    assert np.abs(a - a_ref).max() < 0.05
    for t in (40, 180):
        d = fit_default.irf_draws(at=t, periods=6)
        for j, name in ((0, "x"), (1, "z")):
            ref = np.array([r[f"irf_x_to_{name}_t{t}"] for r in runs]).mean(axis=0)
            got = np.median(d[:, 1:, j, 0], axis=0)
            # long chains: 0.019 at most (response of x at t = 180, where
            # the shock sd is 2); the responses themselves are up to 1.0
            assert np.abs(got - ref).max() < 0.08, (t, name)
