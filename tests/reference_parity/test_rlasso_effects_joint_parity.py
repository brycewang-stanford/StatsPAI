"""Joint inference for ``sp.rlasso_effects`` against ``hdm::rlassoEffects``.

``confint(<rlassoEffects>, joint = TRUE)`` draws from a normal with
covariance ``Omega / n`` and takes a quantile of the largest absolute
t-ratio. Two things are compared, at different levels:

* ``Omega / n`` is deterministic: 1e-10 relative (T2), both methods.
* The critical value is a Monte Carlo quantile. hdm uses 500 draws, so its
  value carries a simulation standard error of about 0.05; StatsPAI uses
  100,000 draws. The comparison is a screen (S), not parity: the two must
  agree within 0.2, and StatsPAI's value must sit between the pointwise
  normal quantile and the Bonferroni one, which bound any sup-t value.

Fixture: ``_generate_rlasso_effects_joint.R`` on the simulated
``rlasso_effect.csv``.

References
----------
[@chernozhukov2016hdm], [@belloni2014inference]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def reference() -> dict:
    path = FIX / "rlasso_effects_joint_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    D = pd.read_csv(FIX / "rlasso_effect.csv")
    return D.drop(columns="y"), D["y"]


@pytest.mark.parametrize("method", ["partialling out", "double selection"])
def test_joint_covariance_matches_hdm(data, reference, method):
    X, y = data
    ref = reference[method.replace(" ", "_")]
    out = sp.rlasso_effects(X, y, index=reference["index"], method=method)
    assert list(out) == ref["names"]
    np.testing.assert_allclose([r.alpha for r in out.values()], ref["coef"], rtol=1e-10)
    np.testing.assert_allclose(out.vcov().to_numpy(), ref["omega_over_n"], rtol=1e-10)


@pytest.mark.parametrize("method", ["partialling out", "double selection"])
def test_joint_band_critical_value(data, reference, method):
    X, y = data
    ref = reference[method.replace(" ", "_")]
    out = sp.rlasso_effects(X, y, index=reference["index"], method=method)
    band = out.conf_int(joint=True)
    crit = band.attrs["critical_value"]
    k = len(out)
    assert stats.norm.ppf(0.975) < crit <= stats.norm.ppf(1 - 0.025 / k) + 1e-3
    assert abs(crit - ref["critical_value_B500"]) < 0.2  # screen, see docstring
    half = (band["upper"] - band["lower"]) / 2
    np.testing.assert_allclose(half, crit * np.sqrt(np.diag(out.vcov())), rtol=1e-12)


def test_pointwise_intervals_are_the_single_target_ones(data, reference):
    X, y = data
    out = sp.rlasso_effects(X, y, index=reference["index"])
    tab = out.conf_int()
    for name, res in out.items():
        lo, hi = res.conf_int()
        assert tab.loc[name, "lower"] == pytest.approx(lo, rel=1e-12)
        assert tab.loc[name, "upper"] == pytest.approx(hi, rel=1e-12)
    assert isinstance(out, dict) and "estimate" in out.summary()


def test_joint_band_covers_at_the_nominal_rate():
    """Simultaneous coverage of three true effects, 300 replications.

    Nominal 95%; a binomial 3-sd band around 0.95 at 300 replications is
    +/- 0.038, and the pointwise intervals should cover jointly clearly
    less often.
    """
    truth = np.array([1.0, -0.5, 0.0])
    joint_hits = point_hits = 0
    reps = 300
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        X = rng.standard_normal((200, 10))
        X[:, 1] += 0.5 * X[:, 0]
        y = X[:, :3] @ truth + 0.5 * X[:, 5] + rng.standard_normal(200)
        out = sp.rlasso_effects(X, y, index=[0, 1, 2])
        jb = out.conf_int(joint=True, n_draws=5000, seed=r)
        pb = out.conf_int()
        joint_hits += bool(((jb["lower"] <= truth) & (truth <= jb["upper"])).all())
        point_hits += bool(((pb["lower"] <= truth) & (truth <= pb["upper"])).all())
    assert 0.95 - 0.045 <= joint_hits / reps <= 0.995
    assert point_hits < joint_hits
