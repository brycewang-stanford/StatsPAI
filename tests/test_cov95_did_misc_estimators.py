"""Coverage campaign — further ``statspai.did`` estimators not in the cov95 set.

Covers:
- ``did_misclassified`` (DiD with misclassified treatment timing, incl. the
  bias-correction path ``pi_misclass > 0`` and clustering);
- ``cohort_anchored_event_study`` (cohort-anchored event study);
- ``overlap_weighted_did`` (overlap-weighted 2×2 DiD with a propensity model);
- ``ddd_heterogeneous`` (heterogeneous triple-difference).

DGP: constant +2 treatment effect from each cohort's adoption date.
Assertions check recovery of the known effect (where the estimand is the ATT)
or finite, well-formed output, never fabricated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 2.0


def _staggered(seed=0, n_units=120, n_periods=8, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        x1 = rng.normal()
        sub = int(u % 2 == 0)  # binary subgroup
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            te = att if d else 0.0
            y = fe + 0.3 * t + 0.5 * x1 + te + rng.normal(0, 0.4)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": y,
                    "g": g,
                    "d": d,
                    "x1": x1,
                    "sub": sub,
                    "cl": u % 10,
                }
            )
    return pd.DataFrame(rows)


def _two_by_two(seed=1, n=500, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    rows = []
    for i in range(n):
        for tt in (0, 1):
            te = att if (grp[i] == 1 and tt == 1) else 0.0
            rows.append(
                {
                    "id": i,
                    "treat": grp[i],
                    "time": tt,
                    "y": 1 + 0.5 * tt + 0.4 * x1[i] + te + rng.normal(0, 0.5),
                    "x1": x1[i],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _staggered()


def _est(result):
    for attr in ("estimate", "att", "overall_att"):
        v = getattr(result, attr, None)
        if v is not None:
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(arr.ravel()[0])
    raise AssertionError("no finite estimate")


def test_did_misclassified_baseline(panel):
    # pi_misclass=0 → standard DiD, recovers the true ATT.
    r = sp.did_misclassified(panel, y="y", treat="g", time="time", id="unit")
    assert abs(_est(r) - TRUE_ATT) < 0.6


def test_did_misclassified_biascorrected_and_cluster(panel):
    # pi_misclass>0 engages the misclassification bias-correction path.
    r = sp.did_misclassified(
        panel, y="y", treat="g", time="time", id="unit", pi_misclass=0.1, cluster="cl"
    )
    assert r is not None and np.isfinite(_est(r))


def test_cohort_anchored_event_study(panel):
    r = sp.cohort_anchored_event_study(
        panel, y="y", treat="g", time="time", id="unit", leads=3, lags=3, cluster="cl"
    )
    assert r is not None
    assert np.isfinite(_est(r))


def test_overlap_weighted_did():
    df = _two_by_two()
    r = sp.overlap_weighted_did(
        df, y="y", treat="treat", time="time", covariates=["x1"]
    )
    assert abs(_est(r) - TRUE_ATT) < 0.7


def test_ddd_heterogeneous(panel):
    r = sp.ddd_heterogeneous(
        panel,
        y="y",
        unit="unit",
        time="time",
        cohort="g",
        subgroup="sub",
        never_value=0,
        n_boot=100,
    )
    assert r is not None
    assert np.isfinite(_est(r))
