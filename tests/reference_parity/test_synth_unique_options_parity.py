"""Classical SCM options on a uniquely identified design.

Track A module 52 certifies ``sp.synth(method='classic')`` with equal
predictor weights on a design whose donor weights are unique: the treated
unit is exactly ``0.5 donor0 + 0.3 donor1 + 0.2 donor2`` in every
pre-period and the effect is 2 in every post-period. This file uses the
same hash-locked CSV for the configurations module 52 does not run.

* **Nested V search** with special predictors (five four-period means and
  two single periods). The treated unit is a convex combination of the
  donors in every predictor, so the zero-loss weights are unique for every
  V: the known truth is the true weights and a gap of 2 (T1). R ``Synth``'s
  own nested search on the same predictors
  (``_fixtures/_generate_synth_unique_R.R``) recovers the same, to 4e-14.
* **Regression V** (Stata ``synth``'s default) on the same predictors:
  the same known truth.
* **The placebo standard error under equal V.** ``sp.synth`` reports the
  population standard deviation of the in-space placebo effects as ``se``.
  Each placebo is simplex least squares on twenty pre-periods with five
  donors; every placebo's weights are checked against the KKT conditions
  of that problem and shown to be its unique optimum, and the placebo
  effects are compared with R ``Synth`` under equal unscaled predictor
  weights.
  Four placebos are interior optima and agree to 1e-12. One (donor0) puts
  all its weight on a single unit; R's interior-point QP solver ``ipop``
  cannot solve that vertex at its tight settings and stops at weight
  0.999998, so that placebo is held to the precision ipop attains there
  (1e-5) while the KKT certificate pins ours exactly. The resulting SD
  agrees with R's to 9e-7.

The placebo donor pool includes the actual treated unit, as in Stata
``synth2`` (``test_synth2_stata_parity.py`` pins the placebo pre-period
MSPEs against it).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_HERE = pathlib.Path(__file__).parent
_DATA = _HERE.parent / "r_parity" / "data" / "52_scm_unique.csv"
_REF = _HERE / "_fixtures" / "synth_unique_R.json"

W_TRUE = {"donor0": 0.5, "donor1": 0.3, "donor2": 0.2, "donor3": 0.0, "donor4": 0.0}
TAU = 2.0
T0 = 20
SPEC = [("y", list(range(a, a + 4)), "mean") for a in range(0, 20, 4)] + [
    ("y", 19, "mean"),
    ("y", 3, "mean"),
]
#: The CLAUDE.md section 5.1 budget.
RTOL = 1e-6


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_DATA)


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(_REF.read_text(encoding="utf-8"))


def _fit(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.synth(
            data,
            outcome="y",
            unit="region",
            time="year",
            treated_unit="treated",
            treatment_time=T0,
            method="classic",
            **kw,
        )


def _weights(res) -> dict:
    table = res.model_info["weights"].set_index("unit")["weight"]
    return {u: float(table.get(u, 0.0)) for u in W_TRUE}


@pytest.fixture(scope="module")
def equal(data):
    return _fit(data, v_method="equal", placebo=True)


@pytest.mark.parametrize("v_method", ["nested", "regression"])
def test_predictor_based_v_recovers_the_known_truth(data, v_method):
    res = _fit(data, v_method=v_method, special_predictors=SPEC, placebo=False)
    scope = sp.validation_scope(res)["configuration"]
    assert scope == {"v_method": v_method, "weights": "unique", "code_path": "native"}
    for unit, w in _weights(res).items():
        assert w == pytest.approx(W_TRUE[unit], abs=1e-9), unit
    assert res.estimate == pytest.approx(TAU, rel=1e-10)


def test_nested_v_matches_r_synth(data, ref):
    res = _fit(data, v_method="nested", special_predictors=SPEC, placebo=False)
    r = ref["nested"]
    assert r["sigf_ipop"] == 20
    for unit, w in _weights(res).items():
        assert w == pytest.approx(r["weights"][unit], abs=1e-9), unit
    assert res.estimate == pytest.approx(r["att"], rel=1e-9)


def _placebo_problem(data, unit):
    wide = data.pivot(index="year", columns="region", values="y")
    pre = wide.index < T0
    donors = [c for c in wide.columns if c != unit]
    return wide.loc[pre, unit].to_numpy(), wide.loc[pre, donors].to_numpy(), donors


def test_each_placebo_is_the_unique_simplex_least_squares_optimum(data, equal):
    """KKT, then uniqueness.

    Optimality: the gradient is constant on the support and no smaller off
    it. Uniqueness: the fitted pre-period path is unique (the objective is
    strictly convex in it), so another optimum would differ by a direction
    d with X d = 0 and sum(d) = 0 that keeps the weights non-negative. The
    placebos for donor3 and donor4 have such a direction in their donor
    matrix -- the actual treated unit is a combination of donor0-2 -- but
    it would push a zero weight negative either way, so the optimum is
    still unique.
    """
    table = equal.model_info["placebo_weights"]
    for unit in table.index:
        y, X, donors = _placebo_problem(data, unit)
        w = table.loc[unit, donors].to_numpy(dtype=float)
        assert w.min() >= -1e-12 and w.sum() == pytest.approx(1.0, abs=1e-12)
        grad = -2.0 * X.T @ (y - X @ w)
        scale = np.abs(grad).max() + np.abs(2.0 * X.T @ y).max()
        support = w > 1e-9
        lam = grad[support].mean()
        assert np.ptp(grad[support]) <= 1e-9 * scale, unit
        assert (grad[~support] - lam >= -1e-9 * scale).all(), unit
        # directions that leave the fit and the adding-up constraint alone
        A = np.vstack([X, np.ones(X.shape[1])])
        _, s, vt = np.linalg.svd(A)
        null = vt[(s > 1e-10 * s[0]).sum() :]
        assert len(null) <= 1, unit
        for d in null:
            for sign in (1.0, -1.0):
                assert (sign * d[~support] < -1e-8).any(), unit


def test_placebo_effects_match_r_synth(equal, ref):
    placebos = ref["placebo_equal_v"]["placebos"]
    atts = dict(
        zip(equal.model_info["placebo_weights"].index, equal.model_info["placebo_atts"])
    )
    table = equal.model_info["placebo_weights"]
    tight = [u for u, p in placebos.items() if p["sigf_ipop"] >= 10]
    assert len(tight) == 4
    for unit, p in placebos.items():
        # the interior optima to ipop's tight precision; the vertex one to
        # the precision ipop reaches at the setting that solves it
        rtol, atol = (1e-9, 1e-9) if unit in tight else (1e-5, 1e-5)
        assert atts[unit] == pytest.approx(p["att"], rel=rtol), unit
        for donor, w in p["weights"].items():
            assert float(table.loc[unit, donor]) == pytest.approx(w, abs=atol)


def test_se_is_the_population_sd_of_the_placebo_effects(equal, ref):
    atts = np.asarray(equal.model_info["placebo_atts"], dtype=float)
    assert equal.se == pytest.approx(float(np.std(atts)), rel=1e-14)
    assert equal.se == pytest.approx(ref["placebo_equal_v"]["sd_population"], rel=RTOL)
    assert sp.validation_scope(equal)["configuration"]["v_method"] == "equal"
