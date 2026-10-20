"""``sp.weightit(method='energy')`` against R ``WeightIt`` 1.7.0.

WeightIt is GPL: its outputs on a shared fixture are the reference
(``_fixtures/_generate_weightit_energy_R.R``), its code was not read.
WeightIt solves the quadratic programme with OSQP and polishes the
solution; ours is solved exactly on the optimality conditions. Where the
programme is well conditioned the weights agree to 1e-9 (OSQP's polish
reaches ~3e-11 here). On the flat case (indicator covariates, Euclidean
distance) the two weight vectors are several hundredths apart while the
objective values agree to 1e-11 -- ours is the lower one -- so that case
asserts optimality, not equal weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching import _energy

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "weightit_energy_R.json").read_text(encoding="utf-8"))
CASES = REF["cases"]


def _data() -> pd.DataFrame:
    d = pd.read_csv(FIX / "weightit_cobalt_data.csv")
    a, b = d.x1 > 0, d.x2 > 0.5
    d["g2"] = ((a & ~b) | (~a & b)).astype(float)
    d["g3"] = (a & b).astype(float)
    return d


def _kwargs(case: dict) -> dict:
    args = case.get("args") or {}
    kw = {}
    if "improved" in args:
        kw["improved"] = bool(args["improved"])
    if "lambda" in args:
        kw["lam"] = float(args["lambda"])
    if "dist.mat" in args:
        kw["dist"] = args["dist.mat"]
    if "moments" in args:
        kw["moments"] = int(args["moments"])
    if "int" in args:
        kw["interactions"] = bool(args["int"])
    if "min.w" in args:
        kw["min_w"] = float(args["min.w"])
    if case.get("focal") is not None:
        kw["focal"] = case["focal"]
    return kw


def _fit(name: str):
    case = CASES[name]
    return sp.weightit(
        _data(),
        case["treat"],
        case["covs"],
        method="energy",
        estimand=case["estimand"],
        **_kwargs(case),
    )


@pytest.mark.parametrize("name", [k for k in CASES if k != "bin_ATE_flat"])
def test_energy_weights_match_weightit(name):
    w = _fit(name).weights.to_numpy()
    np.testing.assert_allclose(w, CASES[name]["w"], rtol=0, atol=1e-9)


def test_flat_programme_ours_is_at_least_as_good(monkeypatch):
    seen = {}
    orig = _energy.solve_qp

    def spy(P, c, A, b, lb):
        seen.update(P=P, c=c, A=A, b=b, lb=lb)
        return orig(P, c, A, b, lb)

    monkeypatch.setattr(_energy, "solve_qp", spy)
    ours = _fit("bin_ATE_flat").weights.to_numpy()
    theirs = np.asarray(CASES["bin_ATE_flat"]["w"])

    def objective(w):
        return 0.5 * w @ seen["P"] @ w + seen["c"] @ w

    # WeightIt's weights are feasible for the same programme ...
    assert np.abs(seen["A"] @ theirs - seen["b"]).max() < 1e-9
    assert theirs.min() >= seen["lb"].min() - 1e-12
    # ... and no better than ours: the objective is the same to 1e-10
    gap = objective(theirs) - objective(ours)
    assert -1e-12 <= gap < 1e-10 * abs(objective(ours))


def test_solution_satisfies_the_optimality_conditions():
    # Identity: at the optimum the gradient of the Lagrangian vanishes on
    # the free weights and is non-negative on the weights at the bound.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 3))
    G = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
    seen = {}
    orig = _energy.solve_qp

    def spy(P, c, A, b, lb):
        w = orig(P, c, A, b, lb)
        seen.update(P=P, c=c, A=A, b=b, lb=lb, w=w)
        return w

    _energy.solve_qp = spy
    try:
        _energy.energy_weights(X, G, 2, "ATE", None, lam=0.0)
    finally:
        _energy.solve_qp = orig
    P, c, A, b, lb, w = (seen[k] for k in ("P", "c", "A", "b", "lb", "w"))
    grad = P @ w + c
    free = w > lb + 1e-12
    nu = np.linalg.lstsq(A[:, free].T, -grad[free], rcond=None)[0]
    lag = grad + A.T @ nu
    scale = np.abs(grad).max()
    assert np.abs(lag[free]).max() < 1e-10 * scale
    assert lag[~free].min() > -1e-10 * scale
    assert np.abs(A @ w - b).max() < 1e-10


def test_primal_active_set_fallback_reaches_the_same_optimum():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(90, 2))
    G = rng.integers(0, 3, size=90)
    seen = {}
    orig = _energy.solve_qp

    def spy(P, c, A, b, lb):
        seen.update(P=P, c=c, A=A, b=b, lb=lb)
        return orig(P, c, A, b, lb)

    _energy.solve_qp = spy
    try:
        w = _energy.energy_weights(X, G, 3, "ATE", None)
    finally:
        _energy.solve_qp = orig
    w2 = _energy._primal_active_set(
        seen["P"], seen["c"], seen["A"], seen["b"], seen["lb"]
    )
    np.testing.assert_allclose(w, w2, atol=1e-8)


def test_exact_mean_balance_with_moments():
    W = _fit("multi_ATE_moments2_int")
    d = _data()
    w = W.weights.to_numpy()
    for g in ("a", "b", "c"):
        m = (d.treat == g).to_numpy()
        for col in ("x1", "x2", "x3"):
            wm = np.sum(w[m] * d[col][m]) / np.sum(w[m])
            assert abs(wm - d[col].mean()) < 1e-9


def test_energy_refuses_continuous_treatment_and_bad_options():
    d = _data()
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.weightit(d, "dose", ["x1", "x2"], method="energy")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.weightit(d, "bin", ["x1", "x2"], method="energy", dist="manhattan")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.weightit(d, "bin", ["x1", "x2"], method="energy", lam=-1.0)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.weightit(d, "bin", ["x1", "x2"], method="energy", estimand="ATO")
