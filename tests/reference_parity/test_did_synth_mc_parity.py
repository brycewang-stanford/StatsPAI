"""Nuclear-norm matrix completion vs the R implementations (fixed lambda).

Functions: ``sp.mc_panel`` (staggered panel) and ``sp.mc_synth`` (single
treated unit). Both call the shared solver
``statspai.matrix_completion._core.mc_nnm_fit``; they differ only in which
cells are masked, the lambda default and the inference layer.

Reference: ``_fixtures/did_synth_mc_R.json`` written by
``_generate_did_synth_mc_R.R`` from

* ``MCPanel::mcnnm_fit`` -- the authors' implementation
  (github.com/susanathey/MCPanel; commit SHA and build note in ``meta``),
* ``fect::fect(method = "mc", CV = FALSE)`` -- and ``gsynth(estimator =
  "mc")``, which in gsynth 1.4.0 is a wrapper that calls ``fect``,

on ``_fixtures/did_synth_mc_panel.csv`` (40 x 25 balanced panel, written by
``_generate_did_synth_mc_data.py``; both sides read the same bytes).

What is compared and the conventions each number depends on
-----------------------------------------------------------
* One minimiser, three lambda scales. StatsPAI's ``lambda_reg`` = theta is
  the singular-value threshold on ``1/2 ||P_O(Y - F)||^2 + theta ||L||_*``.
  MCPanel minimises ``(1/|O|) ||P_O(Y - F)||^2 + lambda_L ||L||_*``, so
  ``lambda_L = 2 theta / |O|``; fect thresholds the singular values of
  ``E / (N T)``, so ``lambda = theta / (N T)``. The fixture stores all three.
* Fixed effects: ``fixed_effects='two-way' | 'unit' | 'time' | 'none'`` is
  MCPanel ``(to_estimate_u, to_estimate_v)`` = (1,1), (1,0), (0,1), (0,0).
  fect ``force = "two-way"`` is the two-way case; fect ``force = "none"``
  keeps an unpenalised grand mean, which no StatsPAI option reproduces, so
  it is not compared.
* Compared quantities: the fitted untreated matrix ``F`` (fixed effects +
  low-rank part) on every cell and the ATT (mean of ``Y - F`` over the
  masked cells). MCPanel's ``u`` / ``v`` split of the fixed effects and its
  ``L`` are not identified separately (a constant can move between them,
  and L need not be double-centred), so they are not compared.
* Convergence: the references are run far past their default stopping
  rules (MCPanel ``rel_tol = 0`` with 3000 sweeps per lambda; fect
  ``tol = 1e-15``) and StatsPAI with ``tol = 1e-14``, so every side reports
  the minimiser. Observed agreement is ~1e-12 relative; the assertion is
  ``1e-9`` (T2 "bit-exact" band): iterative solvers with different update
  orders (EM vs coordinate descent) cannot share every last bit.
* Not compared: ``se`` (StatsPAI unit bootstrap for ``mc_panel``, placebo
  spread for ``mc_synth``; neither reference reports those quantities) and
  the data-driven default lambda (StatsPAI heuristic / own CV vs MCPanel's
  and fect's CV rules with their own fold RNGs).

Reference-free check: the KKT conditions of the convex problem are
asserted directly on StatsPAI's solution (zero residual row/column sums for
the fixed effects, ``U' R V = theta I`` on the active subspace and
``||R_perp||_2 <= theta`` off it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matrix_completion._core import mc_nnm_fit

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "did_synth_mc_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "did_synth_mc_panel.csv")

TOL = 1e-9
SOLVER_TOL = 1e-14
THETAS = [8, 20]
FE_KEY = {"two-way": "twoway", "unit": "unit", "time": "time", "none": "none"}
SYN_UNIT = R["synth"]["treated_unit"]
SYN_T0 = R["synth"]["treatment_time"]


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _panel(theta, fe):
    return sp.mc_panel(
        D,
        y="y",
        unit="unit",
        time="time",
        treat="d",
        lambda_reg=float(theta),
        fixed_effects=fe,
        tol=SOLVER_TOL,
        max_iter=20000,
        n_bootstrap=2,
    )


def _synth(theta, fe):
    return sp.mc_synth(
        D,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=SYN_UNIT,
        treatment_time=SYN_T0,
        lambda_reg=float(theta),
        fixed_effects=fe,
        tol=SOLVER_TOL,
        max_iter=20000,
        placebo=False,
    )


# --------------------------------------------------------------------------
# sp.mc_panel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("fe", ["two-way", "unit", "time", "none"])
def test_mc_panel_matches_mcpanel(theta, fe):
    ref = R["panel"][f"theta_{theta}"][FE_KEY[fe]]
    res = _panel(theta, fe)
    assert res.model_info["converged"]
    assert _rel(res.estimate, ref["att"]) < TOL
    assert _rel(res.model_info["completed_matrix"], ref["fit"]) < TOL
    # lambda mapping: StatsPAI theta -> MCPanel lambda_L = 2 theta / |O|
    assert res.model_info["lambda_mcpanel"] == pytest.approx(ref["lambda_L"], rel=1e-15)
    assert res.model_info["n_control_cells"] == ref["n_obs"]


@pytest.mark.parametrize("theta", THETAS)
def test_mc_panel_matches_fect_two_way(theta):
    ref = R["panel"][f"theta_{theta}"]["fect_twoway"]
    res = _panel(theta, "two-way")
    assert _rel(res.estimate, ref["att"]) < TOL
    assert _rel(res.model_info["completed_matrix"], ref["fit"]) < TOL
    assert res.model_info["lambda_fect"] == pytest.approx(ref["lambda"], rel=1e-15)


def test_mc_panel_matches_gsynth_mc():
    res = _panel(8, "two-way")
    assert _rel(res.estimate, R["panel"]["gsynth_theta_8_att"]) < TOL


def test_mc_panel_default_is_two_way():
    base = dict(
        y="y", unit="unit", time="time", treat="d", lambda_reg=8.0, n_bootstrap=2
    )
    default = sp.mc_panel(D, **base)
    explicit = sp.mc_panel(D, fixed_effects="two-way", **base)
    assert default.model_info["fixed_effects"] == "two-way"
    assert default.estimate == explicit.estimate
    # default tol (1e-10) already lands on the reference minimiser
    assert _rel(default.estimate, R["panel"]["theta_8"]["twoway"]["att"]) < 1e-8


def test_mc_panel_lambda_is_used():
    a = R["panel"]["theta_8"]["twoway"]["att"]
    b = R["panel"]["theta_20"]["twoway"]["att"]
    assert abs(a - b) > 1e-3  # the two fixture lambdas are distinguishable
    assert _rel(_panel(20, "two-way").estimate, b) < TOL


# --------------------------------------------------------------------------
# sp.mc_synth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("fe", ["two-way", "none"])
def test_mc_synth_matches_mcpanel(theta, fe):
    ref = R["synth"][f"theta_{theta}"][FE_KEY[fe]]
    res = _synth(theta, fe)
    assert res.model_info["converged"]
    assert _rel(res.estimate, ref["att"]) < TOL
    assert _rel(res.model_info["completed_matrix"], ref["fit"]) < TOL
    assert res.model_info["lambda_mcpanel"] == pytest.approx(ref["lambda_L"], rel=1e-15)


@pytest.mark.parametrize("theta", THETAS)
def test_mc_synth_matches_fect_two_way(theta):
    ref = R["synth"][f"theta_{theta}"]["fect_twoway"]
    res = _synth(theta, "two-way")
    assert _rel(res.estimate, ref["att"]) < TOL
    assert _rel(res.model_info["completed_matrix"], ref["fit"]) < TOL
    assert res.model_info["lambda_fect"] == pytest.approx(ref["lambda"], rel=1e-15)


def test_mc_synth_and_mc_panel_share_the_solver():
    """Same mask, same lambda -> identical fit from the two front ends."""
    d = D.copy()
    d["d1"] = ((d["unit"] == SYN_UNIT) & (d["time"] >= SYN_T0)).astype(int)
    p = sp.mc_panel(
        d,
        y="y",
        unit="unit",
        time="time",
        treat="d1",
        lambda_reg=8.0,
        tol=SOLVER_TOL,
        max_iter=20000,
        n_bootstrap=2,
    )
    s = _synth(8, "two-way")
    assert _rel(s.estimate, p.estimate) < 1e-10


# --------------------------------------------------------------------------
# Reference-free: KKT conditions of the convex problem
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fe", ["two-way", "none"])
def test_kkt_conditions(fe):
    Y = D.pivot(index="unit", columns="time", values="y").to_numpy()
    W = D.pivot(index="unit", columns="time", values="d").to_numpy()
    obs = W == 0
    theta = 8.0
    sol = mc_nnm_fit(Y, obs, theta, fixed_effects=fe, max_iter=20000, tol=SOLVER_TOL)
    R_ = np.where(obs, Y - sol["fit"], 0.0)
    if fe == "two-way":
        # stationarity in the unpenalised unit / time effects
        assert np.max(np.abs(R_.sum(axis=1))) < 1e-8
        assert np.max(np.abs(R_.sum(axis=0))) < 1e-8
    # R must lie in theta * subdifferential of ||.||_* at L
    U, s, Vt = np.linalg.svd(sol["L"], full_matrices=False)
    k = int(np.sum(s > 1e-8 * s[0]))
    assert k >= 1
    Uk, Vk = U[:, :k], Vt[:k].T
    np.testing.assert_allclose(Uk.T @ R_ @ Vk, theta * np.eye(k), atol=1e-7)
    Pu = np.eye(Y.shape[0]) - Uk @ Uk.T
    Pv = np.eye(Y.shape[1]) - Vk @ Vk.T
    assert np.linalg.norm(Pu @ R_ @ Pv, 2) <= theta * (1 + 1e-8)


# --------------------------------------------------------------------------
# Fail loudly
# --------------------------------------------------------------------------


def test_non_convergence_warns():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = sp.mc_panel(
            D,
            y="y",
            unit="unit",
            time="time",
            treat="d",
            lambda_reg=8.0,
            max_iter=2,
            n_bootstrap=2,
        )
    assert res.model_info["converged"] is False


def test_bad_fixed_effects_raises():
    with pytest.raises(ValueError, match="fixed_effects"):
        sp.mc_panel(
            D,
            y="y",
            unit="unit",
            time="time",
            treat="d",
            fixed_effects="twoway",
            n_bootstrap=2,
        )
    with pytest.raises(ValueError, match="fixed_effects"):
        sp.mc_synth(
            D,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit=SYN_UNIT,
            treatment_time=SYN_T0,
            fixed_effects="both",
            placebo=False,
        )


def test_fixture_meta_recorded():
    meta = R["meta"]
    assert meta["MCPanel_sha"] != "unknown"
    for k in ("r_version", "MCPanel", "fect", "gsynth", "RcppEigen"):
        assert meta[k]
    assert meta["N"] * meta["T"] == len(D)
