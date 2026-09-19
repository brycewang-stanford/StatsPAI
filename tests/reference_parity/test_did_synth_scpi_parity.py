"""``sp.scdata`` / ``sp.scest`` / ``sp.scpi`` vs R ``scpi`` 4.0.1.

Reference: ``_fixtures/did_synth_scpi_R.json`` written by
``_generate_did_synth_scpi_R.R`` (scpi 4.0.1, CVXR 1.9.2 with CLARABEL,
ECOSolveR 0.6.1, Qtools 1.6.0 / quantreg 6.1) on two panels whose bytes both
sides read:

* ``did_synth_scpi_germany.csv`` -- scpi's own vignette panel
  ``scpi::scpi_germany`` (West Germany, 16 donors, T0 = 31, T1 = 13);
* ``did_synth_scpi_california.csv`` -- ``sp.california_prop99()``
  (38 donors > T0 = 19: Z'Z is singular).

Configuration on both sides: features = outcome only, no ``cov.adj``,
``constant = FALSE``, ``V = "separate"`` (identity), ``effect = "unit-time"``,
``u.missp = TRUE``, ``u.sigma = "HC1"``, ``u.order = 1``, ``u.lags = 0``,
``e.order = 1``, ``e.lags = 0``, ``rho = "type-2"``, ``rho.max = 0.2``,
``u.alpha = e.alpha = 0.05``.

What agrees at what tolerance, and why
--------------------------------------
* ``scdata`` matrices A, B, P: identical (0 difference); donor order = R's
  ``sort(B.names)``.
* ``scest`` weights.  StatsPAI solves each weight problem exactly (active set /
  secular equation / lasso path + KKT polish); R hands them to CLARABEL
  (OSQP for lasso) whose default gap tolerances are 1e-8.  Not a tuned
  tolerance but a proof: R's weights are feasible, their objective exceeds
  ours by <= 1e-7 relative (the solver gap), and strong convexity gives
  ``||w_R - w_py||_2 <= sqrt((f(w_R) - f(w_py)) / lambda_min(B'B))``.
  OLS and lasso (closed form / OSQP polish) agree to 1e-9.  The data-driven
  ridge radius Q and lambda agree to 1e-10.
* ``scpi`` deterministic ingredients, given R's weights: rho, Q.star, u.mean,
  u.var (Omega), Sigma, T / parameter counts, e.mean -- 1e-10 relative.
* Out-of-sample moments go through ``Qtools::rrq`` (restricted regression
  quantiles).  R's default fits them with quantreg's Frisch-Newton interior
  point, which stops at a 1e-6 duality gap; StatsPAI solves the LPs exactly.
  Against R re-run with quantreg's exact simplex method (``method = "br"``) the
  e.var and gaussian / ls / qreg bounds agree to 1e-9; against the default
  run they differ by the interior-point error (<= 5e-4 absolute here).
* In-sample simulation: fed R's own N(0,1) draws, every (draw, horizon,
  side) problem is solved exactly (closed form per active set); R uses ECOS
  at its 1e-8 default tolerances on an ill-conditioned problem (cond(Z'Z) ~
  2e6).  Against ECOS re-run at 1e-12 on the same draws the per-draw values
  agree to a median 1e-7 and the quantile bounds to 1e-5; against default
  ECOS, per-draw values differ by up to 6e-3 and the bounds by up to 2e-3
  absolute.  An independent solver (SLSQP) reproduces StatsPAI's per-draw
  solutions to 1e-8, and for ``ols`` they equal the analytic closed form.
* The simulated bounds with StatsPAI's own RNG are Monte Carlo (T3) relative
  to R's; only the draw-fed comparison is a parity statement.
* End-to-end (``sp.scpi`` with its own weights): identical to the draw-fed
  result except for ``simplex``, where R's CLARABEL leaves Norway at
  2.0e-6 >= the 1e-6 threshold scpi uses to count active donors (d0, and df
  of HC1), so R's rho and Sigma use one active donor more than the exact
  solution has.  R's own ``L1-L2`` fit -- the same optimum, its L2 bound is
  slack -- leaves Norway at 2.9e-7 and gets StatsPAI's rho.  Graded as a
  reference-solver artefact (T4), not parity.
* The average-effect interval (``.ci``) is R's ``scdataMulti(effect =
  "unit")`` construction; draw-fed it agrees at the in-sample ECOS tolerance.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth import _scpi_inference as inf_mod
from statspai.synth import _scpi_solvers as sv
from statspai.synth.scpi import _constraint_spec, scpi_inference

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "did_synth_scpi_R.json").read_text(encoding="utf-8"))
GER = pd.read_csv(_FIX / "did_synth_scpi_germany.csv")
CAL = pd.read_csv(_FIX / "did_synth_scpi_california.csv")
KW = dict(
    outcome="gdp",
    unit="country",
    time="year",
    treated_unit="West Germany",
    treatment_time=1991,
)
KW_CAL = dict(
    outcome="packspercapita",
    unit="state",
    time="year",
    treated_unit="California",
    treatment_time=1989,
)
CONSTR = {
    "simplex": "simplex",
    "lasso": "lasso",
    "ridge": "ridge",
    "ols": "ols",
    "L1L2": "L1-L2",
}
TIGHT = 1e-10
SC = sp.scdata(GER, **KW)


def _arr(x):
    return np.asarray(x, dtype=float)


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(_arr(ours), _arr(ref), rtol=rtol, atol=atol)


_CACHE: dict = {}


def _inference_on_R_weights(nm):
    """StatsPAI's inference layer on R's weights and R's draws."""
    if nm not in _CACHE:
        p = R[nm]["scpi"]
        spec = _constraint_spec(SC["A"], SC["B"], CONSTR[nm])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _CACHE[nm] = scpi_inference(
                SC["A"],
                SC["B"],
                SC["P"],
                _arr(p["w"]),
                spec,
                sims=p["sims"],
                draws=_arr(p["zraw"]),
            )
    return _CACHE[nm]


# --------------------------------------------------------------------------
# scdata
# --------------------------------------------------------------------------


def test_scdata_matrices_identical():
    ref = R["scdata"]
    assert SC["donor_names"] == ref["donors"]
    assert (SC["J"], SC["KM"], SC["T0"], SC["T1"]) == (
        ref["J"],
        ref["KM"],
        ref["T0"],
        ref["T1"],
    )
    assert SC["C"] is None and ref["C_is_null"]
    np.testing.assert_array_equal(SC["A"], _arr(ref["A"]))
    np.testing.assert_array_equal(SC["B"], _arr(ref["B"]))
    np.testing.assert_array_equal(SC["P"], _arr(ref["P"]))


# --------------------------------------------------------------------------
# scest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nm", list(CONSTR))
def test_scest_weights_optimal_and_within_solver_gap(nm):
    ref = R[nm]["scest"]
    est = sp.scest(GER, **KW, w_constr=CONSTR[nm])
    A, B = SC["A"], SC["B"]
    w_py, w_R = est["weights"], _arr(ref["w"])
    spec = est["w_constr_spec"]

    def f(w):
        return float(np.sum((A - B @ w) ** 2))

    gap = f(w_R) - f(w_py)
    # StatsPAI's exact optimum is never worse than the conic solver's point
    assert gap >= -1e-14 * f(w_py)
    # CLARABEL / OSQP stop at a 1e-8 relative gap
    assert gap <= 1e-7 * f(w_py)
    # R's point is feasible for the same constraint set ...
    if spec["name"] in ("simplex", "L1-L2"):
        assert abs(w_R.sum() - 1.0) < 1e-12 and w_R.min() >= 0.0
    if spec["name"] == "lasso":
        assert np.abs(w_R).sum() <= spec["Q"] + 1e-9
    if spec["name"] == "ridge":
        assert np.linalg.norm(w_R) <= spec["Q"] + 1e-9
    if spec["name"] == "L1-L2":
        assert np.linalg.norm(w_R) <= spec["Q2"] + 1e-9
    # ... so strong convexity bounds the weight distance by the gap
    lam_min = np.linalg.eigvalsh(B.T @ B).min()
    bound = np.sqrt(max(gap, 0.0) / lam_min)
    assert np.linalg.norm(w_R - w_py) <= bound + 1e-9
    if spec["name"] in ("ols", "lasso"):
        _close(w_py, w_R, rtol=0, atol=1e-9)
    # fitted paths follow the weights
    _close(est["Y_synth_post"], SC["P"] @ w_py, rtol=1e-14)
    _close(SC["P"] @ w_R, ref["Y_post_fit"], rtol=1e-12)


@pytest.mark.parametrize("nm", ["ridge", "L1L2", "simplex", "lasso"])
def test_scest_constraint_radius_matches(nm):
    ref = R[nm]["scest"]
    spec = sp.scest(GER, **KW, w_constr=CONSTR[nm])["w_constr_spec"]
    _close(spec["Q"], ref["Q"])
    if ref["Q2"] is not None:
        _close(spec["Q2"], ref["Q2"])
    if ref["lambda"] is not None:
        _close(spec["lambda"], ref["lambda"])


def test_scest_simplex_norway_weight_is_exactly_zero():
    """R's CLARABEL leaves Norway at 2.0e-6; the exact optimum has 0 with a
    strictly positive KKT multiplier (so it is not a near-tie)."""
    est = sp.scest(GER, **KW)
    w = est["weights"]
    j = SC["donor_names"].index("Norway")
    assert w[j] == 0.0
    assert R["simplex"]["scest"]["w"][j] > 1e-6  # R's count of active donors
    A, B = SC["A"], SC["B"]
    grad = B.T @ (B @ w - A)
    free = w > 0
    nu = -grad[free].mean()
    np.testing.assert_allclose(grad[free] + nu, 0.0, atol=1e-10)
    lam = grad + nu
    assert lam[j] > 1e-3 * np.abs(grad).max()


# --------------------------------------------------------------------------
# scpi: deterministic ingredients on R's weights
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nm", list(CONSTR))
def test_scpi_deterministic_ingredients(nm):
    p = R[nm]["scpi"]
    out = _inference_on_R_weights(nm)
    _close(out["rho"], p["rho"])
    if p["Q_star"] is not None:
        _close(out["Q_star"], p["Q_star"])
    _close(out["u_mean"], p["u_mean"], rtol=1e-9, atol=1e-13)
    _close(out["u_var"], p["u_var"], rtol=1e-9)
    _close(out["Sigma"], p["Sigma"], rtol=1e-9, atol=1e-12 * np.abs(p["Sigma"]).max())
    assert (out["u_T"], out["u_params"], out["u_order"]) == (
        p["u_T"],
        p["u_params"],
        p["u_order"],
    )
    assert (out["e_T"], out["e_params"], out["e_order"]) == (
        p["e_T"],
        p["e_params"],
        p["e_order"],
    )
    _close(out["e_mean"], p["e_mean"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("nm", list(CONSTR))
def test_out_of_sample_bounds_exact_vs_rrq_simplex_method(nm):
    """rrq via quantreg's exact LP (method = 'br'): 1e-9."""
    out = _inference_on_R_weights(nm)
    e = R[nm]["scpi"]["e_br"]
    _close(out["e_var"], e["e_var"], rtol=1e-9, atol=1e-12)
    for ours, theirs in (("subgaussian", "gaussian"), ("ls", "ls"), ("qreg", "qreg")):
        _close(out["e_bounds"][ours][:, 0], e[theirs]["lb"], rtol=0, atol=1e-9)
        _close(out["e_bounds"][ours][:, 1], e[theirs]["ub"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("nm", list(CONSTR))
def test_out_of_sample_bounds_vs_default_interior_point(nm):
    """R's default rrq (Frisch-Newton, 1e-6 duality gap): aligned, 5e-4."""
    out = _inference_on_R_weights(nm)
    e = R[nm]["scpi"]["e_fn"]
    for ours, theirs in (("subgaussian", "gaussian"), ("ls", "ls"), ("qreg", "qreg")):
        _close(out["e_bounds"][ours][:, 0], e[theirs]["lb"], rtol=0, atol=5e-4)
        _close(out["e_bounds"][ours][:, 1], e[theirs]["ub"], rtol=0, atol=5e-4)


# --------------------------------------------------------------------------
# scpi: in-sample simulation fed R's draws
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nm", list(CONSTR))
def test_insample_per_draw_vs_ecos(nm):
    p = R[nm]["scpi"]
    out = _inference_on_R_weights(nm)
    assert out["n_slsqp_fallback"] == 0
    ours = out["vsig"]
    tight = _arr(p["vsig_tight"])
    default = _arr(p["vsig"])
    assert np.isfinite(ours).all() and np.isfinite(default).all()
    d_tight = np.abs(ours - tight)
    assert np.median(d_tight) <= 1e-6
    assert d_tight.max() <= 2e-4
    # default ECOS is further from the exact solution than ECOS at 1e-12
    assert np.median(np.abs(ours - default)) > np.median(d_tight)
    assert np.abs(ours - default).max() <= 1e-2


@pytest.mark.parametrize("nm", list(CONSTR))
def test_insample_bounds_vs_ecos(nm):
    p = R[nm]["scpi"]
    out = _inference_on_R_weights(nm)
    T1 = R["scdata"]["T1"]
    tight = _arr(p["vsig_tight"])
    q_lb = np.quantile(tight[:, :T1], 0.025, axis=0)
    q_ub = np.quantile(tight[:, T1:], 0.975, axis=0)
    _close(out["bounds"]["insample"][:, 0], q_lb, rtol=0, atol=1e-5)
    _close(out["bounds"]["insample"][:, 1], q_ub, rtol=0, atol=1e-5)
    # what an R user gets (ECOS defaults)
    for k in ("insample", "subgaussian", "ls", "qreg", "joint"):
        _close(out["bounds"][k][:, 0], p[k]["lb"], rtol=0, atol=2e-3)
        _close(out["bounds"][k][:, 1], p[k]["ub"], rtol=0, atol=2e-3)
    assert np.all(_arr(p["failed_sims"]) == 0)


def test_insample_solutions_match_independent_solver():
    """SLSQP (independent algorithm) reproduces the active-set optimum."""
    nm = "simplex"
    p = R[nm]["scpi"]
    w = _arr(p["w"])
    spec = _constraint_spec(SC["A"], SC["B"], "simplex")
    res = SC["A"] - SC["B"] @ w
    geom = inf_mod._local_geometry(spec, w, None, 0.2, res, SC["B"])
    out = _inference_on_R_weights(nm)
    zeta = inf_mod._sqrtm_psd(out["Sigma"]) @ _arr(p["zraw"])
    Qm = SC["B"].T @ SC["B"] / SC["T0"]
    rng = np.random.default_rng(0)
    for s in rng.choice(p["sims"], 5, replace=False):
        for h in (0, 6, 12):
            for sign in (-1.0, 1.0):
                c = sign * SC["P"][h]
                y = sv.insample_slsqp(
                    c, Qm, zeta[:, s], w, spec, geom["lb"], geom["Q_star"], None
                )
                col = h if sign < 0 else SC["T1"] + h
                assert abs(-SC["P"][h] @ y - out["vsig"][s, col]) <= 1e-7


def test_ols_insample_equals_closed_form():
    """Known truth: with no constraint, max/min p'y over the ellipsoid
    y'Qy - 2G'y <= 0 is p'Q^{-1}G -/+ sqrt(G'Q^{-1}G p'Q^{-1}p)."""
    p = R["ols"]["scpi"]
    out = _inference_on_R_weights("ols")
    zeta = inf_mod._sqrtm_psd(out["Sigma"]) @ _arr(p["zraw"])
    Qm = SC["B"].T @ SC["B"] / SC["T0"]
    Qi = np.linalg.inv(Qm)
    T1 = SC["T1"]
    for s in range(p["sims"]):
        G = zeta[:, s]
        for h in range(T1):
            x = SC["P"][h]
            centre, rad = x @ Qi @ G, np.sqrt((G @ Qi @ G) * (x @ Qi @ x))
            np.testing.assert_allclose(
                out["vsig"][s, h], -(centre + rad), rtol=1e-9, atol=1e-12
            )
            np.testing.assert_allclose(
                out["vsig"][s, T1 + h], -(centre - rad), rtol=1e-9, atol=1e-12
            )


# --------------------------------------------------------------------------
# End-to-end sp.scpi
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nm", ["lasso", "ridge", "ols", "L1L2"])
def test_scpi_end_to_end(nm):
    p = R[nm]["scpi"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = sp.scpi(
            GER, **KW, w_constr=CONSTR[nm], sims=p["sims"], draws=_arr(p["zraw"])
        )
    mi = r.model_info
    # weights differ from R's by the conic-solver gap only (<= 2e-5 here)
    _close(mi["rho"], p["rho"], rtol=1e-6)
    _close(mi["Sigma"], p["Sigma"], rtol=0, atol=1e-6 * np.abs(p["Sigma"]).max())
    for k in ("insample", "subgaussian", "ls", "qreg", "joint"):
        _close(mi["bounds"][k][:, 0], p[k]["lb"], rtol=0, atol=2e-3)
        _close(mi["bounds"][k][:, 1], p[k]["ub"], rtol=0, atol=2e-3)
    # the reported effect interval is Y_post - (fit + bounds)
    y_post = r.model_info["Y_treated"][-SC["T1"] :]
    fit = SC["P"] @ np.array(list(mi["weights"].values()))
    pr = mi["period_results"]
    _close(pr["pi_lower"], y_post - fit - mi["bounds"]["subgaussian"][:, 1], rtol=1e-12)
    _close(pr["pi_upper"], y_post - fit - mi["bounds"]["subgaussian"][:, 0], rtol=1e-12)


def test_scpi_simplex_end_to_end_differs_only_by_R_solver_noise():
    """T4: R's simplex rho / df count Norway (2.0e-6 from CLARABEL); the
    exact optimum has it at 0.  R's own L1-L2 fit (same optimum; its L2
    bound is slack) does not, and agrees with StatsPAI's rho."""
    p = R["simplex"]["scpi"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = sp.scpi(GER, **KW, sims=p["sims"], draws=_arr(p["zraw"]))
    mi = r.model_info
    w_py = np.array(list(mi["weights"].values()))
    assert np.sum(np.abs(w_py) >= 1e-6) == 6
    assert np.sum(np.abs(_arr(p["w"])) >= 1e-6) == 7
    assert np.sum(np.abs(_arr(R["L1L2"]["scpi"]["w"])) >= 1e-6) == 6
    _close(mi["rho"], R["L1L2"]["scpi"]["rho"], rtol=1e-6)
    assert abs(mi["rho"] / p["rho"] - 1) > 0.05
    assert mi["df"] == 5.0  # simplex: d0 - 1 (R: 6)


def test_scpi_average_effect_interval_is_scdatamulti_unit():
    u = R["unit_simplex"]
    assert u["donors"] == R["scdata"]["donors"]
    np.testing.assert_allclose(u["P"], SC["P"].mean(axis=0), rtol=1e-15)
    spec = _constraint_spec(SC["A"], SC["B"], "simplex")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = scpi_inference(
            SC["A"],
            SC["B"],
            SC["P"],
            _arr(u["w"]),
            spec,
            sims=200,
            draws=_arr(u["zraw"]),
            aggregate=True,
        )
    agg = out["aggregate"]
    for k in ("insample", "subgaussian", "ls", "qreg"):
        _close(agg[k][0], u[k]["lb"], rtol=0, atol=5e-4)
        _close(agg[k][1], u[k]["ub"], rtol=0, atol=5e-4)


# --------------------------------------------------------------------------
# More donors than pre-periods (singular Z'Z): California
# --------------------------------------------------------------------------


def test_california_simplex_singular_design():
    c = R["california_simplex"]
    est = sp.scest(CAL, **KW_CAL)
    sc = est["sc_data"]
    assert est["donor_names"] == c["donors"]
    assert sc["B"].shape[1] > sc["T0"]
    _close(est["weights"], c["scest_w"], rtol=0, atol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = scpi_inference(
            sc["A"],
            sc["B"],
            sc["P"],
            _arr(c["w"]),
            est["w_constr_spec"],
            sims=50,
            draws=_arr(c["zraw"]),
        )
    _close(out["rho"], c["rho"])
    _close(out["Sigma"], c["Sigma"], rtol=0, atol=1e-10 * np.abs(c["Sigma"]).max())
    assert out["n_slsqp_fallback"] == 0
    d = np.abs(out["vsig"] - _arr(c["vsig_tight"]))
    assert np.median(d) <= 1e-5 and d.max() <= 1e-3
    scale = np.abs(_arr(c["insample"]["lb"])).max()
    for k in ("insample", "subgaussian"):
        _close(out["bounds"][k][:, 0], c[k]["lb"], rtol=0, atol=1e-5 * scale)
        _close(out["bounds"][k][:, 1], c[k]["ub"], rtol=0, atol=1e-5 * scale)


# --------------------------------------------------------------------------
# Reference-free identities and API contract
# --------------------------------------------------------------------------


def test_weight_constraints_hold_exactly():
    for name in ("simplex", "lasso", "ridge", "L1-L2"):
        est = sp.scest(GER, **KW, w_constr=name)
        w, spec = est["weights"], est["w_constr_spec"]
        if name in ("simplex", "L1-L2"):
            assert w.min() >= 0 and abs(w.sum() - 1) < 1e-13
        if name == "lasso":
            assert abs(np.abs(w).sum() - spec["Q"]) < 1e-12  # binding here
        if name == "ridge":
            assert abs(np.linalg.norm(w) - spec["Q"]) < 1e-12  # binding here


def test_deprecated_penalty_arguments_warn():
    with pytest.warns(DeprecationWarning, match="lasso_lambda / ridge_lambda"):
        sp.scest(GER, **KW, w_constr="lasso", lasso_lambda=0.5)


def test_scpi_result_contract():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = sp.scpi(GER, **KW, sims=20, seed=1)
    pr = r.model_info["period_results"]
    assert len(pr) == SC["T1"]
    assert (pr["pi_lower"] < pr["pi_upper"]).all()
    assert np.isnan(r.se) and np.isnan(r.pvalue)
    # a prediction interval need not contain the point estimate: it is
    # Y - [fit + upper, fit + lower] with location-shifted bounds (e.mean)
    assert r.ci[0] < r.ci[1]
    assert r.model_info["nominal_coverage"] == pytest.approx(0.90)
