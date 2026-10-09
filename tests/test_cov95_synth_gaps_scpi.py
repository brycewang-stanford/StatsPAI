"""Coverage gaps in the SCPI port: ``scpi`` / ``_scpi_solvers`` / ``_scpi_inference``.

The exact solvers are checked against closed forms (a linear objective over
a ball, the intersection of two circles, least squares inside a norm ball);
the option branches of ``sp.scpi`` are checked through properties that must
hold whatever the simulation draws are (the point estimate does not depend
on the variance estimator, HC scalings are ordered, a rank-deficient OLS
problem has unbounded in-sample bounds).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth import _scpi_inference as si
from statspai.synth import _scpi_solvers as sv
from statspai.synth.scpi import _constraint_spec, scdata, scpi

T_TREAT = 18


def _panel(seed=0, n_donors=6, n_t=24, t0=17, noise=0.3):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n_t))
    rows = []
    for j in range(n_donors + 1):
        lam, a = rng.uniform(0.5, 1.5), rng.normal()
        u = "tr" if j == n_donors else f"d{j}"
        for t in range(n_t):
            eff = 2.0 if (u == "tr" and t >= t0) else 0.0
            rows.append((u, t + 1, a + lam * f[t] + rng.normal(0, noise) + eff))
    return pd.DataFrame(rows, columns=["unit", "time", "y"])


def _fit(df=None, treatment_time=T_TREAT, **kw):
    df = _panel() if df is None else df
    kw.setdefault("sims", 20)
    kw.setdefault("seed", 1)
    return scpi(df, "y", "unit", "time", "tr", treatment_time, **kw)


# ---------------------------------------------------------------------- #
#  Weight solvers
# ---------------------------------------------------------------------- #


def test_qp_bounded_sum_rejects_lower_bounds_above_the_total():
    with pytest.raises(MethodIncompatibility, match="Infeasible simplex"):
        sv.qp_bounded_sum(np.eye(3), np.zeros(3), np.full(3, 0.5), 1.0)


def test_lasso_ball_returns_ols_when_ols_is_inside_the_ball():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(30, 3))
    A = Z @ np.array([0.2, -0.1, 0.3]) + rng.normal(0, 0.01, 30)
    ols = np.linalg.lstsq(Z, A, rcond=None)[0]
    assert np.abs(ols).sum() < 5.0
    np.testing.assert_allclose(sv.lasso_ball(Z, A, 5.0), ols, atol=1e-12)


def test_lasso_ball_rank_deficient_returns_an_interpolating_path_end():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(4, 7))  # more donors than periods: OLS not unique
    A = rng.normal(size=4)
    w = sv.lasso_ball(Z, A, 1e6)
    np.testing.assert_allclose(Z @ w, A, atol=1e-8)
    # the lasso path's end is the minimum-L1 interpolant: at most T non-zeros
    assert int(np.sum(np.abs(w) > 1e-10)) <= 4


def test_ridge_ball_returns_least_squares_when_the_bound_is_slack():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(25, 4))
    A = rng.normal(size=25)
    ols = np.linalg.lstsq(Z, A, rcond=None)[0]
    np.testing.assert_allclose(
        sv.ridge_ball(Z, A, 10 * np.linalg.norm(ols)), ols, atol=1e-10
    )


def test_simplex_l2_binding_bound_gives_weights_on_the_sphere():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(30, 4))
    A = Z[:, 0] + rng.normal(0, 0.01, 30)  # simplex solution ~ e_1, norm ~ 1
    free = sv.qp_bounded_sum(Z.T @ Z, Z.T @ A, np.zeros(4), 1.0)
    assert np.linalg.norm(free) > 0.9
    w = sv.simplex_l2(Z, A, 0.6)
    assert np.linalg.norm(w) == pytest.approx(0.6, abs=1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert (w >= -1e-12).all()
    # tightening the feasible set cannot improve the fit
    assert np.sum((A - Z @ w) ** 2) > np.sum((A - Z @ free) ** 2)


def test_simplex_l2_rejects_a_radius_below_the_uniform_weights():
    Z = np.eye(4)
    with pytest.raises(MethodIncompatibility, match="L1-L2 constraint infeasible"):
        sv.simplex_l2(Z, np.ones(4), 0.4)  # 1/sqrt(4) = 0.5 is the minimum norm


def test_lm_coef_aliases_zero_and_collinear_columns_like_r():
    rng = np.random.default_rng(4)
    x1, x2 = rng.normal(size=20), rng.normal(size=20)
    X = np.column_stack([x1, np.zeros(20), x2, x1 + 2 * x2])
    y = 1.5 * x1 - 0.5 * x2
    coef, rank = sv.lm_coef(X, y)
    assert rank == 2
    assert np.isnan(coef[1]) and np.isnan(coef[3])
    np.testing.assert_allclose(coef[[0, 2]], [1.5, -0.5], atol=1e-12)


def test_shrinkage_ridge_screens_donors_when_the_regression_is_saturated():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(6, 6))  # zero residual degrees of freedom
    A = Z @ np.array([0.6, 0.4, 0, 0, 0, 0]) + rng.normal(0, 0.05, 6)
    out = sv.shrinkage_ridge(A, Z, 6)
    # the screened regression keeps max(T0 - 10, 2) = 2 donors
    assert np.isfinite(out["Q"]) and out["Q"] > 0
    assert np.isfinite(out["lambda"]) and out["lambda"] >= 0


# ---------------------------------------------------------------------- #
#  In-sample QCQP building blocks
# ---------------------------------------------------------------------- #


def test_min_linear_one_quad_on_the_unit_ball():
    g = np.array([1.0, 0.0])
    z, mu = sv._min_linear_one_quad(g, np.eye(2), np.zeros(2), -1.0)
    np.testing.assert_allclose(z, [-1.0, 0.0], atol=1e-14)
    assert mu == pytest.approx(0.5)  # g + 2 mu z = 0


def test_min_linear_one_quad_singular_matrix_ray_and_bounded_cases():
    M = np.diag([1.0, 0.0])  # the constraint does not involve z_2
    out = sv._min_linear_one_quad(np.array([0.0, 1.0]), M, np.zeros(2), -1.0)
    assert out[0] == "ray"
    np.testing.assert_allclose(out[1], [0.0, -1.0], atol=1e-14)
    # objective orthogonal to the null space: pseudo-inverse solution
    z, mu = sv._min_linear_one_quad(np.array([1.0, 0.0]), M, np.zeros(2), -1.0)
    np.testing.assert_allclose(z, [-1.0, 0.0], atol=1e-12)
    assert mu == pytest.approx(0.5)


def test_min_linear_one_quad_empty_set_and_zero_objective():
    assert sv._min_linear_one_quad(np.ones(2), np.eye(2), np.zeros(2), 1.0) is None
    b = np.array([0.3, -0.2])
    z, mu = sv._min_linear_one_quad(np.zeros(2), np.eye(2), b, -1.0)
    np.testing.assert_allclose(z, -b)  # centre of the ball
    assert mu == 0.0


def test_min_linear_two_quads_second_constraint_alone_binds():
    g = np.array([1.0, 0.0])
    big = (np.eye(2), np.zeros(2), -100.0)  # radius 10
    unit = (np.eye(2), np.zeros(2), -1.0)
    z, (mu1, mu2) = sv._min_linear_two_quads(g, big, unit)
    np.testing.assert_allclose(z, [-1.0, 0.0], atol=1e-14)
    assert mu1 == 0.0 and mu2 == pytest.approx(0.5)


def test_min_linear_two_quads_both_bind_at_the_circle_intersection():
    # |z - (1.5, 0)| <= 1 and |z| <= 1 (the latter scaled by 1e-8 so the
    # bracket on the multiplier ratio has to be widened); min z_2 over the
    # lens is its lower corner (0.75, -sqrt(1 - 0.75^2)).
    c = np.array([1.5, 0.0])
    q1 = (np.eye(2), -c, c @ c - 1.0)
    s = 1e-8
    q2 = (s * np.eye(2), np.zeros(2), -s)
    z, (mu1, mu2) = sv._min_linear_two_quads(np.array([0.0, 1.0]), q1, q2)
    np.testing.assert_allclose(z, [0.75, -np.sqrt(1 - 0.75**2)], atol=1e-7)
    assert mu1 > 0 and mu2 > 0


def test_subproblem_degenerate_working_sets():
    c = np.array([1.0, 1.0])
    Q, G, ell = np.eye(2), np.zeros(2), np.zeros(2)
    none_free = np.array([True, True])
    all_free = np.array([False, False])
    assert sv._subproblem(c, Q, G, 0.0, ell, none_free, False, 0.0, None) is None
    # y'y + 1 <= 0 is empty, with or without a ball
    assert sv._subproblem(c, Q, G, 1.0, ell, all_free, False, 0.0, None) is None
    ball = (np.zeros(2), 1.0)
    assert sv._subproblem(c, Q, G, 1.0, ell, all_free, False, 0.0, ball) is None
    # Q singular along the objective: a descent ray, lifted to full length
    Qs = np.diag([1.0, 0.0])
    out = sv._subproblem(
        np.array([0.0, 1.0]),
        Qs,
        np.array([1.0, 0.0]),
        0.0,
        ell,
        all_free,
        False,
        0.0,
        None,
    )
    assert out[0] == "ray"
    np.testing.assert_allclose(out[1], [0.0, -1.0], atol=1e-14)


def test_active_set_reports_unbounded_infeasible_and_bad_start():
    inf = np.full(2, -np.inf)
    Qs = np.diag([1.0, 0.0])
    G = np.array([1.0, 0.0])
    # free direction with zero curvature and no bound: unbounded below
    assert sv.insample_active_set(np.array([0.0, 1.0]), Qs, G, 0.0, inf) == "unbounded"
    # empty quadratic constraint
    assert sv.insample_active_set(np.ones(2), np.eye(2), np.zeros(2), 1.0, inf) is None
    # start y = 0 violates the bound y >= 1
    assert sv.insample_active_set(np.ones(2), np.eye(2), G, 0.0, np.ones(2)) is None


def _random_problem(seed, J=4):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(20, J))
    Qm = Z.T @ Z / 20
    G = rng.normal(0, 0.05, J)
    c = rng.normal(size=J)
    beta = np.abs(rng.normal(0.25, 0.1, J))
    return c, Qm, G, beta


def test_slsqp_l1_fallback_matches_the_closed_form_when_the_ball_is_slack():
    c, Qm, G, beta = _random_problem(0)
    closed, _ = sv._min_linear_one_quad(c, Qm, -G, 0.0)
    y = sv.insample_slsqp(
        c, Qm, G, beta, {"p": "L1", "dir": "<="}, np.full(4, -np.inf), 50.0, None
    )
    assert y @ Qm @ y - 2 * G @ y <= 1e-8
    assert c @ y == pytest.approx(c @ closed, rel=1e-5, abs=1e-8)


def test_slsqp_l2_fallback_matches_the_active_set_solution():
    c, Qm, G, beta = _random_problem(1)
    R = float(np.linalg.norm(beta)) + 0.01  # tight ball around the estimate
    exact = sv.insample_active_set(
        c, Qm, G, 0.0, np.full(4, -np.inf), None, 0.0, (beta, R)
    )
    y = sv.insample_slsqp(
        c, Qm, G, beta, {"p": "L2", "dir": "<="}, np.full(4, -np.inf), R, None
    )
    assert np.linalg.norm(y + beta) <= R + 1e-7
    assert y @ Qm @ y - 2 * G @ y <= 1e-7
    assert c @ y == pytest.approx(c @ exact, rel=1e-5, abs=1e-8)


def test_warn_failed_only_above_twenty_percent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sv.warn_failed(2, 10)
    with pytest.warns(Warning, match="3/10 in-sample simulation problems"):
        sv.warn_failed(3, 10)


# ---------------------------------------------------------------------- #
#  Inference layer
# ---------------------------------------------------------------------- #


def test_hc_scale_family():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(12, 3))
    lev = np.diag(Z @ np.linalg.solve(Z.T @ Z, Z.T))
    np.testing.assert_array_equal(si._hc_scale("HC0", Z, 12, 3), np.ones(12))
    np.testing.assert_allclose(si._hc_scale("HC1", Z, 12, 3), 12 / 9)
    hc2 = si._hc_scale("HC2", Z, 12, 3)
    np.testing.assert_allclose(hc2, 1 / (1 - lev))
    np.testing.assert_allclose(si._hc_scale("HC3", Z, 12, 3), hc2**2)
    hc4 = si._hc_scale("HC4", Z, 12, 3)
    np.testing.assert_allclose(hc4, (1 - lev) ** -np.minimum(4.0, 12 * lev / 3))
    with pytest.raises(MethodIncompatibility, match="u_sigma must be one of"):
        si._hc_scale("HC9", Z, 12, 3)


def test_u_sigma_changes_the_variance_but_not_the_point_estimate():
    fits = {u: _fit(u_sigma=u) for u in ("HC0", "HC2", "HC3")}
    est = {u: r.estimate for u, r in fits.items()}
    assert est["HC0"] == est["HC2"] == est["HC3"]
    tr = {u: float(np.trace(r.model_info["Sigma"])) for u, r in fits.items()}
    # leverage corrections inflate every squared residual
    assert tr["HC0"] < tr["HC2"] < tr["HC3"]


def test_rho_rules_numeric_type1_and_unknown():
    assert _fit(rho=0.07).model_info["rho"] == 0.07
    t1 = _fit(rho="type-1").model_info["rho"]
    t2 = _fit().model_info["rho"]
    assert 0 < t1 <= 0.2 and 0 < t2 <= 0.2 and t1 != t2
    with pytest.raises(MethodIncompatibility, match="rho must be 'type-1'"):
        _fit(rho="type-9")


def test_rho_too_high_keeps_only_the_largest_weight_active():
    with pytest.warns(RuntimeWarning, match="rho was too high"):
        res = _fit(rho=5.0)
    w = res.model_info["weights"]
    assert res.model_info["rho"] == 5.0
    # every weight is below rho, so each lower bound is the weight itself
    np.testing.assert_allclose(
        res.model_info["lb"], [w[k] for k in sorted(w)], atol=1e-12
    )


def test_rho_too_low_is_raised_when_the_pre_fit_is_exact():
    df = _panel()
    wide = df.pivot(index="time", columns="unit", values="y")
    # the treated unit is an exact convex combination of two donors
    mix = 0.6 * wide["d0"] + 0.4 * wide["d1"]
    df.loc[df["unit"] == "tr", "y"] = mix.to_numpy()
    with pytest.warns(RuntimeWarning, match="rho was too low"):
        res = _fit(df)
    assert res.model_info["rho"] == 0.2  # falls back to rho_max
    w = res.model_info["weights"]
    assert w["d0"] == pytest.approx(0.6, abs=1e-8)
    assert w["d1"] == pytest.approx(0.4, abs=1e-8)
    assert abs(res.estimate) < 1e-8


@pytest.mark.parametrize("kind", ["type-1", "type-2"])
def test_constant_donor_is_refused_by_the_rho_rule(kind):
    df = _panel()
    df.loc[df["unit"] == "d3", "y"] = 1.0
    with pytest.raises(DataInsufficient, match="no variation in the pre-treatment"):
        _fit(df, rho=kind)


def test_e_order_zero_uses_a_constant_out_of_sample_design():
    res = _fit(e_order=0)
    mi = res.model_info
    assert mi["e_order"] == 0 and mi["e_params"] == 1
    # constant design: the conditional mean is the same in every period
    np.testing.assert_allclose(mi["e_mean"], mi["e_mean"][0])


def test_u_order_zero_and_no_misspecification():
    r0 = _fit(u_order=0)
    assert r0.model_info["u_params"] == 1
    np.testing.assert_allclose(r0.model_info["u_mean"], r0.model_info["u_mean"][0])
    r_no = _fit(u_missp=False)
    assert r_no.model_info["u_T"] == 0 and r_no.model_info["u_params"] == 0
    np.testing.assert_array_equal(r_no.model_info["u_mean"], 0.0)
    assert r_no.estimate == r0.estimate


def test_ridge_with_user_radius_has_zero_degrees_of_freedom():
    res = _fit(w_constr="ridge", Q=0.8)
    w = np.array(list(res.model_info["weights"].values()))
    assert np.linalg.norm(w) <= 0.8 + 1e-9
    assert res.model_info["df"] == 0.0  # R: lambda is NULL -> df = 0


def test_short_pre_period_collapses_the_out_of_sample_design():
    # ridge keeps all 6 donors active; with 8 pre-periods the design
    # [donors, constant] has too few residual degrees of freedom (T0 - 10 <=
    # columns), so R falls back to a constant
    df = _panel(n_t=12, t0=8)
    res = _fit(df, treatment_time=9, w_constr="ridge")
    assert res.model_info["n_pre_periods"] == 8
    assert res.model_info["e_params"] == 1
    assert len(res.detail) == 4


def test_radius_rule_with_fewer_than_five_pre_periods():
    df = _panel(n_donors=3, n_t=8, t0=4)
    d = scdata(df, "y", "unit", "time", "tr", 5)
    est = sv.shrinkage_ridge(d["A"], d["B"], 3)
    ridge = _constraint_spec(d["A"], d["B"], "ridge")
    assert ridge["Q"] == max(est["Q"], 0.5)  # R floors the radius at 0.5
    assert np.isnan(ridge["lambda"])  # R: lambda is NA for ridge here
    res = _fit(df, treatment_time=5, w_constr="L1-L2")
    spec = res.model_info["w_constr_spec"]
    assert spec["Q2"] == ridge["Q"] and spec["lambda"] == 0.0
    w = np.array(list(res.model_info["weights"].values()))
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert np.linalg.norm(w) <= spec["Q2"] + 1e-9


def test_ols_with_more_donors_than_pre_periods_has_unbounded_insample_bounds():
    df = _panel(n_donors=8, n_t=10, t0=6)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = _fit(df, treatment_time=7, w_constr="ols")
    msgs = " | ".join(str(w.message) for w in rec)
    assert "OLS weights are not identified" in msgs
    assert "more degrees of freedom than observations" in msgs
    assert "Every in-sample simulation problem was unbounded" in msgs
    mi = res.model_info
    assert mi["df"] == 5  # capped at T0 - 1
    assert np.isnan(mi["bounds"]["insample"]).all()
    np.testing.assert_array_equal(mi["failed_sims"], 100.0)
    # minimum-norm least squares interpolates the pre-period exactly
    assert mi["pre_rmspe"] < 1e-8


def test_scpi_inference_argument_checks_and_default_rng():
    d = scdata(_panel(), "y", "unit", "time", "tr", T_TREAT)
    A, B, P = d["A"], d["B"], d["P"]
    spec = _fit().model_info["w_constr_spec"]
    w = sv.qp_bounded_sum(B.T @ B, B.T @ A, np.zeros(B.shape[1]), 1.0)
    with pytest.raises(NotImplementedError, match="u_order / e_order > 1"):
        si.scpi_inference(A, B, P, w, spec, u_order=2)
    with pytest.raises(MethodIncompatibility, match=r"draws must have shape \(6, 12\)"):
        si.scpi_inference(A, B, P, w, spec, sims=12, draws=np.zeros((6, 3)))
    out = si.scpi_inference(A, B, P, w, spec, sims=12)  # draws its own stream
    ins = out["bounds"]["insample"]
    assert ins.shape == (P.shape[0], 2)
    assert np.isfinite(ins).all() and (ins[:, 0] <= ins[:, 1]).all()


# ---------------------------------------------------------------------- #
#  Data preparation and argument checks
# ---------------------------------------------------------------------- #


def test_scdata_rejects_duplicates_and_missing_treated_outcomes():
    df = _panel()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="Duplicate"):
        scdata(dup, "y", "unit", "time", "tr", T_TREAT)
    bad = df.copy()
    bad.loc[(bad["unit"] == "tr") & (bad["time"] == 3), "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="treated unit has missing"):
        scdata(bad, "y", "unit", "time", "tr", T_TREAT)


def test_scdata_drops_donors_missing_pre_and_refuses_missing_post():
    df = _panel()
    pre = df.copy()
    pre.loc[(pre["unit"] == "d2") & (pre["time"] == 3), "y"] = np.nan
    with pytest.warns(UserWarning, match=r"Dropping donors.*d2"):
        d = scdata(pre, "y", "unit", "time", "tr", T_TREAT)
    assert d["donor_names"] == ["d0", "d1", "d3", "d4", "d5"]
    assert d["B"].shape == (17, 5)
    post = df.copy()
    post.loc[(post["unit"] == "d2") & (post["time"] == 20), "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing in the post-treatment"):
        scdata(post, "y", "unit", "time", "tr", T_TREAT)


def test_scpi_requires_at_least_ten_simulations():
    with pytest.raises(MethodIncompatibility, match="sims must be >= 10"):
        sp.scpi(_panel(), "y", "unit", "time", "tr", T_TREAT, sims=5)
