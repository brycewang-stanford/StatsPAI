"""Tests for sp.choo_siow (Choo-Siow TU matching).

Evidence, by kind:

* Identities: the nonparametric surplus formula and its delta-method
  covariance; scale invariance; household data == table input; saturated
  basis == nonparametric (estimate and SE); the parametric fit's
  first-order conditions (margins and moment matching); the IPFP fixed
  point; a model-generated table returns its own Phi / beta exactly.
* Known truth: an individual-level market with type-I extreme-value tastes
  over partner types, solved as a linear program (the TU stable matching
  maximises total surplus), recovers Phi.

Reference parity against cupid_matching is in
tests/reference_parity/test_choo_siow_cupid_parity.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import linear_sum_assignment, linprog

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)
from statspai.structural._choo_siow_core import ipfp_equilibrium

MU = np.array([[300.0, 80.0, 20.0], [90.0, 260.0, 70.0], [25.0, 85.0, 310.0]])
SM = np.array([120.0, 100.0, 90.0])
SW = np.array([110.0, 105.0, 95.0])
X, Y = MU.shape
DIST = np.abs(np.subtract.outer(np.arange(X), np.arange(Y))).astype(float)
BASIS = {"const": np.ones((X, Y)), "distance": DIST}


def _households(mu, sm, sw):
    rows = []
    for x in range(mu.shape[0]):
        for y in range(mu.shape[1]):
            rows += [{"h": f"m{x}", "w": f"w{y}"}] * int(mu[x, y])
    for x, k in enumerate(sm):
        rows += [{"h": f"m{x}", "w": None}] * int(k)
    for y, k in enumerate(sw):
        rows += [{"h": None, "w": f"w{y}"}] * int(k)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0)


# ---------------------------------------------------------------- identities


def test_nonparametric_formula_and_covariance():
    res = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW)
    phi = 2 * np.log(MU) - np.log(SM)[:, None] - np.log(SW)[None, :]
    np.testing.assert_allclose(res.phi.to_numpy(), phi, rtol=0, atol=1e-14)
    var = 4 / MU + 1 / SM[:, None] + 1 / SW[None, :]
    np.testing.assert_allclose(res.phi_se.to_numpy() ** 2, var, rtol=1e-14)
    V = res.vcov_params.to_numpy()
    # Phi_00 and Phi_01 share the single men of type 0 only
    assert V[0, 1] == pytest.approx(1 / SM[0], rel=1e-14)
    # Phi_00 and Phi_10 share the single women of type 0 only
    assert V[0, Y] == pytest.approx(1 / SW[0], rel=1e-14)
    # Phi_00 and Phi_11 share nothing
    assert V[0, Y + 1] == 0.0
    assert res.method == "nonparametric"
    assert list(res.params.index[:2]) == ["0|0", "0|1"]


def test_scale_invariance():
    a = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW)
    b = sp.choo_siow(mu=10 * MU, singles_men=10 * SM, singles_women=10 * SW)
    np.testing.assert_allclose(a.phi, b.phi, atol=1e-13)
    np.testing.assert_allclose(a.phi_se / np.sqrt(10), b.phi_se, rtol=1e-13)


def test_household_data_equals_table():
    df = _households(MU, SM, SW)
    a = sp.choo_siow(df, man="h", woman="w")
    b = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW)
    np.testing.assert_allclose(a.phi.to_numpy(), b.phi.to_numpy(), atol=1e-14)
    assert list(a.phi.index) == ["m0", "m1", "m2"]
    assert list(a.phi.columns) == ["w0", "w1", "w2"]
    assert a.n_households == len(df)
    assert a.model_info["input"] == "households"


def test_saturated_basis_equals_nonparametric():
    sat = {
        f"c{x}{y}": np.eye(1, X * Y, x * Y + y).reshape(X, Y)
        for x in range(X)
        for y in range(Y)
    }
    par = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW, basis=sat)
    np_ = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW)
    # one free surplus per cell: moment matching forces mu(beta) = mu_hat
    np.testing.assert_allclose(par.params.to_numpy(), np_.params.to_numpy(), atol=1e-10)
    # and the GLM sandwich equals the delta-method covariance
    np.testing.assert_allclose(
        par.vcov_params.to_numpy(), np_.vcov_params.to_numpy(), rtol=1e-9, atol=1e-14
    )


def test_parametric_first_order_conditions():
    res = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW, basis=BASIS)
    fm = res.fitted_mu.to_numpy()
    n = MU.sum(1) + SM
    m = MU.sum(0) + SW
    # accounting identities at the observed margins
    np.testing.assert_allclose(fm.sum(1) + res.fitted_singles_men, n, rtol=1e-12)
    np.testing.assert_allclose(fm.sum(0) + res.fitted_singles_women, m, rtol=1e-12)
    # moment matching: sum phi_k mu(beta) = sum phi_k mu_hat
    mm = res.diagnostics["moment_matching"]
    np.testing.assert_allclose(mm["fitted"], mm["observed"], rtol=1e-12)
    # fitted cells are a Choo-Siow equilibrium for the fitted Phi
    phi = res.phi.to_numpy()
    lhs = np.sqrt(np.outer(res.fitted_singles_men, res.fitted_singles_women)) * np.exp(
        phi / 2
    )
    np.testing.assert_allclose(fm, lhs, rtol=1e-10)
    assert res.diagnostics["converged"]
    se_phi = np.sqrt(
        np.einsum(
            "ik,kl,il->i",
            np.stack([BASIS["const"].ravel(), DIST.ravel()], 1),
            res.vcov_params.to_numpy(),
            np.stack([BASIS["const"].ravel(), DIST.ravel()], 1),
        )
    ).reshape(X, Y)
    np.testing.assert_allclose(res.phi_se.to_numpy(), se_phi, rtol=1e-12)


def test_ipfp_fixed_point_and_margins():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(4, 5))
    n = rng.uniform(50, 150, 4)
    m = rng.uniform(50, 150, 5)
    eq = ipfp_equilibrium(phi, n, m, tol=1e-14)
    mu = eq["mu"]
    np.testing.assert_allclose(mu.sum(1) + eq["singles_men"], n, rtol=1e-13)
    np.testing.assert_allclose(mu.sum(0) + eq["singles_women"], m, rtol=1e-13)
    np.testing.assert_allclose(
        mu,
        np.sqrt(np.outer(eq["singles_men"], eq["singles_women"])) * np.exp(phi / 2),
        rtol=1e-13,
    )


def test_model_generated_table_returns_its_surplus():
    beta = np.array([1.5, -1.2])
    phi = beta[0] * BASIS["const"] + beta[1] * DIST
    n = np.array([400.0, 500.0, 300.0])
    m = np.array([350.0, 450.0, 420.0])
    eq = ipfp_equilibrium(phi, n, m, tol=1e-15)
    np_ = sp.choo_siow(
        mu=eq["mu"], singles_men=eq["singles_men"], singles_women=eq["singles_women"]
    )
    np.testing.assert_allclose(np_.phi.to_numpy(), phi, atol=1e-12)
    par = sp.choo_siow(
        mu=eq["mu"],
        singles_men=eq["singles_men"],
        singles_women=eq["singles_women"],
        basis=BASIS,
    )
    np.testing.assert_allclose(par.params.to_numpy(), beta, atol=1e-10)


def test_counterfactual_identities():
    np_ = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW)
    same = np_.counterfactual()
    # nonparametric Phi at the observed margins reproduces the data table
    np.testing.assert_allclose(same.mu.to_numpy(), MU, rtol=1e-11)
    np.testing.assert_allclose(same.singles_men, SM, rtol=1e-11)
    par = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW, basis=BASIS)
    eq = par.counterfactual()
    np.testing.assert_allclose(eq.mu.to_numpy(), par.fitted_mu.to_numpy(), rtol=1e-10)

    men = np.array([500.0, 500.0, 500.0])
    women = pd.Series([400.0, 500.0, 600.0], index=par.mu.columns)
    cf = par.counterfactual(new_margins=(men, women))
    np.testing.assert_allclose(cf.mu.sum(1) + cf.singles_men, men, rtol=1e-11)
    np.testing.assert_allclose(cf.mu.sum(0) + cf.singles_women, women, rtol=1e-11)
    assert cf.max_margin_error < 1e-11
    # one-sided change via dict; equilibrium is homogeneous of degree one
    cf2 = par.counterfactual(new_margins={"men": 2 * (MU.sum(1) + SM)})
    np.testing.assert_allclose(cf2.men_margins, 2 * (MU.sum(1) + SM))
    cf3 = par.counterfactual(new_margins=(2 * (MU.sum(1) + SM), 2 * (MU.sum(0) + SW)))
    np.testing.assert_allclose(cf3.mu.to_numpy(), 2 * eq.mu.to_numpy(), rtol=1e-10)


def test_counterfactual_surplus_override_no_sorting():
    par = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW, basis=BASIS)
    flat = np.full((X, Y), float(par.phi.to_numpy().mean()))
    cf = par.counterfactual(phi=flat)
    # with a surplus constant in y, every man type spreads over women types
    # in proportion to sqrt(mu_0y): the rows of mu are proportional
    rows = cf.mu.to_numpy() / cf.mu.to_numpy().sum(1, keepdims=True)
    np.testing.assert_allclose(rows, np.broadcast_to(rows[0], rows.shape), rtol=1e-12)
    w = np.sqrt(cf.singles_women.to_numpy())
    np.testing.assert_allclose(rows[0], w / w.sum(), rtol=1e-12)
    with pytest.raises(MethodIncompatibility, match="shape"):
        par.counterfactual(phi=np.zeros((2, 2)))
    with pytest.raises(MethodIncompatibility, match="NaN"):
        par.counterfactual(
            phi=pd.DataFrame(flat, index=[9, 8, 7], columns=par.mu.columns)
        )
    with pytest.raises(MethodIncompatibility, match="positive"):
        par.counterfactual(new_margins=([1.0, -1.0, 1.0], [1.0, 1.0, 1.0]))


# ---------------------------------------------------------------- known truth


def _market_lp(phi, xm, yw, eps, eta):
    """Individual-level TU market: maximise total surplus.

    a[i, y] = 1 if man i takes a type-y partner (y = 0: single), b[j, x]
    likewise for women; coupling rows force #men of type x choosing y =
    #women of type y choosing x. Each column has two nonzeros with the
    sign pattern of a network matrix, so the LP has integral vertices.
    """
    Xn, Yn = phi.shape
    N, M = xm.size, yw.size
    nA = N * (Yn + 1)
    ca = eps.copy()
    ca[:, 1:] += phi[xm]
    c = -np.concatenate([ca.ravel(), eta.ravel()])
    rows, cols, vals = [], [], []
    for y in range(Yn + 1):
        rows.append(np.arange(N))
        cols.append(np.arange(N) * (Yn + 1) + y)
        vals.append(np.ones(N))
    for x in range(Xn + 1):
        rows.append(N + np.arange(M))
        cols.append(nA + np.arange(M) * (Xn + 1) + x)
        vals.append(np.ones(M))
    for x in range(Xn):
        for y in range(Yn):
            k = N + M + x * Yn + y
            im, jw = np.where(xm == x)[0], np.where(yw == y)[0]
            rows += [np.full(im.size, k), np.full(jw.size, k)]
            cols += [im * (Yn + 1) + 1 + y, nA + jw * (Xn + 1) + 1 + x]
            vals += [np.ones(im.size), -np.ones(jw.size)]
    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N + M + Xn * Yn, nA + M * (Xn + 1)),
    )
    b = np.concatenate([np.ones(N + M), np.zeros(Xn * Yn)])
    sol = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1), method="highs-ds")
    assert sol.status == 0
    a = sol.x[:nA].reshape(N, Yn + 1)
    bb = sol.x[nA:].reshape(M, Xn + 1)
    mu = np.array([a[xm == x, 1:].sum(0) for x in range(Xn)])
    sm = np.array([a[xm == x, 0].sum() for x in range(Xn)])
    sw = np.array([bb[yw == y, 0].sum() for y in range(Yn)])
    frac = max(np.max(np.minimum(a, 1 - a)), np.max(np.minimum(bb, 1 - bb)))
    return mu, sm, sw, -sol.fun, frac


def test_market_lp_equals_brute_force_assignment():
    phi = np.array([[1.0, -0.5, 0.2], [-0.3, 0.8, -1.0]])
    rng = np.random.default_rng(5)
    xm = np.repeat([0, 1], [40, 50])
    yw = np.repeat([0, 1, 2], [30, 35, 25])
    eps = rng.gumbel(size=(xm.size, 4))
    eta = rng.gumbel(size=(yw.size, 3))
    *_, value, frac = _market_lp(phi, xm, yw, eps, eta)
    # brute force: N x M gains relative to both single, with own-single slots
    N, M = xm.size, yw.size
    gain = (
        phi[xm][:, yw]
        + eps[np.arange(N)][:, 1 + yw]
        + eta[:, 1 + xm].T
        - eps[:, [0]]
        - eta[:, 0][None, :]
    )
    big = 1e9
    top = np.hstack([gain, np.where(np.eye(N, dtype=bool), 0.0, -big)])
    bot = np.hstack([np.where(np.eye(M, dtype=bool), 0.0, -big), np.zeros((M, N))])
    A = np.vstack([top, bot])
    r, c = linear_sum_assignment(A, maximize=True)
    brute = A[r, c].sum() + eps[:, 0].sum() + eta[:, 0].sum()
    assert value == pytest.approx(brute, rel=1e-12)
    assert frac == 0.0


def test_gumbel_large_market_recovers_phi():
    """5,000+ agents per type, tastes Gumbel over partner types. Observed
    errors are below 0.09 (|z| <= 1.34). Bounds: |z| < 3.5 for each of the
    six cells, and |error| < 0.15 (about 1.5 delta-method SEs)."""
    phi = np.array([[1.0, -0.5, 0.2], [-0.3, 0.8, -1.0]])
    rng = np.random.default_rng(11)
    nn = 5000
    xm = np.repeat([0, 1], [nn, int(1.1 * nn)])
    yw = np.repeat([0, 1, 2], [nn, int(0.9 * nn), int(0.8 * nn)])
    eps = rng.gumbel(size=(xm.size, 4))
    eta = rng.gumbel(size=(yw.size, 3))
    mu, sm, sw, _, frac = _market_lp(phi, xm, yw, eps, eta)
    assert frac == 0.0  # integral matching
    res = sp.choo_siow(mu=mu, singles_men=sm, singles_women=sw)
    err = res.phi.to_numpy() - phi
    z = err / res.phi_se.to_numpy()
    assert np.all(np.abs(z) < 3.5)
    assert np.all(np.abs(err) < 0.15)


# ---------------------------------------------------------------- edge cases


def test_empty_couple_cell_warns_and_parametric_uses_it():
    mu = MU.copy()
    mu[0, 2] = 0.0
    with pytest.warns(AssumptionWarning, match="empty"):
        res = sp.choo_siow(mu=mu, singles_men=SM, singles_women=SW)
    assert res.phi.iloc[0, 2] == -np.inf
    assert np.isnan(res.phi_se.iloc[0, 2])
    assert np.isnan(res.vcov_params.iloc[2, 0])
    assert res.diagnostics["zero_couple_cells"] == [(0, 2)]
    par = sp.choo_siow(mu=mu, singles_men=SM, singles_women=SW, basis=BASIS)
    assert np.all(np.isfinite(par.phi.to_numpy()))
    # the -inf cell still solves (exp(-inf / 2) = 0): that pair never forms
    eq = res.counterfactual()
    assert eq.mu.iloc[0, 2] == 0.0


def test_errors():
    with pytest.raises(DataInsufficient, match="no singles"):
        sp.choo_siow(mu=MU, singles_men=[0.0, 1.0, 1.0], singles_women=SW)
    with pytest.raises(MethodIncompatibility, match="either"):
        sp.choo_siow(pd.DataFrame({"h": [1]}), man="h", woman="w", mu=MU)
    with pytest.raises(MethodIncompatibility, match="singles_men"):
        sp.choo_siow(mu=MU)
    with pytest.raises(MethodIncompatibility, match="single counts"):
        sp.choo_siow(mu=MU, singles_men=[1.0, 2.0], singles_women=SW)
    with pytest.raises(MethodIncompatibility, match="no data"):
        sp.choo_siow()
    with pytest.raises(ColumnNotFound):
        sp.choo_siow(pd.DataFrame({"h": [1]}), man="h", woman="w")
    with pytest.raises(DataInsufficient, match="neither"):
        sp.choo_siow(pd.DataFrame({"h": [1, None], "w": [2, None]}), man="h", woman="w")
    with pytest.raises(IdentificationFailure, match="collinear"):
        sp.choo_siow(
            mu=MU,
            singles_men=SM,
            singles_women=SW,
            basis={"a": DIST, "b": 2 * DIST},
        )
    with pytest.raises(MethodIncompatibility, match="shape"):
        sp.choo_siow(
            mu=MU, singles_men=SM, singles_women=SW, basis={"a": np.ones((2, 2))}
        )
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.choo_siow(mu=-MU, singles_men=SM, singles_women=SW)


def test_result_protocol():
    res = sp.choo_siow(mu=MU, singles_men=SM, singles_women=SW, basis=BASIS)
    tab = res.coef_table()
    assert list(tab.index) == ["const", "distance"]
    assert (tab["ci_lower"] < tab["estimate"]).all()
    assert "Choo-Siow" in res.summary()
    assert "IPFP" in res.counterfactual().summary()
    json.dumps(res.to_dict(), default=str)
    td = res.tidy()
    assert len(td) == 2
    assert res.cite()
    assert np.allclose(res.vcov().to_numpy(), res.vcov_params.to_numpy())
    long = res.counterfactual().to_frame()
    assert len(long) == X * Y + X + Y
