"""sp.efc_welfare(demand='probit' | 'quadratic', demand_through=...).

The Online Appendix specifications of Einav, Finkelstein and Cullen (2010),
Table A2. The welfare numbers of the published table are checked in
tests/external_parity/test_efc2010_appendix_table_a2.py; here: the general
solver against the closed-form linear one, recovery of known curves, the
restricted least squares and the probit sandwich.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp
from statspai.adverse_selection._efc_core import _efc_quantities
from statspai.adverse_selection._efc_nonlinear import efc_quantities, rows_for
from statspai.exceptions import IdentificationFailure, MethodIncompatibility


def test_general_solver_equals_the_closed_form_for_linear_demand():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.uniform(0.3, 1.5), -rng.uniform(1e-4, 3e-3)
        g, d = rng.uniform(100, 500), rng.uniform(-0.5, 0.8)
        old = _efc_quantities(a, b, g, d)
        new = efc_quantities("linear", [a, b], g, d)
        np.testing.assert_allclose(new[:8], old[:8], rtol=1e-10, atol=1e-12)
        # The deadweight loss is a difference of two surpluses; the two
        # formulas round differently, so compare on the surplus scale.
        scale = max(abs(old[9]), 1.0)
        assert abs(new[8] - old[8]) < 1e-11 * scale
        assert abs(new[9] - old[9]) < 1e-11 * scale


def _probit_design(seed: int, n: int = 40000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.choice([300.0, 400.0, 500.0, 600.0], size=n)
    buy = (1.1 - 0.0018 * p + rng.normal(size=n) > 0).astype(int)
    cost = np.where(buy == 1, 390.0 + 0.15 * p + rng.normal(0, 60, n), np.nan)
    return pd.DataFrame({"p": p, "buy": buy, "cost": cost})


def test_probit_demand_recovers_the_population_welfare():
    res = sp.efc_welfare(_probit_design(1), "p", "buy", "cost", demand="probit")
    truth = dict(
        zip(rows_for("probit"), efc_quantities("probit", [1.1, -0.0018], 390.0, 0.15))
    )
    t = res.table
    for k in ("coverage_eq", "coverage_eff", "dwl"):
        # 4 delta-method standard errors
        assert abs(t.loc[k, "estimate"] - truth[k]) < 4 * t.loc[k, "se"], k
    assert res.demand == "probit" and "probit demand" in res.method


def test_probit_delta_variance_is_the_hc0_sandwich():
    df = _probit_design(2, n=3000)
    res = sp.efc_welfare(df, "p", "buy", "cost", demand="probit")
    ref = sp.probit(data=df, y="buy", x=["p"], robust="robust", tol=1e-12)
    n = len(df)
    # Stata's vce(robust) is HC0 times n / (n - 1).
    np.testing.assert_allclose(
        res.table.loc[["alpha", "beta"], "se"].to_numpy(),
        ref.std_errors.reindex(["Intercept", "p"]).to_numpy() * np.sqrt((n - 1) / n),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        res.table.loc[["alpha", "beta"], "estimate"].to_numpy(),
        ref.params.reindex(["Intercept", "p"]).to_numpy(),
        rtol=1e-9,
    )


def _quadratic_design(seed: int, n: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.choice([200.0, 350.0, 500.0, 650.0], size=n)
    b1 = -0.0004
    b2 = -(1 + 800 * b1) / 800**2  # through (1, 0) and (0, 800)
    buy = (rng.uniform(size=n) < 1 + b1 * p + b2 * p * p).astype(int)
    cost = np.where(buy == 1, 390.0 + 0.15 * p + rng.normal(0, 60, n), np.nan)
    return pd.DataFrame({"p": p, "buy": buy, "cost": cost})


def test_quadratic_through_two_points_is_restricted_least_squares():
    df = _quadratic_design(3)
    res = sp.efc_welfare(
        df, "p", "buy", "cost", demand="quadratic", demand_through=[(1, 0), (0, 800)]
    )
    c = res.curves
    assert c["alpha"] == pytest.approx(1.0, abs=1e-12)
    assert c["alpha"] + 800 * c["beta"] + 800**2 * c["beta2"] == pytest.approx(
        0.0, abs=1e-12
    )
    # KKT system of the restricted least squares, solved directly.
    p = df["p"].to_numpy()
    X = np.column_stack([np.ones(len(p)), p, p * p])
    R = np.array([[1.0, 0.0, 0.0], [1.0, 800.0, 800.0**2]])
    kkt = np.block([[2 * X.T @ X, R.T], [R, np.zeros((2, 2))]])
    rhs = np.concatenate([2 * X.T @ df["buy"].to_numpy(dtype=float), [1.0, 0.0]])
    b = np.linalg.solve(kkt, rhs)[:3]
    np.testing.assert_allclose([c["alpha"], c["beta"], c["beta2"]], b, rtol=1e-9)
    assert c["beta"] == pytest.approx(-0.0004, abs=4 * res.table.loc["beta", "se"])
    # Two constraints leave one free coefficient: alpha and the curvature
    # have no sampling variance of their own.
    assert res.table.loc["alpha", "se"] < 1e-12


def test_linear_demand_through_a_point_matches_the_appendix_row_3_shape():
    df = _quadratic_design(4)
    res = sp.efc_welfare(df, "p", "buy", "cost", demand_through=[(1, 0)])
    assert res.curves["alpha"] == pytest.approx(1.0, abs=1e-12)
    assert res.demand == "linear" and "linear demand" in res.method


def test_default_path_is_the_linear_closed_form():
    df = _quadratic_design(5, n=4000)
    a = sp.efc_welfare(df, "p", "buy", "cost")
    b = sp.efc_welfare(df, "p", "buy", "cost", demand="linear")
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.method == "Einav-Finkelstein-Cullen welfare (linear curves)"
    assert list(a.table.index) == list(rows_for("linear"))


def test_probit_bootstrap_runs_and_is_reproducible():
    df = _probit_design(6, n=3000)
    kw = dict(demand="probit", se_method="bootstrap", n_boot=40, seed=1)
    a = sp.efc_welfare(df, "p", "buy", "cost", **kw)
    b = sp.efc_welfare(df, "p", "buy", "cost", **kw)
    assert a.table["se"].tolist() == b.table["se"].tolist()
    assert (a.table.loc[["coverage_eq", "dwl"], "se"] > 0).all()


def test_errors():
    df = _probit_design(7, n=2000)
    df["x"] = np.random.default_rng(0).normal(size=len(df))
    with pytest.raises(MethodIncompatibility):
        sp.efc_welfare(df, "p", "buy", "cost", demand="logit")
    with pytest.raises(MethodIncompatibility):
        sp.efc_welfare(df, "p", "buy", "cost", demand="probit", demand_through=[(1, 0)])
    with pytest.raises(MethodIncompatibility):
        sp.efc_welfare(df, "p", "buy", "cost", ["x"], demand_through=[(1, 0)])
    with pytest.raises(MethodIncompatibility):
        sp.efc_welfare(df, "p", "buy", "cost", demand_through=[(1, 0), (0, 800)])
    up = df.assign(buy=(df["p"] > 450).astype(int))
    up["cost"] = np.where(up["buy"] == 1, 400.0 + 0.1 * up["p"], np.nan)
    with pytest.raises(IdentificationFailure):
        sp.efc_welfare(up, "p", "buy", "cost", demand="quadratic")


def test_probit_demand_has_no_interior_corner_at_full_coverage():
    # With probit demand the price of full coverage is minus infinity, so
    # the efficient allocation is interior; the surplus is finite there.
    out = dict(
        zip(
            rows_for("probit"),
            efc_quantities("probit", [1.149, -0.00183], 391.7, 0.155),
        )
    )
    assert 0 < out["coverage_eff"] < 1
    assert np.isfinite(out["price_eff"]) and np.isfinite(out["surplus_eff"])
    assert norm.cdf(1.149 - 0.00183 * out["price_eff"]) == pytest.approx(
        out["coverage_eff"], rel=1e-12
    )
